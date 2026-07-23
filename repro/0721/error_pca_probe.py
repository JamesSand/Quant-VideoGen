#!/usr/bin/env python3
"""Probe quantizer-first PCA correction with the existing INT2 storage layout.

This is a research-only candidate.  It keeps the decoder and BPE unchanged:

    Xc --INT2--> Q(Xc), E = Xc - Q(Xc)
    E --rank-r basis + INT2 coefficients--> Lq
    Xhat = mu + Q(Xc) + Lq

For a fixed Q(Xc) and unquantized rank-r branch, truncated SVD of E is the
globally optimal rank-r correction (Eckart--Young).  Here we retain the
production coefficient quantizer and fast fixed-step subspace solver.
"""
import os
import sys
import torch

os.chdir("/home/zhizhousha/workspace/video-project/Quant-VideoGen")
sys.path.insert(0, "repro/0720/kernel")
sys.path.insert(0, ".")

from bp_quant import FP8, bp_encode_fast, bp_bytes
from bp_triton import triton_decode_packed256
from quant_videogen.functions import (
    triton_prq_dequantize_tensor,
    triton_prq_quantize_tensor,
)


_CACHE = {}


def _get_error_core(r, grid, block, axis, iters=5):
    key = (r, grid, block, axis, iters)
    if key in _CACHE:
        return _CACHE[key]

    def core(X):
        BH, S, D = X.shape
        mu = X.mean(dim=1, keepdim=True)
        Xc = X - mu

        # First store the full-dimensional INT2 backbone.
        t2 = Xc.transpose(1, 2) if axis == "channel" else Xc
        L = t2.shape[-1]
        pad = (block - L % block) % block
        tb = torch.nn.functional.pad(t2, (0, pad)).reshape(
            t2.shape[0], t2.shape[1], -1, block
        )
        dims = (-2, -1) if axis == "channel" else None

        def fmax(t):
            if dims:
                return t.abs().amax(dim=dims, keepdim=True).clamp_min(1e-12)
            return t.abs().amax().clamp_min(1e-12)

        if grid == "ternary":
            sc0 = tb.abs().amax(-1, keepdim=True).clamp_min(1e-8)
            f_sc = fmax(sc0)
            if dims:
                f_sc = torch.exp2(torch.ceil(torch.log2(f_sc)))
            sc = ((sc0 / f_sc).to(FP8).float() * f_sc).clamp_min(1e-8)
            q = (torch.clamp(torch.round(tb / sc), -1, 1) + 1).to(torch.uint8)
            qhat = (q.float() - 1) * sc
            zp8 = None
            f_zp = f_sc
        else:
            mn = tb.amin(-1, keepdim=True)
            mx = tb.amax(-1, keepdim=True)
            sc0 = ((mx - mn) / 3).clamp_min(1e-8)
            f_sc, f_zp = fmax(sc0), fmax(mn)
            if dims:
                f_sc = f_zp = torch.exp2(
                    torch.ceil(torch.log2(torch.maximum(f_sc, f_zp)))
                )
            sc = ((sc0 / f_sc).to(FP8).float() * f_sc).clamp_min(1e-8)
            zp = (mn / f_zp).to(FP8).float() * f_zp
            q = torch.clamp(torch.round((tb - zp) / sc), 0, 3).to(torch.uint8)
            qhat = q.float() * sc + zp
            zp8 = (zp / f_zp).squeeze(-1).to(FP8)

        qflat = q.reshape(t2.shape[0], t2.shape[1], -1, 4)
        packed = (
            qflat[..., 0]
            | (qflat[..., 1] << 2)
            | (qflat[..., 2] << 4)
            | (qflat[..., 3] << 6)
        )
        backbone = qhat.reshape(t2.shape[0], t2.shape[1], -1)[..., :L]
        if axis == "channel":
            backbone = backbone.transpose(1, 2)

        # Fit the low-rank branch to what INT2 failed to represent.
        err = Xc - backbone
        eb = err.to(torch.bfloat16)
        cov = torch.baddbmm(
            torch.zeros(1, device=X.device, dtype=torch.bfloat16),
            eb.transpose(1, 2),
            eb,
            beta=0,
        ).float() / S
        V = cov[:, :, :r].clone()
        for _ in range(iters):
            V = cov @ V
            G = V.transpose(1, 2) @ V
            scale = G.diagonal(dim1=-2, dim2=-1).amax(-1)[:, None, None]
            G = G + 1e-6 * torch.eye(
                r, device=V.device, dtype=V.dtype
            ) * scale
            chol = torch.linalg.cholesky(G)
            V = torch.linalg.solve_triangular(
                chol, V.transpose(1, 2), upper=False
            ).transpose(1, 2)

        c = torch.bmm(err, V)
        cmn, cmx = c.amin(-1, keepdim=True), c.amax(-1, keepdim=True)
        cs0 = ((cmx - cmn) / 3).clamp_min(1e-8)
        f_cs = cs0.amax().clamp_min(1e-8)
        f_cz = cmn.abs().amax().clamp_min(1e-8)
        cs = ((cs0 / f_cs).to(FP8).float() * f_cs).clamp_min(1e-8)
        cz = (cmn / f_cz).to(FP8).float() * f_cz
        cq = torch.clamp(torch.round((c - cz) / cs), 0, 3).to(torch.uint8)
        if r % 4:
            cq = torch.nn.functional.pad(cq, (0, 4 - r % 4))
        cq = cq.reshape(BH, S, -1, 4)
        coef_pack = (
            cq[..., 0]
            | (cq[..., 1] << 2)
            | (cq[..., 2] << 4)
            | (cq[..., 3] << 6)
        )
        return (
            mu.to(torch.bfloat16),
            V.to(torch.bfloat16),
            coef_pack.reshape(BH, S, -1),
            (cs / f_cs).squeeze(-1).to(FP8),
            (cz / f_cz).squeeze(-1).to(FP8),
            packed,
            (sc / f_sc).squeeze(-1).to(FP8),
            zp8,
            f_cs,
            f_cz,
            f_sc,
            f_zp,
        )

    fn = torch.compile(core, dynamic=False, mode="reduce-overhead")
    _CACHE[key] = fn
    return fn


@torch.no_grad()
def error_pca_encode_fast(x, r=9, grid="asym", block=64, axis="channel", iters=5):
    B, H, S, D = x.shape
    BH = B * H
    L = S if axis == "channel" else D
    pad = (block - L % block) % block
    vals = _get_error_core(r, grid, block, axis, iters)(
        x.reshape(BH, S, D).float()
    )
    # reduce-overhead uses reusable graph buffers.
    (
        mu,
        V,
        coef_pack,
        cs8,
        cz8,
        packed,
        sc8,
        zp8,
        f_cs,
        f_cz,
        f_sc,
        f_zp,
    ) = (v.clone() if torch.is_tensor(v) else v for v in vals)
    out = {
        "shape": (B, H, S, D),
        "r": r,
        "grid": grid,
        "block": block,
        "axis": axis,
        "mu": mu,
        "basis": V,
        "coef_pack": coef_pack,
        "coef_scale": cs8,
        "coef_zp": cz8,
        "res_pack": packed,
        "res_scale": sc8,
        "res_zp": zp8,
        "res_g": block,
        "res_pad": pad,
        "f_cs": float(f_cs),
        "f_cz": float(f_cz),
    }
    if axis == "channel":
        out["f_exp"] = torch.log2(f_sc.reshape(BH, -1)).round().to(torch.int8)
    else:
        out["f_sc"], out["f_zp"] = float(f_sc), float(f_zp)
    return out


def timed(fn, reps=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return sorted(samples)[len(samples) // 2]


def rel_l2(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def probe(chunk):
    data = torch.load(
        f"repro/0720/chunks/hy/chunk_{chunk:03d}.pt", map_location="cuda"
    )
    k, v = data["k"], data["v"]
    k1, k2 = k[..., :128].contiguous(), k[..., 128:].contiguous()
    v1, v2 = v[..., :128].contiguous(), v[..., 128:].contiguous()

    def current_k():
        return {
            "halves": (
                bp_encode_fast(
                    k1, r=9, grid="asym", block=64, axis="channel",
                    cmode="reduce-overhead",
                ),
                bp_encode_fast(
                    k2, r=0, grid="ternary", block=64, axis="channel",
                    cmode="reduce-overhead",
                ),
            )
        }

    def error_k():
        return {
            "halves": (
                error_pca_encode_fast(
                    k1, r=9, grid="asym", block=64, axis="channel"
                ),
                bp_encode_fast(
                    k2, r=0, grid="ternary", block=64, axis="channel",
                    cmode="reduce-overhead",
                ),
            )
        }

    def current_v():
        return {
            "halves": (
                bp_encode_fast(
                    v1, r=9, grid="asym", block=128, axis="token",
                    cmode="reduce-overhead",
                ),
                bp_encode_fast(
                    v2, r=0, grid="asym", block=128, axis="token",
                    cmode="reduce-overhead",
                ),
            )
        }

    ck, ek, cv = current_k(), error_k(), current_v()
    dck, dek = triton_decode_packed256(ck), triton_decode_packed256(ek)

    def qvg_encode():
        triton_prq_quantize_tensor(
            k, num_stages=1, num_clusters=256, block_size=64,
            max_iters=2, quantize_fn=lambda _: 2,
        )
        triton_prq_quantize_tensor(
            v, num_stages=1, num_clusters=256, block_size=64,
            max_iters=2, quantize_fn=lambda _: 2,
        )

    qk = triton_prq_quantize_tensor(
        k, num_stages=1, num_clusters=256, block_size=64,
        max_iters=2, quantize_fn=lambda _: 2,
    )
    qv = triton_prq_quantize_tensor(
        v, num_stages=1, num_clusters=256, block_size=64,
        max_iters=2, quantize_fn=lambda _: 2,
    )

    def current_both():
        current_k()
        current_v()

    def error_both():
        error_k()
        current_v()

    print(
        {
            "chunk": chunk,
            "qvg_relL2_k": rel_l2(
                triton_prq_dequantize_tensor(qk, 64, 2), k
            ),
            "current_relL2_k": rel_l2(dck, k),
            "error_pca_relL2_k": rel_l2(dek, k),
            "current_k_bpe": bp_bytes(ck) * 8 / k.numel(),
            "error_k_bpe": bp_bytes(ek) * 8 / k.numel(),
            "qvg_encode_ms_kv": timed(qvg_encode, reps=20, warmup=3),
            "current_encode_ms_kv": timed(current_both),
            "error_pca_encode_ms_kv": timed(error_both),
        },
        flush=True,
    )


if __name__ == "__main__":
    chunks = [int(arg) for arg in sys.argv[1:]] or [1]
    for chunk_id in chunks:
        probe(chunk_id)
