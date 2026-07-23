"""Online fixed-partition PCA-grid hash codec prototype.

The PCA basis and bin thresholds are encoder-only.  Four analytic 2-bit bins
form one uint8 label per token.  Fixed labels index an FP8 table; the remaining
error is packed with a four-level asymmetric channel-axis INT2 grid.
"""
from __future__ import annotations

import torch
import triton

from bp_quant import FP8, _topr_basis, _unpack2
from quant_videogen.kmeans.centroid_update import (
    _centroid_update_chunk_kernel,
    triton_centroid_update_euclid,
)


K = 256
GAUSSIAN_QUARTILE = 0.6744897501960817
_LABEL_CACHE = {}
_RESIDUAL_CACHE = {}


def _fixed_labels(xc: torch.Tensor, r: int, binning: str, iters: int):
    """Return uint8 analytic labels; the PCA state is not stored."""
    bh, s, d = xc.shape
    xb = xc.to(torch.bfloat16)
    cov = torch.baddbmm(
        torch.zeros(1, device=xc.device, dtype=torch.bfloat16),
        xb.transpose(1, 2),
        xb,
        beta=0,
    ).float() / s
    basis = _topr_basis(cov, r, method="subspace", iters=iters)
    coef = torch.bmm(xc, basis)

    if binning == "gaussian":
        sigma = coef.square().mean(1, keepdim=True).sqrt().clamp_min(1e-8)
        z = coef / sigma
        codes = (
            (z > -GAUSSIAN_QUARTILE).to(torch.uint8)
            + (z > 0).to(torch.uint8)
            + (z > GAUSSIAN_QUARTILE).to(torch.uint8)
        )
    elif binning == "uniform":
        lo = coef.amin(1, keepdim=True)
        step = ((coef.amax(1, keepdim=True) - lo) / 4).clamp_min(1e-8)
        codes = (
            (coef > lo + step).to(torch.uint8)
            + (coef > lo + 2 * step).to(torch.uint8)
            + (coef > lo + 3 * step).to(torch.uint8)
        )
    else:
        raise ValueError(f"unknown binning: {binning}")

    if r != 4:
        raise ValueError("the packed prototype currently requires r=4")
    labels = (
        codes[..., 0]
        | (codes[..., 1] << 2)
        | (codes[..., 2] << 4)
        | (codes[..., 3] << 6)
    )
    return labels.contiguous()


def _get_label_core(iters: int):
    if iters in _LABEL_CACHE:
        return _LABEL_CACHE[iters]

    def core(x):
        xf = x.float()
        bh, s, d = xf.shape
        xc = xf - xf.mean(1, keepdim=True)
        xb = xc.to(torch.bfloat16)
        cov = torch.baddbmm(
            torch.zeros(1, device=xf.device, dtype=torch.bfloat16),
            xb.transpose(1, 2),
            xb,
            beta=0,
        ).float() / s
        basis = cov[:, :, :4].clone()
        for _ in range(iters):
            basis = cov @ basis
            gram = basis.transpose(1, 2) @ basis
            scale = gram.diagonal(dim1=-2, dim2=-1).amax(-1)[:, None, None]
            gram = gram + 1e-6 * torch.eye(
                4, device=xf.device, dtype=xf.dtype
            ) * scale
            chol = torch.linalg.cholesky(gram)
            basis = torch.linalg.solve_triangular(
                chol, basis.transpose(1, 2), upper=False
            ).transpose(1, 2)
        coef = torch.bmm(xc, basis)
        sigma = coef.square().mean(1, keepdim=True).sqrt().clamp_min(1e-8)
        z = coef / sigma
        codes = (
            (z > -GAUSSIAN_QUARTILE).to(torch.uint8)
            + (z > 0).to(torch.uint8)
            + (z > GAUSSIAN_QUARTILE).to(torch.uint8)
        )
        return (
            codes[..., 0]
            | (codes[..., 1] << 2)
            | (codes[..., 2] << 4)
            | (codes[..., 3] << 6)
        )

    fn = torch.compile(core, dynamic=False, mode="reduce-overhead")
    _LABEL_CACHE[iters] = fn
    return fn


def fixed_labels_fast(x: torch.Tensor, iters: int = 2):
    labels = _get_label_core(iters)(x)
    return labels.clone()


def _prepare_sorted_labels(labels: torch.Tensor):
    sorted_labels, sorted_indices = torch.sort(labels, dim=-1)
    return sorted_labels.to(torch.int32), sorted_indices.to(torch.int32)


def _group_mean_sorted(
    x: torch.Tensor,
    sorted_labels: torch.Tensor,
    sorted_indices: torch.Tensor,
    block_n: int = 256,
):
    """Fixed-label reduction, reusing one label sort across refit steps."""
    b, n, d = x.shape
    sums = torch.zeros(b, K, d, device=x.device, dtype=torch.float32)
    counts = torch.zeros(b, K, device=x.device, dtype=torch.int32)
    grid = (triton.cdiv(n, block_n), b)
    _centroid_update_chunk_kernel[grid](
        x.contiguous(),
        sorted_indices,
        sorted_labels,
        sums,
        counts,
        b,
        n,
        d,
        K,
        BLOCK_N=block_n,
    )
    return sums / counts.float().unsqueeze(-1).clamp_min(1.0)


def _group_mean(x: torch.Tensor, labels: torch.Tensor, sorted_data=None):
    """Exact fixed-label least-squares table before storage quantization."""
    if sorted_data is not None:
        return _group_mean_sorted(x, *sorted_data)
    old = torch.zeros(
        x.shape[0], K, x.shape[2], device=x.device, dtype=x.dtype
    )
    table = triton_centroid_update_euclid(
        x.contiguous(), labels.long().contiguous(), old
    )
    return table.float()


def _pow2_fp8_table(table: torch.Tensor):
    factor = table.abs().amax(1).clamp_min(2.0 ** -120)
    factor = torch.exp2(torch.ceil(torch.log2(factor)))
    table8 = (table / factor.unsqueeze(1)).to(FP8)
    table_hat = table8.float() * factor.unsqueeze(1)
    exp = torch.log2(factor).round().to(torch.int8)
    return table8, exp, table_hat


def _gather_table(table_hat: torch.Tensor, labels: torch.Tensor):
    index = labels.long().unsqueeze(-1).expand(-1, -1, table_hat.shape[-1])
    return torch.gather(table_hat, 1, index)


def _asym_channel_quant(res: torch.Tensor, block: int):
    """Quantize and pack [BH,S,D], returning actual stored-value reconstruction."""
    bh, s, d = res.shape
    pad = (block - s % block) % block
    t = torch.nn.functional.pad(res.transpose(1, 2), (0, pad))
    tb = t.reshape(bh, d, -1, block)
    mn, mx = tb.amin(-1, keepdim=True), tb.amax(-1, keepdim=True)
    sc0 = ((mx - mn) / 3).clamp_min(1e-8)
    factor = torch.maximum(
        sc0.abs().amax((-2, -1), keepdim=True),
        mn.abs().amax((-2, -1), keepdim=True),
    ).clamp_min(2.0 ** -120)
    factor = torch.exp2(torch.ceil(torch.log2(factor)))
    sc8 = (sc0 / factor).to(FP8)
    zp8 = (mn / factor).to(FP8)
    sc = sc8.float() * factor
    zp = zp8.float() * factor
    q = torch.clamp(torch.round((tb - zp) / sc.clamp_min(1e-8)), 0, 3)
    q8 = q.to(torch.uint8)
    q4 = q8.reshape(bh, d, -1, 4)
    packed = (
        q4[..., 0]
        | (q4[..., 1] << 2)
        | (q4[..., 2] << 4)
        | (q4[..., 3] << 6)
    ).contiguous()
    res_hat = (q * sc + zp).reshape(bh, d, -1)[..., :s].transpose(1, 2)
    exp = torch.log2(factor.reshape(bh, d)).round().to(torch.int8)
    return {
        "res_pack": packed,
        "res_scale": sc8.squeeze(-1),
        "res_zp": zp8.squeeze(-1),
        "res_exp": exp,
        "res_pad": pad,
        "res_g": block,
    }, res_hat.contiguous(), (sc, zp)


def _get_residual_core(block: int, need_hat: bool, slot: int = 0):
    key = (block, need_hat, slot)
    if key in _RESIDUAL_CACHE:
        return _RESIDUAL_CACHE[key]

    def core(x, labels, table8, table_exp):
        xf = x.float()
        bh, s, d = xf.shape
        table_hat = table8.float() * torch.exp2(
            table_exp.float()
        ).unsqueeze(1)
        pred = torch.gather(
            table_hat,
            1,
            labels.long().unsqueeze(-1).expand(-1, -1, d),
        )
        res = xf - pred
        pad = (block - s % block) % block
        tb = torch.nn.functional.pad(
            res.transpose(1, 2), (0, pad)
        ).reshape(bh, d, -1, block)
        mn, mx = tb.amin(-1, keepdim=True), tb.amax(-1, keepdim=True)
        sc0 = ((mx - mn) / 3).clamp_min(1e-8)
        factor = torch.maximum(
            sc0.abs().amax((-2, -1), keepdim=True),
            mn.abs().amax((-2, -1), keepdim=True),
        ).clamp_min(2.0 ** -120)
        factor = torch.exp2(torch.ceil(torch.log2(factor)))
        sc8 = (sc0 / factor).to(FP8)
        zp8 = (mn / factor).to(FP8)
        sc = sc8.float() * factor
        zp = zp8.float() * factor
        q = torch.clamp(
            torch.round((tb - zp) / sc.clamp_min(1e-8)), 0, 3
        )
        q8 = q.to(torch.uint8)
        q4 = q8.reshape(bh, d, -1, 4)
        packed = (
            q4[..., 0]
            | (q4[..., 1] << 2)
            | (q4[..., 2] << 4)
            | (q4[..., 3] << 6)
        )
        exp = torch.log2(factor.reshape(bh, d)).round().to(torch.int8)
        if need_hat:
            z = (q * sc + zp).reshape(bh, d, -1)[..., :s].transpose(1, 2)
            return packed, sc8.squeeze(-1), zp8.squeeze(-1), exp, z
        return packed, sc8.squeeze(-1), zp8.squeeze(-1), exp

    fn = torch.compile(core, dynamic=False, mode="reduce-overhead")
    _RESIDUAL_CACHE[key] = fn
    return fn


def _residual_encode_fast(
    x, labels, table8, table_exp, block, need_hat, slot=0
):
    vals = _get_residual_core(block, need_hat, slot)(
        x, labels, table8, table_exp
    )
    return tuple(v.clone() for v in vals)


def _reround_fixed_grid(
    res: torch.Tensor,
    sc: torch.Tensor,
    zp: torch.Tensor,
    block: int,
):
    """Nearest-code update while holding the stored FP8 grid fixed."""
    bh, s, d = res.shape
    pad = (block - s % block) % block
    tb = torch.nn.functional.pad(res.transpose(1, 2), (0, pad)).reshape(
        bh, d, -1, block
    )
    q = torch.clamp(torch.round((tb - zp) / sc.clamp_min(1e-8)), 0, 3)
    q8 = q.to(torch.uint8)
    q4 = q8.reshape(bh, d, -1, 4)
    packed = (
        q4[..., 0]
        | (q4[..., 1] << 2)
        | (q4[..., 2] << 4)
        | (q4[..., 3] << 6)
    ).contiguous()
    res_hat = (q * sc + zp).reshape(bh, d, -1)[..., :s].transpose(1, 2)
    return packed, res_hat.contiguous()


def _select_heads(a: torch.Tensor, b: torch.Tensor, use_b: torch.Tensor):
    shape = (a.shape[0],) + (1,) * (a.ndim - 1)
    return torch.where(use_b.reshape(shape), b, a)


def _choose_stored_refit(
    x,
    labels,
    residual,
    table8_0,
    table_exp_0,
    table_hat_0,
    table8_1,
    table_exp_1,
    table_hat_1,
):
    """Retain a refit per head only when decoded BF16 SSE strictly improves."""
    pred0 = _gather_table(table_hat_0, labels)
    pred1 = _gather_table(table_hat_1, labels)
    error0 = (x - (pred0 + residual).to(torch.bfloat16).float()).square().sum(
        (1, 2)
    )
    error1 = (x - (pred1 + residual).to(torch.bfloat16).float()).square().sum(
        (1, 2)
    )
    use_refit = error1 < error0
    return (
        _select_heads(table8_0, table8_1, use_refit),
        _select_heads(table_exp_0, table_exp_1, use_refit),
    )


@torch.no_grad()
def grid_hash_encode(
    x: torch.Tensor,
    *,
    r: int = 4,
    binning: str = "gaussian",
    block: int = 64,
    iters: int = 5,
    refine: int = 1,
    labels: torch.Tensor | None = None,
    sorted_data=None,
):
    """Encode [B,H,S,D] into an actual byte-countable packed state."""
    b, h, s, d = x.shape
    bh = b * h
    xf = x.reshape(bh, s, d).float()
    if labels is None:
        xc = xf - xf.mean(1, keepdim=True)
        labels = _fixed_labels(xc, r, binning, iters)
    if sorted_data is None:
        sorted_data = _prepare_sorted_labels(labels)

    table0 = _group_mean(xf, labels, sorted_data)
    table8_0, table_exp_0, table_hat0 = _pow2_fp8_table(table0)
    pred0 = _gather_table(table_hat0, labels)
    residual_state, z0, grid = _asym_channel_quant(xf - pred0, block)
    recon0 = (pred0 + z0).to(torch.bfloat16).float()

    table8, table_exp = table8_0, table_exp_0
    packed, z = residual_state["res_pack"], z0
    selected_refine = torch.zeros(bh, device=x.device, dtype=torch.bool)

    if refine:
        # Fixed labels and fixed residual payload: one exact table LS update.
        target1 = xf - z0
        table1 = _group_mean(target1, labels, sorted_data)
        table8_1, table_exp_1, table_hat1 = _pow2_fp8_table(table1)
        pred1 = _gather_table(table_hat1, labels)
        recon1 = (pred1 + z0).to(torch.bfloat16).float()
        e0 = (xf - recon0).square().sum((1, 2))
        e1 = (xf - recon1).square().sum((1, 2))
        selected_refine = e1 < e0
        table8 = _select_heads(table8_0, table8_1, selected_refine)
        table_exp = _select_heads(table_exp_0, table_exp_1, selected_refine)

    if refine >= 2:
        # Optional second coordinate cycle: reround on the frozen FP8 grid,
        # refit the table once more, and retain only actual stored-value wins.
        packed1, z1 = _reround_fixed_grid(xf - pred1, *grid, block)

        target2 = xf - z1
        table2 = _group_mean(target2, labels, sorted_data)
        table8_2, table_exp_2, table_hat2 = _pow2_fp8_table(table2)
        pred2 = _gather_table(table_hat2, labels)
        packed2, z2 = _reround_fixed_grid(xf - pred2, *grid, block)
        recon2 = (pred2 + z2).to(torch.bfloat16).float()

        e2 = (xf - recon2).square().sum((1, 2))
        selected_second = e2 < torch.minimum(e0, e1)
        table8 = _select_heads(table8, table8_2, selected_second)
        table_exp = _select_heads(table_exp, table_exp_2, selected_second)
        packed = _select_heads(packed, packed2, selected_second)
        z = _select_heads(z0, z2, selected_second)

    state = {
        "shape": (b, h, s, d),
        "r": r,
        "binning": binning,
        "labels": labels,
        "table": table8,
        "table_exp": table_exp,
        "res_pack": packed,
        "res_scale": residual_state["res_scale"],
        "res_zp": residual_state["res_zp"],
        "res_exp": residual_state["res_exp"],
        "res_pad": residual_state["res_pad"],
        "res_g": block,
    }
    return state


@torch.no_grad()
def grid_hash_encode_fast(
    x: torch.Tensor,
    *,
    block: int = 64,
    iters: int = 2,
    refine: int = 1,
    labels: torch.Tensor | None = None,
    sorted_data=None,
):
    """Compiled hot-path prototype; refinement keeps the residual payload fixed."""
    b, h, s, d = x.shape
    bh = b * h
    xf = x.reshape(bh, s, d).float()
    if labels is None:
        labels = fixed_labels_fast(xf, iters)
    if sorted_data is None:
        sorted_data = _prepare_sorted_labels(labels)

    table0 = _group_mean_sorted(xf, *sorted_data)
    table8, table_exp, table_hat0 = _pow2_fp8_table(table0)
    vals = _residual_encode_fast(
        xf, labels, table8, table_exp, block, need_hat=refine > 0
    )
    if refine:
        packed, sc8, zp8, res_exp, z0 = vals
        table1 = _group_mean_sorted(xf.float() - z0, *sorted_data)
        table8_1, table_exp_1, table_hat1 = _pow2_fp8_table(table1)
        table8, table_exp = _choose_stored_refit(
            xf,
            labels,
            z0,
            table8,
            table_exp,
            table_hat0,
            table8_1,
            table_exp_1,
            table_hat1,
        )
    else:
        packed, sc8, zp8, res_exp = vals

    return {
        "shape": (b, h, s, d),
        "r": 4,
        "binning": "gaussian",
        "labels": labels,
        "table": table8,
        "table_exp": table_exp,
        "res_pack": packed,
        "res_scale": sc8,
        "res_zp": zp8,
        "res_exp": res_exp,
        "res_pad": (block - s % block) % block,
        "res_g": block,
    }


@torch.no_grad()
def grid_hash_encode_kv_fast(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block: int = 64,
    iters: int = 2,
    refine: int = 1,
    shared_labels: str | None = None,
):
    if shared_labels is None:
        return {
            "k": grid_hash_encode_fast(
                k, block=block, iters=iters, refine=refine
            ),
            "v": grid_hash_encode_fast(
                v, block=block, iters=iters, refine=refine
            ),
            "shared_labels": False,
        }
    source = {
        "k": k,
        "v": v,
        "k_rope": k[..., :128],
        "v_rope": v[..., :128],
    }[shared_labels]
    b, h, s, source_d = source.shape
    d = k.shape[-1]
    labels = fixed_labels_fast(source.reshape(b * h, s, source_d), iters)
    sorted_data = _prepare_sorted_labels(labels)
    if d in (128, 256):
        bh = b * h
        kf = k.reshape(bh, s, d).float()
        vf = v.reshape(bh, s, d).float()
        table_k0 = _group_mean_sorted(kf, *sorted_data)
        table_v0 = _group_mean_sorted(vf, *sorted_data)
        tk8, tkexp, tkhat0 = _pow2_fp8_table(table_k0)
        tv8, tvexp, tvhat0 = _pow2_fp8_table(table_v0)
        kvals = _residual_encode_fast(
            kf, labels, tk8, tkexp, block, need_hat=refine > 0, slot=0
        )
        vvals = _residual_encode_fast(
            vf, labels, tv8, tvexp, block, need_hat=refine > 0, slot=1
        )
        if refine:
            kp, ks, kz, ke, z_k = kvals
            vp, vs, vz, ve, z_v = vvals
            table_k1 = _group_mean_sorted(kf.float() - z_k, *sorted_data)
            table_v1 = _group_mean_sorted(vf.float() - z_v, *sorted_data)
            tk8_1, tkexp_1, tkhat1 = _pow2_fp8_table(table_k1)
            tv8_1, tvexp_1, tvhat1 = _pow2_fp8_table(table_v1)
            tk8, tkexp = _choose_stored_refit(
                kf,
                labels,
                z_k,
                tk8,
                tkexp,
                tkhat0,
                tk8_1,
                tkexp_1,
                tkhat1,
            )
            tv8, tvexp = _choose_stored_refit(
                vf,
                labels,
                z_v,
                tv8,
                tvexp,
                tvhat0,
                tv8_1,
                tvexp_1,
                tvhat1,
            )
        else:
            kp, ks, kz, ke = kvals
            vp, vs, vz, ve = vvals

        def make_state(table8, table_exp, packed, scale, zp, rexp):
            return {
                "shape": (b, h, s, d),
                "r": 4,
                "binning": "gaussian",
                "labels": labels,
                "table": table8,
                "table_exp": table_exp,
                "res_pack": packed,
                "res_scale": scale,
                "res_zp": zp,
                "res_exp": rexp,
                "res_pad": (block - s % block) % block,
                "res_g": block,
            }

        return {
            "k": make_state(tk8, tkexp, kp, ks, kz, ke),
            "v": make_state(tv8, tvexp, vp, vs, vz, ve),
            "shared_labels": True,
        }
    return {
        "k": grid_hash_encode_fast(
            k,
            block=block,
            iters=iters,
            refine=refine,
            labels=labels,
            sorted_data=sorted_data,
        ),
        "v": grid_hash_encode_fast(
            v,
            block=block,
            iters=iters,
            refine=refine,
            labels=labels,
            sorted_data=sorted_data,
        ),
        "shared_labels": True,
    }


@torch.no_grad()
def grid_hash_decode(state, dtype=torch.bfloat16):
    b, h, s, d = state["shape"]
    bh = b * h
    table_hat = state["table"].float() * torch.exp2(
        state["table_exp"].float()
    ).unsqueeze(1)
    pred = _gather_table(table_hat, state["labels"])

    lp = s + state["res_pad"]
    g = state["res_g"]
    codes = _unpack2(state["res_pack"].reshape(bh, d, lp // 4), lp)
    codes = codes.float().reshape(bh, d, lp // g, g)
    factor = torch.exp2(state["res_exp"].float()).reshape(bh, d, 1, 1)
    sc = state["res_scale"].float().reshape(bh, d, lp // g, 1) * factor
    zp = state["res_zp"].float().reshape(bh, d, lp // g, 1) * factor
    residual = (codes * sc + zp).reshape(bh, d, lp)[..., :s]
    residual = residual.transpose(1, 2)
    return (pred + residual).reshape(b, h, s, d).to(dtype)


def grid_hash_bytes(state):
    keys = (
        "labels",
        "table",
        "table_exp",
        "res_pack",
        "res_scale",
        "res_zp",
        "res_exp",
    )
    return sum(
        state[k].numel() * state[k].element_size()
        for k in keys
        if torch.is_tensor(state.get(k))
    )


def grid_hash_kv_bytes(state):
    total = grid_hash_bytes(state["k"]) + grid_hash_bytes(state["v"])
    if state.get("shared_labels"):
        labels = state["k"]["labels"]
        total -= labels.numel() * labels.element_size()
    return total
