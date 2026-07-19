"""Budget-PCA real quantization: encode -> packed buffers, decode -> bf16.

Storage per (head, chunk), accounting == fake-quant ledger (kernel-plan.md §二):
  res codes   2-bit packed uint8 (4 vals/byte), blocks along `axis`
  res scale/zp fp8 E4M3 per block (ternary: scale only)
  coef codes  2-bit x r per token (r=4 -> exactly 1 byte/token)
  coef scale/zp fp8 per token
  mu, basis   bf16 per chunk (amortized ~0)

Encode math mirrors repro/backup/scripts/pca_quant.py (residual computed AFTER
coef quantization; grids identical). Basis: 'eigh' (parity with fake path) or
'subspace' (deterministic subspace iteration, the fast kernel path).
"""
import torch

FP8 = torch.float8_e4m3fn


def _fp8_store(t):
    return t.to(FP8)


def _fp8_load(t):
    return t.to(torch.float32)


def _pack2(codes):
    """codes uint8 in [0,3], last dim divisible by 4 -> packed uint8."""
    c = codes.reshape(*codes.shape[:-1], codes.shape[-1] // 4, 4)
    return (c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)).contiguous()


def _unpack2(packed, n):
    out = torch.stack([(packed >> (2 * i)) & 3 for i in range(4)], dim=-1)
    return out.reshape(*packed.shape[:-1], packed.shape[-1] * 4)[..., :n]


def _topr_basis(cov, r, method="subspace", iters=8):
    if method == "eigh":
        _, vecs = torch.linalg.eigh(cov)
        return vecs[:, :, -r:].contiguous()
    V = cov[:, :, :r].clone()          # deterministic init
    for i in range(iters):
        V = cov @ V
        V, _ = torch.linalg.qr(V)
    return V


@torch.no_grad()
def bp_encode(x, r=4, grid="asym", block=128, axis="token",
              basis_method="subspace", coef_bits=2):
    """x: [B,H,S,D] -> packed dict. Deterministic, float32 math."""
    B, H, S, D = x.shape
    X = x.reshape(B * H, S, D).float()
    mu = X.mean(dim=1, keepdim=True)                       # [BH,1,D]
    Xc = X - mu
    out = {"shape": (B, H, S, D), "r": r, "grid": grid, "block": block,
           "axis": axis, "mu": mu.to(torch.bfloat16)}
    if r > 0:
        cov = torch.baddbmm(torch.zeros(1, device=x.device), Xc.transpose(1, 2), Xc,
                            beta=0).div_(S)                 # [BH,D,D]
        V = _topr_basis(cov, r, basis_method)               # [BH,D,r]
        c = torch.bmm(Xc, V)                                # [BH,S,r]
        mn = c.amin(-1, keepdim=True); mx = c.amax(-1, keepdim=True)
        cs = _fp8_load(_fp8_store(((mx - mn) / (2 ** coef_bits - 1)).clamp_min(1e-8))).clamp_min(1e-8)
        cz = _fp8_load(_fp8_store(mn))
        cq = torch.clamp(torch.round((c - cz) / cs), 0, 2 ** coef_bits - 1)
        c_hat = cq * cs + cz
        lowrank = torch.bmm(c_hat, V.transpose(1, 2))
        out.update(basis=V.to(torch.bfloat16),
                   coef_pack=_pack2(cq.to(torch.uint8)) if r % 4 == 0 else cq.to(torch.uint8),
                   coef_scale=_fp8_store(cs.squeeze(-1)), coef_zp=_fp8_store(cz.squeeze(-1)))
        res = Xc - lowrank
    else:
        res = Xc
    t = res.transpose(1, 2).contiguous() if axis == "channel" else res  # [BH,D,S] or [BH,S,D]
    L = t.shape[-1]
    g = block
    pad = (g - L % g) % g
    if pad:
        t = torch.nn.functional.pad(t, (0, pad))
    Lp = L + pad
    tb = t.reshape(t.shape[0], t.shape[1], Lp // g, g)
    if grid == "ternary":
        sc = _fp8_load(_fp8_store(tb.abs().amax(-1, keepdim=True).clamp_min(1e-8))).clamp_min(1e-8)
        q = torch.clamp(torch.round(tb / sc), -1, 1) + 1     # {0,1,2}
        out.update(res_pack=_pack2(q.to(torch.uint8).reshape(t.shape[0], t.shape[1], Lp)),
                   res_scale=_fp8_store(sc.squeeze(-1)), res_zp=None, res_g=g, res_pad=pad)
    else:
        mn = tb.amin(-1, keepdim=True); mx = tb.amax(-1, keepdim=True)
        sc = _fp8_load(_fp8_store(((mx - mn) / 3).clamp_min(1e-8))).clamp_min(1e-8)
        zp = _fp8_load(_fp8_store(mn))
        q = torch.clamp(torch.round((tb - zp) / sc), 0, 3)
        out.update(res_pack=_pack2(q.to(torch.uint8).reshape(t.shape[0], t.shape[1], Lp)),
                   res_scale=_fp8_store(sc.squeeze(-1)), res_zp=_fp8_store(zp.squeeze(-1)), res_g=g, res_pad=pad)
    return out


@torch.no_grad()
def bp_decode(d, dtype=torch.bfloat16):
    B, H, S, D = d["shape"]
    ax_ch = d["axis"] == "channel"
    L = S if ax_ch else D
    rows = D if ax_ch else S
    g = d["res_g"]
    Lp = L + d.get("res_pad", 0)
    codes = _unpack2(d["res_pack"], Lp).reshape(B * H, rows, Lp // g, g).float()
    sc = _fp8_load(d["res_scale"]).unsqueeze(-1)
    if d["grid"] == "ternary":
        res = (codes - 1) * sc
    else:
        res = codes * sc + _fp8_load(d["res_zp"]).unsqueeze(-1)
    res = res.reshape(B * H, rows, Lp)[..., :L]
    if ax_ch:
        res = res.transpose(1, 2)                            # -> [BH,S,D]
    out = res + _fp8_load(d["mu"]) if d["mu"].dtype == FP8 else res + d["mu"].float()
    if d["r"] > 0:
        cq = _unpack2(d["coef_pack"], d["r"]).float() if d["r"] % 4 == 0 else d["coef_pack"].float()
        c_hat = cq * _fp8_load(d["coef_scale"]).unsqueeze(-1) + _fp8_load(d["coef_zp"]).unsqueeze(-1)
        out = out + torch.bmm(c_hat, d["basis"].float().transpose(1, 2))
    return out.reshape(B, H, S, D).to(dtype)


@torch.no_grad()
def bp_encode_packed256(x, r_h1=9, r_h2=0, **kw):
    """HY: [B,H,S,256] = rope||prope halves, separate rank/grid per half."""
    h1 = bp_encode(x[..., :128].contiguous(), r=r_h1, **{**kw})
    kw2 = dict(kw); kw2["grid"] = kw.get("grid_h2", "ternary"); kw2.pop("grid_h2", None)
    kw2["block"] = kw.get("block_h2", 64); kw2.pop("block_h2", None)
    kw.pop("grid_h2", None); kw.pop("block_h2", None)
    h2 = bp_encode(x[..., 128:].contiguous(), r=r_h2, **{k: v for k, v in kw2.items() if k not in ("grid_h2", "block_h2")})
    return {"halves": (h1, h2)}


@torch.no_grad()
def bp_decode_packed256(d, dtype=torch.bfloat16):
    return torch.cat([bp_decode(d["halves"][0], dtype), bp_decode(d["halves"][1], dtype)], dim=-1)


def bp_bytes(d):
    """Actual stored bytes (compression-ratio audit)."""
    if "halves" in d:
        return sum(bp_bytes(h) for h in d["halves"])
    n = 0
    for k in ("mu", "basis", "coef_pack", "coef_scale", "coef_zp", "res_pack", "res_scale", "res_zp"):
        t = d.get(k)
        if torch.is_tensor(t):
            n += t.numel() * t.element_size()
    return n
