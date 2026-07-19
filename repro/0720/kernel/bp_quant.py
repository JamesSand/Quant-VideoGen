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
_COMPILE = True   # torch.compile fusion for quant/dequant elementwise chains


def _maybe_compile(fn):
    return torch.compile(fn, dynamic=False) if _COMPILE else fn


@_maybe_compile
def _qp_asym_ch(res, g: int):
    """res [BH,S,D] -> channel-axis blocks, transpose+pad fused."""
    t2 = res.transpose(1, 2)
    L = t2.shape[-1]
    pad = (g - L % g) % g
    t3 = torch.nn.functional.pad(t2, (0, pad))
    tb = t3.reshape(res.shape[0], res.shape[2], -1, g)
    mn = tb.amin(-1, keepdim=True); mx = tb.amax(-1, keepdim=True)
    sc = ((mx - mn) / 3).clamp_min(1e-8).to(FP8).float().clamp_min(1e-8)
    zp = mn.to(FP8).float()
    q = torch.clamp(torch.round((tb - zp) / sc), 0, 3).to(torch.uint8)
    qf = q.reshape(res.shape[0], res.shape[2], -1, 4)
    packed = qf[..., 0] | (qf[..., 1] << 2) | (qf[..., 2] << 4) | (qf[..., 3] << 6)
    return packed, sc.squeeze(-1).to(FP8), zp.squeeze(-1).to(FP8)


@_maybe_compile
def _qp_tern_ch(res, g: int):
    t2 = res.transpose(1, 2)
    L = t2.shape[-1]
    pad = (g - L % g) % g
    t3 = torch.nn.functional.pad(t2, (0, pad))
    tb = t3.reshape(res.shape[0], res.shape[2], -1, g)
    sc = tb.abs().amax(-1, keepdim=True).clamp_min(1e-8).to(FP8).float().clamp_min(1e-8)
    q = (torch.clamp(torch.round(tb / sc), -1, 1) + 1).to(torch.uint8)
    qf = q.reshape(res.shape[0], res.shape[2], -1, 4)
    packed = qf[..., 0] | (qf[..., 1] << 2) | (qf[..., 2] << 4) | (qf[..., 3] << 6)
    return packed, sc.squeeze(-1).to(FP8)


@_maybe_compile
def _qp_asym(tb):
    """tb: [.., nblk, g] -> packed uint8, fp8 scale, fp8 zp (per block)."""
    mn = tb.amin(-1, keepdim=True); mx = tb.amax(-1, keepdim=True)
    sc = ((mx - mn) / 3).clamp_min(1e-8).to(FP8).float().clamp_min(1e-8)
    zp = mn.to(FP8).float()
    q = torch.clamp(torch.round((tb - zp) / sc), 0, 3).to(torch.uint8)
    qf = q.reshape(*tb.shape[:-2], -1, 4)
    packed = qf[..., 0] | (qf[..., 1] << 2) | (qf[..., 2] << 4) | (qf[..., 3] << 6)
    return packed, sc.squeeze(-1).to(FP8), zp.squeeze(-1).to(FP8)


@_maybe_compile
def _qp_tern(tb):
    sc = tb.abs().amax(-1, keepdim=True).clamp_min(1e-8).to(FP8).float().clamp_min(1e-8)
    q = (torch.clamp(torch.round(tb / sc), -1, 1) + 1).to(torch.uint8)
    qf = q.reshape(*tb.shape[:-2], -1, 4)
    packed = qf[..., 0] | (qf[..., 1] << 2) | (qf[..., 2] << 4) | (qf[..., 3] << 6)
    return packed, sc.squeeze(-1).to(FP8)


@_maybe_compile
def _dq_full_ch(packed, sc, zp, ternary: bool, g: int, L: int,
                mu, c_hat, basis, f_sc: float = 1.0, f_zp: float = 1.0):
    """channel-axis fused decode: unpack -> dequant -> crop -> transpose ->
    + mu + c_hat @ basis^T, all in one compiled graph. Returns [BH,S,D] f32."""
    c = torch.stack([(packed >> (2 * i)) & 3 for i in range(4)], dim=-1)
    codes = c.reshape(*packed.shape[:-1], packed.shape[-1] * 4).float()
    codes = codes.reshape(*codes.shape[:-1], codes.shape[-1] // g, g)
    scf = sc.float().unsqueeze(-1) * f_sc
    res = (codes - 1) * scf if ternary else codes * scf + zp.float().unsqueeze(-1) * f_zp
    res = res.reshape(*res.shape[:-2], -1)[..., :L]           # [BH,D,S]
    out = res.transpose(1, 2) + mu.float()
    if basis is not None:
        out = out + torch.bmm(c_hat, basis.float().transpose(1, 2))
    return out.to(torch.bfloat16)


@_maybe_compile
def _dq_res(packed, sc, zp, ternary: bool, g: int, f_sc: float = 1.0, f_zp: float = 1.0):
    """packed [.., rows, Lp/4] -> dequantized [.., rows, Lp] float32."""
    c = torch.stack([(packed >> (2 * i)) & 3 for i in range(4)], dim=-1)
    codes = c.reshape(*packed.shape[:-1], packed.shape[-1] * 4).float()
    codes = codes.reshape(*codes.shape[:-1], codes.shape[-1] // g, g)
    scf = sc.float().unsqueeze(-1) * f_sc
    out = (codes - 1) * scf if ternary else codes * scf + zp.float().unsqueeze(-1) * f_zp
    return out.reshape(*out.shape[:-2], -1)


def _fp8_store(t):
    return t.to(FP8)


def _fp8n_store(t):
    """Normalized fp8: returns (fp8 tensor, float factor)."""
    f = t.abs().amax().clamp_min(1e-8)
    return (t / f).to(FP8), f


def _fp8_load(t):
    return t.to(torch.float32)


def _pack2(codes):
    """codes uint8 in [0,3], last dim divisible by 4 -> packed uint8."""
    c = codes.reshape(*codes.shape[:-1], codes.shape[-1] // 4, 4)
    return (c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)).contiguous()


def _unpack2(packed, n):
    out = torch.stack([(packed >> (2 * i)) & 3 for i in range(4)], dim=-1)
    return out.reshape(*packed.shape[:-1], packed.shape[-1] * 4)[..., :n]


def _topr_basis(cov, r, method="subspace", iters=5):
    if method == "eigh":
        _, vecs = torch.linalg.eigh(cov)
        return vecs[:, :, -r:].contiguous()
    V = cov[:, :, :r].clone()          # deterministic init
    for i in range(iters):
        V = cov @ V
        # orthonormalize via inverse Cholesky of the tiny r x r gram
        # (much faster than batched QR on [BH, D, r])
        G = V.transpose(1, 2) @ V
        G = G + 1e-6 * torch.eye(r, device=V.device, dtype=V.dtype) * G.diagonal(dim1=-2, dim2=-1).amax(-1)[:, None, None]
        Lc = torch.linalg.cholesky(G)
        V = torch.linalg.solve_triangular(Lc, V.transpose(1, 2), upper=False).transpose(1, 2)
    return V


_CORE_CACHE = {}


def _get_encode_core(grid, axis, r, block, iters=5):
    key = (grid, axis, r, block, iters)
    if key in _CORE_CACHE:
        return _CORE_CACHE[key]

    def core(X):
        BH, S, D = X.shape
        mu = X.mean(dim=1, keepdim=True)
        Xc = X - mu
        Xb = Xc.to(torch.bfloat16)
        cov = torch.baddbmm(torch.zeros(1, device=X.device, dtype=torch.bfloat16),
                            Xb.transpose(1, 2), Xb, beta=0).float() / S
        V = cov[:, :, :r].clone()
        for _ in range(iters):
            V = cov @ V
            G = V.transpose(1, 2) @ V
            G = G + 1e-6 * torch.eye(r, device=V.device, dtype=V.dtype) * G.diagonal(dim1=-2, dim2=-1).amax(-1)[:, None, None]
            Lc = torch.linalg.cholesky(G)
            V = torch.linalg.solve_triangular(Lc, V.transpose(1, 2), upper=False).transpose(1, 2)
        c = torch.bmm(Xc, V)
        mn = c.amin(-1, keepdim=True); mx = c.amax(-1, keepdim=True)
        cs0 = ((mx - mn) / 3).clamp_min(1e-8)
        f_cs = cs0.amax().clamp_min(1e-8)
        cs = ((cs0 / f_cs).to(FP8).float() * f_cs).clamp_min(1e-8)
        f_cz = mn.abs().amax().clamp_min(1e-8)
        cz = (mn / f_cz).to(FP8).float() * f_cz
        cq = torch.clamp(torch.round((c - cz) / cs), 0, 3)
        c_hat = cq * cs + cz
        res = Xc - torch.bmm(c_hat, V.transpose(1, 2))
        cq8 = cq.to(torch.uint8)
        if r % 4:
            cq8 = torch.nn.functional.pad(cq8, (0, 4 - r % 4))
        cqf = cq8.reshape(BH, S, -1, 4)
        coef_pack = cqf[..., 0] | (cqf[..., 1] << 2) | (cqf[..., 2] << 4) | (cqf[..., 3] << 6)
        t2 = res.transpose(1, 2) if axis == "channel" else res
        L = t2.shape[-1]
        pad = (block - L % block) % block
        t3 = torch.nn.functional.pad(t2, (0, pad))
        tb = t3.reshape(t2.shape[0], t2.shape[1], -1, block)
        if grid == "ternary":
            sc0 = tb.abs().amax(-1, keepdim=True).clamp_min(1e-8)
            f_sc = sc0.amax().clamp_min(1e-8)
            sc = ((sc0 / f_sc).to(FP8).float() * f_sc).clamp_min(1e-8)
            q = (torch.clamp(torch.round(tb / sc), -1, 1) + 1).to(torch.uint8)
            zp8 = None
            f_zp = f_sc
        else:
            mn2 = tb.amin(-1, keepdim=True); mx2 = tb.amax(-1, keepdim=True)
            sc0 = ((mx2 - mn2) / 3).clamp_min(1e-8)
            f_sc = sc0.amax().clamp_min(1e-8)
            sc = ((sc0 / f_sc).to(FP8).float() * f_sc).clamp_min(1e-8)
            f_zp = mn2.abs().amax().clamp_min(1e-8)
            zp = (mn2 / f_zp).to(FP8).float() * f_zp
            q = torch.clamp(torch.round((tb - zp) / sc), 0, 3).to(torch.uint8)
            zp8 = (zp / f_zp).squeeze(-1).to(FP8)
        qf = q.reshape(t2.shape[0], t2.shape[1], -1, 4)
        packed = qf[..., 0] | (qf[..., 1] << 2) | (qf[..., 2] << 4) | (qf[..., 3] << 6)
        return (mu.to(torch.bfloat16), V.to(torch.bfloat16), coef_pack.squeeze(-2) if coef_pack.shape[-2] == 1 else coef_pack.reshape(BH, S, -1),
                (cs / f_cs).squeeze(-1).to(FP8), (cz / f_cz).squeeze(-1).to(FP8),
                packed, (sc / f_sc).squeeze(-1).to(FP8), zp8,
                f_cs, f_cz, f_sc, f_zp)

    fn = torch.compile(core, dynamic=False) if _COMPILE else core
    _CORE_CACHE[key] = fn
    return fn


@torch.no_grad()
def bp_encode_fast(x, r=4, grid="asym", block=128, axis="token", iters=5):
    """Whole-graph compiled encode (r>0 only)."""
    B, H, S, D = x.shape
    core = _get_encode_core(grid, axis, r, block, iters)
    (mu, V, coef_pack, cs8, cz8, packed, sc8, zp8,
     f_cs, f_cz, f_sc, f_zp) = core(x.reshape(B * H, S, D).float())
    L = S if axis == "channel" else D
    pad = (block - L % block) % block
    out = {"shape": (B, H, S, D), "r": r, "grid": grid, "block": block, "axis": axis,
           "mu": mu, "basis": V, "coef_pack": coef_pack.reshape(B * H, S, -1),
           "coef_scale": cs8, "coef_zp": cz8,
           "res_pack": packed, "res_scale": sc8, "res_zp": zp8, "res_g": block, "res_pad": pad,
           "f_cs": float(f_cs), "f_cz": float(f_cz), "f_sc": float(f_sc), "f_zp": float(f_zp)}
    return out


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
        Xb = Xc.to(torch.bfloat16)
        cov = torch.baddbmm(torch.zeros(1, device=x.device, dtype=torch.bfloat16),
                            Xb.transpose(1, 2), Xb, beta=0).float().div_(S)   # [BH,D,D]
        V = _topr_basis(cov, r, basis_method)               # [BH,D,r]
        c = torch.bmm(Xc, V)                                # [BH,S,r]
        mn = c.amin(-1, keepdim=True); mx = c.amax(-1, keepdim=True)
        cs = _fp8_load(_fp8_store(((mx - mn) / (2 ** coef_bits - 1)).clamp_min(1e-8))).clamp_min(1e-8)
        cz = _fp8_load(_fp8_store(mn))
        cq = torch.clamp(torch.round((c - cz) / cs), 0, 2 ** coef_bits - 1)
        c_hat = cq * cs + cz
        lowrank = torch.bmm(c_hat, V.transpose(1, 2))
        cq8 = cq.to(torch.uint8)
        if r % 4:
            cq8 = torch.nn.functional.pad(cq8, (0, 4 - r % 4))
        out.update(basis=V.to(torch.bfloat16), coef_pack=_pack2(cq8),
                   coef_scale=_fp8_store(cs.squeeze(-1)), coef_zp=_fp8_store(cz.squeeze(-1)))
        res = Xc - lowrank
    else:
        res = Xc
    g = block
    if axis == "channel":
        L = S
        pad = (g - L % g) % g
        if grid == "ternary":
            packed, sc8 = _qp_tern_ch(res, g)
            out.update(res_pack=packed, res_scale=sc8, res_zp=None, res_g=g, res_pad=pad)
        else:
            packed, sc8, zp8 = _qp_asym_ch(res, g)
            out.update(res_pack=packed, res_scale=sc8, res_zp=zp8, res_g=g, res_pad=pad)
        return out
    L = D
    pad = (g - L % g) % g
    t = torch.nn.functional.pad(res, (0, pad)) if pad else res
    Lp = L + pad
    tb = t.reshape(t.shape[0], t.shape[1], Lp // g, g)
    if grid == "ternary":
        packed, sc8 = _qp_tern(tb)
        out.update(res_pack=packed.reshape(t.shape[0], t.shape[1], Lp // 4),
                   res_scale=sc8, res_zp=None, res_g=g, res_pad=pad)
    else:
        packed, sc8, zp8 = _qp_asym(tb)
        out.update(res_pack=packed.reshape(t.shape[0], t.shape[1], Lp // 4),
                   res_scale=sc8, res_zp=zp8, res_g=g, res_pad=pad)
    return out


@torch.no_grad()
def bp_decode(d, dtype=torch.bfloat16):
    B, H, S, D = d["shape"]
    ax_ch = d["axis"] == "channel"
    L = S if ax_ch else D
    rows = D if ax_ch else S
    g = d["res_g"]
    Lp = L + d.get("res_pad", 0)
    tern = d["grid"] == "ternary"
    if d["r"] > 0:
        cq = _unpack2(d["coef_pack"], d["r"]).float()
        c_hat = (cq * _fp8_load(d["coef_scale"]).unsqueeze(-1) * d.get("f_cs", 1.0)
                 + _fp8_load(d["coef_zp"]).unsqueeze(-1) * d.get("f_cz", 1.0))
    else:
        c_hat = None
    if ax_ch:
        out = _dq_full_ch(d["res_pack"].reshape(B * H, rows, Lp // 4),
                          d["res_scale"].reshape(B * H, rows, Lp // g),
                          (d["res_zp"] if not tern else d["res_scale"]).reshape(B * H, rows, Lp // g),
                          tern, g, L, d["mu"], c_hat, d.get("basis"),
                          d.get("f_sc", 1.0), d.get("f_zp", 1.0))
        out = out.reshape(B, H, S, D)
        return out if dtype == torch.bfloat16 else out.to(dtype)
    res = _dq_res(d["res_pack"].reshape(B * H, rows, Lp // 4),
                  d["res_scale"].reshape(B * H, rows, Lp // g),
                  (d["res_zp"] if not tern else d["res_scale"]).reshape(B * H, rows, Lp // g),
                  tern, g, d.get("f_sc", 1.0), d.get("f_zp", 1.0))
    res = res[..., :L]
    out = res + d["mu"].float()
    if c_hat is not None:
        out = out + torch.bmm(c_hat, d["basis"].float().transpose(1, 2))
    return out.reshape(B, H, S, D).to(dtype)


@torch.no_grad()
def bp_encode_packed256(x, r_h1=9, r_h2=0, grid="asym", block=64, axis="channel",
                        grid_h2="ternary", block_h2=64, axis_h2=None,
                        basis_method="subspace"):
    """HY: [B,H,S,256] = rope||prope halves, separate rank/grid/block per half."""
    h1 = bp_encode(x[..., :128].contiguous(), r=r_h1, grid=grid, block=block,
                   axis=axis, basis_method=basis_method)
    h2 = bp_encode(x[..., 128:].contiguous(), r=r_h2, grid=grid_h2, block=block_h2,
                   axis=axis_h2 or axis, basis_method=basis_method)
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
