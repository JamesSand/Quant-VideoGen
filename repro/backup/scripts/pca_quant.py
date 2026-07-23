"""Mean/PCA low-rank + quantized-residual fake-quant for KV cache (plan:
repro/0715/pca-results.md).

Scheme per (batch, head) over a chunk X [S, D]:
  X ≈ mu + quant(coeff) @ V_r^T + quant(residual)
  - mu: token-mean (kept fp)          - V_r: top-r PCA basis of THIS chunk (kept fp)
  - coeff: per-token asym quant (PCA_COEFF_BITS), one scale/zero per token
  - residual: computed AFTER coeff quant (absorbs its error), blockwise over D
    (B=64), grid = ternary symmetric (QVG-aligned) or 4-level asym

Env: PCA_R (default 8; 0 = mean-only), PCA_COEFF_BITS (2), PCA_RES_GRID
(ternary|asym), PCA_V_MODE (mean|pca — what V gets; K always gets PCA_R).
Deterministic; float32 math; TF32 disabled locally.
"""
import os

import torch

PCA_R = int(os.environ.get("PCA_R", "8"))
PCA_KR = int(os.environ.get("PCA_KR", "0")) or None   # K-side rank override
PCA_VR = int(os.environ.get("PCA_VR", "0")) or None   # V-side rank override
PCA_COEFF_BITS = int(os.environ.get("PCA_COEFF_BITS", "2"))
PCA_RES_GRID = os.environ.get("PCA_RES_GRID", "ternary")
RES_AXIS = os.environ.get("PCA_RES_AXIS", "token")
RES_CHNORM = os.environ.get("PCA_RES_CHNORM", "0") == "1"  # per-channel sigma
# normalization of the residual (fp per-channel scale per chunk, amortized ~0)
# before the token-axis grid — captures KIVI's channel-heterogeneity benefit
# without the channel-axis temporal banding.  # token: group along D per token (default);
# channel: group along tokens per channel (KVQuant/KIVI insight — pre-RoPE K esp.)

import contextlib

@contextlib.contextmanager
def _res_grid_override(which):
    """Per-tensor residual grid/block override: PCA_RES_GRID_K/_V, PCA_RES_BLOCK_K/_V.
    Mixed grids (e.g. K asym + V ternary) stay BPE-legal: ternary counted at 2 bits
    (QVG-symmetric accounting), block overheads averaged across K/V."""
    global PCA_RES_GRID, RES_BLOCK, RES_AXIS, RES_CHNORM
    g = os.environ.get(f"PCA_RES_GRID_{which}", "")
    b = os.environ.get(f"PCA_RES_BLOCK_{which}", "")
    a = os.environ.get(f"PCA_RES_AXIS_{which}", "")
    c = os.environ.get(f"PCA_RES_CHNORM_{which}", "")
    if not g and not b and not a and not c:
        yield; return
    prev = (PCA_RES_GRID, RES_BLOCK, RES_AXIS, RES_CHNORM)
    if g: PCA_RES_GRID = g
    if b: RES_BLOCK = int(b)
    if a: RES_AXIS = a
    if c: RES_CHNORM = c == "1"
    try:
        yield
    finally:
        PCA_RES_GRID, RES_BLOCK, RES_AXIS, RES_CHNORM = prev
PCA_V_MODE = os.environ.get("PCA_V_MODE", "mean")
RES_BLOCK = int(os.environ.get("PCA_RES_BLOCK", "64"))
K_BASIS_FILE = os.environ.get("PCA_K_BASIS_FILE", "")
_K_BASIS = None          # [L, H, D, D] eigvecs ascending（OSCAR 式离线校准基）
_LAYER_CTR = [0]

if K_BASIS_FILE:
    _K_BASIS = torch.load(K_BASIS_FILE, map_location="cpu")["eigvecs"]


PCA_FP8SIM = os.environ.get("PCA_FP8SIM", "0") == "1"
PCA_GRID_HASH = os.environ.get("PCA_GRID_HASH", "0") == "1"
PCA_FACTOR_GRID = os.environ.get("PCA_FACTOR_GRID", "0") == "1"
# M0 precheck: simulate storing quantizer metadata (scale/zero) in fp8 E4M3,
# exactly as the real-kernel accounting assumes (kernel-plan.md).

def _fp8(t, per_row=False):
    # normalized fp8 storage. per_row=True (channel-axis residual scales,
    # t: [BH, D, nblk, 1]): one bf16 factor per channel (amortized ~0) —
    # LC's cross-channel scale dynamic range exceeds E4M3's 2^18 and a global
    # factor underflows small-variance channels (the 0720 -3.6dB regression).
    if per_row:
        f = t.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    else:
        f = t.abs().amax().clamp_min(1e-12)
    return (t / f).to(torch.float8_e4m3fn).to(t.dtype) * f

PCA_RES_MSEOPT = os.environ.get("PCA_RES_MSEOPT", "0") == "1"
# N8: per-block MSE-optimal range shrinkage for the asym residual quantizer.
# Searches a few symmetric-about-center shrink ratios per block and keeps the
# min-MSE one; min-max (ratio 1.0) is in the search set, so never worse in
# block MSE. Same (scale, zero) metadata -> BPE unchanged.
_MSE_RATIOS = (1.0, 0.92, 0.85, 0.78, 0.70, 0.62)


def _asym_quant_lastdim_grouped(x, bits, group, mse_opt=None, fp8_per_row=False):
    """Asymmetric fake-quant with one (scale, zero) per group along the last dim."""
    if mse_opt is None:
        mse_opt = PCA_RES_MSEOPT
    S = x.shape
    xg = x.reshape(*S[:-1], S[-1] // group, group)
    mn = xg.amin(dim=-1, keepdim=True)
    mx = xg.amax(dim=-1, keepdim=True)
    if not mse_opt:
        scale = ((mx - mn) / (2 ** bits - 1)).clamp_min(1e-8)
        if PCA_FP8SIM:
            scale = _fp8(scale, per_row=fp8_per_row).clamp_min(1e-8)
            mn = _fp8(mn, per_row=fp8_per_row)
        q = torch.clamp(torch.round((xg - mn) / scale), 0, 2 ** bits - 1)
        return (q * scale + mn).reshape(S)
    ctr = (mx + mn) / 2
    half0 = (mx - mn) / 2
    best_err = None
    best = None
    for r in _MSE_RATIOS:
        lo = ctr - half0 * r
        scale = (half0 * 2 * r / (2 ** bits - 1)).clamp_min(1e-8)
        q = torch.clamp(torch.round((xg - lo) / scale), 0, 2 ** bits - 1)
        deq = q * scale + lo
        err = (deq - xg).pow(2).sum(dim=-1, keepdim=True)
        if best is None:
            best, best_err = deq, err
        else:
            better = err < best_err
            best = torch.where(better, deq, best)
            best_err = torch.minimum(best_err, err)
    return best.reshape(S)


def _ternary_quant_blocked(x, block, fp8_per_row=False):
    """QVG-aligned symmetric ternary over blocks of `block` along the last dim."""
    S = x.shape
    xb = x.reshape(*S[:-1], S[-1] // block, block)
    scale = xb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)  # max_int = 1
    if PCA_FP8SIM:
        scale = _fp8(scale, per_row=fp8_per_row).clamp_min(1e-8)
    q = torch.clamp(torch.round(xb / scale), -1, 1)
    return (q * scale).reshape(S)


# ---- N19: anchor-fidelity event schedule -------------------------------
# PCA_EVENT_SCHED="3:t" -> the FIRST quantize event's residual uses 3-bit
# asym (the revisit/sink anchor gets INT3-level fidelity), later events use
# packed ternary (1.6 b real accounting). Amortized BPE over a video with
# n>=2 events stays under budget; with a single event (LC's cond window)
# the schedule degenerates to the flat default grid (plain N4).
PCA_EVENT_SCHED = os.environ.get("PCA_EVENT_SCHED", "")
_EVENT_IDX = [0]


def _quant_residual(x, _chnorm_done=False):
    if RES_CHNORM and not _chnorm_done:
        # x: [BH, S, D] — equalize per-channel std over this chunk, then quantize
        # the normalized residual on the standard (token-axis) grid.
        sd = x.float().std(dim=-2, keepdim=True).clamp_min(1e-6)   # [BH,1,D]
        q = _quant_residual((x / sd).to(x.dtype), _chnorm_done=True)
        return (q * sd).to(x.dtype)
    if PCA_EVENT_SCHED:
        first, later = PCA_EVENT_SCHED.split(":")
        if _EVENT_IDX[0] == 0:
            if first == "3":
                return _asym_quant_lastdim_grouped(x, 3, RES_BLOCK)
            if first == "4":
                return _asym_quant_lastdim_grouped(x, 4, RES_BLOCK)
        else:
            if later == "t":
                return _ternary_quant_blocked(x, RES_BLOCK)
            if later == "2":
                return _asym_quant_lastdim_grouped(x, 2, RES_BLOCK)
            if later == "1":
                # 1-bit asym {min, max}: accounting-unambiguous (no packing
                # debate) — the schedule wins under ANY convention
                return _asym_quant_lastdim_grouped(x, 1, RES_BLOCK)
    if RES_AXIS == "channel":
        # per-channel residual: group along the token axis. Pad to a full block
        # (M0 erratum fix: the old divisor-fallback silently used g=96 on
        # S=37440 chunks -> scale overhead 0.167 bpe, OVER budget. Padding
        # keeps g=RES_BLOCK and overhead ceil(S/g)*16/S ~= 0.125).
        if PCA_RES_GRID == "zero":
            return torch.zeros_like(x)
        xt = x.transpose(-1, -2).contiguous()
        S_ = xt.shape[-1]
        g = RES_BLOCK
        pad = (g - S_ % g) % g
        if pad:
            xt = torch.nn.functional.pad(xt, (0, pad))
        q = (_ternary_quant_blocked(xt, g, fp8_per_row=True) if PCA_RES_GRID == "ternary"
             else _asym_quant_lastdim_grouped(xt, 2, g, fp8_per_row=True))
        if pad:
            q = q[..., :S_]
        return q.transpose(-1, -2).contiguous()
    if PCA_RES_GRID == "zero":
        return torch.zeros_like(x)   # drop this residual entirely (BPE saver)
    if PCA_RES_GRID == "ternary":
        return _ternary_quant_blocked(x, RES_BLOCK)
    if RES_BLOCK > x.shape[-1]:
        # flattened grouping across tokens (0715: flattened residual is free)
        S = x.shape
        n = x.numel()
        pad = (RES_BLOCK - n % RES_BLOCK) % RES_BLOCK
        xf = torch.nn.functional.pad(x.reshape(-1), (0, pad))
        return _asym_quant_lastdim_grouped(
            xf.reshape(-1, RES_BLOCK), 2, RES_BLOCK)[: n if pad else None].reshape(S) \
            if pad else _asym_quant_lastdim_grouped(
                xf.reshape(-1, RES_BLOCK), 2, RES_BLOCK).reshape(S)
    return _asym_quant_lastdim_grouped(x, 2, RES_BLOCK)


def pca_fake_quant(x, r, fixed_basis=None, res_bits=None, res_bits_t=None):
    """x: [B, H, S, D] -> fake-quantized same shape/dtype. r=0 -> mean-only.
    fixed_basis: [H, D, D] ascending eigvecs -> use its top-r instead of chunk eigh.
    res_bits: [H, D] integer bits (N6) -> per-dim mixed-bit residual.
    res_bits_t: [S] integer bits (N7) -> per-token mixed-bit residual."""
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        B, H, S, D = x.shape
        X = x.reshape(B * H, S, D).float()
        mu = X.mean(dim=1, keepdim=True)                      # [BH,1,D]
        Xc = X - mu
        if r > 0:
            if fixed_basis is not None:
                Vr = fixed_basis.to(Xc.device, torch.float32)[:, :, -r:]  # [H,D,r]
                Vr = Vr.repeat(B, 1, 1) if B > 1 else Vr
            else:
                cov = torch.einsum("bsd,bse->bde", Xc, Xc) / S    # [BH,D,D]
                _, vecs = torch.linalg.eigh(cov)                  # ascending
                Vr = vecs[:, :, -r:]                              # [BH,D,r] top-r
            c = torch.einsum("bsd,bdr->bsr", Xc, Vr)          # [BH,S,r]
            c_hat = _asym_quant_lastdim_grouped(c, PCA_COEFF_BITS, r)
            lowrank = torch.einsum("bsr,bdr->bsd", c_hat, Vr)
        else:
            lowrank = torch.zeros_like(Xc)
        res = Xc - lowrank
        if res_bits is not None:
            bits_bh = res_bits.to(res.device).repeat(B, 1)          # [BH, D]
            res_hat = _asym_quant_mixedbits(res, bits_bh, RES_BLOCK)
        elif res_bits_t is not None:
            res_hat = _asym_quant_tokenbits(res, res_bits_t, RES_BLOCK)
        else:
            res_hat = _quant_residual(res)
        out = mu + lowrank + res_hat
        return out.reshape(B, H, S, D).to(x.dtype)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32


PCA_SPLIT_D = int(os.environ.get("PCA_SPLIT_D", "0"))  # >0: split D into
# sub-heads of this width before PCA (e.g. 128 on HY's 256-dim heads so the
# subspace granularity matches the LC/SF setup). 0 = off.

PCA_QW_MODE = os.environ.get("PCA_QW_MODE", "scale")  # "scale" (N5) | "bits" (N6)
# N6 "bits": keep the residual's shared per-token min/max scale but reallocate
# integer bits per dim under the same total budget (2*D), greedily by online
# Q-energy (Block-GTQ rule: pick argmax w_d * (3/4) * 4^-b). No range
# stretching (N5's failure mode), no extra metadata, BPE unchanged.


def _greedy_bits(w, budget_per_dim=2, bmin=1, bmax=4):
    """w: [H, D] -> integer bits [H, D], sum per head ≈ budget_per_dim * D.
    budget_per_dim may be fractional (spends the BPE headroom exactly)."""
    H, D = w.shape
    b = torch.full((H, D), bmin, dtype=torch.long, device=w.device)
    extra = int(round((budget_per_dim - bmin) * D))
    gain = w * (4.0 ** (-bmin))                       # ∝ marginal gain
    for _ in range(extra):
        idx = gain.argmax(dim=-1)                     # [H]
        rows = torch.arange(H, device=w.device)
        b[rows, idx] += 1
        gain[rows, idx] = torch.where(
            b[rows, idx] >= bmax, torch.zeros_like(idx, dtype=w.dtype),
            w[rows, idx] * (4.0 ** (-b[rows, idx].float())))
    return b


def _asym_quant_mixedbits(x, bits, group):
    """x: [BH, S, D]; bits: [BH, D] integer per-dim bit widths. Shared per-
    (token, group) min/max exactly like the uniform quantizer (same metadata),
    only the per-dim level count varies."""
    BH, S, D = x.shape
    xg = x.reshape(BH, S, D // group, group)
    bg = bits.reshape(BH, 1, D // group, group).float()
    mn = xg.amin(dim=-1, keepdim=True)
    mx = xg.amax(dim=-1, keepdim=True)
    rng = (mx - mn).clamp_min(1e-8)
    levels = (2.0 ** bg - 1).clamp_min(1.0)
    q = torch.clamp(torch.round((xg - mn) / rng * levels), torch.zeros_like(levels), levels)
    return (q / levels * rng + mn).reshape(BH, S, D)


# ---- N5: online Q-energy-weighted K quantization (OSCAR idea, online) ----
# PCA_QW_ALPHA > 0 enables it. The launcher captures per-(layer, head, dim)
# running E[q^2] during generation (free — no calibration pass) and registers
# a provider; K is scaled dim-wise by importance^{alpha/2} before the N4
# pipeline and unscaled after, so basis + residual quantizer minimize the
# attention-logit-weighted error instead of plain MSE. V is untouched.
# Storage cost of the real implementation: one extra fp16 D-vector per
# (head, chunk) — same amortization class as mu.
PCA_QW_ALPHA = float(os.environ.get("PCA_QW_ALPHA", "0"))
_QW_PROVIDER = [None]   # fn(call_idx) -> [H, D] mean q^2 (any device) or None
_QW_CTR = [0]

# ---- Q2: W_O-metric V shaping. The true V error metric is ||W_O dV||, not
# ||dV||: whiten V by G^{1/2} (G = per-head Gram of the output projection
# block, a MODEL WEIGHT — decoder-reproducible, zero storage), run the N4
# pipeline in whitened space, unwhiten on decode.
PCA_V_METRIC = os.environ.get("PCA_V_METRIC", "")      # "" | "wo"
_VW_PROVIDER = [None]   # fn(call_idx) -> (W [H,D,D], Winv [H,D,D]) or None


def set_vw_provider(fn):
    _VW_PROVIDER[0] = fn

# ---- N7: temporal (per-frame) bit reallocation by future-retrieval prob ----
# The launcher computes, at each quantize event, per-token integer bits from
# the FOV overlap of the quantized frames' cameras with all FUTURE planned
# cameras (the action script is known upfront). Same shared scale metadata,
# budget mean = 2 bits — BPE unchanged. Applied to BOTH K and V residuals
# (revisit retrieval reads both). Models without a pose signal degenerate to
# plain N4 by construction.
_TW_BITS = [None]


def set_tw_bits(bits_t):
    _TW_BITS[0] = bits_t


def _asym_quant_tokenbits(x, bits_t, group):
    """x: [BH, S, D]; bits_t: [S] integer bit widths per token."""
    BH, S, D = x.shape
    xg = x.reshape(BH, S, D // group, group)
    lv = (2.0 ** bits_t.to(x.device).float().view(1, S, 1, 1) - 1).clamp_min(1.0)
    mn = xg.amin(dim=-1, keepdim=True)
    mx = xg.amax(dim=-1, keepdim=True)
    rng = (mx - mn).clamp_min(1e-8)
    q = torch.clamp(torch.round((xg - mn) / rng * lv), torch.zeros_like(lv), lv)
    return (q / lv * rng + mn).reshape(BH, S, D)


def set_qw_provider(fn):
    _QW_PROVIDER[0] = fn


PCA_QW_PAIR = os.environ.get("PCA_QW_PAIR", "")  # ""|"half"|"interleave":
# average importance across RoPE pairs so the weighting is invariant to the
# relative rotation (LC/SF rotate_half -> "half"; HY interleaved -> "interleave").


def _qw_pair_avg(w):
    D = w.shape[-1]
    if PCA_QW_PAIR == "half":
        a = (w[..., : D // 2] + w[..., D // 2:]) / 2
        return torch.cat([a, a], dim=-1)
    if PCA_QW_PAIR == "interleave":
        a = (w[..., 0::2] + w[..., 1::2]) / 2
        return torch.stack([a, a], dim=-1).reshape(*w.shape)
    return w


def _qw_scale(w):
    """[H, D] mean q^2 -> per-head tempered scale vector with geometric mean 1."""
    w = _qw_pair_avg(w)
    w = w / w.mean(dim=-1, keepdim=True).clamp_min(1e-12)
    s = w.clamp_min(1e-4) ** (PCA_QW_ALPHA / 2)
    return s / torch.exp(torch.log(s).mean(dim=-1, keepdim=True))


def _split_d(x):
    B, H, S, D = x.shape
    n = D // PCA_SPLIT_D
    return x.reshape(B, H, S, n, PCA_SPLIT_D).permute(0, 1, 3, 2, 4).reshape(B, H * n, S, PCA_SPLIT_D)


def _unsplit_d(x, H, D):
    B, Hn, S, d = x.shape
    n = D // d
    return x.reshape(B, H, n, S, d).permute(0, 1, 3, 2, 4).reshape(B, H, S, D)


# ---- N13: KLT transform coding (no k-means; PCA rotation + eigenvalue-ordered
# greedy bit allocation — the OSCAR "important directions get more budget" idea
# done online per chunk, as ROTATION not subtraction). Per (head, chunk):
# Y=(X-mu)V (V = self-cov eigenbasis, stored fp8+scale; HY's packed 256-dim
# rows rotate as two 128 halves), per-dim bits by greedy on eigenvalues
# (budget mean = PCA_KLT_BUDGET, b in [0,4]), quantized per (dim, 128-token
# block) asym with fp8 scale/zp. BPE: 2 + 0.125 + basis (LC .035/HY .145/SF .027).
PCA_KLT_BUDGET = float(os.environ.get("PCA_KLT_BUDGET", "2"))
PCA_KLT_SPLIT = int(os.environ.get("PCA_KLT_SPLIT", "0"))
PCA_KLT_SBLOCK = int(os.environ.get("PCA_KLT_SBLOCK", "128"))
PCA_KLT_GRID = os.environ.get("PCA_KLT_GRID", "uniform")  # "uniform" | "lm"
PCA_KLT_BMIN = int(os.environ.get("PCA_KLT_BMIN", "0"))
# bmin=1 keeps every rotated dim alive (b=0 deletes directions — HY's revisit
# retrieval punishes that hard).

# ---- N16 (idea #3, DeltaQuant-inspired): spatio-temporal cube-mean core ----
# Tokens are ordered frame-major/spatial within the chunk, so contiguous
# groups of C tokens ≈ local cubes. Core = group mean stored fp8 (per-head
# scale, same NaN guard) — the position-implied "free centroid": no index
# bits at all. Residual = x - core via the proven asym grid.
# BPE: 8/C core + residual(2 + 16/RES_BLOCK); C=64,B=128 -> 2.25.
PCA_CUBE_C = int(os.environ.get("PCA_CUBE_C", "64"))


def _cube_fake_quant(x4):
    x = x4[0].float()                                     # [H, S, D]
    H, S, D = x.shape
    C = PCA_CUBE_C
    ng = (S + C - 1) // C
    pad = ng * C - S
    xp = torch.nn.functional.pad(x, (0, 0, 0, pad))
    xg = xp.reshape(H, ng, C, D)
    core = xg.mean(2, keepdim=True)                       # [H, ng, 1, D]
    sc = (core.abs().amax(dim=(1, 2, 3), keepdim=True) / 440.0).clamp_min(1e-8)
    core = (core / sc).to(torch.float8_e4m3fn).float() * sc
    res = (xg - core).reshape(H, ng * C, D)[:, :S]
    out = core.expand_as(xg).reshape(H, ng * C, D)[:, :S] + _quant_residual(res)
    return out.unsqueeze(0).to(x4.dtype)
# "lm" (N14): std-normalized Gaussian Lloyd-Max codebooks per dim — the
# constructive density-adaptive centroids (user idea #1). Zero table storage;
# per-(dim, token-block) mean/std replace min/zero-point (same 0.125 bpe).
_LM_CODES = {
    1: [-0.7980, 0.7980],
    2: [-1.5104, -0.4528, 0.4528, 1.5104],
    3: [-2.1520, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1520],
    4: [-2.7326, -2.0690, -1.6181, -1.2562, -0.9424, -0.6568, -0.3881, -0.1284,
        0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6181, 2.0690, 2.7326],
}


def _lm_nearest(x, codes):
    mid = (codes[1:] + codes[:-1]) / 2
    return codes[torch.bucketize(x, mid)]


def _klt_fake_quant(x4):
    """x4: [1, H, S, D] -> transform-coded recon, same shape/dtype."""
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        x = x4[0].float()
        H, S, D = x.shape
        if PCA_KLT_SPLIT and D > PCA_KLT_SPLIT:
            n = D // PCA_KLT_SPLIT
            xs = (x.reshape(H, S, n, PCA_KLT_SPLIT).permute(0, 2, 1, 3)
                  .reshape(H * n, S, PCA_KLT_SPLIT))
            out = _klt_core(xs)
            out = (out.reshape(H, n, S, PCA_KLT_SPLIT).permute(0, 2, 1, 3)
                   .reshape(H, S, D))
        else:
            out = _klt_core(x)
        return out.unsqueeze(0).to(x4.dtype)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32


def _klt_core(x):
    H, S, D = x.shape
    mu = x.mean(1, keepdim=True)
    Xc = x - mu
    cov = torch.einsum("hsd,hse->hde", Xc, Xc) / S
    lam, V = torch.linalg.eigh(cov)
    lam, V = lam.flip(-1), V.flip(-1)                     # descending
    sc = (V.abs().amax(dim=(1, 2), keepdim=True) / 440.0).clamp_min(1e-8)
    V = (V / sc).to(torch.float8_e4m3fn).float() * sc     # fp8-stored basis
    Y = torch.einsum("hsd,hdr->hsr", Xc, V)
    bits = _greedy_bits(lam.clamp_min(0), budget_per_dim=PCA_KLT_BUDGET,
                        bmin=PCA_KLT_BMIN, bmax=4)        # [H, D]
    bs = PCA_KLT_SBLOCK
    nb = (S + bs - 1) // bs
    Yp = torch.nn.functional.pad(Y, (0, 0, 0, nb * bs - S))
    Yb = Yp.reshape(H, nb, bs, D)
    if PCA_KLT_GRID in ("lm", "lm_mm", "mix"):
        deq = torch.zeros_like(Yb)
        mean = Yb.mean(2, keepdim=True)
        std = Yb.std(2, keepdim=True).clamp_min(1e-8)
        mn = Yb.amin(2, keepdim=True)
        mx = Yb.amax(2, keepdim=True)
        z = (Yb - mean) / std
        for b in range(1, 5):
            m = (bits == b)
            if not m.any():
                continue
            if PCA_KLT_GRID == "mix" and b >= 3:
                # high-bit (outlier-carrying leading) dims: exact-range uniform
                rng = (mx - mn).clamp_min(1e-8)
                lv = 2.0 ** b - 1
                d_b = torch.clamp(torch.round((Yb - mn) / rng * lv), 0, lv) / lv * rng + mn
            else:
                codes = torch.tensor(_LM_CODES[b], device=Yb.device, dtype=Yb.dtype)
                d_b = _lm_nearest(z, codes) * std + mean
                if PCA_KLT_GRID in ("lm_mm", "mix"):
                    # outlier preservation: values snapped to the outermost code
                    # are replaced by the block's true min/max (metadata reuse:
                    # min/max stored instead of the outer pair — same budget)
                    hi = z > (codes[-1] + codes[-2]) / 2 * 1.0
                    lo = z < (codes[0] + codes[1]) / 2 * 1.0
                    d_b = torch.where(hi, mx.expand_as(d_b), d_b)
                    d_b = torch.where(lo, mn.expand_as(d_b), d_b)
            deq = torch.where(m.view(H, 1, 1, D).expand_as(Yb), d_b, deq)
    else:
        mn = Yb.amin(2, keepdim=True)
        mx = Yb.amax(2, keepdim=True)
        rng = (mx - mn).clamp_min(1e-8)
        lv = (2.0 ** bits.float().view(H, 1, 1, D) - 1)
        q = torch.where(lv > 0,
                        torch.clamp(torch.round((Yb - mn) / rng * lv),
                                    torch.zeros_like(lv), lv),
                        torch.zeros_like(Yb))
        deq = torch.where(lv > 0, q / lv.clamp_min(1.0) * rng + mn,
                          torch.zeros_like(Yb))
    Yq = deq.reshape(H, nb * bs, D)[:, :S]
    return torch.einsum("hsr,hdr->hsd", Yq, V) + mu


# ---- N10: K-side small-dictionary SAS (diagnosis: HY keys want a dictionary,
# not a subspace; V stays N4). Table policy governs the BPE cost:
# PCA_SAS_REFIT=n -> refit+store the centroid table every n-th quantize event
# (warm-started), reuse frozen in between; amortized table cost = K*16/(N*n).
PCA_K_MODE = os.environ.get("PCA_K_MODE", "pca")      # "pca" | "sas"
PCA_SAS_K = int(os.environ.get("PCA_SAS_K", "256"))
PCA_SAS_ITERS = int(os.environ.get("PCA_SAS_ITERS", "20"))
PCA_SAS_REFIT = int(os.environ.get("PCA_SAS_REFIT", "1"))
PCA_SAS_TAB8 = os.environ.get("PCA_SAS_TAB8", "0") == "1"
PCA_N_LAYERS = int(os.environ.get("PCA_N_LAYERS", "0"))   # required for sas
_SAS_TABLES = {}                                          # layer -> [H, K, D]


def _sas_fake_quant_k(x, layer, event_idx):
    """x: [1, H, S, D] -> dictionary-smoothed + quantized-residual recon."""
    from quant_videogen.kmeans.kmeans_euclid import batch_kmeans_Euclid
    xf = x[0].float()                                     # [H, S, D]
    refit = (event_idx % PCA_SAS_REFIT == 0) or (layer not in _SAS_TABLES)
    if refit:
        init = _SAS_TABLES.get(layer)
        _, cent = batch_kmeans_Euclid(
            xf, PCA_SAS_K, max_iters=PCA_SAS_ITERS, init_centroids=init)[:2]
        if PCA_SAS_TAB8:
            # store the table in fp8 (E4M3) with a per-head fp16 scale —
            # LC's deep-layer V centroids reach |1440| > E4M3's 448 max and
            # e4m3fn overflows to NaN without it. Residual is computed against
            # the dequantized table, so decode stays consistent.
            sc = (cent.abs().amax(dim=(1, 2), keepdim=True) / 440.0).clamp_min(1e-8)
            cent = (cent / sc).to(torch.float8_e4m3fn).float() * sc
        _SAS_TABLES[layer] = cent
    else:
        cent = _SAS_TABLES[layer]
    a = torch.cdist(xf, cent).argmin(-1)                  # [H, S]
    sm = torch.gather(cent, 1, a.unsqueeze(-1).expand(-1, -1, xf.shape[-1]))
    out = sm + _quant_residual(xf - sm)
    return out.unsqueeze(0).to(x.dtype)


PCA_SUBCHUNK = int(os.environ.get("PCA_SUBCHUNK", "1"))
# N9: fit mu/basis (and residual blocks) on temporal sub-chunks — the KV
# distribution drifts within a chunk; finer statistics fit it better. Costs
# one extra (mu + basis) per extra sub-chunk: BPE +0.003 (LC) / +0.011 (HY
# at 2 sub-chunks), still < 2.326.


def _subchunk_apply(x, fn, n):
    S = x.shape[2]
    cuts = [S * i // n for i in range(n + 1)]
    return torch.cat([fn(x[:, :, cuts[i]:cuts[i + 1]]) for i in range(n)], dim=2)


_ROPE_TAB = {}


def _lc_rope_cos_sin(S, D, device):
    """LC rope_3d frequency table (window-relative grid from PCA_ROPE_GRID)."""
    key = (S, D)
    if key in _ROPE_TAB:
        cos, sin = _ROPE_TAB[key]
        return cos.to(device), sin.to(device)
    T, Hh, W = (int(x) for x in os.environ.get("PCA_ROPE_GRID", "19,30,52").split(","))
    assert T * Hh * W == S, f"rope grid {T}x{Hh}x{W} != S={S}"
    dim_t = D - 4 * (D // 6)
    dim_h = 2 * (D // 6)
    dim_w = 2 * (D // 6)
    def fr(dim):
        return 1.0 / (10000 ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    ft = torch.einsum("t,f->tf", torch.arange(T).float(), fr(dim_t)).repeat_interleave(2, -1)
    fh = torch.einsum("h,f->hf", torch.arange(Hh).float(), fr(dim_h)).repeat_interleave(2, -1)
    fw = torch.einsum("w,f->wf", torch.arange(W).float(), fr(dim_w)).repeat_interleave(2, -1)
    freqs = torch.cat([
        ft[:, None, None, :].expand(T, Hh, W, -1),
        fh[None, :, None, :].expand(T, Hh, W, -1),
        fw[None, None, :, :].expand(T, Hh, W, -1)], dim=-1).reshape(S, D)
    cos, sin = freqs.cos(), freqs.sin()
    _ROPE_TAB[key] = (cos, sin)
    return cos.to(device), sin.to(device)


def _rot_half(x):
    x2 = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    a, b = x2.unbind(-1)
    return torch.stack((-b, a), dim=-1).reshape(x.shape)


def _perchannel_quant(kt_in, grid, g=128):
    """kt_in [B,H,D,S] -> per-channel blockwise quant along tokens (padded)."""
    S_ = kt_in.shape[-1]
    pad = (g - S_ % g) % g
    kt = torch.nn.functional.pad(kt_in, (0, pad)) if pad else kt_in
    if grid == "ternary":
        q = _ternary_quant_blocked(kt, g, fp8_per_row=True)
    else:
        q = _asym_quant_lastdim_grouped(kt, 2, g, mse_opt=False, fp8_per_row=True)
    return q[..., :S_] if pad else q


def _kivi_paper_fake_quant_kv(k, v):
    """Paper-style KIVI hypothesis: per-channel THREE-LEVEL SYMMETRIC grid
    applied in POST-RoPE coordinates (R -> quant -> R^-1; the pipeline's
    read-time rope re-applies R, so quantization acted post-RoPE)."""
    B, H, S, D = k.shape
    cos, sin = _lc_rope_cos_sin(S, D, k.device)
    kf = k.float()
    kr = kf * cos + _rot_half(kf) * sin                     # R(k)
    ktq = _perchannel_quant(kr.transpose(-1, -2).contiguous(), "ternary").transpose(-1, -2)
    kq = ktq * cos - _rot_half(ktq) * sin                   # R^-1
    v_q = _ternary_quant_blocked(v.float(), 64)             # V per-token 3-level
    return kq.to(k.dtype), v_q.to(v.dtype)


def _qvgpro_fake_quant_kv(k, v):
    """QVG-Pro 反事实臂(0721 夜,reviewer-proof):QVG kmeans 减法 + BF16 质心
    + 四电平 asym B128 残差格——按名义口径合法的最强 kmeans 变体。
    合法性:LC 仅 token 轴合法(2.3257;channel 轴因 S 补零 2.3295 超线);
    SF 两轴均合法(token 2.297 / channel 2.300);HY 2.738 超线(不跑)。
    PCA_QVGPRO_AXIS=token|channel;PCA_QVGPRO_ITERS(LC 100 / SF 2)。"""
    from quant_videogen.kmeans.kmeans_euclid import batch_kmeans_Euclid
    iters = int(os.environ.get("PCA_QVGPRO_ITERS", "100"))
    axis = os.environ.get("PCA_QVGPRO_AXIS", "token")
    outs = []
    for x in (k, v):
        B, H, S, D = x.shape
        X = x.float().view(B * H, S, D).contiguous()
        lab, cent, _, _ = batch_kmeans_Euclid(X, n_clusters=256, max_iters=iters)
        cent = cent.to(torch.bfloat16).float()          # 质心实存 BF16(名义口径)
        g = torch.gather(cent, 1, lab.long().unsqueeze(-1).expand(-1, -1, D))
        res = (X - g).view(B, H, S, D)
        if axis == "channel":
            rq = _perchannel_quant(res.transpose(-1, -2).contiguous(), "asym", g=128).transpose(-1, -2)
        else:
            # token 轴:fp8 s/z + 全局归一因子(per_row=False;逐 token 因子须计
            # 16/128=0.125,预算装不下——与我们 HY V 的 token 轴记账同规)
            rq = _asym_quant_lastdim_grouped(res, 2, 128, mse_opt=False, fp8_per_row=False)
        outs.append((g.view(B, H, S, D) + rq).to(x.dtype))
    return outs[0], outs[1]


def _qvg4_fake_quant_kv(k, v):
    """反事实臂(0721):QVG 原装 kmeans 减法(per-head 全 D 维,K=256,LC 官配
    iters=100)不动,残差格由其三电平对称换成【四电平非对称 B64 + fp8 s/z】
    ——回答"QVG 若把 2-bit 的四个码字用满会怎样"。BPE 代价:残差 s/z 从
    0.125 涨到 0.25(B64 asym 双参数),质心/索引账不变。"""
    from quant_videogen.kmeans.kmeans_euclid import batch_kmeans_Euclid
    iters = int(os.environ.get("PCA_QVG4_ITERS", "100"))
    outs = []
    for x in (k, v):
        B, H, S, D = x.shape
        X = x.float().view(B * H, S, D).contiguous()
        lab, cent, _, _ = batch_kmeans_Euclid(X, n_clusters=256, max_iters=iters)
        g = torch.gather(cent, 1, lab.long().unsqueeze(-1).expand(-1, -1, D))
        res = (X - g).view(B, H, S, D)
        rq = _asym_quant_lastdim_grouped(res, 2, 64, mse_opt=False, fp8_per_row=True)
        outs.append((g.view(B, H, S, D) + rq).to(x.dtype))
    return outs[0], outs[1]


def _kivi_post_fake_quant_kv(k, v):
    """归因分离臂(0720):满血 KIVI 格(K per-channel asym g128 / V per-token
    asym g64)但施加在 POST-RoPE 坐标(R -> quant -> R^-1)。与诚实 KIVI 臂
    (kivifp8)的唯一差异 = 量化位置,用于分离 pre/post-RoPE 对满血 KIVI 的价值。"""
    B, H, S, D = k.shape
    cos, sin = _lc_rope_cos_sin(S, D, k.device)
    kf = k.float()
    kr = kf * cos + _rot_half(kf) * sin                     # R(k)
    ktq = _perchannel_quant(kr.transpose(-1, -2).contiguous(), "asym").transpose(-1, -2)
    kq = ktq * cos - _rot_half(ktq) * sin                   # R^-1
    v_q = _asym_quant_lastdim_grouped(v, 2, 64, mse_opt=False)
    return kq.to(k.dtype), v_q.to(v.dtype)


def _kivi_fake_quant_kv(k, v):
    """KIVI baseline (Liu et al. 2024) core mechanism: K quantized PER-CHANNEL
    (asym 2-bit, groups of 128 along the token axis), V PER-TOKEN (asym 2-bit,
    groups of 64 along channels). The FP16 recent-window of KIVI corresponds to
    the pipelines' native BF16 recent window (aged-chunk quantization)."""
    grid = os.environ.get("PCA_KIVI_GRID", "asym")
    k_q = _perchannel_quant(k.transpose(-1, -2).contiguous(), grid).transpose(-1, -2).contiguous()
    if grid == "ternary":
        v_q = _ternary_quant_blocked(v.float(), 64).to(v.dtype)
    else:
        v_q = _asym_quant_lastdim_grouped(v, 2, 64, mse_opt=False)
    return k_q.to(k.dtype), v_q.to(v.dtype)


_DUMP_DIR = os.environ.get("PCA_DUMP_DIR", "")
_DUMP_N = int(os.environ.get("PCA_DUMP_N", "8"))


@torch.no_grad()
def _grid_hash_fake_quant_kv(k, v):
    """Decode the actual packed PCA-grid hash payload for end-to-end tests."""
    import sys
    for path in ("repro/0721", "repro/0720/kernel"):
        if path not in sys.path:
            sys.path.insert(0, path)
    from pca_grid_hash import grid_hash_encode_kv_fast
    from pca_grid_hash_triton import triton_grid_hash_decode

    if k.shape[-1] == 256:
        cfg = dict(iters=1, refine=0, shared_labels="v_rope")
    elif k.shape[-2] >= 35000:
        cfg = dict(iters=1, refine=0, shared_labels="v")
    else:
        cfg = dict(iters=5, refine=1, shared_labels=None)
    state = grid_hash_encode_kv_fast(k, v, **cfg)
    return (
        triton_grid_hash_decode(state["k"]).to(k.dtype),
        triton_grid_hash_decode(state["v"]).to(v.dtype),
    )


@torch.no_grad()
def _factor_grid_real_quant_kv(k, v):
    """Actual packed factor-only product grid used by the bounded fallback."""
    import sys
    for path in ("repro/0721", "repro/0720/kernel"):
        if path not in sys.path:
            sys.path.insert(0, path)
    from bp_quant import bp_encode_fast
    from factor_grid_triton import triton_factor_decode

    cfg = dict(
        r=4,
        grid="asym",
        block=128,
        axis="channel",
        iters=2,
    )
    ks, vs = bp_encode_fast(k, **cfg), bp_encode_fast(v, **cfg)
    return (
        triton_factor_decode(ks, bs=64, bd=64).to(k.dtype),
        triton_factor_decode(vs, bs=64, bd=64).to(v.dtype),
    )


def pca_fake_quant_kv(k, v):
    call_idx = _QW_CTR[0]
    _QW_CTR[0] += 1
    if _DUMP_DIR and call_idx < _DUMP_N:
        os.makedirs(_DUMP_DIR, exist_ok=True)
        torch.save({"k": k.detach().to(torch.bfloat16).cpu(),
                    "v": v.detach().to(torch.bfloat16).cpu()},
                   f"{_DUMP_DIR}/chunk_{call_idx:03d}.pt")
    if PCA_FACTOR_GRID:
        if not getattr(pca_fake_quant_kv, "_factor_grid_announced", False):
            pca_fake_quant_kv._factor_grid_announced = True
            print(
                "[pca_quant] factor-only product grid active: rank4 INT2 "
                "coefficients + channel-axis asymmetric INT2",
                flush=True,
            )
        return _factor_grid_real_quant_kv(k, v)
    if PCA_GRID_HASH:
        if not getattr(pca_fake_quant_kv, "_grid_hash_announced", False):
            pca_fake_quant_kv._grid_hash_announced = True
            print(
                "[pca_quant] PCA-grid hash active: analytic uint8 labels + "
                "FP8 table + channel-axis asymmetric INT2",
                flush=True,
            )
        return _grid_hash_fake_quant_kv(k, v)
    if os.environ.get("PCA_QVGPRO", "") == "1":
        if not getattr(pca_fake_quant_kv, "_qvgpro_announced", False):
            pca_fake_quant_kv._qvgpro_announced = True
            print(f"[pca_quant] QVG-Pro: kmeans+BF16 centroids + asym B128 axis={os.environ.get('PCA_QVGPRO_AXIS','token')}", flush=True)
        return _qvgpro_fake_quant_kv(k, v)
    if os.environ.get("PCA_QVG4", "") == "1":
        if not getattr(pca_fake_quant_kv, "_qvg4_announced", False):
            pca_fake_quant_kv._qvg4_announced = True
            print("[pca_quant] QVG-4pt counterfactual: kmeans sub + ASYM 4-level B64 residual", flush=True)
        return _qvg4_fake_quant_kv(k, v)
    if os.environ.get("PCA_KIVI_POST", "") == "1":
        if not getattr(pca_fake_quant_kv, "_kpost_announced", False):
            pca_fake_quant_kv._kpost_announced = True
            print("[pca_quant] KIVI-POST mode: post-RoPE per-channel ASYM g128", flush=True)
        return _kivi_post_fake_quant_kv(k, v)
    if os.environ.get("PCA_KIVI_PAPER", "") == "1":
        if not getattr(pca_fake_quant_kv, "_kp_announced", False):
            pca_fake_quant_kv._kp_announced = True
            print("[pca_quant] PAPER-KIVI hypothesis mode: post-RoPE per-channel ternary", flush=True)
        return _kivi_paper_fake_quant_kv(k, v)
    if os.environ.get("PCA_KIVI", "") == "1":
        if not getattr(pca_fake_quant_kv, "_kivi_announced", False):
            pca_fake_quant_kv._kivi_announced = True
            print("[pca_quant] KIVI baseline mode: K per-channel g128 / V per-token g64", flush=True)
        return _kivi_fake_quant_kv(k, v)
    if os.environ.get("PCA_RTN", "") == "1":
        # RTN baseline: plain per-token blockwise 2-bit (== fork's naive-int2
        # math), routed through this launcher for the SF BSHD store fix.
        if not getattr(pca_fake_quant_kv, "_rtn_announced", False):
            pca_fake_quant_kv._rtn_announced = True
            print(f"[pca_quant] RTN baseline mode: {os.environ.get('PCA_RTN_GRID','asym')} 2-bit B64 per-token", flush=True)
        if os.environ.get("PCA_RTN_GRID", "asym") == "ternary":
            return (_ternary_quant_blocked(k.float(), 64).to(k.dtype),
                    _ternary_quant_blocked(v.float(), 64).to(v.dtype))
        return (_asym_quant_lastdim_grouped(k, 2, 64, mse_opt=False).to(k.dtype),
                _asym_quant_lastdim_grouped(v, 2, 64, mse_opt=False).to(v.dtype))
    if PCA_EVENT_SCHED and PCA_N_LAYERS > 0:
        _EVENT_IDX[0] = call_idx // PCA_N_LAYERS
        if call_idx % PCA_N_LAYERS == 0:
            print(f"[pca_quant] N19 sched event={_EVENT_IDX[0]} "
                  f"grid={'first' if _EVENT_IDX[0]==0 else 'later'}", flush=True)
    if PCA_K_MODE == "cube":                                         # N16
        if not getattr(pca_fake_quant_kv, "_cube_announced", False):
            pca_fake_quant_kv._cube_announced = True
            print(f"[pca_quant] N16 cube-mean active: C={PCA_CUBE_C} "
                  f"res_block={RES_BLOCK} v_mode={PCA_V_MODE}", flush=True)
        k_q = _cube_fake_quant(k)
        if PCA_V_MODE == "cube":
            v_q = _cube_fake_quant(v)
        elif PCA_V_MODE == "klt":
            v_q = _klt_fake_quant(v)
        else:
            v_q = pca_fake_quant(v, PCA_R if PCA_V_MODE == "pca" else 0)
        return k_q, v_q
    if PCA_K_MODE == "klt":                                          # N13
        if not getattr(pca_fake_quant_kv, "_klt_announced", False):
            pca_fake_quant_kv._klt_announced = True
            print(f"[pca_quant] N13 KLT active: budget={PCA_KLT_BUDGET} "
                  f"split={PCA_KLT_SPLIT} sblock={PCA_KLT_SBLOCK} "
                  f"v_mode={PCA_V_MODE}", flush=True)
        k_q = _klt_fake_quant(k)
        if PCA_V_MODE == "klt":
            v_q = _klt_fake_quant(v)
        else:
            v_q = pca_fake_quant(v, PCA_R if PCA_V_MODE == "pca" else 0)
        return k_q, v_q
    if PCA_K_MODE == "sas":                                          # N10
        assert PCA_N_LAYERS > 0, "PCA_N_LAYERS required for sas K mode"
        layer = call_idx % PCA_N_LAYERS
        event_idx = call_idx // PCA_N_LAYERS
        if not getattr(pca_fake_quant_kv, "_sas_announced", False):
            pca_fake_quant_kv._sas_announced = True
            print(f"[pca_quant] N10 K<-SAS active: K={PCA_SAS_K} iters={PCA_SAS_ITERS} "
                  f"refit_every={PCA_SAS_REFIT} layers={PCA_N_LAYERS}", flush=True)
        k_q = _sas_fake_quant_k(k, layer, event_idx)
        if PCA_V_MODE == "sas":                                      # N12
            v_q = _sas_fake_quant_k(v, PCA_N_LAYERS + layer, event_idx)
        else:
            v_q = pca_fake_quant(v, PCA_R if PCA_V_MODE == "pca" else 0)
        return k_q, v_q
    if PCA_SUBCHUNK > 1:                                            # N9
        if not getattr(pca_fake_quant_kv, "_sc_announced", False):
            pca_fake_quant_kv._sc_announced = True
            print(f"[pca_quant] N9 sub-chunk stats active: n={PCA_SUBCHUNK}", flush=True)
        k_q = _subchunk_apply(k, lambda t: pca_fake_quant(t, PCA_R), PCA_SUBCHUNK)
        v_q = _subchunk_apply(
            v, lambda t: pca_fake_quant(t, PCA_R if PCA_V_MODE == "pca" else 0),
            PCA_SUBCHUNK)
        return k_q, v_q
    bt = _TW_BITS[0]
    if bt is not None and bt.numel() == k.shape[2]:                 # N7
        if not getattr(pca_fake_quant_kv, "_tw_announced", False):
            pca_fake_quant_kv._tw_announced = True
            hist = torch.bincount(bt.flatten(), minlength=5).tolist()
            print(f"[pca_quant] TW-N7 active: token-bits histogram(0-4)={hist} "
                  f"mean={bt.float().mean():.3f}", flush=True)
        k_q = pca_fake_quant(k, PCA_R, res_bits_t=bt)
        v_q = pca_fake_quant(v, PCA_R if PCA_V_MODE == "pca" else 0, res_bits_t=bt)
        return k_q, v_q
    if PCA_QW_ALPHA > 0 and _QW_PROVIDER[0] is not None:
        w = _QW_PROVIDER[0](call_idx)
        if w is None and not getattr(pca_fake_quant_kv, "_qw_none_announced", False):
            pca_fake_quant_kv._qw_none_announced = True
            print(f"[pca_quant] QW WARNING: provider returned None at call {call_idx} "
                  "(no q-stats yet) — falling back to plain N4 for this event", flush=True)
        if w is not None:
            assert _K_BASIS is None and not PCA_SPLIT_D, "QW mode is exclusive"
            H, D = k.shape[1], k.shape[-1]

            def to_k_layout(t):
                if t.shape == (H, D):
                    return t
                if t.shape[0] == H and D % t.shape[1] == 0:
                    # HY cache packs rope+prope K/V variants (head_dim = 2x);
                    # both halves map to the same underlying dims -> tile.
                    return t.repeat(1, D // t.shape[1])
                assert t.numel() == H * D, f"q-stats {tuple(t.shape)} vs k {H}x{D}"
                return t.reshape(H, D)

            if PCA_QW_MODE == "bits":                       # N6
                wn = _qw_pair_avg(w.to(k.device, torch.float32))
                # PCA_QW_ALPHA doubles as the greedy temper: 1.0 = raw energy,
                # 0.5 = sqrt-tempered (spreads bits, less plain-MSE damage)
                bits = to_k_layout(_greedy_bits(wn.clamp_min(1e-12) ** PCA_QW_ALPHA))
                if not getattr(pca_fake_quant_kv, "_qw_announced", False):
                    pca_fake_quant_kv._qw_announced = True
                    hist = torch.bincount(bits.flatten(), minlength=5).tolist()
                    print(f"[pca_quant] QW-N6 bits active: pair={PCA_QW_PAIR or 'none'} "
                          f"bits histogram(0-4)={hist} mean={bits.float().mean():.3f}", flush=True)
                k_q = pca_fake_quant(k, PCA_R, res_bits=bits)
                v_q = pca_fake_quant(v, PCA_R if PCA_V_MODE == "pca" else 0)
                return k_q, v_q

            s = to_k_layout(_qw_scale(w.to(k.device, torch.float32)))   # N5
            if not getattr(pca_fake_quant_kv, "_qw_announced", False):
                pca_fake_quant_kv._qw_announced = True
                print(f"[pca_quant] QW active: alpha={PCA_QW_ALPHA} pair={PCA_QW_PAIR or 'none'} "
                      f"s[min/med/max]={s.min():.3f}/{s.median():.3f}/{s.max():.3f}", flush=True)
            sk = s[None, :, None, :]
            k_q = (pca_fake_quant(k.float() * sk, PCA_R) / sk).to(k.dtype)
            v_q = pca_fake_quant(v, PCA_R if PCA_V_MODE == "pca" else 0)
            return k_q, v_q
    kb = None
    if _K_BASIS is not None:
        layer = _LAYER_CTR[0] % _K_BASIS.shape[0]
        _LAYER_CTR[0] += 1
        kb = _K_BASIS[layer]
    if PCA_SPLIT_D and k.shape[-1] > PCA_SPLIT_D:
        H, D = k.shape[1], k.shape[-1]
        assert kb is None, "PCA_SPLIT_D incompatible with external basis"
        k_q = _unsplit_d(pca_fake_quant(_split_d(k), PCA_R), H, D)
        v_q = _unsplit_d(pca_fake_quant(_split_d(v), PCA_R if PCA_V_MODE == "pca" else 0), H, D)
        return k_q, v_q
    # PCA_HALF_R="r1,r2": packed models (HY) — separate ranks for the
    # rope/prope halves at the same total coef budget (r1+r2 = 2*PCA_R).
    hr = os.environ.get("PCA_HALF_R", "")
    hrk = os.environ.get("PCA_HALF_R_K", "") or hr
    hrv = os.environ.get("PCA_HALF_R_V", "") or hr
    if (hrk or hrv) and k.shape[-1] == 256:
        def _hq(t, spec, which=""):
            r1, r2 = (int(x) for x in spec.split(","))
            a = pca_fake_quant(t[..., :128].contiguous(), r1)
            # per-half residual grid: PCA_RES_GRID_{K,V}P overrides the PROPE half only
            with _res_grid_override(which + "P"):
                b = pca_fake_quant(t[..., 128:].contiguous(), r2)
            return torch.cat([a, b], dim=-1)
        if not getattr(pca_fake_quant_kv, "_hr_announced", False):
            pca_fake_quant_kv._hr_announced = True
            print(f"[pca_quant] half-rank active: K={hrk} V={hrv} "
                  f"gridK={os.environ.get('PCA_RES_GRID_K','-')} "
                  f"gridV={os.environ.get('PCA_RES_GRID_V','-')}", flush=True)
        with _res_grid_override("K"):
            k_q = _hq(k, hrk, "K") if hrk else pca_fake_quant(k, PCA_KR or PCA_R)
        with _res_grid_override("V"):
            v_q = _hq(v, hrv, "V") if hrv else pca_fake_quant(v, (PCA_VR or PCA_R))
        return k_q, v_q
    # OSCAR-style BF16 sink window: first PCA_SINK_T tokens of event 0 stay
    # full precision (literature-standard; amortized in BPE accounting).
    sink = int(os.environ.get("PCA_SINK_T", "0"))
    ev0 = PCA_N_LAYERS > 0 and (call_idx // PCA_N_LAYERS) == 0
    with _res_grid_override("K"):
        k_q = pca_fake_quant(k, PCA_KR or PCA_R, fixed_basis=kb)
    if sink > 0 and ev0:
        k_q[:, :, :sink] = k[:, :, :sink]
    if PCA_V_MODE == "klt":                                          # N17a
        v_q = _klt_fake_quant(v)
    elif PCA_V_METRIC == "wo" and _VW_PROVIDER[0] is not None:       # N20
        wpair = _VW_PROVIDER[0](call_idx)
        if wpair is None:
            v_q = pca_fake_quant(v, PCA_R if PCA_V_MODE == "pca" else 0)
        else:
            W, Winv = wpair
            if not getattr(pca_fake_quant_kv, "_vw_announced", False):
                pca_fake_quant_kv._vw_announced = True
                print(f"[pca_quant] N20 V<-W_O metric active W{tuple(W.shape)}", flush=True)
            d = W.shape[-1]
            if v.shape[-1] == d:
                vw = torch.einsum("bhsd,hde->bhse", v.float(), W.to(v.device))
                vq = pca_fake_quant(vw, PCA_R if PCA_V_MODE == "pca" else 0)
                v_q = torch.einsum("bhsd,hde->bhse", vq.float(),
                                   Winv.to(v.device)).to(v.dtype)
            else:
                # HY packs [v_rope; v_prope] (2d wide). PCA_VW_HALF=1: whiten
                # only the first (rope) half — the prope variant may not feed
                # the same to_out path. Default: whiten both with same W.
                B, H, S, D = v.shape
                n = D // d
                half_only = os.environ.get("PCA_VW_HALF", "0") == "1"
                vv = v.float().reshape(B, H, S, n, d)
                if half_only:
                    vw = vv.clone()
                    vw[..., 0, :] = torch.einsum("bhsd,hde->bhse", vv[..., 0, :], W.to(v.device))
                else:
                    vw = torch.einsum("bhsnd,hde->bhsne", vv, W.to(v.device))
                vq = pca_fake_quant(vw.reshape(B, H, S, D),
                                    PCA_R if PCA_V_MODE == "pca" else 0)
                vq = vq.float().reshape(B, H, S, n, d)
                if half_only:
                    out = vq.clone()
                    out[..., 0, :] = torch.einsum("bhsd,hde->bhse", vq[..., 0, :], Winv.to(v.device))
                    vq = out
                else:
                    vq = torch.einsum("bhsnd,hde->bhsne", vq, Winv.to(v.device))
                v_q = vq.reshape(B, H, S, D).to(v.dtype)
    else:
        with _res_grid_override("V"):
            v_q = pca_fake_quant(v, (PCA_VR or PCA_R) if PCA_V_MODE == "pca" else 0)
    if sink > 0 and ev0:
        v_q[:, :, :sink] = v[:, :, :sink]
    return k_q, v_q
