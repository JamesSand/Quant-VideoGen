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
PCA_COEFF_BITS = int(os.environ.get("PCA_COEFF_BITS", "2"))
PCA_RES_GRID = os.environ.get("PCA_RES_GRID", "ternary")
PCA_V_MODE = os.environ.get("PCA_V_MODE", "mean")
RES_BLOCK = int(os.environ.get("PCA_RES_BLOCK", "64"))
K_BASIS_FILE = os.environ.get("PCA_K_BASIS_FILE", "")
_K_BASIS = None          # [L, H, D, D] eigvecs ascending（OSCAR 式离线校准基）
_LAYER_CTR = [0]

if K_BASIS_FILE:
    _K_BASIS = torch.load(K_BASIS_FILE, map_location="cpu")["eigvecs"]


PCA_RES_MSEOPT = os.environ.get("PCA_RES_MSEOPT", "0") == "1"
# N8: per-block MSE-optimal range shrinkage for the asym residual quantizer.
# Searches a few symmetric-about-center shrink ratios per block and keeps the
# min-MSE one; min-max (ratio 1.0) is in the search set, so never worse in
# block MSE. Same (scale, zero) metadata -> BPE unchanged.
_MSE_RATIOS = (1.0, 0.92, 0.85, 0.78, 0.70, 0.62)


def _asym_quant_lastdim_grouped(x, bits, group, mse_opt=None):
    """Asymmetric fake-quant with one (scale, zero) per group along the last dim."""
    if mse_opt is None:
        mse_opt = PCA_RES_MSEOPT
    S = x.shape
    xg = x.reshape(*S[:-1], S[-1] // group, group)
    mn = xg.amin(dim=-1, keepdim=True)
    mx = xg.amax(dim=-1, keepdim=True)
    if not mse_opt:
        scale = ((mx - mn) / (2 ** bits - 1)).clamp_min(1e-8)
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


def _ternary_quant_blocked(x, block):
    """QVG-aligned symmetric ternary over blocks of `block` along the last dim."""
    S = x.shape
    xb = x.reshape(*S[:-1], S[-1] // block, block)
    scale = xb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)  # max_int = 1
    q = torch.clamp(torch.round(xb / scale), -1, 1)
    return (q * scale).reshape(S)


def _quant_residual(x):
    if PCA_RES_GRID == "ternary":
        return _ternary_quant_blocked(x, RES_BLOCK)
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
    """w: [H, D] -> integer bits [H, D], sum per head = budget_per_dim * D."""
    H, D = w.shape
    b = torch.full((H, D), bmin, dtype=torch.long, device=w.device)
    extra = (budget_per_dim - bmin) * D
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


def pca_fake_quant_kv(k, v):
    call_idx = _QW_CTR[0]
    _QW_CTR[0] += 1
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
    k_q = pca_fake_quant(k, PCA_R, fixed_basis=kb)
    v_q = pca_fake_quant(v, PCA_R if PCA_V_MODE == "pca" else 0)
    return k_q, v_q
