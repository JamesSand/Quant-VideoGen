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


def _asym_quant_lastdim_grouped(x, bits, group):
    """Asymmetric fake-quant with one (scale, zero) per group along the last dim."""
    S = x.shape
    xg = x.reshape(*S[:-1], S[-1] // group, group)
    mn = xg.amin(dim=-1, keepdim=True)
    mx = xg.amax(dim=-1, keepdim=True)
    scale = ((mx - mn) / (2 ** bits - 1)).clamp_min(1e-8)
    q = torch.clamp(torch.round((xg - mn) / scale), 0, 2 ** bits - 1)
    return (q * scale + mn).reshape(S)


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


def pca_fake_quant(x, r, fixed_basis=None):
    """x: [B, H, S, D] -> fake-quantized same shape/dtype. r=0 -> mean-only.
    fixed_basis: [H, D, D] ascending eigvecs -> use its top-r instead of chunk eigh."""
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
        res_hat = _quant_residual(res)
        out = mu + lowrank + res_hat
        return out.reshape(B, H, S, D).to(x.dtype)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32


PCA_SPLIT_D = int(os.environ.get("PCA_SPLIT_D", "0"))  # >0: split D into
# sub-heads of this width before PCA (e.g. 128 on HY's 256-dim heads so the
# subspace granularity matches the LC/SF setup). 0 = off.

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
    if PCA_QW_ALPHA > 0 and _QW_PROVIDER[0] is not None:
        w = _QW_PROVIDER[0](call_idx)
        if w is not None:
            assert _K_BASIS is None and not PCA_SPLIT_D, "QW mode is exclusive"
            s = _qw_scale(w.to(k.device, torch.float32))       # [H, D]
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
