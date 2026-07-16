"""Mean/PCA low-rank + quantized-residual fake-quant for KV cache (plan:
repro/0715/pca-psnr-plan.md).

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


def pca_fake_quant(x, r):
    """x: [B, H, S, D] -> fake-quantized same shape/dtype. r=0 -> mean-only."""
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        B, H, S, D = x.shape
        X = x.reshape(B * H, S, D).float()
        mu = X.mean(dim=1, keepdim=True)                      # [BH,1,D]
        Xc = X - mu
        if r > 0:
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


def pca_fake_quant_kv(k, v):
    k_q = pca_fake_quant(k, PCA_R)
    v_q = pca_fake_quant(v, PCA_R if PCA_V_MODE == "pca" else 0)
    return k_q, v_q
