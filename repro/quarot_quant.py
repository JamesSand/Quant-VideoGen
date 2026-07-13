"""QuaRot-style KV-cache fake quantization for Quant-VideoGen.

Faithful port of QuaRot's (arXiv:2404.00456) KV-cache quantization part, per
the QVG paper's baseline description ("we only implement its KV cache
quantization part ... block size 16 settings"):

  1. Rotate K and V along head_dim with an orthonormal Hadamard matrix
     (QuaRot applies H to K online post-RoPE and to V by fusing H into
     v_proj/o_proj; in fake-quant, rotate -> quantize -> dequantize ->
     rotate-back is mathematically identical to computing attention in the
     rotated basis).
  2. Per-block (default 16 along head_dim) asymmetric round-to-nearest
     quantization at INT2/INT4 (QuaRot paper quantizes the KV cache
     asymmetrically).

Config via env vars (read once at import):
  QUAROT_BLOCK    block size along head_dim (default 16)
  QUAROT_SYM      1 = symmetric quantizer instead of asymmetric (default 0)
  QUAROT_ROTATE_K 0 = skip Hadamard on K (default 1)
  QUAROT_ROTATE_V 0 = skip Hadamard on V (default 1)

The tensor passed by compress_kv_cache's NAIVE branch is [B, H, S, D];
this implementation only assumes the LAST dim is head_dim (power of 2).
"""

import math
import os

import torch

QUAROT_BLOCK = int(os.environ.get("QUAROT_BLOCK", "16"))
QUAROT_SYM = os.environ.get("QUAROT_SYM", "0") == "1"
QUAROT_ROTATE_K = os.environ.get("QUAROT_ROTATE_K", "1") == "1"
QUAROT_ROTATE_V = os.environ.get("QUAROT_ROTATE_V", "1") == "1"
# post-rotation clipping (both default OFF):
#   CLIP_RATIO < 1.0 -> per-block range shrink, QuaRot-official semantics
#                       (scale from block min/max scaled by ratio; out-of-range
#                       values absorbed by the code clamp)
#   CLIP_PCT < 100   -> global percentile hard clamp on the ROTATED tensor
#                       before blockwise quant (control arm, same form as the
#                       earlier raw-KV study but applied post-rotation)
QUAROT_CLIP_RATIO = float(os.environ.get("QUAROT_CLIP_RATIO", "1.0"))
QUAROT_CLIP_PCT = float(os.environ.get("QUAROT_CLIP_PCT", "100"))

_HAD_CACHE = {}


def hadamard(n: int, device) -> torch.Tensor:
    """Orthonormal Hadamard matrix (Sylvester), n must be a power of 2."""
    assert n & (n - 1) == 0, f"head_dim {n} is not a power of 2"
    key = (n, str(device))
    if key not in _HAD_CACHE:
        H = torch.ones(1, 1, device=device, dtype=torch.float32)
        while H.shape[0] < n:
            H = torch.cat(
                [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0
            )
        _HAD_CACHE[key] = H / math.sqrt(n)
    return _HAD_CACHE[key]


def blockwise_rtn(x: torch.Tensor, num_bits: int, block_size: int, sym: bool,
                  clip_ratio: float = 1.0) -> torch.Tensor:
    """Round-to-nearest fake quant per block along the last dim (float32 in/out).

    clip_ratio < 1.0 shrinks each block's quantization range (new_max =
    max * ratio, new_min = min * ratio — QuaRot's official clip semantics,
    quant_utils.py:125-126); out-of-range values are absorbed by the clamp.
    """
    D = x.shape[-1]
    assert D % block_size == 0, f"head_dim {D} not divisible by block {block_size}"
    shp = x.shape
    xb = x.reshape(-1, D // block_size, block_size)
    if sym:
        qmax = 2 ** (num_bits - 1) - 1  # int2 -> {-2,-1,0,1}, clamp to [-qmax-1, qmax]
        scale = (xb.abs().amax(dim=-1, keepdim=True) * clip_ratio).clamp_min(1e-8) / qmax
        q = torch.clamp(torch.round(xb / scale), -qmax - 1, qmax)
        deq = q * scale
    else:
        mn = xb.amin(dim=-1, keepdim=True) * clip_ratio
        mx = xb.amax(dim=-1, keepdim=True) * clip_ratio
        scale = ((mx - mn) / (2**num_bits - 1)).clamp_min(1e-8)
        q = torch.clamp(torch.round((xb - mn) / scale), 0, 2**num_bits - 1)
        deq = q * scale + mn
    return deq.reshape(shp)


def quarot_fake_quant(x: torch.Tensor, num_bits: int, rotate: bool) -> torch.Tensor:
    orig_dtype = x.dtype
    # full fp32 for the basis change: TF32 matmul (H100 default) costs ~1e-4
    # relative error per rotation, which would pollute the quantization study
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        xf = x.float()
        if rotate:
            H = hadamard(x.shape[-1], x.device)
            xf = xf @ H
        if QUAROT_CLIP_PCT < 100.0:
            from quant_videogen.functions import compute_percentile_by_sorting
            t = compute_percentile_by_sorting(xf.abs(), QUAROT_CLIP_PCT)
            xf = torch.clamp(xf, min=-t, max=t)
        xf = blockwise_rtn(xf, num_bits, QUAROT_BLOCK, QUAROT_SYM, QUAROT_CLIP_RATIO)
        if rotate:
            H = hadamard(x.shape[-1], x.device)
            xf = xf @ H.T
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    return xf.to(orig_dtype)


def make_kv_quantize_fn(num_bits: int):
    """Return a quantize_fn compatible with compress_kv_cache's NAIVE branch.

    compress_kv_cache calls quantize_fn(k) then quantize_fn(v); we tell K and
    V apart by call order (K first, V second) via a toggling closure — the
    NAIVE branch is the only caller and always quantizes exactly one (k, v)
    pair per call site invocation.
    """
    state = {"is_k": True}

    def quantize_fn(t: torch.Tensor) -> torch.Tensor:
        is_k = state["is_k"]
        state["is_k"] = not is_k
        rotate = QUAROT_ROTATE_K if is_k else QUAROT_ROTATE_V
        return quarot_fake_quant(t, num_bits, rotate)

    return quantize_fn
