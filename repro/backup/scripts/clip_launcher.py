"""Pure-discard percentile clipping for the QVG INT2/INT4 pipeline (variant B).

Before the standard QVG quantization (k-means smoothing + blockwise residual
quant), hard-clamp each K/V tensor to +/- t where t = p-th percentile of |x|
(p from env CLIP_PCT). Clipped-off magnitude is DISCARDED — no side-channel
residual, no add-back, no rescale — so compression is identical to no-clip
and any quality change is attributable to the clamp alone. CLIP_PCT=100
skips the clamp entirely (control arm).

Also isolates k-means RNG (same as longcat_rngiso_launcher) so all sweep
points share identical noise trajectories and are directly comparable.

Usage:
  CLIP_PCT=95 torchrun ... repro/backup/scripts/clip_launcher.py <run_long_t2v args with
  --quant_type triton-nstages-kmeans-int2/int4>
"""

import os
import runpy

import torch

# ---- k-means RNG isolation (identical to longcat_rngiso_launcher) ----
_orig_randint = torch.randint
_gens = {}


def _iso_randint(*args, **kwargs):
    if kwargs.get("generator") is None:
        device = torch.device(kwargs.get("device") or "cpu")
        gen = _gens.get(device.type)
        if gen is None:
            gen = torch.Generator(device=device)
            gen.manual_seed(20260710)
            _gens[device.type] = gen
        kwargs["generator"] = gen
    return _orig_randint(*args, **kwargs)


torch.randint = _iso_randint

# ---- pure-discard percentile clamp on K/V before quantization ----
CLIP_PCT = float(os.environ.get("CLIP_PCT", "100"))

import quant_videogen.compress as _compress
from quant_videogen.functions import compute_percentile_by_sorting

_orig_compress_kv_cache = _compress.compress_kv_cache
_logged = {"done": False}


def _clamp_pct(x: torch.Tensor) -> torch.Tensor:
    t = compute_percentile_by_sorting(x.abs(), CLIP_PCT)
    return torch.clamp(x, min=-t, max=t)


def _patched_compress_kv_cache(k, v, quant_type, quant_config, quantize_fn):
    if CLIP_PCT < 100.0 and isinstance(k, torch.Tensor):
        if not _logged["done"]:
            _logged["done"] = True
            print(f"[clip_launcher] pure-discard clamp at p={CLIP_PCT} percentile "
                  f"(no residual add-back), then {quant_type}", flush=True)
        k = _clamp_pct(k)
        v = _clamp_pct(v)
    return _orig_compress_kv_cache(k, v, quant_type, quant_config, quantize_fn)


_compress.compress_kv_cache = _patched_compress_kv_cache

runpy.run_path("experiments/LongCat/run_long_t2v.py", run_name="__main__")
