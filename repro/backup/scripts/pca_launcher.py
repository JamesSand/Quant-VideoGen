"""Launcher injecting mean/PCA low-rank KV fake-quant (see pca_quant.py).

Hijacks `naive-int*` exactly like quarot_launcher.py: patch
quant_videogen.compress.compress_kv_cache before the target script imports it.

Usage:
  PCA_TARGET=experiments/LongCat/run_long_t2v.py PCA_R=8 ... \
  torchrun ... repro/backup/scripts/pca_launcher.py <target args with --quant_type naive-int2>
"""
import os
import re
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import quant_videogen.compress as _compress
from pca_quant import PCA_COEFF_BITS, PCA_R, PCA_RES_GRID, PCA_V_MODE, pca_fake_quant_kv

_orig = _compress.compress_kv_cache
_announced = {"done": False}


def _patched(k, v, quant_type, quant_config, quantize_fn):
    if re.fullmatch(r"naive-int(\d+)", quant_type) is None:
        return _orig(k, v, quant_type, quant_config, quantize_fn)
    if not _announced["done"]:
        _announced["done"] = True
        print(f"[pca_launcher] hijacking {quant_type} -> mean/PCA fake-quant "
              f"(r={PCA_R}, coeff_bits={PCA_COEFF_BITS}, res={PCA_RES_GRID}, "
              f"v_mode={PCA_V_MODE}) k{tuple(k.shape)}", flush=True)
    return pca_fake_quant_kv(k, v)


_compress.compress_kv_cache = _patched
_target = os.environ.get("PCA_TARGET", "experiments/LongCat/run_long_t2v.py")
runpy.run_path(_target, run_name="__main__")
