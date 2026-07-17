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

# SF upstream inconsistency: its MSE printer wants the fake tensor in BHSD (the
# compress input layout) but ChunkedKVCache.store_quantized forwards tensors to
# write(), which expects BSHD. Fix at store time (PCA_SF_STORE_FIX=1).
if os.environ.get("PCA_SF_STORE_FIX", "0") == "1":
    import torch as _torch
    import quant_videogen.kv_cache as _kvc

    _orig_store = _kvc.ChunkedKVCache.store_quantized

    def _store_fix(self, start_index, end_index, quant_data):
        if _torch.is_tensor(quant_data):
            return self.write(start_index, end_index,
                              quant_data.permute(0, 2, 1, 3).contiguous())
        return _orig_store(self, start_index, end_index, quant_data)

    _kvc.ChunkedKVCache.store_quantized = _store_fix

_target = os.environ.get("PCA_TARGET", "experiments/LongCat/run_long_t2v.py")
runpy.run_path(_target, run_name="__main__")
