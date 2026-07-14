"""Launcher that injects QuaRot KV-cache quantization into Quant-VideoGen.

Runs any integration script with the `naive-int{2,4}` CLI quant_type hijacked
to QuaRot-style quantization (Hadamard rotation + per-block asym RTN, see
repro/quarot_quant.py). No repo files are modified: we patch
quant_videogen.compress.compress_kv_cache BEFORE the target script is
imported, so every integration's top-level `from quant_videogen.compress
import compress_kv_cache` binds the patched function.

Usage:
  QUAROT_TARGET=experiments/LongCat/run_long_t2v.py \
  torchrun ... repro/backup/scripts/quarot_launcher.py <target-script-args with --quant_type naive-int2>

Config env vars: see repro/quarot_quant.py. Set QUAROT_DISABLE=1 to run the
target with the stock naive RTN path (no patch) for A/B.
"""

import os
import re
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if os.environ.get("QUAROT_DISABLE", "0") != "1":
    import quant_videogen.compress as _compress
    from quarot_quant import (
        QUAROT_ROTATE_K,
        QUAROT_ROTATE_V,
        quarot_fake_quant,
    )

    _orig_compress_kv_cache = _compress.compress_kv_cache
    _announced = {"done": False}

    def _patched_compress_kv_cache(k, v, quant_type, quant_config, quantize_fn):
        m = re.fullmatch(r"naive-int(\d+)", quant_type)
        if m is None:
            return _orig_compress_kv_cache(k, v, quant_type, quant_config, quantize_fn)
        num_bits = int(m.group(1))
        if not _announced["done"]:
            _announced["done"] = True
            print(
                f"[quarot_launcher] hijacking quant_type={quant_type} -> QuaRot "
                f"fake-quant (bits={num_bits}, block={os.environ.get('QUAROT_BLOCK', '16')}, "
                f"sym={os.environ.get('QUAROT_SYM', '0')}, "
                f"rotate_k={QUAROT_ROTATE_K}, rotate_v={QUAROT_ROTATE_V})",
                flush=True,
            )
        k_quant = quarot_fake_quant(k, num_bits, QUAROT_ROTATE_K)
        v_quant = quarot_fake_quant(v, num_bits, QUAROT_ROTATE_V)
        return k_quant, v_quant

    _compress.compress_kv_cache = _patched_compress_kv_cache

_target = os.environ.get("QUAROT_TARGET", "experiments/LongCat/run_long_t2v.py")
runpy.run_path(_target, run_name="__main__")
