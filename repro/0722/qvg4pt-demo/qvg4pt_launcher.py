"""qvg4pt 最小 launcher(0722 demo)。

机制与 repro/backup/scripts/pca_launcher.py 相同(monkey-patch,见
qvg4pt-code-walkthrough.md §三),但只保留 qvg4pt 一条路:在目标脚本 import
之前把 quant_videogen.compress.compress_kv_cache 换掉,凡 quant_type 匹配
naive-int\\d+ 一律走 qvg4pt_fake_quant_kv;其余 quant_type 原样放行。

Usage:
  PYTHONPATH=experiments/LongCat torchrun --nproc_per_node=1 --standalone \
    repro/0722/qvg4pt-demo/qvg4pt_launcher.py \
    <run_long_t2v.py 的参数,含 --quant_type naive-int2>
"""
import os
import re
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import quant_videogen.compress as _compress
from qvg4pt_quant import qvg4pt_fake_quant_kv

_orig = _compress.compress_kv_cache
_announced = {"done": False}


def _patched(k, v, quant_type, quant_config, quantize_fn):
    if re.fullmatch(r"naive-int(\d+)", quant_type) is None:
        return _orig(k, v, quant_type, quant_config, quantize_fn)
    if not _announced["done"]:
        _announced["done"] = True
        print(f"[qvg4pt_launcher] hijacking {quant_type} -> qvg4pt "
              f"(kmeans K=256 sub + ASYM 4-level B64 residual, fp8 s/z) "
              f"k{tuple(k.shape)}", flush=True)
    return qvg4pt_fake_quant_kv(k, v)


_compress.compress_kv_cache = _patched

_target = os.environ.get("QVG4PT_TARGET", "experiments/LongCat/run_long_t2v.py")
runpy.run_path(_target, run_name="__main__")
