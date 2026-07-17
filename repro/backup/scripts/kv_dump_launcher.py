"""Dump the first quantize event's per-layer K/V (subset of layers) then exit.
Usage: KV_DUMP_DIR=... KV_DUMP_EVERY=5 PCA_TARGET=... torchrun ... kv_dump_launcher.py <args>"""
import os, runpy, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import quant_videogen.compress as _compress

OUT = os.environ["KV_DUMP_DIR"]
EVERY = int(os.environ.get("KV_DUMP_EVERY", "5"))
os.makedirs(OUT, exist_ok=True)
_orig = _compress.compress_kv_cache
_ctr = [0]

def _patched(k, v, quant_type, quant_config, quantize_fn):
    i = _ctr[0]; _ctr[0] += 1
    if (i % int(os.environ.get("KV_DUMP_L", "30"))) in (0, 15, 25) and i < 90:
        torch.save({"k": k.detach().to(torch.bfloat16).cpu(),
                    "v": v.detach().to(torch.bfloat16).cpu()},
                   f"{OUT}/call_{i:03d}.pt")
        print(f"[kv_dump] saved layer {i} k{tuple(k.shape)}", flush=True)
    return _orig(k, v, quant_type, quant_config, quantize_fn)

_compress.compress_kv_cache = _patched

import atexit
runpy.run_path(os.environ["PCA_TARGET"], run_name="__main__")
