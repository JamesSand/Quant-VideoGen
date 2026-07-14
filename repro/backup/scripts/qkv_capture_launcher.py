"""Capture raw Q/K/V (pre-RoPE, last denoise step) from Self-Forcing attention.

Hooks CausalWanSelfAttention.attn_kv_cache_prerope. For layers in QKV_LAYERS and
blocks whose start frame falls in QKV_WINDOWS, stores q/k/v per (layer, block);
each denoise step overwrites the slot, so the surviving copy is the LAST step.

Env: QKV_OUT (required), QKV_TARGET (script), QKV_LAYERS (csv, default "0,15,29"),
     QKV_WINDOWS (csv of start-frame ranges "0-5,87-92,174-179").
"""
import atexit
import os
import runpy
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../../../experiments/Self-Forcing"))
from wan.modules import causal_model as cm

OUT = os.environ["QKV_OUT"]
TARGET = os.environ["QKV_TARGET"]
LAYERS = {int(x) for x in os.getenv("QKV_LAYERS", "0,15,29").split(",")}
WINDOWS = []
for rng in os.getenv("QKV_WINDOWS", "0-5,87-92,174-179").split(","):
    a, b = rng.split("-")
    WINDOWS.append((int(a), int(b)))

_inst_counter = [0]
_slots = {}          # (layer, start_frame) -> {'q','k','v'}

_orig_init = cm.CausalWanSelfAttention.__init__
_orig_prerope = cm.CausalWanSelfAttention.attn_kv_cache_prerope


def _patched_init(self, *a, **kw):
    _orig_init(self, *a, **kw)
    self._qkv_layer = _inst_counter[0]
    _inst_counter[0] += 1


def _in_windows(f):
    return any(a <= f <= b for a, b in WINDOWS)


def _patched_prerope(self, kv_cache, current_start, q, k, v, block_mask, grid_sizes, freqs, seq_lens, s):
    lyr = getattr(self, "_qkv_layer", -1)
    if lyr in LAYERS:
        frame_seqlen = int(grid_sizes[0][1] * grid_sizes[0][2])
        sf = current_start // frame_seqlen
        if _in_windows(int(sf)):
            _slots[(lyr, int(sf))] = {
                "q": q.detach().to("cpu", copy=True),
                "k": k.detach().to("cpu", copy=True),
                "v": v.detach().to("cpu", copy=True),
            }
    return _orig_prerope(self, kv_cache, current_start, q, k, v, block_mask, grid_sizes, freqs, seq_lens, s)


cm.CausalWanSelfAttention.__init__ = _patched_init
cm.CausalWanSelfAttention.attn_kv_cache_prerope = _patched_prerope


@atexit.register
def _save():
    if not _slots:
        print("[qkv] nothing captured", flush=True)
        return
    torch.save({"slots": _slots, "layers": sorted(LAYERS), "windows": WINDOWS,
                "n_attn_instances": _inst_counter[0]}, OUT)
    print(f"[qkv] saved {OUT}: {len(_slots)} slots, keys e.g. {list(_slots)[:4]}", flush=True)


sys.argv = [TARGET] + sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(TARGET)))
runpy.run_path(TARGET, run_name="__main__")
