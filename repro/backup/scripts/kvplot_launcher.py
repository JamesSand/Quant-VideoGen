"""Capture raw (pre-quant, K pre-RoPE) KV writes from ChunkedKVCache for plotting.

Model-agnostic: patches quant_videogen.kv_cache.ChunkedKVCache.{__init__,write}.
Keeps the first KVPLOT_CAP tokens written to every instance; at exit saves only
the middle-of-network instances (idx within +-KVPLOT_HALO of the median) plus
per-instance metadata (creation idx, caller file:line to disambiguate k vs v).

Env: KVPLOT_OUT (required), KVPLOT_CAP (default 12000), KVPLOT_HALO (default 4),
     KVPLOT_TARGET (script to run).
"""
import atexit
import os
import runpy
import sys

import torch

import quant_videogen.kv_cache as kvc

CAP = int(os.getenv("KVPLOT_CAP", "12000"))
HALO = int(os.getenv("KVPLOT_HALO", "4"))
OUT = os.environ["KVPLOT_OUT"]
TARGET = os.environ["KVPLOT_TARGET"]

_registry = []          # instance order
_store = {}             # idx -> list[Tensor]
_meta = {}              # idx -> dict

_orig_init = kvc.ChunkedKVCache.__init__
_orig_write = kvc.ChunkedKVCache.write


def _patched_init(self, *a, **kw):
    _orig_init(self, *a, **kw)
    self._kvplot_idx = len(_registry)
    self._kvplot_kept = 0
    _registry.append(True)


def _patched_write(self, start_index, end_index, data):
    idx = getattr(self, "_kvplot_idx", None)
    if idx is not None and self._kvplot_kept < CAP:
        f = sys._getframe(1)
        caller = f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno}"
        keep = min(CAP - self._kvplot_kept, data.shape[1])
        _store.setdefault(idx, []).append(data[:, :keep].detach().to("cpu", copy=True))
        m = _meta.setdefault(idx, {"callers": set(), "writes": 0, "shape0": tuple(data.shape)})
        m["callers"].add(caller)
        m["writes"] += 1
        self._kvplot_kept += keep
    return _orig_write(self, start_index, end_index, data)


kvc.ChunkedKVCache.__init__ = _patched_init
kvc.ChunkedKVCache.write = _patched_write

# Second hook: models that manage their own cache (LongCat) still call
# compress_kv_cache(k, v, ...) once per layer. Capture its raw inputs.
import quant_videogen.compress as _cmp

_orig_compress = _cmp.compress_kv_cache
_ckv_calls = []          # list of (k_cpu, v_cpu) capped
_ckv_meta = []


def _patched_compress(k, v, *a, **kw):
    def cap(x):
        S_dim = max(range(x.dim()), key=lambda d: x.shape[d])  # longest dim = tokens
        sl = [slice(None)] * x.dim()
        sl[S_dim] = slice(0, CAP)
        return x[tuple(sl)].detach().to("cpu", copy=True)
    _ckv_calls.append((cap(k), cap(v)))
    _ckv_meta.append({"k_shape": tuple(k.shape), "v_shape": tuple(v.shape)})
    return _orig_compress(k, v, *a, **kw)


_cmp.compress_kv_cache = _patched_compress


@atexit.register
def _save():
    n = len(_registry)
    if n == 0 and _ckv_calls:
        m = len(_ckv_calls)
        mid = m // 2
        sel = range(max(0, mid - HALO), min(m, mid + HALO))
        out = {"mode": "compress_calls", "n_calls": m,
               "selected": {i: {"k": _ckv_calls[i][0], "v": _ckv_calls[i][1]} for i in sel},
               "meta": {i: _ckv_meta[i] for i in range(m)}}
        torch.save(out, OUT)
        print(f"[kvplot] saved {OUT} via compress hook: {m} calls, kept {list(sel)}", flush=True)
        return
    if n == 0:
        print("[kvplot] no ChunkedKVCache instances seen — nothing saved", flush=True)
        return
    mid = n // 2
    sel = [i for i in _store if mid - HALO <= i < mid + HALO]
    out = {
        "n_instances": n,
        "selected": {},
        "meta": {i: {**_meta[i], "callers": sorted(_meta[i]["callers"])} for i in _store},
    }
    for i in sel:
        out["selected"][i] = torch.cat(_store[i], dim=1)
    torch.save(out, OUT)
    kept = {i: tuple(out["selected"][i].shape) for i in sel}
    print(f"[kvplot] saved {OUT}: {n} instances total, kept {kept}", flush=True)


sys.argv = [TARGET] + sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(TARGET)))
runpy.run_path(TARGET, run_name="__main__")
