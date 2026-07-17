"""Dump first-event K/V per layer AND a subsample of post-norm_q Q vectors
for the same layers (HY). For the attention-output error metric.
Usage: KVQ_DUMP_DIR=... PCA_TARGET=...HY... torchrun ... kvq_dump_launcher.py <args>"""
import os, runpy, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import quant_videogen.compress as _compress

OUT = os.environ["KVQ_DUMP_DIR"]
os.makedirs(OUT, exist_ok=True)
LAYERS = {0, 5, 10, 15, 20, 25}
NQ = 2048
_q_store = {}          # layer -> [n, H, D]

import wan.models.dits.arwan_w_action_w_mem_relative_rope as _hy
_seen = {}

def _wrap_qnorm(attn):
    if id(attn) in _seen:
        return
    layer = len(_seen)
    _seen[id(attn)] = layer
    if layer not in LAYERS:
        return
    qn = attn.norm_q
    orig = qn.forward
    heads = attn.heads

    def wrapped(t, _orig=orig, _layer=layer, _h=heads):
        out = _orig(t)
        if out.dim() == 3:
            B, S, HD = out.shape
            buf = _q_store.setdefault(_layer, [])
            have = sum(x.shape[0] for x in buf)
            if have < NQ:
                take = min(NQ - have, S)
                idx = torch.randperm(S, device=out.device)[:take]
                buf.append(out[0, idx].view(take, _h, HD // _h).detach().float().cpu())
        return out

    qn.forward = wrapped
    qn._q_wrapped = True

import functools, inspect
_orig_call = _hy.CausalCameraPRopeWanAttnProcessor2_0.__call__

@functools.wraps(_orig_call)
def _call(self, attn, *a, **kw):
    _wrap_qnorm(attn)
    return _orig_call(self, attn, *a, **kw)

_call.__signature__ = inspect.signature(_orig_call)
_hy.CausalCameraPRopeWanAttnProcessor2_0.__call__ = _call

_orig = _compress.compress_kv_cache
_ctr = [0]

def _patched(k, v, quant_type, quant_config, quantize_fn):
    i = _ctr[0]; _ctr[0] += 1
    if i in LAYERS and i < 30:
        q = torch.cat(_q_store.get(i, [torch.zeros(0, 24, 128)]), 0) if i in _q_store else None
        torch.save({"k": k.detach().to(torch.bfloat16).cpu(),
                    "v": v.detach().to(torch.bfloat16).cpu(),
                    "q": q},
                   f"{OUT}/layer_{i:03d}.pt")
        print(f"[kvq_dump] layer {i} k{tuple(k.shape)} q{None if q is None else tuple(q.shape)}", flush=True)
    return _orig(k, v, quant_type, quant_config, quantize_fn)

_compress.compress_kv_cache = _patched
runpy.run_path(os.environ["PCA_TARGET"], run_name="__main__")
