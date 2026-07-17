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
import pca_quant as _pq
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

# ---- N5: online per-(layer, head, dim) E[q^2] capture for PCA_QW_ALPHA ----
if _pq.PCA_QW_ALPHA > 0:
    import torch as _t

    _qs_sum, _qs_cnt = {}, {}

    def _qw_accum(layer, q):   # q: [B, H, S, D] post q_norm, pre RoPE
        with _t.no_grad():
            m = (q.detach().float() ** 2).sum(dim=(0, 2))       # [H, D]
            n = q.shape[0] * q.shape[2]
            if layer in _qs_sum:
                _qs_sum[layer] += m
                _qs_cnt[layer] += n
            else:
                _qs_sum[layer] = m.clone()
                _qs_cnt[layer] = n

    def _qw_provider(call_idx):
        if not _qs_sum:
            return None
        n_layers = max(_qs_sum) + 1
        layer = call_idx % n_layers
        if layer not in _qs_sum:
            return None
        return _qs_sum[layer] / max(_qs_cnt[layer], 1)

    _pq.set_qw_provider(_qw_provider)

    def _wrap_qnorm_of(att, layer):
        qn = att.q_norm
        if getattr(qn, "_qw_wrapped", False):
            return
        orig = qn.forward

        def wrapped(t, _orig=orig, _layer=layer):
            out = _orig(t)
            if out.dim() == 4:
                _qw_accum(_layer, out)
            return out

        qn.forward = wrapped
        qn._qw_wrapped = True

    _tgt_for_qw = os.environ.get("PCA_TARGET", "experiments/LongCat/run_long_t2v.py")
    if "LongCat" in _tgt_for_qw:
        from longcat_video.modules import attention as _latt

        _qw_idx = [0]
        _orig_att_init = _latt.Attention.__init__
        _orig_att_fwd = _latt.Attention.forward_with_kv_cache

        def _att_init(self, *a, **kw):
            _orig_att_init(self, *a, **kw)
            self._qw_layer = _qw_idx[0]
            _qw_idx[0] += 1

        def _att_fwd(self, *a, **kw):
            _wrap_qnorm_of(self, self._qw_layer)
            return _orig_att_fwd(self, *a, **kw)

        _latt.Attention.__init__ = _att_init
        _latt.Attention.forward_with_kv_cache = _att_fwd
        print(f"[pca_launcher] N5 online q-stats enabled for LongCat "
              f"(alpha={_pq.PCA_QW_ALPHA})", flush=True)
    else:
        print("[pca_launcher] WARNING: PCA_QW_ALPHA set but no q-capture "
              f"path for target {_tgt_for_qw}; QW inactive", flush=True)

_target = os.environ.get("PCA_TARGET", "experiments/LongCat/run_long_t2v.py")
runpy.run_path(_target, run_name="__main__")
