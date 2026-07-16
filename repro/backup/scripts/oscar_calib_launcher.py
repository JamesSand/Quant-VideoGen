"""OSCAR-style pass-1 calibration for LongCat: accumulate per-(layer, head)
Q covariance (QtQ) during a clean generation, save eigenbases for pass-2
PCA-KV (K basis <- directions queries actually probe, cf. OSCAR qqt).

Captures q right after q_norm (pre-RoPE, matching the pre-RoPE K we quantize;
caveat noted in docs). Save: {"eigvecs": [L, H, D, D] float32 (ascending),
"counts": [L]} -> $OSCAR_CALIB_OUT.

Usage: OSCAR_CALIB_OUT=... PYTHONPATH=experiments/LongCat torchrun ...
       oscar_calib_launcher.py <run_long_t2v args, quant_type none>
"""
import atexit
import os
import runpy
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from longcat_video.modules import attention as _att

OUT = os.environ["OSCAR_CALIB_OUT"]

_idx = [0]
_cov = {}      # layer -> [H, D, D] float64 (cpu)
_cnt = {}

_orig_init = _att.Attention.__init__
_orig_fwd = _att.Attention.forward_with_kv_cache


def _patched_init(self, *a, **kw):
    _orig_init(self, *a, **kw)
    self._oscar_layer = _idx[0]
    _idx[0] += 1


def _accumulate(layer, q):
    # q: [B, H, S, D] (post q_norm, pre rope)
    with torch.no_grad():
        qf = q.detach().float()
        B, H, S, D = qf.shape
        c = torch.einsum("bhsd,bhse->hde", qf, qf).double().cpu()
        if layer not in _cov:
            _cov[layer] = c
            _cnt[layer] = B * S
        else:
            _cov[layer] += c
            _cnt[layer] += B * S


def _patched_fwd(self, x, shape, kv_cache_dict, *a, **kw):
    return _orig_fwd(self, x, shape, kv_cache_dict, *a, **kw)


# q is computed inside forward_with_kv_cache; easiest reliable capture point is
# q_norm's output: wrap the q_norm module's forward per instance at first use.
_orig_qnorm_fwd = None


def _patch_qnorm(self):
    qn = self.q_norm
    if getattr(qn, "_oscar_wrapped", False):
        return
    layer = self._oscar_layer
    orig = qn.forward

    def wrapped(t, _orig=orig, _layer=layer):
        out = _orig(t)
        try:
            if out.dim() == 4:
                _accumulate(_layer, out)
        except Exception:
            pass
        return out

    qn.forward = wrapped
    qn._oscar_wrapped = True


def _patched_fwd_hook(self, *a, **kw):
    _patch_qnorm(self)
    return _orig_fwd(self, *a, **kw)


_att.Attention.__init__ = _patched_init
_att.Attention.forward_with_kv_cache = _patched_fwd_hook


@atexit.register
def _save():
    if not _cov:
        print("[oscar-calib] nothing captured", flush=True)
        return
    L = max(_cov) + 1
    H, D = _cov[min(_cov)].shape[0], _cov[min(_cov)].shape[-1]
    eig = torch.zeros(L, H, D, D, dtype=torch.float32)
    counts = torch.zeros(L)
    for l, c in _cov.items():
        cov = c / max(_cnt[l], 1)
        cov = (cov + cov.transpose(-1, -2)) / 2
        _, vecs = torch.linalg.eigh(cov)      # ascending
        eig[l] = vecs.float()
        counts[l] = _cnt[l]
    torch.save({"eigvecs": eig, "counts": counts}, OUT)
    print(f"[oscar-calib] saved {OUT}: layers={L} heads={H} D={D} "
          f"tokens/layer~{int(counts.max())}", flush=True)


_target = os.environ.get("PCA_TARGET", "experiments/LongCat/run_long_t2v.py")
runpy.run_path(_target, run_name="__main__")
