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
        if _pq.PCA_FACTOR_GRID:
            method = "factor-only product grid"
        elif _pq.PCA_GRID_HASH:
            method = "PCA-grid hash"
        else:
            method = "mean/PCA fake-quant"
        print(f"[pca_launcher] hijacking {quant_type} -> {method} "
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

        def _att_init(self, *a, **kw):
            _orig_att_init(self, *a, **kw)
            self._qw_layer = _qw_idx[0]
            _qw_idx[0] += 1
            # wrap immediately: LC quantizes the condition window right after
            # the prefill (before any denoising step), so a lazy forward-hook
            # would miss the only quantize event's stats.
            _wrap_qnorm_of(self, self._qw_layer)

        _latt.Attention.__init__ = _att_init
        print(f"[pca_launcher] N5 online q-stats enabled for LongCat "
              f"(alpha={_pq.PCA_QW_ALPHA})", flush=True)
    elif "HY-WorldPlay" in _tgt_for_qw:
        # Self-attention (KV-cached) uses CausalCameraPRopeWanAttnProcessor2_0;
        # register attn instances in first-call order (= layer order) and wrap
        # their norm_q. norm_q output is [B, S, H*D] -> reshape via attn.heads.
        import wan.models.dits.arwan_w_action_w_mem_relative_rope as _hy

        _seen = {}

        def _wrap_hy_qnorm(attn):
            if id(attn) in _seen:
                return
            layer = len(_seen)
            _seen[id(attn)] = layer
            qn = attn.norm_q
            if qn is None or getattr(qn, "_qw_wrapped", False):
                return
            orig = qn.forward
            heads = attn.heads

            def wrapped(t, _orig=orig, _layer=layer, _h=heads):
                out = _orig(t)
                if out.dim() == 3:
                    B, S, HD = out.shape
                    _qw_accum(_layer, out.view(B, S, _h, HD // _h)
                              .permute(0, 2, 1, 3))
                return out

            qn.forward = wrapped
            qn._qw_wrapped = True

        _orig_proc_call = _hy.CausalCameraPRopeWanAttnProcessor2_0.__call__

        import functools as _ft
        import inspect as _insp

        @_ft.wraps(_orig_proc_call)
        def _proc_call(self, attn, *a, **kw):
            _wrap_hy_qnorm(attn)
            return _orig_proc_call(self, attn, *a, **kw)

        # diffusers filters kwargs by inspecting the processor's __call__
        # signature — mirror the original's signature exactly, or viewmats/Ks
        # get silently dropped.
        _proc_call.__signature__ = _insp.signature(_orig_proc_call)
        _hy.CausalCameraPRopeWanAttnProcessor2_0.__call__ = _proc_call
        print(f"[pca_launcher] N5 online q-stats enabled for HY-WorldPlay "
              f"(alpha={_pq.PCA_QW_ALPHA})", flush=True)
    elif "Self-Forcing" in _tgt_for_qw:
        # CausalWanSelfAttention.norm_q output is [B, S, H*D]; layer index by
        # construction order of the self-attention modules.
        import wan.modules.causal_model as _sfm

        _sf_idx = [0]
        _orig_sf_init = _sfm.CausalWanSelfAttention.__init__

        def _sf_init(self, *a, **kw):
            _orig_sf_init(self, *a, **kw)
            layer = _sf_idx[0]
            _sf_idx[0] += 1
            qn = self.norm_q
            orig = qn.forward
            heads = self.num_heads

            def wrapped(t, _orig=orig, _layer=layer, _h=heads):
                out = _orig(t)
                if out.dim() == 3:
                    B, S, HD = out.shape
                    _qw_accum(_layer, out.view(B, S, _h, HD // _h)
                              .permute(0, 2, 1, 3))
                return out

            qn.forward = wrapped
            qn._qw_wrapped = True

        _sfm.CausalWanSelfAttention.__init__ = _sf_init
        print(f"[pca_launcher] N5 online q-stats enabled for Self-Forcing "
              f"(alpha={_pq.PCA_QW_ALPHA})", flush=True)
    else:
        print("[pca_launcher] WARNING: PCA_QW_ALPHA set but no q-capture "
              f"path for target {_tgt_for_qw}; QW inactive", flush=True)

# ---- N7: temporal bit reallocation by future-retrieval probability (HY) ----
if os.environ.get("PCA_TW", "0") == "1":
    _tgt_tw = os.environ.get("PCA_TARGET", "")
    if "HY-WorldPlay" in _tgt_tw:
        import numpy as _np
        import torch as _t2
        import hyvideo.generate as _hg
        from hyvideo.utils.retrieval_context import (
            calculate_fov_overlap_similarity as _fov)
        import wan.inference.pipeline_wan_w_mem_relative_rope as _pp

        _traj = {}
        _orig_p2i = _hg.pose_to_input

        def _p2i(*a, **kw):
            out = _orig_p2i(*a, **kw)
            _traj["vm"] = _np.asarray(out[0].cpu().numpy(), dtype=_np.float64)
            return out

        _hg.pose_to_input = _p2i

        _TPF = 880
        _TW_TEMPER = float(os.environ.get("PCA_TW_TEMPER", "1.0"))
        _orig_qkv = _pp.WanPipeline.quantize_kv_cache

        def _qkv(self, s, e):
            try:
                vm = _traj.get("vm")
                if vm is not None:
                    fs, fe = s // _TPF, e // _TPF
                    fut = list(range(fe, vm.shape[0], 2))
                    if fut and fe > fs:
                        w = _np.zeros(fe - fs)
                        for i, f in enumerate(range(fs, fe)):
                            w[i] = _np.mean([
                                _fov(vm[q], vm[f], fov_h_deg=60.0, fov_v_deg=35.0,
                                     device="cuda",
                                     points_local=getattr(self, "points_local", None))
                                for q in fut])
                        ww = _t2.tensor(w).clamp_min(1e-6) ** _TW_TEMPER
                        bits_f = _pq._greedy_bits(ww[None, :], budget_per_dim=2,
                                                  bmin=1, bmax=4)[0]
                        _pq.set_tw_bits(bits_f.repeat_interleave(_TPF))
                        print(f"[pca_launcher] N7 frames[{fs},{fe}) bits="
                              f"{bits_f.tolist()} w={['%.3f' % x for x in w]}",
                              flush=True)
            except Exception as ex:
                print(f"[pca_launcher] N7 TW failed: {ex}", flush=True)
            try:
                return _orig_qkv(self, s, e)
            finally:
                _pq.set_tw_bits(None)

        _pp.WanPipeline.quantize_kv_cache = _qkv
        print(f"[pca_launcher] N7 temporal bits enabled (temper={_TW_TEMPER})",
              flush=True)
    else:
        print("[pca_launcher] PCA_TW set but target has no pose signal — "
              "method degenerates to plain N4 (by design)", flush=True)

# ---- N20: W_O-metric V provider (LC) ----
if os.environ.get("PCA_V_METRIC", "") == "wo":
    _tgt_vw = os.environ.get("PCA_TARGET", "")
    if "LongCat" in _tgt_vw:
        import torch as _t3
        from longcat_video.modules import attention as _latt2

        _vw_mods = []
        _vw_cache = {}
        _orig_init_vw = _latt2.Attention.__init__

        def _init_vw(self, *a, **kw):
            _orig_init_vw(self, *a, **kw)
            _vw_mods.append(self)

        _latt2.Attention.__init__ = _init_vw

        def _vw_provider(call_idx):
            if not _vw_mods:
                return None
            layer = call_idx % len(_vw_mods)
            if layer not in _vw_cache:
                mod = _vw_mods[layer]
                with _t3.no_grad():
                    Wt = mod.proj.weight.detach().float()      # [dim, dim]
                    H = mod.num_heads
                    D = mod.head_dim
                    Ws, Wis = [], []
                    for h in range(H):
                        blk = Wt[:, h * D:(h + 1) * D]
                        G = blk.t() @ blk
                        G = G + 1e-3 * G.diagonal().mean() * _t3.eye(D, device=G.device)
                        lam, U = _t3.linalg.eigh(G)
                        Ws.append(U @ _t3.diag(lam.clamp_min(1e-8).sqrt()) @ U.t())
                        Wis.append(U @ _t3.diag(1.0 / lam.clamp_min(1e-8).sqrt()) @ U.t())
                    _vw_cache[layer] = (_t3.stack(Ws), _t3.stack(Wis))
            return _vw_cache[layer]

        _pq.set_vw_provider(_vw_provider)
        print("[pca_launcher] N20 W_O-metric V provider enabled (LongCat)", flush=True)
    elif "HY-WorldPlay" in _tgt_vw:
        import torch as _t4
        import wan.models.dits.arwan_w_action_w_mem_relative_rope as _hy2
        import functools as _ft2
        import inspect as _insp2

        _vw_mods_hy = {}
        _vw_cache_hy = {}
        _orig_call_vw = _hy2.CausalCameraPRopeWanAttnProcessor2_0.__call__

        @_ft2.wraps(_orig_call_vw)
        def _call_vw(self, attn, *a, **kw):
            if id(attn) not in _vw_mods_hy:
                _vw_mods_hy[id(attn)] = (len(_vw_mods_hy), attn)
            return _orig_call_vw(self, attn, *a, **kw)

        _call_vw.__signature__ = _insp2.signature(_orig_call_vw)
        _hy2.CausalCameraPRopeWanAttnProcessor2_0.__call__ = _call_vw
        _vw_by_layer = {}

        def _vw_provider_hy(call_idx):
            if not _vw_mods_hy:
                return None
            if not _vw_by_layer:
                for _id, (lay, m) in _vw_mods_hy.items():
                    _vw_by_layer[lay] = m
            L = len(_vw_by_layer)
            layer = call_idx % L
            if layer not in _vw_cache_hy:
                mod = _vw_by_layer.get(layer)
                if mod is None:
                    return None
                with _t4.no_grad():
                    Wt = mod.to_out[0].weight.detach().float()
                    H = mod.heads
                    D = Wt.shape[1] // H
                    Ws, Wis = [], []
                    for h in range(H):
                        blk = Wt[:, h * D:(h + 1) * D]
                        G = blk.t() @ blk
                        G = G + 1e-3 * G.diagonal().mean() * _t4.eye(D, device=G.device)
                        lam, U = _t4.linalg.eigh(G)
                        Ws.append(U @ _t4.diag(lam.clamp_min(1e-8).sqrt()) @ U.t())
                        Wis.append(U @ _t4.diag(1.0 / lam.clamp_min(1e-8).sqrt()) @ U.t())
                    _vw_cache_hy[layer] = (_t4.stack(Ws), _t4.stack(Wis))
            return _vw_cache_hy[layer]

        _pq.set_vw_provider(_vw_provider_hy)
        print("[pca_launcher] N20 W_O-metric V provider enabled (HY)", flush=True)
    elif "Self-Forcing" in _tgt_vw:
        import torch as _t5
        import wan.modules.causal_model as _sfm2

        _vw_mods_sf = []
        _vw_cache_sf = {}
        _orig_sf_init_vw = _sfm2.CausalWanSelfAttention.__init__

        def _sf_init_vw(self, *a, **kw):
            _orig_sf_init_vw(self, *a, **kw)
            _vw_mods_sf.append(self)

        _sfm2.CausalWanSelfAttention.__init__ = _sf_init_vw

        def _vw_provider_sf(call_idx):
            if not _vw_mods_sf:
                return None
            layer = call_idx % len(_vw_mods_sf)
            if layer not in _vw_cache_sf:
                mod = _vw_mods_sf[layer]
                with _t5.no_grad():
                    Wt = mod.o.weight.detach().float()
                    H = mod.num_heads
                    D = mod.head_dim
                    Ws, Wis = [], []
                    for h in range(H):
                        blk = Wt[:, h * D:(h + 1) * D]
                        G = blk.t() @ blk
                        G = G + 1e-3 * G.diagonal().mean() * _t5.eye(D, device=G.device)
                        lam, U = _t5.linalg.eigh(G)
                        Ws.append(U @ _t5.diag(lam.clamp_min(1e-8).sqrt()) @ U.t())
                        Wis.append(U @ _t5.diag(1.0 / lam.clamp_min(1e-8).sqrt()) @ U.t())
                    _vw_cache_sf[layer] = (_t5.stack(Ws), _t5.stack(Wis))
            return _vw_cache_sf[layer]

        _pq.set_vw_provider(_vw_provider_sf)
        print("[pca_launcher] N20 W_O-metric V provider enabled (SF)", flush=True)
    else:
        print("[pca_launcher] PCA_V_METRIC=wo: no provider for this target", flush=True)

_target = os.environ.get("PCA_TARGET", "experiments/LongCat/run_long_t2v.py")
runpy.run_path(_target, run_name="__main__")
