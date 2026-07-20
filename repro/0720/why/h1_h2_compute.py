#!/usr/bin/env python3
"""Why-analysis H1+H2 computation on real dumped chunks.

H1: singular spectra + subtraction-efficiency curves (rank-r sweep vs K-means
    K sweep at matched metadata bits). Uses QVG's own kmeans for the dictionary.
H2: per-channel variance spectra + per-channel residual error ratio under
    (QVG kmeans+RTN residual) vs (Budget-PCA) vs (KIVI).
Outputs npz for plotting.
"""
import os, sys
import numpy as np
import torch

os.chdir("/home/zhizhousha/workspace/video-project/Quant-VideoGen")
sys.path.insert(0, "repro/0720/kernel")
sys.path.insert(0, ".")
from bp_quant import bp_encode_fast, bp_decode
from quant_videogen.functions import triton_prq_quantize_tensor, triton_prq_dequantize_tensor

OUT = "repro/0720/why"
os.makedirs(OUT, exist_ok=True)
res = {}

for model in ("lc", "sf", "hy"):
    d = torch.load(f"repro/0720/chunks/{model}/chunk_001.pt", map_location="cuda")
    k = d["k"].float()
    B, H, S, D = k.shape
    X = k.reshape(B * H, S, D)
    Xc = X - X.mean(1, keepdim=True)
    # --- H1a: singular spectrum (per head, averaged) ---
    cov = torch.einsum("bsd,bse->bde", Xc, Xc) / S
    ev = torch.linalg.eigvalsh(cov)                     # ascending
    spec = ev.flip(-1).mean(0).cpu().numpy()            # descending, head-avg
    res[f"{model}_spec"] = spec
    # --- H1b: subtraction-efficiency: energy removed vs metadata bits ---
    # rank sweep: coef bits/elem = (r*2 + 24)/D  (2bit codes + fp8 s/z + amortized)
    tot = float((Xc * Xc).sum())
    ranks = [1, 2, 4, 8, 16]
    r_pts = []
    _, vecs = torch.linalg.eigh(cov)
    for r in ranks:
        Vr = vecs[:, :, -r:]
        c = torch.einsum("bsd,bdr->bsr", Xc, Vr)
        en = float((c * c).sum()) / tot
        bits = (r * 2 + 24) / D
        r_pts.append((bits, en))
    res[f"{model}_rank_pts"] = np.array(r_pts)
    # kmeans sweep (QVG's own kernel; K clusters on 64-dim blocks like their config)
    k_pts = []
    for K in (16, 64, 256, 1024):
        try:
            q = triton_prq_quantize_tensor(k, num_stages=1, num_clusters=K, block_size=64,
                                           max_iters=100, quantize_fn=lambda t: 2)
            # energy removed by centroids alone: reconstruct WITHOUT residual codes:
            # dequant then subtract residual part is entangled; approximate via
            # dequantized-minus-input energy of the residual stage input if exposed.
            # Fallback: centroid assignment energy = ||X||^2 - ||X - centroid||^2
            deq = triton_prq_dequantize_tensor(q, block_size=64, num_bits=2).float()
            # deq includes residual quantization; isolate centroid contribution via
            # re-encoding with residual bits contribution removed is not exposed ->
            # measure TOTAL relL2 instead and record with its total bits.
            err = float(((deq - k) ** 2).sum()) / float((k * k).sum())
            idx_bits = 8 / 64                      # uint8 assignment per 64-dim block
            cb_amort = K * 64 * 16 / k.numel()     # codebook bf16 amortized
            k_pts.append((K, idx_bits + cb_amort, 1 - err))
        except Exception as e:
            print(f"{model} K={K} failed: {e}")
    res[f"{model}_kmeans_pts"] = np.array(k_pts)
    # --- H2: per-channel variance + per-channel error ratios ---
    var_ch = Xc.var(dim=1).mean(0).cpu().numpy()        # [D] head-avg
    res[f"{model}_var_ch"] = np.sort(var_ch)[::-1]
    # errors per channel under three methods (K tensor)
    q_qvg = triton_prq_dequantize_tensor(
        triton_prq_quantize_tensor(k, num_stages=1, num_clusters=256, block_size=64,
                                   max_iters=100 if model == "lc" else 2, quantize_fn=lambda t: 2),
        block_size=64, num_bits=2).float()
    if model == "hy":
        ours = torch.cat([
            bp_decode(bp_encode_fast(k[..., :128].contiguous(), r=9, grid="asym", block=64, axis="channel"), dtype=torch.float32),
            bp_decode(bp_encode_fast(k[..., 128:].contiguous(), r=0, grid="ternary", block=64, axis="channel"), dtype=torch.float32)], dim=-1)
    else:
        ours = bp_decode(bp_encode_fast(k, r=4, grid="asym", block=128, axis="channel"), dtype=torch.float32)
    import importlib
    os.environ["PCA_KIVI"] = "1"; os.environ["PCA_FP8SIM"] = "1"
    sys.path.insert(0, "repro/backup/scripts")
    import pca_quant; importlib.reload(pca_quant)
    kivi, _ = pca_quant.pca_fake_quant_kv(k, k.clone())
    os.environ["PCA_KIVI"] = "0"
    def ch_err(y):
        e = ((y.float() - k) ** 2).reshape(B * H, S, D).mean(1).mean(0)
        v = (k ** 2).reshape(B * H, S, D).mean(1).mean(0).clamp_min(1e-12)
        return (e / v).cpu().numpy()
    order = np.argsort(var_ch)[::-1]
    res[f"{model}_cherr_qvg"] = ch_err(q_qvg)[order]
    res[f"{model}_cherr_ours"] = ch_err(ours)[order]
    res[f"{model}_cherr_kivi"] = ch_err(kivi)[order]
    print(f"{model} done", flush=True)

np.savez(f"{OUT}/h1_h2_data.npz", **res)
print("saved", f"{OUT}/h1_h2_data.npz")
