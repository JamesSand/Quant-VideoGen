#!/usr/bin/env python3
"""【已废弃,仅留痕】H1 kmeans 侧旧版:全局 64 维块聚类——口径错误。
QVG 真实实现是 per-head、全 D 维 token 聚类(prq_quantize_tensor,
centroids (B,H,K,D));block_size=64 只作用于残差量化。
正确版见 h1_real_path.py(0720 二审勘误)。
"""
import os, sys
import numpy as np
import torch

os.chdir("/home/zhizhousha/workspace/video-project/Quant-VideoGen")
sys.path.insert(0, ".")
from quant_videogen.kmeans.kmeans_euclid import batch_kmeans_Euclid

res = dict(np.load("repro/0720/why/h1_h2_data.npz"))
for model in ("lc", "sf", "hy"):
    d = torch.load(f"repro/0720/chunks/{model}/chunk_001.pt", map_location="cuda")
    k = d["k"].float()
    blocks = k.reshape(-1, 64)
    tot = float((blocks ** 2).sum())
    sub = blocks[torch.randperm(blocks.shape[0], device=blocks.device)[:150000]]
    pts = []
    for K in (16, 64, 256, 1024):
        ids, cent, sizes, nit = batch_kmeans_Euclid(sub.unsqueeze(0), K, max_iters=100)
        cent = cent.squeeze(0)
        recon = cent[torch.cdist(blocks, cent).argmin(-1)]
        removed = 1 - float(((blocks - recon) ** 2).sum()) / tot
        pts.append((K, 8 / 64 + K * 64 * 16 / k.numel(), removed))
        print(f"{model} K={K}: 质心减法消掉 {removed*100:.1f}% (iters={nit})", flush=True)
    res[f"{model}_kmeans_sub_pts"] = np.array(pts)
np.savez("repro/0720/why/h1_h2_data.npz", **res)
print("updated h1_h2_data.npz")
