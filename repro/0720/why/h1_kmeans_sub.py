#!/usr/bin/env python3
"""H1 判决曲线的 kmeans 侧:质心减法单独消掉的能量(与 rank 曲线同口径)。
用 QVG 原装 batch_kmeans_Euclid,64 维块,K=16..1024,iters=100 收敛口径。
输出并入 repro/0720/why/h1_h2_data.npz 的 {model}_kmeans_sub_pts。
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
