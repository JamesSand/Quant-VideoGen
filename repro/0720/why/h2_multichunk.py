#!/usr/bin/env python3
"""H2 跨 chunk 稳健性:小方差通道误差/信号比(QVG vs 我们),全部 8 chunk。
勘误背景:原 H2 判决只用 chunk_001;本脚本验证 LC 核心比值的跨层稳定性,
以及 SF/HY 次要论断是否跨 chunk 成立。
QVG = 原装 prq(num_stages=1, K=256, eval iters)+ int2 B64;我们 = 终版配置。
输出并入 h1_h2_data.npz 的 {model}_h2_multichunk(列:chunk, qvg_small, ours_small, ratio)。
"""
import os, sys
import numpy as np
import torch

os.chdir("/home/zhizhousha/workspace/video-project/Quant-VideoGen")
sys.path.insert(0, ".")
sys.path.insert(0, "repro/backup/scripts")
from quant_videogen.kmeans.kmeans_euclid import batch_kmeans_Euclid
from quant_videogen.sim.quant.lowbit_quantize import blockwise_int2_quantize_triton

ITERS = {"lc": 100, "sf": 2, "hy": 2}
# PCA_FP8SIM=1:终版臂全带 fp8,机制数字必须同口径(0720 外审勘误⑤)
CFG = {
    "lc": dict(PCA_R="4", PCA_RES_GRID="asym", PCA_RES_BLOCK="128", PCA_RES_AXIS_K="channel",
               PCA_FP8SIM="1"),
    "sf": dict(PCA_R="4", PCA_RES_GRID="asym", PCA_RES_BLOCK="128", PCA_RES_AXIS_K="channel",
               PCA_FP8SIM="1"),
    "hy": dict(PCA_R="4", PCA_HALF_R_K="9,0", PCA_RES_GRID="asym", PCA_RES_BLOCK="128",
               PCA_RES_AXIS_K="channel", PCA_RES_BLOCK_K="64",
               PCA_RES_GRID_KP="ternary", PCA_RES_BLOCK_KP="64", PCA_FP8SIM="1"),
}

def qvg_recon(k, iters):
    B, H, S, D = k.shape
    X = k.view(B * H, S, D).contiguous()
    labels, cent, _, _ = batch_kmeans_Euclid(X, n_clusters=256, max_iters=iters)
    g = torch.gather(cent, 1, labels.long().unsqueeze(-1).expand(-1, -1, D))
    rq = blockwise_int2_quantize_triton((X - g).view(B, H, S, D).contiguous(), block_size=64)
    return g.view(B, H, S, D) + rq

def small_ch_ratio(k, recon, n=16):
    var = k.float().var(dim=(0, 1, 2))          # [D]
    idx = var.argsort()[:n]
    err = ((k - recon).float() ** 2).mean(dim=(0, 1, 2))
    sig = (k.float() ** 2).mean(dim=(0, 1, 2))
    return float((err[idx].sum() / sig[idx].sum()))

res = dict(np.load("repro/0720/why/h1_h2_data.npz"))
for model in ("lc", "sf", "hy"):
    for e in list(os.environ):
        if e.startswith("PCA_"):
            del os.environ[e]
    os.environ.update(CFG[model])
    import importlib
    import pca_quant
    importlib.reload(pca_quant)
    rows = []
    for ci in range(8):
        k = torch.load(f"repro/0720/chunks/{model}/chunk_{ci:03d}.pt", map_location="cuda")["k"].float()
        rq = qvg_recon(k, ITERS[model])
        ro, _ = pca_quant.pca_fake_quant_kv(k.to(torch.bfloat16), k.to(torch.bfloat16))
        a, b = small_ch_ratio(k, rq), small_ch_ratio(k, ro.float())
        rows.append((ci, a, b, a / b))
        print(f"{model} chunk{ci:03d}: 小方差16通道 err/sig QVG {a:.4f} vs 我们 {b:.4f} = {a/b:.2f}×", flush=True)
    res[f"{model}_h2_multichunk"] = np.array(rows)
np.savez("repro/0720/why/h1_h2_data.npz", **res)
print("saved h2_multichunk")
