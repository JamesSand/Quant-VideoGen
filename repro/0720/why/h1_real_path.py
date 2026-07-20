#!/usr/bin/env python3
"""H1/H3 修正版:用 QVG 原装真实路径(prq_quantize_tensor,per-head 全 D 维聚类,
与 eval 完全同配置 num_stages=1/K=256/int2 B64)测"质心减法消掉的能量"与最终误差。
勘误背景:旧版 h1_kmeans_sub.py 把所有 head 的 64 维块拼起来做全局聚类,口径错误
(QVG 真口径:centroids (B,H,K,D)、cluster_ids (B,H,S),每 head 全 D 维 token 聚类)。
全部 8 个 dump chunk(=8 个不同层)逐个复核。
输出并入 h1_h2_data.npz 的 {model}_realpath(列:chunk_id, K, removed, relL2sq_final)。
"""
import os, sys
import numpy as np
import torch

os.chdir("/home/zhizhousha/workspace/video-project/Quant-VideoGen")
sys.path.insert(0, ".")
sys.path.insert(0, "repro/backup/scripts")
from quant_videogen.kmeans.kmeans_euclid import batch_kmeans_Euclid
from quant_videogen.sim.quant.lowbit_quantize import blockwise_int2_quantize_triton

ITERS = {"lc": 100, "sf": 2, "hy": 2}  # 与 campaign.sh 的 QVG eval 配置一致

res = dict(np.load("repro/0720/why/h1_h2_data.npz"))
for model in ("lc", "sf", "hy"):
    rows = []
    for ci in range(8):
        f = f"repro/0720/chunks/{model}/chunk_{ci:03d}.pt"
        k = torch.load(f, map_location="cuda")["k"].float()
        B, H, S, D = k.shape
        X = k.view(B * H, S, D).contiguous()
        tot = float((X ** 2).sum())
        for K in (64, 256, 1024):
            labels, cent, _, _ = batch_kmeans_Euclid(X, n_clusters=K, max_iters=ITERS[model])
            gathered = torch.gather(cent, 1, labels.long().unsqueeze(-1).expand(-1, -1, D))
            resid = X - gathered
            removed = 1 - float((resid ** 2).sum()) / tot
            relsq = float("nan")
            if K == 256:  # eval 配置:残差过 QVG 原装 int2 B64 三角格
                rq = blockwise_int2_quantize_triton(resid.view(B, H, S, D).contiguous(), block_size=64)
                recon = gathered.view(B, H, S, D) + rq
                relsq = float(((k - recon) ** 2).sum()) / tot
            rows.append((ci, K, removed, relsq))
            msg = f"{model} chunk{ci:03d} K={K}: 质心消掉 {removed*100:.2f}%"
            if K == 256:
                msg += f"  最终 relL2² {relsq*100:.3f}%  残差回收率 {(1 - relsq/(1-removed))*100:.1f}%"
            print(msg, flush=True)
    res[f"{model}_realpath"] = np.array(rows)
np.savez("repro/0720/why/h1_h2_data.npz", **res)
print("saved h1_h2_data.npz:{model}_realpath")
