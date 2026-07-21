#!/usr/bin/env python3
"""格子×残差交叉分解(0721,用户质询触发):η 差距到底来自电平数、轴、
还是残差结构?

背景:QVG 发布代码的 int2 = 三电平对称 absmax 格(get_intx_max_value(2)=1,
{-1,0,+1}×scale)。本实验在同一批真实 chunk 上,把 {kmeans 残差, PCA 残差}
× {3电平/4电平 × token 轴/channel 轴} 全交叉,所有格子元数据 ≤0.125 bits/elem
(预算公平;3lvl 只存 scale,4lvl asym 存 scale+zp 故用 B128)。
输出并入 h1_h2_data.npz 的 {model}_grid_cross(行=chunk,列=8 个格)。
"""
import os, sys
import numpy as np
import torch

os.chdir("/home/zhizhousha/workspace/video-project/Quant-VideoGen")
sys.path.insert(0, ".")
from quant_videogen.kmeans.kmeans_euclid import batch_kmeans_Euclid
from quant_videogen.sim.quant.lowbit_quantize import blockwise_int2_quantize_triton

ITERS = {"lc": 100, "sf": 2, "hy": 2}


def asym4(x, g):
    sh = x.shape
    pad = (g - sh[-1] % g) % g
    xp = torch.nn.functional.pad(x, (0, pad)) if pad else x
    xb = xp.reshape(*xp.shape[:-1], -1, g)
    mn, mx = xb.amin(-1, keepdim=True), xb.amax(-1, keepdim=True)
    sc = ((mx - mn) / 3).clamp_min(1e-8)
    q = torch.clamp(torch.round((xb - mn) / sc), 0, 3) * sc + mn
    out = q.reshape(*xp.shape)
    return out[..., :sh[-1]] if pad else out


def tern3(x, g):
    sh = x.shape
    pad = (g - sh[-1] % g) % g
    xp = torch.nn.functional.pad(x, (0, pad)) if pad else x
    xb = xp.reshape(*xp.shape[:-1], -1, g)
    sc = xb.abs().amax(-1, keepdim=True).clamp_min(1e-8)
    q = torch.clamp(torch.round(xb / sc), -1, 1) * sc
    out = q.reshape(*xp.shape)
    return out[..., :sh[-1]] if pad else out


def rec(R, Rh):
    return 1 - float(((R - Rh) ** 2).sum()) / float((R ** 2).sum())


COLS = ["km_3tok64", "km_3ch64", "km_4tok128", "km_4ch128",
        "pca_3tok64", "pca_3ch64", "pca_4tok128", "pca_4ch128"]
res = dict(np.load("repro/0720/why/h1_h2_data.npz"))
for m in ("lc", "sf"):
    rows = []
    for ci in range(8):
        k = torch.load(f"repro/0720/chunks/{m}/chunk_{ci:03d}.pt", map_location="cuda")["k"].float()
        B, H, S, D = k.shape
        X = k.view(B * H, S, D).contiguous()
        lab, cent, _, _ = batch_kmeans_Euclid(X, n_clusters=256, max_iters=ITERS[m])
        Rk = (X - torch.gather(cent, 1, lab.long().unsqueeze(-1).expand(-1, -1, D))).view(B, H, S, D)
        mu = X.mean(1, keepdim=True); Xc = X - mu
        cov = torch.einsum("bsd,bse->bde", Xc, Xc) / S
        _, V = torch.linalg.eigh(cov); Vr = V[..., -4:]
        c = torch.einsum("bsd,bdr->bsr", Xc, Vr)
        mn, mx = c.amin(-1, keepdim=True), c.amax(-1, keepdim=True)
        cs = ((mx - mn) / 3).clamp_min(1e-8)
        chat = torch.clamp(torch.round((c - mn) / cs), 0, 3) * cs + mn
        Rp = (Xc - torch.einsum("bsr,bdr->bsd", chat, Vr)).view(B, H, S, D)
        row = []
        for R in (Rk, Rp):
            Rt = R.transpose(-1, -2).contiguous()
            row += [rec(R, blockwise_int2_quantize_triton(R.contiguous(), block_size=64)),
                    rec(Rt, tern3(Rt, 64)),
                    rec(R, asym4(R, 128)),
                    rec(Rt, asym4(Rt, 128))]
        rows.append(row)
        print(f"{m} chunk{ci:03d}: " + "  ".join(f"{c_}={v*100:.1f}" for c_, v in zip(COLS, row)), flush=True)
    arr = np.array(rows)
    res[f"{m}_grid_cross"] = arr
    mean = arr.mean(0) * 100
    print(f"{m} 均值: " + "  ".join(f"{c_}={v:.1f}" for c_, v in zip(COLS, mean)), flush=True)
np.savez("repro/0720/why/h1_h2_data.npz", **res)
print("saved {model}_grid_cross")
