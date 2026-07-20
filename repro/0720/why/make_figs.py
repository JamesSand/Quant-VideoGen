#!/usr/bin/env python3
"""Why-analysis figures 1-5 from h1_h2_data.npz + H3 PC-plane scatter."""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
font_manager.fontManager.addfont(os.path.expanduser("~/.local/share/fonts/NotoSansSC.ttf"))
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans SC"

os.chdir("/home/zhizhousha/workspace/video-project/Quant-VideoGen")
OUT = "repro/0720/why"
d = dict(np.load(f"{OUT}/h1_h2_data.npz"))
MODELS = ("lc", "sf", "hy")
MNAME = {"lc": "LongCat", "sf": "Self-Forcing", "hy": "HY-WorldPlay"}
C = {"lc": "#4A7AB5", "sf": "#3E8E5A", "hy": "#D97706"}

# Fig1: singular spectra
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
for m in MODELS:
    s = d[f"{m}_spec"]
    ax.semilogy(np.arange(1, len(s) + 1), s / s[0], label=f"{MNAME[m]} (top-4 能量 {'%.0f' % (100*s[:4].sum()/s.sum())}%)", color=C[m])
ax.set_xlabel("特征值序号"); ax.set_ylabel("λ_i / λ_1(log)")
ax.set_title("K 协方差谱:低秩薄饼程度 LC ≫ SF ≫ HY")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.savefig(f"{OUT}/fig1_spectra.png", bbox_inches="tight"); plt.close(fig)

# Fig2: subtraction efficiency (energy removed vs metadata bits) — H1 verdict
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=150)
for ax, m in zip(axes, MODELS):
    rp = d[f"{m}_rank_pts"]; kp = d[f"{m}_kmeans_sub_pts"]
    ax.plot(rp[:, 0], rp[:, 1] * 100, "o-", color="#4A7AB5", label="PCA 子空间 (r=1..16)")
    ax.plot(kp[:, 1], kp[:, 2] * 100, "s-", color="#B5484A", label="k-means 字典 (K=16..1024)")
    ax.set_title(f"{MNAME[m]}"); ax.set_xlabel("减法元数据 bits/elem"); ax.set_ylabel("消掉能量 %")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.suptitle("H1 判决:字典减法每比特消掉的能量【更多】——H1 原命题证伪", y=1.03)
fig.savefig(f"{OUT}/fig2_subtraction_efficiency.png", bbox_inches="tight"); plt.close(fig)

# Fig3: residual-stage efficiency (the real mechanism)
rel = {"lc": (0.0940, 0.0805), "sf": (0.0886, 0.0830), "hy": (0.2090, 0.1856)}  # qvg, ours (bench v13)
rem = {"lc": (0.985, 0.821), "sf": (0.951, 0.688), "hy": (0.792, 0.456)}        # kmeans K=256, rank-4
fig, ax = plt.subplots(figsize=(6.5, 4), dpi=150)
x = np.arange(3); w = 0.35
qvg_eff = [rel[m][0] ** 2 / (1 - rem[m][0]) for m in MODELS]
our_eff = [rel[m][1] ** 2 / (1 - rem[m][1]) for m in MODELS]
ax.bar(x - w/2, qvg_eff, w, color="#B5484A", label="QVG(kmeans 残差≈白噪声)")
ax.bar(x + w/2, our_eff, w, color="#4A7AB5", label="Budget-PCA(残差保留通道/时间结构)")
for i, m in enumerate(MODELS):
    ax.text(i - w/2, qvg_eff[i], f"{qvg_eff[i]:.3f}", ha="center", va="bottom", fontsize=8)
    ax.text(i + w/2, our_eff[i], f"{our_eff[i]:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([MNAME[m] for m in MODELS])
ax.set_ylabel("最终误差能量 / 残差能量(越低=残差格效率越高)")
ax.set_title("真机制:2-bit 残差格的能量回收效率差 3-16×")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
fig.savefig(f"{OUT}/fig3_residual_efficiency.png", bbox_inches="tight"); plt.close(fig)

# Fig4: per-channel error ratio (H2)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=150)
for ax, m in zip(axes, MODELS):
    for arm, col in (("qvg", "#B5484A"), ("kivi", "#8A6FB0"), ("ours", "#4A7AB5")):
        ce = d[f"{m}_cherr_{arm}"]
        ax.semilogy(np.convolve(ce, np.ones(8)/8, mode="valid"), color=col, label=arm.upper() if arm != "ours" else "Budget-PCA")
    vr = d[f"{m}_var_ch"]
    ax.set_title(f"{MNAME[m]}(通道方差极差 {vr[0]/max(vr[-1],1e-12):.0f}×)")
    ax.set_xlabel("通道(按方差降序)"); ax.set_ylabel("误差/信号(log,滑窗8)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
fig.suptitle("H2 判决:QVG 的误差集中在小方差通道;通道轴方法免疫", y=1.03)
fig.savefig(f"{OUT}/fig4_channel_error.png", bbox_inches="tight"); plt.close(fig)

# Fig5 (H3): PC-plane scatter with kmeans centroids
sys.path.insert(0, ".")
from quant_videogen.kmeans.kmeans_euclid import batch_kmeans_Euclid
dd = torch.load("repro/0720/chunks/lc/chunk_001.pt", map_location="cuda")
blocks = dd["k"].float().reshape(-1, 64)
sub = blocks[torch.randperm(blocks.shape[0], device=blocks.device)[:150000]]
ids, cent, _, _ = batch_kmeans_Euclid(sub.unsqueeze(0), 256, max_iters=100)
cent = cent.squeeze(0)
mu = blocks.mean(0, keepdim=True)
Xc = blocks - mu
cov = (Xc.T @ Xc) / Xc.shape[0]
_, V = torch.linalg.eigh(cov)
P2 = V[:, -2:]
pts = ((blocks[:40000] - mu) @ P2).cpu().numpy()
cpts = ((cent - mu) @ P2).cpu().numpy()
fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
ax.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.08, color="#4A7AB5", label="真实 KV 块(4 万个)")
ax.scatter(cpts[:, 0], cpts[:, 1], s=14, color="#B5484A", marker="x", label="k-means 256 质心")
ax.set_xlabel("PC-1"); ax.set_ylabel("PC-2")
ax.set_title("H3:LC 的 KV 块云在主成分平面上连续铺开\n256 个离散质心只能撒点覆盖(维度诅咒),连续子空间一步到位")
ax.legend(fontsize=8, markerscale=2)
fig.savefig(f"{OUT}/fig5_pc_plane.png", bbox_inches="tight"); plt.close(fig)
print("figs 1-5 saved")
