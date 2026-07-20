#!/usr/bin/env python3
"""Why-analysis figures 1-5 from h1_h2_data.npz + H3 PC-plane scatter.

0720 勘误版:fig2/fig3/fig5 改为 QVG 真实聚类口径(per-head 全 D 维 token,
prq num_stages=1/K=256,数据键 {m}_realpath/{m}_ourspath,来自 h1_real_path.py
与 h1_ours_path.py);fig1 加跨 chunk(=跨层)稳健性;fig3 的效率比改为
chunk 内自洽口径(同 chunk 的最终误差/残差能量),不再混用视频级 relL2。
"""
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

# Fig1: singular spectra — chunk_001 主图 + 跨 chunk top-4 范围(勘误:排序层依赖)
fig, ax = plt.subplots(figsize=(6.5, 4), dpi=150)
for m in MODELS:
    tops = []
    for ci in range(8):
        k = torch.load(f"repro/0720/chunks/{m}/chunk_{ci:03d}.pt", map_location="cuda")["k"].float()
        B, H, S, D = k.shape
        X = k.reshape(B * H, S, D)
        Xc = X - X.mean(1, keepdim=True)
        ev = torch.linalg.eigvalsh(torch.einsum("bsd,bse->bde", Xc, Xc) / S).flip(-1)
        tops.append(float((ev[:, :4].sum(-1) / ev.sum(-1)).mean()))
        if ci == 1:
            s = ev.mean(0).cpu().numpy()
    ax.semilogy(np.arange(1, len(s) + 1), s / s[0], color=C[m],
                label=f"{MNAME[m]} top-4 能量:chunk_001 {tops[1]*100:.0f}%(8 层均值 {np.mean(tops)*100:.0f}%,极差 {min(tops)*100:.0f}–{max(tops)*100:.0f}%)")
ax.set_xlabel("特征值序号"); ax.set_ylabel("λ_i / λ_1(log)")
ax.set_title("K 协方差谱(chunk_001):低秩薄饼层依赖——早层最扁,跨模型排序不稳")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.savefig(f"{OUT}/fig1_spectra.png", bbox_inches="tight"); plt.close(fig)

# Fig2: H1 判决(修正口径)——原空间消掉能量 vs 元数据 bits,per-head 全 D 维
# kmeans bits = idx 8/D + 质心 K*16/S;ours bits = 系数(2r+16)/D + 基/μ 摊销
SD = {"lc": (29640, 128), "sf": (37440, 128), "hy": (7040, 256)}
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=150)
for ax, m in zip(axes, MODELS):
    S, D = SD[m]
    rp = d[f"{m}_realpath"]      # rows: chunk, K, removed, relsq
    op = d[f"{m}_ourspath"]      # rows: chunk, removed, relsq
    r_eff = 9 if m == "hy" else 4
    ours_bits = (2 * r_eff + 16) / D + (r_eff + 1) * 16 / S
    for i, ci in enumerate(range(8)):
        sel = rp[(rp[:, 0] == ci) & (rp[:, 1] == 256)]
        km_bits = 8 / D + 256 * 16 / S
        ax.plot(km_bits, sel[0, 2] * 100, "o", color="#B5484A", alpha=0.75,
                label="k-means K=256(per-head 全 D 维)" if i == 0 else None)
        osel = op[op[:, 0] == ci]
        ax.plot(ours_bits, osel[0, 1] * 100, "o", color="#4A7AB5", alpha=0.75,
                label=f"μ+PCA r={r_eff}(终版)" if i == 0 else None)
        ax.plot([km_bits, ours_bits], [sel[0, 2] * 100, osel[0, 1] * 100],
                color="#999999", lw=0.6, alpha=0.5, zorder=0)
    ax.set_title(f"{MNAME[m]}(8 chunk,一线一层)")
    ax.set_xlabel("减法元数据 bits/elem"); ax.set_ylabel("消掉能量 %")
    ax.margins(x=0.2, y=0.18)
    ax.legend(fontsize=7, loc="lower center"); ax.grid(alpha=0.3)
fig.suptitle("H1 判决(真实口径,8 chunk):LC/SF 同等比特下字典减得仍更多(证伪不变);HY 字典元数据 4×", y=1.03)
fig.savefig(f"{OUT}/fig2_subtraction_efficiency.png", bbox_inches="tight"); plt.close(fig)

# Fig3: 残差格效率(chunk 内自洽:最终误差能量/残差能量;bar=4 chunk 均值,须=极差)
fig, ax = plt.subplots(figsize=(6.5, 4), dpi=150)
x = np.arange(3); w = 0.35
for off, key, col, lab in ((-w/2, "realpath", "#B5484A", "QVG(kmeans 残差≈白噪声)"),
                           (w/2, "ourspath", "#4A7AB5", "Budget-PCA(残差保留通道结构)")):
    means, lo, hi = [], [], []
    for m in MODELS:
        rows = d[f"{m}_{key}"]
        if key == "realpath":
            rows = rows[rows[:, 1] == 256]
            eff = rows[:, 3] / (1 - rows[:, 2])
        else:
            eff = rows[:, 2] / (1 - rows[:, 1])
        means.append(eff.mean()); lo.append(eff.mean() - eff.min()); hi.append(eff.max() - eff.mean())
    ax.bar(x + off, means, w, color=col, label=lab, yerr=[lo, hi], capsize=3)
    for i, v in enumerate(means):
        ax.text(i + off, v + hi[i] + 0.01, f"{v:.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([MNAME[m] for m in MODELS])
ax.set_ylabel("最终误差能量 / 残差能量(低=格效率高)")
ax.set_title("真机制(修正口径,8 chunk):QVG 回收 ~48%(白噪声水平)vs 我们 ~75%,~2×")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
fig.savefig(f"{OUT}/fig3_residual_efficiency.png", bbox_inches="tight"); plt.close(fig)

# Fig4: per-channel error ratio (H2) — chunk_001,小图注加跨 chunk 比值范围
mc_note = {}
for m in MODELS:
    r = d[f"{m}_h2_multichunk"]      # chunk, qvg, ours, ratio
    mc_note[m] = f"跨 8 层比值 {r[:,3].min():.1f}–{r[:,3].max():.1f}×"
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=150)
for ax, m in zip(axes, MODELS):
    for arm, col in (("qvg", "#B5484A"), ("kivi", "#8A6FB0"), ("ours", "#4A7AB5")):
        ce = d[f"{m}_cherr_{arm}"]
        ax.semilogy(np.convolve(ce, np.ones(8)/8, mode="valid"), color=col,
                    label=arm.upper() if arm != "ours" else "Budget-PCA")
    vr = d[f"{m}_var_ch"]
    ax.set_title(f"{MNAME[m]}(方差极差 {vr[0]/max(vr[-1],1e-12):.0f}×;{mc_note[m]})")
    ax.set_xlabel("通道(按方差降序)"); ax.set_ylabel("误差/信号(log,滑窗8)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
fig.suptitle("H2 判决:QVG 误差集中在小方差通道;LC 判决性(1.8-2.4×,8/8)且跨层稳", y=1.03)
fig.savefig(f"{OUT}/fig4_channel_error.png", bbox_inches="tight"); plt.close(fig)

# Fig5 (H3, 修正口径): per-head 全 D 维 token 云 + 该 head 的 256 质心
sys.path.insert(0, ".")
from quant_videogen.kmeans.kmeans_euclid import batch_kmeans_Euclid
dd = torch.load("repro/0720/chunks/lc/chunk_001.pt", map_location="cuda")
k = dd["k"].float()
B, H, S, D = k.shape
Xh = k.reshape(B * H, S, D)[0]                       # head 0 的全部 token
ids, cent, _, _ = batch_kmeans_Euclid(Xh.unsqueeze(0), 256, max_iters=100)
cent = cent.squeeze(0)
mu = Xh.mean(0, keepdim=True)
Xc = Xh - mu
cov = (Xc.T @ Xc) / Xc.shape[0]
_, V = torch.linalg.eigh(cov)
P2 = V[:, -2:]
pts = ((Xh - mu) @ P2).cpu().numpy()
cpts = ((cent - mu) @ P2).cpu().numpy()
fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
ax.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.10, color="#4A7AB5", label=f"head-0 全部 token({S} 个,128 维)")
ax.scatter(cpts[:, 0], cpts[:, 1], s=14, color="#B5484A", marker="x", label="该 head 的 k-means 256 质心")
ax.set_xlabel("PC-1"); ax.set_ylabel("PC-2")
ax.set_title("H3(真实口径):LC 单 head 的 token 云连续铺开\n256 离散质心撒点覆盖(维度诅咒),连续子空间一步到位")
ax.legend(fontsize=8, markerscale=2)
fig.savefig(f"{OUT}/fig5_pc_plane.png", bbox_inches="tight"); plt.close(fig)
print("figs 1-5 saved")
