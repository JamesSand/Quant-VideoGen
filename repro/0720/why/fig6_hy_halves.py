#!/usr/bin/env python3
"""fig6:HY K 的 rope/prope 半区分谱("能量≠价值"反转的证据图)。"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
font_manager.fontManager.addfont(os.path.expanduser("~/.local/share/fonts/NotoSansSC.ttf"))
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans SC"

os.chdir("/home/zhizhousha/workspace/video-project/Quant-VideoGen")
d = torch.load("repro/0720/chunks/hy/chunk_001.pt", map_location="cuda")
k = d["k"].float()
B, H, S, D = k.shape
X = k.reshape(B * H, S, D)
fig, ax = plt.subplots(figsize=(6.5, 4), dpi=150)
for sl, name, col in ((slice(0, 128), "rope 半区", "#4A7AB5"),
                      (slice(128, 256), "prope 半区", "#B5484A")):
    Xh = X[..., sl]
    Xc = Xh - Xh.mean(1, keepdim=True)
    cov = torch.einsum("bsd,bse->bde", Xc, Xc) / S
    ev = torch.linalg.eigvalsh(cov).flip(-1).mean(0).cpu().numpy()
    top9 = ev[:9].sum() / ev.sum() * 100
    ax.semilogy(np.arange(1, 129), ev / ev[0], color=col, label=f"{name}(top-9 能量 {top9:.0f}%)")
    print(f"{name}: top-9 能量 {top9:.1f}%  总能量占比 {float((Xh*Xh).sum()/(X*X).sum())*100:.1f}%")
ax.set_xlabel("特征值序号"); ax.set_ylabel("λ_i/λ_1(log)")
ax.set_title("HY K 的半区分谱:秩预算 9:0 的谱学依据(反转)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.savefig("repro/0720/why/fig6_hy_halves.png", bbox_inches="tight")
print("fig6 saved")
