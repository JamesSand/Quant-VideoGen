"""SF KV summary, one clean figure per dimension (K row / V row each):
  sf_kv_time.png  — K,V value heatmaps at video begin/mid/end (L15), median norm in titles
  sf_kv_chunk.png — K,V relative token-norm within a 3-latent block + K heatmap of one block
  sf_kv_depth.png — K,V value heatmaps at L0/L15/L29 + norm boxplots + absmax bars
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = "/home/zhizhousha/workspace/video-project/Quant-VideoGen"
FIGS = os.path.join(REPO, "repro/0714/figs")
FSL = 1560

d = torch.load(os.path.join(REPO, "results/kvplot/sf_qkv.pt"), map_location="cpu", weights_only=False)
slots = d["slots"]
WIN = {"begin": (0, 5), "mid": (87, 92), "end": (174, 179)}
LAYERS = [0, 15, 29]
HEAD = 6


def get(layer, win, t):
    a, b = WIN[win]
    frames = sorted(f for (l, f) in slots if l == layer and a <= f <= b)
    return torch.cat([slots[(layer, f)][t] for f in frames], dim=1)[0].float()  # [S,H,D]


def heat(ax, x, title):
    m = x[:, HEAD, :].numpy()
    v = np.percentile(np.abs(m), 99.5)
    ax.imshow(m[::12], aspect="auto", cmap="coolwarm", vmin=-v, vmax=v)
    med = float(x.norm(dim=-1).median())
    mx = float(x.abs().max())
    ax.set_title(f"{title}\nmed norm {med:.1f} | absmax {mx:.1f}", fontsize=9)
    ax.set_xlabel("Channel"); ax.set_ylabel("Token")


# ---------------- Fig A: video begin / mid / end (L15) ----------------
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
for j, w in enumerate(WIN):
    heat(axes[0, j], get(15, w, "k"), f"K — video {w} (L15 H{HEAD})")
    heat(axes[1, j], get(15, w, "v"), f"V — video {w} (L15 H{HEAD})")
fig.suptitle("SF KV values across the video — begin / mid / end (identical walls, no drift)", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "sf_kv_time.png"), dpi=140); plt.close(fig)

# ---------------- Fig B: intra-chunk ----------------
fig = plt.figure(figsize=(14, 5))
ax1 = fig.add_subplot(1, 2, 1)
for t, c in (("k", "tab:blue"), ("v", "tab:red")):
    per_block = []
    for w in WIN:
        x = get(15, w, t)
        tn = x.norm(dim=-1).median(dim=1).values.view(-1, 4680)
        per_block.append(tn)
    curve = torch.cat(per_block, 0).median(dim=0).values.numpy()
    ax1.plot(curve / np.median(curve), lw=0.7, color=c, label=t.upper())
for fb in (FSL, 2 * FSL):
    ax1.axvline(fb, color="gray", lw=0.9, ls="--")
ax1.set_ylim(0.85, 1.25)
ax1.set_xlabel("token position within one 3-latent-frame generation block (dashes = frame starts)")
ax1.set_ylabel("token norm / block median")
ax1.set_title("K/V token norm inside a chunk (L15, 6 blocks pooled)", fontsize=10)
ax1.legend(); ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(1, 2, 2)
x = get(15, "mid", "k")[:4680]
m = x[:, HEAD, :].numpy(); v = np.percentile(np.abs(m), 99.5)
ax2.imshow(m[::6], aspect="auto", cmap="coolwarm", vmin=-v, vmax=v)
for fb in (FSL // 6, 2 * FSL // 6):
    ax2.axhline(fb, color="black", lw=0.8, ls="--")
ax2.set_title("K values of ONE block (3 latent frames, L15 H6)\nchannel walls run straight through frame boundaries", fontsize=9)
ax2.set_xlabel("Channel"); ax2.set_ylabel("Token (dashes = frame starts)")
fig.suptitle("SF KV inside a generation chunk — no sink tokens, walls uniform", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "sf_kv_chunk.png"), dpi=140); plt.close(fig)

# ---------------- Fig C: depth ----------------
fig = plt.figure(figsize=(15, 10))
for j, l in enumerate(LAYERS):
    heat(fig.add_subplot(3, 3, j + 1), get(l, "mid", "k"), f"K — layer {l} (mid window)")
    heat(fig.add_subplot(3, 3, 3 + j + 1), get(l, "mid", "v"), f"V — layer {l} (mid window)")
axk = fig.add_subplot(3, 3, 7)
axk.boxplot([get(l, "mid", "k").norm(dim=-1).flatten().numpy() for l in LAYERS],
            tick_labels=[f"L{l}" for l in LAYERS], flierprops=dict(marker=".", markersize=2))
axk.set_title("K token norms by depth — note detached cluster at L29", fontsize=9)
axk.grid(alpha=0.3, axis="y")
axv = fig.add_subplot(3, 3, 8)
axv.boxplot([get(l, "mid", "v").norm(dim=-1).flatten().numpy() for l in LAYERS],
            tick_labels=[f"L{l}" for l in LAYERS], flierprops=dict(marker=".", markersize=2))
axv.set_title("V token norms by depth — depth-insensitive", fontsize=9)
axv.grid(alpha=0.3, axis="y")
axb = fig.add_subplot(3, 3, 9)
w_ = 0.35
xs = np.arange(3)
axb.bar(xs - w_/2, [float(get(l, "mid", "k").abs().max()) for l in LAYERS], w_, label="K absmax")
axb.bar(xs + w_/2, [float(get(l, "mid", "v").abs().max()) for l in LAYERS], w_, label="V absmax")
axb.set_xticks(xs); axb.set_xticklabels([f"L{l}" for l in LAYERS])
axb.set_title("absmax by depth (quantization range driver)", fontsize=9)
axb.legend(); axb.grid(alpha=0.3, axis="y")
fig.suptitle("SF KV across depth — last layer is the quantization hot spot", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "sf_kv_depth.png"), dpi=140); plt.close(fig)
print("saved 3 figures")
