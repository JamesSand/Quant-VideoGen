"""SF KV summary, one clean figure per dimension (K row / V row each).
Convention (user, 0714): ALL KV value plots are 3D surfaces (X=channel, Y=token,
Z=|value|, coolwarm) — same style as plot_kv3d.py / OScaR Fig.2.
  sf_kv_time.png  — K,V value surfaces at video begin/mid/end (L15)
  sf_kv_chunk.png — K,V token-norm curves within a block + K surface of one block
  sf_kv_depth.png — K,V value surfaces at L0/L15/L29 + norm boxplots + absmax bars
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


ROWS = 500


FIXED_HEAD = 4  # comparison figures use ONE fixed head everywhere (user rule)


def surf(ax, x, title, marks=(), head=None):
    """head=None -> FIXED_HEAD. Pass head explicitly only for single exhibits."""
    if head is None:
        head = FIXED_HEAD
    m = np.abs(x[:, head, :].numpy())
    S, D = m.shape
    stride = max(1, S // ROWS)
    z = m[::stride]
    Y, X = np.mgrid[0:z.shape[0], 0:D]
    ax.plot_surface(X, Y * stride, z, cmap="coolwarm", rstride=2, cstride=2,
                    linewidth=0, antialiased=False)
    for tk in marks:  # frame boundaries drawn on the floor
        ax.plot([0, D - 1], [tk, tk], [0, 0], color="black", lw=1.5)
    med = float(x[:, head].norm(dim=-1).median())
    mx = float(x[:, head].abs().max())
    ax.set_title(f"{title} [H{head}]\nmed norm {med:.1f} | absmax {mx:.1f} (this head)", fontsize=9)
    ax.set_xlabel("Channel", labelpad=6); ax.set_ylabel("Token", labelpad=8)
    ax.view_init(elev=32, azim=-58)
    return head


# ---------------- Fig A: video begin / mid / end (L15) ----------------
fig = plt.figure(figsize=(15, 9))
for j, w in enumerate(WIN):
    surf(fig.add_subplot(2, 3, j + 1, projection="3d"), get(15, w, "k"), f"K — video {w} (L15)")
    surf(fig.add_subplot(2, 3, 3 + j + 1, projection="3d"), get(15, w, "v"), f"V — video {w} (L15)")
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

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
surf(ax2, get(15, "mid", "k")[:4680],
     "K values of ONE block (3 latent frames, L15)\nblack floor lines = frame starts; walls run straight through",
     marks=(FSL, 2 * FSL))
fig.suptitle("SF KV inside a generation chunk — no sink tokens, walls uniform", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "sf_kv_chunk.png"), dpi=140); plt.close(fig)

# ---------------- Fig C: depth ----------------
fig = plt.figure(figsize=(16, 12))
for j, l in enumerate(LAYERS):
    surf(fig.add_subplot(3, 3, j + 1, projection="3d"), get(l, "mid", "k"), f"K — layer {l} (mid window)")
    surf(fig.add_subplot(3, 3, 3 + j + 1, projection="3d"), get(l, "mid", "v"), f"V — layer {l} (mid window)")
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
# ---------------- Fig D: L29 outlier-head exhibit (H4 vs H9) ----------------
fig = plt.figure(figsize=(11, 4.8))
x = get(29, "mid", "k")
surf(fig.add_subplot(1, 2, 1, projection="3d"), x, "K — L29, normal head", head=4)
surf(fig.add_subplot(1, 2, 2, projection="3d"), x, "K — L29, OUTLIER head", head=9)
fig.suptitle("SF L29 K: head 9 is a whole-head outlier (same layer, same window, same scale rules)", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "sf_kv_l29_h9.png"), dpi=140); plt.close(fig)
print("saved 4 figures")
