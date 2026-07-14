"""QKV anatomy plots from sf_qkv.pt (3 layers x 3 time windows, last denoise step).

Fig 1 qkv_time.png   — time windows (layer 15): Q/K/V norm boxplots per window +
                       K/V |value| heatmaps at begin vs end
Fig 2 qkv_chunk.png  — intra-block structure (layer 15): relative norm vs position
                       within a 3-latent block and within a single latent frame
Fig 3 qkv_depth.png  — layer 0 vs 15 vs 29 (mid window): norm boxplots + K/V heatmaps
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = "/home/zhizhousha/workspace/video-project/Quant-VideoGen"
FIGS = os.path.join(REPO, "repro/0714/figs")
FSL = 1560  # tokens per latent frame

d = torch.load(os.path.join(REPO, "results/kvplot/sf_qkv.pt"),
               map_location="cpu", weights_only=False)
slots = d["slots"]
LAYERS = d["layers"]
WIN = {"begin": (0, 5), "mid": (87, 92), "end": (174, 179)}


def get(layer, win):
    a, b = WIN[win]
    frames = sorted(f for (l, f) in slots if l == layer and a <= f <= b)
    out = {}
    for t in ("q", "k", "v"):
        out[t] = torch.cat([slots[(layer, f)][t] for f in frames], dim=1)[0].float()  # [S,H,D]
    return out


def norms(x):          # [S,H,D] -> [S*H]
    return x.norm(dim=-1).flatten().numpy()


def tok_norm_med(x):   # [S,H,D] -> [S]
    return x.norm(dim=-1).median(dim=1).values.numpy()


L = 15


# ---- per-token norm panel: x = token index (full window), y = token norm.
#      one thin line per head + bold median line across heads.
def fig3_panel(ax, x, title):
    nrm = x.norm(dim=-1).numpy()          # [S, H]
    S, H = nrm.shape
    xs = np.arange(S)
    for h in range(H):
        ax.plot(xs, nrm[:, h], lw=0.25, color="steelblue", alpha=0.25)
    ax.plot(xs, np.median(nrm, axis=1), lw=0.9, color="navy", label="median over heads")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Token index", fontsize=8); ax.set_ylabel("Token L2 norm", fontsize=8)
    ax.grid(alpha=0.3)


def grid9(cols, coldata, fname, suptitle):
    """rows = Q/K/V, cols = 3 conditions; row-shared ylim for comparability."""
    fig = plt.figure(figsize=(15, 10))
    for i, tname in enumerate(("q", "k", "v")):
        xs = [coldata(c)[tname] for c in cols]
        lo = min(float(x.norm(dim=-1).min()) for x in xs)
        hi = max(float(x.norm(dim=-1).max()) for x in xs)
        pad = 0.05 * (hi - lo)
        for j, (c, x) in enumerate(zip(cols, xs)):
            ax = fig.add_subplot(3, 3, i * 3 + j + 1)
            fig3_panel(ax, x, f"{tname.upper()} — {c}")
            ax.set_ylim(lo - pad, hi + pad)
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, fname), dpi=140); plt.close(fig)


# ---------------- Fig 1: time windows (rows Q/K/V x cols begin/mid/end, L15) ----------------
grid9(list(WIN), lambda w: get(15, w), "qkv_time.png",
      "SF QKV per-token-position norms (OScaR Fig.3 style) — video begin / mid / end (L15)")

# ---------------- Fig 2: intra-block / intra-frame ----------------
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
for t, c in (("q", "tab:green"), ("k", "tab:blue"), ("v", "tab:red")):
    per_block = []
    for w in WIN:
        x = get(L, w)[t]                       # [2 blocks *4680, H, D]
        S = x.shape[0]
        tn = x.norm(dim=-1).median(dim=1).values.view(S // 4680, 4680)
        per_block.append(tn)
    curve = torch.cat(per_block, 0).median(dim=0).values.numpy()
    axes[0].plot(curve / np.median(curve), lw=0.7, color=c, label=t.upper())
    frame_curve = torch.cat(per_block, 0).view(-1, 3, FSL).flatten(0, 1).median(dim=0).values.numpy()
    axes[1].plot(frame_curve / np.median(frame_curve), lw=0.7, color=c, label=t.upper())
for fb in (FSL, 2 * FSL):
    axes[0].axvline(fb, color="gray", lw=0.8, ls="--")
axes[0].set_title(f"relative token norm within a 3-latent block (L{L}, all windows pooled)", fontsize=10)
axes[0].set_xlabel("position in block (dashes = latent-frame boundaries)")
axes[1].set_title("relative token norm within ONE latent frame (30x52 spatial scan)", fontsize=10)
axes[1].set_xlabel("position in frame (row-major; 52 tokens per row)")
for ax in axes:
    ax.set_ylabel("norm / median"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.suptitle("SF QKV — intra-chunk structure", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "qkv_chunk.png"), dpi=140); plt.close(fig)

# ---------------- Fig 3: depth (rows Q/K/V x cols L0/L15/L29, mid window) ----------------
grid9(LAYERS, lambda l: get(l, "mid"), "qkv_depth.png",
      "SF QKV per-token-position norms (OScaR Fig.3 style) — layer 0 / 15 / 29 (mid window)")

# ---------------- stats for the md ----------------
print("layer,window,tensor,med_norm,toknorm_ratio,absmax")
for l in LAYERS:
    for w in WIN:
        g = get(l, w)
        for t in ("q", "k", "v"):
            tm = tok_norm_med(g[t])
            print(f"L{l},{w},{t.upper()},{np.median(tm):.2f},{tm.max()/tm.min():.2f}x,{g[t].abs().max():.1f}")
