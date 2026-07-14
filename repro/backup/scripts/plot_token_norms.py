"""OScaR-Figure-3-style token-norm statistics for the three QVG models.

Per model, per tensor (K, V):
  left  — boxplots of head-wise L2 norms at 20 evenly spaced token positions
          (OScaR eq.5: N_t = {||t_h||_2, h=1..H})
  right — full-window curve: median-over-heads norm per token (+ min head),
          to expose outlier tokens / frame periodicity.
Usage: plot_token_norms.py <sf|lc|hy> <out.png>
"""
import importlib.util
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

spec = importlib.util.spec_from_file_location(
    "plot_kv3d", "/home/zhizhousha/workspace/video-project/Quant-VideoGen/repro/backup/scripts/plot_kv3d.py")
kv3d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kv3d)

MODEL = {
    "sf": ("Self-Forcing (Wan 1.3B), Layer 15/30", kv3d.load_sf, 1560),
    "lc": ("LongCat-Video (13.6B), Layer 24/48", kv3d.load_lc, 1560),
    "hy": ("HY-WorldPlay (8B), Layer 13/30 (D=256 concat)", kv3d.load_hy, 3520),
}


def fig3_panels(fig, col, x, name):
    """OScaR Figure-3 style: top = norm boxplots per token position, bottom = heatmap."""
    S, H, D = x.shape
    norms = x.norm(dim=-1).numpy()          # [S, H]
    pos = np.linspace(0, S - 1, 16).astype(int)
    axb = fig.add_subplot(2, 2, col + 1)
    axb.boxplot([norms[p] for p in pos], positions=range(len(pos)), widths=0.6,
                flierprops=dict(marker=".", markersize=3))
    axb.set_xticks(range(len(pos)))
    axb.set_xticklabels([str(p) for p in pos], rotation=60, fontsize=7)
    axb.set_xlabel("Token Position"); axb.set_ylabel("L2 Norm Distribution")
    axb.set_title(f"({chr(97+col)}) {name} L2 norm distribution", fontsize=10)
    axb.grid(alpha=0.3, axis="y")

    axh = fig.add_subplot(2, 2, col + 3)
    m = x[:, x.shape[1] // 2, :].numpy()    # middle head, signed values
    v = np.percentile(np.abs(m), 99)
    im = axh.imshow(m[::max(1, S // 800)], aspect="auto", cmap="coolwarm", vmin=-v, vmax=v)
    axh.set_xlabel("Channel Dimension"); axh.set_ylabel("Token Position")
    axh.set_title(f"({chr(99+col)}) {name} heatmap (mid head)", fontsize=10)
    fig.colorbar(im, ax=axh, fraction=0.03, pad=0.02, label="Activation Value")


def norm_panels(fig, row, x, name, frame_len):
    S, H, D = x.shape
    norms = x.norm(dim=-1).numpy()          # [S, H]
    pos = np.linspace(0, S - 1, 20).astype(int)
    axb = fig.add_subplot(2, 2, row * 2 + 1)
    axb.boxplot([norms[p] for p in pos], positions=range(len(pos)), widths=0.6,
                flierprops=dict(marker=".", markersize=3))
    axb.set_xticks(range(len(pos)))
    axb.set_xticklabels([str(p) for p in pos], rotation=60, fontsize=7)
    axb.set_xlabel("Token position"); axb.set_ylabel("L2 norm across heads")
    axb.set_title(f"{name}: head-wise norm boxplots (OScaR eq.5)", fontsize=10)
    axb.grid(alpha=0.3, axis="y")

    axc = fig.add_subplot(2, 2, row * 2 + 2)
    med = np.median(norms, axis=1); mn = norms.min(axis=1)
    axc.plot(med, lw=0.6, color="tab:blue", label="median over heads")
    axc.plot(mn, lw=0.4, color="tab:orange", alpha=0.7, label="min head")
    for fb in range(frame_len, S, frame_len):
        axc.axvline(fb, color="gray", lw=0.4, alpha=0.35)
    axc.set_xlabel("Token index (gray lines = frame boundaries)")
    axc.set_ylabel("L2 norm")
    r = float(med.max() / max(med.min(), 1e-6))
    axc.set_title(f"{name}: per-token norm, max/min(median) = {r:.2f}x", fontsize=10)
    axc.legend(fontsize=8); axc.grid(alpha=0.3)


if __name__ == "__main__":
    arm, out = sys.argv[1], sys.argv[2]
    title, loader, frame_len = MODEL[arm]
    K, V = loader()
    fig = plt.figure(figsize=(13, 8.5))
    fig3_panels(fig, 0, K, "Key")
    fig3_panels(fig, 1, V, "Value")
    fig.suptitle(f"OScaR-Fig.3-style token-norm view — {title}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    # console stats for the md
    for nm, x in (("K", K), ("V", V)):
        n = x.norm(dim=-1)
        med = n.median(dim=1).values
        print(f"{arm} {nm}: median-norm range {float(med.min()):.2f}..{float(med.max()):.2f} "
              f"(ratio {float(med.max()/med.min()):.2f}x); "
              f"p1/p50/p99 = {float(np.percentile(med,1)):.2f}/{float(np.percentile(med,50)):.2f}/{float(np.percentile(med,99)):.2f}")
    print("saved", out)
