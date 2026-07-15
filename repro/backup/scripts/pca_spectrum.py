"""Phase-0 PCA spectrum analysis of KV activations (plan: repro/0715/pca-kv-plan.md).

Per (model, layer, tensor): per-head centered PCA over tokens.
Outputs: figs pca_spectrum / pca_h9 / pca_residual + printed stats for the md.
"""
import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = "/home/zhizhousha/workspace/video-project/Quant-VideoGen"
FIGS = os.path.join(REPO, "repro/0715/figs")
os.makedirs(FIGS, exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "plot_kv3d", os.path.join(REPO, "repro/backup/scripts/plot_kv3d.py"))
kv3d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kv3d)


def load_sf(layer, win="mid"):
    d = torch.load(os.path.join(REPO, "results/kvplot/sf_qkv.pt"), map_location="cpu", weights_only=False)
    WIN = {"begin": (0, 5), "mid": (87, 92), "end": (174, 179)}
    a, b = WIN[win]
    frames = sorted(f for (l, f) in d["slots"] if l == layer and a <= f <= b)
    K = torch.cat([d["slots"][(layer, f)]["k"] for f in frames], dim=1)[0].float()
    V = torch.cat([d["slots"][(layer, f)]["v"] for f in frames], dim=1)[0].float()
    return K, V  # [S,H,D]


DATASETS = {
    "SF-L15": lambda: load_sf(15),
    "SF-L29": lambda: load_sf(29),
    "LC-L24": kv3d.load_lc,
    "HY-L13": kv3d.load_hy,
}


def head_pca(x):  # x [N, D] -> eigvals desc, eigvecs (cols, desc), mean
    mu = x.mean(0)
    xc = x - mu
    cov = (xc.T @ xc) / x.shape[0]
    w, v = torch.linalg.eigh(cov)
    idx = torch.argsort(w, descending=True)
    return w[idx].clamp_min(0), v[:, idx], mu


def cum_energy(w):
    c = torch.cumsum(w, 0)
    return (c / c[-1]).numpy()


stats = {}
fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
for j, (name, loader) in enumerate(DATASETS.items()):
    K, V = loader()
    for i, (tname, x) in enumerate((("K", K), ("V", V))):
        S, H, D = x.shape
        curves, r80s, r95s = [], [], []
        for h in range(H):
            w, _, _ = head_pca(x[:, h, :])
            ce = cum_energy(w)
            curves.append(ce)
            r80s.append(int(np.searchsorted(ce, 0.80) + 1))
            r95s.append(int(np.searchsorted(ce, 0.95) + 1))
        curves = np.stack(curves)             # [H, D]
        med = np.median(curves, 0)
        ax = axes[i, j]
        rs = np.arange(1, D + 1)
        ax.fill_between(rs, curves.min(0), curves.max(0), alpha=0.25, color="steelblue")
        ax.plot(rs, med, color="navy", lw=1.4, label="median head")
        for lvl in (0.8, 0.95):
            ax.axhline(lvl, color="gray", lw=0.6, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("rank r (log)"); ax.set_ylabel("cumulative energy")
        ax.set_title(f"{tname} — {name}\nr80={int(np.median(r80s))} (worst {max(r80s)}), "
                     f"r95={int(np.median(r95s))} (worst {max(r95s)})", fontsize=9)
        ax.grid(alpha=0.3)
        stats[(name, tname)] = dict(r80=int(np.median(r80s)), r80w=max(r80s),
                                    r95=int(np.median(r95s)), r95w=max(r95s))
        print(f"{name} {tname}: r80 med/worst = {int(np.median(r80s))}/{max(r80s)}, "
              f"r95 = {int(np.median(r95s))}/{max(r95s)}")
axes[0, 0].legend(fontsize=8)
fig.suptitle("Per-head centered PCA of KV: cumulative energy vs rank (shaded = min..max over heads)", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "pca_spectrum.png"), dpi=140); plt.close(fig)

# ---------------- H9 check: are PC1/2 the ch95/ch49 directions? ----------------
K29, _ = load_sf(29)
w, v, mu = head_pca(K29[:, 9, :])
tot = float(w.sum())
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i in range(2):
    pc = v[:, i].abs().numpy()
    axes[i].bar(range(128), pc, width=1.0, color="steelblue")
    for ch in (95, 49, 0, 3):
        axes[i].annotate(f"ch{ch}", xy=(ch, pc[ch]), xytext=(ch, pc[ch] + 0.03),
                         ha="center", fontsize=8, color="red")
    axes[i].set_title(f"|PC{i+1}| components — energy {float(w[i])/tot*100:.1f}%", fontsize=10)
    axes[i].set_xlabel("channel within head 9")
    axes[i].grid(alpha=0.3, axis="y")
    print(f"H9 PC{i+1}: energy {float(w[i])/tot*100:.1f}%, |comp| ch95={pc[95]:.3f} ch49={pc[49]:.3f} "
          f"ch0={pc[0]:.3f} ch3={pc[3]:.3f}, max other={np.delete(pc, [95,49,0,3]).max():.3f}")
print(f"H9 top-8 PCs energy: {float(w[:8].sum())/tot*100:.1f}%")
fig.suptitle("SF L29 H9: leading principal components vs the monster channels", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "pca_h9.png"), dpi=140); plt.close(fig)

# ---------------- Residual flattening (SF L15 & L29, K) ----------------
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, lyr in zip(axes, (15, 29)):
    K, _ = load_sf(lyr)
    S, H, D = K.shape
    prof0 = K.permute(1, 2, 0).reshape(H * D, S).pow(2).mean(1).sqrt().numpy()
    ax.plot(prof0, lw=0.5, color="black", alpha=0.8, label="original")
    for r, c in ((8, "tab:blue"), (16, "tab:green"), (32, "tab:orange")):
        res_prof = []
        kur = []
        for h in range(H):
            x = K[:, h, :]
            w, v, mu = head_pca(x)
            xc = x - mu
            proj = xc @ v[:, :r]
            res = xc - proj @ v[:, :r].T
            res_prof.append(res.pow(2).mean(0).sqrt())
            z = res.flatten()
            kur.append(float((z.pow(4).mean() / z.pow(2).mean().pow(2))))
        rp = torch.cat(res_prof).numpy()
        ax.plot(rp, lw=0.5, color=c, alpha=0.8, label=f"residual r={r}")
        contr = float(np.max(rp) / np.median(rp))
        print(f"L{lyr} K residual r={r}: channel contrast {contr:.1f}x "
              f"(original {float(np.max(prof0)/np.median(prof0)):.1f}x), mean kurtosis {np.mean(kur):.1f}")
    ax.set_yscale("log")
    ax.set_title(f"K — L{lyr}: per-channel rms, original vs PCA residual", fontsize=10)
    ax.set_xlabel("channel (12 heads x 128 flattened)"); ax.set_ylabel("rms (log)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.suptitle("PCA residuals flatten the channel walls", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "pca_residual.png"), dpi=140); plt.close(fig)

# ---------------- Basis stationarity (SF L15, K): begin-basis on end window ----------------
Kb, _ = load_sf(15, "begin")
Ke, _ = load_sf(15, "end")
print("basis stationarity (SF L15 K): energy captured on END window")
for r in (8, 16, 32):
    own, cross = [], []
    for h in range(12):
        wb, vb, mub = head_pca(Kb[:, h, :])
        we, ve, mue = head_pca(Ke[:, h, :])
        xe = Ke[:, h, :] - Ke[:, h, :].mean(0)
        tot_e = float(xe.pow(2).sum())
        own.append(float((xe @ ve[:, :r]).pow(2).sum()) / tot_e)
        cross.append(float((xe @ vb[:, :r]).pow(2).sum()) / tot_e)
    print(f"  r={r}: own-basis {np.mean(own)*100:.1f}%  begin-basis {np.mean(cross)*100:.1f}%  "
          f"(retention {np.mean(cross)/np.mean(own)*100:.1f}%)")
print("done")
