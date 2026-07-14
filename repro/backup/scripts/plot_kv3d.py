"""OScaR-Figure-2-style 3D magnitude surfaces of raw K and V for the QVG models.

Usage:
  plot_kv3d.py sf  <out.png>                      # from results/ropestudy/kv_cache_frames180.pt
  plot_kv3d.py kvplot <lc|hy> <k_idx> <v_idx> <out.png> [title]
       # from results/kvplot/<arm>_kv.pt (kvplot_launcher capture); k_idx/v_idx
       # are instance indices in the 'selected' dict (disambiguate via 'meta').

Surfaces: X=channel (head_dim 128), Y=token index (subsampled window), Z=|value|.
Head is auto-picked as the K head with the strongest channel-outlier contrast
(max channel median / overall median); same head used for V.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

TOK_CAP = 12000   # analysis window
ROWS = 600        # surface rows after subsampling


def pick_head(K):  # K: [S,H,D] float32
    med = K.abs().median(dim=0).values            # [H,D] per-channel median
    contrast = med.max(dim=1).values / med.median(dim=1).values.clamp_min(1e-6)
    h = int(contrast.argmax())
    return h, contrast


def channel_stats(x):  # x: [S,D] one head
    med = x.abs().median(dim=0).values
    return float(med.max() / med.median().clamp_min(1e-6))


def surface(ax, x, title, zlabel=True):  # x: [S,D] float32
    S, D = x.shape
    stride = max(1, S // ROWS)
    z = x[::stride].abs().numpy()
    Y, X = np.mgrid[0:z.shape[0], 0:D]
    ax.plot_surface(X, Y * stride, z, cmap="coolwarm", rstride=2, cstride=2,
                    linewidth=0, antialiased=False)
    ax.set_xlabel("Channel", labelpad=8)
    ax.set_ylabel("Token index", labelpad=10)
    if zlabel:
        ax.set_zlabel("|value|", labelpad=6)
    ax.set_title(title, fontsize=11)
    ax.view_init(elev=32, azim=-58)


def plot_pair(K, V, model_title, layer_note, out):
    # K,V: [S,H,D] float32 on CPU
    h, _ = pick_head(K)
    kh, vh = K[:, h], V[:, h]
    fig = plt.figure(figsize=(13, 5.2))
    axk = fig.add_subplot(1, 2, 1, projection="3d")
    axv = fig.add_subplot(1, 2, 2, projection="3d")
    surface(axk, kh, f"(a) Key (pre-RoPE) — {layer_note} Head {h}\nchannel contrast {channel_stats(kh):.1f}x")
    surface(axv, vh, f"(b) Value — {layer_note} Head {h}\nchannel contrast {channel_stats(vh):.1f}x")
    fig.suptitle(model_title, fontsize=13, y=0.99)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}  (head={h}, K contrast {channel_stats(kh):.1f}x, V contrast {channel_stats(vh):.1f}x)")


def load_sf():
    d = torch.load("results/ropestudy/kv_cache_frames180.pt", weights_only=False, mmap=True)
    lay = d["kv_cache"][15]
    K = torch.cat([c for c in lay["k"].chunks[:8]], dim=1)[0].float().cpu()  # [S,H,D]
    V = torch.cat([c for c in lay["v"].chunks[:8]], dim=1)[0].float().cpu()
    return K[:TOK_CAP], V[:TOK_CAP]


def load_kvplot(arm, k_idx, v_idx):
    d = torch.load(f"results/kvplot/{arm}_kv.pt", weights_only=False, mmap=True)
    K = d["selected"][k_idx][0].float().cpu()[:TOK_CAP]
    V = d["selected"][v_idx][0].float().cpu()[:TOK_CAP]
    return K, V


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "sf":
        K, V = load_sf()
        plot_pair(K, V, "Self-Forcing (Wan 1.3B) — raw KV cache", "Layer 15/30,", sys.argv[2])
    elif mode == "kvplot":
        arm, k_idx, v_idx, out = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
        title = sys.argv[6] if len(sys.argv) > 6 else arm
        plot_pair(*load_kvplot(arm, k_idx, v_idx), title, f"inst {k_idx}/{v_idx},", out)
    else:
        raise SystemExit("mode must be sf|kvplot")
