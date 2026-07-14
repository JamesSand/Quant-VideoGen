"""Direct (no-kmeans) visualization: why pre-RoPE K is structured and what RoPE
does to that structure. Panels:
  (a)/(b) signed-value heatmaps (token x channel), pre vs post RoPE
  (c) one LOW-frequency temporal channel across frames at a fixed spatial site
  (d) one HIGH-frequency temporal channel, same site
  (e) per-pair std ratio vs theta*T (from ropestudy_data.npz)
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/zhizhousha/workspace/video-project/Quant-VideoGen"
sys.path.insert(0, os.path.join(REPO, "experiments/Self-Forcing"))
from wan.modules.model import rope_params
from wan.modules.causal_model import causal_rope_apply

LAYER, H_GRID, W_GRID, D, HEAD = 15, 30, 52, 128, 8
FSL = H_GRID * W_GRID

d = D
freqs = torch.cat([
    rope_params(1024, d - 4 * (d // 6)),
    rope_params(1024, 2 * (d // 6)),
    rope_params(1024, 2 * (d // 6)),
], dim=1)
gs1 = torch.tensor([[1, H_GRID, W_GRID]])

dump = torch.load(os.path.join(REPO, "results/ropestudy/kv_cache_frames180.pt"),
                  map_location="cpu", mmap=True, weights_only=False)
kcache = dump["kv_cache"][LAYER]["k"]

# ---------- heatmap window: 2 consecutive frames ----------
hm_frames = [40, 41]
pre_hm = torch.cat([kcache.chunks[t].clone() for t in hm_frames], dim=1).float()
post_hm = torch.cat([causal_rope_apply(pre_hm[:, i*FSL:(i+1)*FSL], gs1, freqs,
                                       start_frame=t).float()
                     for i, t in enumerate(hm_frames)], dim=1)
pre_hm_h = pre_hm[0, :, HEAD, :].numpy()     # [3120, 128]
post_hm_h = post_hm[0, :, HEAD, :].numpy()

# ---------- temporal traces: fixed spatial site across frames ----------
site = 15 * W_GRID + 26
trace_frames = list(range(0, 180, 4))
pre_tr, post_tr = [], []
for t in trace_frames:
    xf = kcache.chunks[t].clone().float()
    pre_tr.append(xf[0, site, HEAD, :].numpy().copy())
    post_tr.append(causal_rope_apply(xf, gs1, freqs, start_frame=t)[0, site, HEAD, :].float().numpy().copy())
pre_tr, post_tr = np.array(pre_tr), np.array(post_tr)   # [45, 128]

# ---------- pick channels via theta*T (temporal pairs are pair idx 0..21) ----------
z = np.load(os.path.join(REPO, "repro/0714/ropestudy_data.npz"))
theta_T, axis = z["theta_T"], z["axis"]
temporal = np.where(axis == 0)[0]
low_pair = temporal[np.argmin(theta_T[temporal])]
high_cands = temporal[theta_T[temporal] >= np.pi]
high_pair = high_cands[0] if len(high_cands) else temporal[np.argmax(theta_T[temporal])]
low_ch, high_ch = 2 * int(low_pair), 2 * int(high_pair)

fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])

for j, (m, ttl) in enumerate([(pre_hm_h, "(a) pre-RoPE K: vertical stripes = per-channel structure"),
                              (post_hm_h, "(b) post-RoPE K: high-freq channels shredded, low-freq stripes survive")]):
    ax = fig.add_subplot(gs[0, j])
    v = np.percentile(np.abs(m), 99)
    ax.imshow(m[::6], aspect="auto", cmap="coolwarm", vmin=-v, vmax=v)
    ax.set_title(ttl, fontsize=10)
    ax.set_xlabel("Channel"); ax.set_ylabel("Token (2 frames)")

ax = fig.add_subplot(gs[0, 2])
colors = np.where(z["disp_pair_mask"], "tab:red", "tab:blue")
ax.scatter(theta_T, z["std_ratio_per_pair"], c=colors, s=28)
ax.axvline(np.pi, ls="--", c="gray"); ax.set_xscale("log")
ax.text(np.pi*1.1, 0.6, "theta*T = pi", fontsize=9, color="gray")
ax.set_xlabel("theta * T  (rotation swept over window)"); ax.set_ylabel("std ratio (post/pre)")
ax.set_title("(e) dispersion is frequency-localized\nred = predicted dispersed pairs", fontsize=10)

for j, (ch, name) in enumerate([(low_ch, f"(c) LOW-freq temporal channel {low_ch} (theta*T={theta_T[low_pair]:.3f})"),
                                (high_ch, f"(d) HIGH-freq temporal channel {high_ch} (theta*T={theta_T[high_pair]:.1f})")]):
    ax = fig.add_subplot(gs[1, j])
    ax.plot(trace_frames, pre_tr[:, ch], "o-", ms=3, label="pre-RoPE", color="tab:blue")
    ax.plot(trace_frames, post_tr[:, ch], "o-", ms=3, label="post-RoPE", color="tab:red", alpha=0.8)
    ax.set_xlabel("Latent frame index (fixed spatial site)"); ax.set_ylabel("channel value")
    ax.set_title(name + "\nsame content, different position ->" +
                 (" rotation ~ identity" if j == 0 else " value sweeps a circle"), fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 2])
std_pre = pre_tr.std(axis=0); std_post = post_tr.std(axis=0)
ax.plot(std_pre, label="pre-RoPE", color="tab:blue")
ax.plot(std_post, label="post-RoPE", color="tab:red", alpha=0.8)
ax.set_xlabel("Channel"); ax.set_ylabel("std across frames (fixed site)")
ax.set_title("(f) temporal std per channel: RoPE injects variance\nonly where theta*T is large", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle("Self-Forcing K, Layer 15 Head 8 — what RoPE actually does to the distribution (no k-means anywhere)",
             fontsize=12)
fig.tight_layout()
out = os.path.join(REPO, "repro/0714/figs/rope_before_after.png")
fig.savefig(out, dpi=140)
print("saved", out, "| low_ch", low_ch, "high_ch", high_ch)
