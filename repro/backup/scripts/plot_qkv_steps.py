"""Timestep-axis dynamics of Q/K/V (DeltaQuant-Fig.4 analog, on SF).

Data: sf_qkv_steps.pt — one block (start frame 87), ALL denoise steps kept.
Figure: rows = layer {15, 29}, cols = Q/K/V; each panel overlays the per-channel
rms profile (over tokens, all heads flattened to 1536 channels) of every denoise
step. If outlier channels move/rescale across steps, the lines diverge.
Also prints step-to-step channel-profile correlations and norm medians.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = "/home/zhizhousha/workspace/video-project/Quant-VideoGen"
d = torch.load(os.path.join(REPO, "results/kvplot/sf_qkv_steps.pt"),
               map_location="cpu", weights_only=False)
slots = d["slots"]

LAYERS = [15, 29]
FRAME = 87
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
print("layer,tensor,step,med_token_norm,absmax")
for i, lyr in enumerate(LAYERS):
    recs = slots[(lyr, FRAME)]
    n_steps = len(recs)
    cmap = plt.cm.viridis(np.linspace(0.05, 0.9, n_steps))
    for j, t in enumerate(("q", "k", "v")):
        ax = axes[i, j]
        profs = []
        for s_idx, rec in enumerate(recs):
            x = rec[t][0].float()                    # [S, H, D]
            prof = x.permute(1, 2, 0).reshape(-1, x.shape[0]).pow(2).mean(1).sqrt()  # [H*D]
            profs.append(prof)
            ax.plot(prof.numpy(), lw=0.5, color=cmap[s_idx], alpha=0.85,
                    label=f"step {s_idx}")
            med = float(x.norm(dim=-1).median()); mx = float(x.abs().max())
            print(f"L{lyr},{t.upper()},{s_idx},{med:.2f},{mx:.1f}")
        # step0 vs last-step channel-profile correlation
        c = float(np.corrcoef(profs[0].numpy(), profs[-1].numpy())[0, 1])
        ax.set_title(f"{t.upper()} — L{lyr}   corr(step0, step{n_steps-1}) = {c:.3f}", fontsize=10)
        ax.set_xlabel("channel (12 heads x 128 flattened)")
        ax.set_ylabel("per-channel rms over tokens")
        if i == 0 and j == 0:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
fig.suptitle("SF Q/K/V per-channel profiles at EVERY denoise step (one block, frame 87-89)\n"
             "diverging lines = distribution shifts across the denoising-timestep axis",
             fontsize=12)
fig.tight_layout()
out = os.path.join(REPO, "repro/0714/figs/qkv_timestep_dynamics.png")
fig.savefig(out, dpi=140)
print("saved", out)
