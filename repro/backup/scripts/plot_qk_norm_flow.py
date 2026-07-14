"""QK-Norm execution flow diagram (Wan whole-dim vs Qwen3 per-head)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = "/home/zhizhousha/workspace/video-project/Quant-VideoGen/repro/0714/figs/qk_norm_flow.png"


def box(ax, x, y, w, h, text, fc="#deebf7", ec="#3182bd", fs=8.5, style="round,pad=0.02"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x0, y0, x1, y1, text=None, fs=7.5):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
                                 color="#555555", lw=1.2))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.045, text, ha="center", fontsize=fs, color="#555555")


fig, axes = plt.subplots(2, 1, figsize=(13, 7.2))
for ax, (title, steps, note) in zip(axes, [
    ("Wan / Self-Forcing:  whole-dim QK-Norm  (norm BEFORE head split)",
     [("token hidden x\n[1536]", "#f0f0f0", "#888888"),
      ("Linear W_q\n(learned)", "#deebf7", "#3182bd"),
      ("raw q\n[1536]", "#f0f0f0", "#888888"),
      ("rms(q) over ALL 1536 dims\nPER TOKEN, computed at runtime", "#fee0d2", "#de2d26"),
      ("divide:  q / rms\n(now unit scale)", "#fee0d2", "#de2d26"),
      ("multiply gain  g [1536]\nLEARNED, one per layer,\nSHARED by all tokens", "#e5f5e0", "#31a354"),
      ("reshape\n12 heads x 128", "#f0f0f0", "#888888"),
      ("RoPE ->\nattention", "#deebf7", "#3182bd")],
     "pins each token's TOTAL norm; heads may still differ (=> whole-head outliers like L29 H9 survive)"),
    ("Qwen3:  per-head QK-Norm  (norm AFTER head split)",
     [("token hidden x\n[4096]", "#f0f0f0", "#888888"),
      ("Linear W_q\n(learned)", "#deebf7", "#3182bd"),
      ("reshape\nH heads x 128", "#f0f0f0", "#888888"),
      ("rms over 128 dims\nPER TOKEN PER HEAD", "#fee0d2", "#de2d26"),
      ("divide:  q_h / rms\n(each head unit scale)", "#fee0d2", "#de2d26"),
      ("multiply gain  g [128]\nLEARNED, SHARED by all\ntokens AND all heads", "#e5f5e0", "#31a354"),
      ("RoPE ->\nattention", "#deebf7", "#3182bd")],
     "pins EVERY head's norm to ~||g||  (=> whole-head outliers largely impossible)"),
]):
    ax.set_xlim(0, 13); ax.set_ylim(0, 2.2); ax.axis("off")
    ax.set_title(title, fontsize=11, loc="left")
    n = len(steps)
    w, gap = 12.4 / n * 0.82, 12.4 / n * 0.18
    for i, (txt, fc, ec) in enumerate(steps):
        x = 0.3 + i * (w + gap)
        box(ax, x, 0.75, w, 0.95, txt, fc=fc, ec=ec)
        if i:
            arrow(ax, x - gap, 1.22, x, 1.22)
    ax.text(0.3, 0.28, "red = per-token dynamic   green = learned constant   |   " + note,
            fontsize=8.5, color="#333333")

fig.suptitle("QK-Norm: where g comes from and what is per-token  (y = g * x / rms(x); V is never normalized)",
             fontsize=12)
fig.tight_layout()
fig.savefig(FIG, dpi=150)
print("saved", FIG)
