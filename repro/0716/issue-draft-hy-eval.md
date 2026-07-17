# GitHub Issue Draft: HY-WorldPlay Table 1 evaluation protocol（短版）

> 用途：问作者 HY 三指标怎么测的。贴到上游 repo issues。

---

**Title:** Cannot reproduce HY-WorldPlay PSNR/SSIM/LPIPS in Table 1 — how were they measured?

Hi, thanks for releasing the code. The LongCat-Video numbers in Table 1 reproduce well on our side (QVG INT2: 28.9 vs reported 28.716, first-generated-chunk protocol). However, we cannot reproduce the **HY-WorldPlay** rows (QVG INT2: 29.174 / 0.882 / 0.094), and §5.1 doesn't specify the similarity protocol for HY.

What we tried (all with `run_qvg.sh` as released; BF16 reference regenerated with the same config, since `run_bf16.sh` uses a different one; metrics per `longcat_video/utils/metric.py`):

- **Full-video mean** (189 frames): 18.8 / 0.46 / 0.37 — far below Table 1. The quantized run's content diverges from the BF16 reference abruptly around the pose reversal (~frame 29).
- **Pre-divergence frames** (1–28): 35.1 / 0.966 / 0.054 — far above Table 1.
- **Early windows straddling the divergence point** can bracket the reported numbers (e.g., frames [20,32): 31.1 / 0.882 / 0.099), but that seems protocol-fragile.

Could you share how the HY PSNR/SSIM/LPIPS in Table 1 were computed — over which frame range, with which generation config (chunk size 12 frames per §5.2 vs 16 in the released scripts; pose; `num_chunk`; `memory_frames`), and whether your BF16/quantized runs stayed content-aligned over the full video?

Thanks!
