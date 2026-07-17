# GitHub Issue Draft: HY-WorldPlay Table 1 evaluation protocol（短版，可直接发）

> 附注（不进 issue）：12 帧/chunk 实测结论 = 不可检验，发布代码 pred_latent_size 只能为 4 的倍数
> （generate.py:205 断言 memory=context+pred；utils.py:132 记忆帧按 4 帧块分配 → utils.py:143 数量
> 断言恒炸）。paper §5.2 的 "chunk sizes of 12 and 16" 与代码 HY=16/SF=12 正好对调，判为笔误。

---

**Title:** Cannot reproduce HY-WorldPlay PSNR/SSIM/LPIPS in Table 1 — how were they measured?

Hi, thanks for releasing the code. The LongCat-Video numbers in Table 1 reproduce well on our side (QVG INT2: 28.9 vs reported 28.716, first-generated-chunk protocol). However, we cannot reproduce the **HY-WorldPlay** rows (QVG INT2: 29.174 / 0.882 / 0.094), and §5.1 doesn't specify the similarity protocol for HY.

What we tried (QVG via `run_qvg.sh` as released; BF16 reference regenerated with the same generation config, since the released `run_bf16.sh` uses a different one; metrics per `longcat_video/utils/metric.py`):

- **Full-video mean** (189 frames): 18.8 / 0.46 / 0.37 — far below Table 1. The quantized run's content diverges from the BF16 reference abruptly around the pose reversal (~frame 29).
- **Pre-divergence frames** (1–28): 35.1 / 0.966 / 0.054 — far above Table 1.
- **Early windows straddling the divergence point** can bracket the reported numbers (e.g., frames [20,32): 31.1 / 0.882 / 0.099), but that seems protocol-fragile.

Could you share how the HY PSNR/SSIM/LPIPS in Table 1 were computed — over which frame range, with which generation config (pose, `num_chunk`, `memory_frames`; we assumed the released `run_qvg.sh` values), and did your BF16/quantized runs stay content-aligned over the full video?

(Minor: §5.2 says chunk sizes of "12 and 16 frames" for HY-WorldPlay and Self-Forcing respectively, while the released configs are HY = 16 frames/chunk (`pred_latent_size=4`) and SF = 12 (`num_frame_per_block=3`) — the two numbers appear swapped. We also note 12-frame chunks cannot run in the released HY code: memory-frame selection allocates in 4-frame blocks, so `pred_latent_size=3` always fails the count assertion in `select_mem_frames_wan`.)

Thanks!
