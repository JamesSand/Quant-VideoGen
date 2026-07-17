# GitHub Issue Draft: HY-WorldPlay Table 1 evaluation protocol（短版，可直接发）

> 附注（不进 issue）：核心问法按用户 0717 定稿——①全程评测 PSNR 差很多（摆数字）；
> ②直接问是不是只测了前几个 chunk；③带上 12/16 笔误确认。12 帧/chunk 实测结论 =
> 发布代码结构性跑不通（generate.py:205 断言 memory=context+pred；utils.py 记忆帧按
> 4 帧块分配 → 数量断言恒炸），佐证笔误判断。

---

**Title:** HY-WorldPlay Table 1: full-video PSNR is far lower — were only the first few chunks evaluated?

Hi, thanks for releasing the code. The LongCat-Video numbers in Table 1 reproduce well on our side (QVG INT2: 28.9 vs reported 28.716, first-generated-chunk protocol). For **HY-WorldPlay**, however, **full-video evaluation gives much lower numbers than Table 1**, and §5.1 doesn't specify the frame range used for HY.

Setup: QVG via `run_qvg.sh` as released (189 frames); BF16 reference regenerated with the same generation config (the released `run_bf16.sh` uses a different config/length, so the two scripts' outputs can't be compared directly); metrics per `longcat_video/utils/metric.py`.

- **Full-video mean: 18.8 / 0.46 / 0.37** vs Table 1's 29.174 / 0.882 / 0.094 — about **10 dB lower**. The quantized run's content diverges from the BF16 reference around the first pose reversal (~frame 29) and never re-aligns, so every later frame sits at a ~15–19 dB floor.
- The **first chunks before divergence** (frames 1–28) give 35.1 / 0.966 / 0.054, and early windows around the divergence point bracket the reported numbers (e.g., frames [20,32): 31.1 / 0.882 / 0.099).

**Question:** were the HY PSNR/SSIM/LPIPS in Table 1 computed **only over the first few chunks** (before content divergence) rather than the full video? If so, over which frame range exactly — and if they really are full-video means, could you share the generation config (pose, `num_chunk`, `memory_frames`) under which the BF16/quantized runs stay content-aligned for the whole video?

**Also:** §5.2 says chunk sizes of "12 and 16 frames" for HY-WorldPlay and Self-Forcing respectively, while the released configs are HY = 16 frames/chunk (`pred_latent_size=4`) and SF = 12 (`num_frame_per_block=3`) — are the two numbers swapped (typo)? We note 12-frame chunks cannot run in the released HY code: memory-frame selection allocates in 4-frame blocks, so `pred_latent_size=3` always fails the count assertion in `select_mem_frames_wan`.

Thanks!
