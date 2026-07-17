# GitHub Issue Draft: HY-WorldPlay Table 1 evaluation protocol

> 用途：向 QVG 作者询问 HY 三指标的确切测法。贴到上游 repo issues。
> 发之前可把「12-frame chunk 实验结果」一节按实测补上（跑完后更新）。

---

**Title:** Question: exact evaluation protocol for the HY-WorldPlay PSNR/SSIM/LPIPS numbers in Table 1

Hi, thanks for releasing the code — the LongCat-Video results in Table 1 reproduce nicely on our side under the "first generated chunk" protocol described in §5.1 (e.g., we get 28.9±0.2 PSNR for QVG INT2 vs the reported 28.716).

We are having trouble reproducing the **HY-WorldPlay** rows, and §5.1 only specifies the similarity protocol for LongCat ("All other metrics are reported under the long-generation setting"). Could you clarify how the HY numbers (e.g., QVG INT2: 29.174 / 0.882 / 0.094) were computed?

**What we did**

- QVG run: `scripts/HY-WorldPlay/run_qvg.sh` as released (`triton-nstages-kmeans-int2`, K=V=256 centroids, block 64, iters 2, nstages 1; `num_chunk 12`, `memory_frames 48`, `temporal_context_size 44`, `pred_latent_size 4`, the bundled prompt/image/pose).
- BF16 reference: regenerated with the **same** generation config as the QVG run (note the released `run_bf16.sh` uses a different config — 56/52 memory, 14 chunks, an extra `left-8` pose segment — so the two scripts' outputs have different lengths and cannot be compared directly).
- Metrics: per-frame PSNR/SSIM/LPIPS following `experiments/LongCat/longcat_video/utils/metric.py` (SSIM via 11×11 avg-pool windows; LPIPS(vgg) on [0,1] inputs).

**What we observe (INT2, 189 frames)**

The per-frame error has a two-phase structure:

| Frames | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| 1–28 (before divergence) | ~35.1 | ~0.966 | ~0.054 |
| 29 onwards (content diverges) | ~15.8 | ~0.37 | ~0.43 |
| **Full-video mean** | **18.8** | **0.46** | **0.37** |

The trajectory bifurcates abruptly around the pose reversal (forward→backward), after which the quantized run's content no longer matches the BF16 reference — consistent with the "content starts to diverge" behavior §5.1 describes for LongCat. As a result, no whole-video average we can produce comes close to 29.174/0.882/0.094, while early-window averages bracket it (e.g., frames [20,32): 31.1 / 0.882 / 0.099).

**Questions**

1. For the HY rows in Table 1, are PSNR/SSIM/LPIPS averaged over the **entire** generated video? Over which frame range / how many frames (the paper's Fig. 1 shows 285 frames)?
2. Did the BF16 and quantized runs stay content-aligned for the whole video in your experiments (no trajectory bifurcation)? If so, was there anything different from the released config that prevents divergence?
3. §5.2 setup says HY uses a chunk size of **12 frames**, but the released scripts use `pred_latent_size 4` (= 16 frames per chunk) — and `pred_latent_size 3` cannot run in the released code: `generate.py` asserts `memory_frames == temporal_context_size + pred_latent_size`, while `select_mem_frames_wan` allocates memory frames in blocks of 4, so its final count assertion always fails when the memory quota is 3. Which chunk size was used for Table 1, and with what memory-selection settings?
4. Which pose script / `num_chunk` / `memory_frames` / `temporal_context_size` were used for Table 1, and which config was used for the BF16 reference (given `run_bf16.sh` and `run_qvg.sh` currently differ)?
5. Can you confirm the metric implementation for Table 1 is `longcat_video/utils/metric.py` (in particular the SSIM window and the LPIPS input range)?

Happy to share our per-frame arrays / videos if useful. Thanks!

---

> 12-frame chunk 复验结论（0717）：**不可检验**。`pred_latent_size=3` 在发布代码里结构性
> 跑不通（generate.py 的 memory=context+pred 断言 vs select_mem_frames_wan 的 4 帧块
> 分配恒冲突）——该发现已并入问题 ③。issue 可以直接发。
