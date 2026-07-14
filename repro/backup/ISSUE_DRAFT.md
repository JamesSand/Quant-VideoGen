# Draft GitHub issue for svg-project/Quant-VideoGen

**Title:** Cannot reproduce Table 1 PSNR with released scripts — memory/compression and rel-L2 reproduce exactly; video PSNR does not. What is the exact similarity-evaluation protocol?

---

Hi, thanks for releasing the code. We set up the repo on 8× H100 80GB (torch 2.8.0+cu128, triton 3.4, flash-attn 2.8.3 cxx11abiTRUE) and can reproduce the memory side of the paper exactly:

- LongCat-Video INT2: 67.32 MB/layer, 3231.28 MB total (matches README to the digit), compression 6.89× (paper 6.94×)
- LongCat-Video INT4: 125.43 MB/layer → 3.70× (paper 3.72×)
- HY-WorldPlay INT2: 141.18 MB/layer (matches README)
- Tensor-level rel-L2 on real KV (1-stage, B=64, K=256): K≈0.27 / V≈0.45 for INT2 — consistent with Figure 7(b–c)
- Triton kernels verified numerically equivalent to a pure-PyTorch reference of the same algorithm

However, we cannot reproduce the Table 1 similarity numbers (PSNR/SSIM/LPIPS vs the BF16 KV-cache baseline). Using the shipped scripts (`scripts/LongCat/{base,run_bf16,run_qvg}.sh`, prompt_idx=1, seed=0) and `experiments/LongCat/longcat_video/utils/metric.py` with `--skip_frames 93` (the shared init prefix), we get:

| Setting | Ours | Table 1 |
|---|---:|---:|
| LongCat INT2 (segment_10, frames 94–293) | 12.7 dB | 28.716 |
| LongCat INT4 | 12.5 dB | 37.141 |
| LongCat INT2, first segment only (20 frames) | 17.9 dB | — |
| HY-WorldPlay INT2 (aligned 12-chunk geometry) | 18.7 dB | 29.174 |
| HY-WorldPlay INT4 | 22.5 dB | 34.454 |

Things we ruled out: kernel numerics (triton == simulation path), k-means RNG desync between the two runs (isolated it via a wrapper; results unchanged), evaluation window (no single window reproduces INT2 and INT4 jointly), the `480p_long_gen_fullkv` workload (per-segment PSNR decays 19.9→13.5→14.6 for INT2; also the BF16 baseline OOMs at segment 4 on 80GB), HY-WorldPlay's two script geometries (48/44 vs 56/52 give the same result), and QVG-Pro config (S=4, B=16 → 22.1 dB).

Two runs of the same INT2 config that differ only in the k-means random init already differ from each other by 22.4 dB PSNR, which upper-bounds any quant-vs-BF16 comparison from this pipeline well below 28.7.

Could you clarify:
1. The exact protocol behind Table 1's PSNR/SSIM/LPIPS: which frames are compared (the paper mentions "the number of the first generated chunk" for LongCat — what is a chunk here, and what frame range/`--skip_frames` was used)?
2. The generation configuration used for those runs (workload, number of inference steps, quantization cadence, whether k-means used centroid caching — `init_centroids` exists in `batch_kmeans_Euclid` but is never passed by any caller in the release)?
3. Whether the evaluation scripts used for the paper can be released?

Happy to share full logs and our reproduction harness. Thanks!
