# PCA-Grid Hash 三指标验证终局

## 总结

**最终判定：FAIL。** PCA-Grid Hash 通过实际 BPE、8-chunk reconstruction
MSE 和 encode+decode 总延迟，但未通过 paired 端到端 canary。按预注册规则，
没有为该候选启动 MP100，也没有接入 `quant_videogen/compress.py` 主路径。

## PCA-Grid Hash

### G1 + G2：PASS

| model | all-in BPE (hash≤QVG) | K MSE ratio | V MSE ratio | worst chunk K/V |
|---|---:|---:|---:|---:|
| LC | 2.3864≤2.4639 | 0.491 | 0.522 | 0.703/0.955 |
| SF | 2.3364≤2.4063 | 0.573 | 0.583 | 0.966/0.794 |
| HY | 2.5588≤3.3199 | 0.507 | 0.471 | 0.544/0.500 |

BPE 来自实际 state tensor 的逐字节计数，包含 labels、table、INT2 codes、
FP8 scale/zero、padding 和 int8 normalization exponent。

### G3：PASS

同一 H100、同一 8 个 paired chunks、CUDA event median，延迟为一次 K+V
encode 加一次 K+V decode：

| model | QVG total mean | hash total mean | geomean speedup | bootstrap 95% lower |
|---|---:|---:|---:|---:|
| LC | 172.2800 ms | 23.4893 ms | 7.326× | 7.092× |
| SF | 3.3762 ms | 3.0484 ms | 1.108× | 1.105× |
| HY | 3.3244 ms | 2.5956 ms | 1.281× | 1.271× |

所有 24 个 paired chunks 均严格快于 QVG。

### G4：FAIL

LC canary 通过，但 SF/HY 出现越过预注册 margin 的端到端退化：

| model | failed metric | candidate−QVG | allowed |
|---|---|---:|---:|
| SF | Image Quality | -2.84 | ≥-0.30 |
| HY | LPIPS | +0.0085 | ≤+0.003 |
| HY | BG Consistency | -0.30 | ≥-0.20 |
| HY | Subject Consistency | -0.70 | ≥-0.30 |
| HY | Aesthetic | -3.66 | ≥-0.30 |
| HY | Image Quality | -1.08 | ≥-0.30 |

这也给出一个明确反例：chunk MSE 大幅优于 QVG，不能推出视频质量不劣。

## 有限 factor-only product-grid 回退

回退同样使用实际 packed payload 和 fused decoder。G1–G3 通过：

| model | BPE (factor≤QVG) | K/V aggregate MSE ratio | speedup (95% lower) |
|---|---:|---:|---:|
| SF | 2.3185≤2.4063 | 0.669/0.762 | 1.067× (1.063×) |
| HY | 2.2313≤3.3199 | 0.582/0.616 | 1.426× (1.420×) |

但 paired canary 仍失败：SF IQ -2.14；HY PSNR -0.188 dB、LPIPS
+0.0066、AQ -2.07、IQ -1.53。因 canary 失败，没有启动新的 MP100 generation。

作为额外复核，既有 factor-grid Budget-PCA 完整数据（LC/SF 各 100 prompts，
HY 10 seeds）按预注册 90% paired TOST 重算。LC 与 SF 全通过，HY primary
PSNR/SSIM/LPIPS 显著优于 QVG，但 HY guardrail 仍失败：

| metric | oriented mean | 90% CI | margin | result |
|---|---:|---:|---:|---|
| BC | +0.211 | [-0.195,+0.617] | ±0.20 | FAIL |
| IQ | -0.315 | [-0.619,-0.011] | ±0.30 | FAIL |
| AQ | -0.484 | [-1.152,+0.184] | ±0.30 | FAIL |

## 数学结论

可以证明且实现对应的结论：

1. 固定 analytic labels 与实际 decoded residual payload，group mean 是实值
   table 子问题的全局最优解；
2. 再固定 FP8 factor，nearest FP8 code 是离散 table 子问题的全局最优解；
3. 固定 table 与四电平 grid，round/clamp 是 residual code 子问题的全局最优解；
4. 实现按 BF16 decoded SSE 做逐 head accept/reject，保证 refit 不增加实际输出
   SSE。

不能宣称 PCA、labels、factor、grid、rank 或端到端视频质量的联合全局最优。
显式数值反例也证明：PCA 最小化连续 residual energy，不保证 post-INT2 MSE
全局最优。

## 复现入口

- 判据：`repro/0721/int2-gates.md`
- G1/G2：`repro/0721/grid_hash_final_screen.py`
- G3：`repro/0721/grid_hash_bench.py`
- G4 canary：`repro/0721/hash_canary_score.py`
- factor G1–G3：`repro/0721/factor_grid_validate.py`
- factor canary：`repro/0721/factor_canary_score.py`
- MP100 TOST：`repro/0721/factor_mp100_tost.py`
- 定理与边界：`repro/0721/pca-grid-hash-proof.md`
