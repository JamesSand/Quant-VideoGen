# B 扫描实验矩阵：QVG / QuaRot / QuaRot+clip × B ∈ {16, 64, 128}

目标：隔离**块大小 B** 单一变量，看三种方法对 B 的敏感度。
固定量：INT2、LongCat 单段续写（seed 0、prompt_idx 1、cond 73 帧）、评测 = frame-93 首帧 PSNR vs `results/longcat/bf16/1-0/segment_1.mp4`。

## 总矩阵（PSNR dB ｜ 括号内 BPE@LC chunk 29,640）

| 方法 \ B | **16** | **64** | **128** |
|---|---:|---:|---:|
| **QVG**（S=1, K=256, iters=100） | ⬜ 待跑（2.70） | **28.88 ± 0.18** (n=3)（2.326） | ⬜ 待跑（2.263） |
| **QuaRot**（非对称，K/V 双旋转） | **30.38 ± 0.003** (n=3)（3.0） | **28.85** (n=1)（2.25） | **24.54 ± 0.002** (n=3)（2.125） |
| **QuaRot + clip**（非对称 + 逐块收缩 r=0.99） | **30.68** (n=1)（3.0） | ⬜ 待跑（2.25） | ⬜ 待跑（2.125） |

已有 5 格，待跑 4 格。BPE 记账：残差 2 + scale 8/B +（QuaRot 非对称再加 zero-point 8/B）+（QVG 再加索引 0.0625 + 质心 4096/29640≈0.138）；clip 不占存储。

## 已有数字的出处

| 格子 | 数值 | 来源 |
|---|---|---|
| QVG B=64 | 28.718 / 29.130 / 28.805 → 28.88±0.18 | 0713 方差研究（`results/varstudy/var_qvg_run{1..3}`） |
| QuaRot B=16 | 30.377 / 30.382 / 30.383 → 30.38±0.003 | 0713 方差研究（asym16） |
| QuaRot B=64 | 28.85 | 0714 补跑（`results/varstudy/var_quarot_asym64_run1`，report-0714 §2.5） |
| QuaRot B=128 | 24.546 / 24.542 / 24.546 → 24.54±0.002 | 0713 方差研究（asym128） |
| QuaRot+clip B=16 | 30.68 | 0713 qclip 扫描 ratio=0.99 臂（`results/qclip/qclip_int2_r0.99`） |

辅助参考（不进主矩阵）：对称 QuaRot B16=21.42±0.000 (n=3)、B64=19.14 (n=1)；QVG-Pro（S=4,B=16）=31.04±0.005——**注意 QVG-Pro 不能填进 QVG B=16 格**，S=4 是混杂变量，本矩阵 QVG 行固定 S=1。

## 待跑 4 格的执行方式（确认后启动）

```bash
# QVG B=16 / B=128（S=1 隔离 B；直跑发布版 run_long_t2v.py）
--quant_type triton-nstages-kmeans-int2 --quant_block_size {16|128} \
--cache_num_k_centroids 256 --cache_num_v_centroids 256 --kmeans_max_iters 100 --num_prq_stages 1

# QuaRot+clip B=64 / B=128（qclip runner 加 B 参数）
QUAROT_BLOCK={64|128} QUAROT_SYM=0 QUAROT_CLIP_RATIO=0.99 QUAROT_CLIP_PCT=100 \
  quarot_launcher.py ... --quant_type naive-int2 --quant_block_size {64|128}
```

4 条 1-GPU pod，单段续写每条 ~10 分钟，全部并行 <20 分钟出数。

## 讨论点（定了再跑）

1. **clip 固定点**：默认沿用 0713 的最优 ratio=0.99。但 0.99 是在 B=16 下扫出来的——收缩比的最优点可能随 B 移动（块越大，块内 min/max 越受离群支配，理论上更粗的块可能偏好更强的收缩）。方案 A：先只跑 r=0.99 两格补齐矩阵；方案 B：B=64/128 各带一条小 ratio 扫描（0.95/0.975/0.99，共 6 条 pod）。建议 A 先看形状。
2. **QVG 的随机性**：QVG 是非确定性的（k-means 无种子），新跑的 B=16/128 两格建议直接 n=3（+4 条 pod），否则和 B=64 的 mean±std 不同权。QuaRot 类确定性方法 n=1 足够（σ≤0.003 已验证）。
3. **预期形状**（可证伪）：QuaRot 对 B 单调恶化且陡（30.38→28.85→24.54，离群主导 scale）；QVG 应当平缓得多（残差已被质心削平，B=64→16 的收益可能 <1 dB）；clip 的相对增益应随 B 增大（clip 正是治"块内离群拉爆 scale"的药）——如果成立，B=128 的 clip 格会是全矩阵最有信息量的点。
