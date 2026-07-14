# B 扫描实验矩阵：QVG / QuaRot / QuaRot+clip × B ∈ {16, 64, 128}

目标：隔离**块大小 B** 单一变量，看三种方法对 B 的敏感度。
固定量：INT2、LongCat 单段续写（seed 0、prompt_idx 1、cond 73 帧）、评测 = frame-93 首帧 PSNR vs `results/longcat/bf16/1-0/segment_1.mp4`。

## 总矩阵

**格子格式**：`PSNR ± std (n=独立重复次数)（BPE）`
- `n=3` = 三次独立运行的 mean±std（QVG 非确定性，必须重复）；`n=1` = 单次（QuaRot 类确定性方法 σ≤0.003 已验证，单次即精确值）
- 括号 = 该配置的 BPE（bit/元素，@LC chunk 29,640）——每格的"存储价格"，横向比较 PSNR 时须对照


| 方法 \ B | **16** | **64** | **128** |
|---|---:|---:|---:|
| **QVG**（S=1, K=256, iters=100） | **30.96 ± 0.026** (n=3)（2.70） | **28.88 ± 0.18** (n=3)（2.326） | **28.41 ± 0.043** (n=3)（2.263） |
| **QuaRot**（非对称，K/V 双旋转） | **30.38 ± 0.003** (n=3)（3.0） | **28.85** (n=1)（2.25） | **24.54 ± 0.002** (n=3)（2.125） |
| **QuaRot + clip**（非对称 + 逐块收缩 r=0.99） | **30.68** (n=1)（3.0） | **29.07** (n=1)（2.25） | **25.35** (n=1)（2.125） |

**9 格全部完成**（B=128 的 QVG 需修 `quant_pack` autotune 才能跑，见下）。BPE 记账：残差 2 + scale 8/B +（QuaRot 非对称再加 zero-point 8/B）+（QVG 再加索引 0.0625 + 质心 4096/29640≈0.138）；clip 不占存储。

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

## 结果读数（0714 实测后回填）

原始数据：QVG B16 = 30.925/30.965/30.975；QVG B128 = 28.359/28.443/28.416（`results/varstudy/var_qvg_b{16,128}_run{1..3}`）；clip 两格在 `results/qclip/qclip_int2_r0.99_b{64,128}`。

### 三个预期的裁决

1. **"QuaRot 对 B 陡峭恶化" —— ✅ 成立**：30.38 → 28.85 → 24.54，全程 −5.84 dB，且 64→128 一段就崩 4.31 dB（原始 KV 的离群随块变粗迅速拉爆 min/max scale）。
2. **"QVG 平缓" —— ❌ 大部分被证伪**：B=16→64 掉 2.08 dB，比 QuaRot 同区间（1.53 dB）还陡——细 scale 对 QVG 残差同样值钱。成立的一半在大 B 端：64→128 只掉 0.47 dB（QuaRot 是它的 9 倍），质心削平残差确实让 QVG 在粗块下有韧性。
3. **"clip 增益随 B 增大" —— ✅ 方向成立但非单调**：+0.30 (B16) / +0.22 (B64) / +0.81 (B128)。B=128 处增益最大符合"clip 治块内离群"的机制；B=64 的小凹陷提示 r=0.99（在 B=16 下调出的）不是各 B 的统一最优，如需精确可在 B=128 补一条 ratio 细扫。

### 计划外的重要发现

- **QVG-Pro 的优势几乎全来自 B=16，而非 S=4**：QVG S=1/B=16 = 30.96，QVG-Pro S=4/B=16 = 31.04——四轮渐进 k-means（4 套质心表 + 4 份索引，metadata 翻倍、4 倍聚类成本）只贡献 **+0.08 dB**；而同样 S=1 下 B=64→16 贡献 +2.08 dB。**"QVG-Pro ≈ QVG + 细 scale"**。
- **同预算对比**：QVG B=128（BPE 2.263）vs QuaRot B=64（2.25）——28.41 vs 28.85，等价预算下 QuaRot 反超；但 QVG B=16（2.70）vs QuaRot B=16（3.0）——30.96 vs 30.38，QVG 又赢且更便宜。两条曲线在 BPE ~2.3-2.7 之间交叉，没有全局赢家。
- **方差随 B 的形状**：QVG σ = 0.026 (B16) / 0.18 (B64) / 0.043 (B128)——B=64 的高方差是异常点而非趋势，提示 0713 测到的 σ=0.18 可能高估了 QVG 的固有噪声。

### 工程注记

`quant_block_size=128` 触发发布 kernel 的 bug：`quant_pack` 的 autotune 含 `BLOCK_D=64` 配置，`SCALE_BLOCK_D = next_power_of_2(64//128) = 0` → `tl.arange(0,0)` 崩。修复 = autotune 剪枝 `BLOCK_D < Q_BLOCK_SIZE` 的配置（`quant_videogen/real/quant_pack.py`，提交 8b81883）。B=128 的三次运行均在修复后完成。
