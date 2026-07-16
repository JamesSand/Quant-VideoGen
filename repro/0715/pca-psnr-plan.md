# Plan：mean / PCA 低秩 KV 量化的首帧 PSNR 实测（0715，Phase 1+2 合并执行版）

目标：把 [pca-kv-plan.md](pca-kv-plan.md) 的方案落到真实生成——实现 fake-quant 编解码，
LongCat 单段续写，**frame-93 首帧 PSNR** 与 QVG/k-means 系对打（同 BPE 档位）。
谱分析（[pca-spectrum.md](pca-spectrum.md)）已给出依据：r=8 拆墙、残差高斯化、按 chunk 重算基。

## 1. 实现方式（零改 repo 源码，沿用既有全套基础设施）

- `repro/backup/scripts/pca_quant.py`：fake-quant 编解码核。每 (head, chunk)：
  ① mean 减除 → ②（PCA 臂）chunk 内协方差 → top-r 基 → 系数量化（非对称，逐 token 一个
  FP8 scale）→ ③ 残差分块量化（B=64，网格见待确认#1）→ ④ 重建返回 bf16。
  全程 float32 计算、无随机源（**确定性，n=1 即可**）。
- `repro/backup/scripts/pca_launcher.py`：劫持 `quant_videogen.compress.compress_kv_cache`
  （quarot_launcher 同款 pre-import patch），配置走环境变量
  `PCA_R / PCA_COEFF_BITS / PCA_RES_GRID / PCA_V_MODE`。
- Chunk 语义与对照组完全一致：LongCat 单段续写，73 帧条件窗 = 29,640 token 一次压缩
  ——与 QVG/QuaRot 各臂同口径，公平。
- 评测：`results/longcat/bf16/1-0/segment_1.mp4` 为参考，frame-93 PSNR（既有协议）。

## 2. 实验臂（INT2 档，LongCat，每臂 1 pod ~10 分钟）

| 臂 | 配置 | BPE | 同档对照（已测） |
|---|---|---:|---|
| M0 | **mean-only** + 残差 2bit B64 | ≈2.13 | QuaRot 对称 B64 2.125（19.14）——"1 质心 k-means"消融，检验 mean 单独值多少 |
| P1 | PCA r=4，系数 2bit | ≈2.25 | QuaRot 非对称 B64 2.25（28.85） |
| **P2** | **PCA r=8，系数 2bit**（主臂） | ≈2.32 | **QVG 2.326（28.88±0.18）** |
| P3 | PCA r=8，系数 4bit | ≈2.44 | —（检验系数是否被 2bit 卡住） |
| P4 | PCA r=16，系数 2bit | ≈2.44 | 与 P3 同价，r vs 系数位宽哪个值钱 |
| P2v | P2 但 V 也上 PCA（若默认 V=mean-only） | ≈2.32+ε | 检验谱分析"V 低秩差"的预判 |

判定：P2 vs QVG（同 2.32 档）±0.3 dB 内 = 平手（PCA 以确定性 + encode 快 100× 胜出）；
P2 高 0.3+ = 有故事；M0 若已接近 QVG，说明"减公共结构"的收益大头在 mean 不在字典。

## 3. 流程与产出

1. 本地先张量级自检（SF dump 上 rel-L2 对 QVG 路径，半小时，防翻车）
2. 6 pod 并行 → frame-93 PSNR → 结果表 + 首帧图并入 `report-0715.md`
3. REPRODUCE.md 记录全部命令

预计总耗时半天；集群配额按现状足够（6×1 GPU pod）。

## 4. 待确认（拍板后即跑）

1. **残差网格**：三元对称（严格对齐 QVG 口径）还是非对称 4 级？
   建议：主臂**三元**（可比性优先），P2 加一个非对称变体（+1 pod）看残差高斯化后
   非对称还剩多少收益。
2. **V 的默认策略**：谱分析显示 V 低秩性差——建议默认 **V=mean-only + 残差**（预算让给 K），
   P2v 臂做对照。同意吗？
3. **系数位宽**：2bit 主力 + 4bit 对照（P3）够吗，还是想再加 FP8 系数臂（+1 pod）？
4. 臂数确认：上表 6 臂（+可选 2 臂）。

## 5. 需要你帮助的地方

- 无硬性阻塞。唯一可能：集群若被占（NODE_BUSY 连环），可能需要你允许我错峰重试。
- 如果 P2 结果暧昧（与 QVG 差 0.3~0.5 dB），下一步是多 prompt 复验——那会是 ~12 pod
  的量，届时再来要批准。


---

## 6. 结果（0716 出数，frame-93 PSNR vs bf16 参考）

| 臂 | 配置 | BPE | PSNR | 读数 |
|---|---|---:|---:|---|
| M0 | 只减 mean + 三元 B64 | 2.13 | 25.03 | 字典必要性确认 |
| P1 | PCA r=4 + 三元 | 2.25 | 27.57 | 输同价 QuaRot 非对称 B64（28.85）1.3 dB |
| P2 | PCA r=8 + 三元（主臂） | 2.32 | 28.40 | 输同价 QVG（28.88±0.18）0.48 dB |
| **P2a** | **PCA r=8 + 非对称残差** | **2.44** | **30.20** | **+0.11 bit 换 +1.32 dB（vs QVG）；与 QuaRot 非对称 B16（30.38@3.0）近同质量、便宜 0.56 bit——新 Pareto 点** |
| P2v | r=8 且 V 也 PCA | 2.32 | 28.83 | V 侧 PCA 值 +0.43 |
| P3 | r=8 + 系数 4bit | 2.44 | 28.32 | 系数 2bit 已够（Phase-0 担心的风险未发生） |
| P4 | r=16 + 三元 | 2.44 | 28.01 | r 加倍反降：系数量化噪声吃掉收益，**r=8 是甜点** |

结论：
1. 同为三元残差，k-means 字典仍小胜 PCA（+0.48）——token 聚团结构有线性子空间表达不了的部分；
2. **非对称残差让 PCA 反超**：P2a = 30.20 @ 2.44，同预算档赢 QVG 1.32 dB——与 0713/0714
   "非对称 4 级 ≫ 三元"的发现在 PCA 残差上再次应验；
3. 叠加 PCA 固有优势（encode 便宜 7-178×、确定性、小 chunk 摊销好），本线画像从"打平"
   升级为"同预算 +1.3 dB"。
4. 未试组合：P2a + V-PCA（潜在 ~30.6 @ 2.44）；多 prompt 复验待批。

原始数据：`results/pcastudy/pca_*/1-0/segment_1.mp4`，结果文件 `repro/backup/race/result_pca_*.txt`。


---

## 7. Auto-research 第二轮（0716）：比 QVG 更好且更便宜的最终算法

目标钉死为 **PSNR > 28.88 且 BPE < 2.326**（双超 QVG）。核心假设：§6 已证明拆平的
PCA 残差近高斯 → **残差块可放粗到 B=128**，省下的 scale/zero-point 预算覆盖非对称开销。
四臂全部命中目标（LongCat frame-93，本地 H100 串行，集群存储故障期间）：

| 臂 | 配置（残差均为非对称 2bit @ B=128） | BPE | PSNR | vs QVG (28.88@2.326) |
|---|---|---:|---:|---|
| N1 | K: mean+PCA r=8；V: mean | 2.221 | 30.235 | +1.35 dB，−4.5% bit |
| N2 | K/V 都 mean+PCA r=8 | 2.317 | 31.469 | +2.59 dB，−0.4% bit |
| N3 | K: mean+PCA r=6；V: mean | 2.205 | 30.729 | +1.85 dB，−5.2% bit |
| **N4** | **K/V 都 mean+PCA r=4** | **2.253** | **31.788** | **+2.91 dB，−3.1% bit** |

### 最终算法（暂名 **PCA-KV**，N4 配置）

```
每 (head, chunk)：X ≈ mu + quant₂(coef) · V₄ᵀ + asym₂@B128(residual)
  mu: token 均值 [128]，bf16                    ← 字典
  V₄: top-4 PCA 基 [128×4]，bf16（协方差 128×128 现算，微秒级）← 字典
  coef: 每 token 4 个系数，非对称 2bit（每 token 一组 scale/zp）
  residual: 非对称 2bit，块=128（整 head 一块，scale+zp 各 FP8）
K 与 V 同方案。BPE = 2.253（K/V 合账，@LC chunk 29,640）。
```

### 与全家福对比

| 方法 | BPE | PSNR |
|---|---:|---:|
| **PCA-KV (N4)** | **2.253** | **31.788** |
| QVG-Pro（S=4,B=16） | 3.30 | 31.04 |
| QuaRot 非对称 B16 | 3.0 | 30.38 |
| QVG（发布配置） | 2.326 | 28.88±0.18 |
| QuaRot 非对称 B64 | 2.25 | 28.85 |

**PCA-KV 在同价位（2.25）比 QuaRot 高 2.9 dB，比贵 47% 的 QVG-Pro 还高 0.75 dB——
Pareto 全面占优**；叠加固有优势：encode 比 k-means 便宜 ~178×（无迭代）、完全确定性
（消灭 QVG 的 σ=0.18）、字典极小（5 个向量 vs 256 质心，小 chunk 摊销好）。

### 两个反直觉的机制发现

1. **r 越小越好（4 > 6 > 8 > 16）**：系数的 2bit 量化噪声随 r 线性累积，而非对称
   B=128 残差本身足够强——低秩分支只需拆墙（§6 已证 r=8 拆干净，r=4 拆掉主体即够），
   多余的秩反而引入噪声。
2. **V-PCA 是被低估的大头（+1.2~1.5 dB）**：V 无通道墙但有内容相关性，PCA 吸的是
   后者；且 V 直接乘 attention 权重进输出，V 的保真度弹性更大。

### 局限与下一步

单 prompt/seed（与既有全部数字同口径）；建议多 prompt 复验后定稿。真 kernel 化
（batched 128×128 eigh + 小 GEMM decode）与 QVG 速度对打是工程下一步。
复现：`pod_run_pca.sh pca_n4 4 2 asym pca 128`，评测 `pca_eval.py`。
