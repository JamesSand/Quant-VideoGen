# PCA-KV 实验结果（Phase-1 七臂 + Auto-research 最终算法）

目标：把"mean/PCA 低秩 + 2bit 残差"方案落到真实生成——实现 fake-quant 编解码，
LongCat 单段续写，**frame-93 首帧 PSNR** 与 QVG/k-means 系对打（同 BPE 档位）。
谱分析（[pca-spectrum.md](pca-spectrum.md)）已给出依据：r=8 拆墙、残差高斯化、按 chunk 重算基。

## 方法（fake-quant 实现，零改 repo 源码）

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

## 结果一：Phase-1 七臂（frame-93 PSNR vs bf16 参考）

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

## 结果二：Auto-research 最终算法（双超 QVG）

目标钉死为 **PSNR > 28.88 且 BPE < 2.326**（双超 QVG）。核心假设：结果一已证明拆平的
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

### 补充验证（0716）

N4 的优势在 SSIM/LPIPS 上同样成立（SSIM 0.942 vs QVG 0.907；LPIPS 官方口径 0.067 vs
0.089），排除过度平滑担忧——详见 `../0716/ssim-lpips-validation.md`。

### 局限与下一步

单 prompt/seed（与既有全部数字同口径）；建议多 prompt 复验后定稿。真 kernel 化
（batched 128×128 eigh + 小 GEMM decode）与 QVG 速度对打是工程下一步。
复现：`pod_run_pca.sh pca_n4 4 2 asym pca 128`，评测 `pca_eval.py`。


---

## 结果三：OSCAR 式 attention-aware 基的检验（负结果，0716）

借鉴 OSCAR（arXiv 2605.17757）的 qqt 目标——K 的基不用自协方差、改用**校准集上 Q 的协方差**
（"query 实际探测的方向"）。两遍法实现：pass-1 干净生成钩 `q_norm` 输出累计每 (层,head)
QᵀQ（873k token/层）→ 特征基存盘；pass-2 用该固定基替换 K 的 chunk 自协方差基
（`oscar_calib_launcher.py` + `pca_quant.py` 的 `PCA_K_BASIS_FILE`）。

| 臂 | K 基 | r | BPE | frame-93 PSNR |
|---|---|---:|---:|---:|
| N4（对照） | 自协方差（chunk 现算） | 4 | 2.253 | **31.788** |
| O1 | QᵀQ（离线校准） | 4 | 2.253 | 28.883（−2.91） |
| O1b | QᵀQ | 8 | 2.317 | 31.327（−0.46，且更贵） |

**结论：attention-aware 基不适用于减法式方案**。机制：OSCAR 的 qqt 服务于**旋转**
（保留全 128 维，基只决定能量在量化块间怎么排布，query 偏好的排序对分块有利）；
我们的方案是**减法**（只保 top-r，其余进残差）——需要的是"K 的能量/墙住在哪"
（自协方差），不是"query 往哪看"。Q 的 top-4 方向漏掉了 K 的部分墙 → 残差留墙 →
掉回 QVG 水平；r=8 靠交集变大追回大半，但任何同价点都严格劣于自协方差。
**边界划清：数据自适应基的目标函数必须与基的用法匹配——旋转配 attention-aware，
减法配能量主导。**（校准基础设施保留，可复用于未来旋转式变体。）
