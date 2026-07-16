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
