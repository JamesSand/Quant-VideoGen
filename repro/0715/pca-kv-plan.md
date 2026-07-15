# Plan：KV activation 的 PCA 分析 + 低秩 2bit / 残差 2bit 量化（0715）

核心想法：`K ≈ (低秩投影系数 → 2bit) × PCA 基 + (残差 → 2bit)`——把 SVDQuant 的
"低秩吸结构、残差好量化"思路搬到 KV cache，与 QVG 的 k-means（非线性码本减法）正面对比。

## 0. 为什么值得做（基于已有发现）

| 已有发现 | 对本计划的含义 |
|---|---|
| K 有强通道墙（SF 5.4× / LC 9.0× / HY 12.8×），跨 token 的结构性成分 | 正是低秩子空间该吸走的东西；PCA top 分量预计以极小 r 捕获墙 |
| L29 H9 整头离群 = ch95/ch49 两根通道支配 | PCA 会把它们自动收进前几个主成分——残差随之被拉平 |
| KV 沿视频位置轴平稳（三窗实测） | **一套基全程适用**；可离线校准基、在线只算系数 |
| cache 只存干净重编码那一步的 K/V | timestep 漂移不进 cache，PCA 基无需按步适配 |
| 与 k-means 的机制对照 | k-means = 非线性 256 点码本（每 token/head 8bit 索引）；PCA = 线性 r 维子空间（每 token r 个系数）——同一哲学的两种实现，应同 BPE 对打 |
| 成本 | 对 KV 做 PCA 极便宜：per-head 协方差 128×128，特征分解微秒级——DeltaQuant 说"在线 SVD 贵"针对的是激活全矩阵，对 head_dim=128 的 KV 不成立 |

**两个预判（写明，供证伪）**：V 的低秩性大概率差（本来就是均匀噪声状，PCA 吸不到东西），
胜负手几乎全在 K 侧；K 的谱预计衰减快（通道墙即低秩结构），关键看残差拉平后 2bit 是否回血。

## 1. 存储方案与 BPE 预算（先算账再动手）

每 (head, chunk) 存：基 `V_r ∈ r×128`（bf16，按 chunk 分摊）+ 每 token r 个系数
（2bit + FP8 scale）+ 128 维残差（2bit，三元或非对称 4 级 + FP8 scale/B）。

```
BPE ≈ 2(残差) + 2r/128(系数) + 8/B(残差scale) + 8·⌈r/16⌉/128(系数scale) + r·128·16/N(基摊销)
```

| r | BPE @ N=29,640 | 同档对照 |
|---|---:|---|
| 8 | ≈2.30 | QVG 的 2.326 |
| 16 | ≈2.44 | QVG 与 Pro 之间 |
| 32 | ≈2.72 | 低于 QVG-Pro 的 3.30 |

→ 实验矩阵按"同 BPE 档位"设计：r=8 打 QVG、r=32 打 QVG-Pro。

## 2. 分阶段计划

### Phase 0 —— 谱分析（纯本地，半天，零 pod）

数据全在手：SF 49GB dump、`sf_qkv.pt`（含 Q）、`lc_kv.pt`、`hy_kv.pt`。

- 每模型中间层 + 末层：per-head K/V 的奇异值谱衰减曲线；捕获 50/80/95% 能量所需的 r
- L29 H9 专项：top-2 主成分是否就是 ch95/ch49 方向
- 投影后残差画像（per-channel 3D 曲面 + kurtosis）：是否比原始 K 更各向同性（预判三元
  残差量化的友好度）
- 基的平稳性：开头窗算的基投到结尾窗，能量保留率（决定"离线基"还是"每 chunk 重算"）
- 产出：`pca-spectrum.md` + 图，据此定 r 档位

### Phase 1 —— 张量级编解码原型（本地，半天）

- 实现 `pca_quant.py`（quarot_quant.py 同款 fake-quant 风格）：per-head per-chunk
  协方差 → eig → top-r 投影 → 系数 2bit 非对称 → 残差 2bit（三元 / 非对称各一版）
- 张量级 rel-L2 同 BPE 对打：PCA(r=8/16/32) vs QVG k-means 路径 vs 非对称 QuaRot B64
- 止损线：张量级就输给 k-means ≥20% → 停

### Phase 2 —— 管线集成 + 首帧 PSNR（~8 pod，半天）

- `pca_launcher.py` 劫持 `compress_kv_cache`（quarot_launcher 同款手法），LongCat
  单段续写、frame-93 口径
- 臂：r∈{8,16,32} × 残差{三元, 非对称}，n=1（确定性）；对照 QVG 28.88 /
  QuaRot asym B64 28.85 / QVG-Pro 31.04
- 判定：任一档位同 BPE 下 ≥ QVG +0.3 dB → 有故事；全面平手也有价值（证明"减结构再
  量化"的收益与手段无关，是数据结构本身的属性）

### Phase 3 —— 写入 report-0715（present 版）

谱衰减图、同 BPE 对决表、"k-means vs PCA：非线性码本 vs 线性子空间"的机制结论。

## 3. 待确认后启动

1. PCA 范围：per-head（默认，与 QVG 质心对齐）还是跨 head？
2. 系数位宽：固定 2bit，还是加 4bit 对照臂？
3. K/V 都做，还是先只做 K（V 预计低秩性差；建议都测、预期写明）？
