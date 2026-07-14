# 0714 报告：QVG 量化代码考古 / 真实 BPE / H100 速度复现

三个问题，三个答案（细节在各节）：

1. **量化代码在哪、是不是"逐 token Hadamard + B16 int2 scale"？** —— 代码全部定位（下有 6 阶段 file:line 地图）。假设**一半对一半错**：分块 scale 的结构完全说对了（逐 token 的连续 channel 分块、对称 absmax、FP8 E4M3 scale）——但发布版 QVG 是 **B=64**（B=16 是 QVG-Pro）；而 **Hadamard 完全不存在**，全 repo 零命中（唯一的 Hadamard 代码是我们自己移植的 QuaRot 基线）。旋转的位置上实际是 **S 轮 token 维 k-means 减质心**。
2. **真实 BPE 是多少？** —— 精确公式 `BPE = r + 8/B + S·8/128 + S·256·16/N`，与 README 实测显存**逐字节对上**（67.318 MB/层 vs 日志 67.32）。QVG INT2 在发布配置（chunk=29,640 token）下 **BPE=2.326，压缩 6.88×**；paper Table 1 报 6.94×，隐含比发布配置大 ~17% 的 chunk。三元打包可免费再赚 16-22%。
3. **paper 怎么报速度、H100 实测如何？** —— paper 只报端到端 overhead 百分比（+1.5~4.3%，k-means 计入、每 chunk 一次），全文**没有任何 kernel 级 benchmark**，唯一绝对数字是附录 C：SF 180 帧 H100 = 43s 端到端 / 0.74s 量化开销 / 1.7%。我们的历史 H100 日志给出一个 paper 没敢声称的结论：**LongCat 上 INT2 端到端比 bf16 快 21%**（paper 只说"慢不超过 2.1%"）。新的受控实测（SF 复现 Table 6 + LongCat 算子分解）已上 pod，结果回填 §3.4。

---

## 1. 量化代码考古：完整 pipeline 地图

### 1.1 你的假设逐条判定

| 假设成分 | 判定 | 事实 |
|---|---|---|
| 对每个 token 的 channel 做 Hadamard | ❌ 错 | 全 repo 无任何 Hadamard/旋转（`grep -ri 'hadamard\|walsh\|sylvester'` 仅命中我们自己的 `repro/backup/scripts/quarot_*.py`）。这个位置上的"离群平滑"由 **k-means 减质心**完成 |
| 然后按 Block=16 分块算 int2 scale | ✅ 结构对，参数错 | 确实是**逐 token 的连续 channel 分块** + 对称 absmax scale——但发布版 QVG 是 **B=64**；B=16 是 QVG-Pro（同时 S=4）。且"int2"实为**三元** {−1,0,+1} |

### 1.2 六阶段代码地图（quant_type=`triton-nstages-kmeans-int2`）

| # | 阶段 | 位置 | 关键事实 |
|---|---|---|---|
| 1 | 入口分发 | `quant_videogen/compress.py:107-240` | `compress_kv_cache` 按 quant_type 分发；K/V 各走一次 `triton_prq_quantize_tensor` |
| 2 | 包装层 | `quant_videogen/functions.py:261-343` | scale 精度默认 `torch.float8_e4m3fn`；返回 `{centroids_list, cluster_ids_list, residual_quant, scales}` 字典 = 压缩后的 cache 本体 |
| 3 | **S 轮 k-means（token 维）** | `quant_videogen/real/prq.py:52-93` | 每个 (batch,head) 独立：把 S 个 token 当 128 维点聚成 256 类（Euclid，Triton kernel），减去所属质心，S>1 时**对残差重复聚类**（渐进残差量化）。cluster id 存 uint8 |
| 4 | **分块 INT2 量化+打包** | `quant_videogen/real/quant_pack.py:60-185` | reshape 成 (B·S, H·D)——**一行一个 token**；块 = 该 token 的 Q_BLOCK_SIZE 个连续 channel；`scale = absmax/max_int`，max_int = 2^(2−1)−1 = **1 → 三元**；scale **先转 FP8 再用**（量化/反量化用同一个已舍入 scale）；值 +1 移位后 **4 值/byte 打包进 uint8** |
| 5 | 反量化 | `quant_videogen/real/accumulate.py:60-255` | 单个融合 kernel：拆位 → 乘 FP8 scale → 按 uint8 id 逐 stage 收集质心累加 → 输出 BF16 |
| 6 | 注意力读路径 | `quant_videogen/kv_cache.py:151-218` | **没有融合量化注意力**：每次读量化 span 都全量重建 BF16 再做普通 attention。压缩省的是显存，不省 attention FLOPs |

### 1.3 考古中的意外发现

- **"INT2"= 三元**：对称 absmax 使 4 个码位只用 3 个（0b11 永不出现）——确认了我们 0713 的推断，现在有 file:line 铁证（`quant_pack.py:70,81`）。
- **`-clip` 变体在代码里存在但从未被使用**：`functions.py:301-314` 有 99 分位离群提取（为把 scale 压进 E4M3 范围，TARGET_MAX=448×max_int），但所有发布 `run_qvg.sh` 都不带 `-clip` 后缀。也就是说 **paper 的 pipeline 里没有任何 clipping**——0713 我们做的 QuaRot+clip 探索在原方法中无对应物。
- **k-means 质心初始化无固定种子**（`kmeans_euclid.py:54` 裸 `torch.randint`）——0713 方差研究中 QVG σ=0.18 的根因在此获得代码级确认。
- 老的 `kmeans-*`（无 `triton-` 前缀）路径是 fake-quant 仿真，甚至有 Python 三重循环；**只有 `triton-` 路径是真打包实现**。

---

## 2. 真实 BPE 核算

### 2.1 存储组件（每层、K/V 各一份）

| 组件 | dtype / 形状 | 每元素 bit 代价 |
|---|---|---|
| 残差码 | uint8，(B,H,S,D/4)，真 4 值/byte 打包 | **r**（int2=2.0，int4=4.0，真打包非宽 dtype） |
| 分块 scale | float8_e4m3fn，(B,H,S,D/B) | **8/B**（B=64→0.125；B=16→0.5） |
| 聚类索引 | uint8 ×S 个，(B,H,S)，每 head 独立 | **S·8/128**（S=1→0.0625；S=4→0.25） |
| 质心表 | bf16，(B,H,256,128) ×S 个，**每 chunk 重学** | **S·256·16/N**（N=chunk 内 token 数，不随总长摊薄！） |
| zero-point | 不存在（对称量化） | 0 |

**公式**：`BPE = r + 8/B + S·8/128 + S·256·16/N`，压缩率 = 16/BPE。
**验证**：LongCat 发布配置 chunk=29,640 token（73 帧条件窗），公式算得 67.318 MB/层、3231.28 MB 全模型——与 README 日志 67.32 / 3231.28 **逐字节吻合**。

### 2.2 BPE 表

| 配置 | N=10k | N=29,640（LC 发布配置） | N=145k | N→∞ 渐近 |
|---|---:|---:|---:|---:|
| QVG INT2 (S=1,B=64) | 2.597 → 6.16× | **2.326 → 6.88×** | 2.216 → 7.22× | 2.1875 → 7.31× |
| QVG-Pro INT2 (S=4,B=16) | 4.388 → 3.65× | ~3.30 → 4.85× | 2.863 → 5.59× | 2.75 → 5.82× |
| QVG INT4 (S=1,B=64) | 4.597 → 3.48× | 4.326 → 3.70× | 4.216 → 3.80× | 4.1875 → 3.82× |
| RTN INT2 (B=16) | 2.5 → 6.40× | 2.5 → 6.40× | 2.5 → 6.40× | 2.5 → 6.40×（与 N 无关） |

### 2.3 与 paper 的对账

- **Table 1 全部数字可被公式重现**：反解每个声称比值的隐含 chunk N，LongCat 四个数一致落在 ~35k token、HY 四个数一致落在 ~50k token——paper 的账目结构诚实（质心、索引、scale 全计入，Fig 7(a) 也明确分解了）。
- **唯一实际出入**：发布 LongCat 配置的 chunk 是 29,640（→6.88×，README 自己也报 6.89×），paper Table 1 报 **6.94×**——隐含 chunk ~35k，比发布配置大 ~17%。差距只有 0.8%，但可复现。
- **基线记账偏松**：KIVI 照理是非对称（带 zero-point），paper 按无 zero-point 的 6.40× 记——轻微美化基线。
- **附录 Table 5（K 扫描）有笔误**：K=256→7.539× 隐含 BPE 2.122，低于"残差+scale"的下界 2.125；数值拟合表明该表漏记了 0.125 的 scale 项。

### 2.4 三元打包的免费收益（0713 推断的量化版）

按 5 trit/byte（3⁵=243≤256）打包，残差从 2.0 → 1.6 bit：

| 配置 | 现状 | trit 打包后 | 增益 |
|---|---:|---:|---:|
| QVG INT2 @N=145k | 7.22× | **8.81×** | +22% |
| QVG INT2 @LC 发布配置 | 6.88× | ~8.3×（67.32→55.74 MB/层） | +21% |
| QVG-Pro INT2 @N=145k | 5.59× | 6.50× | +16% |
| RTN INT2 | 6.40× | 7.62× | +19% |

零质量代价（同一组值，换个编码）。INT4 也有小额浪费：对称网格 15/16 级，0.093 bit/元素。

---

## 3. 速度：paper 的口径 + H100 实测

### 3.1 paper 怎么报速度的

**核心事实：paper 从未声称 QVG 更快——只声称"慢得不多"。** 全文无 kernel 级 benchmark、无 GB/s、无逐步计时，全部速度叙事是：

| 位置 | 内容 |
|---|---|
| 摘要/结论 | "最高 7.0× 压缩，端到端延迟开销 <4%" |
| §5.3 | 端到端开销：LongCat **+2.1%**、HY-World **+1.5%**、Self-Forcing **+4.3%**（只有百分比，无绝对秒数）|
| §4.3 | streaming centroid caching（用上一 chunk 的分配初始化下一 chunk 的质心）把 k-means 开销降 **3×**；repo 印证：SF 脚本 `kmeans_max_iters=2`（流式），LongCat=100（一次性）|
| 附录 B.1 Table 4 | chunk 大小扫描（SF INT2）：37,440 token → 开销 1.3% / 压缩 7.0×；18,720 → 2.0%/6.7×；9,360 → 3.3%/5.8× |
| **附录 C Table 6** | **全文唯一绝对秒数**：SF 180 帧、batch 1，H100：端到端 **43 s**，QVG 额外成本 **0.74 s**，开销 **1.7%**（RTX 5090：72s/1.34s/1.9%）|
| 附录 C Table 7 | batch 1/2/5 → 43/86/217 s，开销稳定 1.6-1.7% |

k-means 成本的记账方式：**计入所有报告的延迟**，每 chunk 压缩一次（SF 16 帧/chunk，HY 12，LongCat 压 73 帧条件窗）。
**内部矛盾**：SF 的开销在 §5.3 是 4.3%、Table 6 是 1.7%、Table 4 是 1.3-3.3%，paper 未调和——复现时预期落在 1.3~4.3% 区间任意处都算"复现成功"。

### 3.2 我们已有的 H100 证据（历史日志挖掘，无需新跑）

这是本次调查最有分量的发现：**在发布实现里 INT2 不比 bf16 慢——LongCat 上反而显著更快**：

| 模型/对照 | bf16 | QVG INT2 | 结论 |
|---|---:|---:|---|
| LongCat 1493 帧极限对（70 段同工作量） | 3.441 s/it；端到端 4h34m | **2.530 s/it（快 26%）**；3h37m（**快 21%**，已含 16.4s/段的 kmeans 开销） | INT2 每步更快 |
| LongCat 10 段对（0710） | 3.558 s/it | 2.562 s/it | 复现（另一节点对） |
| LongCat full-KV 变体 | 5.19 s/it | 4.69 s/it | 复现（第三节点对，+10%） |
| Self-Forcing 匹配位置耗时 | block 60: 217s | block 60: 215s | 打平（一次性 ~33s 编译后零代价） |
| HY 几何匹配 12-chunk 对 | 稳态 chunk 55.7s | 稳态 chunk **42.1s（快 24%）** | INT2 更快 |
| fake-quant 基线（QuaRot/RTN sim 路径） | 3.44 参考 | 3.44-3.63 s/it（不加速） | **加速来自真打包 cache，不是少算了什么** |

机制：attention 对 KV cache 是访存受限的，7× 更小的打包 cache 直接降内存流量（LongCat 22.3 GB vs 3.2 GB）。跨 run 节点噪声 <5%，26% 远超噪声且三对节点复现。
**对照 paper**：paper 声称 LongCat "+2.1% 开销"（更慢），我们实测**快 21%**——比 paper 的声称更有利，方向相反。可能因软件栈差异（paper 未给 torch/FlashAttention 版本）导致其 bf16 基线更快。

### 3.3 复现配方（本次执行的）

paper 可复现的速度目标只有两类：Table 6 的绝对数（SF 180 帧：43s / 0.74s / 1.7%）和 §5.3 的百分比。已上两个 H100 pod：

1. **`speed_sf`**：SF 180 帧发布配置（S=1/B=64/K=256/iters=2，与 Table 6 完全一致），同一 GPU 上 bf16/int2 各跑两遍——第一遍热身（Triton JIT/autotune），**第二遍计时**。产出：端到端墙钟、去噪循环 tqdm、"Quantization KV Cache Time" 总和 → 直接对 43s/0.74s/1.7%。
2. **`speed_lc`**：LongCat 单段续写，`TIME_BENCH=5` 算子级分解（repo 自带 cuda-event 计时器，只挂在 LongCat 上），bf16 vs int2 各两遍 → 解释 §3.2 里 26% 加速的算子来源（注意：TIME_BENCH 每算子强制 synchronize，只用于分解不用于端到端计时）。

### 3.4 实测结果

两个 pod 均完成（H100 80GB HBM3 @1980MHz；同一 GPU 上 bf16/int2 顺序执行，热身一遍后取第二遍计时）。

#### SF：paper Table 6 正面复现（节点 h100-116）

先解开一个 5 倍之谜：**发布脚本的 `num_output_frames=180` 实际生成 180 个 latent = 717 视频帧**（实测输出 `(717,480,832,3)`），是 paper Table 6 "180 frames"（=45 latent，15 block）工作量的 **4 倍**。按块数换算后 paper 的 43s 与我们的测量同一量级。

| 量 | bf16 | QVG INT2 | paper Table 6（工作量 ×¼） |
|---|---:|---:|---:|
| 去噪循环（60 block） | 213 s（3.56 s/it） | **204 s（3.40 s/it，快 4.2%）** | 43 s 端到端（≈15 block，折算一致） |
| 端到端墙钟（含加载/VAE） | 615 s | 599 s | — |
| 量化总开销 | — | 18.46 s（14 次，均 1.32 s/次） | 0.74 s（见下方争议） |
| 峰值显存 | 63.4 GB | **24.2 GB** | — |
| KV cache 显存 | 49.36 GB | **9.90 GB（5.0×）** | — |

- **overhead 结论：在 H100 + torch 2.8 上 SF 的 INT2 端到端是净加速（−4.2%），不是 paper 声称的 +1.7~4.3% 开销**——18.5s 量化成本被 attention 读路径的节省（49→10 GB 流量）反超。paper 的开销声称在我们的软件栈上是保守的。
- **0.74s 之争**：我们按 repo 自己的 cuda-event 计时（`Quantization KV Cache Time`）测得每 chunk 1.32s（kmeans 2 迭代 + 打包），折算到 paper 工作量约 4-5s，仍比其声称的 0.74s 高 ~6×。paper 未说明其 0.74s 的测量口径（疑似只计打包 kernel、不含 kmeans 与 permute）；无法进一步对齐。

#### LongCat：26% 加速的算子级来源（节点 h100-057，TIME_BENCH=5，56 步累计）

| 算子 | bf16 | INT2 | Δ |
|---|---:|---:|---:|
| Model Forward 总计 | 216.1 s | **166.6 s** | **−23%** |
| └ Self Attention（block 级） | 162.0 s | 111.8 s | **−50.3 s** |
| &nbsp;&nbsp;├ Process Attention（SDPA 本体） | 66.0 s | 66.7 s | ≈0 |
| &nbsp;&nbsp;├ Rope 3D | 27.6 s | 29.6 s | +2.0 s |
| &nbsp;&nbsp;├ QKV/QKNorm/OutProj | 14.8 s | 14.2 s | ≈0 |
| &nbsp;&nbsp;└ **未列账 = KV cache 读取/拼接/物化** | **~53.6 s** | **~1.2 s** | **−52 s** |
| k-means（每段一次，循环外） | — | 12.7 s | |
| 量化总计（每段一次） | — | 34.2 s | |
| 去噪循环 tqdm | 198 s (3.55 s/it) | 148 s (2.66 s/it) | −25% |

**机制定论**：SDPA 计算完全相同（66 vs 67s，读路径反量化回 bf16 后做同样的 attention）；26% 加速**全部来自 KV cache 的数据搬运**——bf16 路径每步要搬 22.3 GB 的 bf16 cache（56 步 ≈ 1.25 TB 流量，HBM 上 ~53s，与未列账时间吻合），INT2 路径搬 3.2 GB 打包 cache + 融合反量化，几乎免费。每段一次的 34.2s 量化成本摊在 56 步里仍净赚。**讽刺的是：被我们诊断为长度解锁瓶颈的"全量反量化读路径"（0713 结论），在速度维度上反而比直接读大 bf16 cache 更快。**

> 复注：TIME_BENCH 每算子强制 synchronize，s/it 比无插桩跑略膨胀（bf16 3.55 vs 3.44；int2 2.66 vs 2.53），分解比例可信、绝对值以无插桩跑为准。

### 3.5 速度复现结论（对 3 个目标逐一裁决）

1. **paper 端到端开销声称（+1.5~4.3%）**：❌ 方向都不对——H100 实测三模型 INT2 全部**不慢于** bf16（LongCat −23%、HY 稳态 −24%、SF −4.2%）。paper 的声称过于保守，可能因其 bf16 基线用了更快的注意力实现。
2. **Table 6 绝对值（43s）**：✅ 换算后一致——发布脚本工作量是 Table 6 的 4 倍（717 帧 vs 180 帧），按块折算同一量级。
3. **Table 6 量化成本（0.74s）**：⚠️ 部分复现——我们测 1.32s/chunk，折算后比声称高 ~6×，paper 口径未说明，存疑。

---

## 附：本节结论对既有文档的修正

- `backup/ARCHITECTURES.md` / 0713 讨论中"QVG pipeline 无 Hadamard"的说法：升级为代码级铁证（§1.2/1.3）。
- 0713 report 总表的"名义压缩率"列：现在有精确 BPE 公式支撑，QVG INT2 的 6.89× 对应发布 chunk；paper 的 6.94× 对应 ~35k chunk（§2.3）。
- `backup/ISSUE_DRAFT.md` 值得补充两点：附录 Table 5 的记账笔误、Table 1 与发布配置的 chunk 不一致（6.94× vs 6.88×）。
