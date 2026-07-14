# 0714 报告：QVG 量化代码考古 / 真实 BPE / H100 速度复现

三个问题，三个答案（细节在各节）：

1. **量化代码在哪、是不是"逐 token Hadamard + B16 int2 scale"？** —— 代码全部定位（下有 6 阶段 file:line 地图）。假设**一半对一半错**：分块 scale 的结构完全说对了（逐 token 的连续 channel 分块、对称 absmax、FP8 E4M3 scale）——但发布版 QVG 是 **B=64**（B=16 是 QVG-Pro）；而 **Hadamard 完全不存在**，全 repo 零命中（唯一的 Hadamard 代码是我们自己移植的 QuaRot 基线）。旋转的位置上实际是 **S 轮 token 维 k-means 减质心**。
2. **真实 BPE 是多少？** —— 精确公式 `BPE = r + 8/B + S·8/128 + S·256·16/N`，与 README 实测显存**逐字节对上**（67.318 MB/层 vs 日志 67.32）。QVG INT2 在发布配置（chunk=29,640 token）下 **BPE=2.326，压缩 6.88×**；paper Table 1 报 6.94×，隐含比发布配置大 ~17% 的 chunk。三元打包可免费再赚 16-22%。
3. **paper 是怎么测速度的？** —— 只测一种东西：**量化/反量化给端到端生成时间带来的开销**（QVG vs BF16 各跑一遍完整生成，H100 + CUDA 12.8），k-means 成本计入、每 chunk 压缩一次；全文**没有任何 kernel 级 benchmark**。唯一绝对秒数在附录 C：SF 180 帧 = 端到端 43s / QVG 额外成本 0.74s / 开销 1.7%。方法细节见 §3。

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

### 1.3 QVG 与 QVG-Pro 参数全表

**两者共享的机制**（写死在代码里，不可配）：

| 机制 | 实现 | 代码位置 |
|---|---|---|
| Pipeline | S 轮 token 维 k-means 减质心 → 残差逐 token 连续 channel 分块量化 | `real/prq.py:52-93` |
| 残差网格 | 对称 absmax；INT2 → **三元 {−1,0,+1}**（max_int=2^(b−1)−1=1）；INT4 → {−7..+7} 15 级 | `real/quant_pack.py:70,81` |
| scale | **FP8 E4M3**，每 (token, head, B 个连续 channel) 一个；先舍入到 FP8 再用作除数 | `quant_pack.py:75-79`，默认值 `functions.py:267` |
| 残差打包 | 4 值/uint8（INT2）、2 值/uint8（INT4），真位打包 | `quant_pack.py:91-108`，`PACK_OUTPUT_INT8=True` 写死于 `functions.py:324` |
| 聚类索引 | uint8，每 (token, head, stage) 一个（要求 K≤256） | `prq.py:74-76`，`CLUSTER_ID_INT8=True` 写死于 `functions.py:325` |
| 质心表 | bf16 (B,H,K,128)，每层、K/V 各自、每 stage 一份，**每 chunk 重学** | `prq.py:63-72` |
| K 的缓存时机 | RoPE 之前（pre-RoPE），读取时重施 RoPE | `kv_cache.py` 读路径 |
| 没有的东西 | 无 zero-point（对称）、无 Hadamard/旋转、无 clipping（`-clip` 变体存在但未启用） | §1.2/§1.4 |

**可配参数对照**（发布脚本实际值，`grep` 自 `scripts/*/run_qvg.sh`）：

| 参数 | **QVG** | **QVG-Pro** | 出处 |
|---|---|---|---|
| `num_prq_stages`（S，k-means 轮数） | **1** | **4** | dataclass 默认=4（`sim/quant/quantize_config.py:24-26`）|
| `quant_block_size`（B，channel 块大小） | **64** | **16** | dataclass 默认=16（`quantize_config.py:21`）|
| `cache_num_k_centroids` / `_v_`（K） | 256 / 256 | 256 / 256 | 各 run_qvg.sh |
| `kmeans_max_iters` | LongCat **100**（一次性压缩条件窗）；SF / HY **2**（流式 + 质心缓存，paper §4.3 的 3× 优化） | 100（我们 0713 var 实验的跑法；无官方发布脚本） | 各 run_qvg.sh |
| `quant_type` | `triton-nstages-kmeans-int2`（int4 同名换后缀） | 同左 | |
| 逐 token 存储（残差+scale+索引；不含质心，= chunk→∞ 的渐近 BPE） | 2 + 8/64 + 1×8/128 = **2.1875** | 2 + 8/16 + 4×8/128 = **2.75** | §2 公式 |
| 真实 BPE @ LC 发布 chunk（= 上行 + 质心表分摊：S×256质心×128维×16bit ÷ (29,640token×128维)） | **2.326 → 6.88×** | 3.30 → 4.85× | §2.2 |
| 首帧 PSNR（0713 实测，INT2，mean±std n=3） | 28.88 ± 0.18 | **31.04 ± 0.005** | 0713 report |

三个值得注意的点：
1. **repo 的 dataclass 默认值就是 QVG-Pro**（S=4/B=16），但所有发布视频脚本都显式覆盖成 QVG（S=1/B=64）。**QVG-Pro 没有任何官方发布脚本**，paper Table 1 的 QVG-Pro 行只能靠 argparse 默认或手工传参复现（0713 我们的做法）。
2. `kmeans_max_iters` 在官方脚本里差 50 倍（LongCat 100 vs SF/HY 2）：LongCat 对固定 73 帧条件窗一次性压缩可以奢侈迭代；流式模型靠上一 chunk 质心热启动 + 2 次迭代（paper §4.3 声称的 3× k-means 加速正是这条路径）。
3. Pro 对 QVG 的交换：每元素多付 0.5625 bit 元数据 + 4 倍 k-means 成本（0713 实测 s/it 2.94 vs 2.69），买到 +2.16 dB 首帧 PSNR 和更稳的方差（σ 0.005 vs 0.18，四轮平均稀释了质心随机性）。**（0714 B 扫描修正：这 +2.16 dB 中约 +2.08 来自 B=64→16 的细 scale，S=1→4 仅贡献 ~+0.08 dB，见 §2.6。）**

### 1.4 考古中的意外发现

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
| 质心表 | bf16，(B,H,256,128) ×S 个，**每 chunk 重学** | **S·(256·128·16)/(N·128) = S·256·16/N**（N=chunk 内 token 数；每 head 一套 64 KB 的表摊给该 head 的 N×128 个元素，不随总长摊薄！） |
| zero-point | 不存在（对称量化） | 0 |

**公式**：`BPE = r + 8/B + S·8/128 + S·(256·128·16)/(N·128)`（末项分子分母的 128 数值上可约，= S·256·16/N），压缩率 = 16/BPE。
**验证**：LongCat 发布配置 chunk=29,640 token（73 帧条件窗），公式算得 67.318 MB/层、3231.28 MB 全模型——与 README 日志 67.32 / 3231.28 **逐字节吻合**。

### 2.2 BPE 表

| 配置 | N=10k | N=29,640（LC 发布配置） | N=145k | N→∞ 渐近 |
|---|---:|---:|---:|---:|
| QVG INT2 (S=1,B=64) | 2.597 → 6.16× | **2.326 → 6.88×** | 2.216 → 7.22× | 2.1875 → 7.31× |
| QVG-Pro INT2 (S=4,B=16) | 4.388 → 3.65× | ~3.30 → 4.85× | 2.863 → 5.59× | 2.75 → 5.82× |
| QVG INT4 (S=1,B=64) | 4.597 → 3.48× | 4.326 → 3.70× | 4.216 → 3.80× | 4.1875 → 3.82× |
| RTN INT2 (B=16) | 2.5 → 6.40× | 2.5 → 6.40× | 2.5 → 6.40× | 2.5 → 6.40×（与 N 无关） |

> RTN 行注：repo 的 `naive-int2`（RTN 基线）**也是三元对称**——`int_max = 2^(n−1)−1 = 1`，`clamp(round(x/s), −1, +1)`，无 zero-point（`sim/quant/lowbit_quantize.py:675-678,764`）。所以 2.5 = 2 bit 码 + 8/16 scale，没有 zero-point 项是诚实的；RTN 无索引、无质心表，故整行与 N 无关。若换成非对称 4 级（{0..3}+FP8 zero-point，即我们 0713 的 30.38 dB 跑法），BPE = 2 + 0.5 + 0.5 = 3.0 → 5.33×：多付 0.5 bit 买 +8.5 dB。

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

### 2.5 同 B=64 对齐的 QuaRot 对照（0714 新增实验）

0713 的 QuaRot 基线用 B=16（对齐 paper 基线记账 6.40×）；应用户要求补跑与 QVG 同 B=64 的两臂（单段续写、seed 0、frame-93 口径，节点 h100-110/116）：

| 配置（INT2，B=64） | 账单（块级标量记 FP8） | BPE | 压缩率 | frame-93 PSNR |
|---|---|---:|---:|---:|
| 对称 QuaRot | 2 + 8/64 | 2.125 | 7.53× | 19.14 |
| 非对称 QuaRot | 2 + 8/64 + 8/64（scale+zero-point） | **2.25** | 7.11× | **28.85** |
| **QVG（对照）** | 2 + 8/64 + 索引 0.0625 + 质心 0.138 | **2.326** | 6.88× | **28.88 ± 0.18 (n=3)** |

**读数：同块大小、几乎同 BPE 下，非对称 QuaRot 与 QVG 统计打平**（差 0.03 dB，在 QVG 自身 σ=0.18 的 0.2σ 内），且 QuaRot 略便宜、完全确定性、无 k-means 开销。即 QVG 的"索引+字典"（0.201 bit）在这个工作点上没有胜过一个简单 zero-point（0.125 bit）。对称 B=64 崩至 19.14（比对称 B16 再低 2.3 dB：三元档位下块越粗 absmax scale 越吃离群）。

**含义**：paper Table 1 的 INT2 优势(QVG 28.716 vs QuaRot 21.573)几乎全部来自基线被做成了三元对称——换成正确的非对称实现并对齐 B，优势消失。注意本行是单 prompt / 单 run（QuaRot 类确定性 σ≤0.003 已验证），多 prompt 复验前不下最终结论。候选去向:ISSUE_DRAFT 补充点 #3。

### 2.6 B 扫描九宫格（完整矩阵与台账见 [b-sweep.md](b-sweep.md)）

三方法 × B∈{16,64,128}，INT2，frame-93 口径（QVG 各格 n=3，QuaRot 类确定性 n=1）：

| 方法 \ B | 16 | 64 | 128 |
|---|---:|---:|---:|
| QVG（S=1） | **30.96 ± 0.026**（BPE 2.70） | 28.88 ± 0.18（2.326） | 28.41 ± 0.043（2.263） |
| QuaRot 非对称 | 30.38 ± 0.003（3.0） | 28.85（2.25） | 24.54 ± 0.002（2.125） |
| QuaRot + clip r=0.99 | 30.68（3.0） | 29.07（2.25） | **25.35**（2.125） |

四个要点：

1. **QVG-Pro 的优势解构**：QVG S=1/B=16 = 30.96 vs QVG-Pro S=4/B=16 = 31.04——**S=4 的四轮渐进 k-means 只值 +0.08 dB**，细 scale（B=64→16）才值 +2.08 dB。"QVG-Pro ≈ QVG + 细 scale"，S=4 的 4 倍聚类成本与翻倍 metadata 几乎白付。
2. **无全局赢家**：等预算 BPE≈2.25 处 QuaRot 反超 QVG B=128（28.85 vs 28.41）；BPE≈2.7-3.0 处 QVG 反超（30.96 vs 30.38）——两条质量-预算曲线交叉。
3. **对 B 的敏感度**：QuaRot 全程 −5.84 dB（64→128 崩 4.31）；QVG 16→64 同样陡（−2.08）但大 B 端有韧性（64→128 仅 −0.47，质心削平残差的功劳）。clip 增益 +0.30/+0.22/+0.81，B=128 处最大，方向符合"clip 治块内离群"。
4. **发布 kernel 无法跑 B=128**：`quant_pack` autotune 含 `BLOCK_D<Q_BLOCK_SIZE` 非法配置 → `tl.arange(0,0)` 崩溃；已修复（提交 8b81883）。发布方的 B 设计空间受此隐性限制。

---

## 3. paper 说的测速方法

paper 只测一种东西：**量化/反量化给端到端生成时间带来的开销**（相对 BF16 KV cache 基线）。没有 kernel 级 benchmark、没有吞吐表、没有逐步计时。三处口径（原文引文均出自 2602.02958v5）：

### 3.1 主文 §5.3 — 端到端开销百分比

> "We evaluate the **end-to-end latency** to quantify the **overhead introduced by quantization and dequantization** in our method."

- 测法：BF16 和 QVG 各跑一遍**完整生成管线**，报生成总时间的增幅百分比。
- 平台（§5.1）：NVIDIA H100，CUDA 12.8。
- k-means 成本**计入**（不排除、不摊销掉）；压缩每 chunk 一次（SF 16 latent 帧/chunk、HY 12、LongCat 压 73 帧条件窗），流式模型用上一 chunk 质心热启动（§4.3，`kmeans_max_iters=2`）。
- 报告值：LongCat **+2.1%**、HY-World **+1.5%**、Self-Forcing **+4.3%**。

### 3.2 附录 B.1 — 量化管线占比（按 chunk 扫描）

> "We **profile the runtime overhead of the k-means clustering and quantization pipeline** of QVG under different chunk sizes on Self-Forcing with INT2 quantization."

- 测法：k-means+量化管线耗时 ÷ 端到端运行时间，SF INT2，扫 chunk 大小。
- Table 4：chunk 37,440 / 18,720 / 9,360 token → **1.3% / 2.0% / 3.3%**。

### 3.3 附录 C — 唯一的绝对秒数

> "a breakdown of the **end-to-end latency** and the **QVG-related overhead** on Self-Forcing for both an NVIDIA H100 and an NVIDIA RTX 5090"

- 测法：端到端秒数与 "QVG extra cost" 两个量分开报。
- Table 6（SF，batch 1）：H100 **43 s / 0.74 s / 1.7%**；RTX 5090 72 s / 1.34 s / 1.9%。
- Table 7（batch 扫描）：batch 1/2/5 → 43/86/217 s，开销稳定 1.6-1.7%。

### 3.4 按 paper 方法实际操作时的对应物与注意事项

- paper **没写**的：计时器实现、warmup 次数、prompt 数量、"QVG extra cost" 的统计范围（是否含 k-means/permute）均未说明。
- repo 里与 "QVG extra cost" 对应的现成计时点：三个模型的推理代码都打印 `Quantization KV Cache Time: X s`（`torch.cuda.Event` 计时；SF 在 `experiments/Self-Forcing/pipeline/causal_inference.py:79-132`）——对 int2 日志求和即得。端到端 = 完整生成的墙钟。
- 具体命令：`bash scripts/Self-Forcing/run_bf16.sh` 与 `bash scripts/Self-Forcing/run_qvg.sh` 各计一次端到端墙钟，差值/基线 = §5.3 的 overhead%；量化成本从 qvg 日志求和。
- ⚠️ 换算陷阱：SF 发布脚本 `num_output_frames=180` 实际生成 **180 latent = 717 视频帧**（实测确认），是 Table 6 "180 frames"（45 latent）的 4 倍工作量——对齐附录 C 的绝对秒数时必须按块数折算。

### 3.5 按 paper 方法的实测结果（H100，同 GPU 顺序跑 bf16 → QVG）

| 模型（发布工作量） | 口径 | bf16 | QVG INT2 | 差 |
|---|---|---:|---:|---:|
| LongCat（10 段全量） | 端到端墙钟 | 2977 s | 2385 s | **−19.9%** |
| Self-Forcing（180 latent = 717 帧） | 端到端墙钟（稳态遍） | 605 s | 575 s（首遍含编译 599 s） | **−5.0%** |
| HY-WorldPlay（12 chunk） | **日志内稳态 chunk 4-11 生成计时** | 60.1 s（7.52 s/chunk） | 42.0 s（5.25 s/chunk） | **−30.2%** |

结论：paper 声称 INT2 有 +1.5~4.3% 的端到端开销；我们按其自己的方法在 H100 上实测，
**三个模型全部为负开销（INT2 更快）**，来源是 KV cache 搬运量的缩减（访存收益）。

HY 的口径说明（为什么不用墙钟）：result 文件的墙钟（bf16 635s vs qvg 稳态遍 200s，−68%）
被**运行顺序的缓存效应**污染——bf16 臂第一个跑，独自支付冷 PVC 权重加载与首次初始化
（其墙钟中约 431s 为非生成开销；qvg 两遍只有 ~164s/~95s），连 VAE 首次 decode（14.4s vs
0.2s）都不对称。改用日志内逐 chunk 生成计时（`Generate time for chunk N`，稳态段 4-11）后：
7.52 vs 5.25 s/chunk = −30.2%。交叉核对：QVG 侧与 0714 早晨异节点旧证据几乎逐字重合
（42.0 vs 42.1s），bf16 侧 60.1 vs 55.7s（±8% 节点差）——稳态结论稳健于 −25~−30%。

## 附：本节结论对既有文档的修正

- `backup/ARCHITECTURES.md` / 0713 讨论中"QVG pipeline 无 Hadamard"的说法：升级为代码级铁证（§1.2/1.3）。
- 0713 report 总表的"名义压缩率"列：现在有精确 BPE 公式支撑，QVG INT2 的 6.89× 对应发布 chunk；paper 的 6.94× 对应 ~35k chunk（§2.3）。
- `backup/ISSUE_DRAFT.md` 值得补充两点：附录 Table 5 的记账笔误、Table 1 与发布配置的 chunk 不一致（6.94× vs 6.88×）。
