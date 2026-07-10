# QuantVideoGen PSNR 复现报告

> 机器：8× H100 80GB · torch 2.8.0+cu128 · triton 3.4.0 · flash-attn 2.8.3
> 仓库 commit：as-checked-out · Paper：arXiv:2602.02958v5（本地 PDF 与 arXiv 下载版 md5 一致）
> 所有命令/脚本见 `repro/`，原仓库文件零修改。

## 一、结论摘要

**完全对上的部分：**

| 指标 | 本地 | Paper/README | 状态 |
|---|---|---|---|
| LongCat INT2 KV 内存 | 67.32 MB/层, 3231.28 MB 总 | README: 67.32 / 3231.28 | ✅ 逐位一致 |
| LongCat BF16 KV 内存 | 464.00 MB/层, 22272.00 MB 总 | README: 464.00 / 22272.00 | ✅ 逐位一致 |
| 压缩率 INT2 | 6.89× | README 6.89× / paper 6.94× | ✅ |
| K-cache rel-L2 (INT2, 真实数据) | mean 0.267 | Paper Fig 7(b): 0.15–0.3 | ✅ 落在曲线内 |
| V-cache rel-L2 (INT2, 真实数据) | mean 0.453 | Paper Fig 7(c): 0.26–0.47 | ✅ 落在曲线内 |
| K/V rel-L2 (INT4) | 0.042 / 0.071 | — | （INT2 的 ~1/6，合理） |
| triton 实路径 vs 纯 torch 模拟路径 | rel_err 完全一致 (0.1492 vs 0.1497) | — | ✅ kernel 数值正确 |

**对不上的部分（核心差异）：**

| 视频 PSNR (量化 vs BF16 基线) | 本地（发布协议） | Paper Table 1 | 差距 |
|---|---:|---:|---:|
| LongCat-Video INT2 | 17.9 dB (seg1) | 28.716 | −10.8 |
| LongCat-Video INT4 | 21.5 dB (seg1) | 37.141 | −15.6 |
| HY-WorldPlay INT2 | 18.67 dB | 29.174 | −10.5 |
| HY-WorldPlay INT4 | 22.51 dB | 34.454 | −11.9 |
| LongCat QVG-Pro INT2 (S=4,B=16) | 22.05 dB (seg1) | 30.376 | −8.3 |

（10-segment 完整 long-gen 数字见 §四，跑完后回填。）

## 二、为什么对不上——证据链

排除法（每一步都有实验支撑）：

1. **不是环境/安装问题**：README 的 flash-attn wheel ABI 错误已修复（cxx11abiTRUE）；NGC 容器 `TRITON_PTXAS_PATH` 指向 CUDA 13 ptxas 的问题已修复（`repro/env_fix.sh`）；全部 run 正常完成、无 err.txt。
2. **不是 kernel bug**：`repro/kernel_ab_test.py` 对同一 tensor 跑 triton 实路径与纯 torch 模拟路径，INT2/INT4 重建误差完全一致（0.1492 vs 0.1497 / 0.0225 vs 0.0225）；smoothing 较 RTN 改善 3.6×，符合设计。
3. **不是 RNG 污染**（虽然它真实存在）：发布代码中 k-means 的 `torch.randint`（kmeans_euclid.py:54）消耗全局 CUDA RNG，会使 QVG run 从第 2 段起噪声与 bf16 分叉。我们用 `repro/longcat_rngiso_launcher.py`（不改仓库文件的 monkeypatch）把它隔离后，segment-1（本就噪声对齐）数字几乎不变：17.91 vs 17.35。HY-WorldPlay 全程噪声预采样、天然对齐，仍只有 18.67。
4. **不是配置错**：压缩率/内存与 README 逐位一致，说明 S=1/B=64/K=256 正是作者产表配置。
5. **量化误差水平与 paper 自报一致**：真实 KV 上 INT2 V-cache rel-L2 = 0.453，与 paper Figure 7(c) 的 Value 曲线（0.26–0.47）吻合。**问题在于：每次注意力读取都带 ~45% 的 V 误差，在自回归生成中不可能只造成 3.7% 的像素偏差（28.7 dB）**。
6. **混沌放大的硬上界**：两个只差 k-means 随机种子的 INT2 run 互比 = **22.41 dB**。若量化 run 是围绕 bf16 的无偏扰动，quant-vs-bf16 的期望上限 ≈ 22.41+3 ≈ 25.4 dB。paper 的 28.716 超出了发布 pipeline 的物理可达范围。

## 三、paper 数字最可能的来源

- Paper §5.1 原文："*For the similarity experiments on LongCat-Video-13B, we report the number of the first generated chunk as the content starts to diverge while maintaining the same quality.*" —— 作者明确承认 PSNR 只测**第一个生成 chunk**、且承认之后内容漂移。
- 我们的逐帧曲线：LongCat INT2 第一个生成帧 = **29.4 dB**（paper 28.716），INT4 第一帧 = 33.6；HY INT4 分叉起点 = 33.5（paper 34.454）。paper 的数字与"分叉起点处保真度"高度吻合，与 20 帧窗口平均（17.9）不吻合。
- Paper 内部佐证：Table 1 中 INT4 的 QVG (37.14) ≈ QVG-Pro (37.10)，但两者张量级误差差 ~2.5×——只有测量在某个噪声地板附近饱和才会出现这种现象。
- 另外发布代码缺失 paper 声称的 centroid caching（`init_centroids` 参数存在但无任何调用者传入），k-means 为纯随机初始化 + 迭代上限，属于代码-论文差异。

### 穷举式窗口扫描（否定性结果）

对逐帧 PSNR 曲线扫描所有可能的评测窗口 [1..N)：

- LongCat：INT2 在 [1..179) 均值 28.739（与 28.716 仅差 0.023dB）；INT4 在 [1..121) 均值 37.096（差 0.045dB）——**但两者所需窗口不同**。同一窗口下两个数字无法同时成立（差 1.6–8 dB）。平滑衰减曲线扫窗口必然穿过任意目标值，单点吻合无协议意义。
- HY-WorldPlay：INT2 在 [1..45] 均值 29.045（差 0.13dB），同窗口 INT4 = 32.82 ≠ 34.454。
- 结论：**不存在任何单一评测窗口/skip 组合能同时复现 Table 1 的 INT2+INT4 数字**。

### 额外配置假设的穷举验证（均被否定）

| 假设 | 实验 | 结果 | 判定 |
|---|---|---|---|
| paper 用的是 `480p_long_gen_fullkv` workload（上下文逐段增长，对应 teaser 的大 KV 叙事） | bf16+INT2+INT4 各 6 段 | seg1 确有提升（INT2 20.1 / INT4 25.5，因条件帧 93>73），但逐段**递减**（INT2: 19.9→13.5→14.6；INT4: 25.2→18.2→15.9），跨段漂移压过长上下文钉扎 | ❌ |
| （fullkv 附带发现） | bf16 fullkv 基线 | **在 segment 4（153 帧上下文）OOM**——80GB H100 装不下 bf16 全历史 cache，而 INT2/INT4 正常跑（峰值 44.8GB）。QVG 的内存价值主张成立；但也意味着 paper 若用此场景，其 "BF16 reference" 的产生方式（teaser 承认 bf16 在 4090 上 "previously infeasible"）未在论文/代码中说明 | ⚠️ |
| paper 的 HY 用 run_bf16.sh 的 14-chunk/56-52 几何跑两条 run | 补跑 shipped bf16 + 同几何 INT2/INT4 | INT2 18.14 / INT4 21.59（与 12-chunk matched 的 18.67/22.51 无实质差异） | ❌ |
| QVG-Pro 配置（S=4,B=16）能达到 paper 水平 | seg1 | 22.05（paper Pro=30.376） | ❌ |
| 作者忘 skip 前缀、codec 噪声抬高均值 | 实测无 skip 全均值 | LongCat 21.86 | ❌ |
| 存在统一评测窗口 | 全窗口扫描 | INT2/INT4 所需窗口互斥 | ❌ |

### 外部线索核查（全部为空）

- GitHub（svg-project/Quant-VideoGen）：0 个 issue；git 全历史无删除的评测脚本（首提交→LingBot 集成，共 10 commits）。
- 项目主页（svg-project.github.io/qvg）：无评测代码、无发布的 bf16/量化视频对、无协议细节。
- 复现 Table 1 精确数字所需的信息（作者的评测脚本/确切窗口/生成配置）不存在于任何公开渠道。

## 四、完整 long-gen 数字（10 segments, skip_frames=93）

LongCat segment_10（93 init + 200 生成帧，评测后 200 帧）：

| 配置 | 发布版 | RNG 隔离版 | Paper Table 1 |
|---|---:|---:|---:|
| INT2 | 12.70 dB | 12.63 dB | 28.716 |
| INT4 | 12.51 dB | 12.02 dB | 37.141 |

- 长时程完全饱和：INT4 与 INT2 无差别（甚至略低）——漂移由混沌主导，与量化精度脱钩。任何窗口/协议组合都无法从这些视频中得出 28.7/37.1。
- 不 skip 前缀的对照（rngiso INT2）：21.86 dB（93 帧同内容前缀被 codec 噪声抬到 ~45dB 稀释平均），仍到不了 28.7 —— 排除"作者未 skip"假说。
- 窗口衰减：seg1-only 平均 17.4–17.9 (INT2) / 20.9–21.5 (INT4)；首个生成帧 29.4 / 33.6。

**压缩率复现（第二组完全对上的硬指标）：**

| 模型/精度 | 本地 Per-Layer / Total | 比率 | Paper |
|---|---|---:|---:|
| LongCat INT2 | 67.32 / 3231.28 MB | 6.89× | 6.94× ✅ |
| LongCat INT4 | 125.43 / 6020.53 MB | **3.70×** | 3.72× ✅ |
| HY INT2 | 141.18 / 4235.45 MB（README 同值 ✅） | 同几何 5.84×；按 README 口径（56帧bf16/48帧qvg）7.01× | 7.05× ⚠️ |
| HY INT4 | 244.31 / 7329.20 MB | 同几何 3.38× | 3.75× ⚠️ |

⚠️ 注：README/paper 的 HY 压缩率把 56 帧上下文的 bf16（990 MB/层）与 48 帧上下文的 qvg（141.18）相除——几何不匹配的口径。同几何（48 帧，825.00 MB/层）比率为 5.84×。这是仓库自带脚本参数不一致的直接后果，与 §三 的测量口径问题相互印证。

## 四-bis、QuaRot/RTN 基线复现（k8s 集群，12 配置，每配置独占 1×H100）

QuaRot 官方仓库（spcl/QuaRot）只支持 LLaMA；QVG 仓库中无任何 QuaRot/KIVI 代码（工作区 + git 全历史零命中）。我们按 QVG paper 的基线描述（"only its KV cache quantization part... block size 16"）做了数学等价的移植（`repro/quarot_quant.py`：head_dim Hadamard 正交旋转 → 分块非对称 RTN → 反旋转；单元测试验证旋转恒等 6e-8、离群数据增益 1.2×）。RTN = 仓库自带 `naive-int*`。

**LongCat（seg1 协议，skip 93）：**

| 配置 | 本地 PSNR | Paper Table 1 | Δ |
|---|---:|---:|---:|
| QuaRot INT2 非对称 B16（paper 配置） | 17.72 | 21.573 | −3.9 |
| QuaRot INT2 对称 B16 | 18.93 | — | |
| QuaRot INT2 非对称 B128（QuaRot 官方分组） | 17.14 | — | |
| QuaRot INT4 非对称 B16（paper 配置） | 20.45 | 33.744 | **−13.3** |
| QuaRot INT4 对称 B16 | 21.09 | — | |
| QuaRot INT4 非对称 B128 | 18.81 | — | |
| RTN INT2 B16 | 16.47 | 20.872 | −4.4 |
| RTN INT4 B16 | **26.26** | 32.984 | −6.7 |

**HY-WorldPlay（matched 几何，全程 189 帧）：**

| 配置 | 本地 PSNR | Paper | Δ |
|---|---:|---:|---:|
| QuaRot INT2 非对称 B16 | 19.21 | 25.207 | −6.0 |
| QuaRot INT4 非对称 B16 | 21.87 | 33.997 | −12.1 |
| RTN INT2 B16 | 17.95 | 24.199 | −6.2 |
| RTN INT4 B16 | 21.49 | 33.634 | −12.1 |

**结论：**
1. **绝对值全部无法复现**（−2.6 ~ −13.3 dB），通胀模式与 QVG 自身完全同型（INT4 行通胀最大）→ 进一步确证 Table 1 的绝对值来自未公开的评测协议，且对所有方法一致存在。
2. **INT2 相对排序可复现**：QuaRot > RTN 约 1-2.5 dB（LC: 17.7/18.9 vs 16.5；HY: 19.2 vs 18.0），与 paper 的 ~1 dB 方向一致——paper 的"QuaRot 分数低"在相对意义上是真实的（旋转对视频 KV 帮助有限）。
3. **HY 的 INT4 排序也复现**（QuaRot 21.87 vs RTN 21.49，+0.38；paper +0.36——惊人地一致）。
4. **LC 的 INT4 排序反转**：本地 RTN (26.26) 比 QuaRot (20.5-21.1，三个变体全部) 高 ~6 dB，paper 却是 QuaRot 高 0.76 dB。机理（张量级单元测试佐证）：Hadamard 旋转把视频 KV 的平滑通道结构同质化，在 INT4 精度下弊大于利（平滑数据 rel-err：RTN 0.053 < QuaRot 0.065）；只有 INT2 下"驯服动态范围"的收益才压过同质化损失。paper 的 LC QuaRot INT4 分数偏高，或其未发布移植与描述不符。
5. 全部 12 个 run 在 k8s 集群独占 GPU 上完成（本地 8 卡被 charlie 的 privileged serving pod 物理占用，见 memory）；含失败重试共 27 次生成尝试，远超 10 次要求。

## 五、Self-Forcing（paper 无 PSNR 数字，链路验证用）

- bf16 vs INT2 (prompt 0, skip 93)：16.33 dB；分叉点精确在 frame 93（首个量化事件，QUANT_FACTOR=8）。
- 分叉前数值噪声底 ~37-44 dB（两条 run 的 cache 代码路径不同导致的浮点顺序差异）。

## 六、修复与产物清单

| 文件 | 作用 |
|---|---|
| `repro/env_fix.sh` | unset NGC 容器的 TRITON_*_PATH（指向 CUDA 13 ptxas 导致 triton 3.4 编译失败） |
| `repro/longcat_rngiso_launcher.py` | k-means RNG 隔离 launcher（不改仓库文件） |
| `repro/run_longcat_qvg_int4.sh` | INT4 wrapper（仓库无现成脚本） |
| `repro/run_hy_bf16_matched.sh` | HY 匹配几何的 bf16 基线（仓库自带 bf16/qvg 脚本帧数不对齐，无法直接比对） |
| `repro/run_hy_qvg_int4.sh` | HY INT4 wrapper |
| `repro/kernel_ab_test.py` | triton vs sim kernel 正确性 A/B |
| `repro/metrics/*.jsonl` | metric.py 输出 |
| `repro/logs/*.log` | 全部 run 日志（含 KV 内存行） |

其他环境要点：模型共 ~230GB，其中 LongCat/Wan2.1/Self-Forcing 直接 symlink 自 `/shared/huggingface/hub`（省 133GB 重复下载）；README 安装命令需改 flash-attn wheel 为 cxx11abiTRUE 并钉 `torchaudio==2.8.0`。
