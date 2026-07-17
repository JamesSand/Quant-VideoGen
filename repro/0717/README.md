# 0717 工作目录

昨日总况读 [../0716/report-0716.md](../0716/report-0716.md)；接手须知读
[../0716/HANDOFF.md](../0716/HANDOFF.md)。本页 = 当前待办（已解决项已清除：
issue 已处理、W8A8 挂起项、上游 issue 池、集群善后均已了结）。
（收尾三件套 report-0717/HANDOFF/REPRODUCE 按惯例日末补。）

## 待办一：HY −3.1 dB 归因

**问题**：HY 上 N4 的 drop 前段（帧 1-28）输 QVG 3.1 dB（31.98 vs 35.11），是六格
矩阵里 N4 唯一落败的格子。LC 上 N4 反而赢 3.1 dB——同一算法在两个模型上方向相反，
说明差距来自 HY 的 KV 结构本身，值得搞清机制。

**已排除的假设**（0717 凌晨，两个 arm 实测无差异）：
- r=8（秩翻倍）：31.99，不动
- `PCA_SPLIT_D=128` 半头分裂（与 LC 128 维同构）：31.98，不动
→ **瓶颈不在子空间秩/粒度**。

**下一步三刀（按顺序做，每步都可能提前定位）**：
1. **层×头×K/V 误差分解**：在 HY 的真实 KV 上（fake-quant 路径里顺手 dump 每
   层/头/K/V 的重构 MSE，N4 vs QVG-SAS 各一份），找出差距集中在哪——若集中在
   K 侧中层，与 LC 张量级发现一致（SAS 的 256 原子字典在 K 中层本就胜 4 维子空间，
   见 [../0716/mse-reduction.md](../0716/mse-reduction.md)），则方向明确：K 侧换刀。
2. **K/V 不对称预算**：既然 N4 的赢面历来在 V 侧，试 K 用更强配置（如 K 残差 3-bit
   / V 残差降 2-bit 平账，或 K 用小字典+PCA 混合）、V 维持 N4——BPE 持平下重分配。
3. **残差位宽/块尺寸扫描**：HY 的 256 维行有 2 个 B=128 块，块内动态范围可能比
   LC 更花——扫 B∈{64,128,256}×残差 bit∈{2,3}，看 HY 是否偏好不同残差配置。

**工具**：`pca_quant.py`（加 dump 开关即可）、既有数组 `sf_hy_n4*.npz`、
`pod_run_pca.sh`。每 arm ~6 min（HY 189 帧单卡）。读数一律两段协议。

## 待办二：六格矩阵补空

现状见 [../0716/metric-matrix.md](../0716/metric-matrix.md)。
**注意：SF 已移出参考三指标矩阵（用户 0717 决定，无条件前缀→onset 落近无损区、
判别力低且无 paper 锚点；SF 质量只走 VBench IQ）**，矩阵为 (LC×HY)×(INT2×INT4) 四格。
剩两个空格：

1. **N4 的 INT4 档定义**（研究决策，先讨论再跑）：`pca_quant.py` 残差写死 2-bit。
   候选配置 = coef 4-bit + 残差 asym 4-bit B128，估算 BPE≈4.5 vs QVG INT4 的 4.30
   ——略吃亏，可选补偿：r 降到 2（coef 开销减半）或残差 B=256。定义完成后 LC/HY
   两格一天内可填满（每格一次生成+一次评测）。
   **决策点**：INT4 档要不要追求 BPE 也压过 QVG（改动大），还是先出质量数字（改动小）？
2. **QVG-Pro 的 paper 口径 LPIPS**：旧 Pro 视频已清理、旧 LPIPS 是废弃口径。
   需重生成 Pro（S=4,B=16，~25 min）后用 `sf_ref_metrics.py` 补测 f93 三指标
   ——顺手把 Pro 的 SSIM 也落成 paper 实现版。

## 待办三：多 prompt 复验（可信度卡的核心）

**为什么必须做**：目前全部头条数字 = 单 prompt（滑板手/湖桥）× 单 seed。写 paper
需要：N4 vs QVG 的差距带置信区间；QVG 非确定性（k-means atomic_add，σ≈0.18、
长尾 ±1-2 dB）要求每 prompt n≥3，N4 确定性 n=1 即可——"确定性"本身也是一张对比卡。

**设计（草案，开跑前确认）**：
- **prompt 套件**：paper 同款 = MovieGen benchmark 套件（Self-Forcing 官方设定，
  `experiments/Self-Forcing/prompts/MovieGenVideoBench_extended.txt`；LC 入口
  `assets/t2v.txt`）。取前 10 条（或分层抽 10 条）。
- **矩阵**：LC×{QVG n=3, N4 n=1}×10 prompts ≈ 40 次生成（LC 一段 ~15 min 单卡，
  8 卡并行一晚跑完）；SF 同理但用 onset 协议；HY 是 I2V+pose 协议、文本套件不适用，
  维持单场景多 seed（QVG n=3）。
- **产出**：N4−QVG 差距的 mean±std（分模型）；VBench 长尾裁决（我们 60.5 vs paper
  67.3@1400f 的分歧是否由多 prompt 平均解释——10 prompts 各测 1400f VBench IQ）。
- **决策点**：10 prompts 够不够？要不要包含 QVG-Pro 臂（+10 次生成）？

## 待办四：N4 真 kernel 化（M0-M4）

**为什么必须做**：现在 N4 是 fake-quant（cache 实存 bf16）——QVG 的三个系统卖点
（端到端 −20% 显存、~3× 长度解锁、7× KV 压缩）我们全是零对应物；SF 上 N4 因此
卡在 777 帧。计划全文见 [../0716/n4-int2-impl-plan.md](../0716/n4-int2-impl-plan.md)。

- **M0 风险先探**（半天）：FP8-scale 的质量代价未测——真实现里 scale/zp 若用 FP8
  存储（BPE 账本 2.3125 的前提），需先在 fake 路径里模拟 FP8 精度确认不掉点。
- **M1 位打包存储**（核心，~2 天）：coef/残差按 2-bit 打包进 uint8 张量 + 元数据布局
  ——这一步就解锁长度/显存故事（可立即重测 SF 极限帧数 vs QVG 的 2397 帧）。
- **M2 融合 decode kernel**（~2 天）：X̂ = mu + coef·V₄ᵀ + res 一次瘦 GEMM+逐元素加，
  Triton 实现；decode 侧两方本来都可忽略，目标是不引入回退。
- **M3 基准**（1 天）：encode/decode/端到端三档 vs QVG 全配置，干净计时协议
  （沿用 [../0716/kernel-speed.md](../0716/kernel-speed.md) 的同 chunk 口径）。
- **M4 流式 encode 热启动**（风险项，可裁）：对流式 iters=2 QVG 慢 23× 的追赶——
  低精度协方差 GEMM + 基跨 chunk 热启动（KV 平稳性已证，子空间应漂得慢；若漂移
  假设不成立则此项止损）。
- **顺序建议**：M0 → M1 → 立即重测 SF 长度（最便宜的新故事）→ M2 → M3；M4 视
  M3 结果决定。

## 今日实验记录

（待添加）
