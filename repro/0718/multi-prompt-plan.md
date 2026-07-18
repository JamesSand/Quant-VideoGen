# Multi-Prompt 复验计划（0718 主任务；待办三落地版，开跑前经用户确认）

> 目的：把 Budget-PCA 的三关头条数字从 n=1 升级为带置信区间的结论——这是
> [../0717/final-method-results.md](../0717/final-method-results.md) 诚实边界里
> 唯一挡在"定案"前面的项。配置**冻结**为 final 版，不允许按 prompt 重调
> （预注册原则）。0718 新增（用户指示）：**VBench 打分扩展到 LC 和 HY**，
> 三模型全部出 VBench 结果，不再是 SF 专属。

## 一、要回答的四个问题（判定标准前置）

| # | 问题 | 判定标准 |
|---|---|---|
| Q1 | LC/SF 的领先（+3.06 dB / +1.20 VBench）多 prompt 下是否保持 | 逐 prompt 配对差 mean±std；胜率 ≥ 8/10 且配对符号检验方向一致 → 领先定案 |
| Q2 | HY 的 +0.13 dB（单次在 ±0.2 混沌噪声内）是否真实 | 多 seed 配对差的 mean ± std/√n 区间不含 0 → 升级为真领先；含 0 → 头条改写为"PSNR 持平 + SSIM/LPIPS 领先"（后两者单次已超噪声） |
| Q3 | SF 1400f VBench 60.5 vs paper 67.3 的分歧是否由多 prompt 平均解释 | 10 prompts 的 1400f VBench IQ 均值落在 67±2 → 分歧解释为 prompt 方差；仍 ≤62 → 上游问题立案 |
| Q4 | **LC/HY 的 VBench（新增）**：无参考质量分上 Budget-PCA 是否同样 ≥ QVG | 三模型 VBench 均给出 (BF16 / QVG / ours) 三行；ours ≥ QVG 即过，同时报 vs BF16 的保真差 |

## 二、套件与抽样（可复现）

- **池**：`../0717/MovieGenVideoBench_extended.txt`（1003 条，Self-Forcing 官方套件，
  paper 同源）；
- **抽样**：均匀间隔取 10 条——**行号 1, 101, 201, …, 901**（确定性、无挑选嫌疑、
  内容自然分层：人物/动物/风景/科幻等）；
- **seed**：QVG 臂每 prompt n=3（seed 0/1/2；k-means atomic_add 非确定性 σ≈0.18、
  长尾 ±1-2 dB）；Budget-PCA 臂 n=1（确定性，同 seed 复跑逐位一致——**"确定性"
  本身单列一张对比卡**）；BF16 参考臂 n=1（seed 0）。

## 三、逐模型矩阵与算力预算

### LC（frame-93 三指标 + VBench，参考=同 prompt BF16 续写）

| 臂 | n/prompt | 生成次数 | 指标 |
|---|---|---|---|
| BF16 参考 | 1 | 10 | （参考基准）+ VBench |
| QVG（k=64, iters 官配） | 3 | 30 | PSNR/SSIM/LPIPS@f93 + VBench |
| Budget-PCA（r=4, asym B128） | 1 | 10 | 同上 |

50 次 × ~15 min 单卡 ≈ **12.5 GPU·h**（8 卡并行一晚）。入口 `assets/t2v.txt` 换行
注入 prompt，复用 `pod_run_paperspeed.sh` 的 LC 分支参数化。

### SF（VBench700，无参考；顺带 Q3 的 1400f 臂）

| 臂 | n/prompt | 700f | 1400f（仅 IQ，Q3 用） |
|---|---|---|---|
| QVG（naive-int2 B64） | 3 | 30 | 10（seed 0） |
| Budget-PCA（r=4, ternary B64） | 1 | 10 | 10 |
| BF16 上限参照 | 1 | 10 | — |

生成 ~50+20 次 × ~25 min ≈ **30 GPU·h**。
**强制卫生**：命令必须含 `--quant_type naive-int2 --quant_block_size 64 --use_ema
--local_attn_size 195`，日志验证 hijack>0 + 事件数，否则该 run 作废（N12 教训）。

### HY（文本套件不适用——I2V+pose 协议，改多 seed 配对；+VBench）

场景维持湖桥（`assets/hyworld.png` + 6 向 pose cycle），变异轴 = 生成 seed 0-4：

| 臂 | n | 生成次数 | 指标 |
|---|---|---|---|
| BF16 参考 | 每 seed 1 | 5 | （参考基准）+ VBench |
| QVG | 每 seed 1（自身非确定性已含在内） | 5 | 全程 PSNR/SSIM/LPIPS + VBench |
| Budget-PCA（K=9:0/V=8:1, asym B128） | 每 seed 1 | 5 | 同上 |

15 次 × ~50 min ≈ **13 GPU·h**。配对方式：同 seed 的 (ours−QVG) 全程三指标差，
n=5 做符号检验 + mean±sem。固定 `--memory_frames 48 --temporal_context_size 44`。

### VBench 打分（三模型统一，新增）

全部落盘视频（LC 50 + SF 60 + HY 15 = 125 段）跑同一套 VBench 管线（与 SF 已有
VBench700 打分脚本同款、同维度集），一次批量 ≈ **10 GPU·h**。注意：LC/HY 视频
帧数/分辨率与 SF 不同，VBench 绝对值**只做臂间横比（同模型内 BF16/QVG/ours）**，
不跨模型比较、不与他文横比（与 LPIPS 口径纪律同理）。

**总算力 ≈ 75 GPU·h**，k8s 自开 pod（repro/k8s + turboskill 授权，Weka ENG-91011），
8×H100 两晚可完成。

## 四、执行阶段

1. **Phase 0 冒烟**（半天）：prompt#1 全臂各 1 次 + 三模型视频各过一遍 VBench
   管线，验证 prompt 注入、hijack 计数、指标管线（LPIPS paper 口径 [0,1] 直喂
   vgg；SSIM 用 paper metric.py 11×11 avg_pool）端到端通；
2. **Phase 1 LC**（1 晚）→ Q1-LC + Q4-LC 判定；
3. **Phase 2 SF 700f + 1400f**（1 晚）→ Q1-SF、Q3 判定；
4. **Phase 3 HY 多 seed**（与 Phase 2 并行，卡够即同晚）→ Q2 + Q4-HY 判定；
5. **Phase 4 汇总**（半天）：`multi-prompt-results.md`——逐 prompt 明细表、
   mean±std、胜率、符号检验、四个 Q 的裁决；同步改写
   final-method-results.md 诚实边界与 report 战绩表（加置信区间列 + VBench 列）。

## 五、预注册的诚实条款

- 配置冻结：三模型配置 = final 版，任何 prompt 上单独调参 → 该结果只能标
  "diagnostic"，不进主表；
- 全部 run 留 log + 视频落 `results/multiprompt/{lc,sf,hy}/{arm}/{prompt或seed}/`，
  npz/视频不进 git，明细表进 git；
- 失败/异常 run（OOM、hijack=0、NaN）如实计入明细表并标注原因，不静默重跑换数；
- 若 Q1 胜率落在 6-7/10 灰区：扩到 20 prompts（行号 51, 151, …, 951 追加），
  不允许只对失利 prompt 复跑。

## 六、决策点（已定）

- **10 prompts 起步**：够出方向与量级；灰区自动扩 20（见上条）；
- **QVG-Pro 臂**：本轮不做（+50 次生成；且其 LPIPS 口径复测另案在待办队列）。
