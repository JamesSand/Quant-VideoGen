# 0720 工作目录 — 总索引

> **新人从这里进**:[REPRODUCE-0720.md](REPRODUCE-0720.md)(全链路指令级复现
> 指南);**可直接 present 的成果报告**见 [report-0720.md](report-0720.md)
> (方法公式 + BPE 推导 + MP100 表 + kernel 速度 + why);交接见
> [HANDOFF-0720.md](HANDOFF-0720.md)。

## 一句话

kernel 三重门与 why 判决双收官,经外部核查二审勘误后全部结论以真口径重立;
与 paper 原文的四类数字差异全部定位到 paper 侧——**方法、账单、机制、横比,
四条线都闭环了**。

## 核心结论四件套

1. **终表**([mp100-table.md](mp100-table.md)):MovieGen 100 随机 prompt
   (seed=42),vs QVG **18 列 = 11 显著胜 + 7 统计平局 + 0 负**;LC
   31.68/0.9370/0.0547,SF VBench 四维全第一,HY 18.77 + SC 94.68。全部数字
   为补零块长 + fp8 元数据的诚实口径(KIVI 的 LC 从虚标 33.47 重赛回 30.55);
2. **kernel 三重门**:① BPE 逐字节审计合规([bpe-audit.md](bpe-audit.md):
   LC 2.3195 / SF 2.3185 / HY cache 2.3256(归一因子入账),全 ≤2.326);② 同输入 encode 快过
   QVG kmeans([kernel-results.md](kernel-results.md):LC 32.5× / HY 1.4× /
   SF 1.0-1.1×,Triton fused decode 与参考数值等价 ≤1 ulp);③ 质量无一列输 QVG;
3. **why 机制判决**([why-budget-pca-wins.md](why-budget-pca-wins.md),只留
   成立假说,每节带"大白话+公式+例子"导读框):真机制 = **残差结构分工**
   ——kmeans 残差≈白噪声,2-bit 格回收率在 LC/SF 全部 16 测点钉死 46-49%
   (教科书白噪声水平)vs 我们 ~76%,格效率差 ~2.2×,chunk 级最终误差我们
   20/24 格更小;H2 通道劫持成立(LC 1.8-2.4×,8/8 跨层稳)、H3 维度诅咒
   成立、H4 排除"没调好"(iters ×50 换 ±0.5dB);证伪/撤回/勘误全部在
   [why-refuted-and-errata.md](why-refuted-and-errata.md)(负结果台账,
   引用正报告须成对看);
4. **paper 差异定位**([paper-diff-plan.md](paper-diff-plan.md)):四类差异
   全部结案且无一指向我们管线——LC baseline 低 ~10dB = paper 实现弱(三点位
   格 + post-RoPE,E1 复现至 ±1-3dB);HY 高 7-12dB = PSNR 窗口口径;VBench
   水位差 = 我们 AQ 重实现比官方低 ~4.7(BC/IQ/SC 与官方 ±0.4 一致);paper
   的 baseline VBench 崩坏用官方打分器也复现不出 = 其视频真坏。QVG 本尊行
   −0.5dB 一致反向背书我们管线。**E1-C 归因矩阵补齐**(kivipost 臂):格子
   ×位置 2×2 分离——满血 KIVI 只挪 post-RoPE 掉 **7.5dB**(30.22→22.74,
   仅比 RTN 高 0.8)、只换三点位格掉 11.4dB,两因子叠加亚可加(地板 ~19dB);
   附带把 KVQuant 位置红利在 LC 上量为 +7.5dB。

## 文档地图

| 文件 | 内容 |
|---|---|
| [REPRODUCE-0720.md](REPRODUCE-0720.md) | 全链路复现指南(环境→生成→评分→三重门→why,含预期数字与已知坑) |
| [mp100-table.md](mp100-table.md) | MP100 定案表 + vs QVG 裁决 + 与 paper 横比头注(源 = ../0718/) |
| [bpe-audit.md](bpe-audit.md) | BPE 逐字节审计(kernel 数 bytes 终审) |
| [kernel-results.md](kernel-results.md) | 速度对决 v13 + fp8 归一化返工记录 |
| [why-budget-pca-wins.md](why-budget-pca-wins.md) | why 判决正报告(成立假说,导读框 + 6 图内嵌) |
| [why-refuted-and-errata.md](why-refuted-and-errata.md) | 负结果台账(H1 原命题证伪、撤回/降级、二审三勘误、历史证伪索引) |
| [method-explainer.md](method-explainer.md) | 方法讲解(三模型逐个,行号可点;头部有 0720 勘误横幅) |
| [paper-diff-plan.md](paper-diff-plan.md) | paper vs 复现四类差异的立案-排查-结案全记录 |
| [kivi-paper-repro.md](kivi-paper-repro.md) | KIVI@LC paper 数字复现台账(chunk 级鉴别矩阵) |
| [report-0720.md](report-0720.md) / [HANDOFF-0720.md](HANDOFF-0720.md) | present 版成果报告(方法/战绩/kernel/why 四段)/ 交接 |

## 代码与数据资产

| 路径 | 内容 |
|---|---|
| `kernel/` | 真实现:`bp_quant.py`(encode,torch.compile)、`bp_triton.py`(fused decode)、`bpe_audit.py`、`bench_speed.py` + `bench_report.json`、pod 清单 |
| `why/` | 判决脚本(`h1_real_path.py`/`h1_ours_path.py`/`h2_multichunk.py`/`h1_h2_compute.py`/`make_figs.py`/`fig6_hy_halves.py`)+ fig1-6 + `h1_h2_data.npz`;`h1_kmeans_sub.py` 已废弃留痕 |
| `chunks/{lc,sf,hy}/` | 三管线真实 dump chunk 各 8 个(why 分析素材) |
| `score_fp8.py` + `aggregate_fp8.py` | MP100 终版评分/汇总入口(0720 诚实口径) |
| `score_e1.py` / `vbench_official.py` / `e2b/` | paper-diff 的 E1 复刻评分 / 官方 VBench 对打脚本与原始结果 |
| `jobs_m1regen.txt` / `jobs_kivipost.txt` 等 | 记账修正后重赛 / E1-C 归因臂的任务清单 |
| `lc_p1_f93_comparison.png` + `lc_f93_grids/` | 可视化素材(f93 对比 + 10 网格;grids 已复制到 ../0721/,HY 6 个 mp4 已移至 ../0721/hy_videos/) |

## 诚实记录(本日三轮)

- **一审(kernel 审计触发)**:数字节抓出 4 个记账 bug(块长回退×2、fp8 饱和、
  fp8 下溢);受污染数字全删,LC/SF 双方含 baseline 全部真实口径重赛(~700 次
  重生成),KIVI@LC 33.47→30.55;
- **二审(外部核查触发)**:三处勘误全部核实修正——kmeans 聚类口径(全局
  64 维块 → per-head 全 D 维真口径,fig2/3/5 重算)、"0.7%"引用错误、单 chunk
  过度概括(全部改 8 chunk 复核);效率差 3-16× 修正为 ~2.2×,H1 证伪反而升级
  为 24/24 全向;fig1 排序论断撤回、SF"无红利"降级。正报告与负结果台账拆分,
  勘误全部留痕;
- **三审(外部工程复审触发)**:五条批评全部核实修复(report-0720 §五)——
  HY BPE 曾因归一因子漏账实为 2.3295 超线(改 pow2-int8 因子后 2.3256 ✓,
  质量不降反微升);QVG 同口径实测 BPE 2.464/2.406/3.320(质心实存 fp32),
  终表已改真实值;"逐位一致"降级为"数值等价 ≤1 ulp";score_fp8 臂名/HY 路径
  修复(旧臂名会静默评到作废目录);why 脚本补 PCA_FP8SIM=1 重跑,结论不变。

## 结转(接 HANDOFF)

0717/0718 头条文档按 0720 终版数字重写;kernel 深度集成(真 cache 路径接入
三管线,解锁 SF 1400f/Q3);论文写作(素材已齐:方法 + 终表 + 速度 + why
判决 + 可视化)。
