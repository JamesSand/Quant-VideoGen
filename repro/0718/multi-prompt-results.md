# Multi-prompt campaign 结果（基线轮 + 优化循环进行中）

> 依据 [multi-prompt-plan.md](multi-prompt-plan.md)。生成台账 `logs/ledger.txt`，
> 逐帧指标缓存 `npz/`，自动统计表 [stats-output.md](stats-output.md)（stats.py 生成）。
> 状态：**基线轮完成，SF/HY 的 VBench 有欠账 → 按 goal 进入算法优化循环（sweep 1 在跑）**。

## 执行偏差记录（诚实条款）

1. **QVG n=3 = 同 seed 三重复**（plan 原文写 seed 0/1/2 是设计错误：参考型指标必须与
   BF16 参考同 seed；QVG 的非确定性来自 k-means atomic_add，同 seed 重复即可捕获）；
2. **SF 帧数语义修正**：`--num_output_frames` 是 latent 帧（%3==0），180 latents=717px，
   VBench 取 700 前缀窗（=原 headline 口径）；
3. **HY 为 173 帧**（num_chunk 11 = 44 latents；与旧 headline 189f 差一个 chunk 配置，
   campaign 内三臂严格同长配对，绝对值不与旧 headline 横比）；
4. **Q3（1400f）结构性受阻・双边对称**：360-latent 下 QVG 臂与 PCA 臂**全部**死于
   同一个上游 raise（`ValueError: KV cache size is too small`，pre-RoPE 路径无滑窗
   实现）——本代码库 195-latent 窗口下 1400f 两边都不可测，此项裁决改判"无法在
   released code 复现 paper 的 1400f 设定"，待 kernel 化（M1）解锁；
5. LC 指标窗口：`first_div` 被 ~60dB 编解码噪声污染（帧 0 起），协议固定为
   **f93 单帧**（headline 门）+ 生成区均值 [93:]（辅助）；HY 固定为 **[13:] 均值**
   （去掉 BF16 上下文 chunk）。

## 基线轮裁决（PCA final 配置 vs QVG，配对统计）

| Q | 结果 | 数据 |
|---|---|---|
| **Q1-LC** | **✓ 胜**（参考三指标） | f93 ΔPSNR +1.40±2.79，胜 9/10，符号检验 p=0.011；ΔSSIM/ΔLPIPS 同 9/10 p=0.011 |
| **Q1-SF** | **✗ 败**（VBench700） | Δ −1.46±2.57，胜 3/10，p=0.945——**旧 headline（71.61>70.41）不泛化** |
| **Q2-HY** | **参考指标胜、显著性部分达标** | ΔPSNR +0.176（胜 4/5，p=0.19，区间含 0）；**ΔSSIM +0.0176 胜 5/5 p=0.031 显著**；ΔLPIPS 胜 4/5。head line 改写成立："SSIM 显著领先，PSNR 方向为正但 n=5 未过显著线" |
| **Q3** | **⊘ 双边受阻**（见偏差 4） | QVG 10/10 raise、PCA 10/10 raise |
| **Q4-LC** | ✓（VBench 持平微胜） | Δ +0.04，胜 6/8（两平），p=0.14；与 BF16 无差 |
| **Q4-HY** | **✗ 败**（VBench） | Δ −1.16，胜 0/5——但参考指标同时在赢（见解读） |

单 prompt 明细表见 [stats-output.md](stats-output.md)。LC p9 备注：PCA 仅 f93 单帧
瞬态 −6.2dB（f99 即恢复至 QVG 同级），生成区均值差距远小——但按预注册协议照记。

## 关键解读：VBench(MUSIQ IQ) 上的"忠实悖论"

SF p1 的 700f 读数：BF16=62.46，QVG=66.32（**高于无损参考 +3.9**），PCA=63.78。
QVG 的 k-means 字典伪影产生 MUSIQ 偏好的锐化效果，得分**超过 BF16 本身**；我们的
方法忠实于 BF16（参考指标全线胜的同一性质），于是被 BF16 自身的 IQ 衰减封顶。
HY 的 VBench 0/5 与参考指标 5/5 同时发生，是同一现象的第二例。
**协议以 VBench 高者为胜，照此执行优化**，但此解读写入定案文档。

## 优化循环 sweep 1（在跑，全部 BPE < 2.326 合法）

| 模型 | 变体 | BPE | 动机 |
|---|---|---|---|
| LC | r=6 asym B128 | 2.28 | p9 瞬态 = 坏 basis 情形，加秩兜底 |
| SF | asym B128 r4 | 2.257 | 4 电平格（旧 N4 默认）多 prompt 复验 |
| SF | ternary B64 r6 | 2.29 | 更多减法 |
| SF | ternary B64 r4 + V=mean | <2.26 | V 不做低秩平滑，保锐度 |
| SF | ternary B64 r6 + V=mean | ~2.27 | 组合 |
| HY | K9:0 / V9:0 | 2.320 | 已知参考指标合格（18.824/0.4907），测 VBench |
| HY | K9:0 / V0:0（V 仅均值） | <2.32 | V 保锐度假说 |

判定规则不变：配对胜率 + 符号检验；参考指标不得回退到输 QVG。
