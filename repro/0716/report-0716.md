# Report 2026-07-16：评测体系考古与勘误、六格矩阵定稿、N4 三模型画像修正

> **一句话**：把 PSNR/SSIM/LPIPS 评测体系彻底对齐 paper 并勘误（SSIM 实现、HY 协议、
> paper 笔误各一处），六格覆盖矩阵 (LC/SF/HY × INT2/INT4) 基本填满；我们的方法
> PCA-KV N4 的画像修正为 **LC 大胜 / SF 打平 / HY 落败 3.1 dB**（早先"无一处输"撤回）。
> 名词：**N4** = 本项目新方法（per head×chunk：均值 + top-4 PCA 基 + 2-bit 系数 +
> 2-bit 非对称残差 B=128，K/V 同构；BPE 2.253 fake / 2.3125 real vs QVG 2.326），
> **QVG** = paper 原方法（k-means 语义平滑 + 渐进残差量化）。

## 一、三项勘误（全部影响既有结论，须知）

### 1. SSIM 实现错误（我们的 bug）

`sf_ref_metrics.py` 曾用**整帧单窗"全局 SSIM"**（一个 mean/var），读数严重虚高；
paper 的 `metric.py` 是 **11×11 avg_pool 局部窗**。示例：HY 同一窗口 0.966（错）→
0.761（对）。脚本已修，受影响数组补 `ssim_paper` 键；`precompute_arrays.py` 一直正确
（原始 REPORT 的数组不受影响）。PSNR/LPIPS 无恙。**此后 SSIM 一律 paper 实现。**

### 2. HY 评测协议（paper 的坑）

- paper §5.1 只给 LC 指定了相似度协议（首个生成 chunk）；**HY 的协议未指定**，
  字面 "long-generation setting"（全程均值）与物理事实矛盾——我们全程实测
  18.77/0.458/0.374（paper 报 29.174/0.882/0.094），且确定性的 N4 全程同样 ~18 dB，
  证明是自回归混沌而非管线问题。
- **HY 的逐帧结构 = 平台 + 断崖**：帧 1-28 平台 ~35 dB；帧 29 断崖（恰为 pose
  `w→s` 首次回访点——需从量化 memory 检索旧内容的时刻）；崖后 14-18 dB 永不回归
  （吸收态：六段 pose 的五个切换点中只有第一个产生断崖，其余全平）。
- 旧 [23,36) 窗口横跨断崖，"匹配 paper"是两段混合平均的巧合——作废。
- 官方脚本无法构成对照：`run_bf16.sh` 与 `run_qvg.sh` 配置不配对
  （memory 56/52 vs 48/44、14 vs 12 chunks、pose 7 段 vs 6 段，帧数都不同）。
- **定稿（两段协议，0717 拍板）**：drop 前 [1,断崖) 与 drop 后 [断崖,末)
  分别报三指标 + 断崖帧位置（断崖 = 首个 PSNR<28 帧）。
- 已起草上游 issue 询问作者 HY 帧范围（[issue-draft-hy-eval.md](issue-draft-hy-eval.md)，可直接发）。

### 3. paper §5.2 chunk 尺寸笔误（证据闭环）

paper 称 HY/SF chunk 为 "12 and 16 frames, respectively"；代码实际 **HY=16 帧**
（`pred_latent_size=4`）、**SF=12 帧**（`num_frame_per_block=3`×4 帧/latent）——两数
交叉对调。且 12 帧 chunk 在发布代码里**结构性跑不通**（`generate.py:205` 断言
memory−context=pred 与 `select_mem_frames_wan` 的 4 帧块分配在 pred=3 时恒冲突），
作者用这份代码也跑不了 HY@12——笔误实锤，chunk 粒度嫌疑排除。

## 二、六格矩阵（本日定稿口径下的全部数字）

格式 PSNR / SSIM(paper 实现) / LPIPS(paper 口径)。协议：LC=首生成帧 f93；SF=onset 帧
（paper 无 SF 行，自建对比）；HY=两段协议。

### INT2

| 模型 | QVG（我们） | QVG（paper Table 1） | N4（我们） |
|---|---|---|---|
| LC | 28.73 / 0.9033 / 0.089 | 28.716 / 0.909 / 0.065 → **PSNR+SSIM 双 match** | **31.79 / 0.9424 / 0.067**（+3.1 dB，三指标全胜） |
| SF | 38.65 / 0.9736 / 0.041 | —（无 SF 行） | 38.52 / 0.9730 / 0.043（打平） |
| HY drop 前（断崖帧） | **35.11 / 0.9655 / 0.0544**（29） | 29.174 / 0.882 / 0.094¹ | 31.98 / 0.9439 / 0.0770（29）→ **输 3.1 dB** |
| HY drop 后 | 15.79 / 0.370 / 0.432 | | 15.62 / 0.380 / 0.435（打平——噪声地板） |

¹ paper 的 HY INT2 值落在我们两段之间，形状 = 跨崖平均（我们 [20,32) = 31.1/0.882/0.099，
SSIM/LPIPS 与其精确重合）；最可能解释 = 作者 run 分岔更晚的全程均值。待 issue 裁决。

### INT4

| 模型 | QVG（我们） | QVG（paper） | 判定 |
|---|---|---|---|
| LC | 33.75 / 0.9535 / 0.056 | 37.141 / 0.978 / 0.024 | −3.39，在 QVG 已知 ±1-2 dB 运行间散布内（0713 n=3：34.36±1.28） |
| HY drop 前（断崖 35） | 35.14 / 0.9640 / 0.0500 | 34.454 / 0.954 / 0.051 | **三指标精确吻合（+0.7/+0.010/−0.001）——INT4 列复现成立** |
| HY drop 后 | 19.72 / 0.664 / 0.222 | | INT4 断崖晚 6 帧且崖后温和（SSIM 0.66 vs 0.37）——位宽推迟/减缓分岔 |
| SF | 空（需生成 195 配置 INT4） | — | N4 无 INT4 档（配置待定义） |

### 第四指标 VBench Image Quality（SF，逐行复刻官方 imaging_quality）

700f 处 BF16−QVG 相对差 **1.10 vs paper 1.04（精确吻合）**；N4 与 QVG 打平
（−0.15~−0.36，噪声内）——无参考指标看不见保真度，读作"**保真提升不付画质税**"。
长尾分歧（我们 60.5 vs paper 67.3 @1400f）待多 prompt 裁决。详见 [vbench-repro.md](vbench-repro.md)。

## 三、N4 三模型画像（修正版）

**LC 大胜（+3.1 dB、SSIM/LPIPS 同向、BPE 更低）/ SF 起点打平 / HY drop 前落败 3.1 dB。**

- 早先"HY 打平、无一处输"= 跨崖窗口 + 错误 SSIM 的双重伪影，撤回。
- HY 差距**不是**调参问题：r=8、128 维半头分裂（`PCA_SPLIT_D=128`）两个补救 arm
  平台均值分毫不动（31.98/31.99/31.98）——双双证伪。结合张量级发现（K 侧中层
  SAS 字典本就胜 4 维子空间），下一步 = 按层×头×K/V 误差分解定位差距来源。
- 断崖帧 N4 与 QVG 相同（29）——回访鲁棒性打平，输的是平台保真。
- OSCAR（FutureMLS 同事项目）QᵀQ 校准基已验证为负结果：减法式方案须自协方差，
  attention-aware 基属旋转式（[../0715/pca-results.md](../0715/pca-results.md) §结果三）。

## 四、系统侧现状（无新实验，勘误后口径）

encode 三档：N4 朴素 torch ≈ QVG-100 / 胜 QVG-Pro 3.2× / **败流式 iters=2 QVG 23×**；
decode 双方可忽略；早前 "N4 快 2.4×" 为脏管线计时，已勘误（[kernel-speed.md](kernel-speed.md)）。
真差距是整条打包存储路径缺席（fake-quant 存 bf16 → 长度/显存/端到端三个系统卖点
零对应物）。实施计划 M0-M4 见 [n4-int2-impl-plan.md](n4-int2-impl-plan.md)（待讨论）。

## 五、遗留问题（按优先级）

1. **发 issue** 问 HY 帧范围（草稿就绪）；作者一句话可定 INT2 列复现性
2. **HY 差距归因**：层×头×K/V 误差分解（调参已证伪，须换刀）
3. **多 prompt 复验**：全部头条仍是单 prompt/seed；MovieGen 套件 + QVG n≥3
4. **N4 kernel 化**（M1 位打包 → M2 融合 decode → M3 基准 → M4 流式热启动）
5. 空格：SF×INT4、N4-INT4 档定义；W8A8 计划挂起；Weka 集群恢复确认 + 节点赦免

## 附：本日文档索引

[metric-matrix.md](metric-matrix.md)（六格约定与现状）· [hy-ref-metrics.md](hy-ref-metrics.md)
（HY 考古+两段协议全文）· [sf-ref-metrics.md](sf-ref-metrics.md) ·
[ssim-lpips-validation.md](ssim-lpips-validation.md) · [vbench-repro.md](vbench-repro.md) ·
[mse-reduction.md](mse-reduction.md) · [kernel-speed.md](kernel-speed.md) ·
[qvg-evaluation.md](qvg-evaluation.md) · [issue-draft-hy-eval.md](issue-draft-hy-eval.md) ·
[n4-int2-impl-plan.md](n4-int2-impl-plan.md)
