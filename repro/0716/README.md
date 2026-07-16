# 0716 工作目录

收尾三件套（report-0716.md / HANDOFF.md / REPRODUCE.md）**待写**（见 repro/README.md 约定）。

## 文档地图（本目录）

| 文档 | 内容 |
|---|---|
| [qvg-evaluation.md](qvg-evaluation.md) | QVG 评测体系两板块梳理：①质量指标（PSNR/SSIM/LPIPS/VBench-IQ，含首帧口径与 LPIPS 怪癖）②Ablation Studies（分辨率/chunk/K/S/B/batch + 我们补的消融） |
| [ssim-lpips-validation.md](ssim-lpips-validation.md) | 首帧口径推广到 SSIM ✓；LPIPS 官方 [0,1] 直喂怪癖发现与复刻；**N4 三指标对 QVG/QVG-Pro 全胜**；LPIPS 口径决策（此后一律 paper 口径） |
| [mse-reduction.md](mse-reduction.md) | Fig.6 量化 MSE 缩减复现（K 10.9×/V 3.6×，定性成立）+ N4 张量级对比（V 侧全层胜 SAS）；附录含 INT2/INT4 全量原始数据 |
| [kernel-speed.md](kernel-speed.md) | 同 chunk 干净 encode/decode 对决：N4 朴素 torch ≈ QVG-100、比 Pro 快 3.2×、**对流式 iters=2 QVG 慢 23×**；含对早前 "2.4×" 说法的勘误 |
| [vbench-repro.md](vbench-repro.md) | VBench A.1 复现（700f 相对差与 paper 精确吻合）+ **N4 第四指标与 QVG 打平**（保真无画质税）+ 3 条上游口径缺口 |
| figs/ | 本日图表 |

## 今天干了什么（含跨 0715/0716 凌晨的连续工作）

1. **PCA-KV 研究闭环**（结果在 [../0715/pca-results.md](../0715/pca-results.md)）：
   Phase-1 七臂 → auto-research 第二轮 **N4 冠军（r=4 双侧 PCA + 非对称残差 B=128：
   31.79 PSNR @ BPE 2.253，双超 QVG）** → OSCAR（arXiv 2605.17757，FutureMLS-Lab）的
   QᵀQ 校准基检验 = **负结果**（减法式方案必须用自协方差；attention-aware 基属于旋转式）
2. **N4 四指标验证全部完成**：PSNR/SSIM/LPIPS 三项大胜 + VBench IQ 持平——质量故事定稿
3. **评测体系全面对齐 paper**：SSIM 首帧口径 ✓、LPIPS 怪癖复刻 ✓、VBench 协议 ✓
   （逐行复刻 imaging_quality，前缀等价性免裁剪）、Fig.6 MSE ✓、kernel 速度干净对比 ✓
4. **两个 OSCAR 的辨析**：撞名项目分清（美团 OScaR=TNI 论文 vs FutureMLS OSCAR=谱旋转
   项目），后者 K/V 双侧协方差旋转（K←QᵀQ、V←SST）的机制厘清
5. **基础设施动荡与恢复**：Weka CSI 集群级故障（版本错配 4.4.4 vs 4.4.12，ENG-91011）
   诊断 + 切换本地 8×H100 完成全部实验；容器重建后的权限/gh/kubectl/ffmpeg 修复；
   `repro/k8s/` 集群操作知识库建立
6. **SF 上游兼容修复**：fake-quant 张量路径布局矛盾（`PCA_SF_STORE_FIX`）、
   滑窗不支持与 BF16@1400 不可复现的确认

## 现状判定（0716 晚讨论定稿——三张卡）

| 卡 | 现状 | 修正过的表述 |
|---|---|---|
| **质量卡** | LC 上 PSNR/SSIM/LPIPS/BPE 全胜；SF 上 VBench IQ **打平**（0.15-0.36，协议噪声内） | VBench 不是"打不过"——无参考指标看不见保真度，parity 的正确读法是"**保真提升不付画质税**"（卖点）。真问题是**覆盖矩阵错位**：三指标在 LC、VBench 在 SF、HY 全空白——没有任何一个模型有完整四指标 |
| **系统卡** | encode 三档：平 QVG-100 / 胜 Pro 3.2× / **败流式 23×**；decode 双方可忽略 | 缺口不是"没 kernel"而是**整条打包存储路径缺席**——fake-quant 存 bf16，QVG 的端到端 −20%、~3× 长度解锁、7× 显存三个系统卖点我们零对应物（VBench 表 N4 卡 777 帧即此因） |
| **可信度卡** | 全部头条 = 单 prompt / 单 seed | 写 paper 必须 MovieGen prompt 套件（paper 同款，`assets/t2v.txt` 入口）；且 QVG 非确定性（σ=0.18）每 prompt 需 n≥3，N4 确定性 n=1——这本身是一张对比卡；HY 是 I2V+pose 协议，文本套件不适用需按其官方设定 |

诚实脚注：张量级 MSE 分解中 K 侧中层 SAS 仍略胜（256 原子字典 > 4 维子空间）——赢面来自 V 侧与非对称残差，机制须如实呈现。

## 还需要解决的问题（按优先级）

1. **多 prompt 复验**：全部头条数字单 prompt/seed——N4 的 +2.9 dB 需要置信区间；
   同时裁决 VBench 长尾分歧（我们 60.5 vs paper 67.3 @1400f，嫌疑=多 prompt 平均）
2. **N4 真 kernel 化**：①位打包+融合 decode（解锁长度/显存故事，现 fake-quant 只能
   777 帧）②流式 encode 追平（对 iters=2 QVG 慢 23×；抓手=低精度协方差 GEMM +
   基跨 chunk 热启动——KV 平稳性已证，子空间应漂得慢）
3. **N4 泛化**：HY 一次未测；SF 只有 VBench 没有参考指标；三模型补齐
4. **质量再挖**（有现成假设）：按深度重分配预算（L29 热点）、K/V 分离调秩
5. **收尾债务**：0715/0716 三件套；ISSUE_DRAFT 更新发送（累计 5+ 条上游问题：
   B=128 kernel bug、SF fake 路径矛盾、滑窗缺失、LPIPS 怪癖、BF16@1400 不可复现）
6. **挂起计划**：W8A8+KV2（重启时 KV 侧应换 N4）；集群 Weka 恢复确认 + 黑名单赦免
   （099/079/118/052 是被集群级故障冤枉的）
