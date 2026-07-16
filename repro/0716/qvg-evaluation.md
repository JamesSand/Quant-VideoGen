# QVG 的 Evaluation 体系梳理：指标、benchmark、口径与缺口

来源：paper（arXiv:2602.02958 v5，`papers/2602.02958v5.pdf`）+ codebase 考古
（`repro/0714/details-0714.md`）+ 我们的复现实测（0713-0715）。

## 1. 质量指标（Table 1 主表）

三个指标全部是**对参考的保真度**（量化版生成 vs 同 seed 的 BF16 参考视频），
不是无参考质量评测：

| 指标 | 含义 | 复现时确认的关键口径 |
|---|---|---|
| **PSNR** | 逐像素保真 | **首帧口径**：paper 的数字对应第一张生成帧（LongCat = 全局 frame 93，93 帧共享 init 前缀之后），不是全视频平均——全视频 PSNR 被混沌放大拖到 12-20 dB 且不可复现，这是 0713 复现的核心结论 |
| **SSIM** | 结构相似度 | 结构性指标可与 paper 对上 |
| **LPIPS** | 感知距离（VGG 特征） | repo 自带实现 |

官方实现：`experiments/LongCat/longcat_video/utils/metric.py`——独立 CLI（逐帧
PSNR + LPIPS(vgg)，tabulate 打表），**不被任何主流程调用**，是手动工具。
我们的对应物：`repro/backup/scripts/precompute_arrays.py`（多了 SSIM 和逐帧数组落盘）。

## 2. 压缩率（Table 1 第二轴）

Compression Ratio vs BF16。账目**诚实**：残差码 + FP8 scale + uint8 索引 + bf16 质心表
（质心按 chunk 分摊）全计入（paper Fig 7a 有分解）。我们的公式
`BPE = r + 8/B + S·8/128 + S·(256·128·16)/(N·128)` 与 README 实测显存逐字节吻合。
两个小瑕疵（0714 对账）：①Table 1 的 6.94× 隐含 ~35k token 的 chunk，发布配置实际
29,640 → 6.88×（高报 0.8%）；②附录 Table 5 的 K 扫描漏记 scale 项。

## 3. 速度评测

- **唯一口径 = 端到端生成时间的开销百分比**（§5.3：BF16 与 QVG 各跑一遍完整生成）：
  LongCat +2.1% / HY-World +1.5% / Self-Forcing +4.3%；k-means 成本计入、每 chunk 压缩一次
- **唯一绝对秒数**：附录 C Table 6——SF 180 帧、batch 1、H100：端到端 43s、QVG 额外
  0.74s、开销 1.7%（附录 Table 7 为 batch 1/2/5 扫描）
- **全文没有任何 kernel 级 benchmark**（无 GB/s、无 TFLOPS、无逐 op 计时）
- 内部矛盾：SF 的开销 §5.3 说 4.3%、Table 6 说 1.7%、Table 4 按 chunk 1.3-3.3%，未调和
- 我们按同口径实测**方向反转**：三模型全部负开销（LC −19.9% / SF −5.0% / HY −30%
  稳态口径），加速来自 KV cache 搬运量缩减（`repro/0714/details-0714.md` §3.5）
- 换算陷阱：SF 发布脚本 `num_output_frames=180` 实际生成 717 帧（180 latent），
  是 Table 6 "180 frames" 的 4 倍工作量

## 4. Benchmark 套件与测试设置

| 项 | 设置 |
|---|---|
| 模型 | LongCat-Video 13.6B / HY-WorldPlay 8B / Self-Forcing (Wan) 1.3B，均 480p |
| prompt | MovieGen prompt 套件（经 Self-Forcing 官方设定，paper p.7 脚注）；repo 内 `assets/t2v.txt` |
| 配置 | QVG（S=1/B=64/K=256）与 QVG-Pro（S=4/B=16）两档；基线 RTN/KIVI/QuaRot（paper 私有移植，疑为对称变体——0713 复现：其 QuaRot 21.57 ≈ 我们的对称版 21.42，而正确非对称版 30.38） |
| 消融 | chunk 大小（附录 Table 4：6/12/24 帧）、质心数 K（附录 Table 5：64-512）、PRQ 阶段数（Fig 5c 逐阶段 MSE）、720p 质量（Table 3，无计时） |
| 硬件 | H100 + CUDA 12.8（主）；RTX 5090/4090 侧记 |

**paper 没有用的东西**：VBench（全文 0 次提及；repo 里 `--prompt_source
image_to_video_vbench` 只是上游 LongCat 遗留的死选项，`prompt_loader.py` 选它直接
raise）、FVD、任何无参考质量指标、多 seed 方差、多 prompt 统计。

## 5. 评测注意事项（我们补出的口径缺口）

1. **首帧 vs 全视频的歧义**：paper 未写明 PSNR 是首帧口径——直接按全视频复现必然对不上。
2. **单 prompt / 单 seed**：所有质量数字无方差报告；QVG 自身运行间 σ=0.18
   （k-means 质心初始化无种子，0713 方差研究），Table 1 的 0.1-0.3 dB 级差距在噪声内。
3. **基线记账偏松**：KIVI 本为非对称（应付 zero-point），paper 按无 zero-point 的
   6.40× 记账。
4. **速度测量未说明** warmup 次数、prompt 数、"QVG extra cost" 的统计范围（我们实测
   同款 cuda-event 计时得 1.32s/chunk，折算比其 0.74s 高 ~6×，无法对齐）。
5. **B=128 不可测**：发布 kernel 的 autotune bug 使 quant_block_size=128 直接崩
   （已修，提交 8b81883）——paper 的 B 消融空间受此隐性限制。

## 6. 我们复现采用的评测协议（对照）

- 质量：**frame-93 首帧 PSNR** vs `results/longcat/bf16/1-0/segment_1.mp4`（LongCat
  单段续写，seed 0，prompt_idx 1）；非确定性方法 n=3 报 mean±std
- 压缩：K/V 合账 BPE（公式见 §2），同预算档位对比
- 速度：同 GPU 顺序跑、日志内生成计时（墙钟含加载缓存效应，0714 的 HY 教训）
