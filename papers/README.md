# Papers 索引

本目录收录项目相关论文 PDF。添加新论文时：PDF 以 arXiv id 命名放入本目录，并在下表加一行。

| arXiv | 标题 | 作者/机构 | 与本项目的关系 | 文件 | 相关本地文档 |
|---|---|---|---|---|---|
| [2602.02958](https://arxiv.org/abs/2602.02958) (v5) | **QuantVideoGen: Semantic-Smoothed Ternary KV-Cache Quantization for Long Video Generation**（本 repo 的论文） | QVG 作者团队 | 主复现对象：Table 1 PSNR 复现、QuaRot 基线移植、BPE 对账、速度复现全部针对它 | `2602.02958v5.pdf` | `repro/backup/REPORT.md`、`repro/0713/report-0713.md`、`repro/0714/report-0714.md` |
| [2605.19660](https://arxiv.org/abs/2605.19660) (v1) | **OScaR: The Occam’s Razor for Extreme KV Cache Quantization in LLMs and Beyond** | Su et al.（清华 / **美团 LongCat Team** / HKU 等，2026-05） | 极低比特 KV 量化（LLM/多模态）：诊断 per-channel 范式的瓶颈为 **Token Norm Imbalance (TNI)**（共享量化参数跨越 norm 差异大的 token 组时误差被系统性放大），方法 = Canalized Rotation + Omni-Token Scaling，附 CUDA kernel；INT2 近无损，decoding 最高 3.0× 加速。与我们的关联：①出自 LongCat 团队（我们三模型之一的作者方）；②TNI 视角与我们的 b-sweep 发现互补（块越粗 scale 越被离群/norm 差异拖垮）；③它的旋转+逐token缩放 与 QVG 的 k-means 去均值是同一问题的两条路线；④有开源代码可对比 | `2605.19660.pdf` | `repro/0714/b-sweep.md`（TNI 与 B 敏感度的印证） |
| [2605.26266](https://arxiv.org/abs/2605.26266) (v1) | **Quantized Keys Steal Attention: Bias Correction for KV-Cache Compression in Video Diffusion** | Tuncer, Becker, Pfeil（TUM / Tensordyne，2026-05） | 高度相关的并行工作：指出 K 量化噪声经 softmax 指数的凸性产生 **Jensen bias**（量化 key 系统性抢走未量化当前 chunk 的注意力质量），给出按 attention score 的在线偏差校正（二阶 Taylor，零额外显存）；在 MAGI-1 / SkyReels-V2 / HY-WorldPlay 的 INT2 上接近 BF16 质量。与我们 0714 的 proxy 讨论（"注意力权重如何变形量化误差"）和 clip/MSE-悖论刨析直接互补——它给出的正是"误差如何过 softmax"的解析修正 | `2605.26266.pdf` | 0714 proxy 排行榜讨论（对话记录）、`repro/0714/report-0714.md` §"Clip 张量级刨析"背景 |

## 备注

- 项目根目录外还有一份 QVG 论文副本：`/home/zhizhousha/workspace/video-project/2602.02958v5.pdf`（历史文档中的引用多指向该路径，保留不动）。
- 2605.26266 的 Jensen-bias 校正是纯读路径的后处理，理论上可以叠加在 QVG / QuaRot 任何一种量化之上——候选后续实验：在我们的 frame-93 协议下测"QVG + bias correction"与"非对称 QuaRot B64 + bias correction"。
