# QVG 的 Evaluation 梳理：质量评测 × Ablation Studies

来源：paper（arXiv:2602.02958 v5）+ codebase 考古 + 我们 0713-0715 的复现实测。
（速度与压缩率的评测口径不在本文档，见 `repro/0714/details-0714.md` §2/§3。）

---

# 板块一：视频生成质量评测（Quality Assessment）

## 1.1 指标体系

| 指标 | 类型 | 口径 | 复现要点 |
|---|---|---|---|
| **PSNR** | 有参考（保真度） | 量化版生成 vs **同 seed 的 BF16 参考视频**，逐像素 | **首帧口径**：paper 数字对应第一张生成帧（LongCat = 全局 frame 93），非全视频平均——全视频 PSNR 被混沌放大拖到 12-20 dB 不可复现（0713 核心结论） |
| **SSIM** | 有参考 | 同上，结构相似度 | 可与 paper 对上 |
| **LPIPS** | 有参考 | 同上，VGG 感知距离 | repo 自带实现 |
| **VBench Image Quality** | **无参考** | VBench 套件中的单一维度（非全套 16 维） | 只在附录 A.1 的长视频实验中使用（表见 §1.3） |

前三者衡量"量化改变了多少"，VBench IQ 衡量"画面本身好不好"——长视频场景参考会漂移，
无参考指标是必要补充。

## 1.2 主表协议（Table 1）

- **模型**：LongCat-Video 13.6B / HY-WorldPlay 8B / Self-Forcing (Wan) 1.3B，480p
- **prompt**：MovieGen prompt 套件（经 Self-Forcing 官方设定，paper p.7 脚注）；repo 内 `assets/t2v.txt`
- **对比配置**：QVG（S=1/B=64/K=256）、QVG-Pro（S=4/B=16）；基线 RTN/KIVI/QuaRot
  （paper 私有移植——0713 复现：其 QuaRot 21.57 ≈ 我们的**对称**变体 21.42，正确的非对称
  实现是 30.38，基线被系统性弱化）
- 同表并报 Compression Ratio（账目诚实，细节见 details-0714 §2.3）

## 1.3 VBench 长视频质量（附录 A.1，Table 2）

SF 延长至 1400 帧（~90s），VBench Image Quality (%)：

| 方法 | 350 帧 | 700 | 1050 | 1400 |
|---|---:|---:|---:|---:|
| BF16 | 74.33 | 70.56 | 68.80 | 65.87 |
| KIVI | 67.57 | 57.18 | 44.31 | 35.85 |
| QuaRot | 48.33 | 45.17 | 45.58 | 44.45 |
| **QVG** | **74.36** | **69.52** | **67.23** | **67.28** |

读数：QVG 全程贴 BF16（长时漂移被有效抑制）；KIVI 随长度崩坏、QuaRot 起点就崩
（48.33@350 帧——again 弱对称移植的旁证）。

## 1.4 官方实现与我们的对应工具

- 官方：`experiments/LongCat/longcat_video/utils/metric.py`——独立 CLI（逐帧 PSNR +
  LPIPS(vgg)），不被主流程调用；VBench 评测代码未随 repo 发布（`--prompt_source
  image_to_video_vbench` 是上游 LongCat 的遗留死选项，与 A.1 无关）
- 我们：`repro/backup/scripts/precompute_arrays.py`（PSNR/SSIM/LPIPS 逐帧数组落盘）+
  frame-93 首帧协议（`repro/0713/REPRODUCE.md` §3）

## 1.5 口径缺口与注意事项

1. **首帧 vs 全视频歧义**：paper 未写明 PSNR 是首帧口径，按全视频复现必然对不上
2. **单 prompt / 单 seed、无方差**：QVG 运行间 σ=0.18（k-means 质心无种子），Table 1 的
   0.1-0.3 dB 级差距在噪声内
3. **基线记账偏松**：KIVI 本为非对称（应付 zero-point 存储），按无 zero-point 记 6.40×
4. **工具教训**：`pdftotext` 对本 PDF 附录页抽取为空——"全文没有 X"类断言必须用页面
   渲染复核（本文档初版因此漏掉附录 A）

---

# 板块二：Ablation Studies

## 2.1 分辨率（附录 A.2：480p → 720p，LongCat，INT2）

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| KIVI | 17.66 | 0.5518 | 0.3441 |
| QuaRot | 20.88 | 0.6659 | 0.2395 |
| **QVG** | **25.99** | **0.8174** | **0.1177** |

**质心配置与 480p 完全相同**（不需要加 K）。机制：QVG 按**定长 token chunk** 聚类——
分辨率升高只是每帧 token 变多、每 chunk 覆盖帧数变少，聚类规模不变。720p 只测质量、
无计时。

## 2.2 Chunk 大小（附录 B.1，Table 4，SF INT2）

| Chunk（token） | 帧/chunk | 每层 KV 显存 | 压缩率 | 开销 |
|---:|---:|---:|---:|---:|
| 37,440 | 24 | 220 MB | 7.0× | 1.3% |
| 18,720 | 12 | 230 MB | 6.7× | 2.0% |
| 9,360 | 6 | 264 MB | 5.8× | 3.3% |

大 chunk 双赢（质心摊薄 + k-means 次数少）；代价是未压缩窗口更大。我们的公式验证：
37,440 点公式给 6.97 vs 声称 7.0 ✓；9,360 点公式 6.10 vs 声称 5.8（paper 偏保守）。

## 2.3 质心数 K（附录 Table 5，S=1，INT2）

K=64→7.760×、128→7.676×、256→7.539×、512→7.307×。趋势：簇越多质量越好、压缩率越低；
K=256 恰好是 uint8 索引的上限（超过就要 9-bit 索引）。**记账笔误**（0714 对账）：
K=256→7.539× 隐含 BPE 2.122，低于"残差+scale"的下界 2.125——该表漏记了 0.125 的 scale 项。

## 2.4 PRQ 阶段数 S（Fig 5c + 我们的解构）

paper 只给逐阶段 MSE 缩减（stage 1 = 5.83×，递减到 ~1.09×）。**我们 0714 b-sweep 的
解构结论更尖锐**：QVG-Pro（S=4/B=16）对 QVG（S=1/B=64）的 +2.16 dB 里，**S=1→4 只贡献
+0.08 dB**，其余全来自 B=64→16 的细 scale——四轮渐进 k-means 的复杂度几乎白付。

## 2.5 Block size B（paper 没做，我们补的——0714 九宫格）

paper 无 B 消融；部分原因是**发布 kernel 跑不了 B=128**（`quant_pack` autotune bug，
`tl.arange(0,0)` 崩，我们已修：提交 8b81883）。我们的九宫格（INT2，frame-93）：

| 方法 \ B | 16 | 64 | 128 |
|---|---:|---:|---:|
| QVG (S=1) | 30.96 | 28.88 | 28.41 |
| QuaRot 非对称 | 30.38 | 28.85 | 24.54 |
| QuaRot 非对称+clip r=0.99 | 30.68 | 29.07 | 25.35 |

读数：QVG 大 B 端有韧性（64→128 仅 −0.47，QuaRot 崩 4.31）——质心把残差削平后对 scale
粒度不敏感；细节见 `repro/0714/b-sweep.md`。

## 2.6 Batch size（附录 C Table 7，速度侧 ablation）

batch 1/2/5 → 端到端 43/86/217 s，QVG 开销稳定 1.6-1.7%——开销由 chunk 大小决定、
与 batch/序列长度无关（paper 声称；我们的速度复现见 details-0714 §3.5）。

## 2.7 paper 未覆盖、我们已补的 ablation 汇总

| 维度 | 我们的结论 | 出处 |
|---|---|---|
| Block size B | 见 §2.5 九宫格 | 0714/b-sweep.md |
| 对称 vs 非对称残差网格 | 非对称 4 级 ≫ 三元（+1.8~9 dB，全线成立） | 0713/0714/0715 |
| 方差（n=3 重复） | 确定性方法 σ≤0.003；QVG σ=0.18（质心无种子） | 0713 方差研究 |
| clip（旋转后收缩/分位） | r=0.99 小赚，增益随 B 增大 | 0713 qclip + 0714 |
| 低秩字典 vs k-means 字典 | PCA-KV r=4：31.79 @ BPE 2.253，双超 QVG | 0715/pca-results.md |
