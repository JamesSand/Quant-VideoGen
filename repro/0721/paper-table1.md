# QVG paper (2602.02958v5) Table 1 原始数字(INT2 段,对账基准)

> 从 PDF 第 6 页逐字抄录(0721);协议要点(§5.1/5.2,第 7-8 页)一并固化。
> 用途:MPFULL 全量评测的 VBench 对账(用户 0721 要求)。

## LongCat-Video-13B,INT2 KV Cache,480p

| Method | Compression(BF16) | PSNR | SSIM | LPIPS | BC | IQ | SC | AQ |
|---|---|---|---|---|---|---|---|---|
| BF16 参考行 | — | — | — | — | 96.22 | 72.72 | 95.51 | 64.83 |
| RTN | 6.40× | 20.872 | 0.719 | 0.203 | 84.84 | 59.60 | 70.63 | 43.38 |
| KIVI | 6.40× | 20.317 | 0.719 | 0.208 | 84.84 | 38.10 | 75.25 | 41.58 |
| Quarot | 6.40× | 21.573 | 0.759 | 0.171 | 86.12 | 50.70 | 80.61 | 49.49 |
| QVG-Pro(paper 版) | 4.97× | **30.376** | **0.935** | **0.048** | 96.20 | 71.74 | 94.92 | 63.88 |
| QVG | **6.94×** | 28.716 | 0.909 | 0.065 | 95.06 | 71.47 | 94.11 | 62.22 |

## HY-WorldPlay-8B,INT2 KV Cache,480p

| Method | Compression | PSNR | SSIM | LPIPS | BC | IQ | SC | AQ |
|---|---|---|---|---|---|---|---|---|
| BF16 参考行 | — | — | — | — | 97.92 | 74.33 | 97.90 | 69.85 |
| RTN | 6.40× | 24.199 | 0.696 | 0.229 | 96.16 | 71.86 | 96.08 | 69.15 |
| KIVI | 6.40× | 24.272 | 0.701 | 0.230 | 96.95 | 71.40 | 95.89 | 68.19 |
| Quarot | 6.40× | 25.207 | 0.738 | 0.205 | 97.34 | 72.26 | 96.64 | 69.38 |
| QVG-Pro(paper 版) | 5.20× | **31.562** | **0.923** | **0.069** | 98.00 | 74.15 | 97.96 | 69.45 |
| QVG | **7.05×** | 29.174 | 0.882 | 0.094 | 97.98 | 73.87 | 97.90 | 69.80 |

(SF 无 Table 1 行;paper 以 Figure 5 的"每 50 帧 IQ 曲线"呈现 SF。)

## 协议与实现要点(§5.1-5.3 原文摘录)

- **QVG 配置:S=1 stage,B=64;QVG-Pro(paper 版):S=4 stages,B=16**;
  K=256 质心,uint8 assignment;FP8 E4M3 per-group scaling factors;
- **"adopt pre-RoPE key caching"**(其方法侧;baseline 实现未描述);
- **QuaRot baseline 用 block size 16**("for fair comparison"——注意与我们
  B64 的 QuaRot 口径不同);
- **数据集:MovieGen benchmark,"follow Self-Forcing's official prompt
  settings"**,脚注指向 Self-Forcing 仓库的 MovieGenVideoBench_extended.txt
  ——**与我们 MPFULL 用的 1003 条为同一文件**(全量对账可行性 ✓);
- LC 协议:"report the number of the first generated chunk"(≈ 我们的 f93);
- 端到端开销:QVG LC 2.1% / HY 1.5% / SF 4.3%;kmeans 质心缓存提速 3×。

## 记账对照(0721 注)

- paper 名义压缩比 QVG 6.94× ⇒ BPE = 16/6.94 = **2.3054**——比我们一直沿用
  的 2.326 预算线更紧,且其实存字节(fp32 质心)实测为 2.464(6.49×):
  **paper 的名义压缩比按其发布代码的存储无法达成**;
- **命名冲突警示:paper 的 "QVG-Pro"(4 段渐进,BPE≈3.22)≠ 我们的反事实
  臂 qvgprot/qvgproc(单段 + BF16 质心 + asym B128,名义 2.3257)**。我们的
  臂在文档中一律称"QVG 名义合法变体(qvg-nom)",避免混淆。
