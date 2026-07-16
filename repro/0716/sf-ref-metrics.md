# Self-Forcing 参考指标（PSNR / SSIM / LPIPS）：QVG vs PCA-KV N4

## 先说清协议前提

1. **paper Table 1 没有 SF 行**（已核对原文 6-7 页：Table 1 只有 LongCat-13B 和
   HY-WorldPlay-8B 两个模型）。SF 的质量在 paper 里只用 VBench IQ 报（Fig 5 / 附录 A.1），
   所以 SF 的三指标**不存在"和 paper match"一说**——本表是我们自建的同协议方法间对比。
2. **onset（起点帧）协议**：与 LC frame-93 / HY [23,36) 同一思想（测"量化误差刚注入、
   未被自回归混沌放大"的纯净信号）。同 seed/同配置下，量化 run 与 BF16 参考在分歧前
   逐帧一致（编码噪声底 ~48-51 dB）；onset = 首个 PSNR < 40 的帧。SF 的量化几乎立即
   生效，onset = 帧 1。
3. **配置必须匹配**：`num_output_frames` 改变初始噪声张量形状 → 完全不同的采样路径。
   QVG 按 195 latents（=777 帧，与 BF16 参考同配置）重新生成；早先 600-latent 极限视频
   与 BF16@195 的对比无效（13.5 dB 是噪声路径差异的伪影，已作废）。
4. LPIPS = paper 口径（[0,1] 直喂 vgg，不归一化），见
   [ssim-lpips-validation.md](ssim-lpips-validation.md) 的口径决策。

## 结果（onset = 帧 1，vs BF16@195 参考）

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS (paper 口径) ↓ |
|---|---:|---:|---:|
| QVG（INT2，195 匹配配置） | 38.65 | 0.9991 | 0.0413 |
| **PCA-KV N4** | 38.52 | 0.9990 | 0.0427 |

参考项（非头条）：全视频均值（777 帧，混沌漂移主导，与量化精度基本脱钩）：
QVG 19.71 dB / N4 19.01 dB。

## 读数

- **SF 起点上 QVG 与 N4 打平**（Δ0.13 dB，编码噪声级）——与 LC 上 N4 +2.9 dB 的大胜
  不同。诚实结论：N4 的保真优势目前只在 LC 上验证到，SF 起点处两方法都几乎无损、
  拉不开差距。
- 与 VBench IQ 的 parity（[vbench-repro.md](vbench-repro.md)）一致：SF 上 N4 = QVG 档。

## 数据与复现

- 逐帧数组：`repro/backup/protosearch/sf_sf_195_qvg.npz`、`sf_sf_full_n4.npz`
  （字段 psnr/ssim/lpips，帧 0 起）
- 视频：`results/pcastudy/sf_195_qvg/0-0_ema.mp4`、`results/pcastudy/sf_full_n4/0-0_ema.mp4`；
  BF16 参考 = 既有 777f 极限视频
- 复现：`repro/backup/scripts/sf_ref_metrics.py <bf16_ref> 777 <test_video>`
  （自动找首分歧帧并存 npz）；N4 生成命令见 vbench-repro.md §复现命令

## 待补

HY 列（同思想、既定窗口 [23,36)）——N4 的 HY 生成在修复 PYTHONPATH 问题后重跑中，
QVG 窗口值已备好（26.782 / 0.9663 / 0.1597 vs paper 29.174 / 0.882 / 0.094，PSNR 在
原复现 ±2.6 dB 判定标准内；SSIM/LPIPS 绝对值有系统偏移，只做方法间同协议对比）。
