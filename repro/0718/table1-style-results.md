# QVG Paper Table-1 风格结果表(我们的 multi-prompt campaign 版)

> 指标集对齐 QVG paper Table 1:PSNR / SSIM / LPIPS(vs 各自 BF16 参考)+ VBench
> 四维(Background Consistency / Image Quality / Subject Consistency / Aesthetic
> Quality)。**已有数字填入,空格 = 打分器建设中**(BC/SC/AQ 三维正在补,视频全部
> 在盘,只需重打分)。绝对值不与 paper 横比(prompt 集/帧数/协议不同);同表内
> 臂间对比协议严格一致。

## LongCat-Video-13B(10 prompts × f93 单帧;QVG=同 seed 3 重复均值)

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | BG Consist.↑ | Image Quality↑ | Subj Consist.↑ | Aesthetic↑ | BPE↓ |
|---|---|---|---|---|---|---|---|---|
| BF16 KV(参考) | — | — | — | | 69.87 | | | 16 |
| QVG | 28.18 | 0.9302 | 0.0624 | | 69.84 | | | 2.326 |
| **Budget-PCA(r4 asym B128)** | **29.58** | **0.9391** | **0.0520** | | **69.88** | | | **2.3125** |

## HY-WorldPlay-8B(湖桥场景 × 10 seeds × frames[13:] 均值)

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | BG Consist.↑ | Image Quality↑ | Subj Consist.↑ | Aesthetic↑ | BPE↓ |
|---|---|---|---|---|---|---|---|---|
| BF16 KV(参考) | — | — | — | | 78.07 | | | 16 |
| QVG | 17.45 | 0.4204 | 0.3886 | | **78.50** | | | 2.326 |
| **Budget-PCA(K9:0/V9:0+KP 三值)** | **17.54** | **0.4415** | **0.3756** | | 77.71 | | | **2.258** |

## Self-Forcing-Wan(20 prompts × 700 帧前缀窗;参考三指标按协议不适用*)

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | BG Consist.↑ | Image Quality↑ | Subj Consist.↑ | Aesthetic↑ | BPE↓ |
|---|---|---|---|---|---|---|---|---|
| BF16 KV | — | — | — | | 64.10ᵃ | | | 16 |
| QVG | * | * | * | | 65.12 | | | 2.326 |
| **Budget-PCA(r4 asym B128)** | * | * | * | | **65.65** | | | **2.3125** |

## 注

- **Image Quality** = VBench 的 MUSIQ-SPAQ 口径(×100);LC/HY 取全视频窗,SF 取
  700 帧前缀窗(paper Figure 5(a) 同轴)。BC/SC/AQ 三列打分器在建(open_clip 已装,
  CLIP-B/32 一致性 + DINO 一致性 + LAION aesthetic head),视频无需重生成;
- \* SF 的参考三指标按 0716 协议移出([[sf-no-prefix-eval-exclusion]]:无条件前缀
  → onset=帧 1,近无损区无判别力);paper 的 Table 1 也只报 LC/HY 两模型的保真列;
- ᵃ SF BF16 的 IQ 目前 n=10(p1-10),QVG/PCA 为 n=20;
- 配对显著性(比绝对均值更强的证据)见 [multi-prompt-results.md](multi-prompt-results.md):
  LC 三指标各胜 9/10(p=0.011)、SF IQ 胜 15/20(p=0.021)、HY SSIM 9/10 / LPIPS 10/10 /
  PSNR 持平 / IQ 负 0.78(结构性,详见该文档);
- PSNR 绝对值口径提醒:LC 是 f93 单帧、HY 是全程均值,与 paper 的口径(LC 报
  "first divergent chunk"、HY 长生成设定)不同,勿直接对 paper 的 28.716/29.174。
