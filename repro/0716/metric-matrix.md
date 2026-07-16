# 标准评测矩阵：(LC × SF × HY) × (INT2 × INT4) 的 PSNR / SSIM / LPIPS

> **约定（用户 2026-07-16 定）**：以后所有 PSNR/SSIM/LPIPS 评测按
> **(LC × SF × HY) × (INT2 × INT4) 六格矩阵**报，不再只报单格。
> LPIPS 一律 paper 口径（[0,1] 直喂 vgg）。

## 各格的既定起点协议

| 格 | 协议 | 依据 |
|---|---|---|
| LC（INT2/INT4） | 首个生成帧 frame 93 | [0713 report](../0713/report-0713.md)，13/13 match 验证 |
| HY INT2 | 起点窗口 [23,36) 均值 | [backup/REPORT.md](../backup/REPORT.md) §起点窗口 |
| HY INT4 | 起点窗口 [0,32) 均值（PSNR 取有限值均值，零误差帧剔除） | 同上 |
| SF（两位宽） | onset 帧（首个 PSNR<40；paper Table 1 **无 SF 行**，故为自建同协议对比） | [sf-ref-metrics.md](sf-ref-metrics.md) |

参考视频与被测必须**同 seed 同配置**（SF 注意 `num_output_frames` 改噪声形状，见
sf-ref-metrics.md §前提 3）。

## 矩阵现状（QVG = paper-match 锚点；N4 = 我们的方法）

格式：PSNR / SSIM / LPIPS(paper 口径)。paper 列 = Table 1。

### INT2

| 模型 | QVG（我们） | QVG（paper） | PSNR 差 | N4（我们） |
|---|---|---|---:|---|
| LC | 28.97±0.16 / 0.907 / 0.089 | 28.716 / 0.909 / 0.065 | +0.25 ✓ | **31.79 / 0.942 / 0.067** |
| SF | 38.65 / 0.9991 / 0.041（onset） | —（无 SF 行） | — | 38.52 / 0.9990 / 0.043（打平） |
| HY | 26.78 / 0.966 / 0.160 | 29.174 / 0.882 / 0.094 | −2.39 ✓¹ | **空**（生成仍失败，见下） |

### INT4

| 模型 | QVG（我们） | QVG（paper） | PSNR 差 | N4（我们） |
|---|---|---|---:|---|
| LC | 33.75 / 0.997 / 0.056 | 37.141 | −3.39 ⚠² | **空**（需定义 INT4 档配置） |
| SF | **空**（需生成 195 配置 INT4 视频） | —（无 SF 行） | — | 空 |
| HY | **35.711** / 0.998 / 0.046 | 34.454 / 0.930 / 0.062 | +1.26 ✓ | 空 |

¹ 在原复现 13/13@±2.6 dB 判定标准内；HY 的 SSIM/LPIPS 绝对值与 paper 系统偏移
（参考源/实现差异），只做方法间同协议对比，PSNR 才对 paper。
² 单视频测量；QVG INT4 已知运行间波动 ±1-2 dB（k-means 质心 atomic_add），0713 的
n=3 均值 34.36±1.28、REPORT 三测 33.71/33.54/35.84——本值在既有散布内，非新异常。
**HY INT4 的 35.711 与原 REPORT 逐位一致**，INT4 协议复现成立。

## 数据位置

- 新增逐帧数组（paper 口径 LPIPS）：`repro/backup/protosearch/sf_lc_qvg_int4.npz`、
  `sf_hy_qvg_int4.npz`（视频软链在 `results/pcastudy/{lc,hy}_qvg_int4/`，
  指向既有 `triton-nstages-kmeans-int4_64` 输出）
- 其余格来源：LC INT2 见 [ssim-lpips-validation.md](ssim-lpips-validation.md)；
  SF 见 [sf-ref-metrics.md](sf-ref-metrics.md)；HY INT2 QVG =
  `sf_kc_256_vc_256_nstages_1.npz` 的 [23,36) 窗口
- 复现：`sf_ref_metrics.py <bf16_ref> <n_frames> <test>` → 按上表窗口读数

## 待补的空格（按优先级）

1. **HY × N4（INT2）**：生成持续失败（`results/pcastudy/hy_n4/err.txt`，
   PYTHONPATH 修复后 chunk 0 能出、后续 chunk 报错）——需先修生成
2. **N4 的 INT4 档定义**：`pca_quant.py` 残差写死 2-bit；需定一个 INT4-class 配置
   （候选：残差 asym 4-bit B128 + coef 4-bit，BPE ≈4.5 vs QVG INT4 的 4.30）——
   研究决策，待讨论
3. **SF × INT4（QVG + N4）**：需按 195-latent 匹配配置生成 QVG INT4 视频
