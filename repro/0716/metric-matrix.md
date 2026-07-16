# 标准评测矩阵：(LC × SF × HY) × (INT2 × INT4) 的 PSNR / SSIM / LPIPS

> **约定（用户 2026-07-16 定）**：以后所有 PSNR/SSIM/LPIPS 评测按
> **(LC × SF × HY) × (INT2 × INT4) 六格矩阵**报，不再只报单格。
> LPIPS 一律 paper 口径（[0,1] 直喂 vgg）。

## 各格的既定起点协议

| 格 | 协议 | 依据 |
|---|---|---|
| LC（INT2/INT4） | 首个生成帧 frame 93 | [0713 report](../0713/report-0713.md)，13/13 match 验证；paper 版 SSIM 下 QVG f93 = 28.73/0.9033 vs paper 28.716/0.909 **双指标精确 match** |
| HY INT2 | **平台期 [1,断崖) 均值 + 断崖帧位置**（断崖=首个 PSNR<28 帧；~~[23,36)~~ 跨崖窗口作废） | [hy-ref-metrics.md](hy-ref-metrics.md) 协议考古：HY 是平台+断崖结构，paper 三元组是跨崖平均的形状、其参考管线未发布 |
| HY INT4 | 起点窗口 [0,32) 均值（PSNR 取有限值均值；含零误差帧 → SSIM 均值偏高，注意） | [backup/REPORT.md](../backup/REPORT.md) §起点窗口 |
| SF（两位宽） | onset 帧（首个 PSNR<40；paper Table 1 **无 SF 行**，故为自建同协议对比） | [sf-ref-metrics.md](sf-ref-metrics.md) |

**SSIM 一律 paper 实现（metric.py 的 11×11 avg_pool 局部窗）**——`sf_ref_metrics.py`
曾用的全局 SSIM 严重虚高已废（0716 勘误，脚本已修）；受影响 npz 用 `ssim_paper` 键。

参考视频与被测必须**同 seed 同配置**（SF 注意 `num_output_frames` 改噪声形状，见
sf-ref-metrics.md §前提 3）。

## 矩阵现状（QVG = paper-match 锚点；N4 = 我们的方法）

格式：PSNR / SSIM / LPIPS(paper 口径)。paper 列 = Table 1。

### INT2

| 模型 | QVG（我们） | QVG（paper） | 判定 | N4（我们） |
|---|---|---|---|---|
| LC | 28.73 / **0.9033** / 0.089（f93，paper 版 SSIM） | 28.716 / 0.909 / 0.065 | PSNR+SSIM 双 match ✓ | **31.79 / 0.9424 / 0.067**（大胜） |
| SF | 38.65 / 0.9736 / 0.041（onset） | —（无 SF 行） | — | 38.52 / 0.9730 / 0.043（打平） |
| HY | **35.11 / 0.9655 / 0.0544**（平台 [1,29)） | 29.174 / 0.882 / 0.094¹ | 形状级 match¹ | 31.98 / 0.9439 / 0.0770（**输 3.1 dB**；断崖同帧 29） |

### INT4

| 模型 | QVG（我们） | QVG（paper） | PSNR 差 | N4（我们） |
|---|---|---|---:|---|
| LC | 33.75 / 0.9535 / 0.056 | 37.141 / 0.978 / 0.024 | −3.39 ⚠² | **空**（需定义 INT4 档配置） |
| SF | **空**（需生成 195 配置 INT4 视频） | —（无 SF 行） | — | 空 |
| HY | **35.711** / 0.9694³ / 0.046（[0,32)） | 34.454 / 0.930 / 0.062 | +1.26 ✓ | 空 |

¹ paper 的 HY 三元组是**跨崖窗口平均的形状**（我们 [20,32) = 31.1/0.882/0.099，
SSIM/LPIPS 与 paper 精确重合），其参考管线未发布、无法逐位复现——协议考古全文见
[hy-ref-metrics.md](hy-ref-metrics.md)。HY 行按平台期协议报，不再直接对 paper 数。
² 单视频测量；QVG INT4 已知运行间波动 ±1-2 dB（k-means 质心 atomic_add），0713 的
n=3 均值 34.36±1.28、REPORT 三测 33.71/33.54/35.84——本值在既有散布内，非新异常。
**HY INT4 的 35.711 与原 REPORT 逐位一致**，INT4 协议复现成立。
³ [0,32) 含零误差帧（SSIM=1），均值偏高；与 paper 0.930 的差主要是窗口含未量化帧。

## 数据位置

- 新增逐帧数组（paper 口径 LPIPS）：`repro/backup/protosearch/sf_lc_qvg_int4.npz`、
  `sf_hy_qvg_int4.npz`（视频软链在 `results/pcastudy/{lc,hy}_qvg_int4/`，
  指向既有 `triton-nstages-kmeans-int4_64` 输出）
- 其余格来源：LC INT2 见 [ssim-lpips-validation.md](ssim-lpips-validation.md)；
  SF 见 [sf-ref-metrics.md](sf-ref-metrics.md)；HY INT2 QVG =
  `sf_kc_256_vc_256_nstages_1.npz` 的 [23,36) 窗口
- 复现：`sf_ref_metrics.py <bf16_ref> <n_frames> <test>` → 按上表窗口读数

## 待补的空格（按优先级）

1. ~~HY × N4（INT2）~~ 已补（关键坑：HY 严禁 `PCA_SF_STORE_FIX`，见 hy-ref-metrics.md）
2. **HY × N4 的 256 维头补救**：平台期输 3.1 dB，候选 r=8 / 128 维半头分裂——待拍板
3. **N4 的 INT4 档定义**：`pca_quant.py` 残差写死 2-bit；需定一个 INT4-class 配置
   （候选：残差 asym 4-bit B128 + coef 4-bit，BPE ≈4.5 vs QVG INT4 的 4.30）——
   研究决策，待讨论
4. **SF × INT4（QVG + N4）**：需按 195-latent 匹配配置生成 QVG INT4 视频
