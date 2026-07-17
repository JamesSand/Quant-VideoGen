# 标准评测矩阵：(LC × HY) × (INT2 × INT4) 的 PSNR / SSIM / LPIPS

> **约定（用户 2026-07-16 定，0717 修订）**：以后所有 PSNR/SSIM/LPIPS 评测按
> **(LC × HY) × (INT2 × INT4) 四格矩阵**报，不再只报单格。
> LPIPS 一律 paper 口径（[0,1] 直喂 vgg）。
>
> **SF 已从参考三指标矩阵中移除（用户 0717 决定）**：SF 是 T2V、**没有条件前缀**
> ——量化从第一个 block 就生效，onset 即帧 1，落在 38-39 dB 的近无损区（编码噪声底
> ~48 dB），方法间拉不开差距（QVG 38.65 vs N4 38.52），判别力太低；且 paper Table 1
> 本无 SF 行，没有对照锚点。**SF 的质量评测只走 VBench IQ**
> （[vbench-repro.md](vbench-repro.md)）。历史读数留档于
> [sf-ref-metrics.md](sf-ref-metrics.md)，不再更新。

## 各格的既定起点协议

| 格 | 协议 | 依据 |
|---|---|---|
| LC（INT2/INT4） | 首个生成帧 frame 93 | [0713 report](../0713/report-0713.md)，13/13 match 验证；paper 版 SSIM 下 QVG f93 = 28.73/0.9033 vs paper 28.716/0.909 **双指标精确 match** |
| HY（INT2/INT4） | **两段协议（用户 0717 定稿）：drop 前 [1,断崖) 与 drop 后 [断崖,末) 分别报三指标 + 断崖帧位置**（断崖=首个 PSNR<28 帧；~~[23,36)~~ 跨崖窗口、~~[0,32)~~ 起点窗口均作废） | [hy-ref-metrics.md](hy-ref-metrics.md)：HY 是平台+断崖结构；INT4 drop 前与 paper 三指标精确吻合，INT2 的 paper 值为跨崖均值形状 |

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
| HY | drop前 **35.11 / 0.9655 / 0.0544**，drop后 15.79/0.370/0.432（断崖 29） | 29.174 / 0.882 / 0.094¹ | 形状级 match¹ | drop前 31.98 / 0.9439 / 0.0770（**输 3.1 dB**），drop后 15.62/0.380/0.435（断崖同帧 29） |

### INT4

| 模型 | QVG（我们） | QVG（paper） | PSNR 差 | N4（我们） |
|---|---|---|---:|---|
| LC | 33.75 / 0.9535 / 0.056 | 37.141 / 0.978 / 0.024 | −3.39 ⚠² | **空**（需定义 INT4 档配置） |
| HY | drop前 **35.14 / 0.9640 / 0.0500**，drop后 19.72/0.664/0.222（断崖 35） | 34.454 / 0.954 / 0.051 | **三指标精确吻合**（+0.7 / +0.010 / −0.001）✓ | 空 |

¹ paper 的 HY 三元组是**跨崖窗口平均的形状**（我们 [20,32) = 31.1/0.882/0.099，
SSIM/LPIPS 与 paper 精确重合），其参考管线未发布、无法逐位复现——协议考古全文见
[hy-ref-metrics.md](hy-ref-metrics.md)。HY 行按两段协议报；INT2 不直接对 paper 数
（paper 值落在两段之间），**INT4 的 drop 前段与 paper 三指标精确吻合**。
² 单视频测量；QVG INT4 已知运行间波动 ±1-2 dB（k-means 质心 atomic_add），0713 的
n=3 均值 34.36±1.28、REPORT 三测 33.71/33.54/35.84——本值在既有散布内，非新异常。
**HY INT4 的 35.711 与原 REPORT 逐位一致**，INT4 协议复现成立。
³（作废）旧 [0,32) 起点窗口的 35.711/0.9694 已由两段协议取代；另原引 paper INT4
SSIM/LPIPS 0.930/0.062 为误抄，PDF 实值 0.954/0.051。

## 数据位置

- 新增逐帧数组（paper 口径 LPIPS）：`repro/backup/protosearch/sf_lc_qvg_int4.npz`、
  `sf_hy_qvg_int4.npz`（视频软链在 `results/pcastudy/{lc,hy}_qvg_int4/`，
  指向既有 `triton-nstages-kmeans-int4_64` 输出）
- 其余格来源：LC INT2 见 [ssim-lpips-validation.md](ssim-lpips-validation.md)；
  SF 见 [sf-ref-metrics.md](sf-ref-metrics.md)；HY 两段读数 =
  `sf_{kc_256_vc_256_nstages_1,hy_n4,hy_qvg_int4}.npz`（`ssim_paper` 键），
  断崖判据 PSNR<28
- 复现：`sf_ref_metrics.py <bf16_ref> <n_frames> <test>` → 按上表窗口读数

## 待补的空格（按优先级）

1. ~~HY × N4（INT2）~~ 已补（关键坑：HY 严禁 `PCA_SF_STORE_FIX`，见 hy-ref-metrics.md）
2. **HY × N4 的差距归因**：drop 前输 3.1 dB；r=8 与 128 维半头分裂两个补救 arm
   已双双证伪（见 hy-ref-metrics.md §三补救实验）——下一步是 层×头×K/V 误差分解定位
3. **N4 的 INT4 档定义**：`pca_quant.py` 残差写死 2-bit；需定一个 INT4-class 配置
   （候选：残差 asym 4-bit B128 + coef 4-bit，BPE ≈4.5 vs QVG INT4 的 4.30）——
   研究决策，待讨论
4. ~~SF × INT4~~ 取消（SF 已移出参考三指标矩阵，用户 0717 决定）
