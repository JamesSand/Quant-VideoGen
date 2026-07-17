# HANDOFF — 2026-07-16/17 凌晨收尾

## 现在的状态（接手即用）

- **无在飞实验**，本地 8×H100 全空。（集群 Weka 故障 ENG-91011 已于 0717 修复，
  集群恢复可用——开 pod 姿势见 `repro/k8s/`。）
- 全部结论已推送 together/zhizhou-dev（最新链：`8ed37bc → 1da7d22 → 48c88aa →
  6b19945 → d5593cd → 175beea → 014bfb8 → 本次三件套`）。
- 结论总览读 [report-0716.md](report-0716.md)；数字查 [metric-matrix.md](metric-matrix.md)；
  HY 协议细节读 [hy-ref-metrics.md](hy-ref-metrics.md)。

## 必须知道的三件事（否则会重蹈覆辙）

1. **SSIM 一律 paper 实现**（11×11 avg_pool，`metric.py` 原版）；
   `sf_ref_metrics.py` 已修；0716 前由它产出的 npz 里 `ssim` 键作废、用 `ssim_paper`。
2. **HY 用两段协议**（drop 前 [1,断崖) / drop 后 + 断崖帧；断崖=首个 PSNR<28 帧）。
   [23,36)、[0,32) 等旧窗口全部作废。paper 的 HY INT2 值不可逐位复现（管线未发布），
   INT4 的 drop 前段与 paper 精确吻合。
3. **HY 跑 PCA launcher 的两个坑**：PYTHONPATH 必须含
   `experiments/HY-WorldPlay:experiments/HY-WorldPlay/wan`；**严禁** `PCA_SF_STORE_FIX=1`
   （HY 原生 BHSD，permute 会把 24 头当 token 数炸断言）。SF 则必须开它。

## 等用户拍板的决策

1. **issue 发不发**：[issue-draft-hy-eval.md](issue-draft-hy-eval.md) 定稿可发；
   作者答"HY 帧范围"一项即可定 INT2 列复现性。
2. **HY −3.1 dB 归因**：调参（r=8/半头分裂）已证伪；下一刀 = 层×头×K/V 误差分解、
   K/V 不对称预算、残差位宽/块扫描。
3. **N4 INT4 档定义**（残差 asym 4-bit B128 + coef 4-bit，BPE≈4.5？）；SF×INT4 补格。
4. **多 prompt 战役 / N4 kernel 化 M0-M4 / W8A8 重启**（均有现成计划文档）。

## 关键数据路径

- 逐帧数组：`repro/backup/protosearch/*.npz`（HY 新数组注意 `ssim_paper` 键）
- HY 视频：QVG=`results/hyworldplay/triton-nstages-kmeans-int2_64/kc_256_vc_256_nstages_1/`，
  N4=`results/pcastudy/hy_n4/`，补救 arm=`hy_n4_r8/`、`hy_n4_split128/`，
  参考=`results/hyworldplay/bf16_matched/`
- 12 帧 chunk 尝试的残骸（证明结构性跑不通）：`results/hyworldplay/{bf16,qvg}_chunk12/err.txt`
- paper 全文文本：`papers/2602.02958v5.pdf`（pdftotext 可用）
