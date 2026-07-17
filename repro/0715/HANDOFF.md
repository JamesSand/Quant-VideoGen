# HANDOFF — 2026-07-15

> 本页大部分已被 0716 的工作推进/修正，接手请先读
> [../0716/HANDOFF.md](../0716/HANDOFF.md)；此处只留 0715 特有的状态。

- **N4 = 当前最优方法**（配置见 report-0715.md §二），fake-quant 实现在
  `repro/backup/scripts/pca_quant.py`，启动器 `pca_launcher.py`，
  批量脚本 `pod_run_pca.sh <tag> <R> <coeff_bits> <res_grid> <v_mode> [res_block] [k_basis_file]`。
- **OSCAR QᵀQ 基**：`results/pcastudy/lc_qqt_basis.pt`（48 层×32 头×128×128，
  由 `oscar_calib_launcher.py` pass-1 捕获）——已证负结果，仅留档。
- **W8A8+KV2 计划未启动**（w8a8-kv2-plan.md），Phase 0 三个决策点等讨论；
  重启时 KV 侧应直接用 N4 而非 naive-int2。
- arm 视频都在 `results/pcastudy/pca_*/`；评测数字用 `pca_eval.py`（BPE 账 + f93 PSNR）。
