# REPRODUCE — 0715 头条数字的指令级复现

前置同 [../0716/REPRODUCE.md](../0716/REPRODUCE.md)（env_fix.sh + LIBRARY_PATH）。

## 1. N4 冠军（LC INT2，31.79 @ BPE 2.253）

```bash
# 生成（LC，prompt_idx=1 滑板手，seed 0；~15 min/arm 单卡）
bash repro/backup/scripts/pod_run_pca.sh pca_n4 4 2 asym pca 128
# 参数: <tag> <R> <coeff_bits> <res_grid:ternary|asym> <v_mode:mean|pca> [res_block=64] [k_basis_file]

# 评测（frame-93 PSNR + BPE 账 + WIN 判定，vs results/longcat/bf16/1-0/segment_1.mp4）
.venv/bin/python repro/backup/scripts/pca_eval.py pca_n4
# 期望: PSNR 31.788, BPE 2.253 (fake; real 含 coef zero-point = 2.3125), WIN vs QVG 28.88@2.326
```

## 2. auto-research 的四条机制结论（对应 arm）

```bash
bash .../pod_run_pca.sh pca_r6  6 2 asym pca 128    # 秩扫描: r4 > r6 > r8 > r16
bash .../pod_run_pca.sh pca_r8  8 2 asym pca 128
bash .../pod_run_pca.sh pca_vmean 4 2 asym mean 128 # V mean-only: 比 v_mode=pca 低 1.2-1.5 dB
bash .../pod_run_pca.sh pca_tern 4 2 ternary pca 128 # 残差 ternary: 比 asym 低 ~1.8 dB
bash .../pod_run_pca.sh pca_b64 4 2 asym pca 64      # B=64: 与 128 同质量、BPE 更差
```

## 3. OSCAR QᵀQ 负结果

```bash
# pass-1: 捕获 Q^T Q 校准基（wrap q_norm.forward，48×32×128×128）
CUDA_VISIBLE_DEVICES=7 .venv/bin/torchrun --standalone --nproc_per_node=1 \
  repro/backup/scripts/oscar_calib_launcher.py   # → results/pcastudy/lc_qqt_basis.pt
# pass-2: 用外部基跑 N4 配置
bash .../pod_run_pca.sh pca_o1 4 2 asym pca 128 results/pcastudy/lc_qqt_basis.pt
# 期望: r=4 → 28.88 (−2.9 vs 自协方差), r=8 → 31.33（仍差且更贵）
```

## 4. Phase-0 谱分析图

```bash
.venv/bin/python repro/backup/scripts/kv_spectrum.py   # r80 分布、L29 放大系数、逐 chunk 基漂移
# 图与数字见 pca-spectrum.md
```
