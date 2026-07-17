# REPRODUCE — 0716 全部头条数字的指令级复现

前置（每个 shell 都要）：

```bash
cd /home/zhizhousha/workspace/video-project/Quant-VideoGen
. repro/backup/scripts/env_fix.sh          # TRITON_PTXAS_PATH（CUDA-13 坑）
export LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 1. 六格矩阵的读数（已有视频，只算指标，~2 min/条）

```bash
# 逐帧三指标（paper 版 SSIM + paper 口径 LPIPS），存 repro/backup/protosearch/sf_<tag>.npz
CUDA_VISIBLE_DEVICES=7 .venv/bin/python repro/backup/scripts/sf_ref_metrics.py \
  results/hyworldplay/bf16_matched/0-0.mp4 189 \
  results/hyworldplay/triton-nstages-kmeans-int2_64/kc_256_vc_256_nstages_1/0-0.mp4 \
  results/pcastudy/hy_n4/0-0.mp4

# HY 两段协议读数（drop 前/后 + 断崖帧）
.venv/bin/python - <<'EOF'
import numpy as np
for tag in ['kc_256_vc_256_nstages_1','hy_n4','hy_qvg_int4']:
    d = np.load(f'repro/backup/protosearch/sf_{tag}.npz')
    P,S,L = d['psnr'], d.get('ssim_paper', d['ssim']), d['lpips']
    c = next(i for i in range(1,len(P)) if P[i]<28)
    print(tag, f"断崖={c}",
          f"前: {P[1:c].mean():.2f}/{S[1:c].mean():.4f}/{L[1:c].mean():.4f}",
          f"后: {P[c:].mean():.2f}/{S[c:].mean():.4f}/{L[c:].mean():.4f}")
EOF
# 期望: QVG INT2 断崖29 前35.11/0.9655/0.0544；N4 断崖29 前31.98/0.9439/0.0770；
#       QVG INT4 断崖35 前35.14/0.9640/0.0500

# LC f93 / SF onset(f1) 单帧读数（LC QVG 28.73/0.9033、N4 31.79/0.9424；SF 38.65 vs 38.52）
# 参考: LC=results/longcat/bf16/1-0/segment_1.mp4 (f93)
#       SF=repro/backup/limit_videos/selfforcing_bf16_777frames_48.6s.mp4 (f1)
```

旧 npz 补 paper 版 SSIM 键：
`.venv/bin/python repro/backup/scripts/paper_ssim_recalc.py <ref> <n> "<video>:<tag>"`

## 2. HY N4 生成（含两个坑的正确姿势，~6 min）

```bash
PROMPT='<scripts/HY-WorldPlay/run_qvg.sh 里的湖桥 prompt 原文>'
CUDA_VISIBLE_DEVICES=7 PCA_R=4 PCA_COEFF_BITS=2 PCA_RES_GRID=asym PCA_V_MODE=pca PCA_RES_BLOCK=128 \
PCA_TARGET=experiments/HY-WorldPlay/wan/generate.py \
PYTHONPATH=experiments/HY-WorldPlay:experiments/HY-WorldPlay/wan \
.venv/bin/torchrun --nproc_per_node=1 --standalone repro/backup/scripts/pca_launcher.py \
  --input "$PROMPT" --image_path assets/hyworld.png --num_chunk 12 \
  --pose "w-8,s-8,a-8,d-8,up-8,down-8" \
  --ar_model_path ckpts/HY-WorldPlay/wan_transformer \
  --ckpt_path ckpts/HY-WorldPlay/wan_distilled_model/model.pt \
  --offload_text_encoder --out results/pcastudy/hy_n4 \
  --memory_frames 48 --temporal_context_size 44 --pred_latent_size 4 --quant_type naive-int2
```

坑：①漏 PYTHONPATH 的 `wan` → `No module named 'models'`；②加 `PCA_SF_STORE_FIX=1`
→ `data tokens (24)` 断言（HY 原生 BHSD）。补救 arm 复现：r=8 加 `PCA_R=8`；
半头分裂加 `PCA_SPLIT_D=128`（`pca_quant.py` 新选项）。

## 3. 三项勘误的验证

```bash
# ① SSIM 实现差异：对任意一对帧分别算全局版/11×11 版，差 ~0.2（0.966 vs 0.761）
# ② 12 帧 chunk 结构性跑不通：--pred_latent_size 3 --memory_frames 47 --temporal_context_size 44
#    → select_mem_frames_wan (utils.py:143) 数量断言必炸；err 在 results/hyworldplay/*_chunk12/err.txt
# ③ chunk 笔误：grep 'num_frame_per_block' experiments/Self-Forcing/configs/self_forcing_dmd.yaml  # =3 → 12帧
#    grep -n 'pred_latent_size' scripts/HY-WorldPlay/run_qvg.sh                                    # =4 → 16帧
pdftotext papers/2602.02958v5.pdf - | grep -A2 'chunk sizes of'   # paper 原文 "12 and 16"（写反）
```

## 4. 断崖机制验证（逐边界扫描）

```bash
.venv/bin/python - <<'EOF'
import numpy as np
P = np.load('repro/backup/protosearch/sf_kc_256_vc_256_nstages_1.npz')['psnr']
for b in [29,61,93,125,157]:   # 六段 pose 的五个切换帧
    print(f"边界{b}: 前13帧均值 {P[b-13:b].mean():.1f} → 后13帧 {P[b:b+13].mean():.1f}")
EOF
# 期望：只有 29 处跌 ~15 dB，其余 ±0.5 内 —— 分岔是一次性吸收态
EOF
```

## 5. VBench IQ（第四指标）

```bash
CUDA_VISIBLE_DEVICES=7 .venv/bin/python repro/backup/scripts/vbench_iq.py <video.mp4>
# 700f 处期望 BF16 71.51 / QVG 70.41 / N4 70.26（相对差 1.10 vs paper 1.04）
```
