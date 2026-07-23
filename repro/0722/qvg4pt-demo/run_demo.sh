#!/bin/bash
# qvg4pt 最小 demo:单 prompt、单 segment 的 LongCat 续写,KV cache 用 qvg4pt 量化。
# Usage: bash repro/0722/qvg4pt-demo/run_demo.sh [prompt_idx]   (默认 1;需 1 张 GPU)
# 产出: repro/0722/qvg4pt-demo/demo_out/<P>-0/segment_1.mp4(不入 git)
set -eu
P=${1:-1}
cd /home/zhizhousha/workspace/video-project/Quant-VideoGen
. .venv/bin/activate
. repro/backup/scripts/env_fix.sh
export TRITON_CACHE_DIR=/tmp/qvg_triton/gpu${CUDA_VISIBLE_DEVICES:-x}
export PCA_FP8SIM=1            # fp8 合法元数据口径(终值口径)
BASE=results/multiprompt/mp100/lc/base/$P-0.mp4
[ -f "$BASE" ] || { echo "missing base video $BASE (先跑 lc_base:$P)"; exit 1; }
OUTD=repro/0722/qvg4pt-demo/demo_out
mkdir -p $OUTD

PYTHONPATH=experiments/LongCat \
  torchrun --nproc_per_node=1 --standalone repro/0722/qvg4pt-demo/qvg4pt_launcher.py \
  --checkpoint_dir=ckpts/LongCat-Video --workload 480p_long_gen \
  --init_video_path $BASE --num_segments 1 --num_cond_frames 73 \
  --seed 0 --prompt_source text_to_video_from_file \
  --prompt repro/0718/prompts100/selected.txt --prompt_idx $P \
  --output_dir $OUTD \
  --quant_type naive-int2 --quant_block_size 64

OUT=$OUTD/$P-0/segment_1.mp4
[ -f "$OUT" ] && echo "DEMO OK: $OUT" || { echo "DEMO FAILED: no output"; exit 1; }
