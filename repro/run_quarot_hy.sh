#!/bin/bash
# QuaRot baseline run for HY-WorldPlay (matched 12-chunk geometry).
# Usage: run_quarot_hy.sh <int2|int4> <block> <sym:0|1> <rotate:1|0> <outtag>
set -e
BITS=${1:?bits}; BLOCK=${2:?block}; SYM=${3:?sym}; ROT=${4:?rotate}; TAG=${5:?tag}

PROMPT='A paved pathway leads towards a stone arch bridge spanning a calm body of water.  Lush green trees and foliage line the path and the far bank of the water. A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. The water reflects the surrounding greenery and the sky.  The scene is bathed in soft, natural light, creating a tranquil and serene atmosphere. The pathway is composed of large, rectangular stones, and the bridge is constructed of light gray stone.  The overall composition emphasizes the peaceful and harmonious nature of the landscape.'

export QUAROT_BLOCK=$BLOCK QUAROT_SYM=$SYM QUAROT_ROTATE_K=$ROT QUAROT_ROTATE_V=$ROT
export QUAROT_TARGET=experiments/HY-WorldPlay/wan/generate.py
# generate.py does bare `import models` etc. resolved from its own dir; a
# direct torchrun puts wan/ on sys.path[0] but runpy.run_path does not, so
# add it to PYTHONPATH explicitly.
export PYTHONPATH=experiments/HY-WorldPlay:experiments/HY-WorldPlay/wan
export LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}

torchrun --nproc_per_node=1 --standalone repro/quarot_launcher.py \
  --input "$PROMPT" \
  --image_path assets/hyworld.png \
  --num_chunk 12 \
  --pose "w-8,s-8,a-8,d-8,up-8,down-8" \
  --ar_model_path ckpts/HY-WorldPlay/wan_transformer \
  --ckpt_path ckpts/HY-WorldPlay/wan_distilled_model/model.pt \
  --offload_text_encoder \
  --out results/quarot/${TAG} \
  --memory_frames 48 --temporal_context_size 44 --pred_latent_size 4 \
  --quant_type naive-${BITS} --quant_block_size ${BLOCK}
