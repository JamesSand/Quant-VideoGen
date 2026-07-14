#!/bin/bash
# repro wrapper: HY QVG at the SHIPPED run_bf16.sh geometry (56/52, 14 chunks,
# 7-move pose incl. left-8) — hypothesis: paper Table 1 used this geometry for
# both arms, and run_qvg.sh's 48/44 was a later change.
# Usage: run_hy_qvg_bf16geom.sh int2|int4
set -e
BITS=${1:?usage: run_hy_qvg_bf16geom.sh int2|int4}

PROMPT='A paved pathway leads towards a stone arch bridge spanning a calm body of water.  Lush green trees and foliage line the path and the far bank of the water. A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. The water reflects the surrounding greenery and the sky.  The scene is bathed in soft, natural light, creating a tranquil and serene atmosphere. The pathway is composed of large, rectangular stones, and the bridge is constructed of light gray stone.  The overall composition emphasizes the peaceful and harmonious nature of the landscape.'

IMAGE_PATH=assets/hyworld.png

MEMORY_FRAMES=56
TEMPORAL_CONTEXT_SIZE=52
PRED_LATENT_SIZE=4

quant_type="triton-nstages-kmeans-${BITS}"
cache_num_k_centroids=256
cache_num_v_centroids=256
kmeans_max_iters=2
quant_block_size=64
num_prq_stages=1

output_folder=results/hyworldplay_bf16geom/${BITS}

export PYTHONPATH=experiments/HY-WorldPlay
export LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}

torchrun --nproc_per_node=1 --standalone experiments/HY-WorldPlay/wan/generate.py \
  --input "$PROMPT" \
  --image_path "$IMAGE_PATH" \
  --num_chunk 14 \
  --pose "w-8,s-8,a-8,d-8,up-8,down-8,left-8" \
  --ar_model_path ckpts/HY-WorldPlay/wan_transformer \
  --ckpt_path ckpts/HY-WorldPlay/wan_distilled_model/model.pt \
  --offload_text_encoder \
  --out "$output_folder" \
  --memory_frames $MEMORY_FRAMES \
  --temporal_context_size $TEMPORAL_CONTEXT_SIZE \
  --pred_latent_size $PRED_LATENT_SIZE \
  --quant_type $quant_type \
  --quant_block_size $quant_block_size \
  --cache_num_k_centroids $cache_num_k_centroids \
  --cache_num_v_centroids $cache_num_v_centroids \
  --kmeans_max_iters $kmeans_max_iters \
  --num_prq_stages $num_prq_stages
