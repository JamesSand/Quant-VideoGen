#!/bin/bash
# repro wrapper: BF16 baseline MATCHED to scripts/HY-WorldPlay/run_qvg.sh geometry
# (shipped run_bf16.sh uses num_chunk=14/pose+left-8/MEM 56/52 -> 221 frames,
#  which cannot be PSNR-compared with the 189-frame qvg output; this run uses
#  the qvg settings + quant_type none so the pair is frame-aligned.)
PROMPT='A paved pathway leads towards a stone arch bridge spanning a calm body of water.  Lush green trees and foliage line the path and the far bank of the water. A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. The water reflects the surrounding greenery and the sky.  The scene is bathed in soft, natural light, creating a tranquil and serene atmosphere. The pathway is composed of large, rectangular stones, and the bridge is constructed of light gray stone.  The overall composition emphasizes the peaceful and harmonious nature of the landscape.'

IMAGE_PATH=assets/hyworld.png

MEMORY_FRAMES=48
TEMPORAL_CONTEXT_SIZE=44
PRED_LATENT_SIZE=4

output_folder=results/hyworldplay/bf16_matched

export PYTHONPATH=experiments/HY-WorldPlay

torchrun --nproc_per_node=1 --standalone experiments/HY-WorldPlay/wan/generate.py \
  --input "$PROMPT" \
  --image_path "$IMAGE_PATH" \
  --num_chunk 12 \
  --pose "w-8,s-8,a-8,d-8,up-8,down-8" \
  --ar_model_path ckpts/HY-WorldPlay/wan_transformer \
  --ckpt_path ckpts/HY-WorldPlay/wan_distilled_model/model.pt \
  --offload_text_encoder \
  --out "$output_folder" \
  --memory_frames $MEMORY_FRAMES \
  --temporal_context_size $TEMPORAL_CONTEXT_SIZE \
  --pred_latent_size $PRED_LATENT_SIZE \
  --quant_type none
