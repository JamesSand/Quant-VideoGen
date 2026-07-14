#!/bin/bash
# Capture raw KV (middle layers) from LongCat / HY for distribution plots.
# Usage: pod_run_kvplot.sh <lc|hy>
set -u
ARM=${1:?lc|hy}
TAG=kvplot_${ARM}

cd /home/zhizhousha/workspace/video-project/Quant-VideoGen
. .venv/bin/activate
. repro/backup/scripts/env_fix.sh
export TRITON_CACHE_DIR=$PWD/repro/backup/triton_cache/$TAG
mkdir -p $TRITON_CACHE_DIR repro/backup/race repro/backup/logs results/kvplot
RESULT=repro/backup/race/result_$TAG.txt
echo "START $(date +%F_%T) tag=$TAG node=${NODE_NAME:-?}" > $RESULT

FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$FREE_MB" -lt 72000 ]; then echo "NODE_BUSY free=${FREE_MB}MiB" >> $RESULT; exit 42; fi

export KVPLOT_OUT=results/kvplot/${ARM}_kv.pt KVPLOT_CAP=12000 KVPLOT_HALO=4

case $ARM in
  lc)
    export KVPLOT_TARGET=experiments/LongCat/run_long_t2v.py PYTHONPATH=experiments/LongCat
    torchrun --nproc_per_node=1 --standalone repro/backup/scripts/kvplot_launcher.py \
      --checkpoint_dir=ckpts/LongCat-Video --workload 480p_long_gen \
      --init_video_path results/longcat/base/1-0.mp4 --output_dir results/kvplot/lc_gen \
      --num_segments 1 --num_cond_frames 73 --seed 0 \
      --prompt_source text_to_video_from_file --prompt assets/t2v.txt --prompt_idx 1 \
      --quant_type triton-nstages-kmeans-int2 --quant_block_size 64 \
      --cache_num_k_centroids 256 --cache_num_v_centroids 256 \
      --kmeans_max_iters 100 --num_prq_stages 1 \
      > repro/backup/logs/$TAG.log 2>&1 ;;
  hy)
    PROMPT='A paved pathway leads towards a stone arch bridge spanning a calm body of water.  Lush green trees and foliage line the path and the far bank of the water. A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. The water reflects the surrounding greenery and the sky.  The scene is bathed in soft, natural light, creating a tranquil and serene atmosphere. The pathway is composed of large, rectangular stones, and the bridge is constructed of light gray stone.  The overall composition emphasizes the peaceful and harmonious nature of the landscape.'
    export LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}
    export KVPLOT_TARGET=experiments/HY-WorldPlay/wan/generate.py PYTHONPATH=experiments/HY-WorldPlay
    torchrun --nproc_per_node=1 --standalone repro/backup/scripts/kvplot_launcher.py \
      --input "$PROMPT" --image_path assets/hyworld.png --num_chunk 12 \
      --pose "w-8,s-8,a-8,d-8,up-8,down-8" \
      --ar_model_path ckpts/HY-WorldPlay/wan_transformer \
      --ckpt_path ckpts/HY-WorldPlay/wan_distilled_model/model.pt \
      --offload_text_encoder --out results/kvplot/hy_gen \
      --memory_frames 48 --temporal_context_size 44 --pred_latent_size 4 \
      --quant_type triton-nstages-kmeans-int2 --quant_block_size 64 \
      --cache_num_k_centroids 256 --cache_num_v_centroids 256 \
      --kmeans_max_iters 2 --num_prq_stages 1 \
      > repro/backup/logs/$TAG.log 2>&1 ;;
  *) echo "unknown arm $ARM" >> $RESULT; exit 2 ;;
esac
RC=$?
if [ $RC -ne 0 ] || [ ! -f "results/kvplot/${ARM}_kv.pt" ]; then
  echo "FAILED rc=$RC err=$(grep -m1 -oE 'OutOfMemoryError|Error[^\"]{0,80}' repro/backup/logs/$TAG.log | head -1)" >> $RESULT
  exit 1
fi
echo "OK" >> $RESULT
