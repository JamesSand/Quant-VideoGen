#!/bin/bash
# Paper-method speed measurement (§5.3 / Appendix C): end-to-end wall of the
# SHIPPED workloads, bf16 vs qvg on the SAME GPU. QVG runs twice where cheap
# (pass1 exposes triton compile, pass2 = steady state; LC amortizes over 10 seg).
# Usage: pod_run_paperspeed.sh <sf|lc|hy>
set -u
ARM=${1:?sf|lc|hy}
TAG=paperspeed_${ARM}

cd /home/zhizhousha/workspace/video-project/Quant-VideoGen
. .venv/bin/activate
. repro/backup/scripts/env_fix.sh
export TRITON_CACHE_DIR=$PWD/repro/backup/triton_cache/$TAG
mkdir -p $TRITON_CACHE_DIR repro/backup/race repro/backup/logs results/paperspeed
RESULT=repro/backup/race/result_$TAG.txt
echo "START $(date +%F_%T) tag=$TAG node=${NODE_NAME:-?}" > $RESULT

FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$FREE_MB" -lt 72000 ]; then echo "NODE_BUSY free=${FREE_MB}MiB" >> $RESULT; exit 42; fi
nvidia-smi --query-gpu=name --format=csv,noheader >> $RESULT

t() { # t <label> <cmd...>: time one full generation, log wall to RESULT
  local label=$1; shift
  local t0=$(date +%s)
  "$@" > repro/backup/logs/${TAG}_${label}.log 2>&1
  local rc=$?
  echo "PASS $label rc=$rc wall=$(( $(date +%s) - t0 ))s" >> $RESULT
  return $rc
}

FAIL=0
case $ARM in
  sf)
    SFQ="--quant_type triton-nstages-kmeans-int2 --cache_num_k_centroids 256 --cache_num_v_centroids 256 --kmeans_max_iters 2 --quant_block_size 64 --num_prq_stages 1"
    sf_run() { PYTHONPATH=experiments/Self-Forcing:. DUMP_KV_LEVEL=0 \
      torchrun --nproc_per_node=1 --standalone experiments/Self-Forcing/inference.py \
      --config_path experiments/Self-Forcing/configs/self_forcing_dmd.yaml \
      --checkpoint_path ckpts/Self-Forcing/self_forcing_dmd.pt \
      --data_path assets/t2v.txt --output_folder results/paperspeed/sf_$1 \
      --num_samples 1 --num_output_frames 180 --local_attn_size 180 \
      --use_ema --save_with_index $2; }
    t bf16      sf_run bf16 "--quant_type none"       || FAIL=1
    t qvg_pass1 sf_run qvg1 "$SFQ"                    || FAIL=1
    t qvg_pass2 sf_run qvg2 "$SFQ"                    || FAIL=1 ;;
  lc)
    LCC="--checkpoint_dir=ckpts/LongCat-Video --workload 480p_long_gen \
      --init_video_path results/longcat/base/1-0.mp4 --num_segments 10 --num_cond_frames 73 \
      --seed 0 --prompt_source text_to_video_from_file --prompt assets/t2v.txt --prompt_idx 1"
    lc_run() { PYTHONPATH=experiments/LongCat \
      torchrun --nproc_per_node=1 --standalone experiments/LongCat/run_long_t2v.py \
      $LCC --output_dir results/paperspeed/lc_$1 $2; }
    t bf16 lc_run bf16 "--quant_type none" || FAIL=1
    t qvg  lc_run qvg  "--quant_type triton-nstages-kmeans-int2 --quant_block_size 64 --cache_num_k_centroids 256 --cache_num_v_centroids 256 --kmeans_max_iters 100 --num_prq_stages 1" || FAIL=1 ;;
  hy)
    PROMPT='A paved pathway leads towards a stone arch bridge spanning a calm body of water.  Lush green trees and foliage line the path and the far bank of the water. A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. The water reflects the surrounding greenery and the sky.  The scene is bathed in soft, natural light, creating a tranquil and serene atmosphere. The pathway is composed of large, rectangular stones, and the bridge is constructed of light gray stone.  The overall composition emphasizes the peaceful and harmonious nature of the landscape.'
    export LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}
    HYQ="--quant_type triton-nstages-kmeans-int2 --quant_block_size 64 --cache_num_k_centroids 256 --cache_num_v_centroids 256 --kmeans_max_iters 2 --num_prq_stages 1"
    hy_run() { PYTHONPATH=experiments/HY-WorldPlay \
      torchrun --nproc_per_node=1 --standalone experiments/HY-WorldPlay/wan/generate.py \
      --input "$PROMPT" --image_path assets/hyworld.png --num_chunk 12 \
      --pose "w-8,s-8,a-8,d-8,up-8,down-8" \
      --ar_model_path ckpts/HY-WorldPlay/wan_transformer \
      --ckpt_path ckpts/HY-WorldPlay/wan_distilled_model/model.pt \
      --offload_text_encoder --out results/paperspeed/hy_$1 \
      --memory_frames 48 --temporal_context_size 44 --pred_latent_size 4 $2; }
    t bf16      hy_run bf16 "--quant_type none" || FAIL=1
    t qvg_pass1 hy_run qvg1 "$HYQ"              || FAIL=1
    t qvg_pass2 hy_run qvg2 "$HYQ"              || FAIL=1 ;;
  *) echo "unknown arm $ARM" >> $RESULT; exit 2 ;;
esac

[ $FAIL -ne 0 ] && { echo "FAILED (see logs)" >> $RESULT; exit 1; }
echo "OK" >> $RESULT
