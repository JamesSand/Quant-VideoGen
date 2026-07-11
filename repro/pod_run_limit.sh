#!/bin/bash
# Push each model to its generation limit on a dedicated 1-GPU cluster pod.
# Usage: pod_run_limit.sh <lc_bf16|lc_int2|sf_bf16|sf_int2|hy_bf16|hy_int2>
# bf16 configs sized near the 80GB VRAM edge auto-step-down on OOM — the
# largest size that completes IS the measured limit.
set -u
CFG=${1:?config}

cd /home/zhizhousha/workspace/video-project/Quant-VideoGen
. .venv/bin/activate
. repro/env_fix.sh
export TRITON_CACHE_DIR=$PWD/repro/triton_cache/limit_$CFG
mkdir -p $TRITON_CACHE_DIR repro/race repro/logs results/limits
RESULT=repro/race/result_limit_$CFG.txt
echo "START $(date +%F_%T) cfg=$CFG node=${NODE_NAME:-?}" > $RESULT

FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$FREE_MB" -lt 72000 ]; then echo "NODE_BUSY free=${FREE_MB}MiB" >> $RESULT; exit 42; fi

QVG_ARGS_LC="--quant_type triton-nstages-kmeans-int2 --quant_block_size 64 --cache_num_k_centroids 256 --cache_num_v_centroids 256 --kmeans_max_iters 100 --num_prq_stages 1"
QVG_ARGS_SF="--quant_type triton-nstages-kmeans-int2 --quant_block_size 64 --cache_num_k_centroids 256 --cache_num_v_centroids 256 --kmeans_max_iters 2 --num_prq_stages 1"

run_lc() {  # $1 = quant args or "--quant_type none", $2 = tag
  PYTHONPATH=experiments/LongCat torchrun --nproc_per_node=1 --standalone \
    experiments/LongCat/run_long_t2v.py --checkpoint_dir=ckpts/LongCat-Video \
    --workload 480p_long_gen --init_video_path results/longcat/base/1-0.mp4 \
    --output_dir results/limits/$2 --num_segments 70 --num_cond_frames 73 --seed 0 \
    --prompt_source text_to_video_from_file --prompt assets/t2v.txt --prompt_idx 1 \
    $1 > repro/logs/limit_$2.log 2>&1
}

run_sf() {  # $1 = latents, $2 = quant args, $3 = tag
  PYTHONPATH=experiments/Self-Forcing:. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --nproc_per_node=1 --standalone \
    experiments/Self-Forcing/inference.py \
    --config_path experiments/Self-Forcing/configs/self_forcing_dmd.yaml \
    --checkpoint_path ckpts/Self-Forcing/self_forcing_dmd.pt \
    --data_path repro/prompt0.txt --output_folder results/limits/$3 \
    --num_samples 1 --num_output_frames $1 --local_attn_size $1 \
    --use_ema --save_with_index $2 > repro/logs/limit_$3.log 2>&1
}

run_hy() {  # $1 = num_chunk, $2 = memory_frames, $3 = pose, $4 = quant args, $5 = tag
  PROMPT='A paved pathway leads towards a stone arch bridge spanning a calm body of water.  Lush green trees and foliage line the path and the far bank of the water. A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. The water reflects the surrounding greenery and the sky.  The scene is bathed in soft, natural light, creating a tranquil and serene atmosphere. The pathway is composed of large, rectangular stones, and the bridge is constructed of light gray stone.  The overall composition emphasizes the peaceful and harmonious nature of the landscape.'
  PYTHONPATH=experiments/HY-WorldPlay:experiments/HY-WorldPlay/wan \
  LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/stubs \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --nproc_per_node=1 --standalone experiments/HY-WorldPlay/wan/generate.py \
    --input "$PROMPT" --image_path assets/hyworld.png \
    --num_chunk $1 --pose "$3" \
    --ar_model_path ckpts/HY-WorldPlay/wan_transformer \
    --ckpt_path ckpts/HY-WorldPlay/wan_distilled_model/model.pt \
    --offload_text_encoder --out results/limits/$5 \
    --memory_frames $2 --temporal_context_size $(($2-4)) --pred_latent_size 4 \
    $4 > repro/logs/limit_$5.log 2>&1
}

pose_cycle() {  # $1 = number of 8-latent entries
  local moves=(w s a d up down) out="" i
  for ((i=0;i<$1;i++)); do out+="${moves[$((i%6))]}-8,"; done
  echo "${out%,}"
}

oom_in_log() { grep -q "OutOfMemoryError" "repro/logs/limit_$1.log" 2>/dev/null; }

case $CFG in
  lc_bf16)
    run_lc "--quant_type none" lc_bf16
    RC=$?; OUT=results/limits/lc_bf16/1-0/segment_70.mp4 ;;
  lc_int2)
    run_lc "$QVG_ARGS_LC" lc_int2
    RC=$?; OUT=results/limits/lc_int2/1-0/segment_70.mp4 ;;
  sf_bf16)
    for L in 228 210 195; do
      run_sf $L "--quant_type none" sf_bf16
      RC=$?; OUT=results/limits/sf_bf16/0-0_ema.mp4
      if [ $RC -eq 0 ] && [ -f "$OUT" ]; then echo "LIMIT latents=$L" >> $RESULT; break; fi
      oom_in_log sf_bf16 && { echo "OOM_AT latents=$L, stepping down" >> $RESULT; continue; } || break
    done ;;
  sf_int2)
    for L in 720 600 480; do
      run_sf $L "$QVG_ARGS_SF" sf_int2
      RC=$?; OUT=results/limits/sf_int2/0-0_ema.mp4
      if [ $RC -eq 0 ] && [ -f "$OUT" ]; then echo "LIMIT latents=$L" >> $RESULT; break; fi
      oom_in_log sf_int2 && { echo "OOM_AT latents=$L, stepping down" >> $RESULT; continue; } || break
    done ;;
  hy_bf16)
    for NC in 20 18 16; do
      run_hy $NC $((NC*4+4)) "$(pose_cycle $((NC/2)))" "--quant_type none" hy_bf16
      RC=$?; OUT=results/limits/hy_bf16/0-0.mp4
      if [ $RC -eq 0 ] && [ -f "$OUT" ] && [ ! -s results/limits/hy_bf16/err.txt ]; then echo "LIMIT chunks=$NC" >> $RESULT; break; fi
      { oom_in_log hy_bf16 || grep -q "OutOfMemoryError" results/limits/hy_bf16/err.txt 2>/dev/null; } && { echo "OOM_AT chunks=$NC, stepping down" >> $RESULT; rm -f results/limits/hy_bf16/err.txt; continue; } || break
    done ;;
  hy_int2)
    for NC in 60 52 44; do
      run_hy $NC $((NC*4+4)) "$(pose_cycle $((NC/2)))" "--quant_type triton-nstages-kmeans-int2 --quant_block_size 64 --cache_num_k_centroids 256 --cache_num_v_centroids 256 --kmeans_max_iters 2 --num_prq_stages 1" hy_int2
      RC=$?; OUT=results/limits/hy_int2/0-0.mp4
      if [ $RC -eq 0 ] && [ -f "$OUT" ] && [ ! -s results/limits/hy_int2/err.txt ]; then echo "LIMIT chunks=$NC" >> $RESULT; break; fi
      { oom_in_log hy_int2 || grep -q "OutOfMemoryError" results/limits/hy_int2/err.txt 2>/dev/null; } && { echo "OOM_AT chunks=$NC, stepping down" >> $RESULT; rm -f results/limits/hy_int2/err.txt; continue; } || break
    done ;;
  *) echo "unknown cfg $CFG" >> $RESULT; exit 2 ;;
esac

if [ "${RC:-1}" -ne 0 ] || [ ! -f "$OUT" ]; then
  echo "FAILED rc=${RC:-?} out_missing=$([ -f "$OUT" ] && echo no || echo yes) err=$(grep -m1 -oE 'OutOfMemoryError|RuntimeError[^\"]{0,80}|ValueError[^\"]{0,80}' repro/logs/limit_$CFG.log | head -1)" >> $RESULT
  exit 1
fi
NFRAMES=$(python - "$OUT" <<'PY'
import sys, imageio
print(sum(1 for _ in imageio.get_reader(sys.argv[1])))
PY
)
PEAK=$(grep -oE "Peak Memory Usage: [0-9.]+" repro/logs/limit_$CFG.log | tail -1)
echo "OK frames=$NFRAMES ${PEAK:-peak=n/a}" >> $RESULT
