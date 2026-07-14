#!/bin/bash
# 0714 speed reproduction. Usage: pod_run_speed.sh <sf|lc>
#   sf: paper Appendix C Table 6 target (SF 180f: 43s e2e / 0.74s QVG cost / 1.7%)
#       4 passes on ONE GPU: bf16 warm, bf16 measured, int2 warm, int2 measured
#   lc: TIME_BENCH=5 op-level decomposition, single-segment continuation,
#       bf16 vs int2, warm+measured each (explains the 26% int2 speedup)
set -u
ARM=${1:?sf|lc}
TAG=speed_${ARM}

cd /home/zhizhousha/workspace/video-project/Quant-VideoGen
. .venv/bin/activate
. repro/backup/scripts/env_fix.sh
export TRITON_CACHE_DIR=$PWD/repro/backup/triton_cache/$TAG
mkdir -p $TRITON_CACHE_DIR repro/backup/race repro/backup/logs
RESULT=repro/backup/race/result_$TAG.txt
echo "START $(date +%F_%T) tag=$TAG node=${NODE_NAME:-?}" > $RESULT

FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$FREE_MB" -lt 72000 ]; then echo "NODE_BUSY free=${FREE_MB}MiB" >> $RESULT; exit 42; fi
nvidia-smi --query-gpu=name,clocks.max.sm --format=csv,noheader >> $RESULT

run_sf() { # $1=mode(bf16|int2) $2=pass(warm|meas)
  local mode=$1 pass=$2 t0 t1
  local Q=""
  [ "$mode" = int2 ] && Q="--quant_type triton-nstages-kmeans-int2 \
    --cache_num_k_centroids 256 --cache_num_v_centroids 256 \
    --kmeans_max_iters 2 --quant_block_size 64 --num_prq_stages 1"
  [ "$mode" = bf16 ] && Q="--quant_type none"
  t0=$(date +%s)
  PYTHONPATH=experiments/Self-Forcing:. DUMP_KV_LEVEL=0 \
  torchrun --nproc_per_node=1 --standalone experiments/Self-Forcing/inference.py \
    --config_path experiments/Self-Forcing/configs/self_forcing_dmd.yaml \
    --checkpoint_path ckpts/Self-Forcing/self_forcing_dmd.pt \
    --data_path assets/t2v.txt --output_folder results/speed/sf_${mode}_${pass} \
    --num_samples 1 --num_output_frames 180 --local_attn_size 180 \
    --use_ema --save_with_index $Q \
    > repro/backup/logs/${TAG}_${mode}_${pass}.log 2>&1
  local rc=$?
  t1=$(date +%s)
  echo "PASS ${mode}_${pass} rc=$rc wall=$((t1-t0))s" >> $RESULT
  return $rc
}

run_lc() { # $1=mode(bf16|int2) $2=pass(warm|meas)
  local mode=$1 pass=$2 t0 t1
  local COMMON="--checkpoint_dir=ckpts/LongCat-Video --workload 480p_long_gen \
    --init_video_path results/longcat/base/1-0.mp4 --output_dir results/speed/lc_${mode}_${pass} \
    --num_segments 1 --num_cond_frames 73 --seed 0 \
    --prompt_source text_to_video_from_file --prompt assets/t2v.txt --prompt_idx 1"
  local Q="--quant_type none"
  [ "$mode" = int2 ] && Q="--quant_type triton-nstages-kmeans-int2 --quant_block_size 64 \
    --cache_num_k_centroids 256 --cache_num_v_centroids 256 \
    --kmeans_max_iters 100 --num_prq_stages 1"
  t0=$(date +%s)
  TIME_BENCH=5 PYTHONPATH=experiments/LongCat \
  torchrun --nproc_per_node=1 --standalone experiments/LongCat/run_long_t2v.py $COMMON $Q \
    > repro/backup/logs/${TAG}_${mode}_${pass}.log 2>&1
  local rc=$?
  t1=$(date +%s)
  echo "PASS ${mode}_${pass} rc=$rc wall=$((t1-t0))s" >> $RESULT
  return $rc
}

FAIL=0
case $ARM in
  sf)
    run_sf bf16 warm || FAIL=1
    run_sf bf16 meas || FAIL=1
    run_sf int2 warm || FAIL=1
    run_sf int2 meas || FAIL=1 ;;
  lc)
    run_lc bf16 warm || FAIL=1
    run_lc bf16 meas || FAIL=1
    run_lc int2 warm || FAIL=1
    run_lc int2 meas || FAIL=1 ;;
  *) echo "unknown arm $ARM" >> $RESULT; exit 2 ;;
esac

if [ $FAIL -ne 0 ]; then echo "FAILED (some pass rc!=0, see logs)" >> $RESULT; exit 1; fi
echo "OK" >> $RESULT
