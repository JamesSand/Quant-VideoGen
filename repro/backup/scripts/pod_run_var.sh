#!/bin/bash
# Variance study: rerun one INT2 method config. Usage: pod_run_var.sh <method> <runidx>
# methods: qvgpro | qvg | quarot_asym16 | quarot_sym16 | quarot_asym128 | rtn16
set -u
M=${1:?}; IDX=${2:?}
TAG=var_${M}_run${IDX}

cd /home/zhizhousha/workspace/video-project/Quant-VideoGen
. .venv/bin/activate
. repro/backup/scripts/env_fix.sh
export TRITON_CACHE_DIR=$PWD/repro/backup/triton_cache/$TAG
mkdir -p $TRITON_CACHE_DIR repro/backup/race repro/backup/logs
RESULT=repro/backup/race/result_$TAG.txt
echo "START $(date +%F_%T) tag=$TAG node=${NODE_NAME:-?}" > $RESULT

FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$FREE_MB" -lt 72000 ]; then echo "NODE_BUSY free=${FREE_MB}MiB" >> $RESULT; exit 42; fi

COMMON="--checkpoint_dir=ckpts/LongCat-Video --workload 480p_long_gen \
  --init_video_path results/longcat/base/1-0.mp4 --output_dir results/varstudy/$TAG \
  --num_segments 1 --num_cond_frames 73 --seed 0 \
  --prompt_source text_to_video_from_file --prompt assets/t2v.txt --prompt_idx 1"
export PYTHONPATH=experiments/LongCat

case $M in
  qvgpro)
    torchrun --nproc_per_node=1 --standalone repro/backup/scripts/longcat_rngiso_launcher.py $COMMON \
      --quant_type triton-nstages-kmeans-int2 --quant_block_size 16 \
      --cache_num_k_centroids 256 --cache_num_v_centroids 256 \
      --kmeans_max_iters 100 --num_prq_stages 4 ;;
  qvg)
    torchrun --nproc_per_node=1 --standalone experiments/LongCat/run_long_t2v.py $COMMON \
      --quant_type triton-nstages-kmeans-int2 --quant_block_size 64 \
      --cache_num_k_centroids 256 --cache_num_v_centroids 256 \
      --kmeans_max_iters 100 --num_prq_stages 1 ;;
  quarot_asym16)
    export QUAROT_BLOCK=16 QUAROT_SYM=0 QUAROT_ROTATE_K=1 QUAROT_ROTATE_V=1 QUAROT_TARGET=experiments/LongCat/run_long_t2v.py
    torchrun --nproc_per_node=1 --standalone repro/backup/scripts/quarot_launcher.py $COMMON \
      --quant_type naive-int2 --quant_block_size 16 ;;
  quarot_sym16)
    export QUAROT_BLOCK=16 QUAROT_SYM=1 QUAROT_ROTATE_K=1 QUAROT_ROTATE_V=1 QUAROT_TARGET=experiments/LongCat/run_long_t2v.py
    torchrun --nproc_per_node=1 --standalone repro/backup/scripts/quarot_launcher.py $COMMON \
      --quant_type naive-int2 --quant_block_size 16 ;;
  quarot_asym128)
    export QUAROT_BLOCK=128 QUAROT_SYM=0 QUAROT_ROTATE_K=1 QUAROT_ROTATE_V=1 QUAROT_TARGET=experiments/LongCat/run_long_t2v.py
    torchrun --nproc_per_node=1 --standalone repro/backup/scripts/quarot_launcher.py $COMMON \
      --quant_type naive-int2 --quant_block_size 128 ;;
  rtn16)
    torchrun --nproc_per_node=1 --standalone experiments/LongCat/run_long_t2v.py $COMMON \
      --quant_type naive-int2 --quant_block_size 16 ;;
  *) echo "unknown method $M" >> $RESULT; exit 2 ;;
esac > repro/backup/logs/$TAG.log 2>&1
RC=$?
OUT=results/varstudy/$TAG/1-0/segment_1.mp4
if [ $RC -ne 0 ] || [ ! -f "$OUT" ]; then
  echo "FAILED rc=$RC err=$(grep -m1 -oE 'OutOfMemoryError|RuntimeError[^\"]{0,60}' repro/backup/logs/$TAG.log | head -1)" >> $RESULT
  exit 1
fi
echo "OK" >> $RESULT
