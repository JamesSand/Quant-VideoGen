#!/bin/bash
# Run one baseline config on one GPU with sentinel handoff.
# Usage: race_run_one.sh <gpu> <lc|hy> <int2|int4> <block> <sym> <rotate> <tag>
# Assumes a sentinel is already holding memory on <gpu>; we signal it to decay
# and immediately start the run so combined memory stays high while loading.
set -u
GPU=${1:?}; KIND=${2:?}; BITS=${3:?}; BLOCK=${4:?}; SYM=${5:?}; ROT=${6:?}; TAG=${7:?}

cd /home/zhizhousha/workspace/video-project/Quant-VideoGen
. .venv/bin/activate
. repro/env_fix.sh
export TRITON_CACHE_DIR=$PWD/repro/triton_cache/$TAG
mkdir -p $TRITON_CACHE_DIR repro/race

touch repro/race/release_gpu${GPU}   # tell the sentinel to start decaying

if [ "$KIND" = "lc" ]; then
  if [ "$ROT" = "0" ]; then
    # stock RTN path, no launcher
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=experiments/LongCat torchrun --nproc_per_node=1 --standalone \
      experiments/LongCat/run_long_t2v.py --checkpoint_dir=ckpts/LongCat-Video \
      --workload 480p_long_gen --init_video_path results/longcat/base/1-0.mp4 \
      --output_dir results/quarot/$TAG --num_segments 1 --num_cond_frames 73 --seed 0 \
      --prompt_source text_to_video_from_file --prompt assets/t2v.txt --prompt_idx 1 \
      --quant_type naive-$BITS --quant_block_size $BLOCK \
      > repro/logs/quarot_$TAG.log 2>&1
  else
    CUDA_VISIBLE_DEVICES=$GPU bash repro/run_quarot_longcat.sh $BITS $BLOCK $SYM $ROT $TAG \
      > repro/logs/quarot_$TAG.log 2>&1
  fi
  RC=$?
  OUT=results/quarot/$TAG/1-0/segment_1.mp4
  METRIC_ARGS="--v1 results/longcat/bf16/1-0/segment_1.mp4 --v2 $OUT --skip_frames 93"
else
  if [ "$ROT" = "0" ]; then
    CUDA_VISIBLE_DEVICES=$GPU QUAROT_DISABLE=1 bash repro/run_quarot_hy.sh $BITS $BLOCK $SYM 0 $TAG \
      > repro/logs/quarot_$TAG.log 2>&1
  else
    CUDA_VISIBLE_DEVICES=$GPU bash repro/run_quarot_hy.sh $BITS $BLOCK $SYM $ROT $TAG \
      > repro/logs/quarot_$TAG.log 2>&1
  fi
  RC=$?
  OUT=results/quarot/$TAG/0-0.mp4
  METRIC_ARGS="--v1 results/hyworldplay/bf16_matched/0-0.mp4 --v2 $OUT"
  if [ -s results/quarot/$TAG/err.txt ]; then
    echo "RESULT $TAG FAILED err.txt: $(head -1 results/quarot/$TAG/err.txt)" >> repro/race/results.txt
    exit 1
  fi
fi

if [ $RC -ne 0 ] || [ ! -f "$OUT" ]; then
  echo "RESULT $TAG FAILED rc=$RC out_missing=$([ -f "$OUT" ] && echo no || echo yes)" >> repro/race/results.txt
  exit 1
fi

CUDA_VISIBLE_DEVICES=$GPU python experiments/LongCat/longcat_video/utils/metric.py $METRIC_ARGS \
  --prompt_idx 0 --seed 0 --output_path repro/metrics/quarot_$TAG.jsonl \
  > repro/logs/metric_$TAG.log 2>&1
PSNR=$(grep -oP 'PSNR\s*\|\s*\K[0-9.]+' repro/logs/metric_$TAG.log | head -1)
SSIM=$(grep -oP 'SSIM\s*\|\s*\K[0-9.]+' repro/logs/metric_$TAG.log | head -1)
LPIPS=$(grep -oP 'LPIPS\s*\|\s*\K[0-9.]+' repro/logs/metric_$TAG.log | head -1)
echo "RESULT $TAG OK psnr=$PSNR ssim=$SSIM lpips=$LPIPS" >> repro/race/results.txt
