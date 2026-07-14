#!/bin/bash
# Pod-side runner: one baseline config on a dedicated 1-GPU k8s pod.
# Usage: pod_run_one.sh <lc|hy> <int2|int4> <block> <sym> <rotate> <tag>
set -u
KIND=${1:?}; BITS=${2:?}; BLOCK=${3:?}; SYM=${4:?}; ROT=${5:?}; TAG=${6:?}

cd /home/zhizhousha/workspace/video-project/Quant-VideoGen
. .venv/bin/activate
. repro/backup/scripts/env_fix.sh
export TRITON_CACHE_DIR=$PWD/repro/backup/triton_cache/pod_$TAG
mkdir -p $TRITON_CACHE_DIR repro/backup/race repro/backup/logs repro/backup/metrics
RESULT=repro/backup/race/result_$TAG.txt
echo "START $(date +%F_%T) tag=$TAG kind=$KIND bits=$BITS block=$BLOCK sym=$SYM rot=$ROT host=$(hostname) node=${NODE_NAME:-unknown}" > $RESULT

# Fail fast if the "dedicated" GPU is actually occupied by a
# scheduler-bypassing pod on this node (cluster-wide double-booking issue).
FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$FREE_MB" -lt 72000 ]; then
  echo "NODE_BUSY free=${FREE_MB}MiB node=${NODE_NAME:-$(hostname)}" >> $RESULT
  exit 42
fi

if [ "$KIND" = "lc" ]; then
  if [ "$ROT" = "0" ]; then
    PYTHONPATH=experiments/LongCat torchrun --nproc_per_node=1 --standalone \
      experiments/LongCat/run_long_t2v.py --checkpoint_dir=ckpts/LongCat-Video \
      --workload 480p_long_gen --init_video_path results/longcat/base/1-0.mp4 \
      --output_dir results/quarot/$TAG --num_segments 1 --num_cond_frames 73 --seed 0 \
      --prompt_source text_to_video_from_file --prompt assets/t2v.txt --prompt_idx 1 \
      --quant_type naive-$BITS --quant_block_size $BLOCK \
      > repro/backup/logs/quarot_$TAG.log 2>&1
  else
    bash repro/backup/scripts/run_quarot_longcat.sh $BITS $BLOCK $SYM $ROT $TAG > repro/backup/logs/quarot_$TAG.log 2>&1
  fi
  RC=$?
  OUT=results/quarot/$TAG/1-0/segment_1.mp4
  METRIC_ARGS="--v1 results/longcat/bf16/1-0/segment_1.mp4 --v2 $OUT --skip_frames 93"
else
  if [ "$ROT" = "0" ]; then
    QUAROT_DISABLE=1 bash repro/backup/scripts/run_quarot_hy.sh $BITS $BLOCK $SYM 0 $TAG > repro/backup/logs/quarot_$TAG.log 2>&1
  else
    bash repro/backup/scripts/run_quarot_hy.sh $BITS $BLOCK $SYM $ROT $TAG > repro/backup/logs/quarot_$TAG.log 2>&1
  fi
  RC=$?
  OUT=results/quarot/$TAG/0-0.mp4
  METRIC_ARGS="--v1 results/hyworldplay/bf16_matched/0-0.mp4 --v2 $OUT"
  if [ -s results/quarot/$TAG/err.txt ]; then
    echo "FAILED err.txt: $(head -1 results/quarot/$TAG/err.txt)" >> $RESULT
    exit 1
  fi
fi

if [ $RC -ne 0 ] || [ ! -f "$OUT" ]; then
  echo "FAILED rc=$RC out_missing=$([ -f "$OUT" ] && echo no || echo yes) last_err=$(grep -m1 -oE 'OutOfMemoryError|ModuleNotFoundError[^\"]*|RuntimeError[^\"]*' repro/backup/logs/quarot_$TAG.log | head -1)" >> $RESULT
  exit 1
fi

python experiments/LongCat/longcat_video/utils/metric.py $METRIC_ARGS \
  --prompt_idx 0 --seed 0 --output_path repro/backup/metrics/quarot_$TAG.jsonl \
  > repro/backup/logs/metric_$TAG.log 2>&1
PSNR=$(grep -oP 'PSNR\s*\|\s*\K[0-9.]+' repro/backup/logs/metric_$TAG.log | head -1)
SSIM=$(grep -oP 'SSIM\s*\|\s*\K[0-9.]+' repro/backup/logs/metric_$TAG.log | head -1)
LPIPS=$(grep -oP 'LPIPS\s*\|\s*\K[0-9.]+' repro/backup/logs/metric_$TAG.log | head -1)
echo "OK psnr=$PSNR ssim=$SSIM lpips=$LPIPS" >> $RESULT
