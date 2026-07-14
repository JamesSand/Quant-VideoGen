#!/bin/bash
# Pod-side: one QuaRot+clip sweep point (post-rotation clipping).
# Usage: pod_run_qclip.sh <int2|int4> <clip_ratio> <clip_pct> <tag>
set -u
BITS=${1:?}; RATIO=${2:?}; PCT=${3:?}; TAG=${4:?}; BLK=${5:-16}

cd /home/zhizhousha/workspace/video-project/Quant-VideoGen
. .venv/bin/activate
. repro/backup/scripts/env_fix.sh
export TRITON_CACHE_DIR=$PWD/repro/backup/triton_cache/$TAG
mkdir -p $TRITON_CACHE_DIR repro/backup/race repro/backup/logs
RESULT=repro/backup/race/result_$TAG.txt
echo "START $(date +%F_%T) tag=$TAG bits=$BITS ratio=$RATIO pct=$PCT node=${NODE_NAME:-?}" > $RESULT

FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$FREE_MB" -lt 72000 ]; then echo "NODE_BUSY free=${FREE_MB}MiB" >> $RESULT; exit 42; fi

export QUAROT_BLOCK=$BLK QUAROT_SYM=0 QUAROT_ROTATE_K=1 QUAROT_ROTATE_V=1
export QUAROT_CLIP_RATIO=$RATIO QUAROT_CLIP_PCT=$PCT
export QUAROT_TARGET=experiments/LongCat/run_long_t2v.py
export PYTHONPATH=experiments/LongCat

torchrun --nproc_per_node=1 --standalone repro/backup/scripts/quarot_launcher.py --checkpoint_dir=ckpts/LongCat-Video \
    --workload 480p_long_gen --init_video_path results/longcat/base/1-0.mp4 \
    --output_dir results/qclip/$TAG --num_segments 1 --num_cond_frames 73 --seed 0 \
    --prompt_source text_to_video_from_file --prompt assets/t2v.txt --prompt_idx 1 \
    --quant_type naive-$BITS --quant_block_size $BLK \
    > repro/backup/logs/$TAG.log 2>&1
RC=$?
OUT=results/qclip/$TAG/1-0/segment_1.mp4
if [ $RC -ne 0 ] || [ ! -f "$OUT" ]; then
  echo "FAILED rc=$RC err=$(grep -m1 -oE 'OutOfMemoryError|RuntimeError[^\"]{0,60}' repro/backup/logs/$TAG.log | head -1)" >> $RESULT
  exit 1
fi
echo "OK" >> $RESULT
