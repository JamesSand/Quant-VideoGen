#!/bin/bash
# Pod-side: LC baseline with N segments (for long-window protocol search).
# Usage: pod_run_long.sh <int2|int4> <block> <sym> <rotate> <tag> <num_segments>
set -u
BITS=${1:?}; BLOCK=${2:?}; SYM=${3:?}; ROT=${4:?}; TAG=${5:?}; NSEG=${6:?}

cd /home/zhizhousha/workspace/video-project/Quant-VideoGen
. .venv/bin/activate
. repro/backup/scripts/env_fix.sh
export TRITON_CACHE_DIR=$PWD/repro/backup/triton_cache/pod_$TAG
mkdir -p $TRITON_CACHE_DIR repro/backup/race repro/backup/logs
RESULT=repro/backup/race/result_$TAG.txt
echo "START $(date +%F_%T) tag=$TAG bits=$BITS block=$BLOCK sym=$SYM rot=$ROT nseg=$NSEG node=${NODE_NAME:-?}" > $RESULT

FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$FREE_MB" -lt 72000 ]; then echo "NODE_BUSY free=${FREE_MB}MiB" >> $RESULT; exit 42; fi

export QUAROT_BLOCK=$BLOCK QUAROT_SYM=$SYM QUAROT_ROTATE_K=$ROT QUAROT_ROTATE_V=$ROT
export QUAROT_TARGET=experiments/LongCat/run_long_t2v.py
export PYTHONPATH=experiments/LongCat
LAUNCHER=repro/backup/scripts/quarot_launcher.py
[ "$ROT" = "0" ] && export QUAROT_DISABLE=1

torchrun --nproc_per_node=1 --standalone $LAUNCHER --checkpoint_dir=ckpts/LongCat-Video \
    --workload 480p_long_gen --init_video_path results/longcat/base/1-0.mp4 \
    --output_dir results/quarot/$TAG --num_segments $NSEG --num_cond_frames 73 --seed 0 \
    --prompt_source text_to_video_from_file --prompt assets/t2v.txt --prompt_idx 1 \
    --quant_type naive-$BITS --quant_block_size $BLOCK \
    > repro/backup/logs/quarot_$TAG.log 2>&1
RC=$?
OUT=results/quarot/$TAG/1-0/segment_$NSEG.mp4
if [ $RC -ne 0 ] || [ ! -f "$OUT" ]; then
  echo "FAILED rc=$RC out_missing=$([ -f "$OUT" ] && echo no || echo yes)" >> $RESULT; exit 1
fi
echo "OK segments=$NSEG" >> $RESULT
