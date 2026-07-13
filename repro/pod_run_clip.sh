#!/bin/bash
# Pod-side: one clip-sweep point. Usage: pod_run_clip.sh <int2|int4> <pct>
set -u
BITS=${1:?}; PCT=${2:?}
TAG=clip_${BITS}_p${PCT}

cd /home/zhizhousha/workspace/video-project/Quant-VideoGen
. .venv/bin/activate
. repro/env_fix.sh
export TRITON_CACHE_DIR=$PWD/repro/triton_cache/$TAG
mkdir -p $TRITON_CACHE_DIR repro/race repro/logs
RESULT=repro/race/result_$TAG.txt
echo "START $(date +%F_%T) tag=$TAG node=${NODE_NAME:-?}" > $RESULT

FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$FREE_MB" -lt 72000 ]; then echo "NODE_BUSY free=${FREE_MB}MiB" >> $RESULT; exit 42; fi

CLIP_PCT=$PCT PYTHONPATH=experiments/LongCat torchrun --nproc_per_node=1 --standalone \
  repro/clip_launcher.py --checkpoint_dir=ckpts/LongCat-Video \
  --workload 480p_long_gen --init_video_path results/longcat/base/1-0.mp4 \
  --output_dir results/clipstudy/$TAG --num_segments 1 --num_cond_frames 73 --seed 0 \
  --prompt_source text_to_video_from_file --prompt assets/t2v.txt --prompt_idx 1 \
  --quant_type triton-nstages-kmeans-$BITS --quant_block_size 64 \
  --cache_num_k_centroids 256 --cache_num_v_centroids 256 \
  --kmeans_max_iters 100 --num_prq_stages 1 \
  > repro/logs/$TAG.log 2>&1
RC=$?
OUT=results/clipstudy/$TAG/1-0/segment_1.mp4
if [ $RC -ne 0 ] || [ ! -f "$OUT" ]; then
  echo "FAILED rc=$RC err=$(grep -m1 -oE 'OutOfMemoryError|RuntimeError[^\"]{0,60}' repro/logs/$TAG.log | head -1)" >> $RESULT
  exit 1
fi
echo "OK" >> $RESULT
