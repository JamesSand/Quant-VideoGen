#!/bin/bash
# Run a job file (one campaign.sh job spec per line) across N GPUs.
# Usage: gpu_queue.sh <jobs_file> [num_gpus]
set -u
JOBS=${1:?jobs file}
NG=${2:-8}
GPUS=${GPUS:-$(seq 0 $((NG-1)) | tr '\n' ' ')}   # override with GPUS="3 4 5"
DIR=$(cd "$(dirname "$0")" && pwd)
QD=$(mktemp -d)
i=0
grep -v '^\s*\(#\|$\)' "$JOBS" | while read -r j; do echo "$j" > $QD/$(printf '%04d' $i).job; i=$((i+1)); done
worker() {
  local gpu=$1
  while true; do
    local f
    f=$(ls $QD/*.job 2>/dev/null | head -1) || true
    [ -z "$f" ] && break
    mv "$f" "$f.taken.$gpu" 2>/dev/null || continue  # atomic claim
    local job=$(cat "$f.taken.$gpu")
    # wait until the GPU is actually free (leftover process guard), max 15 min
    local waited=0
    while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu)" -gt 2000 ]; do
      sleep 30; waited=$((waited+30))
      if [ $waited -ge 900 ]; then echo "[gpu$gpu] GPUBUSY giving job back: $job"; mv "$f.taken.$gpu" "$f"; sleep 60; continue 2; fi
    done
    echo "[gpu$gpu] START $job $(date +%T)"
    CUDA_VISIBLE_DEVICES=$gpu bash $DIR/campaign.sh "$job"
    echo "[gpu$gpu] DONE  $job $(date +%T)"
    rm -f "$f.taken.$gpu"
  done
}
for g in $GPUS; do worker $g & done
wait
rm -rf $QD
echo "QUEUE COMPLETE: $JOBS"
