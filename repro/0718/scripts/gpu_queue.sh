#!/bin/bash
# Run a job file (one campaign.sh job spec per line) across N GPUs.
# Usage: gpu_queue.sh <jobs_file> [num_gpus]
set -u
JOBS=${1:?jobs file}
NG=${2:-8}
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
    echo "[gpu$gpu] START $job $(date +%T)"
    CUDA_VISIBLE_DEVICES=$gpu bash $DIR/campaign.sh "$job"
    echo "[gpu$gpu] DONE  $job $(date +%T)"
    rm -f "$f.taken.$gpu"
  done
}
for g in $(seq 0 $((NG-1))); do worker $g & done
wait
rm -rf $QD
echo "QUEUE COMPLETE: $JOBS"
