#!/bin/bash
# mean/PCA KV fake-quant arm. Usage: pod_run_pca.sh <tag> <R> <coeff_bits> <res_grid> <v_mode>
set -u
TAG=${1:?}; R=${2:?}; CB=${3:?}; RG=${4:?}; VM=${5:?}
cd /home/zhizhousha/workspace/video-project/Quant-VideoGen
. .venv/bin/activate
. repro/backup/scripts/env_fix.sh
export TRITON_CACHE_DIR=$PWD/repro/backup/triton_cache/$TAG
mkdir -p $TRITON_CACHE_DIR repro/backup/race repro/backup/logs
RESULT=repro/backup/race/result_$TAG.txt
echo "START $(date +%F_%T) tag=$TAG R=$R cb=$CB res=$RG vm=$VM node=${NODE_NAME:-?}" > $RESULT
FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$FREE_MB" -lt 72000 ]; then echo "NODE_BUSY free=${FREE_MB}MiB" >> $RESULT; exit 42; fi
export PCA_R=$R PCA_COEFF_BITS=$CB PCA_RES_GRID=$RG PCA_V_MODE=$VM
export PCA_TARGET=experiments/LongCat/run_long_t2v.py PYTHONPATH=experiments/LongCat
torchrun --nproc_per_node=1 --standalone repro/backup/scripts/pca_launcher.py \
  --checkpoint_dir=ckpts/LongCat-Video --workload 480p_long_gen \
  --init_video_path results/longcat/base/1-0.mp4 --output_dir results/pcastudy/$TAG \
  --num_segments 1 --num_cond_frames 73 --seed 0 \
  --prompt_source text_to_video_from_file --prompt assets/t2v.txt --prompt_idx 1 \
  --quant_type naive-int2 --quant_block_size 64 \
  > repro/backup/logs/$TAG.log 2>&1
RC=$?
OUT=results/pcastudy/$TAG/1-0/segment_1.mp4
if [ $RC -ne 0 ] || [ ! -f "$OUT" ]; then
  echo "FAILED rc=$RC err=$(grep -m1 -oE 'OutOfMemoryError|Error[^\"]{0,80}' repro/backup/logs/$TAG.log | head -1)" >> $RESULT
  exit 1
fi
echo "OK" >> $RESULT
