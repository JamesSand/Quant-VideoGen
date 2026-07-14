#!/bin/bash
# SF QKV anatomy capture: bf16 180-frame run, 3 layers x 3 time windows,
# Q at last denoise step. Usage: pod_run_qkv.sh
set -u
TAG=qkv_sf

cd /home/zhizhousha/workspace/video-project/Quant-VideoGen
. .venv/bin/activate
. repro/backup/scripts/env_fix.sh
export TRITON_CACHE_DIR=$PWD/repro/backup/triton_cache/$TAG
mkdir -p $TRITON_CACHE_DIR repro/backup/race repro/backup/logs results/kvplot
RESULT=repro/backup/race/result_$TAG.txt
echo "START $(date +%F_%T) tag=$TAG node=${NODE_NAME:-?}" > $RESULT

FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$FREE_MB" -lt 72000 ]; then echo "NODE_BUSY free=${FREE_MB}MiB" >> $RESULT; exit 42; fi

export QKV_OUT=results/kvplot/sf_qkv.pt
export QKV_TARGET=experiments/Self-Forcing/inference.py
export QKV_LAYERS=0,15,29
export QKV_WINDOWS=0-5,87-92,174-179

PYTHONPATH=experiments/Self-Forcing:. DUMP_KV_LEVEL=0 \
torchrun --nproc_per_node=1 --standalone repro/backup/scripts/qkv_capture_launcher.py \
  --config_path experiments/Self-Forcing/configs/self_forcing_dmd.yaml \
  --checkpoint_path ckpts/Self-Forcing/self_forcing_dmd.pt \
  --data_path assets/t2v.txt --output_folder results/kvplot/sf_qkv_gen \
  --num_samples 1 --num_output_frames 180 --local_attn_size 180 \
  --use_ema --save_with_index --quant_type none \
  > repro/backup/logs/$TAG.log 2>&1
RC=$?
if [ $RC -ne 0 ] || [ ! -f "results/kvplot/sf_qkv.pt" ]; then
  echo "FAILED rc=$RC err=$(grep -m1 -oE 'OutOfMemoryError|Error[^\"]{0,80}' repro/backup/logs/$TAG.log | head -1)" >> $RESULT
  exit 1
fi
echo "OK" >> $RESULT
