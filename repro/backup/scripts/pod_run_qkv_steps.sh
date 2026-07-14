#!/bin/bash
# Capture Q/K/V at EVERY denoise step for one block (timestep-axis dynamics).
set -u
TAG=qkv_steps_sf
cd /home/zhizhousha/workspace/video-project/Quant-VideoGen
. .venv/bin/activate
. repro/backup/scripts/env_fix.sh
export TRITON_CACHE_DIR=$PWD/repro/backup/triton_cache/$TAG
mkdir -p $TRITON_CACHE_DIR repro/backup/race repro/backup/logs results/kvplot
RESULT=repro/backup/race/result_$TAG.txt
echo "START $(date +%F_%T) tag=$TAG node=${NODE_NAME:-?}" > $RESULT
FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$FREE_MB" -lt 72000 ]; then echo "NODE_BUSY free=${FREE_MB}MiB" >> $RESULT; exit 42; fi
export QKV_OUT=results/kvplot/sf_qkv_steps.pt QKV_TARGET=experiments/Self-Forcing/inference.py
export QKV_LAYERS=0,15,29 QKV_WINDOWS=87-89 QKV_ALL_STEPS=1
PYTHONPATH=experiments/Self-Forcing:. DUMP_KV_LEVEL=0 \
torchrun --nproc_per_node=1 --standalone repro/backup/scripts/qkv_capture_launcher.py \
  --config_path experiments/Self-Forcing/configs/self_forcing_dmd.yaml \
  --checkpoint_path ckpts/Self-Forcing/self_forcing_dmd.pt \
  --data_path assets/t2v.txt --output_folder results/kvplot/sf_qkv_steps_gen \
  --num_samples 1 --num_output_frames 180 --local_attn_size 180 \
  --use_ema --save_with_index --quant_type none \
  > repro/backup/logs/$TAG.log 2>&1
RC=$?
if [ $RC -ne 0 ] || [ ! -f "results/kvplot/sf_qkv_steps.pt" ]; then
  echo "FAILED rc=$RC" >> $RESULT; exit 1
fi
echo "OK" >> $RESULT
