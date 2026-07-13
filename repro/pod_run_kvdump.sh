#!/bin/bash
# Dump Self-Forcing's real (pre-RoPE) KV cache for the RoPE dispersion study.
set -u
cd /home/zhizhousha/workspace/video-project/Quant-VideoGen
. .venv/bin/activate
. repro/env_fix.sh
export TRITON_CACHE_DIR=$PWD/repro/triton_cache/kvdump
mkdir -p $TRITON_CACHE_DIR repro/race repro/logs results/ropestudy
RESULT=repro/race/result_kvdump.txt
echo "START $(date +%F_%T) node=${NODE_NAME:-?}" > $RESULT

FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$FREE_MB" -lt 72000 ]; then echo "NODE_BUSY free=${FREE_MB}MiB" >> $RESULT; exit 42; fi

DUMP_KV_LEVEL=1 KV_DUMP_DIR=results/ropestudy PYTHONPATH=experiments/Self-Forcing:. \
torchrun --nproc_per_node=1 --standalone experiments/Self-Forcing/inference.py \
  --config_path experiments/Self-Forcing/configs/self_forcing_dmd.yaml \
  --checkpoint_path ckpts/Self-Forcing/self_forcing_dmd.pt \
  --data_path repro/prompt0.txt --output_folder results/ropestudy/video \
  --num_samples 1 --num_output_frames 180 --local_attn_size 180 \
  --use_ema --save_with_index --quant_type none \
  > repro/logs/kvdump.log 2>&1
RC=$?
DUMP=$(ls results/ropestudy/kv_cache_frames*.pt 2>/dev/null | head -1)
if [ $RC -ne 0 ] || [ -z "$DUMP" ]; then
  echo "FAILED rc=$RC dump_missing=$([ -n "$DUMP" ] && echo no || echo yes)" >> $RESULT; exit 1
fi
echo "OK dump=$DUMP size=$(du -h $DUMP | cut -f1)" >> $RESULT
