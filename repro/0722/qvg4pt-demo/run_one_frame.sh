#!/bin/bash
# qvg4pt 一帧最小 demo:bash repro/0722/qvg4pt-demo/run_one_frame.sh [prompt_idx]
set -eu
cd /home/zhizhousha/workspace/video-project/Quant-VideoGen
. .venv/bin/activate
. repro/backup/scripts/env_fix.sh
export TRITON_CACHE_DIR=/tmp/qvg_triton/gpu${CUDA_VISIBLE_DEVICES:-x}
export ONEFRAME_P=${1:-1}
torchrun --nproc_per_node=1 --standalone repro/0722/qvg4pt-demo/minimal_one_frame.py
ls -la repro/0722/qvg4pt-demo/demo_out/oneframe_f93.png
