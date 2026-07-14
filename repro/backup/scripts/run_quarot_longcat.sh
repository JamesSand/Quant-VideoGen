#!/bin/bash
# QuaRot baseline run for LongCat (1 segment, seg1 protocol).
# Usage: run_quarot_longcat.sh <int2|int4> <block> <sym:0|1> <rotate:1|0> <outtag>
set -e
BITS=${1:?bits}; BLOCK=${2:?block}; SYM=${3:?sym}; ROT=${4:?rotate}; TAG=${5:?tag}

export QUAROT_BLOCK=$BLOCK QUAROT_SYM=$SYM QUAROT_ROTATE_K=$ROT QUAROT_ROTATE_V=$ROT
export QUAROT_TARGET=experiments/LongCat/run_long_t2v.py
export PYTHONPATH=experiments/LongCat

torchrun --nproc_per_node=1 --standalone repro/backup/scripts/quarot_launcher.py --checkpoint_dir=ckpts/LongCat-Video \
    --workload 480p_long_gen \
    --init_video_path results/longcat/base/1-0.mp4 \
    --output_dir results/quarot/${TAG} \
    --num_segments 1 --num_cond_frames 73 --seed 0 \
    --prompt_source text_to_video_from_file --prompt assets/t2v.txt --prompt_idx 1 \
    --quant_type naive-${BITS} --quant_block_size ${BLOCK}
