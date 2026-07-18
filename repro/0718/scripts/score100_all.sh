#!/bin/bash
# run after generation: 8 scoring shards, one per GPU
cd /home/zhizhousha/workspace/video-project/Quant-VideoGen
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python repro/0718/scripts/score100.py $i 8 > repro/0718/logs/score_shard$i.log 2>&1 &
done
wait
.venv/bin/python repro/0718/scripts/aggregate100.py
echo SCORING DONE
