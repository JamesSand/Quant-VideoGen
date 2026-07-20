#!/bin/bash
cd /home/zhizhousha/workspace/video-project/Quant-VideoGen
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python repro/0720/score_fp8.py $i 8 > repro/0718/logs/scorefp8_$i.log 2>&1 &
done
wait
echo FP8 SCORING DONE
