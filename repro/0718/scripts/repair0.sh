#!/bin/bash
cd /home/zhizhousha/workspace/video-project/Quant-VideoGen
GPUS="0 1" bash repro/0718/scripts/gpu_queue.sh repro/0718/jobs_repair1.txt
GPUS="0 1 2 5" bash repro/0718/scripts/gpu_queue.sh repro/0718/jobs_repair2.txt
echo "REPAIR0 DONE"
