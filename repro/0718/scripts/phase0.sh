#!/bin/bash
set -u
cd /home/zhizhousha/workspace/video-project/Quant-VideoGen
D=repro/0718/scripts
bash $D/gpu_queue.sh repro/0718/jobs_phase0a.txt 8
bash $D/gpu_queue.sh repro/0718/jobs_phase0b.txt 8
echo "PHASE0 GENERATION DONE $(date +%T)"
