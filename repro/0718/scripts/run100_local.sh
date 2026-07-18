#!/bin/bash
cd /home/zhizhousha/workspace/video-project/Quant-VideoGen
export CAMPAIGN_NS=mp100 CAMPAIGN_PROMPTS=repro/0718/prompts100/selected.txt CAMPAIGN_SF_DIR=repro/0718/prompts100
D=repro/0718/scripts
echo "=== MP100 bases $(date +%T)"; bash $D/gpu_queue.sh repro/0718/jobs100_bases.txt 8
unset CAMPAIGN_NS CAMPAIGN_PROMPTS CAMPAIGN_SF_DIR
echo "=== HY baselines $(date +%T)"; bash $D/gpu_queue.sh repro/0718/jobs100_hy.txt 8
export CAMPAIGN_NS=mp100 CAMPAIGN_PROMPTS=repro/0718/prompts100/selected.txt CAMPAIGN_SF_DIR=repro/0718/prompts100
echo "=== MP100 LC/SF main $(date +%T)"; bash $D/gpu_queue.sh repro/0718/jobs100_part0.txt 8
echo "MP100 LOCAL DONE $(date +%T)"
