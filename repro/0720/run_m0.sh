#!/bin/bash
cd /home/zhizhousha/workspace/video-project/Quant-VideoGen
export CAMPAIGN_NS=mp100 CAMPAIGN_PROMPTS=repro/0718/prompts100/selected.txt CAMPAIGN_SF_DIR=repro/0718/prompts100
grep -v '^hy' repro/0720/jobs_m0.txt > /tmp/m0_lcsf.txt
bash repro/0718/scripts/gpu_queue.sh /tmp/m0_lcsf.txt 8
unset CAMPAIGN_NS CAMPAIGN_PROMPTS CAMPAIGN_SF_DIR
grep '^hy' repro/0720/jobs_m0.txt > /tmp/m0_hy.txt
bash repro/0718/scripts/gpu_queue.sh /tmp/m0_hy.txt 8
echo M0 GEN DONE
