#!/bin/bash
cd /home/zhizhousha/workspace/video-project/Quant-VideoGen
export CAMPAIGN_NS=mp100 CAMPAIGN_PROMPTS=repro/0718/prompts100/selected.txt CAMPAIGN_SF_DIR=repro/0718/prompts100
grep -v '^hy' repro/0718/jobs_final1.txt > /tmp/final1_sf.txt
bash repro/0718/scripts/gpu_queue.sh /tmp/final1_sf.txt 8
unset CAMPAIGN_NS CAMPAIGN_PROMPTS CAMPAIGN_SF_DIR
grep '^hy' repro/0718/jobs_final1.txt > /tmp/final1_hy.txt
bash repro/0718/scripts/gpu_queue.sh /tmp/final1_hy.txt 8
# LC kaxvax full-100 vbench
.venv/bin/python repro/0718/scripts/vbench4.py $(ls results/multiprompt/mp100/lc/pcakaxvax_rep0/p*/*/segment_1.mp4)
echo FINAL1 DONE
