#!/bin/bash
cd /home/zhizhousha/workspace/video-project/Quant-VideoGen
export CAMPAIGN_NS=mp100 CAMPAIGN_PROMPTS=repro/0718/prompts100/selected.txt CAMPAIGN_SF_DIR=repro/0718/prompts100
grep -v '^hy' repro/0718/jobs_kax4.txt > /tmp/kax4_lc.txt
grep '^hy' repro/0718/jobs_kax4.txt > /tmp/kax4_hy.txt
bash repro/0718/scripts/gpu_queue.sh /tmp/kax4_lc.txt 8
unset CAMPAIGN_NS CAMPAIGN_PROMPTS CAMPAIGN_SF_DIR
bash repro/0718/scripts/gpu_queue.sh /tmp/kax4_hy.txt 8
# score the SF kax arm (regenerated in round 2, unscored due to crash)
.venv/bin/python repro/0718/scripts/vbench4.py $(ls results/multiprompt/mp100/sf/pcaa128kax_rep0_f180/p*/*.mp4) --max-frames 700
echo KAX4 DONE
