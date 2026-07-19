#!/bin/bash
cd /home/zhizhousha/workspace/video-project/Quant-VideoGen
export CAMPAIGN_NS=mp100 CAMPAIGN_PROMPTS=repro/0718/prompts100/selected.txt CAMPAIGN_SF_DIR=repro/0718/prompts100
grep -v '^hy' repro/0718/jobs_kax5.txt > /tmp/kax5_sf.txt
grep '^hy' repro/0718/jobs_kax5.txt > /tmp/kax5_hy.txt
bash repro/0718/scripts/gpu_queue.sh /tmp/kax5_sf.txt 8
unset CAMPAIGN_NS CAMPAIGN_PROMPTS CAMPAIGN_SF_DIR
bash repro/0718/scripts/gpu_queue.sh /tmp/kax5_hy.txt 8
echo KAX5 DONE
