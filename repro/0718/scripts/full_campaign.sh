#!/bin/bash
cd /home/zhizhousha/workspace/video-project/Quant-VideoGen
D=repro/0718/scripts
echo "=== PHASE1a LC bases $(date +%T)"; bash $D/gpu_queue.sh repro/0718/jobs_phase1a.txt 8
echo "=== PHASE1b LC continuations $(date +%T)"; bash $D/gpu_queue.sh repro/0718/jobs_phase1b.txt 8
echo "=== PHASE3 HY seeds $(date +%T)"; bash $D/gpu_queue.sh repro/0718/jobs_phase3.txt 8
echo "=== PHASE2a SF 700 $(date +%T)"; bash $D/gpu_queue.sh repro/0718/jobs_phase2a.txt 8
echo "=== PHASE2b SF long $(date +%T)"; bash $D/gpu_queue.sh repro/0718/jobs_phase2b_360lat.txt 8
echo "FULL CAMPAIGN GENERATION DONE $(date +%T)"
