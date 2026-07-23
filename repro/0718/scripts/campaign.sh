#!/bin/bash
# Multi-prompt campaign single-job entry (repro/0718/multi-prompt-plan.md).
# Usage: campaign.sh <job>
#   lc_base:<P>            LC 480p_init base video for prompt P (1-10), seed 0
#   lc:<P>:<arm>:<rep>     LC continuation; arm in {bf16,qvg,pca}; rep = repeat idx (seed always 0)
#   sf:<P>:<arm>:<rep>[:frames]  SF generation, default 700 frames
#   hy:<S>:<arm>           HY lake-bridge, seed S (0-4)
# All QVG repeats use seed 0 (nondeterminism source = kmeans atomic_add, not seed;
# reference-paired metrics require same seed as BF16 ref — plan deviation noted).
set -u
JOB=${1:?job spec}
cd /home/zhizhousha/workspace/video-project/Quant-VideoGen
. .venv/bin/activate
. repro/backup/scripts/env_fix.sh
PROMPTS=${CAMPAIGN_PROMPTS:-repro/0718/prompts/selected.txt}
BASE=results/multiprompt${CAMPAIGN_NS:+/$CAMPAIGN_NS}
LOGD=repro/0718/logs; mkdir -p $LOGD
TAG=$(echo "$JOB" | tr ':,' '__')
# pod-local per-GPU triton cache: no quota usage, one writer per cache (race-free)
export TRITON_CACHE_DIR=/tmp/qvg_triton/gpu${CUDA_VISIBLE_DEVICES:-x}
mkdir -p $TRITON_CACHE_DIR
LOG=$LOGD/$TAG.log

QVG_LC="--quant_type triton-nstages-kmeans-int2 --quant_block_size 64 --cache_num_k_centroids 256 --cache_num_v_centroids 256 --kmeans_max_iters 100 --num_prq_stages 1"
QVG_SFHY="--quant_type triton-nstages-kmeans-int2 --quant_block_size 64 --cache_num_k_centroids 256 --cache_num_v_centroids 256 --kmeans_max_iters 2 --num_prq_stages 1"

pca_env_lc()  { export PCA_R=4 PCA_COEFF_BITS=2 PCA_RES_GRID=asym PCA_V_MODE=pca PCA_RES_BLOCK=128; }
pca_env_hy()  { pca_env_lc; export PCA_HALF_R_K=9,0 PCA_HALF_R_V=8,1; }
pca_env_sf()  { export PCA_R=4 PCA_COEFF_BITS=2 PCA_RES_GRID=ternary PCA_V_MODE=pca PCA_RES_BLOCK=64 PCA_SF_STORE_FIX=1; }
# sweep variants (all BPE-legal; suffix after 'pca'):
apply_variant() {  # $1 = arm string like pcar6 / pcaa128 / pcavmean / pcar6vmean / pcav90 / pcav00
  case $1 in
    *hash*)  export PCA_GRID_HASH=1 ;;
  esac
  case $1 in
    *factor*) export PCA_FACTOR_GRID=1 ;;
  esac
  case $1 in
    *r6*)    export PCA_R=6 ;;
  esac
  case $1 in
    *a128*)  export PCA_RES_GRID=asym PCA_RES_BLOCK=128 ;;
    *kptern128*) export PCA_RES_GRID=asym PCA_RES_BLOCK=128 PCA_RES_GRID_KP=ternary PCA_RES_BLOCK_KP=128 ;;
    *kp0*)   export PCA_RES_GRID=asym PCA_RES_BLOCK=128 PCA_RES_GRID_KP=zero ;;
    *kptern*) export PCA_RES_GRID=asym PCA_RES_BLOCK=128 PCA_RES_GRID_KP=ternary PCA_RES_BLOCK_KP=64 ;;
    *vtern*) export PCA_RES_GRID=asym PCA_RES_BLOCK=128 PCA_RES_GRID_V=ternary PCA_RES_BLOCK_V=64 ;;
    *ktern*) export PCA_RES_GRID=asym PCA_RES_BLOCK=128 PCA_RES_GRID_K=ternary PCA_RES_BLOCK_K=64 ;;
    *tern*)  export PCA_RES_GRID=ternary PCA_RES_BLOCK=64 ;;
  esac
  case $1 in
    *r8*)    export PCA_R=8 ;;
  esac
  case $1 in
    *vmean*) export PCA_V_MODE=mean ;;
  esac
  case $1 in
    *v90*)   export PCA_HALF_R_V=9,0 ;;
    *v00*)   export PCA_HALF_R_V=0,0 ;;
  esac
  case $1 in
    *vptern*) export PCA_RES_GRID_VP=ternary PCA_RES_BLOCK_VP=64 ;;
  esac
  case $1 in
    *kax*)   export PCA_RES_AXIS_K=channel ;;
  esac
  case $1 in
    *kptok*) export PCA_RES_AXIS_KP=token ;;
  esac
  case $1 in
    *kb64*)  export PCA_RES_BLOCK_K=64 ;;
  esac
  case $1 in
    *chnk*)  export PCA_RES_CHNORM_K=1 ;;
  esac
  case $1 in
    *ktern32*) export PCA_RES_GRID_K=ternary PCA_RES_BLOCK_K=32 ;;
  esac
  case $1 in
    *fp8*)   export PCA_FP8SIM=1 ;;
  esac
  case $1 in
    *dump*)  export PCA_DUMP_DIR=repro/0720/chunks/$KIND PCA_DUMP_N=8 ;;
  esac
  case $1 in
    *chnv*)  export PCA_RES_CHNORM_V=1 ;;
  esac
  case $1 in
    *vax*)   export PCA_RES_AXIS_V=channel ;;
  esac
  case $1 in
    *vb64*)  export PCA_RES_BLOCK_V=64 ;;
  esac
}

IFS=: read -r KIND A B C D <<< "$JOB"
t0=$(date +%s)
case $KIND in
  lc_base)
    P=$A
    PYTHONPATH=experiments/LongCat torchrun --nproc_per_node=1 --standalone \
      experiments/LongCat/run_long_t2v.py --checkpoint_dir=ckpts/LongCat-Video \
      --workload 480p_init --seed 0 --quant_type none \
      --prompt_source text_to_video_from_file --prompt $PROMPTS --prompt_idx $P \
      --output_dir $BASE/lc/base > $LOG 2>&1
    RC=$?; OUT=$BASE/lc/base/$P-0.mp4 ;;
  lc)
    P=$A; ARM=$B; REP=${C:-0}
    _w=0
    until [ -f $BASE/lc/base/$P-0.mp4 ]; do
      sleep 60; _w=$((_w+60))
      [ $_w -ge 10800 ] && { echo "$JOB BASEWAIT-TIMEOUT" >> repro/0718/logs/ledger_$(hostname -s).txt; exit 3; }
    done
    OUTD=$BASE/lc/${ARM}_rep$REP/p$P
    LCC="--checkpoint_dir=ckpts/LongCat-Video --workload 480p_long_gen \
      --init_video_path $BASE/lc/base/$P-0.mp4 --num_segments 1 --num_cond_frames 73 \
      --seed 0 --prompt_source text_to_video_from_file --prompt $PROMPTS --prompt_idx $P \
      --output_dir $OUTD"
    case $ARM in
      bf16) PYTHONPATH=experiments/LongCat torchrun --nproc_per_node=1 --standalone \
              experiments/LongCat/run_long_t2v.py $LCC --quant_type none > $LOG 2>&1; RC=$? ;;
      qvg)  PYTHONPATH=experiments/LongCat torchrun --nproc_per_node=1 --standalone \
              experiments/LongCat/run_long_t2v.py $LCC $QVG_LC > $LOG 2>&1; RC=$? ;;
      qvgi*) IT=${ARM#qvgi}
            PYTHONPATH=experiments/LongCat torchrun --nproc_per_node=1 --standalone \
              experiments/LongCat/run_long_t2v.py $LCC --quant_type triton-nstages-kmeans-int2 \
              --quant_block_size 64 --cache_num_k_centroids 256 --cache_num_v_centroids 256 \
              --kmeans_max_iters $IT --num_prq_stages 1 > $LOG 2>&1; RC=$? ;;
      qvg4*) export PCA_QVG4=1 PCA_FP8SIM=1
            PCA_TARGET=experiments/LongCat/run_long_t2v.py PYTHONPATH=experiments/LongCat \
              torchrun --nproc_per_node=1 --standalone repro/backup/scripts/pca_launcher.py \
              $LCC --quant_type naive-int2 --quant_block_size 64 > $LOG 2>&1; RC=$? ;;
      qvgpro*) export PCA_QVGPRO=1 PCA_QVGPRO_ITERS=100 PCA_FP8SIM=1
            case $ARM in *proc*) export PCA_QVGPRO_AXIS=channel;; *) export PCA_QVGPRO_AXIS=token;; esac
            PCA_TARGET=experiments/LongCat/run_long_t2v.py PYTHONPATH=experiments/LongCat \
              torchrun --nproc_per_node=1 --standalone repro/backup/scripts/pca_launcher.py \
              $LCC --quant_type naive-int2 --quant_block_size 64 > $LOG 2>&1; RC=$? ;;
      rtn*|kivi*) case $ARM in
              kivipost*) export PCA_KIVI_POST=1 PCA_ROPE_GRID=19,30,52 ;;
              kivipaper*) export PCA_KIVI_PAPER=1 PCA_ROPE_GRID=19,30,52 ;;
              rtn*) export PCA_RTN=1 ;;
              *) export PCA_KIVI=1 ;;
            esac
            case $ARM in *3pt*) export PCA_KIVI_GRID=ternary PCA_RTN_GRID=ternary;; esac
            case $ARM in *fp8*) export PCA_FP8SIM=1;; esac
            PCA_TARGET=experiments/LongCat/run_long_t2v.py PYTHONPATH=experiments/LongCat \
              torchrun --nproc_per_node=1 --standalone repro/backup/scripts/pca_launcher.py \
              $LCC --quant_type naive-int2 --quant_block_size 64 > $LOG 2>&1; RC=$? ;;
      quarot) QUAROT_BLOCK=64 QUAROT_SYM=0 QUAROT_ROTATE_K=1 QUAROT_ROTATE_V=1 \
            QUAROT_TARGET=experiments/LongCat/run_long_t2v.py PYTHONPATH=experiments/LongCat \
              torchrun --nproc_per_node=1 --standalone repro/backup/scripts/quarot_launcher.py \
              $LCC --quant_type naive-int2 --quant_block_size 64 > $LOG 2>&1; RC=$? ;;
      pca*) pca_env_lc; apply_variant $ARM
            PCA_TARGET=experiments/LongCat/run_long_t2v.py PYTHONPATH=experiments/LongCat \
              torchrun --nproc_per_node=1 --standalone repro/backup/scripts/pca_launcher.py \
              $LCC --quant_type naive-int2 --quant_block_size 64 > $LOG 2>&1; RC=$? ;;
    esac
    OUT=$OUTD/$P-0/segment_1.mp4 ;;
  sf)
    P=$A; ARM=$B; REP=${C:-0}; NF=${D:-180}   # NF is in LATENT frames (must be %3==0); 180 latents = 717px -> VBench700 window
    OUTD=$BASE/sf/${ARM}_rep${REP}_f$NF/p$P
    SFC="--config_path experiments/Self-Forcing/configs/self_forcing_dmd.yaml \
      --checkpoint_path ckpts/Self-Forcing/self_forcing_dmd.pt \
      --data_path ${CAMPAIGN_SF_DIR:-repro/0718/prompts}/p$P.txt --output_folder $OUTD \
      --num_samples 1 --num_output_frames $NF --local_attn_size 195 --use_ema --seed 0 --save_with_index"
    case $ARM in
      bf16) PYTHONPATH=experiments/Self-Forcing:. torchrun --nproc_per_node=1 --standalone \
              experiments/Self-Forcing/inference.py $SFC --quant_type none > $LOG 2>&1; RC=$? ;;
      qvg)  PYTHONPATH=experiments/Self-Forcing:. torchrun --nproc_per_node=1 --standalone \
              experiments/Self-Forcing/inference.py $SFC $QVG_SFHY > $LOG 2>&1; RC=$? ;;
      qvgpro*) export PCA_QVGPRO=1 PCA_QVGPRO_ITERS=2 PCA_SF_STORE_FIX=1 PCA_FP8SIM=1
            case $ARM in *proc*) export PCA_QVGPRO_AXIS=channel;; *) export PCA_QVGPRO_AXIS=token;; esac
            PCA_TARGET=experiments/Self-Forcing/inference.py PYTHONPATH=experiments/Self-Forcing:. \
              torchrun --nproc_per_node=1 --standalone repro/backup/scripts/pca_launcher.py \
              $SFC --quant_type naive-int2 --quant_block_size 64 > $LOG 2>&1; RC=$? ;;
      rtn*|kivi*) case $ARM in rtn*) export PCA_RTN=1;; *) export PCA_KIVI=1;; esac
            case $ARM in *fp8*) export PCA_FP8SIM=1;; esac
            export PCA_SF_STORE_FIX=1
            PCA_TARGET=experiments/Self-Forcing/inference.py PYTHONPATH=experiments/Self-Forcing:. \
              torchrun --nproc_per_node=1 --standalone repro/backup/scripts/pca_launcher.py \
              $SFC --quant_type naive-int2 --quant_block_size 64 > $LOG 2>&1; RC=$? ;;
      quarot) QUAROT_BLOCK=64 QUAROT_SYM=0 QUAROT_ROTATE_K=1 QUAROT_ROTATE_V=1 QUAROT_SF_STORE_FIX=1 \
            QUAROT_TARGET=experiments/Self-Forcing/inference.py PYTHONPATH=experiments/Self-Forcing:. \
              torchrun --nproc_per_node=1 --standalone repro/backup/scripts/quarot_launcher.py \
              $SFC --quant_type naive-int2 --quant_block_size 64 > $LOG 2>&1; RC=$? ;;
      pca*) pca_env_sf; apply_variant $ARM
            PCA_TARGET=experiments/Self-Forcing/inference.py PYTHONPATH=experiments/Self-Forcing:. \
              torchrun --nproc_per_node=1 --standalone repro/backup/scripts/pca_launcher.py \
              $SFC --quant_type naive-int2 --quant_block_size 64 > $LOG 2>&1; RC=$? ;;
    esac
    OUT=$(ls $OUTD/*.mp4 2>/dev/null | head -1) ;;
  hy)
    S=$A; ARM=$B
    PROMPT='A paved pathway leads towards a stone arch bridge spanning a calm body of water.  Lush green trees and foliage line the path and the far bank of the water. A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. The water reflects the surrounding greenery and the sky.  The scene is bathed in soft, natural light, creating a tranquil and serene atmosphere. The pathway is composed of large, rectangular stones, and the bridge is constructed of light gray stone.  The overall composition emphasizes the peaceful and harmonious nature of the landscape.'
    POSE="w-8,s-8,a-8,d-8,up-8"
    OUTD=$BASE/hy/${ARM}_s$S
    export LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    HYC="--input \$PROMPT --image_path assets/hyworld.png --num_chunk 11 --pose $POSE \
      --ar_model_path ckpts/HY-WorldPlay/wan_transformer \
      --ckpt_path ckpts/HY-WorldPlay/wan_distilled_model/model.pt \
      --offload_text_encoder --out $OUTD --seed $S \
      --memory_frames 48 --temporal_context_size 44 --pred_latent_size 4"
    case $ARM in
      bf16) PYTHONPATH=experiments/HY-WorldPlay:experiments/HY-WorldPlay/wan \
              torchrun --nproc_per_node=1 --standalone experiments/HY-WorldPlay/wan/generate.py \
              --input "$PROMPT" --image_path assets/hyworld.png --num_chunk 11 --pose "$POSE" \
              --ar_model_path ckpts/HY-WorldPlay/wan_transformer \
              --ckpt_path ckpts/HY-WorldPlay/wan_distilled_model/model.pt \
              --offload_text_encoder --out $OUTD --seed $S \
              --memory_frames 48 --temporal_context_size 44 --pred_latent_size 4 \
              --quant_type none > $LOG 2>&1; RC=$? ;;
      qvg)  PYTHONPATH=experiments/HY-WorldPlay:experiments/HY-WorldPlay/wan \
              torchrun --nproc_per_node=1 --standalone experiments/HY-WorldPlay/wan/generate.py \
              --input "$PROMPT" --image_path assets/hyworld.png --num_chunk 11 --pose "$POSE" \
              --ar_model_path ckpts/HY-WorldPlay/wan_transformer \
              --ckpt_path ckpts/HY-WorldPlay/wan_distilled_model/model.pt \
              --offload_text_encoder --out $OUTD --seed $S \
              --memory_frames 48 --temporal_context_size 44 --pred_latent_size 4 \
              $QVG_SFHY > $LOG 2>&1; RC=$? ;;
      rtn|kivi) [ "$ARM" = rtn ] && export PCA_RTN=1 || export PCA_KIVI=1
            PCA_TARGET=experiments/HY-WorldPlay/wan/generate.py \
              PYTHONPATH=experiments/HY-WorldPlay:experiments/HY-WorldPlay/wan \
              torchrun --nproc_per_node=1 --standalone repro/backup/scripts/pca_launcher.py \
              --input "$PROMPT" --image_path assets/hyworld.png --num_chunk 11 --pose "$POSE" \
              --ar_model_path ckpts/HY-WorldPlay/wan_transformer \
              --ckpt_path ckpts/HY-WorldPlay/wan_distilled_model/model.pt \
              --offload_text_encoder --out $OUTD --seed $S \
              --memory_frames 48 --temporal_context_size 44 --pred_latent_size 4 \
              --quant_type naive-int2 --quant_block_size 64 > $LOG 2>&1; RC=$? ;;
      quarot) QUAROT_BLOCK=64 QUAROT_SYM=0 QUAROT_ROTATE_K=1 QUAROT_ROTATE_V=1 \
            QUAROT_TARGET=experiments/HY-WorldPlay/wan/generate.py \
              PYTHONPATH=experiments/HY-WorldPlay:experiments/HY-WorldPlay/wan \
              torchrun --nproc_per_node=1 --standalone repro/backup/scripts/quarot_launcher.py \
              --input "$PROMPT" --image_path assets/hyworld.png --num_chunk 11 --pose "$POSE" \
              --ar_model_path ckpts/HY-WorldPlay/wan_transformer \
              --ckpt_path ckpts/HY-WorldPlay/wan_distilled_model/model.pt \
              --offload_text_encoder --out $OUTD --seed $S \
              --memory_frames 48 --temporal_context_size 44 --pred_latent_size 4 \
              --quant_type naive-int2 --quant_block_size 64 > $LOG 2>&1; RC=$? ;;
      pca*) pca_env_hy; apply_variant $ARM
            PCA_TARGET=experiments/HY-WorldPlay/wan/generate.py \
              PYTHONPATH=experiments/HY-WorldPlay:experiments/HY-WorldPlay/wan \
              torchrun --nproc_per_node=1 --standalone repro/backup/scripts/pca_launcher.py \
              --input "$PROMPT" --image_path assets/hyworld.png --num_chunk 11 --pose "$POSE" \
              --ar_model_path ckpts/HY-WorldPlay/wan_transformer \
              --ckpt_path ckpts/HY-WorldPlay/wan_distilled_model/model.pt \
              --offload_text_encoder --out $OUTD --seed $S \
              --memory_frames 48 --temporal_context_size 44 --pred_latent_size 4 \
              --quant_type naive-int2 --quant_block_size 64 > $LOG 2>&1; RC=$? ;;
    esac
    OUT=$OUTD/0-$S.mp4 ;;
  *) echo "unknown job $JOB"; exit 2 ;;
esac

WALL=$(( $(date +%s) - t0 ))
HIJACK=$(grep -ci 'hijack' $LOG 2>/dev/null || true)
STATUS=OK
[ $RC -ne 0 ] && STATUS=FAIL
[ ! -f "${OUT:-/nonexistent}" ] && STATUS=NOFILE
# hygiene: pca arms must show hijack>0
case "$JOB" in *:pca*|*pca:*|*:rtn*|*:kivi*) [ "${HIJACK:-0}" -eq 0 ] && STATUS=NOHIJACK ;; esac
echo "$JOB $STATUS rc=$RC wall=${WALL}s hijack=$HIJACK out=$OUT" >> repro/0718/logs/ledger_$(hostname -s).txt
echo "$JOB $STATUS rc=$RC wall=${WALL}s hijack=$HIJACK"
[ "$STATUS" = OK ]
