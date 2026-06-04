#!/bin/bash

example_dir=experiments/LingBot/examples/05
prompt=$(cat ${example_dir}/prompt.txt)
ckpt_path=ckpts/LingBot/lingbot-world-base-cam
frame_num=961
seed=42
quant_factor=8

#########################################################
# Quantization Configuration
#########################################################
quant_type="triton-nstages-kmeans-int2"
# quant_type="triton-nstages-kmeans-int4"
cache_num_k_centroids=256
cache_num_v_centroids=256
kmeans_max_iters=4
quant_block_size=16
num_prq_stages=2

quant_dir=${quant_type}_${quant_block_size}/kc_${cache_num_k_centroids}_vc_${cache_num_v_centroids}_nstages_${num_prq_stages}
output_folder=results/lingbot/${quant_dir}

echo "Running inference with checkpoint $ckpt_path and example ${example_dir}"
echo "Output will be saved to $output_folder"

export PYTHONPATH=experiments/LingBot:.

torchrun --nproc_per_node=8 --standalone experiments/LingBot/generate_fast.py \
  --task i2v-A14B \
  --size 480*832 \
  --ckpt_dir $ckpt_path \
  --image ${example_dir}/image.jpg \
  --action_path ${example_dir} \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 8 \
  --frame_num $frame_num \
  --base_seed $seed \
  --save_dir $output_folder \
  --prompt "$prompt" \
  --use_chunked_kv \
  --quant_type $quant_type \
  --cache_num_k_centroids $cache_num_k_centroids \
  --cache_num_v_centroids $cache_num_v_centroids \
  --kmeans_max_iters $kmeans_max_iters \
  --quant_block_size $quant_block_size \
  --num_prq_stages $num_prq_stages \
  --quant_factor $quant_factor
