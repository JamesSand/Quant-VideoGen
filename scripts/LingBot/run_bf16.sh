#!/bin/bash

example_dir=experiments/LingBot/examples/05
prompt=$(cat ${example_dir}/prompt.txt)
ckpt_path=ckpts/LingBot/lingbot-world-base-cam
frame_num=961
seed=42

output_folder=results/lingbot/bf16

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
  --prompt "$prompt"
