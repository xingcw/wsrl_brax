#!/bin/bash
# Example script for finetuning a policy in Brax environment

CKPT_IDX=$1

python finetune_brax.py \
    --brax_env="ant" \
    --brax_backend="generalized" \
    --brax_episode_length=1000 \
    --brax_ckpt_path="$HOME/learning2sim2real/checkpoints/ant_sac/20260122_012045" \
    --brax_ckpt_idx=$CKPT_IDX \
    --brax_hidden_dims="32, 32" \
    --brax_num_eval_envs=128 \
    --load_brax_q_network=True \
    --agent="sac" \
    --seed=0 \
    --num_online_steps=100000 \
    --utd=4 \
    --batch_size=256 \
    --eval_interval=10000 \
    --reward_scale=10.0 \
    --reward_bias=0.0 \
    --debug=False \
    --config=experiments/configs/brax_sac_config.py \
    --exp_name="brax_ant_finetune"
