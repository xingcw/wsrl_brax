#!/bin/bash
# Example script for finetuning a policy in Brax environment
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
# Use system CUDA 12.9 for ptxas (supports RTX 5090 CC 12.0)
# Disable autotuning to avoid cuBLAS issues on Blackwell (compute capability 12.0)
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda-12.9 --xla_gpu_autotune_level=0"
export PATH="/usr/local/cuda-12.9/bin:$PATH"
# Set cuDNN library path from pip package
CUDNN_PATH=$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])" 2>/dev/null)
if [ -n "$CUDNN_PATH" ]; then
    export LD_LIBRARY_PATH="$CUDNN_PATH/lib:/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH"
else
    export LD_LIBRARY_PATH="/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH"
fi

CKPT_IDX=$1

python finetune_brax.py \
    --brax_env="ant" \
    --brax_backend="generalized" \
    --brax_episode_length=1000 \
    --brax_ckpt_path="/home/chxing/projects/repos/learning2sim2real/checkpoints/ant_sac/20260121_003948" \
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
