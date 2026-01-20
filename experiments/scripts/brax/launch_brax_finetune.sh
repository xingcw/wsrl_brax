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

# Default values
BRAX_ENV=${BRAX_ENV:-"ant"}
BRAX_EPISODE_LENGTH=${BRAX_EPISODE_LENGTH:-1000}
BRAX_CKPT_PATH=${BRAX_CKPT_PATH:-"/home/chxing/projects/repos/learning2sim2real/checkpoints/ant_sac/20260120_002608"}  # Path to Brax SAC checkpoint (optional)
BRAX_CKPT_IDX=${BRAX_CKPT_IDX:--1}    # Checkpoint index (-1 for latest)
BRAX_HIDDEN_DIMS=${BRAX_HIDDEN_DIMS:-"32, 32"}  # Hidden layer sizes
SEED=${SEED:-0}
AGENT=${AGENT:-"sac"}
NUM_ONLINE_STEPS=${NUM_ONLINE_STEPS:-500000}
UTD=${UTD:-4}
BATCH_SIZE=${BATCH_SIZE:-256}
EVAL_INTERVAL=${EVAL_INTERVAL:-10000}
DEBUG=${DEBUG:-"True"}  # Set to False to enable wandb logging

# Run the finetuning script
python finetune_brax.py \
    --brax_env="${BRAX_ENV}" \
    --brax_backend="generalized" \
    --brax_episode_length=${BRAX_EPISODE_LENGTH} \
    --brax_ckpt_path="${BRAX_CKPT_PATH}" \
    --brax_ckpt_idx=${BRAX_CKPT_IDX} \
    --brax_hidden_dims="${BRAX_HIDDEN_DIMS}" \
    --brax_native_eval=True \
    --brax_num_eval_envs=128 \
    --load_brax_q_network=True \
    --agent="${AGENT}" \
    --seed=${SEED} \
    --num_online_steps=${NUM_ONLINE_STEPS} \
    --utd=${UTD} \
    --batch_size=${BATCH_SIZE} \
    --eval_interval=${EVAL_INTERVAL} \
    --reward_scale=1.0 \
    --reward_bias=0.0 \
    --debug=${DEBUG} \
    --config=experiments/configs/brax_sac_config.py \
    --exp_name="brax_${BRAX_ENV}_finetune"
