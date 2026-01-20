export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
# Use system CUDA 12.9 for ptxas (supports RTX 5090 CC 12.0)
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda-12.9"
# Set cuDNN library path from pip package
CUDNN_PATH=$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])" 2>/dev/null)
if [ -n "$CUDNN_PATH" ]; then
    export LD_LIBRARY_PATH="$CUDNN_PATH/lib:/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH"
fi

python3 finetune.py \
--agent cql \
--config experiments/configs/train_config.py:locomotion_cql \
--env halfcheetah-medium-replay-v2 \
--project locomotion-finetune \
--reward_scale 1.0 \
--reward_bias 0.0 \
--num_offline_steps 250_000 \
$@
