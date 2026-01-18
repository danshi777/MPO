#!/bin/bash

# =========================================
# srun.sh -- Run LLaMA-Factory training with srun
# Usage: bash dfki_srun.sh <config.yaml>
# Example: 
# cd /netscratch/dshi/projects/llama-factory-and-verl/LLaMA-Factory
# bash dfki_srun.sh dfki_sft_qwen3_4b_think.yaml

# bash srun.sh examples/train_mpo/qwen2.5_mpo.yaml
# =========================================
# 日志会生成在：train_log/dfki_sft_qwen3_4b_think_20251029_145512.log

# 检查参数
# if [ $# -ne 1 ]; then
#     echo "Usage: $0 <config.yaml>"
#     exit 1
# fi

# CONFIG_FILE="$1"

# 设置工作目录
WORKDIR="/netscratch/dshi/projects/MPO/"
cd "$WORKDIR" || { echo "Failed to cd $WORKDIR"; exit 1; }

# 创建日志目录
LOGDIR="$WORKDIR/train_logs"
mkdir -p "$LOGDIR"

# 生成带时间戳的日志文件
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="$LOGDIR/$(basename "$CONFIG_FILE" .yaml)_${TIMESTAMP}.log"

export WANDB_API_KEY="1be0fb12f1f3c5f2b090fc4f64a51f36d74ca634"
# 运行训练
srun -p H200 --mem=280GB --nodes=1 --ntasks-per-node=1 --gpus-per-task=2 --cpus-per-task=8 \
    --container-image=/enroot/ubuntu22.04+py3.10+cu126.sqsh \
    --container-mounts=/netscratch/dshi:/netscratch/dshi,/ds:/ds,$WORKDIR:$WORKDIR \
    --container-workdir="$WORKDIR" \
    bash -c '
        source "/netscratch/dshi/projects/envs/anaconda/etc/profile.d/conda.sh"
        conda activate mpo
        python src/llamafactory/data/generate_mean_harmless.py --model_id_or_path=/ds/models/llms/Llama-3.1-8B-Instruct --model_name=llama31-8b
        python src/llamafactory/data/generate_mean_harmless.py --model_id_or_path=/netscratch/dshi/models/Qwen2.5-7B-Instruct --model_name=qwen25-7b
    '
    # 2>&1 | tee "$LOGFILE"

# echo "Training finished. Log saved to $LOGFILE"

# FORCE_TORCHRUN=1 CUDA_LAUNCH_BLOCKING=1 python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m llamafactory.cli train examples/train_mpo/qwen2.5_mpo.yaml
# FORCE_TORCHRUN=1 CUDA_LAUNCH_BLOCKING=1 llamafactory-cli train examples/train_mpo/qwen2.5_mpo.yaml