#!/bin/bash

# =========================================
# srun_train.sh -- Run LLaMA-Factory training with srun
# Usage: bash srun_train.sh <config.yaml>
# Example:
# bash srun_train_hpo.sh examples/train_hpo/llama3.1_hpo.yaml cpts_hpo
# bash srun_train_hpo.sh examples/train_hpo/llama3.1_simpo.yaml cpts_simpo
# bash srun_train_hpo.sh examples/train_hpo/llama3.1_wpo.yaml cpts_wpo-promptretain0.2-2epoch
# bash srun_train_hpo.sh examples/train_hpo/llama3.1_hwpo.yaml cpts_hwpo
# =========================================
# 日志会生成在：train_log/dfki_sft_qwen3_4b_think_20251029_145512.log

# 检查参数
# if [ $# -ne 1 ]; then
#     echo "Usage: $0 <config.yaml>"
#     exit 1
# fi

# CONFIG_FILE="$1"


if [ $# -ne 2 ]; then
    echo "Usage: $0 <config.yaml> <logdir>"
    exit 1
fi

CONFIG_FILE="$1"
LOGDIR="$2"

# 设置工作目录
WORKDIR="/netscratch/dshi/projects/MPO/"
cd "$WORKDIR" || { echo "Failed to cd $WORKDIR"; exit 1; }

# 创建日志目录
# LOGDIR="$WORKDIR/train_logs"
mkdir -p "$LOGDIR"

# 生成带时间戳的日志文件
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="$LOGDIR/$(basename "$CONFIG_FILE" .yaml)_${TIMESTAMP}.log"

export WANDB_API_KEY="1be0fb12f1f3c5f2b090fc4f64a51f36d74ca634"
# 运行训练
srun -p B200 --mem=280GB --nodes=1 --ntasks-per-node=1 --gpus-per-task=2 --cpus-per-task=8 \
    --container-image=/enroot/ubuntu22.04+py3.10+cu126.sqsh \
    --container-mounts=/netscratch/dshi:/netscratch/dshi,/ds:/ds,$WORKDIR:$WORKDIR \
    --container-workdir="$WORKDIR" \
    bash -c '
        source /netscratch/dshi/projects/envs/anaconda/etc/profile.d/conda.sh
        conda activate mpo
        FORCE_TORCHRUN=1 CUDA_LAUNCH_BLOCKING=1 llamafactory-cli train '"$CONFIG_FILE"'
    ' \
    2>&1 | tee "$LOGFILE"

echo "Training finished. Log saved to $LOGFILE"

# FORCE_TORCHRUN=1 CUDA_LAUNCH_BLOCKING=1 python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m llamafactory.cli train examples/train_mpo/qwen2.5_mpo.yaml
# FORCE_TORCHRUN=1 CUDA_LAUNCH_BLOCKING=1 llamafactory-cli train examples/train_mpo/qwen2.5_mpo.yaml
# FORCE_TORCHRUN=1 CUDA_LAUNCH_BLOCKING=1 llamafactory-cli train examples/train_hpo/llama3.1_hpo.yaml