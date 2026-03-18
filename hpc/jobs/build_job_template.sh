#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --job-name="build_config_W16A16_LOGITS"
#SBATCH -c4
#SBATCH --time=00-00:30:00
#SBATCH --begin=now

BASE="/workspace/trt-scripts"
MODEL_DIR="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
CONFIG="build_config_W16A16_LOGITS.json"

srun apptainer exec --nv --writable-tmpfs \
  $BASE/hpc/trtllm-tools.sif \
  python $BASE/model/build_engine.py \
    --config $BASE/model/configs/hpc/qwen/$CONFIG \
    --base $BASE \
    --model-dir $MODEL_DIR
