#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --job-name="build_W16A16_INT8KV"
#SBATCH -c2
#SBATCH --time=00:25:00
#SBATCH --output=/cluster/home/krisskn/master-thesis/trt-scripts/hpc/logs/build_W16A16_INT8KV_%j.out

BASE="/cluster/home/krisskn/master-thesis/trt-scripts"
MODEL_DIR="/cluster/home/krisskn/master-thesis/hf-cache/models/Qwen2.5-3B"

srun \
  /usr/bin/apptainer exec --nv --writable-tmpfs \
  $BASE/hpc/trtllm-tools.sif \
  python $BASE/model/build_engine.py \
    --config $BASE/model/configs/hpc/qwen/build_config_W16A16_INT8KV_LOGITS.json \
    --base $BASE \
    --model-dir $MODEL_DIR
