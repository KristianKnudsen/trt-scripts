#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --job-name="build_W8A8_SQ_LOGITS"
#SBATCH -c4
#SBATCH --time=00-00:30:00
#SBATCH --begin=now

BASE="/cluster/home/krisskn/master-thesis/trt-scripts"
MODEL_DIR="/cluster/home/krisskn/master-thesis/hf-cache/models/Qwen2.5-3B"
CONFIG="build_config_W8A8_SQ_LOGITS.json"

srun /usr/bin/apptainer exec --nv --writable-tmpfs \
  $BASE/hpc/trtllm-tools.sif \
  python $BASE/model/build_engine.py \
    --config $BASE/model/configs/hpc/qwen/$CONFIG \
    --base $BASE \
    --model-dir $MODEL_DIR
