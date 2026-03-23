#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --job-name="build_qwen"
#SBATCH -c2
#SBATCH --time=00-00:30:00
#SBATCH --begin=now
#SBATCH --array=0-5
#SBATCH --output=/cluster/home/krisskn/master-thesis/trt-scripts/hpc/logs/build_%A_%a.out

BASE="/cluster/home/krisskn/master-thesis/trt-scripts"
MODEL_DIR="/cluster/home/krisskn/master-thesis/hf-cache/models/Qwen2.5-3B"

CONFIGS=(
  "build_config_W16A16_LOGITS.json"   # 0
  "build_config_W8A16_LOGITS.json"    # 1
  "build_config_W8A8_SQ_LOGITS.json"  # 2
  "build_config_W4A16_LOGITS.json"    # 3
  "build_config_W4A16_AWQ_LOGITS.json" # 4
  "build_config_W4A8_AWQ_LOGITS.json"  # 5
)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

srun /usr/bin/apptainer exec --nv --writable-tmpfs \
  $BASE/hpc/trtllm-tools.sif \
  python $BASE/model/build_engine.py \
    --config $BASE/model/configs/hpc/qwen/$CONFIG \
    --base $BASE \
    --model-dir $MODEL_DIR
