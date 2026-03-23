#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --job-name="build"
#SBATCH -c4
#SBATCH --time=00-00:30:00
#SBATCH --begin=now
#SBATCH --array=0-N
#SBATCH --output=/cluster/home/krisskn/master-thesis/trt-scripts/hpc/logs/build_%A_%a.out

BASE="/cluster/home/krisskn/master-thesis/trt-scripts"
MODEL_DIR="/cluster/home/krisskn/master-thesis/hf-cache/models/<model>"

# One config per quant variant. Update --array=0-N (0-indexed).
CONFIGS=(
  "build_config_W16A16_LOGITS.json"   # 0
)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

srun /usr/bin/apptainer exec --nv --writable-tmpfs \
  $BASE/hpc/trtllm-tools.sif \
  python $BASE/model/build_engine.py \
    --config $BASE/model/configs/hpc/<model>/$CONFIG \
    --base $BASE \
    --model-dir $MODEL_DIR
