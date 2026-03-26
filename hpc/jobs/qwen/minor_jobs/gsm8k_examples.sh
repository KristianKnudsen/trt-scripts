#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --job-name="gsm8k_examples"
#SBATCH -c2
#SBATCH --time=00:04:00
#SBATCH --array=0-4

BASE="/cluster/home/krisskn/master-thesis/trt-scripts"
MODEL_DIR="/cluster/home/krisskn/master-thesis/hf-cache/models/Qwen2.5-3B"
LOGS="$BASE/hpc/logs"

QUANTS=(
  "W16A16"
  "W8A16"
  "W8A8_SQ"
  "W4A16_AWQ"
  "W4A16"
)

QUANT=${QUANTS[$SLURM_ARRAY_TASK_ID]}

srun --output=$LOGS/gsm8k_examples_${QUANT}_%j.out \
  /usr/bin/apptainer exec --nv --writable-tmpfs \
  $BASE/hpc/trtllm-tools.sif \
  python $BASE/eval/custom_lmeval_wrapper.py \
    --config $BASE/eval/configs/qwen/${QUANT}/eval_gsm8k_examples.json \
    --base $BASE \
    --model-dir $MODEL_DIR
