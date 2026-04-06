#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --job-name="test_eval_W8A16_winogrande"
#SBATCH -c2
#SBATCH --time=00:12:30
#SBATCH --begin=now
#SBATCH --output=/cluster/home/krisskn/master-thesis/trt-scripts/hpc/logs/eval_%j.out

BASE="/cluster/home/krisskn/master-thesis/trt-scripts"
MODEL_DIR="/cluster/home/krisskn/master-thesis/hf-cache/models/Qwen2.5-3B"

srun /usr/bin/apptainer exec --nv --writable-tmpfs \
  $BASE/hpc/trtllm-tools.sif \
  python $BASE/eval/custom_lmeval_wrapper.py \
    --config $BASE/eval/configs/Qwen25_3B/W8A16/eval_winogrande.json \
    --base $BASE \
    --model-dir $MODEL_DIR
