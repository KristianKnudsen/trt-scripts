#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --job-name="eval_W16A16"
#SBATCH -c2
#SBATCH --time=00-00:10:00
#SBATCH --begin=now
#SBATCH --array=0-8

BASE="/cluster/home/krisskn/master-thesis/trt-scripts"
MODEL_DIR="/cluster/home/krisskn/master-thesis/hf-cache/models/Qwen2.5-3B"
QUANT="W16A16"

CONFIGS=(
  "eval_mmlu_${QUANT}.json"           # 0
  "eval_mmlu_pro_${QUANT}.json"       # 1
  "eval_hellaswag_${QUANT}.json"      # 2
  "eval_winogrande_${QUANT}.json"     # 3
  "eval_gpqa_${QUANT}.json"           # 4
  "eval_gsm8k_${QUANT}.json"          # 5
  "eval_humaneval_plus_${QUANT}.json" # 6
  "eval_mbpp_plus_${QUANT}.json"      # 7
  "eval_multiple_py_${QUANT}.json"    # 8
)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

srun /usr/bin/apptainer exec --nv --writable-tmpfs \
  $BASE/hpc/trtllm-tools.sif \
  python $BASE/eval/custom_lmeval_wrapper.py \
    --config $BASE/eval/configs/$CONFIG \
    --base $BASE \
    --model-dir $MODEL_DIR
