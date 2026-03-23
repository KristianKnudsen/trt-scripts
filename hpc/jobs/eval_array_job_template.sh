#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --job-name="eval_W16A16"
#SBATCH -c4
#SBATCH --time=00-02:00:00
#SBATCH --begin=now
#SBATCH --array=0-8
#SBATCH --output=/cluster/home/krisskn/master-thesis/trt-scripts/hpc/logs/eval_%A_%a.out

BASE="/cluster/home/krisskn/master-thesis/trt-scripts"
MODEL_DIR="/cluster/home/krisskn/master-thesis/hf-cache/models/Qwen2.5-3B"
QUANT="W16A16"

# One config per task — all for the same quant.
# Update --array=0-N if you add/remove tasks (0-indexed).
CONFIGS=(
  "eval_mmlu_${QUANT}.json"           # 0 - loglikelihood
  "eval_mmlu_pro_${QUANT}.json"       # 1 - loglikelihood
  "eval_hellaswag_${QUANT}.json"      # 2 - loglikelihood
  "eval_winogrande_${QUANT}.json"     # 3 - loglikelihood
  "eval_gpqa_${QUANT}.json"           # 4 - loglikelihood
  "eval_gsm8k_${QUANT}.json"          # 5 - generation
  "eval_humaneval_${QUANT}.json" # 6 - generation
  "eval_mbpp_${QUANT}.json"      # 7 - generation

)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

srun /usr/bin/apptainer exec --nv --writable-tmpfs \
  $BASE/hpc/trtllm-tools.sif \
  python $BASE/eval/custom_lmeval_wrapper.py \
    --config $BASE/eval/configs/$CONFIG \
    --base $BASE \
    --model-dir $MODEL_DIR
