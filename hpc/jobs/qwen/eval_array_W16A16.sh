#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --job-name="eval_W16A16"
#SBATCH -c2
#SBATCH --time=00:12:30
#SBATCH --begin=now
#SBATCH --output=/cluster/home/krisskn/master-thesis/trt-scripts/hpc/logs/eval_%A_%a.out

BASE="/cluster/home/krisskn/master-thesis/trt-scripts"
MODEL_DIR="/cluster/home/krisskn/master-thesis/hf-cache/models/Qwen2.5-3B"
QUANT="W16A16"

CONFIGS=(
  "eval_humaneval_${QUANT}.json" 
  "eval_mbpp_${QUANT}.json"      
  "eval_mmlu_${QUANT}.json"           
  "eval_mmlu_pro_${QUANT}.json"       
  "eval_hellaswag_${QUANT}.json"      
  "eval_winogrande_${QUANT}.json"     
  "eval_gpqa_${QUANT}.json"           
  "eval_gsm8k_${QUANT}.json"          
)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

srun /usr/bin/apptainer exec --nv --writable-tmpfs \
  $BASE/hpc/trtllm-tools.sif \
  python $BASE/eval/custom_lmeval_wrapper.py \
    --config $BASE/eval/configs/$CONFIG \
    --base $BASE \
    --model-dir $MODEL_DIR
