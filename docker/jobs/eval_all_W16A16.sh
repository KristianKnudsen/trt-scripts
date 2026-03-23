#!/bin/bash

BASE="/workspace/trt-scripts"
MODEL_DIR="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
QUANT="W16A16"

CONFIGS=(
  "eval_humaneval_${QUANT}.json"
  "eval_mbpp_${QUANT}.json"
  "eval_gpqa_${QUANT}.json"
  "eval_mmlu_${QUANT}.json"
  "eval_mmlu_pro_${QUANT}.json"
  "eval_hellaswag_${QUANT}.json"
  "eval_winogrande_${QUANT}.json"
  "eval_gsm8k_${QUANT}.json"
)

for CONFIG in "${CONFIGS[@]}"; do
  echo "Running $CONFIG..."
  python $BASE/eval/custom_lmeval_wrapper.py \
    --config $BASE/eval/configs/$CONFIG \
    --base $BASE \
    --model-dir $MODEL_DIR
done
