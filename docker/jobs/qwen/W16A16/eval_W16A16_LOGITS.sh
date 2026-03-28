#!/bin/bash

BASE="/workspace/trt-scripts"
MODEL_DIR="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
CONFIG="eval_mmlu_pro_W16A16.json"

python $BASE/eval/custom_lmeval_wrapper.py \
  --config $BASE/eval/configs/$CONFIG \
  --base $BASE \
  --model-dir $MODEL_DIR
