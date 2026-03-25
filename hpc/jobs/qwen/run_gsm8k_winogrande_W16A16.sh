#!/bin/bash
BASE="/cluster/home/krisskn/master-thesis/trt-scripts"
MODEL_DIR="/cluster/home/krisskn/master-thesis/hf-cache/models/Qwen2.5-3B"
QUANT="W16A16"

/usr/bin/apptainer exec --nv --writable-tmpfs \
  $BASE/hpc/trtllm-tools.sif \
  python $BASE/eval/custom_lmeval_wrapper.py \
    --config $BASE/eval/configs/qwen/${QUANT}/eval_gsm8k.json \
    --base $BASE \
    --model-dir $MODEL_DIR

/usr/bin/apptainer exec --nv --writable-tmpfs \
  $BASE/hpc/trtllm-tools.sif \
  python $BASE/eval/custom_lmeval_wrapper.py \
    --config $BASE/eval/configs/qwen/${QUANT}/eval_winogrande.json \
    --base $BASE \
    --model-dir $MODEL_DIR
