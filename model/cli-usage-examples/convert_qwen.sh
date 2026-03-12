#!/bin/bash
set -e

python /workspace/TensorRT-LLM/examples/models/core/qwen/convert_checkpoint.py \
  --model_dir /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b/ \
  --output_dir ./checkpoints/qwen2/int4-weight-only \
  --dtype float16 \
  --use_weight_only \
  --weight_only_precision int4