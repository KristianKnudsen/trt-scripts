#!/bin/bash

python /workspace/TensorRT-LLM/examples/models/core/llama/convert_checkpoint.py \
  --model_dir /root/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da/ \
  --output_dir ./checkpoints/mistral_checkpoint_kv_int8 \
  --dtype float16 \
  --int8_kv_cache 
