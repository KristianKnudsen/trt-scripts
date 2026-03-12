#!/bin/bash
set -e

# CONFIGURATION
TOKENIZER="/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/"

BASE_NAME="Qwen_Synthetic"

INPUT_MEAN=2048
OUTPUT_MEAN=256
INPUT_STDEV=128
OUTPUT_STDEV=16
NUM_REQUESTS=1000

# AUTO-GENERATED OUTPUT FILE NAME
OUTFILE="datasets/${BASE_NAME}_${INPUT_MEAN}_${OUTPUT_MEAN}_${INPUT_STDEV}_${OUTPUT_STDEV}.txt"

# RUN
python /workspace/TensorRT-LLM/benchmarks/cpp/prepare_dataset.py \
    --stdout \
    --tokenizer "$TOKENIZER" \
    token-norm-dist \
    --input-mean "$INPUT_MEAN" \
    --output-mean "$OUTPUT_MEAN" \
    --input-stdev "$INPUT_STDEV" \
    --output-stdev "$OUTPUT_STDEV" \
    --num-requests "$NUM_REQUESTS" \
    > "$OUTFILE"
