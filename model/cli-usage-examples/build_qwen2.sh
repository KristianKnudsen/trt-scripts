#!/usr/bin/env bash
set -euo pipefail

# ----- Paths -----
MODEL_DIR="./checkpoints/qwen2_fp16"                 # Your converted TRT-LLM checkpoint
OUT_DIR="./trt_engines/qwen2/fp16_batch16"           # Where to put the built engines

# ----- Sequence length parameters -----
MAX_INPUT_LEN=3072       # Max input tokens
MAX_SEQ_LEN=6144         # Max total tokens (prompt + generated)

# ----- Batch / tokens -----
MAX_BATCH_SIZE=16
MAX_BEAM_WIDTH=1
OPT_NUM_TOKENS=2048
MAX_NUM_TOKENS=3072

# ----- Precision / plugins -----
LOGITS_DTYPE="float16"   # What the doc calls --logits_dtype
GEMM_PLUGIN="auto"       # Let TRT choose
GPT_ATTN_PLUGIN="auto"   # IMPORTANT: do NOT force float16 here
WORKERS=4

echo "========================================"
echo " Building TensorRT-LLM Engine (FP16)"
echo "----------------------------------------"
echo " Model directory:   ${MODEL_DIR}"
echo " Output directory:  ${OUT_DIR}"
echo " Max batch size:    ${MAX_BATCH_SIZE}"
echo " Max input length:  ${MAX_INPUT_LEN}"
echo " Max seq length:    ${MAX_SEQ_LEN}"
echo " Opt tokens:        ${OPT_NUM_TOKENS}"
echo " Max tokens:        ${MAX_NUM_TOKENS}"
echo "========================================"

mkdir -p "${OUT_DIR}"

trtllm-build \
  --checkpoint_dir "${MODEL_DIR}" \
  --output_dir "${OUT_DIR}" \
  --max_batch_size ${MAX_BATCH_SIZE} \
  --max_beam_width ${MAX_BEAM_WIDTH} \
  --max_input_len ${MAX_INPUT_LEN} \
  --max_seq_len ${MAX_SEQ_LEN} \
  --opt_num_tokens ${OPT_NUM_TOKENS} \
  --max_num_tokens ${MAX_NUM_TOKENS} \
  --kv_cache_type paged \
  --remove_input_padding enable \
  --gpt_attention_plugin ${GPT_ATTN_PLUGIN} \
  --gemm_plugin ${GEMM_PLUGIN} \
  --context_fmha disable \
  --use_paged_context_fmha disable \
  --use_fp8_context_fmha disable \
  --use_fused_mlp disable \
  --logits_dtype ${LOGITS_DTYPE} \
  --workers ${WORKERS}

echo "Build complete: ${OUT_DIR}"
