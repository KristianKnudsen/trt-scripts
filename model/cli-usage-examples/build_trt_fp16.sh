#!/usr/bin/env bash
set -euo pipefail

# ----- Paths -----
MODEL_DIR="./checkpoints/qwen2_fp16"   # Converted checkpoint
OUT_DIR="./trt_engines/qwen2/fp16_batch16"         # Output engine directory

# ----- Sequence length parameters -----
# Dataset: input/output = 2048 ± 256 -> safe upper bound ≈ 6144 tokens total
MAX_INPUT_LEN=3072       # Hard cap for input tokens
MAX_SEQ_LEN=6144         # Hard cap for total context (input + output)

# ----- Expected batch shape -----
MAX_BATCH_SIZE=16         # Single-request inference
MAX_BEAM_WIDTH=1         # No beam search for benchmarks

# ----- Kernel optimization hints -----
OPT_NUM_TOKENS=2048      # Typical prompt size
MAX_NUM_TOKENS=3072      # Safe upper bound for prompt size variation

# ----- Precision and plugins -----
PRECISION="fp16"
LOGITS_DTYPE="float16"
GEMM_PLUGIN="auto"
ATTN_PLUGIN="float16"

# ----- Parallel build workers -----
WORKERS=4

echo "========================================"
echo " Building TensorRT-LLM Engine"
echo "----------------------------------------"
echo " Model directory:   ${MODEL_DIR}"
echo " Output directory:  ${OUT_DIR}"
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
  --gpt_attention_plugin ${ATTN_PLUGIN} \
  --remove_input_padding enable \
  --context_fmha enable \
  --use_paged_context_fmha enable \
  --use_fused_mlp enable \
  --gemm_plugin ${GEMM_PLUGIN} \
  --logits_dtype ${LOGITS_DTYPE} \
  --workers ${WORKERS}

echo "Build complete: ${OUT_DIR}"

