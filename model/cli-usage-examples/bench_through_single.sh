#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------
# ENGINE + MODEL
# ----------------------------------------

#ENGINE_DIR="./trt_engines/mistral/int8_batch16"

#ENGINE_DIR="./trt_engines/mistral/w16a16-kv16-b64/"
#MODEL_HF_NAME="mistralai/Mistral-7B-v0.1"
#MODEL_PATH="/root/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da"

#ENGINE_DIR="./trt_engines/qwen3/w16a16-kv16-b64"
#MODEL_HF_NAME="Qwen/Qwen3-8B"
#MODEL_PATH="/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"

ENGINE_DIR="./trt_engines/Qwen25_3B/w4a16"
MODEL_HF_NAME="Qwen/Qwen2.5-3B"
MODEL_PATH="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b/"


# ----------------------------------------
# DATASET (pick ONE)
# ----------------------------------------

#DATASET="./datasets/Mistral_Synthethic_512_512_64_64.txt"

#DATASET="./datasets/Mistral_Synthethic_2048_2048_256_256.txt"

DATASET="./datasets/Qwen_Synthethic_512_512_64_64.txt"

#DATASET="./datasets/Qwen_Synthethic_2048_2048_256_256.txt"

# ----------------------------------------
# OUTPUT + REPEATS
# ----------------------------------------

TAG="1_10"

OUTDIR="./results/qwen2"
mkdir -p "$OUTDIR"

REPEATS=1
NUM_REQ=10
SMI_INTERVAL_MS=5000

# ----------------------------------------
# SAMPLER OPTIONS
# ----------------------------------------

SAMPLER_YAML="$OUTDIR/sampler.yaml"
cat > "$SAMPLER_YAML" <<EOF
temperature: 0.0
top_p: 1.0
repetition_penalty: 1.0
EOF

# ----------------------------------------
# Parse dataset signature (last four numbers)
# ----------------------------------------

DATASET_BASE=$(basename "$DATASET")
DATASET_SIG=$(echo "$DATASET_BASE" | sed -E 's/.*_([0-9]+_[0-9]+_[0-9]+_[0-9]+)\.txt/\1/')

# ----------------------------------------
# Model/quant tags
# ----------------------------------------

MODEL_TAG=$(basename "$(dirname "$ENGINE_DIR")")   # e.g. mistral
PREC_TAG=$(basename "$ENGINE_DIR")                # e.g. int8

# ----------------------------------------
# Create ONE folder for all repeats in this config
# ----------------------------------------

STAMP_HM=$(date +%H%M)
RUN_DIR="${OUTDIR}/${MODEL_TAG}_${PREC_TAG}_${DATASET_SIG}_${STAMP_HM}_${TAG}"
mkdir -p "$RUN_DIR"

echo "Run directory: $RUN_DIR"

# ----------------------------------------
# RUN FUNCTION (per repetition)
# ----------------------------------------

run_one () {
  local rid="$1"
  local dataset="$2"

  echo "Running repeat: r${rid}"

  # SMI logging
  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,power.draw \
             --format=csv -lms "$SMI_INTERVAL_MS" \
             > "${RUN_DIR}/r${rid}_smi.csv" &
  smi_pid=$!

  trtllm-bench -m "$MODEL_HF_NAME" --model_path "$MODEL_PATH" throughput \
      --engine_dir "$ENGINE_DIR" \
      --backend tensorrt \
      --dataset "$dataset" \
      --max_batch_size 1 \
      --concurrency 1 \
      --num_requests ${NUM_REQ} \
      --kv_cache_free_gpu_mem_fraction 0.9 \
      --sampler_options "$SAMPLER_YAML" \
      --warmup 1 \
      --streaming \
      --report_json "${RUN_DIR}/r${rid}_report.json" \
      --request_json "${RUN_DIR}/r${rid}_requests.json"

  kill $smi_pid || true
}

# ----------------------------------------
# MAIN LOOP — all output into one folder
# ----------------------------------------

for i in $(seq 1 "$REPEATS"); do
  run_one "$i" "$DATASET"
done

