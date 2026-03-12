#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------
# ENGINE + MODEL
# ----------------------------------------

ENGINE_DIR="./trt_engines/mistral/int8"
MODEL_HF_NAME="mistralai/Mistral-7B-v0.1"
MODEL_PATH="/root/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da"

# ----------------------------------------
# DATASET (pick ONE)
# ----------------------------------------

# ./datasets/Mistral_Synthethic_512_512_64_64.txt
# ./datasets/Mistral_Synthethic_2048_2048_256_256.txt
DATASET="./datasets/Mistral_Synthethic_512_512_64_64.txt"

# ----------------------------------------
# OUTPUT + REPEATS
# ----------------------------------------

OUTDIR="./results"
mkdir -p "$OUTDIR"

REPEATS=1          # how many times to run the SAME dataset
NUM_REQ=15

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
# RUN FUNCTION
# ----------------------------------------

run_one () {
  local name="$1"
  local dataset="$2"
  local stamp
  stamp=$(date +%Y%m%d_%H%M%S)

  # SMI logging
  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,power.draw \
             --format=csv -lms "$SMI_INTERVAL_MS" \
             > "$OUTDIR/${name}_${stamp}_smi.csv" &

  smi_pid=$!

  trtllm-bench -m "$MODEL_HF_NAME" --model_path "$MODEL_PATH" latency \
      --engine_dir "$ENGINE_DIR" \
      --backend tensorrt \
      --dataset "$dataset" \
      --num_requests ${NUM_REQ} \
      --kv_cache_free_gpu_mem_fraction 0.9 \
      --sampler_options "$SAMPLER_YAML" \
      --warmup 0 \
      --report_json "$OUTDIR/${name}_${stamp}_report.json" \
      --iteration_log "$OUTDIR/${name}_${stamp}_iter.log"

  kill $smi_pid || true
}

# ----------------------------------------
# NAMING
# ----------------------------------------

MODEL_TAG=$(basename "$(dirname "$ENGINE_DIR")")   # e.g. mistral
PREC_TAG=$(basename "$ENGINE_DIR")                # e.g. int8

# ----------------------------------------
# MAIN LOOP — SINGLE DATASET, MANY RUNS
# ----------------------------------------

for i in $(seq 1 "$REPEATS"); do
  run_one "${MODEL_TAG}_${PREC_TAG}_latency_r${i}" "$DATASET"
done

