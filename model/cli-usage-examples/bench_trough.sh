#!/usr/bin/env bash
set -euo pipefail

# "./trt_engines/qwen/fp16"
# "./trt_engines/mistral/int8"
ENGINE_DIR="./trt_engines/mistral/int8"
#"Qwen/Qwen3-8B" 
#"mistralai/Mistral-7B-v0.1"
MODEL_HF_NAME="mistralai/Mistral-7B-v0.1"  # HF name

#/root/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da
#"/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
MODEL_PATH="/root/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da"

# ./datasets/Mistral_Synthethic_2048_2048_256_256.txt
# ./datasets/Mistral_Synthethic_512_512_64_64.txt 
# "./datasets/Qwen_Synthethic_512_512_64_64.txt"
# "./datasets/Qwen_Synthethic_2048_2048_256_256.txt"

DATA_SHORT="./datasets/Mistral_Synthethic_512_512_64_64.txt"
DATA_LONG="./datasets/Mistral_Synthethic_2048_2048_256_256.txt"
OUTDIR="./results"
mkdir -p "$OUTDIR"

SAMPLER_YAML="$OUTDIR/sampler.yaml"
cat > "$SAMPLER_YAML" <<EOF
temperature: 0.0
top_p: 1.0
repetition_penalty: 1.0
EOF

run_one () {
  local name="$1"
  local dataset="$2"
  local stamp
  stamp=$(date +%Y%m%d_%H%M%S)

  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,power.draw \
             --format=csv -lms 5000 > "$OUTDIR/${name}_${stamp}_smi.csv" &
  smi_pid=$!

  NUM_REQ=25

trtllm-bench -m "$MODEL_HF_NAME" --model_path "$MODEL_PATH" throughput \
    --engine_dir "$ENGINE_DIR" \
    --backend tensorrt \
    --dataset "$dataset" \
    --max_batch_size 1 \
    --concurrency 1 \
    --num_requests ${NUM_REQ} \
    --kv_cache_free_gpu_mem_fraction 0.9 \
    --sampler_options "$SAMPLER_YAML" \
    --warmup 0 \
    --streaming \
    --report_json "$OUTDIR/${name}_${stamp}_report.json" \
    --request_json "$OUTDIR/${name}_${stamp}_requests.json" \
    --iteration_log "$OUTDIR/${name}_${stamp}_iter.log"


  kill $smi_pid || true
}

MODEL_TAG=$(basename "$(dirname "$ENGINE_DIR")")      # e.g. qwen
PREC_TAG=$(basename "$ENGINE_DIR")                    # e.g. fp16

for i in 1 2; do
  run_one "${MODEL_TAG}_${PREC_TAG}_short_r${i}" "$DATA_SHORT"
  run_one "${MODEL_TAG}_${PREC_TAG}_long_r${i}"  "$DATA_LONG"
done
