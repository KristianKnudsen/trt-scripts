#!/bin/bash

concurrency_list="1"
multi_round=5
isl=1024
osl=1024
result_dir=./
model_path=/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b/

for concurrency in ${concurrency_list}; do
    num_prompts=$((concurrency * multi_round))
    python -m tensorrt_llm.serve.scripts.benchmark_serving \
        --model ${model_path} \
        --backend openai \
        --dataset-name "random" \
	--random-ids \
        --random-input-len ${isl} \
        --random-output-len ${osl} \
        --random-prefix-len 0 \
        --num-prompts ${num_prompts} \
        --max-concurrency ${concurrency} \
        --ignore-eos \
        --save-result \
        --result-dir "${result_dir}" \
        --result-filename "concurrency_${concurrency}.json" \
        --percentile-metrics "ttft,tpot,itl,e2el"
done
