lm-eval \
    --model local-completions \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 32 \
    --limit 256 \
    --model_args \
"base_url=http://localhost:8000/v1/completions,"\
"model=W4A16,"\
"tokenizer=/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b,"\
"tokenized_requests=True,"\
"max_length=6144,"\
"max_gen_toks=3072,"\
"num_concurrent=1,"\
"timeout=240" \
    --gen_kwargs "temperature=0.0" \
    --output_path ./results_mistral_32_test
