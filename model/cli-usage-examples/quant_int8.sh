python /workspace/TensorRT-LLM/examples/quantization/quantize.py \
  --model_dir /root/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da \
  --dtype float16 \
  --qformat int8_sq \
  --kv_cache_dtype int8 \
  --calib_size 32 \
  --batch_size 1 \
  --calib_max_seq_length 256 \
  --tokenizer_max_seq_length 4096 \
  --tp_size 1 \
  --pp_size 1 \
  --output_dir ./checkpoints/mistral_sqint8_kvint8_quant

