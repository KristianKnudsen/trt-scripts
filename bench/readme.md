# Bench

This folder contains the TensorRT-LLM latency benchmark wrapper, dataset generators, configs, and result files.

The main runner is:

```bash
bench/bench_latency.py
```

It runs `trtllm-bench latency`, applies the local patches we need, and appends one row per run to:

```bash
bench/results/results.csv
```

## Config

Configs live under:

```bash
bench/configs
```

The default config is:

```bash
bench/configs/base_config.json
```

Current fields:

```json
{
    "num_requests": 10,
    "warmup": 1,
    "backend": "tensorrt",
    "kv_cache_free_gpu_mem_fraction": 0.9,
    "concurrency": 1,
    "max_batch_size": 1,
    "smi_logging": true,
    "smi_interval_ms": 1000,
    "sampler_options": {
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0
    },
    "keep_raw": false
}
```

Field notes:

- `num_requests`: number of requests to benchmark.
- `warmup`: warmup iterations before measuring.
- `backend`: passed to TensorRT-LLM bench.
- `kv_cache_free_gpu_mem_fraction`: passed to TensorRT-LLM bench.
- `concurrency`: request concurrency. Usually `1` for latency runs.
- `max_batch_size`: kept in config for clarity. Current runs use batch size `1`.
- `smi_logging`: records GPU stats with `nvidia-smi` during the run.
- `smi_interval_ms`: `nvidia-smi` sample interval.
- `sampler_options`: generation sampler settings patched into TensorRT-LLM bench.
- `keep_raw`: when `false`, raw per-run files are deleted after the CSV row is written.

## Arguments

```bash
python bench/bench_latency.py \
  --base /workspace/trt-scripts \
  --engine Qwen25_3B/W16A16 \
  --dataset Qwen_Synthetic_short_in_short_out_256_256_16_16.txt \
  --config base_config.json \
  --tokenizer Qwen25
```

Arguments:

- `--base`: repo root. Defaults to `/workspace/trt-scripts`.
- `--engine`: engine folder under `model/trt_engines`.
- `--dataset`: dataset file under `bench/datasets`. Subfolders are supported.
- `--config`: config file under `bench/configs`. The `.json` suffix is optional.
- `--tokenizer`: tokenizer folder under `model/tokenizers`.

Examples:

```bash
--engine Qwen25_3B/W16A16
```

resolves to:

```bash
model/trt_engines/Qwen25_3B/W16A16
```

```bash
--tokenizer Qwen25
```

resolves to:

```bash
model/tokenizers/Qwen25
```

## Generate Datasets

Dataset scripts live under:

```bash
bench/datasets
```

Generate Qwen datasets:

```bash
bench/datasets/GenerateQwenData.sh
```

Generate Mistral datasets:

```bash
bench/datasets/GenerateMistralData.sh
```

Each script creates four synthetic datasets:

- short input, short output
- long input, short output
- short input, long output
- long input, long output

The Qwen files are:

```bash
bench/datasets/Qwen_Synthetic_short_in_short_out_256_256_16_16.txt
bench/datasets/Qwen_Synthetic_long_in_short_out_2560_256_128_16.txt
bench/datasets/Qwen_Synthetic_short_in_long_out_256_2560_16_128.txt
bench/datasets/Qwen_Synthetic_long_in_long_out_2560_2560_128_128.txt
```

The Mistral files use the same names with `Mistral_Synthetic_` instead of `Qwen_Synthetic_`.

The dataset scripts call:

```bash
/workspace/TensorRT-LLM/benchmarks/cpp/prepare_dataset.py
```

## Run One Benchmark

Example Qwen25_3B W16A16 short input, short output run:

```bash
python /workspace/trt-scripts/bench/bench_latency.py \
  --base /workspace/trt-scripts \
  --engine Qwen25_3B/W16A16 \
  --dataset Qwen_Synthetic_short_in_short_out_256_256_16_16.txt \
  --config base_config.json \
  --tokenizer Qwen25
```

The command prints the CSV path when it finishes.

## Run All Qwen25_3B Benchmarks

This script runs all four Qwen datasets for every Qwen25_3B quant:

```bash
docker/jobs/Qwen25_3B/bench_all_qwen25_3b.sh
```

It appends one row per run to:

```bash
bench/results/results.csv
```

## Results

The main output is:

```bash
bench/results/results.csv
```

Each benchmark appends one row. The row includes:

- model, quant, engine, dataset, config, and request count
- engine metadata such as dtype, quantization, max input length, and max sequence length
- average dataset input length, output length, and sequence length
- latency, TTFT, TPOT, throughput, and generation TPS percentiles
- peak GPU util, memory util, memory used, and power from `nvidia-smi`

Raw per-run files are written under:

```bash
bench/results/raw
```

When `keep_raw` is `false`, the raw folder is deleted after the CSV row is written. Set `keep_raw` to `true` if you want to keep `report.json`, `iteration.log`, `smi.csv`, and the temporary workspace.

## Files To Edit

For different hardware, models, or benchmark shapes, these are the usual files to change.

Dataset generation:

```bash
bench/datasets/GenerateQwenData.sh
bench/datasets/GenerateMistralData.sh
```

Edit these when you need different input or output lengths, standard deviations, request counts, tokenizer paths, or dataset names. The workload values should fit the engine limits for the hardware you are testing.

Benchmark config:

```bash
bench/configs/base_config.json
```

Edit this for request count, warmup, KV cache free memory fraction, concurrency, SMI logging, sampler options, or whether raw files are kept.

Benchmark sweep jobs:

```bash
docker/jobs/Qwen25_3B/bench_all_qwen25_3b.sh
```

Edit this when the model name, tokenizer name, quant list, dataset list, config name, or repo base path changes.

Engine folders:

```bash
model/trt_engines/<model>/<quant>
```

The `--engine` argument points under this folder. The sweep scripts assume these engine folders already exist.

Tokenizer folders:

```bash
model/tokenizers/<tokenizer>
```

The `--tokenizer` argument points under this folder. The benchmark runner uses this as `--model_path` for TensorRT-LLM bench.

Benchmark runner:

```bash
bench/bench_latency.py
```

Only edit this if you need to change the CSV columns, result extraction, SMI parsing, path rules, or TensorRT-LLM bench patching behavior.

## Notes

The runner patches TensorRT-LLM bench in-process so `--model_path` can point at the tokenizer folder instead of a full Hugging Face model folder.

The runner also patches sampler options in-process, so no temporary sampler YAML file is needed.
