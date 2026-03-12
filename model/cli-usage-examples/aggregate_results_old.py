#!/usr/bin/env python3
import os
import sys
import json
import csv
import math
from glob import glob

def percentile(values, p):
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    return vals[f] * (c - k) + vals[c] * (k - f)

def parse_smi_file(path, util_threshold=30.0):
    mem_used = []
    power = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return None, None

    header = [h.strip().lower() for h in rows[0]]
    idx = {name: i for i, name in enumerate(header)}

    def col(name_options):
        for opt in name_options:
            if opt in idx:
                return idx[opt]
        raise KeyError(f"SMI column not found. Tried: {name_options}\nHeader was: {header}")

    gpu_util_col = col(["utilization.gpu [%]", "utilization.gpu %", "utilization.gpu"])
    mem_used_col = col(["memory.used [mib]", "memory.used"])
    power_col = col(["power.draw [w]", "power.draw"])

    for row in rows[1:]:
        row = [c.strip() for c in row]
        util_str = row[gpu_util_col]
        if not util_str:
            continue
        try:
            util = float(util_str.split()[0])
        except:
            continue
        if util < util_threshold:
            continue

        # memory parsing
        try:
            mem_val = float(row[mem_used_col].split()[0])
        except:
            mem_val = None

        # power parsing
        try:
            power_val = float(row[power_col].split()[0])
        except:
            power_val = None

        if mem_val is not None:
            mem_used.append(mem_val)
        if power_val is not None:
            power.append(power_val)

    max_mem = max(mem_used) if mem_used else None
    avg_power = sum(power) / len(power) if power else None

    return max_mem, avg_power

def main(run_dir):
    print(f"[INFO] Aggregating folder: {run_dir}")

    report_files = sorted(glob(os.path.join(run_dir, "r*_report.json")))
    request_files = sorted(glob(os.path.join(run_dir, "r*_requests.json")))
    smi_files = sorted(glob(os.path.join(run_dir, "r*_smi.csv")))

    print(f"[INFO] Found {len(report_files)} report files")
    print(f"[INFO] Found {len(request_files)} request files")
    print(f"[INFO] Found {len(smi_files)} SMI logs")

    if not report_files or not request_files or not smi_files:
        print("[ERROR] Missing required file types.")
        sys.exit(1)

    num_runs = min(len(report_files), len(request_files), len(smi_files))
    print(f"[INFO] Aggregating {num_runs} repeats")

    with open(report_files[0], "r", encoding="utf-8") as f:
        first_report = json.load(f)

    engine = first_report["engine"]
    world_info = first_report["world_info"]
    dataset_info = first_report["dataset"]
    request_info = first_report["request_info"]

    # Concurrency detection
    avg_conc = request_info.get("avg_num_concurrent_requests")
    concurrency = int(round(avg_conc)) if avg_conc is not None else None
    print(f"[INFO] Detected concurrency: {concurrency}")

    # Batchsize = min(max_batch_size, concurrency)
    max_batch_size = world_info.get("max_batch_size", 1)
    batchsize = min(max_batch_size, concurrency) if concurrency else max_batch_size
    print(f"[INFO] Detected batchsize: {batchsize}")

    # Other static metadata
    model_name = engine.get("model")
    backend = engine.get("backend")
    quant_method = engine.get("quantization")
    kv_dtype = engine.get("kv_cache_dtype")
    kv_fraction = world_info.get("kv_cache_percentage")

    max_input_len = engine.get("max_input_length")
    max_seq_len = engine.get("max_sequence_length")

    dataset_mean_input = dataset_info.get("avg_isl", dataset_info["isl_stats"]["average"])
    dataset_mean_output = dataset_info.get("avg_osl", dataset_info["osl_stats"]["average"])

    all_e2e_ms = []
    all_ttft_ms = []
    all_tpot_ms = []
    all_in_tokens = []
    all_out_tokens = []

    total_duration_ns = 0
    total_in = 0
    total_out = 0
    total_req = 0

    for i in range(num_runs):
        with open(request_files[i], "r", encoding="utf-8") as f:
            requests = json.load(f)

        if not requests:
            continue

        start_ts = min(r["start_timestamp"] for r in requests)
        end_ts   = max(r["end_timestamp"] for r in requests)
        total_duration_ns += (end_ts - start_ts)

        for r in requests:
            in_tok = r["num_input_tokens"]
            out_tok = r.get("num_total_output_tokens", r.get("output_tokens", 0))

            all_in_tokens.append(in_tok)
            all_out_tokens.append(out_tok)

            all_e2e_ms.append(r["end_to_end_latency"] / 1e6)
            all_ttft_ms.append(r["time_to_first_token"] / 1e6)
            all_tpot_ms.append(r["intertoken_latency"] / 1e6)

            total_in += in_tok
            total_out += out_tok
            total_req += 1

    # Means
    avg_in = sum(all_in_tokens) / len(all_in_tokens)
    avg_out = sum(all_out_tokens) / len(all_out_tokens)

    # Percentiles
    lat_p50 = percentile(all_e2e_ms, 50)
    lat_p90 = percentile(all_e2e_ms, 90)
    lat_p95 = percentile(all_e2e_ms, 95)
    lat_p99 = percentile(all_e2e_ms, 99)

    tpot_p50 = percentile(all_tpot_ms, 50)
    tpot_p90 = percentile(all_tpot_ms, 90)
    tpot_p95 = percentile(all_tpot_ms, 95)
    tpot_p99 = percentile(all_tpot_ms, 99)

    ttft_p50 = percentile(all_ttft_ms, 50)
    ttft_p90 = percentile(all_ttft_ms, 90)
    ttft_p95 = percentile(all_ttft_ms, 95)
    ttft_p99 = percentile(all_ttft_ms, 99)

    # Throughput
    if total_duration_ns > 0:
        total_time_s = total_duration_ns / 1e9
        req_per_s = total_req / total_time_s
        tokens_per_s = (total_in + total_out) / total_time_s
        output_tokens_per_s = total_out / total_time_s
    else:
        req_per_s = tokens_per_s = output_tokens_per_s = None

    all_mem_peaks = []
    all_power = []
    all_gpu_util = []

    for path in smi_files:
        mem_peak, power_avg = parse_smi_file(path)
        if mem_peak:
            all_mem_peaks.append(mem_peak)

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        header = [h.strip().lower() for h in rows[0]]
        idx = {name: i for i, name in enumerate(header)}

        def col(opts):
            for opt in opts:
                if opt in idx:
                    return idx[opt]
            return None

        gpu_util_col = col(["utilization.gpu [%]", "utilization.gpu", "utilization.gpu %"])
        power_col = col(["power.draw [w]", "power.draw"])

        if gpu_util_col is None or power_col is None:
            continue

        for row in rows[1:]:
            row = [c.strip() for c in row]
            try:
                util = float(row[gpu_util_col].split()[0])
            except:
                continue
            if util < 30.0:
                continue

            try:
                p = float(row[power_col].split()[0])
                all_power.append(p)
            except:
                pass

            try:
                u = float(row[gpu_util_col].split()[0])
                all_gpu_util.append(u)
            except:
                pass

    bench_vram_peak = max(all_mem_peaks) if all_mem_peaks else None
    bench_power_avg = sum(all_power) / len(all_power) if all_power else None
    bench_power_peak = max(all_power) if all_power else None
    gpu_util_avg = sum(all_gpu_util) / len(all_gpu_util) if all_gpu_util else None
    gpu_util_peak = max(all_gpu_util) if all_gpu_util else None

    result = {
        "model_name": model_name,
        "system": None,
        "serving_stack": "tensorrt-llm",
        "engine_backend": backend,
        "quant_method": quant_method,
        "kv_fraction": kv_fraction,
        "kv_quant_method": kv_dtype,

        "max_input_len": max_input_len,
        "max_seq_len": max_seq_len,

        "dataset_mean_input_tokens": dataset_mean_input,
        "dataset_mean_output_tokens": dataset_mean_output,

        "avg_input_tokens": avg_in,
        "avg_output_tokens": avg_out,

        "batchsize": batchsize,
        "concurrency": concurrency,
        "rounds": num_runs,
        "num_requests": total_req,

        "latency_p50_ms": lat_p50,
        "latency_p90_ms": lat_p90,
        "latency_p95_ms": lat_p95,
        "latency_p99_ms": lat_p99,

        "token_latency_p50_ms": tpot_p50,
        "token_latency_p90_ms": tpot_p90,
        "token_latency_p95_ms": tpot_p95,
        "token_latency_p99_ms": tpot_p99,

        "ttft_p50_ms": ttft_p50,
        "ttft_p90_ms": ttft_p90,
        "ttft_p95_ms": ttft_p95,
        "ttft_p99_ms": ttft_p99,

        "req_per_s": req_per_s,
        "tokens_per_s": tokens_per_s,
        "output_tokens_per_s": output_tokens_per_s,

        "bench_vram_peak_mb": bench_vram_peak,
        "bench_power_avg_w": bench_power_avg,
        "bench_power_peak_w": bench_power_peak,

        "gpu_util_avg": gpu_util_avg,
        "gpu_util_peak": gpu_util_peak,

        "mmlu_accuracy": None,
        "gsm8k_accuracy": None,
        "humaneval_pass1": None
    }

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} RUN_DIR")
        sys.exit(1)
    main(sys.argv[1])
