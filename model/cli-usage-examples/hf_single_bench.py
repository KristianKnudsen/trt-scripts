import time
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen3-8B"
#model_name = "mistralai/Mistral-7B-v0.1"
device = "cuda"

num_requests = 10
warmup_requests = 1
dataset_path = "./datasets/Qwen_Synthethic_512_512_64_64.txt"  # JSONL: one object per line

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map={"": device},
    trust_remote_code=True,
)
model.eval()
print("Model loaded on:", next(model.parameters()).device)


def load_trtllm_dataset_jsonl(path: str):
    prompts, out_lens = [], []

    with open(path, "r") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            try:
                ids = obj["input_ids"]
                out_len = obj["output_tokens"]
            except KeyError as e:
                raise KeyError(
                    f"Line {line_idx}: missing required field {e.args[0]!r}"
                ) from e

            t = torch.tensor([ids], dtype=torch.long, device=device)
            prompts.append(t)
            out_lens.append(int(out_len))

    if not prompts:
        raise ValueError("Dataset is empty or has no valid entries")

    print(f"Loaded {len(prompts)} samples from {path}")
    return prompts, out_lens


@torch.no_grad()
def run_one(req_id: int, input_ids: torch.Tensor, out_len: int):
    print(f"[Request {req_id}] starting")

    prompt = input_ids.clone()

    t0 = time.perf_counter()
    step_lat = []

    # first step: full prompt, with KV cache
    s0 = time.perf_counter()
    out = model(input_ids=prompt, use_cache=True)
    if device == "cuda":
        torch.cuda.synchronize()
    s1 = time.perf_counter()

    logits = out.logits[:, -1, :]
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    generated = next_token
    past_kv = out.past_key_values

    ttft = s1
    step_lat.append(s1 - s0)

    # remaining steps: only last token plus past_kv
    for _ in range(1, out_len):
        s0 = time.perf_counter()
        out = model(
            input_ids=generated[:, -1:],
            use_cache=True,
            past_key_values=past_kv,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        s1 = time.perf_counter()

        logits = out.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        past_kv = out.past_key_values

        step_lat.append(s1 - s0)

    t1 = time.perf_counter()
    print(f"[Request {req_id}] finished")

    return {
        "lat_ms": (t1 - t0) * 1000.0,
        "ttft_ms": (ttft - t0) * 1000.0,
        "tok_lat_ms": [x * 1000.0 for x in step_lat],
        "input_tokens": int(prompt.shape[1]),
        "output_tokens": int(out_len),
    }


def main():
    prompts, out_lens = load_trtllm_dataset_jsonl(dataset_path)

    needed = warmup_requests + num_requests
    if len(prompts) < needed:
        raise ValueError(
            f"Dataset has {len(prompts)} samples, but {needed} are required "
            f"({warmup_requests} warmup + {num_requests} measured)."
        )

    print("\n--- Warmup ---")
    for i in range(warmup_requests):
        _ = run_one(f"warmup-{i}", prompts[i], out_lens[i])

    print("\n--- Benchmark ---")
    results = []
    bench_start = time.perf_counter()
    for i in range(num_requests):
        idx = warmup_requests + i
        res = run_one(i, prompts[idx], out_lens[idx])
        results.append(res)
    bench_end = time.perf_counter()

    latencies = np.array([r["lat_ms"] for r in results])
    ttfts = np.array([r["ttft_ms"] for r in results])
    tok_lat_all = np.concatenate([np.array(r["tok_lat_ms"]) for r in results])

    total_input_tokens = sum(r["input_tokens"] for r in results)
    total_output_tokens = sum(r["output_tokens"] for r in results)
    total_tokens = total_input_tokens + total_output_tokens

    wall = bench_end - bench_start

    def p(x, q):
        return float(np.percentile(x, q)) if len(x) else float("nan")

    latency_p50_ms = p(latencies, 50)
    latency_p90_ms = p(latencies, 90)
    latency_p95_ms = p(latencies, 95)
    latency_p99_ms = p(latencies, 99)

    token_latency_p50_ms = p(tok_lat_all, 50)
    token_latency_p90_ms = p(tok_lat_all, 90)
    token_latency_p95_ms = p(tok_lat_all, 95)
    token_latency_p99_ms = p(tok_lat_all, 99)

    ttft_p50_ms = p(ttfts, 50)
    ttft_p90_ms = p(ttfts, 90)
    ttft_p95_ms = p(ttfts, 95)
    ttft_p99_ms = p(ttfts, 99)

    req_per_s = num_requests / wall
    tokens_per_s = total_tokens / wall
    output_tokens_per_s = total_output_tokens / wall

    header = (
        "latency_p50_ms\tlatency_p90_ms\tlatency_p95_ms\tlatency_p99_ms\t"
        "token_latency_p50_ms\ttoken_latency_p90_ms\t"
        "token_latency_p95_ms\ttoken_latency_p99_ms\t"
        "ttft_p50_ms\tttft_p90_ms\tttft_p95_ms\tttft_p99_ms\t"
        "req_per_s\ttokens_per_s\toutput_tokens_per_s"
    )

    values = (
        f"{latency_p50_ms:.3f}\t{latency_p90_ms:.3f}\t{latency_p95_ms:.3f}\t{latency_p99_ms:.3f}\t"
        f"{token_latency_p50_ms:.3f}\t{token_latency_p90_ms:.3f}\t{token_latency_p95_ms:.3f}\t{token_latency_p99_ms:.3f}\t"
        f"{ttft_p50_ms:.3f}\t{ttft_p90_ms:.3f}\t{ttft_p95_ms:.3f}\t{ttft_p99_ms:.3f}\t"
        f"{req_per_s:.6f}\t{tokens_per_s:.3f}\t{output_tokens_per_s:.3f}"
    )

    print("\n" + header)
    print(values)


if __name__ == "__main__":
    main()
