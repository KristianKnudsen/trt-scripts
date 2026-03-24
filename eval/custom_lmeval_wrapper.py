import os

from tensorrt_llm.evaluate.lm_eval import LmEvalWrapper
import argparse
import csv
import copy
import json
import re
import sys
import tempfile
import torch
from contextlib import contextmanager
from tqdm import tqdm
from dataclasses import dataclass, fields
from typing import Optional
from pathlib import Path
from tensorrt_llm.sampling_params import SamplingParams


def _loglikelihood_tokens(self: LmEvalWrapper, requests, disable_tqdm=False, **kwargs):
    """
    Full MMLU task will crash on certain hardware (HPC will work, allocate a lot of ram). Divide the tasks as much as possible for stable evaluation. Making big requests with gather logits enabled
    seems to have some kind of memory accumulation in the eninge.
    """
    profiler.start("trtllm exec")
    results = []

    sp = copy.deepcopy(self.sampling_params) if self.sampling_params else SamplingParams()
    sp.max_tokens = 1
    sp.temperature = 0.0
    sp.return_context_logits = True

    def _build_input(request):
        _, prompt_tokens, target_tokens = request
        prompt_tokens = list(prompt_tokens)
        target_tokens = list(target_tokens)
        inp = prompt_tokens + target_tokens[:-1]
        return inp, prompt_tokens, target_tokens

    def _process(gen_output, prompt_tokens, target_tokens):
        output = gen_output.result()
        start = len(prompt_tokens) - 1
        end = start + len(target_tokens)
        logits = output.context_logits[start:end].clone()
        del gen_output, output

        logits_f = logits.float()
        target = torch.tensor(target_tokens, device=logits_f.device)
        token_lp = logits_f[torch.arange(len(target_tokens)), target]
        logprob_sum = float((token_lp - torch.logsumexp(logits_f, dim=-1)).sum())
        is_greedy = bool((logits_f.argmax(dim=-1) == target).all())
        return logprob_sum, is_greedy

    for request in tqdm(requests, desc="Processing requests", disable=disable_tqdm):
        inp, pt, tt = _build_input(request)
        future = self.llm.generate(inp, sampling_params=sp, use_tqdm=False)
        results.append(_process(future, pt, tt))

    profiler.stop("trtllm exec")
    elapsed_time = profiler.elapsed_time_in_sec("trtllm exec")
    logger.info(f"TRTLLM execution time: {elapsed_time:.3f} seconds.")
    profiler.reset("trtllm exec")

    return results


# Perplexity tasks
def loglikelihood_rolling(self, requests, disable_tqdm=False):
    pass

LmEvalWrapper._loglikelihood_tokens = _loglikelihood_tokens
LmEvalWrapper.loglikelihood_rolling = loglikelihood_rolling

from tensorrt_llm.evaluate.lm_eval import *
from tensorrt_llm.llmapi.llm_args import KvCacheConfig

@dataclass
class EvalConfig:
    # Engine config
    free_gpu_memory_fraction: float = 0.8
    max_batch_size: int = 1
    max_tokens:int = 1,

    # Sampling Params
    temperature:float = 0.0,
    return_context_logits: bool = True,
    seed: int = 0

    # LM eval params
    num_samples: Optional[int] = 100
    task: str = "hellaswag"
    few_shots: int = 5
    scores_filter: Optional[str] = "acc_norm,none"

    # Unannotated for printing
    model_dir = Path("/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b")
    engine_dir = Path("/workspace/code/trt_engines/qwen2/W16A16_LOGITS")

def load_config(config_path: str, base: str, model_dir: str) -> EvalConfig:
    with open(config_path) as f:
        data = json.load(f)

    engine_root = Path(base) / "model" / "trt_engines"

    field_names = {f.name for f in fields(EvalConfig)}
    unknown = set(data) - field_names - {"model_dir", "engine_dir"}
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")

    kwargs = {k: v for k, v in data.items() if k in field_names}
    cfg = EvalConfig(**kwargs)

    cfg.model_dir = Path(data["model_dir"]) if "model_dir" in data else Path(model_dir)

    if "engine_dir" in data:
        p = Path(data["engine_dir"])
        cfg.engine_dir = p if p.is_absolute() else engine_root / p

    return cfg

# 

_MEM_PATTERNS = {
    "exec_context_mib":    re.compile(r"Allocated ([0-9.]+) (MiB) for execution context memory"),
    "runtime_buffers_mib": re.compile(r"Allocated ([0-9.]+) (KB|MB|MiB|GiB) GPU memory for runtime buffers"),
    "decoder_mib":         re.compile(r"Allocated ([0-9.]+) (KB|MB|MiB|GiB) GPU memory for decoder"),
    "kv_cache_alloc_mib":  re.compile(r"Allocated ([0-9.]+) (GiB) for max tokens in paged KV cache"),
}
_UNIT_TO_MIB = {"KB": 1/1024, "KiB": 1/1024, "MB": 1, "MiB": 1, "GB": 1024, "GiB": 1024}


@contextmanager
def _capture_trtllm_logs():
    """Redirects stdout/stderr at fd level to capture C++ TRT-LLM log output.
    Yields a dict populated with parsed memory stats after the block exits."""
    stats = {}
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".trtlog")
    tmp_path = tmp.name; tmp.close()

    saved = (os.dup(1), os.dup(2))
    cap = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    sys.stdout.flush(); sys.stderr.flush()
    os.dup2(cap, 1); os.dup2(cap, 2); os.close(cap)

    try:
        yield stats
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        for dst, src in zip((1, 2), saved):
            os.dup2(src, dst); os.close(src)

        text = open(tmp_path).read(); os.unlink(tmp_path)
        sys.stdout.write(text); sys.stdout.flush()

        for key, pat in _MEM_PATTERNS.items():
            if m := pat.search(text):
                stats[key] = round(float(m.group(1)) * _UNIT_TO_MIB[m.group(2)], 2)


def _engine_disk_size_mib(engine_dir: Path) -> float:
    return round(sum(p.stat().st_size for p in engine_dir.glob("*.engine")) / 1024 ** 2, 1)


_CODE_EVAL_TASKS = {"humaneval", "mbpp"}
_GATED_TASK_PREFIXES = {"gpqa"}

def _prepare_task_env(task: str, base: str):
    if task in _CODE_EVAL_TASKS:
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    if any(task.startswith(p) for p in _GATED_TASK_PREFIXES):
        token_path = Path(base) / ".hf_token"
        if token_path.exists():
            os.environ["HF_TOKEN"] = token_path.read_text().strip()

    # TRT-LLM doesnt expose confirm run unsafe code nor log samples, so we patch them manually.
    if task in _CODE_EVAL_TASKS:
        import lm_eval.evaluator as _lm_evaluator
        _orig_eval = _lm_evaluator.evaluate
        _lm_evaluator.evaluate = lambda *a, **kw: _orig_eval(*a, confirm_run_unsafe_code=True, log_samples=False, **kw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON eval config file")
    parser.add_argument("--base", type=str, required=True, help="Path to trt-scripts root directory")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to HuggingFace model snapshot")
    args = parser.parse_args()

    e_config = load_config(args.config, args.base, args.model_dir)
    _prepare_task_env(e_config.task, args.base)

    eval = LmEvalEvaluator(
        task_name=e_config.task,
        num_samples=e_config.num_samples,
    )

    stack = list(eval.task_dict.values())

    while stack:
        obj = stack.pop()
        if isinstance(obj, dict):
            stack.extend(obj.values())
        else:
            obj.set_config(key="num_fewshot", value=e_config.few_shots)
            obj.set_fewshot_seed(seed=e_config.seed)

    with _capture_trtllm_logs() as mem_stats:
        llm = LLM(
            model=e_config.engine_dir,
            tokenizer=e_config.model_dir,
            kv_cache_config=KvCacheConfig(
                free_gpu_memory_fraction=e_config.free_gpu_memory_fraction,
            ),
            max_batch_size=e_config.max_batch_size,
        )

    mem_stats["engine_disk_mib"] = _engine_disk_size_mib(e_config.engine_dir)

    sampling_params = SamplingParams(
        max_tokens=e_config.max_tokens,
        temperature=e_config.temperature,
        return_context_logits=e_config.return_context_logits,
        seed=e_config.seed
    )

    result = eval.evaluate(llm, sampling_params, scores_filter=e_config.scores_filter)

    append_result_csv(e_config, str(result), mem_stats)

    print(f"Measured accuracy: {result}")
    print(f"Memory stats: {mem_stats}")


_MEM_STAT_KEYS = ["engine_disk_mib", "exec_context_mib", "kv_cache_alloc_mib",
                  "runtime_buffers_mib", "decoder_mib"]


def append_result_csv(e_config: EvalConfig, result: str, mem_stats: dict = None, csv_path="results.csv"):
    file_exists = os.path.exists(csv_path)
    mem_stats = mem_stats or {}

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["model", "task", "result", "config params"] + _MEM_STAT_KEYS)

        writer.writerow([e_config.engine_dir.name,
                         e_config.task,
                         result,
                         str(e_config)] + [mem_stats.get(k, "") for k in _MEM_STAT_KEYS])

if __name__ == "__main__":
    main()
    print("finished")
