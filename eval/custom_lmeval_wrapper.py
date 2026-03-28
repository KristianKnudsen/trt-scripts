import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import csv
import json
import re
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import Optional
from pathlib import Path
from tensorrt_llm.sampling_params import SamplingParams

import lmeval_patches

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
    print_outputs: int = 0

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
# LmEvalEvaluator multiplies all metrics by 100; perplexity metrics must be divided back
_PERPLEXITY_TASKS = {"wikitext", "ptb", "lambada_openai"}

def _prepare_task_env(task: str, base: str):
    if task in _CODE_EVAL_TASKS:
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    if any(task.startswith(p) for p in _GATED_TASK_PREFIXES):
        token_path = Path(base) / ".hf_token"
        if token_path.exists():
            os.environ["HF_TOKEN"] = token_path.read_text().strip()

    # TRT-LLM doesnt expose confirm run unsafe code nor log samples, so we patch them manually.
    lmeval_patches.patch_evaluator(task)


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
    lmeval_patches.config["print_outputs"] = e_config.print_outputs

    sampling_params = SamplingParams(
        max_tokens=e_config.max_tokens,
        temperature=e_config.temperature,
        return_context_logits=e_config.return_context_logits,
        seed=e_config.seed
    )

    result = eval.evaluate(llm, sampling_params, scores_filter=e_config.scores_filter)

    # LmEvalEvaluator.evaluate() blindly multiplies all metrics by 100 to normalize
    # accuracy scores into a 0-100 range. Perplexity is not an accuracy metric,
    # so we undo that scaling here.
    if e_config.task in _PERPLEXITY_TASKS:
        result /= 100
        stats = lmeval_patches.ppl_stats
        # It also doesnt do Token perplexity by default so we extract it from the patched thingi.
        if stats["num_tokens"] > 0:
            import math
            token_ppl = math.exp(-stats["logprob_sum"] / stats["num_tokens"])
            print(f"Token perplexity: {token_ppl:.4f}")

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
