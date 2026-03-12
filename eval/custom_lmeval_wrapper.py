from tensorrt_llm.evaluate.lm_eval import LmEvalWrapper
import csv
import os
import copy
import torch
import torch.nn.functional as F
from tqdm import tqdm
from tensorrt_llm.sampling_params import SamplingParams
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

# Context logprobs, teacher forces...
# TODO: fix memory leakage
def _loglikelihood_tokens(self: LmEvalWrapper, requests, disable_tqdm=False, **kwargs):
    profiler.start("trtllm exec")
    results = []

    sp = copy.deepcopy(self.sampling_params) if self.sampling_params else SamplingParams()
    sp.max_tokens = 1
    sp.temperature = 0.0
    sp.return_context_logits = True

    with torch.no_grad():
        for request in tqdm(requests, desc="Submitting requests", disable=disable_tqdm):
            _, prompt_tokens, target_tokens = request

            prompt_tokens = list(prompt_tokens)
            target_tokens = list(target_tokens)

            if not target_tokens:
                results.append((0.0, True))
                continue

            inp = prompt_tokens + target_tokens[:-1]

            gen_output = self.llm.generate(inp, sampling_params=sp)
            output = gen_output.result()

            logits = output.context_logits

            start = len(prompt_tokens) - 1
            end = start + len(target_tokens)
            logits = logits[start:end]

            logprob_sum = 0.0
            is_greedy = True

            for i, tok in enumerate(target_tokens):
                token_logits = logits[i]

                row = token_logits.float()
                logprob_sum += float(row[tok] - torch.logsumexp(row, dim=-1))

                if tok != int(token_logits.argmax()):
                    is_greedy = False

            results.append((logprob_sum, is_greedy))

            # Memory issues...
            del logits
            del output
            del gen_output

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
    num_samples: Optional[int] = 1
    task: str = "mmlu"
    few_shots: int = 5
    free_gpu_memory_fraction: float = 0.5
    max_batch_size: int = 1
    max_tokens:int = 1,
    temperature:float = 0.0,
    return_context_logits: bool = True,
    seed: int = 0

    # Unannotated for printing
    model_dir = Path("/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b")
    engine_dir = Path("/workspace/code/trt_engines/qwen2/W16A16_LOGITS")

def main():

    e_config = EvalConfig()

    eval = LmEvalEvaluator(
        task_name=e_config.task,
        num_samples=e_config.num_samples
    )

    stack = list(eval.task_dict.values())

    while stack:
        obj = stack.pop()
        if isinstance(obj, dict):
            stack.extend(obj.values())
        else:
            obj.set_config(key="num_fewshot", value=e_config.few_shots)
            obj.set_fewshot_seed(seed=e_config.seed)

    llm = LLM(
        model=e_config.engine_dir,
        tokenizer=e_config.model_dir,
        kv_cache_config=KvCacheConfig(
            free_gpu_memory_fraction=e_config.free_gpu_memory_fraction,
        ),
        max_batch_size=e_config.max_batch_size,
    )

    sampling_params = SamplingParams(
        max_tokens=e_config.max_tokens,
        temperature=e_config.temperature,
        return_context_logits=e_config.return_context_logits,
        seed=e_config.seed
    )

    result = eval.evaluate(llm, sampling_params)

    append_result_csv(e_config, str(result))

    print(f"Measured accuracy: {result}")


def append_result_csv(e_config: EvalConfig, result: str, csv_path="results.csv"):

    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["model", "task", "result", "config params"])

        writer.writerow([e_config.engine_dir.name, 
                         e_config.task, 
                         result,
                         str(e_config)])


if __name__ == "__main__":
    main()
    print("finished")
