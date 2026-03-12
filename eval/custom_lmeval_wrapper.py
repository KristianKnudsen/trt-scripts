from tensorrt_llm.evaluate.lm_eval import LmEvalWrapper

import copy
import torch
import torch.nn.functional as F
from tqdm import tqdm
from tensorrt_llm.sampling_params import SamplingParams

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

def main():
    eval = LmEvalEvaluator(
        task_name="mmlu",
        #num_samples=1
    )

    stack = list(eval.task_dict.values())

    while stack:
        obj = stack.pop()
        if isinstance(obj, dict):
            stack.extend(obj.values())
        else:
            obj.set_config(key="num_fewshot", value=5)
            obj.set_fewshot_seed(seed=0)

    model_dir = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
    engine_dir = "/workspace/code/trt_engines/qwen2/W16A16_LOGITS"

    llm = LLM(
        model=engine_dir,
        tokenizer=model_dir,
        kv_cache_config=KvCacheConfig(
            free_gpu_memory_fraction=0.5,
        ),
        max_batch_size=1,
    )

    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        return_context_logits=True
    )

    result = eval.evaluate(llm, sampling_params)

    print(result)


if __name__ == "__main__":
    main()
    print("finished")
