import copy
import torch
from tqdm import tqdm
from tensorrt_llm.evaluate.lm_eval import LmEvalWrapper, profiler, logger
from tensorrt_llm.sampling_params import SamplingParams


def generate_until(self, requests, disable_tqdm: bool = False):
    profiler.start("trtllm exec")
    results = []
    for request in tqdm(requests, desc="Submitting requests", disable=disable_tqdm):
        prompt, gen_kwargs = request.args
        sampling_params = self._get_sampling_params(gen_kwargs)
        output = self.llm.generate_async(prompt, sampling_params=sampling_params, streaming=self.streaming)
        results.append(output)

    outputs = []
    for output in tqdm(results, desc="Fetching responses", disable=disable_tqdm):
        outputs.append(output.result())

    profiler.stop("trtllm exec")
    logger.info(f"TRTLLM execution time: {profiler.elapsed_time_in_sec('trtllm exec'):.3f} seconds.")
    profiler.reset("trtllm exec")

    texts = [output.outputs[0].text for output in outputs]
    n = min(config["print_outputs"], len(texts))
    for req, text in zip(requests[:n], texts[:n]):
        print(f"\n[DEBUG] Input: ...{req.args[0][-300:]}")
        print(f"[DEBUG] Output: {text}")
    return texts


def _loglikelihood_tokens(self, requests, disable_tqdm=False, **kwargs):
    """
    Full MMLU task will crash on certain hardware (HPC will work, allocate a lot of ram). Divide the tasks as much as possible for stable evaluation. Making big requests with gather logits enabled
    seems to have some kind of memory accumulation in the engine.
    """
    profiler.start("trtllm exec")

    sp = copy.deepcopy(self.sampling_params) if self.sampling_params else SamplingParams()
    sp.max_tokens = 1
    sp.temperature = 0.0
    sp.return_context_logits = True

    results = []
    for _, prompt_tokens, target_tokens in tqdm(requests, desc="Processing requests", disable=disable_tqdm):
        output = self.llm.generate(prompt_tokens + target_tokens[:-1], sampling_params=sp, use_tqdm=False).result()
        start = len(prompt_tokens) - 1
        logits_f = output.context_logits[start:start + len(target_tokens)].clone()

        target = torch.tensor(target_tokens, device=logits_f.device)
        # log-softmax of the correct token at each position, summed as the overall sequence score
        logprob_sum = float((logits_f[torch.arange(len(target_tokens)), target] - torch.logsumexp(logits_f, dim=-1)).sum())
        # true if the model's top-1 prediction matched the target at every position
        is_greedy = bool((logits_f.argmax(dim=-1) == target).all())
        results.append((logprob_sum, is_greedy))

    profiler.stop("trtllm exec")
    logger.info(f"TRTLLM execution time: {profiler.elapsed_time_in_sec('trtllm exec'):.3f} seconds.")
    profiler.reset("trtllm exec")

    return results


# Perplexity tasks
def loglikelihood_rolling(self, requests, disable_tqdm=False):
    pass


LmEvalWrapper._loglikelihood_tokens = _loglikelihood_tokens
LmEvalWrapper.loglikelihood_rolling = loglikelihood_rolling
LmEvalWrapper.generate_until = generate_until

config = {
    "print_outputs": 0,
}


_CODE_EVAL_TASKS = {"humaneval", "mbpp"}

def patch_evaluator(task: str):
    import lm_eval.evaluator as _lm_evaluator
    _orig_eval = _lm_evaluator.evaluate
    if task in _CODE_EVAL_TASKS:
        _lm_evaluator.evaluate = lambda *a, **kw: _orig_eval(*a, confirm_run_unsafe_code=True, log_samples=False, **kw)
