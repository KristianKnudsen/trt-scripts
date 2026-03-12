from tensorrt_llm.runtime import ModelRunnerCpp

def main():
    r = ModelRunnerCpp.from_dir("/workspace/code/trt_engines/qwen2/W16A16_LOGITS")
    print("gather_context_logits:", getattr(r, "gather_context_logits", None))

if __name__ == "__main__":
    main()
