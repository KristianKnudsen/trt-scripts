#!/usr/bin/env python3
import argparse

import json

from pathlib import Path


SMI_QUERY = (
    "timestamp,name,utilization.gpu,utilization.memory,"
    "memory.total,memory.used,memory.free,power.draw"
)


def load_config(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def run_benchmarks(args) -> int:
    # TODO: Implement logging
    pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TensorRT-LLM latency benchmark sweeps.")
    parser.add_argument("--base", type=Path, required=True, help="Path to base trt-scripts folder")
    parser.add_argument("--engine", type=str, required=True, help="Engine folder name under trt_engines in the model folder")
    parser.add_argument("--tokenizer", type=Path,required=True, help="Location of original model tokenizer")
    parser.add_argument("--dataset", tyoe=str, required=True,help="name or folder + name of the dataset to use in the bench/datasets folder")
    parser.add_argument("--config", type=str, required=True,help="config name or folder + config of the config to use in the bench/configs folder")

    return run_benchmarks(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
