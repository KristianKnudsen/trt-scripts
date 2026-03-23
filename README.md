# trt-scripts

Scripts for building, evaluating, and serving quantized TensorRT-LLM engines, developed as part of a master thesis project.

The general workflow is: build a TensorRT engine from a HuggingFace model → evaluate it across benchmarks → optionally serve it via an API.

### Contents

- [model/](#model) — engine building, checkpoints, and serving
- [eval/](#eval) — benchmark evaluation
- [hpc/](#hpc) — HPC cluster environment and SLURM jobs
- [docker/](#docker) — local development environment

---

## `model/`

Contains everything related to building and running TensorRT-LLM engines.

- [`model/build_engine.py`](model/build_engine.py) — main script for compiling a TensorRT engine from a HuggingFace model checkpoint
- `model/configs/` — build configs organized by environment and model
- `model/trt_checkpoints/` — intermediate quantized checkpoints
- `model/trt_engines/` — compiled TensorRT engines, organized by model and quantization
- [`model/api/`](model/api/) — scripts for serving a built engine via `trtllm-serve`
- `model/cli-usage-examples/` — miscellaneous CLI usage examples and benchmarking scripts

Refer to the `model/` directory for documentation on build configs and quantization variants.

---

## `eval/`

Contains the evaluation pipeline for running benchmarks against built engines using `lm-eval`.

- [`eval/custom_lmeval_wrapper.py`](eval/custom_lmeval_wrapper.py) — wrapper script that runs `lm-eval` tasks against a TensorRT-LLM engine
- `eval/configs/` — benchmark configs organized by model and quantization variant

Refer to the `eval/` directory for documentation on supported tasks and config structure.

---

## `hpc/`

Contains the Apptainer container definition and SLURM batch job scripts for running builds and evaluations on the HPC cluster.

- `hpc/jobs/` — SLURM job scripts and templates, organized by model
- [`hpc/readme.md`](hpc/readme.md) — full documentation for the HPC environment, including how to build the container and submit jobs

---

## `docker/`

Contains a Docker-based local development environment for building and testing outside of the HPC cluster.

- `docker/Dockerfile` — image based on `nvcr.io/nvidia/tensorrt-llm/release:1.1.0rc5`
- `docker/Makefile` — shortcuts for building the image and launching containers (`make build`, `make run-shell`, `make run-jupyter`)
- `docker/jobs/` — local equivalents of the HPC job scripts for running builds and evals inside Docker
