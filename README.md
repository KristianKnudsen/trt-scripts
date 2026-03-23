# TensorRT-LLM Evaluation Tools

Scripts for quantizing, building, evaluating, and serving TensorRT-LLM engines, developed as part of a master thesis project.

### Contents

- [model/](#model) — quantization, checkpoints, engine builds, configs, and serving
- [eval/](#eval) — benchmark evaluation and configs
- [hpc/](#hpc) — HPC cluster container and SLURM job launchers
- [docker/](#docker) — local Docker environment and job launchers

---

## `model/`

Contains everything related to preparing and running TensorRT-LLM engines. The general flow is: quantize a HuggingFace model → build a quantized checkpoint → compile a TensorRT engine.

- [model/build_engine.py](model/build_engine.py) — main script that handles quantization, checkpoint creation, and engine compilation
- [model/configs/](model/configs/) — build configs organized by environment and model
- [model/trt_checkpoints/](model/trt_checkpoints/) — quantized model checkpoints
- [model/trt_engines/](model/trt_engines/) — compiled TensorRT engines, organized by model and quantization variant
- [model/api/](model/api/) — scripts for serving a built engine via `trtllm-serve`
- [model/cli-usage-examples/](model/cli-usage-examples/) — miscellaneous CLI usage and benchmarking scripts

Refer to the [model/](model/) directory for documentation on build configs and quantization variants.

---

## `eval/`

Contains the evaluation pipeline for running benchmarks against built engines using `lm-eval`.

- [eval/custom_lmeval_wrapper.py](eval/custom_lmeval_wrapper.py) — wrapper script that runs `lm-eval` tasks against a TensorRT-LLM engine
- [eval/configs/](eval/configs/) — benchmark configs organized by model and quantization variant

Refer to the [eval/](eval/) directory for documentation on supported tasks and config structure.

---

## `hpc/`

Contains the Apptainer container definition and SLURM batch job scripts for running builds and evaluations on the HPC cluster. Jobs are launched from [hpc/jobs/](hpc/jobs/).

Refer to the [hpc/](hpc/) directory for documentation on the container setup and how to submit jobs.

---

## `docker/`

Contains a Docker-based local development environment for building and testing outside of the HPC cluster. Jobs are launched from [docker/jobs/](docker/jobs/).

- [docker/Dockerfile](docker/Dockerfile) — image based on `nvcr.io/nvidia/tensorrt-llm/release:1.1.0rc5`
- [docker/Makefile](docker/Makefile) — shortcuts for building the image and launching containers (`make build`, `make run-shell`, `make run-jupyter`)
