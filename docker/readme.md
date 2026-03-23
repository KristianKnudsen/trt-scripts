# docker/

Local development environment for running builds and evaluations outside of the HPC cluster. Uses the same base image as the HPC Apptainer container (`nvcr.io/nvidia/tensorrt-llm/release:1.1.0rc5`).

### Contents

- [Getting Started](#getting-started)
- [Makefile Commands](#makefile-commands)
- [Running Jobs](#running-jobs)
- [jobs/](#jobs)

---

## Getting Started

Build the image from the [Dockerfile](Dockerfile):

```bash
make build
```

Launch a detached container with host networking:

```bash
make run-shell-detached-hosted
```

Then attach to it:

```bash
docker exec -it trt_shell bash
```

The container mounts:
- `~/.cache/huggingface` → `/root/.cache/huggingface` — HuggingFace model cache
- `trtllm_data` (Docker volume) → `/workspace` — persistent workspace

---

## Makefile Commands

| Command | Description |
|---|---|
| `make build` | Build the Docker image |
| `make clean` | Remove dangling Docker images |
| `make run-shell` | Launch an interactive shell with GPU access |
| `make run-jupyter` | Launch JupyterLab on port `25566` |
| `make run-shell-detached` | Launch a detached shell container |
| `make run-shell-detached-hosted` | Launch a detached shell using host networking |

---

## Running Jobs

Unlike the HPC setup, there is no SLURM — jobs are run directly inside the container. From a shell inside the container, invoke the build or eval scripts directly:

```bash
bash /workspace/trt-scripts/docker/jobs/<model>/your_job.sh
```

---

## `jobs/`

Job scripts for running builds and evals inside the container, organized by model. These mirror the HPC job scripts but without the SBATCH headers.

```
jobs/
  <model>/
    build_<QUANT>.sh
    eval_<QUANT>.sh
```
