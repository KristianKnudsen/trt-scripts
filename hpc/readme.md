## Container (`trtllm-tools.sif`)

The Apptainer image (`trtllm-tools.sif`) is built from `trtllm-tools.def` and provides the runtime environment for all build and eval jobs. It is based on `nvcr.io/nvidia/tensorrt-llm/release:1.1.0rc5`.

### Build

```bash
cd /localscratch
apptainer build --fakeroot trtllm-tools.sif ~/trt-scripts/hpc/trtllm-tools.def
mv trtllm-tools.sif ~/trt-scripts/hpc/
```

> `/localscratch` is used due to limitations on network-mounted home directories. Rebuild by rerunning after updating `.def`.

---

## Engine Builds (SLURM)

Build scripts live under `hpc/jobs/<model>/`. Each script is an sbatch array job — one task per quantization config.

### Run all quants (Qwen)

```bash
sbatch hpc/jobs/qwen/build_array_all.sh
```

### Create a new build job

1. Copy `hpc/jobs/build_array_job_template.sh` into `hpc/jobs/<model>/`
2. Fill in the placeholders at the top: `<BASE_PATH>`, `<MODEL>`, `<MODEL_DIR>`
3. Add one entry per quant to the `CONFIGS` array and update `--array=0-N`

---

## Evaluation (SLURM)

Each quant has two files:
- `eval_array_<QUANT>.sh` — the sbatch array job (one task per benchmark)
- `submit_eval_<QUANT>.sh` — submits task groups with their own `--time` and resource overrides

### Run all quants (Qwen)

```bash
bash hpc/jobs/qwen/submit_eval_all.sh
```

### Run a single quant

```bash
bash hpc/jobs/qwen/submit_eval_W16A16.sh
```

### Monitor jobs

```bash
squeue --me
```

### Create a new eval job

1. Copy `hpc/jobs/eval_array_job_template.sh` into `hpc/jobs/<model>/` and fill in `<BASE_PATH>`, `<MODEL>`, `<MODEL_DIR>`, `<QUANT>`
2. Copy `hpc/jobs/eval_job_template.sh` (or write a matching submit script) and update the `--array` and `--time` groups
3. Add eval configs under `eval/configs/<model>/<QUANT>/` — one JSON per benchmark

---

## Serving (optional)

A built engine can be served via `trtllm-serve`. See `model/api/serve.sh`.
