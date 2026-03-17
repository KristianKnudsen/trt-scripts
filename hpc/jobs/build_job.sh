#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --job-name="build_qwen_w8a16"
#SBATCH -c2
#SBATCH --time=00-00:05:00
#SBATCH --begin=now

module load OpenMPI/4.1.6-GCC-13.2.0

BASE=/cluster/home/krisskn/master-thesis/trt-scripts

mpirun --allow-run-as-root -n 1 \
  apptainer exec --nv --writable-tmpfs \
  $BASE/hpc/trtllm-tools.sif \
  bash $BASE/model/run-build.sh \
    --config $BASE/model/build_config_W4A16_AWQ_LOGITS.json \
    --paths $BASE/paths/paths.json
