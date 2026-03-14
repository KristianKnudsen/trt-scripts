#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --job-name="build_qwen_w8a16"
#SBATCH -c2
#SBATCH --time=00-00:05:00
#SBATCH --begin=now

module load OpenMPI/4.1.6-GCC-13.2.0

mpirun --allow-run-as-root -n 1 \
  apptainer exec --nv --writable-tmpfs \
  /cluster/home/krisskn/master-thesis/trt-scripts/hpc/trtllm-tools.sif \
  bash /cluster/home/krisskn/master-thesis/trt-scripts/model/run-build.sh
