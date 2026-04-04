#!/bin/bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --job-name="download_model"
#SBATCH -c4
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/home/krisskn/master-thesis/trt-scripts/hpc/logs/download_%j.out

MODEL_ID="mistralai/Mistral-7B-v0.3"
LOCAL_DIR="/cluster/home/krisskn/master-thesis/hf-cache/models/Mistral-7B-v0.3"

srun /usr/bin/apptainer exec --writable-tmpfs \
  /cluster/home/krisskn/master-thesis/trt-scripts/hpc/trtllm-tools.sif \
  python -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL_ID}', local_dir='${LOCAL_DIR}')
print('Download complete.')
"
