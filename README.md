# trt-scripts


# Load MPI module required on HPC nodes (ensures proper runtime environment)
module load OpenMPI/4.1.6-GCC-13.2.0

# Start an interactive container shell with GPU access on the HPC node
apptainer shell --nv --writable-tmpfs ../../trtllm-tools.sif

# Execute the lm-eval test script inside the container environment on the HPC node
apptainer exec --nv --writable-tmpfs ../../trtllm-tools.sif bash ./lmeval-test.sh
