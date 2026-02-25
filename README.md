# trt-scripts

## HPC Container Build (Apptainer)

The container image was built from the existing Docker-based setup using an Apptainer definition file (`trtllm-tools.def`) to produce a persistent `.sif` image for the HPC environment.

### Build Steps

```bash
cd /localscratch
apptainer build --fakeroot trtllm-tools.sif ~/trt-scripts/docker/trtllm-tools.def
mv trtllm-tools.sif ~/
```

### Result

Persistent container image:

```
~/trtllm-tools.sif
```

### Notes

- `/localscratch` is used for building due to limitations on network-mounted home directories.
- Rebuild by repeating the same command after updating the `.def` file.

## HPC use

### Load MPI module required on HPC nodes
```
module load OpenMPI/4.1.6-GCC-13.2.0
```

### Start an interactive container shell with GPU access on the HPC node
```
apptainer shell --nv --writable-tmpfs ../../trtllm-tools.sif
```
### Execute the lm-eval test script inside the container environment on the HPC node
```
apptainer exec --nv --writable-tmpfs ../../trtllm-tools.sif bash ./lmeval-test.sh
```
