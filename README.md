# Large-Scale Experiments & Profiling

This project includes a full pipeline for generating input configurations, running large batches of experiments on DAS-5, profiling GPU implementations, and analyzing results.

## Overview of Workflow

The experimental pipeline consists of four stages:

1. **Generate input files** → `generate_experiment_inputs.py`  
2. **Run experiments in batch** → `batch_experiments.sh` (SLURM)  
3. **(Optional) Profile runs** → `job.sh` (nvprof)  
4. **Analyze results & visualize** → `analyze_experiments.py`  

---

## Load CUDA Module
Before running experiments on DAS-5, load the CUDA module:
```bash
module load cuda12.6/toolkit
```

## Setup Python Environment
To run the Python scripts for generating inputs and analyzing results, set up a virtual environment:

```bash
pip install uv
uv venv
uv sync
```

NOTE: The python scripts will not run on the DAS-5 login node due to module versioning and restrictions. You will need to pull the experimental results from the cluster to your local machine. This can be accomplished using `scp`, `rsync`, or just copy pasting from the cluster to transfer the `_logs/experiment_results.csv` file and any generated figures.

## Build Binaries
Make sure to build the CPU and CUDA implementations before running experiments:
```bash
make all
```

## Generate Experiment Inputs

You can generate a large set of structured `.in` files using:

```bash
uv run generate_experiment_inputs.py
```

This will:

Create a test_files/ directory (if not present)
Generate input files of the form:
```exp_<rows>x<cols>_<scenario>_c<clouds>_ex<factor>_t<threshold>_m<minutes>.in```

Example:

```exp_512x512_V_c256_ex10_t0p000001_m1000.in```

### Design Notes
The generator spans:
- Grid sizes: small → very large (up to 8192×8192)
- Scenarios: M, V, D, d
- Cloud densities: sparse → dense


## Run Batch Experiments (SLURM)

To execute all experiments across all implementations:

```bash
sbatch batch_experiments.sh
```

This runs the sequential, CUDA AoS, and CUDA SoA implementations on all generated input files, multiple times per configuration, and extracts key metrics from the output.

The sequential runs are a bottleneck in runtime, so you may also choose to run each implementation individually, running the sequential implementation once to get baseline metrics, and then running the CUDA implementations many times separately to speed up the overall process.

```bash
sbatch batch_experiments_seq.sh
sbatch batch_experiments_cuda.sh
sbatch batch_experiments_cuda_soa.sh
```

This requires the resulting CSV files to be merged later into a singular experimental dataset, but allows you to run the CUDA experiments much faster by not waiting for many sequential runs to complete.


### What This Script Does
Runs:
- flood_seq (CPU baseline)
- flood_cuda
- flood_cuda_soa

Iterates over:
1. All .in files in test_files/
2. Multiple runs per configuration (RUNS=100)
3. Extracts key metrics from program output
4. Writes results to:
```_logs/experiment_results.csv```

### Alternative Cloud-Parallel Implementation
The provided CUDA implementations are parallelized over the grid. For an alternative approach, you can parallelize the rainfall step over the clouds instead. Because of time constraints explained in the discussion section of our report, this version can only be enabled by uncommenting the relevant sections in `flood_cuda_soa.cu` and `flood_cuda.cu` and is not included in the batch experiment scripts. You can run this version manually by uncommenting the relevant sections in the CUDA source files and re-commenting the original grid-parallel rainfall kernel, then running the modified binaries on the cluster.

This implementation will pass all of the `check_correctness.py` tests, but just didn't make into the final implementation due to time constraints.

## Output Format

Each row contains:
``` csv
run,binary,input_file,minute,max_spillage_minute,max_spillage_scenario,
max_water_scenario,total_rain,total_water,total_water_loss,
precision_loss,runtime
```

## GPU Profiling (Optional)

To profile CUDA implementations using nvprof, submit:

``` bash
sbatch job.sh
```

This runs:

``` bash
./flood_seq $(< test_files/input_file.in)
nvprof ./flood_cuda $(< test_files/input_file.in)
nvprof ./flood_cuda_soa $(< test_files/input_file.in)
```

Change the input file as needed for different configurations.

## Profiling Output
Saved in:
- ```_logs/flood_output_<jobid>.txt```
- ```_logs/flood_error_<jobid>.txt```


## Analysis & Visualization

After experiments complete, run:

``` bash
make build_visualizations
```

or directly:

``` bash
uv run analyze_experiments.py
```
