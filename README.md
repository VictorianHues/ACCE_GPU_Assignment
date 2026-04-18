# Large-Scale Experiments & Profiling

This project includes a full pipeline for generating input configurations, running large batches of experiments on DAS-5, profiling GPU implementations, and analyzing results.

## Overview of Workflow

The experimental pipeline consists of four stages:

1. **Generate input files** → `generate_experiment_inputs.py`  
2. **Run experiments in batch** → `batch_experiments.sh` (SLURM)  
3. **(Optional) Profile runs** → `job.sh` (nvprof)  
4. **Analyze results & visualize** → `analyze_experiments.py`  

---

## Setup Python Environment
To run the Python scripts for generating inputs and analyzing results, set up a virtual environment:

```bash
pip install uv
uv venv
uv sync
```

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

## Outputs

Aggregated results:

```text
_logs/experiment_results_averaged.csv
Figures (PDF):
figure_speedup_vs_clouds.pdf
figure_speedup_vs_grid_size.pdf
figure_precision_loss_vs_rain.pdf
figure_runtime_distribution_by_impl.pdf
figure_correlation_heatmap.pdf
figure_heatmap_*.pdf
figure_bubble_*.pdf
```