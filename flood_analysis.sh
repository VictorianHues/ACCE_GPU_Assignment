#!/bin/bash
#SBATCH --job-name=flood_analysis
#SBATCH --output=_logs/flood_analysis_output_%j.txt
#SBATCH --error=_logs/flood_analysis_error_%j.txt
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=defq
#SBATCH --constraint="gpunode,TitanX"
#SBATCH --gres=gpu:1

set -euo pipefail

INPUT_FILE="test_files/large_mountains.in"
RUN_ID="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
TARGETS=("flood_cuda" "flood_cuda_soa")

mkdir -p _logs

get_kernel_filter() {
	local target="$1"
	if [[ "$target" == "flood_cuda_soa" ]]; then
		echo "alt_calc_rainfall_kernel_soa|compute_spillage_kernel|compute_private_spillage_propagation_kernel|compute_spillage_propagation_kernel|cloud_movement_kernel_soa|reset_ancillary_structures_kernel"
	else
		echo "alt_calc_rainfall_kernel|compute_spillage_kernel|compute_private_spillage_propagation_kernel|compute_spillage_propagation_kernel|cloud_movement_kernel|reset_ancillary_structures_kernel"
	fi
}

get_hot_kernels() {
	local target="$1"
	if [[ "$target" == "flood_cuda_soa" ]]; then
		echo "alt_calc_rainfall_kernel_soa|compute_spillage_kernel|compute_private_spillage_propagation_kernel|compute_spillage_propagation_kernel"
	else
		echo "alt_calc_rainfall_kernel|compute_spillage_kernel|compute_private_spillage_propagation_kernel|compute_spillage_propagation_kernel"
	fi
}

# Discover supported counters on this GPU
nvprof --query-metrics > "_logs/nvprof_query_metrics_${RUN_ID}.txt" 2>&1
nvprof --query-events > "_logs/nvprof_query_events_${RUN_ID}.txt" 2>&1

for target in "${TARGETS[@]}"; do
	if [[ ! -x "./$target" ]]; then
		echo "Skipping $target (binary not found or not executable)"
		continue
	fi

	KERNEL_FILTER="$(get_kernel_filter "$target")"
	HOT_KERNELS="$(get_hot_kernels "$target")"

	# Baseline runtime profile
	nvprof \
		--log-file "_logs/nvprof_runtime_${target}_${RUN_ID}.txt" \
		"./$target" $(< "$INPUT_FILE")

	# Full API/GPU trace
	nvprof \
		--print-gpu-trace \
		--print-api-trace \
		--csv \
		--log-file "_logs/nvprof_trace_${target}_${RUN_ID}.csv" \
		"./$target" $(< "$INPUT_FILE")

	# Focused kernel timeline (post-filtered from full GPU trace)
	nvprof \
		--print-gpu-trace \
		--csv \
		--log-file "_logs/nvprof_gpu_trace_${target}_${RUN_ID}.csv" \
		"./$target" $(< "$INPUT_FILE")

	grep -E "^\"|$KERNEL_FILTER" "_logs/nvprof_gpu_trace_${target}_${RUN_ID}.csv" > "_logs/nvprof_kernels_${target}_${RUN_ID}.csv" || true

	# Occupancy + divergence metrics
	nvprof \
		--kernels "$HOT_KERNELS" \
		--metrics achieved_occupancy,sm_efficiency,warp_execution_efficiency,branch_efficiency \
		--log-file "_logs/nvprof_occ_div_${target}_${RUN_ID}.txt" \
		"./$target" $(< "$INPUT_FILE")

	# Memory behavior metrics
	nvprof \
		--kernels "$HOT_KERNELS" \
		--metrics gld_efficiency,gst_efficiency,gld_throughput,gst_throughput,dram_read_throughput,dram_write_throughput,l2_read_hit_rate,l2_write_hit_rate \
		--log-file "_logs/nvprof_memory_${target}_${RUN_ID}.txt" \
		"./$target" $(< "$INPUT_FILE")

	# One-shot derived analysis summary.
	# --kernels must come before --analysis-metrics so the analysis is scoped.
	nvprof \
		--kernels "$HOT_KERNELS" \
		--analysis-metrics \
		--export-profile "_logs/nvprof_analysis_${target}_${RUN_ID}.nvprof" \
		-u us \
		--log-file "_logs/nvprof_analysis_${target}_${RUN_ID}.txt" \
		"./$target" $(< "$INPUT_FILE")

	# Atomic-metric discovery (first 8 atomic metrics, if available)
	ATOMIC_METRICS=$(awk '/^[[:alnum:]_]+/ && tolower($1) ~ /atomic/ {print $1}' "_logs/nvprof_query_metrics_${RUN_ID}.txt" | head -n 8 | paste -sd, -)
	if [[ -n "$ATOMIC_METRICS" ]]; then
		nvprof \
			--kernels "$HOT_KERNELS" \
			--metrics "$ATOMIC_METRICS" \
			--log-file "_logs/nvprof_atomic_${target}_${RUN_ID}.txt" \
			"./$target" $(< "$INPUT_FILE")
	fi
done

SUMMARY_FILE="_logs/nvprof_summary_${RUN_ID}.txt"
{
	echo "nvprof comparison summary (RUN_ID=${RUN_ID})"
	echo ""
	for target in "${TARGETS[@]}"; do
		runtime_log="_logs/nvprof_runtime_${target}_${RUN_ID}.txt"
		if [[ ! -f "$runtime_log" ]]; then
			echo "[$target] no runtime log found"
			echo ""
			continue
		fi

		echo "[$target]"
		awk '/GPU activities:/,/API calls:/' "$runtime_log"
		echo ""
	done
} > "$SUMMARY_FILE"

echo "Wrote summary: $SUMMARY_FILE"

