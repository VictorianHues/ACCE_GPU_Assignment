#!/bin/bash
#SBATCH --job-name=cuda_batch_experiments
#SBATCH --output=_logs/cuda_batch_experiments_out_%j.txt
#SBATCH --error=_logs/cuda_batch_experiments_err_%j.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=defq
#SBATCH --constraint="gpunode,TitanX"
#SBATCH --gres=gpu:1

set -euo pipefail

BINS=("flood_cuda")
INPUTS=(test_files/*.in)
OUTCSV="_logs/cuda_batch_experiments_results.csv"
RUNS=10

echo "run,binary,input_file,minute,max_spillage_minute,max_spillage_scenario,max_water_scenario,total_rain,total_water,total_water_loss,precision_loss,runtime" > "$OUTCSV"

for bin in "${BINS[@]}"; do
  if [[ ! -x ./$bin ]]; then
    echo "Skipping $bin (not executable)" >&2
    continue
  fi
  for input in "${INPUTS[@]}"; do
    for run in $(seq 1 $RUNS); do
      outfile=$(mktemp)
      ./$bin $(< "$input") > "$outfile" 2>&1 || echo "Run failed: $bin $input (run $run)" >&2
      # Robust extraction of all Result fields (ignore commas)
      result_fields=$(awk '/Result:/ {gsub(/,/," "); for(i=2;i<=8;i++) printf "%s ", $i; exit}' "$outfile")
      set -- $result_fields
      minute=$1
      max_spillage_minute=$2
      max_spillage_scenario=$3
      max_water_scenario=$4
      total_rain=$5
      total_water=$6
      total_water_loss=$7
      precision_loss=$(awk '/Check precision loss:/ {print $4; exit}' "$outfile")
      runtime=$(awk '/Time:/ {print $2; exit}' "$outfile")
      echo "$run,$bin,$input,$minute,$max_spillage_minute,$max_spillage_scenario,$max_water_scenario,$total_rain,$total_water,$total_water_loss,$precision_loss,$runtime" >> "$OUTCSV"
      rm "$outfile"
    done
  done

done

echo "Results written to $OUTCSV"
