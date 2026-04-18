#!/bin/bash
#SBATCH --job-name=flood_cuda
#SBATCH --output=_logs/flood_output_%j.txt
#SBATCH --error=_logs/flood_error_%j.txt
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=defq
#SBATCH --constraint="gpunode,TitanX"
#SBATCH --gres=gpu:1

## This is an example of a SLURM job script to run the program on a GPU node
nvprof ./flood_cuda_soa 3072 2048 M 0.0000001 800 10 30 70 45 240 80 3 80 32 35 83766
nvprof ./flood_cuda_soa 3072 2048 M 0.0000001 800 10 30 70 45 240 800 0.1 80 32 35 83766