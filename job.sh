#!/bin/bash
#SBATCH --job-name=flood_cuda
#SBATCH --output=flood_output_%j.txt
#SBATCH --error=flood_error_%j.txt
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=defq
#SBATCH --constraint="gpunode,TitanX"
#SBATCH --gres=gpu:1

<<<<<<< Updated upstream
## This is an example of a SLURM job script to run the program on a GPU node
./flood_cuda $(< test_files/debug.in)
=======
#./flood_seq $(< test_files/large_mountains.in)
#nvprof ./flood_cuda $(< test_files/large_mountains.in)
nvprof ./flood_cuda_soa $(< test_files/large_mountains_extreme.in)
>>>>>>> Stashed changes
