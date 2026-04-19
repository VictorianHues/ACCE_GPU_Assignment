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

################# SMALL MOUNTAINS ############################# 

printf "SMALL MOUNTAINS - flood_cuda_soa\n"

./flood_seq $(cat test_files/small_mountains.in) > res_seq_m.out
./flood_cuda_soa $(cat test_files/small_mountains.in) > res_cuda_soa_m.out
python3 test_files/check_correctness.py res_seq_m.out res_cuda_soa_m.out

printf "SMALL MOUNTAINS - flood_cuda\n"

./flood_cuda $(cat test_files/small_mountains.in) > res_cuda_m.out
python3 test_files/check_correctness.py res_seq_m.out res_cuda_m.out

############################# CUSTOM CLOUDS #############################

printf "CUSTOM CLOUDS - flood_cuda_soa\n"

./flood_seq $(cat test_files/custom_clouds.in) > res_seq_c.out
./flood_cuda_soa $(cat test_files/custom_clouds.in) > res_cuda_soa_c.out
python3 test_files/check_correctness.py res_seq_c.out res_cuda_soa_c.out

printf "CUSTOM CLOUDS - flood_cuda\n"

./flood_cuda $(cat test_files/custom_clouds.in) > res_cuda_c.out
python3 test_files/check_correctness.py res_seq_c.out res_cuda_c.out

############################# MEDIUM LOWER DAM #############################

printf "MEDIUM LOWER DAM - flood_cuda_soa\n"

./flood_seq $(cat test_files/medium_lower_dam.in) > res_seq_d.out
./flood_cuda_soa $(cat test_files/medium_lower_dam.in) > res_cuda_soa_d.out
python3 test_files/check_correctness.py res_seq_d.out res_cuda_soa_d.out

printf "MEDIUM LOWER DAM - flood_cuda\n"

./flood_cuda $(cat test_files/medium_lower_dam.in) > res_cuda_d.out
python3 test_files/check_correctness.py res_seq_d.out res_cuda_d.out

############################# MEDIUM HIGHER DAM #############################

printf "MEDIUM HIGHER DAM - flood_cuda_soa\n"

./flood_seq $(cat test_files/medium_higher_dam.in) > res_seq_h.out
./flood_cuda_soa $(cat test_files/medium_higher_dam.in) > res_cuda_soa_h.out
python3 test_files/check_correctness.py res_seq_h.out res_cuda_soa_h.out

printf "MEDIUM HIGHER DAM - flood_cuda\n"

./flood_cuda $(cat test_files/medium_higher_dam.in) > res_cuda_h.out
python3 test_files/check_correctness.py res_seq_h.out res_cuda_h.out