#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH -t 03:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=20G
#SBATCH --output="./log.out"
#SBATCH --job-name="experiment"

now=$(date)
echo "$now"
#cd /gpfs/home5/obevza/Master-thesis/   
# Loading modules
# module load 2024
# module load Python/3.10.4-GCCcore-11.3.0

# Activate your conda environment
conda activate toy

# Display GPU name
nvidia-smi --query-gpu=name --format=csv,noheader

# Run the script
python 2_toy_problem.py 
    
echo "JOB DONE"