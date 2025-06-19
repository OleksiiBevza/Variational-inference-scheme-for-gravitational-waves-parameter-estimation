#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu_a100
#SBATCH -t 01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=20G
#SBATCH --output="./log.out"
#SBATCH --job-name="experiment"

now=$(date)
echo "$now"

# Loading modules
# module load 2024
# module load Python/3.10.4-GCCcore-11.3.0

# Activate your conda environment
conda activate /home/obevza/miniconda3/envs/flowMC

# Display GPU name
nvidia-smi --query-gpu=name --format=csv,noheader

# Run the script
python experiment.py \
    --experiment-type "gaussian" \
    --outdir ./outdir/ \
    --n-dims 2 \
    
echo "JOB DONE"