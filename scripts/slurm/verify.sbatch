#!/bin/bash
# scripts/slurm/verify.sbatch
# SLURM script for verification

#SBATCH --job-name=neuro_stylometry_verify
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/verify_%j.out
#SBATCH --error=logs/verify_%j.err

# Load modules
module load python/3.11
module load cuda/12.1

# Activate virtual environment
source /home/$USER/venvs/neuro_stylometry/bin/activate

export CUDA_VISIBLE_DEVICES=0

# Run verification
python -m neuro_stylometry verify \
    --model-a /scratch/$USER/artifacts/phase_d/model_dirty \
    --model-b /scratch/$USER/artifacts/phase_d/model_clean \
    --output /scratch/$USER/artifacts/reports/comparison_report.md \
    --config conf/experiments/my_experiment.yaml \
    --mode hpc

echo "Verification completed at $(date)"
