#!/bin/bash
# scripts/slurm/verify.sbatch
# SLURM script for CHG + SVS verification

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

# Load modules (adjust based on your HPC environment)
module load python/3.11
module load cuda/12.1

# Activate virtual environment
source /home/$USER/venvs/neuro_stylometry/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

export CUDA_VISIBLE_DEVICES=0

# Verification artifacts directory structure
# Input: /scratch/$USER/artifacts/phase_d/baseline/
#        /scratch/$USER/artifacts/phase_d/constrained/
#        /scratch/$USER/artifacts/phase_a/clean_dataset.arrow
# Output: /scratch/$USER/artifacts/phase_d/chg/chg_gates_{baseline,constrained}.pt
#         /scratch/$USER/artifacts/phase_d/chg/svs_{baseline,constrained}.json
#         /scratch/$USER/artifacts/phase_d/chg/head_classification_{baseline,constrained}.json
#         /scratch/$USER/artifacts/phase_d/chg/verification_summary.json

echo "Starting CHG + SVS verification at $(date)"

# Run verification (CHG learns head gates, SVS computes stylometric validity)
python -m neuro_stylometry verify \
    --dataset /scratch/$USER/artifacts/phase_a/clean_dataset.arrow \
    --phase-d-dir /scratch/$USER/artifacts/phase_d \
    --artifacts-dir /scratch/$USER/artifacts/phase_a \
    --output-dir /scratch/$USER/artifacts/phase_d/chg

echo "Verification completed at $(date)"

# Optional: Generate Phase D report after verification
# Uncomment the following to generate HTML report automatically
# echo "Generating Phase D report at $(date)"
# python -m neuro_stylometry report-phase-d \
#     --phase-d-dir /scratch/$USER/artifacts/phase_d \
#     --output-dir /scratch/$USER/artifacts/reports \
#     --dataset /scratch/$USER/artifacts/phase_a/clean_dataset.arrow
# echo "Report generation completed at $(date)"
