#!/bin/bash
# scripts/slurm/phase_d.sbatch
# SLURM script for Phase D training (baseline + constrained)

#SBATCH --job-name=neuro_stylometry_phase_d
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/phase_d_%j.out
#SBATCH --error=logs/phase_d_%j.err

# Load modules (adjust based on your HPC environment)
module load python/3.11
module load cuda/12.1
module load cudnn/8.9.7

# Activate virtual environment
source /home/$USER/venvs/neuro_stylometry/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# NUMA pinning for better GPU performance
NUMA_NODE=$(nvidia-smi topo -m | grep GPU0 | awk '{print $NF}' | head -1)
export CUDA_VISIBLE_DEVICES=0

# Phase D artifacts directory structure
# Input: /scratch/$USER/artifacts/phase_a/clean_dataset.arrow
#        /scratch/$USER/artifacts/phase_a/projection_matrix.pt
# Output: /scratch/$USER/artifacts/phase_d/baseline/
#         /scratch/$USER/artifacts/phase_d/constrained/
#         /scratch/$USER/artifacts/phase_d/phase_d_comparative_metrics.json

echo "Starting Phase D training at $(date)"

# Run Phase D with NUMA binding and HPC configuration
numactl --cpunodebind=${NUMA_NODE} --membind=${NUMA_NODE} \
    python -m neuro_stylometry run-phase-d \
        --dataset /scratch/$USER/artifacts/phase_a/clean_dataset.arrow \
        --output-dir /scratch/$USER/artifacts/phase_d \
        --artifacts-dir /scratch/$USER/artifacts/phase_a \
        --mode hpc

echo "Phase D training completed at $(date)"

# Optional: Run verification immediately after training
# Uncomment the following to run CHG + SVS verification automatically
# echo "Starting verification at $(date)"
# python -m neuro_stylometry verify \
#     --dataset /scratch/$USER/artifacts/phase_a/clean_dataset.arrow \
#     --phase-d-dir /scratch/$USER/artifacts/phase_d \
#     --artifacts-dir /scratch/$USER/artifacts/phase_a \
#     --output-dir /scratch/$USER/artifacts/phase_d/chg
# echo "Verification completed at $(date)"
