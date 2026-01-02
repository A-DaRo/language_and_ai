#!/bin/bash
# scripts/slurm/phase_a.sbatch
# SLURM script for Phase A

#SBATCH --job-name=neuro_stylometry_phase_a
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/phase_a_%j.out
#SBATCH --error=logs/phase_a_%j.err

# Load modules
module load python/3.11
module load cuda/12.1
module load cudnn/8.9.7

# Activate virtual environment
source /home/$USER/venvs/neuro_stylometry/bin/activate

# NUMA pinning: bind to the NUMA node with the GPU
NUMA_NODE=$(nvidia-smi topo -m | grep GPU0 | awk '{print $NF}' | head -1)
export CUDA_VISIBLE_DEVICES=0

# Run Phase A with NUMA binding
numactl --cpunodebind=${NUMA_NODE} --membind=${NUMA_NODE} \
    python -m neuro_stylometry run-phase-a \
        --dataset /scratch/$USER/data/sobr_unified.arrow \
        --output-dir /scratch/$USER/artifacts/phase_a \
        --config conf/experiments/my_experiment.yaml \
        --mode hpc

echo "Phase A completed at $(date)"
