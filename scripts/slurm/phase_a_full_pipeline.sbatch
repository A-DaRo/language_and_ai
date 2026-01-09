#!/bin/bash
# scripts/slurm/phase_a_full_pipeline.sbatch
# End-to-end Phase A run (HPC): run pipeline + validate handover artifacts.

#SBATCH --job-name=neuro_stylometry_phase_a_full
#SBATCH --partition=tue.gpu2.q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/phase_a_full_%j.out
#SBATCH --error=logs/phase_a_full_%j.err

set -euo pipefail

# Load modules (adjust to your cluster)
module load python/3.11
module load cuda/12.1
module load cudnn/8.9.7

# Activate virtual environment (adjust if different)
source /home/$USER/venvs/langai/bin/activate

# Resolve repo root from script location
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "${REPO_ROOT}/logs"

# Inputs/outputs (override via env vars if desired)
DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/artifacts/data/sobr.arrow}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifacts/phase_a}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/conf/experiments/my_experiment.yaml}"

# GPU pinning (optional)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Build optional config arg if file exists
CONFIG_ARG=()
if [[ -f "${CONFIG_PATH}" ]]; then
  CONFIG_ARG=(--config "${CONFIG_PATH}")
fi

echo "Repo root: ${REPO_ROOT}"
echo "Dataset:   ${DATASET_PATH}"
echo "Output:    ${OUTPUT_DIR}"
if [[ -f "${CONFIG_PATH}" ]]; then
  echo "Config:    ${CONFIG_PATH}"
else
  echo "Config:    (none)"
fi

# Run Phase A via CLI (HPC strategy)
python -m neuro_stylometry run-phase-a \
  --dataset "${DATASET_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --mode hpc \
  "${CONFIG_ARG[@]}"

# Validate handover artifacts
python scripts/validate_phase_a_handover.py "${OUTPUT_DIR}"

echo "Phase A full pipeline completed at $(date)"
