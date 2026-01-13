#!/bin/bash
# scripts/slurm/phase_d_full_pipeline.sbatch
# SLURM script for full Phase D pipeline (training + verification + report)

#SBATCH --job-name=neuro_stylometry_phase_d_full
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=256G
#SBATCH --time=16:00:00
#SBATCH --output=logs/phase_d_full_%j.out
#SBATCH --error=logs/phase_d_full_%j.err

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

# Define paths
DATASET=/scratch/$USER/artifacts/phase_a/clean_dataset.arrow
ARTIFACTS_DIR=/scratch/$USER/artifacts/phase_a
OUTPUT_DIR=/scratch/$USER/artifacts/phase_d
REPORTS_DIR=/scratch/$USER/artifacts/reports

echo "=========================================="
echo "Phase D Full Pipeline"
echo "=========================================="
echo "Start time: $(date)"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Step 1: Validate Phase A artifacts
echo ""
echo "[1/4] Validating Phase A artifacts..."
python -m neuro_stylometry validate-handover --artifacts-dir $ARTIFACTS_DIR
if [ $? -ne 0 ]; then
    echo "ERROR: Phase A validation failed!"
    exit 1
fi
echo "Phase A artifacts validated successfully"

# Step 2: Train baseline + constrained models
echo ""
echo "[2/4] Training baseline and constrained models..."
numactl --cpunodebind=${NUMA_NODE} --membind=${NUMA_NODE} \
    python -m neuro_stylometry run-phase-d \
        --dataset $DATASET \
        --output-dir $OUTPUT_DIR \
        --artifacts-dir $ARTIFACTS_DIR \
        --mode hpc
if [ $? -ne 0 ]; then
    echo "ERROR: Phase D training failed!"
    exit 1
fi
echo "Phase D training completed successfully"

# Step 3: Run CHG + SVS verification
echo ""
echo "[3/4] Running CHG + SVS verification..."
python -m neuro_stylometry verify \
    --dataset $DATASET \
    --phase-d-dir $OUTPUT_DIR \
    --artifacts-dir $ARTIFACTS_DIR \
    --output-dir $OUTPUT_DIR/chg
if [ $? -ne 0 ]; then
    echo "ERROR: Verification failed!"
    exit 1
fi
echo "Verification completed successfully"

# Step 4: Generate Phase D report
echo ""
echo "[4/4] Generating Phase D comparative report..."
python -m neuro_stylometry report-phase-d \
    --phase-d-dir $OUTPUT_DIR \
    --output-dir $REPORTS_DIR \
    --dataset $DATASET
if [ $? -ne 0 ]; then
    echo "ERROR: Report generation failed!"
    exit 1
fi
echo "Report generated successfully"

echo ""
echo "=========================================="
echo "Phase D Full Pipeline Complete"
echo "=========================================="
echo "End time: $(date)"
echo "Results available at:"
echo "  - Models: $OUTPUT_DIR"
echo "  - Report: $REPORTS_DIR/phase_d_report.html"
echo "=========================================="
