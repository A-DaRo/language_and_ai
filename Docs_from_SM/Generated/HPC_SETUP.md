# HPC Setup and Execution Guide

This guide provides instructions for running Phase D training on HPC environments with rented GPUs (e.g., cloud GPU providers, university HPC clusters).

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Phase D Configuration](#phase-d-configuration)
4. [Running Phase D](#running-phase-d)
5. [SLURM Job Submission](#slurm-job-submission)
6. [Non-SLURM Execution](#non-slurm-execution)
7. [Monitoring and Debugging](#monitoring-and-debugging)
8. [Expected Outputs](#expected-outputs)

## Prerequisites

### Required Artifacts from Phase A
Before running Phase D, ensure you have completed Phase A and have the following artifacts:
- `clean_dataset.arrow` - Cleaned dataset with masked text
- `projection_matrix.pt` - LEACE projection matrix
- `pollution_logs.arrow` - Pollution detection logs (optional, for validation)

Validate Phase A artifacts:
```bash
python -m neuro_stylometry validate-handover --artifacts-dir /path/to/phase_a_artifacts
```

### Hardware Requirements
**Minimum:**
- 1x GPU with 16GB VRAM (e.g., NVIDIA V100, A10)
- 32 CPU cores
- 128GB RAM
- 500GB storage

**Recommended:**
- 1x GPU with 40GB+ VRAM (e.g., NVIDIA A100)
- 64 CPU cores
- 256GB RAM
- 1TB storage

### Software Requirements
- Python 3.10+
- CUDA 12.1+
- PyTorch 2.0+
- All dependencies from `pyproject.toml`

## Environment Setup

### 1. Clone Repository
```bash
cd /scratch/$USER  # or your preferred work directory
git clone <repository-url> neuro_stylometry
cd neuro_stylometry
```

### 2. Create Virtual Environment
```bash
# Using venv
python3 -m venv venv
source venv/bin/activate

# Or using conda
conda create -n neuro_stylometry python=3.11
conda activate neuro_stylometry
```

### 3. Install Dependencies
```bash
pip install -e .
```

### 4. Verify Installation
```bash
python -m neuro_stylometry --help
python -m neuro_stylometry hardware-info
```

### 5. Set Up Directory Structure
```bash
mkdir -p /scratch/$USER/artifacts/phase_a
mkdir -p /scratch/$USER/artifacts/phase_d
mkdir -p /scratch/$USER/artifacts/reports
mkdir -p logs
```

### 6. Transfer Phase A Artifacts
```bash
# From your local machine or Phase A output location
scp -r phase_a_artifacts/* user@hpc:/scratch/$USER/artifacts/phase_a/
```

## Phase D Configuration

Phase D uses YAML configuration files with mode-specific overrides:
- `conf/base/phase_d.yaml` - Base configuration (defaults)
- `conf/laptop/phase_d.yaml` - Laptop mode (small-scale testing)
- `conf/hpc/phase_d.yaml` - HPC mode (production training)

### HPC Configuration (`conf/hpc/phase_d.yaml`)
```yaml
training:
  num_epochs: 10              # More epochs for HPC
  batch_size: 32              # Larger batches with more VRAM
  learning_rate: 3e-5
  layerwise_lr_decay: 0.9
  gradient_accumulation_steps: 2
  precision: "bf16"           # Mixed precision for speed

scheduler:
  name: "cosine"              # Cosine annealing for better convergence
  num_warmup_steps: 500

execution:
  autotuning:
    enabled: true             # Automatically tune batch sizes
    initial_token_budget: 65536
    max_token_budget: 393216
  dynamic_batching:
    enabled: true
    max_batch_size: 128
  telemetry:
    enabled: true             # Log performance metrics
```

### Custom Experiment Configuration (Optional)
Create `conf/experiments/my_experiment.yaml` to override specific settings:
```yaml
training:
  num_epochs: 15
  batch_size: 64
  learning_rate: 5e-5

data:
  split_ratios:
    train: 0.7
    val: 0.15
    test: 0.15
```

## Running Phase D

### Method 1: CLI Direct Execution (Interactive)
```bash
# Allocate GPU node interactively
salloc --partition=gpu --gpus=1 --cpus-per-task=32 --mem=256G --time=12:00:00

# Run Phase D
python -m neuro_stylometry run-phase-d \
  --dataset /scratch/$USER/artifacts/phase_a/clean_dataset.arrow \
  --output-dir /scratch/$USER/artifacts/phase_d \
  --artifacts-dir /scratch/$USER/artifacts/phase_a \
  --mode hpc

# Optional: Run verification
python -m neuro_stylometry verify \
  --dataset /scratch/$USER/artifacts/phase_a/clean_dataset.arrow \
  --phase-d-dir /scratch/$USER/artifacts/phase_d \
  --artifacts-dir /scratch/$USER/artifacts/phase_a \
  --output-dir /scratch/$USER/artifacts/phase_d/chg

# Optional: Generate report
python -m neuro_stylometry report-phase-d \
  --phase-d-dir /scratch/$USER/artifacts/phase_d \
  --output-dir /scratch/$USER/artifacts/reports \
  --dataset /scratch/$USER/artifacts/phase_a/clean_dataset.arrow
```

### Method 2: SLURM Batch Submission (Recommended)

#### Full Pipeline (Training + Verification + Report)
```bash
sbatch scripts/slurm/phase_d_full_pipeline.sbatch
```

This runs all steps sequentially:
1. Validates Phase A artifacts
2. Trains baseline and constrained models
3. Runs CHG + SVS verification
4. Generates HTML report

#### Individual Steps

**Training only:**
```bash
sbatch scripts/slurm/phase_d.sbatch
```

**Verification only (after training):**
```bash
sbatch scripts/slurm/verify.sbatch
```

### Method 3: Custom Experiment Config
```bash
python -m neuro_stylometry run-phase-d \
  --dataset /scratch/$USER/artifacts/phase_a/clean_dataset.arrow \
  --output-dir /scratch/$USER/artifacts/phase_d_exp1 \
  --artifacts-dir /scratch/$USER/artifacts/phase_a \
  --mode hpc \
  --config conf/experiments/my_experiment.yaml
```

## SLURM Job Submission

### Editing SLURM Scripts

Before submitting, update the SLURM scripts to match your HPC environment:

1. **Module loading** (lines 16-19):
   ```bash
   module load python/3.11
   module load cuda/12.1
   module load cudnn/8.9.7
   ```

2. **Virtual environment path** (line 22):
   ```bash
   source /home/$USER/venvs/neuro_stylometry/bin/activate
   ```

3. **Partition name** (line 6):
   ```bash
   #SBATCH --partition=gpu  # Change to your GPU partition name
   ```

### Job Monitoring
```bash
# Check job status
squeue -u $USER

# View job output (while running)
tail -f logs/phase_d_full_<job_id>.out

# Cancel job
scancel <job_id>

# View completed job details
sacct -j <job_id> --format=JobID,JobName,Elapsed,State,ExitCode
```

### Resource Adjustments

For smaller models or datasets, reduce resources in SLURM scripts:
```bash
#SBATCH --cpus-per-task=16  # Instead of 32
#SBATCH --mem=128G          # Instead of 256G
#SBATCH --time=06:00:00     # Instead of 12:00:00
```

## Non-SLURM Execution

For cloud GPU providers without SLURM (e.g., Lambda Labs, RunPod, Vast.ai):

### 1. SSH into GPU Instance
```bash
ssh user@gpu-instance
```

### 2. Run Full Pipeline Script
```bash
#!/bin/bash
# run_phase_d.sh

export CUDA_VISIBLE_DEVICES=0

DATASET=/workspace/artifacts/phase_a/clean_dataset.arrow
ARTIFACTS_DIR=/workspace/artifacts/phase_a
OUTPUT_DIR=/workspace/artifacts/phase_d
REPORTS_DIR=/workspace/artifacts/reports

# Validate Phase A
python -m neuro_stylometry validate-handover --artifacts-dir $ARTIFACTS_DIR

# Train models
python -m neuro_stylometry run-phase-d \
  --dataset $DATASET \
  --output-dir $OUTPUT_DIR \
  --artifacts-dir $ARTIFACTS_DIR \
  --mode hpc

# Verify models
python -m neuro_stylometry verify \
  --dataset $DATASET \
  --phase-d-dir $OUTPUT_DIR \
  --artifacts-dir $ARTIFACTS_DIR \
  --output-dir $OUTPUT_DIR/chg

# Generate report
python -m neuro_stylometry report-phase-d \
  --phase-d-dir $OUTPUT_DIR \
  --output-dir $REPORTS_DIR \
  --dataset $DATASET

echo "Complete! Results at $OUTPUT_DIR"
```

### 3. Run in Background with nohup
```bash
chmod +x run_phase_d.sh
nohup ./run_phase_d.sh > phase_d.log 2>&1 &

# Monitor progress
tail -f phase_d.log
```

## Monitoring and Debugging

### Real-time GPU Monitoring
```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1
```

### Check Training Progress
```bash
# View training logs (JSONL format)
tail -f /scratch/$USER/artifacts/phase_d/baseline/training_log.jsonl
tail -f /scratch/$USER/artifacts/phase_d/constrained/training_log.jsonl

# Pretty-print latest epoch
tail -1 /scratch/$USER/artifacts/phase_d/baseline/training_log.jsonl | python -m json.tool
```

### Common Issues

#### 1. Out of Memory (OOM)
**Solution:** Reduce batch size in HPC config
```yaml
training:
  batch_size: 16  # Reduce from 32
  gradient_accumulation_steps: 4  # Increase to maintain effective batch size
```

#### 2. CUDA Version Mismatch
**Solution:** Check CUDA compatibility
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
nvidia-smi
```

#### 3. Missing Phase A Artifacts
**Solution:** Validate and re-run Phase A if needed
```bash
python -m neuro_stylometry validate-handover --artifacts-dir /path/to/phase_a
```

#### 4. Slow Training
**Solution:** Enable autotuning and check GPU utilization
- Verify `execution.autotuning.enabled: true` in HPC config
- Check GPU utilization with `nvidia-smi` (should be >80%)
- Ensure NUMA pinning is enabled (see SLURM scripts)

## Expected Outputs

### Phase D Training Output
```
/scratch/$USER/artifacts/phase_d/
├── baseline/
│   ├── model.pt                          # Baseline transformer weights
│   ├── head.pt                           # Classification head weights
│   ├── checkpoint.pt                     # Full checkpoint (model + head + optimizer)
│   ├── training_log.jsonl                # Per-batch training logs
│   ├── phase_d_metrics.json              # Final metrics summary
│   ├── test_metrics.json                 # Test set evaluation
│   └── test_details.json                 # Per-class test metrics
├── constrained/
│   ├── model.pt                          # Constrained transformer (with Affine Guard)
│   ├── head.pt
│   ├── checkpoint.pt
│   ├── training_log.jsonl
│   ├── phase_d_metrics.json
│   ├── test_metrics.json
│   └── test_details.json
└── phase_d_comparative_metrics.json      # Baseline vs Constrained comparison
```

### Verification Output
```
/scratch/$USER/artifacts/phase_d/chg/
├── chg_gates_baseline.pt                 # Learned attention head gates (baseline)
├── chg_gates_constrained.pt              # Learned attention head gates (constrained)
├── head_classification_baseline.json     # Head type classification (baseline)
├── head_classification_constrained.json  # Head type classification (constrained)
├── svs_baseline.json                     # Stylometric Validity Score (baseline)
├── svs_constrained.json                  # Stylometric Validity Score (constrained)
└── verification_summary.json             # Comparative verification summary
```

### Report Output
```
/scratch/$USER/artifacts/reports/
├── phase_d_report.html                   # Interactive HTML report
└── phase_d_assets/                       # Report images and plots
    ├── loss_baseline.png
    ├── loss_constrained.png
    ├── accuracy_comparison.png
    ├── confusion_baseline_*.png
    └── confusion_constrained_*.png
```

### Typical Training Time

| Dataset Size | GPU | Epochs | Training Time | Verification Time |
|-------------|-----|--------|---------------|-------------------|
| 1K samples  | V100 | 10 | ~30 min | ~10 min |
| 10K samples | V100 | 10 | ~3 hours | ~30 min |
| 100K samples | A100 | 10 | ~12 hours | ~2 hours |

## Performance Optimization Tips

1. **Use mixed precision training** (`precision: "bf16"` in HPC config)
2. **Enable autotuning** for automatic batch size optimization
3. **Use NUMA pinning** (enabled in SLURM scripts)
4. **Increase gradient accumulation** to simulate larger batches
5. **Monitor GPU utilization** - should be >80% during training
6. **Use cosine scheduler** for better convergence
7. **Enable telemetry** to track performance metrics

## Transferring Results Back to Local Machine

```bash
# Download trained models
scp -r user@hpc:/scratch/$USER/artifacts/phase_d ./local_artifacts/

# Download report
scp -r user@hpc:/scratch/$USER/artifacts/reports ./local_reports/

# View report locally
open local_reports/phase_d_report.html
```

## Support and Troubleshooting

For issues specific to your HPC environment:
1. Check HPC documentation for module names and partition names
2. Contact HPC support for CUDA/GPU issues
3. Verify PyTorch CUDA compatibility: `python -c "import torch; print(torch.cuda.is_available())"`

For Phase D pipeline issues:
1. Run with `--verbose` flag for detailed logging
2. Check training logs in `artifacts/phase_d/*/training_log.jsonl`
3. Validate Phase A artifacts before starting
4. Review `HPC_SETUP.md` (this file) for common issues
