# Phase D Quick Start Guide

This guide provides quick commands to run Phase D training on both laptop and HPC environments.

## Prerequisites

Ensure Phase A is complete and you have:
- `clean_dataset.arrow`
- `projection_matrix.pt`

Validate Phase A artifacts:
```bash
python -m neuro_stylometry validate-handover --artifacts-dir artifacts/data/output
```

## Laptop Mode (Local Testing)

### Full Pipeline (Training + Verification + Report)
```bash
# Step 1: Train baseline and constrained models (2 epochs, small batch)
python -m neuro_stylometry run-phase-d \
  --dataset artifacts/data/output/clean_dataset.arrow \
  --output-dir artifacts/phase_d \
  --artifacts-dir artifacts/data/output \
  --mode laptop

# Step 2: Run CHG + SVS verification
python -m neuro_stylometry verify \
  --dataset artifacts/data/output/clean_dataset.arrow \
  --phase-d-dir artifacts/phase_d \
  --artifacts-dir artifacts/data/output \
  --output-dir artifacts/phase_d/chg

# Step 3: Generate HTML report
python -m neuro_stylometry report-phase-d \
  --phase-d-dir artifacts/phase_d \
  --output-dir artifacts/reports \
  --dataset artifacts/data/output/clean_dataset.arrow

# View report
open artifacts/reports/phase_d_report.html
```

### Laptop Configuration
- **Epochs:** 2
- **Batch size:** 4
- **Precision:** fp32
- **Time:** ~10-30 minutes (depending on dataset size)

## HPC Mode (Production Training)

### Option 1: SLURM Full Pipeline (Recommended)
```bash
# Submit job that runs everything (training + verification + report)
sbatch scripts/slurm/phase_d_full_pipeline.sbatch

# Monitor job
squeue -u $USER
tail -f logs/phase_d_full_*.out
```

### Option 2: SLURM Individual Steps
```bash
# Train models only
sbatch scripts/slurm/phase_d.sbatch

# After training completes, run verification
sbatch scripts/slurm/verify.sbatch
```

### Option 3: Interactive HPC Session
```bash
# Allocate GPU node
salloc --partition=gpu --gpus=1 --cpus-per-task=32 --mem=256G --time=12:00:00

# Run Phase D
python -m neuro_stylometry run-phase-d \
  --dataset /scratch/$USER/artifacts/phase_a/clean_dataset.arrow \
  --output-dir /scratch/$USER/artifacts/phase_d \
  --artifacts-dir /scratch/$USER/artifacts/phase_a \
  --mode hpc
```

### Option 4: Cloud GPU (No SLURM)
```bash
# SSH into GPU instance
ssh user@gpu-instance

# Set GPU
export CUDA_VISIBLE_DEVICES=0

# Run full pipeline
python -m neuro_stylometry run-phase-d \
  --dataset artifacts/phase_a/clean_dataset.arrow \
  --output-dir artifacts/phase_d \
  --artifacts-dir artifacts/phase_a \
  --mode hpc

python -m neuro_stylometry verify \
  --dataset artifacts/phase_a/clean_dataset.arrow \
  --phase-d-dir artifacts/phase_d \
  --artifacts-dir artifacts/phase_a \
  --output-dir artifacts/phase_d/chg

python -m neuro_stylometry report-phase-d \
  --phase-d-dir artifacts/phase_d \
  --output-dir artifacts/reports \
  --dataset artifacts/phase_a/clean_dataset.arrow
```

### HPC Configuration
- **Epochs:** 10
- **Batch size:** 32 (with 2x gradient accumulation = effective 64)
- **Precision:** bf16 (mixed precision)
- **Scheduler:** Cosine with 500 warmup steps
- **Autotuning:** Enabled (automatically adjusts batch size)
- **Dynamic batching:** Enabled
- **Time:** ~4-12 hours (depending on dataset size and GPU)

## Custom Experiment Configuration

Create `conf/experiments/my_experiment.yaml`:
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

Run with custom config:
```bash
python -m neuro_stylometry run-phase-d \
  --dataset artifacts/data/output/clean_dataset.arrow \
  --output-dir artifacts/phase_d_custom \
  --artifacts-dir artifacts/data/output \
  --mode hpc \
  --config conf/experiments/my_experiment.yaml
```

## Output Structure

```
artifacts/phase_d/
├── baseline/                              # Baseline model (no masking, no Affine Guard)
│   ├── model.pt                          # Transformer weights (500MB)
│   ├── head.pt                           # Classification head weights
│   ├── checkpoint.pt                     # Full checkpoint for resuming (1.5GB)
│   ├── training_log.jsonl                # Per-batch training logs
│   ├── phase_d_metrics.json              # Final metrics summary
│   ├── test_metrics.json                 # Test set evaluation
│   └── test_details.json                 # Per-class test metrics + confusion matrices
├── constrained/                           # Constrained model (masked text + Affine Guard)
│   └── [same structure as baseline]
├── phase_d_comparative_metrics.json       # Baseline vs Constrained comparison
└── chg/                                   # Verification outputs
    ├── chg_gates_baseline.pt             # Attention head gates
    ├── chg_gates_constrained.pt
    ├── head_classification_baseline.json  # Head type classification
    ├── head_classification_constrained.json
    ├── svs_baseline.json                 # Stylometric Validity Score
    ├── svs_constrained.json
    └── verification_summary.json          # Comparative summary

artifacts/reports/
├── phase_d_report.html                    # Interactive HTML report
└── phase_d_assets/                        # Report plots
    ├── loss_baseline.png
    ├── loss_constrained.png
    ├── accuracy_comparison.png
    └── confusion_*.png
```

## Monitoring Training

### View Training Progress
```bash
# Live training logs
tail -f artifacts/phase_d/baseline/training_log.jsonl

# Pretty-print latest epoch
tail -1 artifacts/phase_d/baseline/training_log.jsonl | python -m json.tool

# GPU utilization (should be >80%)
watch -n 1 nvidia-smi
```

### Check Metrics
```bash
# View test metrics
cat artifacts/phase_d/baseline/test_metrics.json | python -m json.tool

# View comparative summary
cat artifacts/phase_d/phase_d_comparative_metrics.json | python -m json.tool
```

## Common Commands

### View Configuration
```bash
# Show merged configuration for a mode
python -m neuro_stylometry config-show --mode laptop
python -m neuro_stylometry config-show --mode hpc
```

### Check Hardware
```bash
# Display detected hardware profile
python -m neuro_stylometry hardware-info
```

### Resume Training
```bash
# Training automatically creates checkpoints
# To resume from checkpoint, the trainer will detect and load checkpoint.pt automatically
# Manual resumption is built into the training loop
```

## Troubleshooting

### Out of Memory
Reduce batch size in config:
```yaml
training:
  batch_size: 16  # Reduce from 32
  gradient_accumulation_steps: 4  # Increase to maintain effective batch size
```

### Slow Training
- Enable autotuning: `execution.autotuning.enabled: true`
- Check GPU utilization: `nvidia-smi` (should be >80%)
- Use mixed precision: `training.precision: "bf16"`

### Missing Artifacts
Validate Phase A first:
```bash
python -m neuro_stylometry validate-handover --artifacts-dir artifacts/data/output
```

## Expected Training Time

| Dataset Size | Mode | GPU | Time (Training) | Time (Verification) |
|-------------|------|-----|-----------------|---------------------|
| 100 samples | Laptop | CPU/GPU | 5-10 min | 2-5 min |
| 1K samples | Laptop | GPU | 20-40 min | 5-10 min |
| 1K samples | HPC | V100 | 30-60 min | 10-20 min |
| 10K samples | HPC | V100 | 3-6 hours | 30-60 min |
| 100K samples | HPC | A100 | 8-16 hours | 2-4 hours |

## Tips for HPC Execution

1. **Test locally first:** Always run in laptop mode on a small subset to verify the pipeline works
2. **Use SLURM scripts:** They handle resource allocation and NUMA pinning automatically
3. **Monitor GPU utilization:** Should be >80% during training
4. **Enable autotuning:** Automatically finds optimal batch sizes
5. **Use mixed precision:** `bf16` provides 2-3x speedup with minimal accuracy loss
6. **Check logs regularly:** Monitor `logs/phase_d_*.out` for errors
7. **Validate Phase A:** Always validate artifacts before training

## Next Steps

After Phase D completes:
1. View the HTML report at `artifacts/reports/phase_d_report.html`
2. Examine comparative metrics in `phase_d_comparative_metrics.json`
3. Review verification results in `artifacts/phase_d/chg/`
4. Compare baseline vs constrained model performance
5. Analyze attention head classifications and SVS scores

For detailed setup and troubleshooting, see [HPC_SETUP.md](HPC_SETUP.md).
