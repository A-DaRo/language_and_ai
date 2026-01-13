# Phase D Status Report (Implementation Snapshot)

**Document Type:** Status + Gap Analysis
**Scope:** Phase D (Neural Stylometry)
**Context:** Phase A artifacts available under `artifacts/phase_a/`
**Last Updated:** 2026-01-12 (Updated with evaluation + CHG/SVS improvements + HPC config)  

---

## 1. Executive Summary

Phase D is now fully implemented for Track A (training pipeline) with configuration-driven architecture. The core model wrapper, Affine Guard injection, tokenizer alignment, dataset loading, baseline vs constrained training loop, checkpointing, metrics reporting, and CHG/SVS verification entry points are in place. The pipeline now uses YAML-based configuration with mode-specific overrides (laptop/HPC), providing a clean separation between configuration and execution. Evaluation/reporting is now implemented (report + plots + per-class tables + confusion matrices), and CHG/SVS have been upgraded for rigor (head gating + POS-aware SVS + query aggregation). Remaining work is primarily in expanded tests and minor evaluation polish.

---

## 2. Implemented in This Update (Chat Session Summary)

### 2.0 Configuration System (NEW - January 2026)

**Files:**
- `conf/base/phase_d.yaml` - Base Phase D configuration
- `conf/laptop/phase_d.yaml` - Laptop-specific overrides
- `src/neuro_stylometry/config.py` - Configuration loading with `load_phase_d_config()`
- `src/neuro_stylometry/phase_d_pipeline.py` - Training orchestrator

**What it does:**
- YAML-based configuration with mode-specific overrides (base → laptop/hpc → experiment)
- Orchestrates training of both baseline and constrained models
- Provides comparative metrics and summary reporting
- Supports split column detection with fallback to author-stratified split
- Configurable training parameters (epochs, batch size, learning rate, checkpointing)

**Configuration Merge Order:**
1. `conf/base/phase_d.yaml` - Base defaults
2. `conf/{mode}/phase_d.yaml` - Mode-specific (laptop/hpc)
3. Optional experiment config - User overrides

**Why it matters:**
Provides clean separation between configuration and execution, making it easy to switch between laptop and HPC modes, run experiments with different hyperparameters, and maintain reproducible training configurations.

**Artifacts per pipeline run:**
- `baseline/` directory with baseline model artifacts
- `constrained/` directory with constrained model artifacts
- `phase_d_comparative_metrics.json` - Comparative summary with accuracy deltas

---

### 2.1 Tokenizer Alignment (Phase A Compatibility)

**Files:**
- `src/neuro_stylometry/stylometry_net/tokenizer.py`

**What it does:**
- Loads mask tokens from `conf/base/gliner_taxonomy.yaml`
- Registers typed masks as special tokens (single-token enforcement)
- Exposes `encode_batch()` and `resize_model_embeddings()`

**Why it matters:**
Ensures `[MASK:*]` tokens from Phase A remain single tokens in Phase D.

---

### 2.2 Affine Guard + Transformer Wrapper

**Files:**
- `src/neuro_stylometry/stylometry_net/affine_guard.py`
- `src/neuro_stylometry/stylometry_net/transformer.py`

**What it does:**
- Loads `projection_matrix.pt` from Phase A
- Injects Affine Guard after embeddings
- Uses Phase D tokenizer wrapper for mask alignment
- Adds `from_phase_a(artifacts_dir=...)` constructor

---

### 2.3 Phase D Dataset Loading + Collation

**Files:**
- `src/neuro_stylometry/stylometry_net/phase_d_dataset.py`

**What it does:**
- Loads Arrow dataset (`clean_dataset.arrow`)
- Supports `post` (baseline) and `post_masked` (constrained)
- Builds label maps per demographic field
- Handles missing labels as `-1`
- Supports split column; falls back to author‑stratified split (80/10/10) if missing

---

### 2.4 Baseline vs Constrained Training Loop

**Files:**
- `src/neuro_stylometry/training/trainer.py`
- `src/neuro_stylometry/stylometry_net/classification_head.py`

**What it does:**
- Trains baseline model on `post`
- Trains constrained model on `post_masked` with Affine Guard
- Multi‑task head with masked loss for missing labels
- Checkpointing + metadata + JSONL logging
- Per‑task accuracy and macro‑F1 reported to JSON
- Progress bar during training

**Artifacts per run:**
- `model.pt`, `head.pt`
- `training_log.jsonl`
- `phase_d_metrics.json`
- `checkpoint.pt`
- optional cadence checkpoints under `checkpoints/`

---

### 2.5 CLI Wiring (UPDATED - January 2026)

**Files:**
- `src/neuro_stylometry/__main__.py`

**Run Phase D (New Config-Driven Approach):**
```bash
python -m neuro_stylometry run-phase-d \
  --dataset artifacts/data/output/clean_dataset.arrow \
  --output-dir artifacts/phase_d \
  --artifacts-dir artifacts/data/output \
  --mode laptop
```

**Optional experiment config override:**
```bash
python -m neuro_stylometry run-phase-d \
  --dataset artifacts/data/output/clean_dataset.arrow \
  --output-dir artifacts/phase_d \
  --artifacts-dir artifacts/data/output \
  --mode laptop \
  --config experiments/custom_phase_d.yaml
```

**Verification:**
```bash
python -m neuro_stylometry verify \
  --dataset artifacts/phase_a/clean_dataset.arrow \
  --phase-d-dir artifacts/phase_d \
  --output-dir artifacts/phase_d/chg
```

**Key Changes:**
- Simplified CLI with mode-based configuration loading
- All hyperparameters now controlled via YAML configuration files
- Removed individual parameter flags in favor of clean config files
- CLI paths now point to actual Phase A output location (`artifacts/data/output/`)

---

### 2.6 CHG + SVS Verification

**Files:**
- `src/neuro_stylometry/stylometry_net/chg_verifier.py`
- `src/neuro_stylometry/stylometry_net/svs_calculator.py`
- `src/neuro_stylometry/stylometry_net/verification.py`

**What it does:**
- CHG learns gate parameters per head using attention head masks during forward pass (L1 regularization)
- SVS computes function‑word vs content‑word attention mass ratio with optional POS tagging (spaCy) and lexical fallback
- CLI `verify` runs baseline + constrained verification and writes artifacts

**Artifacts:**
- `chg_gates_{baseline,constrained}.pt`
- `head_classification_{baseline,constrained}.json`
- `svs_{baseline,constrained}.json`

---

## 3. Current State of Phase D (As‑Implemented)

### 3.1 Implemented Components

| Area | Status | Notes |
|------|--------|-------|
| **Configuration System** | ✅ **NEW** | **YAML-based config with mode overrides (laptop/HPC)** |
| **Pipeline Orchestrator** | ✅ **NEW** | **Trains both baseline + constrained with comparative metrics** |
| Tokenizer alignment | ✅ | Mask tokens from taxonomy, enforced single-token |
| Affine Guard | ✅ | Projection matrix buffer, frozen by default |
| Transformer wrapper | ✅ | Injects guard after embeddings |
| Dataset loader | ✅ | Arrow read + split logic + label maps |
| Training loop | ✅ | Baseline + constrained, logs, checkpoints |
| Metrics | ✅ | Per-task accuracy and macro‑F1 |
| CHG verifier | ✅ | Head gating via attention masks |
| SVS | ✅ | POS‑aware (spaCy) + lexical fallback + query aggregation |
| CLI wiring | ✅ **UPDATED** | **Config-driven run-phase-d + verify** |
| Evaluation/reporting | ✅ **NEW** | **Plots + HTML report + report CLI** |
| HPC config | ✅ **NEW** | **conf/hpc/phase_d.yaml** |

### 3.2 Known Limitations / Improvements

1. **Test set evaluation could be deeper**
   - Confusion matrices + per-class tables exist.
   - Improvement: add per-attribute breakdowns and calibration curves.

2. **Report coverage could be expanded**
   - Per-class tables + confusion matrices exist, but no per-attribute breakdowns.

---

## 4. Remaining Phase D Work (Spec‑Aligned)

### 4.1 Evaluation + Reporting (Completed)

**Delivered files:**
- `src/neuro_stylometry/evaluation/attention_analysis.py`
- `src/neuro_stylometry/evaluation/visualizations.py`
- `src/neuro_stylometry/evaluation/comparative_report.py`

**Delivered artifacts:**
- Baseline vs constrained plots (learning curves, accuracy bars)
- Confusion matrices and per-class tables
- Report HTML under `artifacts/reports/phase_d_report.html`
- Aggregated `phase_d_metrics.json` at `artifacts/phase_d/`

---

### 4.2 Verification Enhancements (In Progress)

**Needed improvements:**
- Comparative summary output (delta SVS, delta accuracy) beyond the current JSON summary.

---

### 4.3 Training Enhancements

**Recommended upgrades:**
- Mixed precision + GradScaler ✅
- Learning‑rate scheduler (linear warmup + cosine decay) ✅
- Gradient accumulation ✅
- Checkpoint resumption ✅

---

### 4.4 Tests

**Missing tests:**
- Unit tests for tokenizer alignment (real-model already exists)
- Unit tests for Affine Guard correctness
- Integration test for baseline + constrained training (tiny dataset)
- Golden tests for CHG/SVS outputs

---

## 5. Suggested Next Implementation Order

1. **Test coverage**
2. **Config schema finalization (last per instruction)**

---

## 6. Quick Reference (Current CLI Usage)

**Train baseline + constrained (laptop mode):**
```bash
python -m neuro_stylometry run-phase-d \
  --dataset artifacts/phase_a/clean_dataset.arrow \
  --output-dir artifacts/phase_d \
  --artifacts-dir artifacts/phase_a \
  --mode laptop
```

**Train with custom experiment config:**
```bash
python -m neuro_stylometry run-phase-d \
  --dataset artifacts/phase_a/clean_dataset.arrow \
  --output-dir artifacts/phase_d \
  --artifacts-dir artifacts/phase_a \
  --mode laptop \
  --config experiments/my_config.yaml
```

**Expected outputs:**
- `artifacts/phase_d/baseline/` - Baseline model artifacts (model.pt, head.pt, training_log.jsonl, test_metrics.json)
- `artifacts/phase_d/constrained/` - Constrained model artifacts (same structure)
- `artifacts/phase_d/phase_d_comparative_metrics.json` - Comparative summary with accuracy deltas

**Verify CHG + SVS:**
```bash
python -m neuro_stylometry verify \
  --dataset artifacts/phase_a/clean_dataset.arrow \
  --phase-d-dir artifacts/phase_d \
  --output-dir artifacts/phase_d/chg
```

**Report generation (from existing outputs):**
```bash
python -m neuro_stylometry report-phase-d \
  --phase-d-dir artifacts/phase_d \
  --output-dir artifacts/reports \
  --dataset artifacts/phase_a/clean_dataset.arrow
```

---

## 7. Configuration Files Reference

**Base Configuration:** `conf/base/phase_d.yaml`
- Defines all training parameters with sensible defaults
- Includes model configuration (roberta-base, max_length=512)
- Specifies demographic label fields to predict
- Sets evaluation and output options

**Laptop Configuration:** `conf/laptop/phase_d.yaml`
- Overrides for resource-constrained environments
- Reduced epochs (2 instead of 3)
- Smaller batch size (4 instead of 8)

**HPC Configuration:** `conf/hpc/phase_d.yaml` (DONE)
- Larger batch sizes + accumulation
- Mixed precision training
- Cosine scheduler + warmup

**Creating Custom Experiment Configs:**
```yaml
# experiments/my_experiment.yaml
training:
  num_epochs: 5
  batch_size: 16
  learning_rate: 3e-5

data:
  split_ratios:
    train: 0.7
    val: 0.15
    test: 0.15
```

---

*End of report.*

---

## 8. Phase D HPC Run Order (Recommended)

**Prerequisites:**
- `artifacts/phase_a/clean_dataset.arrow`
- `artifacts/phase_a/projection_matrix.pt`
- `artifacts/phase_a/pollution_logs.arrow`
- `conf/base/phase_d.yaml`
- `conf/hpc/phase_d.yaml`

**Order of execution:**
1. Run Phase A and validate artifacts:
   ```bash
   python -m neuro_stylometry run-phase-a \
     --dataset artifacts/data/sobr.arrow \
     --output-dir artifacts/phase_a \
     --mode hpc

   python -m neuro_stylometry validate-handover \
     --artifacts-dir artifacts/phase_a
   ```
2. Run Phase D using the HPC config:
   ```bash
   python -m neuro_stylometry run-phase-d \
     --dataset artifacts/phase_a/clean_dataset.arrow \
     --output-dir artifacts/phase_d \
     --artifacts-dir artifacts/phase_a \
     --mode hpc
   ```
3. Run verification (CHG + SVS):
   ```bash
   python -m neuro_stylometry verify \
     --dataset artifacts/phase_a/clean_dataset.arrow \
     --phase-d-dir artifacts/phase_d \
     --artifacts-dir artifacts/phase_a \
     --output-dir artifacts/phase_d/chg \
     --svs-query-strategy mean_tokens
   ```
4. Generate report:
   ```bash
   python -m neuro_stylometry report-phase-d \
     --phase-d-dir artifacts/phase_d \
     --output-dir artifacts/reports \
     --dataset artifacts/phase_a/clean_dataset.arrow
   ```
