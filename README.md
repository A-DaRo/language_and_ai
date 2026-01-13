# Neuro-Symbolic Stylometry Pipeline

A dual-mode (Laptop/HPC) implementation of a pollution-aware stylometry pipeline combining symbolic span detection with geometric projection and constrained Transformer training.

## Features

- **Phase A: Pollution Guard** - GLiNER-based span detection + LEACE geometric projection
- **Phase D: Constrained Transformer** - Affine Guard layer for stylometric constraint enforcement
- **Dual-Mode Execution** - Automatic adaptation between laptop debug and HPC production environments
- **Apache Arrow Backend** - Memory-efficient data processing with zero-copy reads

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

For HPC features (cuml):
```bash
pip install -e ".[hpc]" --extra-index-url=https://pypi.nvidia.com
```

To install all (BEWARE cuml is Linux only)
```bash
pip install -e ".[all]" --extra-index-url=https://pypi.nvidia.com
```

**Remark**
FP8 precision requires `TransformerEngine`, to install:
```bash
pip install --no-build-isolation transformer_engine[pytorch]
```

## To begin

1. Make sure to have all SOBR csv files in `datasets/`

2. Create unified arrow dataset

```bash
# Convert raw data to Arrow format
python scripts/convert_pandas_to_arrow.py
```

3. Create laptop partition

```bash
# Create laptop arrow partition from unified table
python scripts/create_laptop_dataset.py
```

## How to use

### The `run_phase_a` command

### The `run_phase_d` command

# End-to-End Phase D Pipeline Analysis

## Command Breakdown

```bash
python -m neuro_stylometry run-phase-d \
    --dataset artifacts/phase_d/tokenized_dataset.arrow \
    --output-dir artifacts/phase_d_test \
    --artifacts-dir artifacts/phase_a \
    --mode hpc \
    --config conf/experiments/quick_test.yaml
```

### CLI Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--dataset` | tokenized_dataset.arrow | Pre-tokenized Arrow dataset with `post`, `post_masked`, labels |
| `--output-dir` | phase_d_test | Output directory for trained models and metrics |
| `--artifacts-dir` | phase_a | Phase A outputs (contains `projection_matrix.pt` for Affine Guard) |
| `--mode` | `hpc` | Uses phase_d.yaml settings (FP8, large batches, AOT compilation) |
| `--config` | quick_test.yaml | Experiment overrides merged on top of base+hpc configs |

---

## Pipeline Execution Flow

### 1. CLI Entry Point (`__main__.py::run_phase_d`)

```
┌─────────────────────────────────────────────────────────────┐
│  1. Check if --dataset exists                                │
│     ├── If missing + --preprocess enabled → auto-tokenize   │
│     └── If missing + --no-preprocess → raise ClickException │
│  2. Load config: base → hpc → experiment overlay            │
│  3. Call run_phase_d_training(...)                          │
└─────────────────────────────────────────────────────────────┘
```

Since `--preprocess` defaults to `True`, if the dataset doesn't exist, the CLI will:
1. Look for `clean_dataset.arrow` in phase_a
2. Tokenize using the model specified in config (default: `roberta-base`)
3. Save to `--dataset` path

### 2. Configuration Loading

The config is layered as:
```
conf/base/phase_d.yaml          # Base settings
    ↓ merge
conf/hpc/phase_d.yaml           # HPC overrides (from attachment)
    ↓ merge  
conf/experiments/quick_test.yaml # Experiment-specific overrides
```

**Key HPC settings from phase_d.yaml:**
```yaml
training:
  num_epochs: 10
  batch_size: 64
  learning_rate: 3e-5
  layerwise_lr_decay: 0.9
  gradient_accumulation_steps: 2
  precision: "fp8"  # Falls back to BF16 if TransformerEngine unavailable

optimization:
  use_aot_mode: true
  use_torch_compile: true
  torch_compile_mode: "reduce-overhead"
  use_fused_optimizer: true
  use_device_prefetch: true
  token_budget: 131072  # ~256 samples @ 512 tokens

execution:
  autotuning:
    enabled: true
    max_token_budget: 524288  # For 80GB+ GPUs
  dynamic_batching:
    enabled: true
    max_batch_size: 4096
```

### 3. Main Training Orchestration (`phase_d_pipeline.py::run_phase_d_training`)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        run_phase_d_training()                            │
├──────────────────────────────────────────────────────────────────────────┤
│  1. Resolve all paths relative to config root                            │
│  2. Optionally add round-robin split column (if --data-change yes)       │
│  3. Load merged config                                                   │
│  4. Create output directories                                            │
│                                                                          │
│  ╔══════════════════════════════════════════════════════════════════╗   │
│  ║                    BASELINE MODEL TRAINING                        ║   │
│  ║  • text_field = "post" (raw text, no masking)                    ║   │
│  ║  • use_affine_guard = False                                       ║   │
│  ║  • Output: artifacts/phase_d_test/baseline/                       ║   │
│  ╚══════════════════════════════════════════════════════════════════╝   │
│                                ↓                                         │
│  ╔══════════════════════════════════════════════════════════════════╗   │
│  ║                  CONSTRAINED MODEL TRAINING                       ║   │
│  ║  • text_field = "post_masked" (pollution spans replaced)         ║   │
│  ║  • use_affine_guard = True (applies projection_matrix.pt)        ║   │
│  ║  • Output: artifacts/phase_d_test/constrained/                    ║   │
│  ╚══════════════════════════════════════════════════════════════════╝   │
│                                ↓                                         │
│  5. Save comparative summary (phase_d_comparative_metrics.json)          │
│  6. Optionally generate HTML report                                      │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Training Steps

### Step 3.1: Baseline Model Training

#### Data Loading
```python
train_loader, label_maps = baseline_trainer._build_loader(
    text_field="post",          # Uses ORIGINAL text (no masking)
    label_maps=None,            # Creates new label mappings
    split="train",
    shuffle=True,
    enable_dynamic_batching=True,  # Token-budget-based batching
)
```

**Dynamic batching** with HPC settings:
- `token_budget: 131072` tokens per batch
- At 512 tokens/sample → ~256 samples per batch
- Autotuning adjusts budget based on OOM events

#### Model Architecture
```python
model, head = baseline_trainer._build_model(use_affine_guard=False)
```

- **Encoder**: `roberta-base` (or configured model)
- **Classification Head**: Multi-task head with one classifier per task (disorder types, severity, etc.)
- **No Affine Guard**: Hidden states pass through unchanged

#### Training Loop
```python
baseline_trainer._train(
    model=model,
    head=head,
    loader=train_loader,
    eval_loader=val_loader,
    run_dir=baseline_dir,
)
```

**With HPC config:**
- **Epochs**: 10 (from `training.num_epochs`)
- **Precision**: FP8 (or BF16 fallback)
- **Optimizer**: Fused AdamW (`use_fused_optimizer: true`)
- **Scheduler**: Cosine with 500 warmup steps
- **Early Stopping**: Enabled, patience=2, metric=`f1_macro`
- **AOT Compilation**: `torch.compile(mode="reduce-overhead")` with CUDA graphs

#### Evaluation
```python
test_metrics = baseline_trainer._evaluate(model, head, test_loader, num_classes)
test_details = baseline_trainer._evaluate_detailed(model, head, test_loader, num_classes)
```

**Outputs saved to baseline:**
- `test_metrics.json` - Per-task accuracy, F1, etc.
- `test_details.json` - Confusion matrices, per-class metrics
- Model checkpoints

---

### Step 3.2: Constrained Model Training

#### Data Loading
```python
train_loader_c, _ = constrained_trainer._build_loader(
    text_field="post_masked",   # Uses MASKED text from Phase A
    label_maps=label_maps,       # Reuses baseline's label mappings
    split="train",
    shuffle=True,
    enable_dynamic_batching=True,
)
```

**Key difference**: Text like `"I have depression"` becomes `"I have [MASK:DISORDER]"` based on GLiNER detection from Phase A.

#### Model Architecture with Affine Guard
```python
model_c, head_c = constrained_trainer._build_model(use_affine_guard=True)
```

**Affine Guard Layer**:
```
Hidden States (H) → P @ H → Projected Hidden States
```
Where `P` is the LEACE projection matrix from projection_matrix.pt.

The projection removes linear information about pollution entities while preserving task-relevant features.

#### Training Loop
Same hyperparameters as baseline (10 epochs, FP8, etc.) but:
- Model has Affine Guard layer injected
- Input text is masked version

**Outputs saved to `artifacts/phase_d_test/constrained/`:**
- `test_metrics.json`
- `test_details.json`
- Model checkpoints

---

### Step 3.3: Comparative Summary

```python
_save_comparative_summary(results, output_dir, num_classes)
```

Creates `artifacts/phase_d_test/phase_d_comparative_metrics.json`:
```json
{
  "baseline": {
    "test": {"disorder": {"accuracy": 0.85, "f1_macro": 0.82}, ...}
  },
  "constrained": {
    "test": {"disorder": {"accuracy": 0.83, "f1_macro": 0.80}, ...}
  },
  "delta": {
    "disorder": {
      "accuracy_delta": -0.02,
      "baseline_accuracy": 0.85,
      "constrained_accuracy": 0.83
    }
  }
}
```

### Step 3.4: Report Generation (Optional)

If `evaluation.write_report: true` in config:
```python
from .evaluation.comparative_report import generate_phase_d_report
generate_phase_d_report(phase_d_dir, reports_dir, dataset_path)
```

---

## When is CHG Run?

**CHG (Causal Head Gating) is NOT run during `run-phase-d`.**

CHG is a separate verification step executed via:
```bash
python -m neuro_stylometry verify \
    --dataset artifacts/phase_a/clean_dataset.arrow \
    --phase-d-dir artifacts/phase_d_test \
    --artifacts-dir artifacts/phase_a \
    --output-dir artifacts/phase_d_test/chg
```

CHG analyzes which attention heads facilitate vs. suppress pollution information after Phase D training.

---

### AOT (Ahead-Of-Time) Pipeline

For maximum throughput, pre-tokenize the dataset:

```bash
# Standard AOT tokenization (variable-length)
python scripts/preprocess_tokens.py \
    --input artifacts/phase_a/clean_dataset.arrow \
    --output artifacts/phase_d/tokenized_dataset.arrow \
    --model roberta-base \
    --max-length 512 \
    --workers 8

# Zero-copy mode (pre-padded fixed-length arrays)
python scripts/preprocess_tokens.py \
    --input artifacts/phase_a/clean_dataset.arrow \
    --output artifacts/phase_d/tokenized_dataset.arrow \
    --model roberta-base \
    --max-length 512 \
    --workers 8 \
    --pre-pad
```

The `--pre-pad` flag creates fixed-length arrays that enable single `torch.stack()` collation (~2ms vs ~15ms/batch).

### Configuration Examples

**HPC Mode** (`conf/hpc/phase_d.yaml`):
```yaml
optimization:
  use_aot_mode: true
  use_torch_compile: true
  torch_compile_mode: "reduce-overhead"
  use_fused_optimizer: true
  use_device_prefetch: true
  quantize_step: 16
  token_budget: 32768

training:
  precision: "fp8"  # Falls back to BF16 if unavailable
  batch_size: 64

execution:
  autotuning:
    enabled: true
    max_token_budget: 393216
  data_loader:
    num_workers: 16
    pin_memory: true
    persistent_workers: true
    prefetch_factor: 8
```

**Laptop Mode** (`conf/laptop/phase_d.yaml`):
```yaml
optimization:
  use_aot_mode: true
  use_torch_compile: false  # Compilation overhead > benefit for small runs
  use_fused_optimizer: true
  use_device_prefetch: true
  quantize_step: 16
  token_budget: 8192

training:
  precision: "fp32"  # Stability on consumer GPUs
  batch_size: 4

execution:
  autotuning:
    enabled: false
  data_loader:
    num_workers: 0  # Single-threaded for stability
    pin_memory: true
```

### Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AOT Pipeline Data Flow                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  preprocess_tokens.py          PhaseDDataset           FastCollator          │
│  ┌─────────────────┐          ┌─────────────┐         ┌─────────────┐       │
│  │ Parallel        │          │ Memory-mapped│         │ Zero-copy   │       │
│  │ tokenization    │ ──Arrow──│ Arrow table │ ──AOT───│ tensor stack│       │
│  │ (CPU workers)   │          │ (mmap)      │         │ (no Python) │       │
│  └─────────────────┘          └─────────────┘         └─────────────┘       │
│                                      │                       │               │
│                                      ▼                       ▼               │
│                              QuantizedBucketSampler   DevicePrefetcher      │
│                              ┌─────────────────┐     ┌─────────────────┐    │
│                              │ Snap-to-Grid    │     │ CUDA stream     │    │
│                              │ token budgeting │     │ async H2D       │    │
│                              └─────────────────┘     └─────────────────┘    │
│                                                              │               │
│                                                              ▼               │
│                                                      Trainer._train()       │
│                                                      ┌─────────────────┐    │
│                                                      │ torch.compile   │    │
│                                                      │ CUDA Graphs     │    │
│                                                      │ Strided telemetry│   │
│                                                      └─────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete I/O Summary

### Inputs

| File | Description |
|------|-------------|
| tokenized_dataset.arrow | Pre-tokenized dataset with `post`, `post_masked`, labels, `split` column |
| projection_matrix.pt | LEACE projection matrix (768×768 for RoBERTa) |
| phase_d.yaml | HPC training config |
| quick_test.yaml | Experiment overrides |
| gliner_taxonomy.yaml | Mask token taxonomy |

### Outputs

```
artifacts/phase_d_test/
├── baseline/
│   ├── test_metrics.json
│   ├── test_details.json
│   └── checkpoints/
├── constrained/
│   ├── test_metrics.json
│   ├── test_details.json
│   └── checkpoints/
├── phase_d_comparative_metrics.json
└── reports/  (if write_report enabled)
    └── phase_d_report.html
```

---

## Timeline Visualization

```
Time ─────────────────────────────────────────────────────────────────────────►

│ Config Load │ Baseline DataLoader │ Baseline Training (10 epochs) │ Baseline Eval │
                                                                                    │
              │ Constrained DataLoader │ Constrained Training (10 epochs) │ Constrained Eval │
                                                                                              │
                                                                          │ Save Comparisons │
                                                                                              │
                                                                              │ Report Gen │
```

## License

MIT License
