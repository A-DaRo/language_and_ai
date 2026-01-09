# Phase D Implementation Plan (Repo-Aligned)

## Current Repository State (Quick Overview)
- Phase A pipeline exists and is wired via `src/neuro_stylometry/phase_a_pipeline.py` with laptop/HPC strategies under `src/neuro_stylometry/pollution_guard/strategies/`.
- GLiNER detection, masking, LEACE, and dataset handling are implemented; Phase A outputs follow the handover contract in `scripts/validate_phase_a_handover.py` and `src/neuro_stylometry/data_engine/schemas.py`.
- Phase D core modules are stubs: `src/neuro_stylometry/stylometry_net/affine_guard.py`, `src/neuro_stylometry/stylometry_net/transformer.py`, `src/neuro_stylometry/stylometry_net/classification_head.py`, `src/neuro_stylometry/stylometry_net/chg_verifier.py`, `src/neuro_stylometry/stylometry_net/svs_calculator.py`.
- Training and evaluation layers are also placeholders: `src/neuro_stylometry/training/*.py` and `src/neuro_stylometry/evaluation/*.py`.
- CLI has Phase A commands in `src/neuro_stylometry/__main__.py`; Phase D entry points and `scripts/run_phase_d.py` are placeholders.
- Configs exist for pipeline and training under `conf/base`, `conf/laptop`, and `conf/hpc`, but Phase D-specific settings are not yet defined.
- SLURM scripts exist under `scripts/slurm/` for Phase A, Phase D, and verification but rely on CLI commands that are not fully implemented.

## Phase D Scope and Preconditions (From towards_project_completion.md)
- Inputs required: `clean_dataset.arrow` with `post_masked` and `projection_matrix.pt` from Phase A.
- Tokenizer must register typed mask tokens as single tokens; repeat Phase A’s special-token registration for Phase D.
- Affine Guard layer must load and freeze the LEACE projection matrix (`projection_matrix.pt`).
- Two-model training is required: a dirty baseline (no masking/no projection) and a constrained model (masking + Affine Guard).
- Verification requires Causal Head Gating (CHG) and Stylometric Validity Score (SVS).
- Expected Phase D artifacts include model checkpoints, training logs, metrics JSON, visualizations, and reports.

## Phase D Implementation Plan (Two-Person Parallel Workstreams)

### Track A: Core Model + Data + Training
- Step A1: Define Phase D configs in `conf/base/training.yaml` (or new `conf/base/phase_d.yaml`) with task map, mask tokens, baseline vs constrained toggles, and artifact paths aligned to current output contracts.
- Step A2: Implement `src/neuro_stylometry/stylometry_net/tokenizer.py` to register mask tokens, control max length, and emit `input_ids`/`attention_mask`.
- Step A3: Implement `src/neuro_stylometry/stylometry_net/affine_guard.py` as a frozen projection buffer with `from_checkpoint()`, following LEACE idempotence requirements.
- Step A4: Implement `src/neuro_stylometry/stylometry_net/transformer.py` to wrap a pretrained RoBERTa encoder, inject Affine Guard after embeddings, and keep tokenizer alignment.
- Step A5: Implement `src/neuro_stylometry/stylometry_net/classification_head.py` with per-task heads and masked-loss computation for missing labels.
- Step A6: Build dataset loader utilities for Phase D (Arrow → tokenized batches) using `SOBR_SCHEMA` and ensuring memory-mapped loading for scale.
- Step A7: Implement training loop in `src/neuro_stylometry/training/trainer.py` plus optimizer/scheduler utilities, producing logs and checkpoints for baseline and constrained runs.
- Step A8: Wire Phase D CLI entry points in `src/neuro_stylometry/__main__.py` and implement `scripts/run_phase_d.py` to run both models and emit `phase_d_metrics.json`.

### Track B: Verification + Evaluation + Reporting
- Step B1: Implement `src/neuro_stylometry/stylometry_net/chg_verifier.py` for Causal Head Gating with gating regularization and head classification output.
- Step B2: Implement `src/neuro_stylometry/stylometry_net/svs_calculator.py` to compute SVS and serialize per-model results.
- Step B3: Implement evaluation metrics in `src/neuro_stylometry/training/metrics.py` and `src/neuro_stylometry/evaluation/` (accuracy/F1, confusion matrices, calibration curves).
- Step B4: Implement Phase D visualizations and a report generator in `src/neuro_stylometry/evaluation/visualizations.py` and `src/neuro_stylometry/evaluation/comparative_report.py`.
- Step B5: Add attention analysis tooling in `src/neuro_stylometry/evaluation/attention_analysis.py` to support head-level inspection and POS/token-type aggregation.

### Synchronization Points (Both Tracks)
- Step S1: Agree on a shared Phase D config schema (tasks, labels, thresholds, output paths) before training and evaluation are connected.
- Step S2: Validate Phase A handover artifacts with `scripts/validate_phase_a_handover.py` and confirm mask-token registration in tokenizer and model.
- Step S3: Define a shared metrics JSON schema (`phase_d_metrics.json`) used by training, evaluation, and reports.
- Step S4: Run a small end-to-end dry run using a tiny Arrow slice to validate that baseline/constrained runs both complete.
- Step S5: Finalize Phase D report output locations under `artifacts/reports/` and ensure paths align with `scripts/slurm/phase_d.sbatch`.

## Expanded Task Breakdown (Concrete Implementation Checklist)

### Core Configuration and Contracts
- Define `phase_d` config keys under `conf/base/training.yaml` or a new `conf/base/phase_d.yaml`.
- Add mode overrides in `conf/laptop/training.yaml` and `conf/hpc/training.yaml` (batch sizes, gradient accumulation, devices).
- Define Phase D artifact paths to match `artifacts/phase_d/{baseline,constrained}/` and `artifacts/reports/`.
- Add a `phase_d_metrics.json` schema reference for evaluation/reporting alignment.

### Data and Tokenization
- Implement Arrow dataset loader to read `clean_dataset.arrow` with memory mapping and validate `post_masked`.
- Ensure tokenizer registers mask tokens as `additional_special_tokens` and that each mask token encodes to one ID.
- Provide a tokenization wrapper that supports `max_length`, padding, truncation, and dynamic batching.

### Model and Training
- Implement Affine Guard with buffer registration and device-safe loading.
- Implement transformer wrapper with Affine Guard injection after embeddings and custom attention mask handling.
- Implement multi-task head and a masked loss function to ignore missing labels.
- Implement trainer with:
  - Baseline vs constrained training toggles.
  - Checkpointing and JSONL logs per run.
  - Consistent seeds for reproducibility.
  - Mixed precision support in HPC mode.

### Evaluation and Verification
- Implement standard metrics (accuracy, macro-F1) per task for both models.
- Implement confusion matrix and calibration curve generation.
- Implement CHG training loop on frozen encoder and per-head gate classification.
- Implement SVS computation with a documented token-type/POS tagging scheme.

### Reporting and Artifacts
- Implement Phase D report generator (HTML) referencing all plots and metrics.
- Implement `experiment_summary.json` aggregation matching the roadmap schema.
- Update SLURM scripts to call final CLI with correct artifact paths and config.

## Phase A HPC Translation Plan (Batching + Shell Scripts)
- Step H1: Add dataset sharding utility (e.g., `scripts/shard_arrow_dataset.py`) to split `clean_dataset.arrow` or raw Arrow into N shards for parallel runs.
- Step H2: Extend `src/neuro_stylometry/pollution_guard/semantic_chunker.py` to implement `parallel_chunking_workers` using `multiprocessing.get_context("spawn")` with worker-side tokenizer/PySBD initialization.
- Step H3: Add a batched HPC Phase A runner that processes shards for GLiNER detection + masking, outputs per-shard `pollution_logs.arrow`, and merges results.
- Step H4: Add a merge step for masked datasets (Arrow concat) and compute LEACE once on the merged masked corpus to preserve a global projection.
- Step H5: Create a simple shell wrapper (e.g., `scripts/hpc/run_phase_a_batches.sh`) that launches shard jobs locally or via SLURM array jobs.
- Step H6: Update `scripts/slurm/phase_a.sbatch` to optionally run as an array and call the batch script with shard index and total shards.
- Step H7: Document HPC batch workflow in `Docs_from_SM/Generated/phase_d_plan.md` and align with `conf/hpc/pipeline.yaml` for batch sizing and device settings.

## Deliverables Checklist (Phase D)
- `src/neuro_stylometry/stylometry_net/` fully implemented: tokenizer, affine guard, transformer, classification head, CHG verifier, SVS calculator.
- `src/neuro_stylometry/training/` provides full training loop, optimizer/scheduler, checkpointing, and metrics.
- `src/neuro_stylometry/evaluation/` provides Phase D visualizations, reports, and attention analysis outputs.
- CLI and scripts support `run-phase-d` for baseline + constrained runs with consistent artifact paths.
- Phase D artifacts: checkpoints, logs, `phase_d_metrics.json`, CHG gates, SVS summaries, and `phase_d_report.html`.

## Testing and Validation
- Unit tests for Affine Guard projection correctness and tokenizer mask-token alignment.
- Integration test to run a tiny Phase D training step (baseline + constrained) and verify artifact creation.
- Handover validation via `scripts/validate_phase_a_handover.py` before Phase D execution.

## Phase D Config Schema (Proposed)

```yaml
# conf/base/phase_d.yaml (or merged into conf/base/training.yaml)
phase_d:
  experiment_name: "phase_d_default"
  seed: 1337
  output_dir: "artifacts/phase_d"
  reports_dir: "artifacts/reports"

data:
  dataset_path: "artifacts/phase_a/clean_dataset.arrow"
  text_field_baseline: "post"         # raw
  text_field_constrained: "post_masked"
  label_fields:
    female: 2
    birth_year: 8
    nationality: 30
    political_leaning: 4
    extrovert: 2
    sensing: 2
    feeling: 2
    judging: 2
  missing_label_value: -1

tokenizer:
  model_name: "roberta-base"
  max_length: 512
  mask_tokens:
    - "[MASK:AGE]"
    - "[MASK:GENDER]"
    - "[MASK:NATIONALITY]"
    - "[MASK:POLITICAL]"
    - "[MASK:MBTI]"

model:
  hidden_dim: 768
  dropout: 0.1
  use_affine_guard: true
  projection_matrix_path: "artifacts/phase_a/projection_matrix.pt"
  pretrained_model: "roberta-base"

training:
  num_epochs: 3
  batch_size: 16
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  eval_steps: 500
  save_steps: 1000
  logging_steps: 100
  mixed_precision: false

optimizer:
  name: "adamw"
  lr: 2e-5
  weight_decay: 0.01

scheduler:
  name: "linear"
  num_warmup_steps: 500

evaluation:
  metrics:
    - "accuracy"
    - "f1_macro"
  write_confusion_matrices: true
  write_calibration_curves: true

chg:
  enabled: true
  num_layers: 12
  num_heads: 12
  init_value: 1.0
  regularization: 0.01
  threshold: 0.5
```

### Mode Overrides (Examples)
- `conf/laptop/training.yaml`: reduce batch size, disable CHG by default, enable smaller subsets.
- `conf/hpc/training.yaml`: increase batch size, enable mixed precision, keep CHG enabled, and set `output_dir` to scratch.
