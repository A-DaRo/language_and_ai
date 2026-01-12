# Phase D Status Report (Implementation Snapshot)

**Document Type:** Status + Gap Analysis  
**Scope:** Phase D (Neural Stylometry)  
**Context:** Phase A artifacts available under `artifacts/phase_a/`  
**Last Updated:** 2026-01-xx  

---

## 1. Executive Summary

Phase D is now partially implemented beyond the original stubs. The core model wrapper, Affine Guard injection, tokenizer alignment, dataset loading, baseline vs constrained training loop, checkpointing, metrics reporting, and CHG/SVS verification entry points are in place. The remaining work is primarily in evaluation/reporting, richer training utilities, and full CHG/SVS rigor.

---

## 2. Implemented in This Update (Chat Session Summary)

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

### 2.5 CLI Wiring

**Files:**
- `src/neuro_stylometry/__main__.py`

**Run Phase D:**
```
python -m neuro_stylometry run-phase-d \
  --dataset artifacts/phase_a/clean_dataset.arrow \
  --output-dir artifacts/phase_d \
  --artifacts-dir artifacts/phase_a
```

**Verification:**
```
python -m neuro_stylometry verify \
  --dataset artifacts/phase_a/clean_dataset.arrow \
  --phase-d-dir artifacts/phase_d \
  --output-dir artifacts/phase_d/chg
```

---

### 2.6 CHG + SVS Verification

**Files:**
- `src/neuro_stylometry/stylometry_net/chg_verifier.py`
- `src/neuro_stylometry/stylometry_net/svs_calculator.py`
- `src/neuro_stylometry/stylometry_net/verification.py`

**What it does:**
- CHG learns gate parameters per head (proxy loss scaling + L1 regularization)
- SVS computes function‑word vs content‑word attention mass ratio
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
| Tokenizer alignment | ✅ | Mask tokens from taxonomy, enforced single-token |
| Affine Guard | ✅ | Projection matrix buffer, frozen by default |
| Transformer wrapper | ✅ | Injects guard after embeddings |
| Dataset loader | ✅ | Arrow read + split logic + label maps |
| Training loop | ✅ | Baseline + constrained, logs, checkpoints |
| Metrics | ✅ | Per-task accuracy and macro‑F1 |
| CHG verifier | ✅ (proxy) | Lightweight gate learning |
| SVS | ✅ (heuristic) | Function vs content mass |
| CLI wiring | ✅ | run‑phase‑d + verify |

### 3.2 Known Limitations / Improvements

1. **CHG is proxy‑based**
   - Current gate learning scales loss by mean gate value.
   - Does not re‑run with gated attention outputs.
   - Improvement: implement true attention‑head gating forward pass.

2. **SVS is heuristic**
   - Uses a fixed function‑word list and CLS attention.
   - Improvement: use POS tagging or expanded function‑word lexicon.

3. **Evaluation on validation split only**
   - Currently uses `split=val` if present.
   - Improvement: add test split evaluation and report both.

4. **Training utilities are minimal**
   - No scheduler/warmup, no AMP, no gradient accumulation.
   - Improvement: wire precision management + scheduler in training.

5. **No top‑level metrics aggregation**
   - Baseline and constrained metrics saved separately.
   - Improvement: aggregate into `artifacts/phase_d/phase_d_metrics.json`.

---

## 4. Remaining Phase D Work (Spec‑Aligned)

### 4.1 Evaluation + Reporting (Primary Blocker)

**Required files:**
- `src/neuro_stylometry/evaluation/attention_analysis.py`
- `src/neuro_stylometry/evaluation/visualizations.py`
- `src/neuro_stylometry/evaluation/comparative_report.py`

**Required artifacts:**
- Baseline vs constrained plots (learning curves, accuracy bars, gate heatmaps)
- Report HTML under `artifacts/reports/phase_d_report.html`
- Aggregated `phase_d_metrics.json` at `artifacts/phase_d/`

---

### 4.2 Verification Enhancements

**Needed improvements:**
- Full CHG implementation with gated attention outputs.
- SVS with POS tagging for robust stylometry vs content categorization.
- Comparative summary output (delta SVS, delta accuracy).

---

### 4.3 Training Enhancements

**Recommended upgrades:**
- Mixed precision + GradScaler
- Learning‑rate scheduler (linear warmup + cosine decay)
- Gradient accumulation
- Checkpoint resumption

---

### 4.4 Tests

**Missing tests:**
- Unit tests for tokenizer alignment (real-model already exists)
- Unit tests for Affine Guard correctness
- Integration test for baseline + constrained training (tiny dataset)
- Golden tests for CHG/SVS outputs

---

## 5. Suggested Next Implementation Order

1. **Phase D evaluation/reporting pipeline**
2. **CHG/SVS rigor improvements**
3. **Training quality upgrades (precision, scheduler)**
4. **Test coverage**
5. **Config schema finalization (last per instruction)**

---

## 6. Quick Reference (Current CLI Usage)

**Train baseline + constrained:**
```
python -m neuro_stylometry run-phase-d \
  --dataset artifacts/phase_a/clean_dataset.arrow \
  --output-dir artifacts/phase_d \
  --artifacts-dir artifacts/phase_a
```

**Verify CHG + SVS:**
```
python -m neuro_stylometry verify \
  --dataset artifacts/phase_a/clean_dataset.arrow \
  --phase-d-dir artifacts/phase_d \
  --output-dir artifacts/phase_d/chg
```

---

*End of report.*
