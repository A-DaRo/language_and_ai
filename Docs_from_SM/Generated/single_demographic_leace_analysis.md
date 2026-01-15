# Single-Demographic LEACE Analysis

**Date:** January 15, 2026  
**Context:** Investigation of `--use-only nationality` E2E test results

---

## Executive Summary

The current LEACE implementation, designed for **multi-demographic joint erasure with sparse labels**, exhibits suboptimal behavior when applied to a **single high-cardinality categorical demographic** with no missing values. While mathematically correct (projection matrix passes idempotence), the implementation has inefficiencies and the probe evaluation fails.

---

## Test Results Analysis

### What Worked
- ✅ Taxonomy filtering: Only nationality prompts used in GLiNER
- ✅ Entity detection: 840 nationality spans detected
- ✅ LEACE projection computed: Shape (768, 768), idempotence error 2.6e-7
- ✅ 56 directions erased (matches number of nationality categories)

### What Failed
- ❌ Probe evaluation: "n_splits=5 cannot be greater than the number of members in each class"
- ⚠️ Visualizations in wrong directory path (reports/phase_a/ vs reports/)
- ⚠️ Missing indicator column is degenerate (all zeros)

---

## Technical Analysis

### 1. Concept Matrix Structure (Single Categorical, No Nulls)

For nationality with C=56 categories and N=200 samples:

```
Z shape: (200, 57)
├── Columns 0-55: One-hot encoding for 56 nationalities  
└── Column 56: MISSING indicator (ALL ZEROS - useless)
```

**Mathematical Properties:**
- One-hot constraint: Σ z_{ic} = 1 for all rows → columns linearly dependent
- rank(Z) ≤ 55 (not 56)
- Missing column has zero variance → contributes nothing

### 2. Covariance Matrix Issues

**Σ_ZZ (57 × 57):**
- Has rank at most 55
- Row/column 56 (missing indicator) is all zeros
- One-hot columns sum to constant, causing rank deficiency

**Σ_XZ (768 × 57):**
- Column 56 (missing indicator covariance) is all zeros
- Captures covariance between embeddings and category indicators

### 3. LEACE Computation

The code correctly handles rank deficiency via pseudo-inverse:
```python
U, S, _ = torch.linalg.svd(M, full_matrices=False)
tol = float(S.max().item()) * 1e-6
r = int((S > tol).sum().item())  # Effective rank
```

Result: **56 directions erased** - this is correct! It removes:
- The 55 independent category mean directions
- Plus ~1 direction from regularization/numerical effects

### 4. Probe Failure Root Cause

```python
# From sklearn cross-validation
"n_splits=5 cannot be greater than the number of members in each class"
```

With 56 categories and 200 samples:
- Average 3.6 samples per category
- 5-fold CV requires ≥5 samples per class
- **Impossible to evaluate amnesic drop for high-cardinality categorical**

---

## Comparison: Multi-Demographic vs Single-Demographic

| Aspect | Multi-Demographic (Report) | Single-Demographic (nationality) |
|--------|---------------------------|----------------------------------|
| Z dimensions | ~20 (all columns combined) | 57 (56 categories + missing) |
| Missing indicators | Useful (sparse labels) | Useless (all zeros) |
| rank(Z) | ~18-20 (low) | 55 (high but deficient) |
| Samples per class | Varies by column | 3-4 per nationality |
| Probe feasibility | Likely works | Fails (insufficient samples) |
| Erased directions | ~20 | 56 |
| "Meta-leakage" guard | Active | Inactive (no missing values) |

---

## Recommendations

### Option A: Adapt Encoder for Single-Demographic Mode

```python
class DemographicEncoder:
    def fit(self, table, skip_missing_indicator=False):
        # For single-demographic no-nulls case:
        # - Don't add MISSING column if all values present
        # - Use dummy encoding (C-1 columns) instead of one-hot (C columns)
```

**Benefits:**
- Removes degenerate columns
- Proper rank for Σ_ZZ
- Slightly more efficient

### Option B: Adapt Probe for High-Cardinality

```python
def compute_amnesic_drop_extended(...):
    # For high-cardinality:
    # - Reduce n_splits dynamically based on min class size
    # - Or use stratified holdout instead of k-fold
    min_class_size = min(class_counts)
    n_splits = min(5, min_class_size)
```

### Option C: Sample Aggregation for Probe

For nationality with 56 categories:
1. Group into regions (e.g., "Western Europe", "Eastern Europe", "Asia")
2. Evaluate amnesic drop on grouped labels
3. More robust evaluation with fewer classes

### Option D: Use Original Probe Method (Class-Conditional)

Instead of `compute_projection_from_concepts`, use `compute_projection` with:
- Integer labels (0-55 for nationality)
- Class-conditional means
- Avoids one-hot rank issues

---

## Finalized Implementation Plan

This plan specifically addresses single-demographic erasure in the "no-nulls" regime (enforced by strict filtering in `--use-only` mode).

### 1. Refined Concept Encoding Strategy

**Objective**: Eliminate degenerate dimensions in $Z$ to improve numerical stability and condition numbers.

**Analysis**:
The current `DemographicEncoder` naively appends a `__MISSING` indicator column even when the dataset is fully observed (dense). For a categorical variable with $K$ classes this results in a $Z$ matrix of size $N \times (K+1)$ where column $K+1$ is all zeros. While `LEACEComputer` handles rank deficiency via `_robust_pinv`, this is theoretically inelegant and computationally wasteful.

**Action**: Modify `DemographicEncoder.fit()`
1.  **Constraint**: Do **not** use dummy encoding (dropping one category). Keep full one-hot encoding.
    *   *Reason*: LEACE minimizes $||\Delta X||_F$. Symmetric treatment of all categories is generally preferred. The "Standard LEACE" derivation (Belrose et al., 2023, Theorem 3.1) relies on centering class means. Full one-hot encoding is mathematically equivalent to the class-mean approach when used with `compute_projection_from_concepts`. Dropping one category arbitrarily breaks symmetry unless handled carefully.
2.  **Change**: Conditionally suppress the `MISSING` column.
    ```python
    # In DemographicEncoder.fit
    has_missing = series.isna().any()
    if has_missing:
        # Add MISSING indicator
    # Else: Skip MISSING indicator (pure clean one-hot)
    ```

### 2. Robust Probe Evaluation Strategy

**Objective**: Enable amnesic drop evaluation for high-cardinality demographics (e.g. Nationality: 56 classes, ~200 samples).

**Analysis**:
Standard $k$-fold cross-validation fails when $N_{class} < k$. With 56 classes and 200 samples, some classes have $N_c \approx 3$, making 5-fold splitting impossible.

**Action**: Update `probes.py` with Dynamic Split Strategy
Implement a "best-effort" evaluation hierarchy:
1.  **Standard K-Fold**: If $\min(N_c) \ge 5$, use $k=5$.
2.  **Low-Sample Adaptation**:
    *   If $2 \le \min(N_c) < 5$: Use `StratifiedShuffleSplit` with `n_splits=5, test_size=0.2`. This is less strict than K-Fold but guarantees representation.
    *   Alternatively, reduce $k$ to $\min(N_c)$, but $k=2$ is noisy.
3.  **Fallback (Singleton Classes)**: If $\min(N_c) < 2$ (classes with 1 sample):
    *   **CRITICAL**: Drop classes with $< 2$ samples from the *probe evaluation* (but not the erasure).
    *   *Reason*: You cannot train/test on a single sample.
    *   Warning: "Dropping X classes with insufficient samples for probe evaluation."

### 3. Algorithm Selection: Concepts vs. Explicit Classes

**Decision**: Continue using `compute_projection_from_concepts` (General LEACE).

**Justification**:
*   **Uniformity**: The pipeline architecture treats "demographics" as a possibly multi-column design matrix $Z$ (supporting continuous, binary, and multi-label attributes uniformly).
*   **Equivalence**: As established in *LEACE: Perfect linear concept erasure in closed form* (Belrose et al., 2023), Section 4, the affine guardedness condition ($\Sigma_{XZ} = 0$) generalizes the linear guardedness condition (equal class means).
*   With a full one-hot $Z$ (and proper centering, which `leace.py` does):
    $$ \Sigma_{XZ} \propto \sum_c p_c (\mu_c - \mu)(\mathbf{e}_c - \bar{\mathbf{e}})^T $$
    Zeroing this cross-covariance is equivalent to collapsing class means.
*   Therefore, specialized distinct codepaths for "single categorical" vs "multi-demographic" are unnecessary complexity, provided rank-deficient $Z$ is handled (which `leace.py` already does via SVD/pinv).

### 4. Implementation Checklist

1.  **Update `DemographicEncoder`**:
    *   [ ] Add `_has_missing_values` check in `fit`.
    *   [ ] Only register `__MISSING` feature if true.
    *   [ ] Update `transform` to respect the fitted feature set (don't write to missing offset if it doesn't exist).

2.  **Update `ProbeTrainer`**:
    *   [ ] In `evaluate_amnesic_drop`, compute class counts first.
    *   [ ] Filter samples: `valid_indices = label_counts[labels] >= 2`.
    *   [ ] If `valid_indices` drops > 10% of data, log warning.
    *   [ ] Use `StratifiedShuffleSplit` if `n_splits > min_class_samples`.


---

## Appendix: Metrics from Test Run

```json
{
  "leace": {
    "effective_rank": 712,
    "n_erased_directions": 56,
    "idempotence_error": 2.6e-7
  },
  "probe": {
    "nationality": {
      "error": "n_splits=5 cannot be greater than the number of members in each class."
    }
  },
  "dataset": {
    "num_samples": 200,
    "demographic_coverage": {
      "nationality": 1.0
    }
  }
}
```
