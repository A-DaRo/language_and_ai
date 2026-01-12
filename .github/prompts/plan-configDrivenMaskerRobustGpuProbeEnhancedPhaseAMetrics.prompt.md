# Plan: Config-Driven Masker, Robust GPU Probe, Enhanced Phase A Metrics

Refactor the Phase A pipeline to eliminate hardcoded mask mappings, fix sklearn compatibility and add GPU-accelerated probing, and introduce additional metrics/visualizations for deeper analysis.

## Steps

1. **Make `SpanMasker` config-driven** — Remove `COLUMN_TO_MASK` and `ENTITY_TO_MASK` class constants in [masker.py](src/neuro_stylometry/pollution_guard/masker.py#L43-L71). Add a `@classmethod from_taxonomy(taxonomy_cfg)` that extracts `mask_token` from each column definition in [gliner_taxonomy.yaml](conf/base/gliner_taxonomy.yaml). Update `get_masker()` in [phase_a_pipeline.py](src/neuro_stylometry/phase_a_pipeline.py#L222-L231) to call this factory, and wire the same in both [laptop.py](src/neuro_stylometry/pollution_guard/strategies/laptop.py) and [hpc.py](src/neuro_stylometry/pollution_guard/strategies/hpc.py) strategy `execute()` methods.

2. **Fix sklearn compatibility and add GPU probe backends** — In [probe.py](src/neuro_stylometry/pollution_guard/probe.py#L75-L99), refactor `LinearProbe` to support three backends: `sklearn` (CPU), `cuml` (GPU via RAPIDS cuML), and `torch` (PyTorch-native logistic regression with `torch.linalg.lstsq` or gradient descent). Add backend auto-detection: use `cuml` if available and on CUDA, else fall back. Remove brittle `multi_class` kwarg handling — modern sklearn auto-selects based on solver.

3. **Add k-fold cross-validation to `compute_amnesic_drop()`** — Replace the single 80/20 split with stratified k-fold (k=5 default) to produce mean ± std accuracy and statistical significance (paired t-test between before/after). Return extended tuple with confidence intervals.

4. **Implement `TorchLogisticProbe` for HPC** — Create a new class in [probe.py](src/neuro_stylometry/pollution_guard/probe.py) using `torch.nn.Linear` with BCEWithLogitsLoss/CrossEntropyLoss, SGD/Adam optimizer, running on GPU with BF16 autocast. Batch the training for large embedding sets (>100k samples).

5. **Extend Phase A metrics** — In [metrics.py](src/neuro_stylometry/evaluation/metrics.py), add: (a) per-document masking rate histogram buckets, (b) embedding separability scores (silhouette, Davies-Bouldin) before/after LEACE, (c) per-column amnesic drop with 95% CI from cross-validation.

6. **Add new visualizations** — In [visualizations_phase_a.py](src/neuro_stylometry/evaluation/visualizations_phase_a.py), add: (a) `plot_amnesic_drop_with_ci()` (error bars), (b) `plot_embedding_separability()` (silhouette before/after bar chart), (c) `plot_probe_learning_curves()` (if torch backend used).

## Further Considerations

1. **cuML dependency** — cuML optional (import guarded) with graceful fallback to sklearn.

2. **Torch probe hyperparameters** — Expose `lr`, `epochs`, `batch_size` via YAML `probe.*` , add `probe.torch_lr`, `probe.torch_epochs` to [pipeline.yaml](conf/base/pipeline.yaml).

3. **Statistical significance threshold** — p-value threshold for amnesic drop configurable, default 0.05. (add it to YAML config)

4. **Benchmark Exact vs. Approximate Solvers** — When dataset size allows (fits in VRAM), compute *both* the `cuml` (exact analytical/quasi-Newton) solution and the `pytorch` (SGD/Adam) solution. Log the discrepancy in accuracy and coefficients. This validates that the HPC `TorchLogisticProbe` is converging correctly and not underfitting compared to the mathematical "gold standard" of the exact solver.

**Metrics (`metrics.py`):**
*   **`solver_accuracy_delta`**: The absolute difference between cuML (exact) accuracy and PyTorch (SGD) accuracy. A delta $> 1.0\%$ triggers a warning that the SGD probe has not converged or is misconfigured.
*   **`weight_cosine_similarity`**: The cosine similarity between the hyperplane normal vector (weights) found by `cuml` and the one found by `torch`. If this is close to 1.0, both solvers found the same decision boundary.
*   **`time_to_convergence_ratio`**: Ratio of wall-clock time taken by Torch (to reach within 1% of optimal) vs. cuML. This quantifies the "cost" of using the scalable backend.

**Plots (`visualizations_phase_a.py`):**
*   **`plot_solver_convergence_benchmark()`**: A line chart showing the PyTorch probe's validation accuracy over *epochs* (x-axis), with a solid horizontal line representing the final static accuracy achieved by the `cuml` exact solver. This visualizes if and when the SGD probe matches the analytical solution.
*   **`plot_weight_correlation_scatter()`**: A scatter plot where the X-axis is the weight coefficients of the `cuml` model and the Y-axis is the weights of the `torch` model. Perfect alignment follows the $y=x$ diagonal, proving the approximate solver learned the same features.

5. **Class Imbalance Handling** — Stylometric and PII attributes are often sparse (e.g., specific entities might appear in only 5% of documents). Ensure the `k-fold` strategy is strictly stratified and consider implementing `WeightedRandomSampler` or `pos_weight` in the `BCEWithLogitsLoss` for the Torch backend to prevent the probe from simply learning the majority class bias.

**Metrics (`metrics.py`):**
*   **`balanced_accuracy_drop`**: Instead of standard accuracy, compute the arithmetic mean of sensitivity (true positive rate) and specificity (true negative rate). This prevents the "Amnesic Drop" from looking impressive just because the probe started guessing the majority class.
*   **`minority_class_f1_delta`**: The specific drop in F1-score for the least frequent class. This is the strictest test of whether the pollution (which is often sparse) was actually removed.
*   **`majority_baseline_gap`**: The difference between the Probe's accuracy and a "ZeroR" baseline (always predicting the majority class). If this gap reaches 0, the pollution is effectively fully removed.

**Plots (`visualizations_phase_a.py`):**
*   **`plot_stratified_confusion_matrices()`**: A 2x2 grid of heatmaps:
    *   Top-Left: Before Masking (Standard)
    *   Top-Right: Before Masking (Normalized by True Label - showing Recall)
    *   Bottom-Left: After Masking (Standard)
    *   Bottom-Right: After Masking (Normalized by True Label)
    *   *Why:* Normalized matrices reveal if the "drop" in accuracy is actually just the model collapsing into predicting the majority class (a common failure mode in unlearning).

6. **Control Probes (Specificity Check)** — Implement an optional "Control Probe" that targets a feature which *should not* change (e.g., document sentiment or length bucket). If the amnesic drop is high for the target variable but also high for the control variable, it indicates the masking strategy is destroying the embedding's general semantic quality rather than surgically removing the specific pollution.

**Metrics (`metrics.py`):**
*   **`control_stability_score`**: The accuracy retention of the control probe (e.g., Sentiment or Length). Calculated as $1.0 - (\text{Acc}_{control\_before} - \text{Acc}_{control\_after})$. High is good.
*   **`specificity_ratio`**: A derived ratio: $\frac{\text{Amnesic Drop (Target)}}{\text{Amnesic Drop (Control) + \epsilon}}$. A ratio $>1$ indicates the masking is attacking the PII/Style, not the general embedding quality. A ratio $\approx 1$ implies the masker is just adding random noise.

**Plots (`visualizations_phase_a.py`):**
*   **`plot_specificity_gap()`**: A grouped bar chart.
    *   X-Axis: Probe Targets (e.g., "Gender [Target]", "Sentiment [Control]").
    *   Y-Axis: Accuracy.
    *   Bars: "Before" (Blue) vs. "After" (Red).
    *   *Visual Goal:* We want to see a massive drop for the Target bars, but equal heights for the Control bars.
*   **`plot_selectivity_frontier()`**: A scatter plot where each point is a different masking strategy (or layer choice).
    *   X-Axis: Control Task Performance (General Utility).
    *   Y-Axis: Target Task Amnesia (Privacy/Safety).
    *   *Optimal:* Points in the top-right corner (High Utility, High Amnesia).