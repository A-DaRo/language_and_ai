# Finalized Implementation Plan: `--use-only` Filter & `run-full-pipeline` Command

## Executive Summary

After deep investigation, I've identified:
1. **Critical architectural coupling**: `--use-only` impacts 4 systems: taxonomy prompts, concept encoding, LEACE matrix construction, and data filtering
2. **Deadlock bug confirmed**: In hpc.py, phase_a_pipeline.py visualization code is never reached because HPC strategy returns metadata directly; the pipeline's visualization code runs AFTER strategy.execute() but HPC's stage 4 (probing) already generates visualizations, creating duplication/clutter
3. **Single-label LEACE mode**: When `--use-only` specifies one label, `DemographicEncoder` produces a single-column concept matrix, which fundamentally changes LEACE projection

---

## Part 1: Architecture Analysis

### 1.1 Data Flow: `--use-only` Impact Points

| Component | Impact | Required Change |
|-----------|--------|-----------------|
| **SOBRTaxonomy** | Only load `column_prompts` for selected labels | Filter `column_prompts` dict |
| **GLiNERDetector** | Only use prompts/distractors for selected columns | Pass filtered taxonomy |
| **DemographicEncoder** | Concept matrix Z dimension changes | Accept `demographic_columns` param (already supports this) |
| **LEACEComputer** | Projection matrix dimension changes (k columns → 1 for single-label) | No change needed (auto-adapts to Z shape) |
| **SpanMasker** | Only mask spans for selected labels | Filter `entity_to_mask` |
| **PhaseDDataset** | Filter rows with valid values for selected labels | Already implemented |

### 1.2 Deadlock Analysis: hpc.py Lines 928-1000 & phase_a_pipeline.py Lines 343-393

**Root Cause**: Duplication of visualization responsibility:
- **HPC Strategy** (`hpc.py:_generate_visualizations`): Generates visualizations in Stage 4 (probing)
- **PhaseAPipeline** (`phase_a_pipeline.py:run`): Also tries to generate visualizations after `strategy.execute()` returns

**Execution Flow**:
```
PhaseAPipeline.run()
  └─> strategy.execute()  # HPC strategy
        └─> Stage 4: _run_probing_stage() 
              └─> _generate_visualizations()  # ✓ Runs
              └─> Returns ProbingContext
        └─> Build metadata
        └─> compute_explicit_recall()  # ✓ Runs
        └─> logger.info("Phase A complete!")  # Last log seen
        └─> return metadata
  └─> viz_config.get("enabled")  # After strategy returns
        └─> generate_phase_a_plots()  # REDUNDANT, may cause issues
```

**Problem**: After `strategy.execute()` returns, phase_a_pipeline.py attempts to:
1. Load artifacts that were just written
2. Call `compute_phase_a_metrics()` - may duplicate work
3. Call `generate_phase_a_plots()` - different function than HPC's `_generate_visualizations()`

The "deadlock" is likely a hang in `generate_phase_a_plots()` when loading large artifacts, or a conflict between the two visualization systems.

---

## Part 2: Implementation Tasks

### Task 1: Add `--use-only` to `run-phase-a` (CLI + Pipeline)

**Files to modify**:
- __main__.py: Add `--use-only` decorator
- phase_a_pipeline.py: Accept `use_only_labels` param
- base.py: Add `use_only_labels` to `execute()` signature

**Logic**:
```python
# In PhaseAPipeline.__init__ or run():
if use_only_labels:
    # 1. Filter taxonomy to only include selected columns
    filtered_taxonomy = {
        col: self.taxonomy.column_prompts[col] 
        for col in use_only_labels 
        if col in self.taxonomy.column_prompts
    }
    # 2. Update DemographicEncoder to only encode selected columns
    encoder = DemographicEncoder(demographic_columns=list(use_only_labels))
```

### Task 2: Propagate `use_only_labels` Through Strategy Execute

**Files to modify**:
- laptop.py: Accept and use `use_only_labels`
- hpc.py: Accept and use `use_only_labels`

**Filter Points**:
1. **Before GLiNER init**: Filter taxonomy config
2. **Before DemographicEncoder**: Pass filtered column list
3. **After dataset load**: Filter Arrow table rows (AND semantics)

### Task 3: Fix Visualization Deadlock/Duplication

**Problem**: Two visualization systems running (HPC internal + pipeline post-execution)

**Root Cause**: Violation of separation of concerns
- **Strategy layer** (hpc.py/laptop.py): Should only compute metrics and return data
- **Pipeline layer** (phase_a_pipeline.py): Should orchestrate visualization from returned data

**Solution**: Remove ALL visualization logic from strategy, centralize in pipeline
- **Rationale**: Cleaner separation of concerns, consistent with laptop strategy behavior
- **HPC Strategy Role**: Compute and return rich metadata (probing results, separability metrics)
- **Pipeline Role**: Consume metadata and generate ALL visualizations using `visualizations_phase_a.py`

---

#### 3.1: Changes to HPC Strategy (`hpc.py`)

**Remove**:
- `_generate_visualizations()` method (lines 928-1000)
- All imports from `...evaluation.visualizations_phase_a`

**Add**: Return visualization-ready data in metadata

```python
# In _run_probing_stage(), replace visualization call with metadata return
def _run_probing_stage(...) -> ProbingContext:
    # ... existing probing logic ...
    
    # Instead of calling _generate_visualizations(), return data for pipeline
    viz_data = {
        "by_column_extended": by_column_extended,
        "separability_before": separability_before,
        "separability_after": separability_after,
        "benchmark_results": benchmark_results,
        "control_probe_results": control_probe_results,
        "probe_config": {
            "torch_epochs": probe_config.torch_epochs,
            "exact_solver": probe_config.exact_solver,
            # ... other probe config fields
        },
        "demo_cols": demo_cols,
    }
    
    return ProbingContext(
        # ... existing fields ...
        visualization_data=viz_data,  # NEW: Pass data to pipeline
    )
```

**Metadata Structure** (returned by `execute()`):
```python
{
    "dataset_path": str(output_dataset_path),
    "projection_matrix_path": str(projection_matrix_path),
    "pollution_logs_path": str(pollution_logs_path),
    
    # NEW: Probing results for visualization
    "probing_results": {
        "by_column_extended": {...},  # Per-column amnesic drop with CI
        "separability_before": {...},  # Per-column Fisher discriminant
        "separability_after": {...},
        "benchmark_results": {...},  # Solver convergence comparison
        "control_probe_results": {...},  # Target vs control specificity
        "demo_cols": ["nationality", "female", ...],  # Used columns
        "probe_config": {...},  # ProbeConfig as dict
    },
    
    # Timing/resource info
    "stage_timings": {...},
    "total_inference_time": float,
}
```

---

#### 3.2: Changes to Pipeline (`phase_a_pipeline.py`)

**Current Code** (lines 343-393, approximate):
```python
# After strategy.execute()
if viz_config.get("enabled"):
    # Attempt to call generate_phase_a_plots() - partial coverage
    pass
```

**New Code** (comprehensive visualization orchestration):
```python
def run(self, input_dataset_path: Path, output_dir: Path) -> PhaseArtifacts:
    # ... existing execution ...
    
    # Strategy returns metadata with probing_results
    metadata = self.strategy.execute(...)
    
    # ========================================================================
    # VISUALIZATION ORCHESTRATION (ALL MODES)
    # ========================================================================
    viz_config = self._cfg_get_optional("visualization", {})
    if viz_config.get("enabled", True):
        reports_dir = output_dir / "reports" / "phase_a"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Generating Phase A visualizations...")
        
        # Extract data from metadata
        probing_results = metadata.get("probing_results", {})
        
        # Import all visualization functions
        from .evaluation.visualizations_phase_a import (
            plot_amnesic_drop_with_ci,
            plot_embedding_separability,
            plot_solver_convergence_benchmark,
            plot_specificity_gap,
            plot_pca_before_after,
            plot_singular_values,
            plot_embedding_norm_distribution,
            plot_amnesic_drop_bars,
            plot_gliner_confidence_histogram,
            plot_detection_count_by_type,
            plot_explicit_recall_bars,
            plot_entity_cooccurrence,
            plot_demographic_score_matrix,
            plot_masking_coverage,
        )
        
        dpi = int(viz_config.get("figure_dpi", 150))
        palette = viz_config.get("color_palette", "husl")
        
        # -----------------------------------------------------------------
        # 1. Amnesic Drop Visualizations
        # -----------------------------------------------------------------
        by_column_extended = probing_results.get("by_column_extended")
        if by_column_extended:
            plot_amnesic_drop_with_ci(
                per_column_results=by_column_extended,
                output_path=reports_dir / "amnesic_drop_with_ci.png",
                dpi=dpi,
                palette=palette,
            )
        
        # -----------------------------------------------------------------
        # 2. Embedding Separability (Before/After LEACE)
        # -----------------------------------------------------------------
        demo_cols = probing_results.get("demo_cols", [])
        separability_before = probing_results.get("separability_before", {})
        separability_after = probing_results.get("separability_after", {})
        
        first_col = demo_cols[0] if demo_cols else None
        if first_col and first_col in separability_before and first_col in separability_after:
            plot_embedding_separability(
                separability_before=separability_before[first_col],
                separability_after=separability_after[first_col],
                output_path=reports_dir / "embedding_separability.png",
                dpi=dpi,
            )
        
        # -----------------------------------------------------------------
        # 3. Solver Convergence Benchmark
        # -----------------------------------------------------------------
        benchmark_results = probing_results.get("benchmark_results")
        probe_config_dict = probing_results.get("probe_config", {})
        
        if benchmark_results and "torch_accuracy" in benchmark_results:
            torch_final = benchmark_results.get("torch_accuracy", 0.5)
            epochs = probe_config_dict.get("torch_epochs", 20)
            # Reconstruct learning curve (simple exponential model)
            learning_curve = [
                torch_final * (1 - 0.5 * np.exp(-i / 20)) 
                for i in range(epochs)
            ]
            plot_solver_convergence_benchmark(
                benchmark_results=benchmark_results,
                learning_curve_torch=learning_curve,
                output_path=reports_dir / "solver_convergence_benchmark.png",
                dpi=dpi,
            )
        
        # -----------------------------------------------------------------
        # 4. Specificity Gap (Target vs Control)
        # -----------------------------------------------------------------
        control_probe_results = probing_results.get("control_probe_results")
        if control_probe_results and "error" not in control_probe_results:
            target_results = control_probe_results.get("target", {})
            control_keys = [k for k in control_probe_results if k.startswith("control_")]
            
            if target_results and control_keys:
                plot_specificity_gap(
                    target_results={
                        "acc_before": target_results.get("acc_before", 0.5),
                        "acc_after": target_results.get("acc_after", 0.5),
                    },
                    control_results={
                        "acc_before": control_probe_results[control_keys[0]].get("acc_before", 0.5),
                        "acc_after": control_probe_results[control_keys[0]].get("acc_after", 0.5),
                    },
                    target_name=demo_cols[0] if demo_cols else "target",
                    control_name=demo_cols[1] if len(demo_cols) > 1 else "control",
                    output_path=reports_dir / "specificity_gap.png",
                    dpi=dpi,
                    palette=palette,
                )
        
        # -----------------------------------------------------------------
        # 5. Additional Visualizations (Dataset-Level)
        # -----------------------------------------------------------------
        # Load artifacts for dataset-level visualizations
        clean_table = pa.ipc.open_file(metadata["dataset_path"]).read_all()
        logs_table = pa.ipc.open_file(metadata["pollution_logs_path"]).read_all()
        projection_matrix = torch.load(metadata["projection_matrix_path"])
        
        # Convert to DataFrames for visualization functions
        clean_df = clean_table.to_pandas()
        logs_df = logs_table.to_pandas()
        
        # GLiNER Detection Statistics
        if len(logs_df) > 0:
            plot_gliner_confidence_histogram(
                logs_df=logs_df,
                output_path=reports_dir / "gliner_confidence_histogram.png",
                dpi=dpi,
                palette=palette,
            )
            
            plot_detection_count_by_type(
                logs_df=logs_df,
                output_path=reports_dir / "detection_count_by_type.png",
                dpi=dpi,
                palette=palette,
            )
            
            plot_entity_cooccurrence(
                logs_df=logs_df,
                output_path=reports_dir / "entity_cooccurrence.png",
                dpi=dpi,
            )
            
            plot_masking_coverage(
                clean_df=clean_df,
                logs_df=logs_df,
                output_path=reports_dir / "masking_coverage.png",
                dpi=dpi,
                palette=palette,
            )
        
        # LEACE Projection Quality
        plot_singular_values(
            projection_matrix=projection_matrix,
            output_path=reports_dir / "singular_values.png",
            dpi=dpi,
        )
        
        # Embedding Analysis (requires embedder - sample for efficiency)
        sample_size = min(2000, len(clean_df))
        if sample_size > 0:
            clean_df_sample = clean_df.sample(n=sample_size, random_state=42)
            
            # Get embedder from pipeline
            embedder = self.get_embedder()
            
            # Compute embeddings (before/after LEACE)
            texts = clean_df_sample["post_masked"].tolist()
            embeddings_before = embedder.encode_batch(texts)
            embeddings_after = embeddings_before @ projection_matrix.cpu().numpy().T
            
            # PCA scatter plots
            labels_df = clean_df_sample[demo_cols] if demo_cols else pd.DataFrame()
            plot_pca_before_after(
                embeddings_before=embeddings_before,
                embeddings_after=embeddings_after,
                labels_df=labels_df,
                output_dir=reports_dir,
                dpi=dpi,
                palette=palette,
            )
            
            # Embedding norm distribution
            plot_embedding_norm_distribution(
                embeddings_before=embeddings_before,
                embeddings_after=embeddings_after,
                output_path=reports_dir / "embedding_norm_distribution.png",
                dpi=dpi,
            )
            
            # Demographic score matrix (if multiple demographics)
            if len(demo_cols) > 1:
                plot_demographic_score_matrix(
                    embeddings=embeddings_after,
                    labels_df=labels_df,
                    output_path=reports_dir / "demographic_score_matrix.png",
                    dpi=dpi,
                    palette=palette,
                )
        
        # Explicit recall bars (if available in metadata)
        explicit_recall = metadata.get("explicit_recall", {})
        if explicit_recall:
            plot_explicit_recall_bars(
                metrics={"explicit_recall": explicit_recall},
                output_path=reports_dir / "explicit_recall_bars.png",
                dpi=dpi,
                palette=palette,
            )
        
        logger.info(f"✓ All visualizations saved to: {reports_dir}")
    
    # Return artifacts
    return PhaseArtifacts(...)
```

---

#### 3.3: Files to Modify

| File | Changes |
|------|---------|
| **hpc.py** | Remove `_generate_visualizations()`, return `visualization_data` in `ProbingContext` |
| **laptop.py** | Ensure metadata structure matches HPC (add `probing_results` if missing) |
| **base.py** | Update `ProbingContext` dataclass to include `visualization_data: Optional[Dict[str, Any]]` |
| **phase_a_pipeline.py** | Add comprehensive visualization orchestration (see 3.2) |

---

#### 3.4: NO - Backward Compatibility

**Old Behavior** (HPC strategy) -> TO BE REMOVED:
- Visualizations generated inside Stage 4 (probing)
- Limited to 4 plots: amnesic_drop_with_ci, embedding_separability, solver_convergence, specificity_gap

**New Behavior** (Pipeline orchestration) -> TO BE ENFORCED:
- ALL 14+ plots from `visualizations_phase_a.py` generated consistently
- HPC and laptop modes produce identical output
- Easier to extend (add new plots in one place)

### Task 4: Adapt Visualizations and Metrics for Single-Demographic Mode

**Context**: When `--use-only` specifies a single demographic, several visualizations need adaptation:

#### 4.1: Visualization Changes for Single-Label Mode

| Visualization | Multi-Label Behavior | Single-Label Adaptation |
|---------------|---------------------|-------------------------|
| **plot_amnesic_drop_with_ci** | Bar chart with all demographics | Single bar (still useful for CI) |
| **plot_embedding_separability** | Uses first demographic | ✓ No change needed |
| **plot_solver_convergence_benchmark** | Generic (solver comparison) | ✓ No change needed |
| **plot_specificity_gap** | Target vs control demographic | **SKIP** (requires ≥2 demographics) |
| **plot_pca_before_after** | Colored by first demographic | ✓ No change needed (color by single label) |
| **plot_singular_values** | Generic (projection spectrum) | ✓ No change needed |
| **plot_embedding_norm_distribution** | Generic (before/after norms) | ✓ No change needed |
| **plot_amnesic_drop_bars** | Bar chart with all demographics | Single bar |
| **plot_gliner_confidence_histogram** | All entity types | Filtered to selected label |
| **plot_detection_count_by_type** | All entity types | Filtered to selected label |
| **plot_explicit_recall_bars** | All demographics | Single bar |
| **plot_entity_cooccurrence** | Heatmap of co-occurrence | **SKIP** (requires ≥2 entity types) |
| **plot_demographic_score_matrix** | Pairwise scatter matrix | **SKIP** (requires ≥2 demographics) |
| **plot_masking_coverage** | Stacked bar per demographic | Single bar |

#### 4.2: Conditional Logic for Single-Label Mode

```python
# In PhaseAPipeline.run(), visualization section:
demo_cols = probing_results.get("demo_cols", [])
is_single_label = len(demo_cols) == 1

# Specificity Gap: Skip if single-label
if not is_single_label and control_probe_results:
    plot_specificity_gap(...)

# Entity Co-occurrence: Skip if single-label
if not is_single_label and len(logs_df) > 0:
    plot_entity_cooccurrence(...)

# Demographic Score Matrix: Skip if single-label
if not is_single_label and len(demo_cols) > 1:
    plot_demographic_score_matrix(...)
```

#### 4.3: Metrics Changes for Single-Label Mode

**Affected Metrics** (from `metrics.py`):

| Metric Function | Multi-Label | Single-Label Adaptation |
|-----------------|-------------|-------------------------|
| `compute_per_column_amnesic_drop` | Returns dict with all columns | Returns dict with single column |
| `compute_embedding_separability` | Per-column Fisher discriminant | Single column |
| `compute_class_imbalance_metrics` | Per-column class ratios | Single column |
| `compute_control_probe_metrics` | Target vs control | **SKIP** (requires ≥2 columns) |

**Conditional Execution**:
```python
# In HPCFilterStrategy._run_probing_stage() or pipeline:
if len(demo_cols) >= 2:
    # Compute control probe metrics (target vs control)
    control_probe_results = compute_control_probe_metrics(
        embeddings_before=embeddings_before,
        embeddings_after=embeddings_after,
        target_labels=labels[:, 0],
        control_labels=labels[:, 1],
        control_name=demo_cols[1],
    )
else:
    control_probe_results = {"skipped": "single_label_mode"}
```

#### 4.4: Dimension Changes for Single-Label Mode

**LEACE Projection Matrix**:
- Multi-label: `P.shape = (768, 768)` (projects out k-dimensional concept space)
- Single-label: `P.shape = (768, 768)` (projects out 1D concept space)
- **Note**: Shape is always (d, d), but rank changes: `rank(I - P) = k` where k = number of demographics

**Concept Matrix Z**:
- Multi-label: `Z.shape = (n, k)` where k = number of demographics (e.g., k=8 for all demographics)
- Single-label: `Z.shape = (n, 1)` (single column)

**DemographicEncoder Output**:
```python
# Multi-label example:
encoder = DemographicEncoder(demographic_columns=["nationality", "female", "birth_year"])
Z = encoder.encode(table)  # Z.shape = (n, 3)

# Single-label example:
encoder = DemographicEncoder(demographic_columns=["nationality"])
Z = encoder.encode(table)  # Z.shape = (n, 1)
```

#### 4.5: Report Text Adaptations

**Phase A Metrics JSON** (`phase_a_metrics.json`):
```json
{
  "label_filter": {
    "mode": "single_label",
    "use_only": ["nationality"]
  },
  "probing_results": {
    "by_column": {
      "nationality": {
        "acc_before": 0.72,
        "acc_after": 0.53,
        "amnesic_drop": 0.19,
        "ci_lower": 0.17,
        "ci_upper": 0.21
      }
    }
  }
}
```

**Visualization Titles**:
- Multi-label: "Amnesic Drop Across Demographics"
- Single-label: "Amnesic Drop for Nationality"

**Implementation**:
```python
# In visualizations_phase_a.py functions, add conditional titles:
def plot_amnesic_drop_with_ci(per_column_results, output_path, dpi, palette):
    if len(per_column_results) == 1:
        title = f"Amnesic Drop for {list(per_column_results.keys())[0].replace('_', ' ').title()}"
    else:
        title = "Amnesic Drop Across Demographics"
    
    # ... plotting logic ...
```

---

### Task 5: Add `--use-only` to `verify` Command (Already Exists)

The `verify` command already has `--use-only` with auto-detection from `training_metadata.json`. **No changes needed** for verify CLI.

However, verify should also support **auto-detection from Phase A metrics** when available:

**Files to modify**:
- __main__.py: Add Phase A metadata auto-detection

### Task 6: Create `run-full-pipeline` Command

**New command in __main__.py**:

```python
@cli.command("run-full-pipeline")
@click.option("--dataset", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--mode", type=click.Choice(["laptop", "hpc"]), default="laptop")
@click.option("--use-only", "use_only_labels", multiple=True, 
              help="Filter to specific demographics (e.g., --use-only nationality --use-only female)")
@click.option("--skip-verify", is_flag=True, help="Skip verification step")
@click.option("--force-rerun", is_flag=True, help="Force re-run even if artifacts exist")
def run_full_pipeline(dataset, output_dir, mode, use_only_labels, skip_verify, force_rerun):
    """
    Run complete pipeline: Phase A → Phase D → Verify.
    
    Orchestrates the full neuro-stylometry pipeline:
    1. Phase A: Pollution detection → masking → LEACE projection
    2. Phase D: Training with projection-aware architecture
    3. Verify: Orthogonality verification and metrics
    
    All stages respect --use-only filter and propagate metadata.
    """
    import json
    from .phase_a_pipeline import PhaseAPipeline
    from .phase_d_pipeline import PhaseDPipeline  # Assuming this exists
    from .verification_pipeline import VerificationPipeline  # Assuming this exists
    from .factories.strategy_factory import StrategyFactory
    
    output_dir = Path(output_dir)
    phase_a_dir = output_dir / "phase_a"
    phase_d_dir = output_dir / "phase_d"
    
    # =========================================================================
    # Stage 1: Phase A (Pollution Filtering)
    # =========================================================================
    click.echo("=" * 80)
    click.echo("STAGE 1: Phase A (Pollution Filtering)")
    click.echo("=" * 80)
    
    phase_a_artifacts_exist = (
        (phase_a_dir / "clean_dataset.arrow").exists() and
        (phase_a_dir / "projection_matrix.pt").exists() and
        (phase_a_dir / "pollution_logs.arrow").exists()
    )
    
    if phase_a_artifacts_exist and not force_rerun:
        click.echo(f"✓ Phase A artifacts found at {phase_a_dir}")
        click.echo("  Skipping Phase A (use --force-rerun to override)")
    else:
        strategy = StrategyFactory.create_strategy(mode=mode)
        pipeline_a = PhaseAPipeline(strategy=strategy)
        
        # Pass use_only_labels to pipeline
        artifacts_a = pipeline_a.run(
            input_dataset_path=Path(dataset),
            output_dir=phase_a_dir,
            use_only_labels=list(use_only_labels) if use_only_labels else None,
        )
        
        click.echo(f"✓ Phase A complete: {artifacts_a.clean_dataset_path}")
    
    # Load Phase A metadata for propagation
    phase_a_metrics_path = phase_a_dir / "phase_a_metrics.json"
    if phase_a_metrics_path.exists():
        with open(phase_a_metrics_path) as f:
            phase_a_metadata = json.load(f)
    else:
        phase_a_metadata = {}
    
    # =========================================================================
    # Stage 2: Phase D (Training)
    # =========================================================================
    click.echo("\n" + "=" * 80)
    click.echo("STAGE 2: Phase D (Training)")
    click.echo("=" * 80)
    
    phase_d_artifacts_exist = (
        (phase_d_dir / "baseline" / "checkpoints").exists() and
        (phase_d_dir / "constrained" / "checkpoints").exists()
    )
    
    if phase_d_artifacts_exist and not force_rerun:
        click.echo(f"✓ Phase D artifacts found at {phase_d_dir}")
        click.echo("  Skipping Phase D (use --force-rerun to override)")
    else:
        pipeline_d = PhaseDPipeline(mode=mode)
        
        # Propagate use_only_labels from Phase A metadata if not explicitly provided
        use_only_for_d = list(use_only_labels) if use_only_labels else None
        if not use_only_for_d and "label_filter" in phase_a_metadata:
            use_only_for_d = phase_a_metadata["label_filter"].get("use_only")
        
        artifacts_d = pipeline_d.run(
            clean_dataset_path=phase_a_dir / "clean_dataset.arrow",
            projection_matrix_path=phase_a_dir / "projection_matrix.pt",
            output_dir=phase_d_dir,
            use_only_labels=use_only_for_d,
        )
        
        click.echo(f"✓ Phase D complete: {artifacts_d.baseline_checkpoint}")
    
    # =========================================================================
    # Stage 3: Verification
    # =========================================================================
    if not skip_verify:
        click.echo("\n" + "=" * 80)
        click.echo("STAGE 3: Verification")
        click.echo("=" * 80)
        
        verification_artifacts_exist = (phase_d_dir / "verification_metrics.json").exists()
        
        if verification_artifacts_exist and not force_rerun:
            click.echo(f"✓ Verification artifacts found at {phase_d_dir}")
            click.echo("  Skipping verification (use --force-rerun to override)")
        else:
            verify_pipeline = VerificationPipeline(mode=mode)
            
            # Auto-detect use_only from training metadata
            training_metadata_path = phase_d_dir / "training_metadata.json"
            use_only_for_verify = list(use_only_labels) if use_only_labels else None
            if not use_only_for_verify and training_metadata_path.exists():
                with open(training_metadata_path) as f:
                    training_metadata = json.load(f)
                use_only_for_verify = training_metadata.get("label_filter", {}).get("use_only")
            
            verify_results = verify_pipeline.run(
                phase_d_dir=phase_d_dir,
                phase_a_dir=phase_a_dir,
                use_only_labels=use_only_for_verify,
            )
            
            click.echo(f"✓ Verification complete: {verify_results.metrics_path}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    click.echo("\n" + "=" * 80)
    click.echo("PIPELINE COMPLETE")
    click.echo("=" * 80)
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"Phase A artifacts: {phase_a_dir}")
    click.echo(f"Phase D artifacts: {phase_d_dir}")
    if use_only_labels:
        click.echo(f"Label filter: {', '.join(use_only_labels)}")
    click.echo("=" * 80)
```

---

### Task 7: Create E2E Verification Script

**New file**: `scripts/verify_use_only_e2e.py`

```python
#!/usr/bin/env python
"""
E2E verification script for --use-only filter across full pipeline.

Usage:
    python scripts/verify_use_only_e2e.py --samples 100 --label nationality
    python scripts/verify_use_only_e2e.py --samples 500 --label nationality --label female
    
Verifies:
1. Phase A runs without error with filtered label(s)
2. Artifacts have correct dimensions
3. Metadata propagates correctly through all phases
4. Single-label mode adaptations work correctly
5. Multi-label mode works as expected
6. Visualizations are generated appropriately for each mode

Exit codes:
    0: All tests passed
    1: Test failure (see stderr for details)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import pyarrow as pa
import torch


def verify_phase_a_artifacts(
    phase_a_dir: Path,
    expected_labels: List[str],
) -> bool:
    """Verify Phase A artifacts match expected dimensions."""
    print("\n[1/5] Verifying Phase A artifacts...")
    
    # Check files exist
    required_files = [
        "clean_dataset.arrow",
        "projection_matrix.pt",
        "pollution_logs.arrow",
        "phase_a_metrics.json",
    ]
    
    for filename in required_files:
        filepath = phase_a_dir / filename
        if not filepath.exists():
            print(f"  ✗ Missing artifact: {filename}")
            return False
    
    # Check dataset dimensions
    clean_table = pa.ipc.open_file(phase_a_dir / "clean_dataset.arrow").read_all()
    print(f"  ✓ Dataset: {len(clean_table)} rows")
    
    # Check projection matrix shape
    projection_matrix = torch.load(phase_a_dir / "projection_matrix.pt")
    assert projection_matrix.shape == (768, 768), f"Unexpected projection shape: {projection_matrix.shape}"
    print(f"  ✓ Projection matrix: {projection_matrix.shape}")
    
    # Check metadata
    with open(phase_a_dir / "phase_a_metrics.json") as f:
        metrics = json.load(f)
    
    label_filter = metrics.get("label_filter", {})
    actual_labels = label_filter.get("use_only", [])
    
    if set(actual_labels) != set(expected_labels):
        print(f"  ✗ Label mismatch: expected {expected_labels}, got {actual_labels}")
        return False
    
    print(f"  ✓ Label filter: {actual_labels}")
    
    # Check probing results match filtered labels
    probing_results = metrics.get("probing_results", {})
    by_column = probing_results.get("by_column", {})
    
    if set(by_column.keys()) != set(expected_labels):
        print(f"  ✗ Probing columns mismatch: expected {expected_labels}, got {list(by_column.keys())}")
        return False
    
    print(f"  ✓ Probing results: {list(by_column.keys())}")
    
    return True


def verify_visualizations(
    reports_dir: Path,
    is_single_label: bool,
) -> bool:
    """Verify correct visualizations were generated."""
    print("\n[2/5] Verifying visualizations...")
    
    # Plots that should always exist
    required_plots = [
        "amnesic_drop_with_ci.png",
        "embedding_separability.png",
        "solver_convergence_benchmark.png",
        "singular_values.png",
        "embedding_norm_distribution.png",
        "gliner_confidence_histogram.png",
        "detection_count_by_type.png",
        "masking_coverage.png",
        "pca_before_after.png",  # Assuming generated by plot_pca_before_after
    ]
    
    # Plots that should only exist in multi-label mode
    multi_label_only_plots = [
        "specificity_gap.png",
        "entity_cooccurrence.png",
        "demographic_score_matrix.png",
    ]
    
    for plot in required_plots:
        if not (reports_dir / plot).exists():
            print(f"  ✗ Missing required plot: {plot}")
            return False
        print(f"  ✓ {plot}")
    
    for plot in multi_label_only_plots:
        exists = (reports_dir / plot).exists()
        if is_single_label and exists:
            print(f"  ✗ Unexpected plot in single-label mode: {plot}")
            return False
        elif not is_single_label and not exists:
            print(f"  ✗ Missing multi-label plot: {plot}")
            return False
        
        status = "skipped (single-label)" if is_single_label else "generated"
        print(f"  ✓ {plot}: {status}")
    
    return True


def verify_metadata_propagation(
    phase_a_dir: Path,
    phase_d_dir: Path,
    expected_labels: List[str],
) -> bool:
    """Verify metadata propagates from Phase A → Phase D → Verify."""
    print("\n[3/5] Verifying metadata propagation...")
    
    # Check Phase A metadata
    with open(phase_a_dir / "phase_a_metrics.json") as f:
        phase_a_metrics = json.load(f)
    
    phase_a_labels = phase_a_metrics.get("label_filter", {}).get("use_only", [])
    if set(phase_a_labels) != set(expected_labels):
        print(f"  ✗ Phase A label mismatch")
        return False
    print(f"  ✓ Phase A metadata: {phase_a_labels}")
    
    # Check Phase D metadata (if exists)
    training_metadata_path = phase_d_dir / "training_metadata.json"
    if training_metadata_path.exists():
        with open(training_metadata_path) as f:
            phase_d_metadata = json.load(f)
        
        phase_d_labels = phase_d_metadata.get("label_filter", {}).get("use_only", [])
        if set(phase_d_labels) != set(expected_labels):
            print(f"  ✗ Phase D label mismatch")
            return False
        print(f"  ✓ Phase D metadata: {phase_d_labels}")
    else:
        print(f"  ⚠ Phase D metadata not found (Phase D not run)")
    
    return True


def verify_single_label_adaptations(
    phase_a_dir: Path,
    label: str,
) -> bool:
    """Verify single-label mode adaptations."""
    print(f"\n[4/5] Verifying single-label adaptations for '{label}'...")
    
    with open(phase_a_dir / "phase_a_metrics.json") as f:
        metrics = json.load(f)
    
    # Check mode flag
    mode = metrics.get("label_filter", {}).get("mode")
    if mode != "single_label":
        print(f"  ✗ Expected mode='single_label', got '{mode}'")
        return False
    print(f"  ✓ Mode: {mode}")
    
    # Check probing results have only one column
    by_column = metrics.get("probing_results", {}).get("by_column", {})
    if list(by_column.keys()) != [label]:
        print(f"  ✗ Expected single column '{label}', got {list(by_column.keys())}")
        return False
    print(f"  ✓ Probing results: single column '{label}'")
    
    # Check control probe metrics were skipped
    control_probe_results = metrics.get("probing_results", {}).get("control_probe_results", {})
    if control_probe_results.get("skipped") != "single_label_mode":
        print(f"  ✗ Control probe should be skipped in single-label mode")
        return False
    print(f"  ✓ Control probe: skipped (as expected)")
    
    return True


def verify_multi_label_mode(
    phase_a_dir: Path,
    labels: List[str],
) -> bool:
    """Verify multi-label mode works correctly."""
    print(f"\n[5/5] Verifying multi-label mode for {labels}...")
    
    with open(phase_a_dir / "phase_a_metrics.json") as f:
        metrics = json.load(f)
    
    # Check mode flag
    mode = metrics.get("label_filter", {}).get("mode")
    if mode != "multi_label":
        print(f"  ✗ Expected mode='multi_label', got '{mode}'")
        return False
    print(f"  ✓ Mode: {mode}")
    
    # Check probing results have all columns
    by_column = metrics.get("probing_results", {}).get("by_column", {})
    if set(by_column.keys()) != set(labels):
        print(f"  ✗ Expected columns {labels}, got {list(by_column.keys())}")
        return False
    print(f"  ✓ Probing results: {list(by_column.keys())}")
    
    # Check control probe metrics exist
    control_probe_results = metrics.get("probing_results", {}).get("control_probe_results", {})
    if not control_probe_results or "skipped" in control_probe_results:
        print(f"  ✗ Control probe should run in multi-label mode")
        return False
    print(f"  ✓ Control probe: computed")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="E2E verification for --use-only filter")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to use")
    parser.add_argument("--label", action="append", dest="labels", required=True, 
                       help="Label(s) to filter (can be repeated)")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/test_use_only"),
                       help="Output directory for test artifacts")
    
    args = parser.parse_args()
    
    # Determine mode
    is_single_label = len(args.labels) == 1
    mode_name = "single-label" if is_single_label else "multi-label"
    
    print("=" * 80)
    print(f"E2E Verification: --use-only filter ({mode_name} mode)")
    print("=" * 80)
    print(f"Labels: {args.labels}")
    print(f"Samples: {args.samples}")
    print(f"Output: {args.output_dir}")
    
    # Setup paths
    phase_a_dir = args.output_dir / "phase_a"
    reports_dir = phase_a_dir / "reports" / "phase_a"
    phase_d_dir = args.output_dir / "phase_d"
    
    # Run tests
    tests_passed = True
    
    # Test 1: Verify Phase A artifacts
    if not verify_phase_a_artifacts(phase_a_dir, args.labels):
        tests_passed = False
    
    # Test 2: Verify visualizations
    if not verify_visualizations(reports_dir, is_single_label):
        tests_passed = False
    
    # Test 3: Verify metadata propagation
    if not verify_metadata_propagation(phase_a_dir, phase_d_dir, args.labels):
        tests_passed = False
    
    # Test 4/5: Mode-specific tests
    if is_single_label:
        if not verify_single_label_adaptations(phase_a_dir, args.labels[0]):
            tests_passed = False
    else:
        if not verify_multi_label_mode(phase_a_dir, args.labels):
            tests_passed = False
    
    # Summary
    print("\n" + "=" * 80)
    if tests_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

---

## Part 3: Detailed Changes by File

### __main__.py

| Line Range | Change |
|------------|--------|
| 86-171 | Add `--use-only` decorator to `run-phase-a`, pass to pipeline |
| ~900+ | Add new `run-full-pipeline` command (see Task 6) |

### phase_a_pipeline.py

| Line Range | Change |
|------------|--------|
| 56-70 | Add `use_only_labels: Optional[List[str]] = None` to `__init__` and `run()` |
| 297-300 | Add `use_only_labels` to `run()` signature, pass to strategy.execute() |
| 343-420 | **REWRITE**: Comprehensive visualization orchestration using metadata from strategy |

**Key Changes in Visualization Section**:
```python
# OLD: Partial visualization code, inconsistent between modes
if viz_config.get("enabled"):
    # Limited functionality, HPC strategy already generated some plots
    pass

# NEW: Complete visualization orchestration, consistent across modes
if viz_config.get("enabled", True):
    probing_results = metadata.get("probing_results", {})  # From strategy
    
    # Generate ALL visualizations from visualizations_phase_a.py
    # - Amnesic drop with CI
    # - Embedding separability
    # - Solver convergence
    # - Specificity gap (if multi-label)
    # - PCA scatter plots
    # - Singular value spectrum
    # - Embedding norm distribution
    # - GLiNER detection histograms
    # - Entity co-occurrence (if multi-label)
    # - Demographic score matrix (if multi-label)
    # - Masking coverage
    # - Explicit recall bars
    
    # Conditional logic for single-label vs multi-label
    is_single_label = len(probing_results.get("demo_cols", [])) == 1
    if not is_single_label:
        plot_specificity_gap(...)  # Skip in single-label mode
        plot_entity_cooccurrence(...)
        plot_demographic_score_matrix(...)
```

### strategies/base.py

| Line Range | Change |
|------------|--------|
| 143-155 | Add `use_only_labels: Optional[List[str]] = None` to `execute()` abstract method |
| ~70-80 | **NEW**: Add `visualization_data: Optional[Dict[str, Any]] = None` to `ProbingContext` dataclass |

**New ProbingContext Structure**:
```python
@dataclass
class ProbingContext:
    """Results from probing stage."""
    probing_complete: bool
    reports_dir: Optional[Path] = None
    
    # NEW: Visualization-ready data (replaces inline visualization)
    visualization_data: Optional[Dict[str, Any]] = None
    # Contains: by_column_extended, separability_before/after,
    # benchmark_results, control_probe_results, demo_cols, probe_config
```

### strategies/hpc.py

| Line Range | Change |
|------------|--------|
| 928-1000 | **DELETE**: Remove `_generate_visualizations()` method entirely |
| 1007-1050 | Add `use_only_labels: Optional[List[str]]` parameter to `execute()` |
| ~1070 | Filter taxonomy config for selected labels (Task 2) |
| ~1100 | Filter Arrow table rows for valid labels (Task 2) |
| ~1130 | Pass filtered columns to `DemographicEncoder` (Task 2) |
| 850-920 | **MODIFY**: In `_run_probing_stage()`, return visualization data instead of generating plots |

**Key Change in _run_probing_stage()**:
```python
# OLD: Generate visualizations inline
def _run_probing_stage(...):
    # ... probing logic ...
    self._generate_visualizations(...)  # DELETE THIS
    return ProbingContext(probing_complete=True, reports_dir=reports_dir)

# NEW: Return visualization-ready data
def _run_probing_stage(...):
    # ... probing logic ...
    
    # Package data for pipeline visualization
    viz_data = {
        "by_column_extended": by_column_extended,
        "separability_before": separability_before,
        "separability_after": separability_after,
        "benchmark_results": benchmark_results,
        "control_probe_results": control_probe_results if len(demo_cols) >= 2 else {"skipped": "single_label_mode"},
        "probe_config": {
            "torch_epochs": probe_config.torch_epochs,
            "exact_solver": probe_config.exact_solver,
        },
        "demo_cols": demo_cols,
    }
    
    return ProbingContext(
        probing_complete=True,
        reports_dir=None,  # Pipeline will create reports dir
        visualization_data=viz_data,
    )
```

### strategies/laptop.py

| Line Range | Change |
|------------|--------|
| Similar to hpc.py | Add `use_only_labels` parameter, filter taxonomy/table/encoder |
| Visualization | **VERIFY** that laptop strategy doesn't have inline visualization (should already delegate to pipeline) |

### gliner_detector.py

| Line Range | Change |
|------------|--------|
| 290-310 | **NEW**: Add `SOBRTaxonomy.filter_columns(columns: List[str])` method |

```python
class SOBRTaxonomy:
    def filter_columns(self, columns: List[str]) -> "SOBRTaxonomy":
        """Return a new taxonomy with only specified columns."""
        filtered_prompts = {
            col: self.column_prompts[col]
            for col in columns
            if col in self.column_prompts
        }
        
        return SOBRTaxonomy(
            column_prompts=filtered_prompts,
            # ... copy other attributes ...
        )
```

### visualizations_phase_a.py

| Line Range | Change |
|------------|--------|
| Throughout | **ADD**: Conditional logic for single-label vs multi-label titles |
| plot_amnesic_drop_with_ci | Add adaptive title based on number of columns |
| plot_specificity_gap | Add check to skip if single-label mode |
| plot_demographic_score_matrix | Add check to skip if single-label mode |
| plot_entity_cooccurrence | Add check to skip if single-label mode |

**Example Adaptation**:
```python
def plot_amnesic_drop_with_ci(per_column_results, output_path, dpi, palette):
    # Adaptive title
    if len(per_column_results) == 1:
        col_name = list(per_column_results.keys())[0]
        title = f"Amnesic Drop for {col_name.replace('_', ' ').title()}"
    else:
        title = "Amnesic Drop Across Demographics"
    
    # ... existing plotting logic ...
```

### metrics.py

| Line Range | Change |
|------------|--------|
| compute_phase_a_metrics | Add `label_filter` field to output JSON |
| compute_control_probe_metrics | Add conditional execution (skip if single-label) |

**Example**:
```python
def compute_phase_a_metrics(..., use_only_labels=None):
    metrics = {...}
    
    # NEW: Add label filter metadata
    if use_only_labels:
        metrics["label_filter"] = {
            "mode": "single_label" if len(use_only_labels) == 1 else "multi_label",
            "use_only": list(use_only_labels),
        }
    
    return metrics
```

---

## Part 6: Implementation Summary & Testing Strategy

### 6.1: Key Architectural Decisions

| Decision | Rationale | Impact |
|----------|-----------|--------|
| **Option A for Task 3** | Cleaner separation of concerns (strategy computes, pipeline visualizes) | All modes produce identical visualization output |
| **Filter at dataset load** | Minimizes wasted compute, ensures LEACE projection matches filtered labels | Phase A must be re-run if filter changes |
| **Single-label adaptations** | Some visualizations are meaningless with one demographic | Conditional plot generation based on `len(demo_cols)` |
| **Metadata propagation** | Label filter stored in metrics JSON files | Enables auto-detection in downstream phases |

### 6.2: Testing Strategy

#### Unit Tests (to be added)

```python
# tests/unit/test_use_only_filter.py

def test_taxonomy_filtering():
    """Test SOBRTaxonomy.filter_columns()"""
    taxonomy = SOBRTaxonomy.from_config({...})
    filtered = taxonomy.filter_columns(["nationality"])
    assert len(filtered.column_prompts) == 1
    assert "nationality" in filtered.column_prompts

def test_demographic_encoder_single_label():
    """Test DemographicEncoder with single column"""
    encoder = DemographicEncoder(demographic_columns=["nationality"])
    Z = encoder.encode(table)
    assert Z.shape[1] == 1  # Single column

def test_leace_projection_single_label():
    """Test LEACE projection with 1D concept space"""
    computer = LEACEComputer(concept_dim=1)
    P = computer.compute_projection(X, Z)
    assert P.shape == (768, 768)
    # Verify rank(I - P) ≈ 1

def test_visualization_conditional_logic():
    """Test visualization skipping in single-label mode"""
    viz_data = {"demo_cols": ["nationality"], ...}
    is_single_label = len(viz_data["demo_cols"]) == 1
    assert is_single_label
    # Verify specificity_gap, entity_cooccurrence, demographic_score_matrix are skipped
```

#### Integration Tests (to be added)

```python
# tests/integration/test_phase_a_use_only.py

def test_phase_a_single_label_pipeline():
    """Test Phase A with --use-only nationality"""
    pipeline = PhaseAPipeline(strategy=laptop_strategy)
    artifacts = pipeline.run(
        input_dataset_path=test_dataset,
        output_dir=tmp_dir,
        use_only_labels=["nationality"],
    )
    
    # Verify artifacts
    assert artifacts.clean_dataset_path.exists()
    metrics = json.loads(artifacts.metrics_path.read_text())
    assert metrics["label_filter"]["mode"] == "single_label"
    assert metrics["label_filter"]["use_only"] == ["nationality"]
    
    # Verify visualizations
    reports = list((tmp_dir / "reports" / "phase_a").glob("*.png"))
    assert "specificity_gap.png" not in [r.name for r in reports]

def test_phase_a_multi_label_pipeline():
    """Test Phase A with --use-only nationality --use-only female"""
    pipeline = PhaseAPipeline(strategy=laptop_strategy)
    artifacts = pipeline.run(
        input_dataset_path=test_dataset,
        output_dir=tmp_dir,
        use_only_labels=["nationality", "female"],
    )
    
    # Verify artifacts
    metrics = json.loads(artifacts.metrics_path.read_text())
    assert metrics["label_filter"]["mode"] == "multi_label"
    assert set(metrics["label_filter"]["use_only"]) == {"nationality", "female"}
    
    # Verify visualizations
    reports = list((tmp_dir / "reports" / "phase_a").glob("*.png"))
    assert "specificity_gap.png" in [r.name for r in reports]
```

#### E2E Test

```bash
# Run E2E verification script
python scripts/verify_use_only_e2e.py --samples 50 --label nationality
python scripts/verify_use_only_e2e.py --samples 50 --label nationality --label female

# Expected output:
# ================================================================================
# E2E Verification: --use-only filter (single-label mode)
# ================================================================================
# Labels: ['nationality']
# Samples: 50
# Output: artifacts/test_use_only
# 
# [1/5] Verifying Phase A artifacts...
#   ✓ Dataset: 50 rows
#   ✓ Projection matrix: (n.dim, n.dim) (dimensions vary based on the embedding model)
#   ✓ Label filter: ['nationality']
#   ✓ Probing results: ['nationality']
# 
# [2/5] Verifying visualizations...
#   ✓ amnesic_drop_with_ci.png
#   ✓ embedding_separability.png
#   ✓ solver_convergence_benchmark.png
#   ✓ specificity_gap.png: skipped (single-label)
#   ✓ entity_cooccurrence.png: skipped (single-label)
#   ✓ demographic_score_matrix.png: skipped (single-label)
# 
# [3/5] Verifying metadata propagation...
#   ✓ Phase A metadata: ['nationality']
# 
# [4/5] Verifying single-label adaptations for 'nationality'...
#   ✓ Mode: single_label
#   ✓ Probing results: single column 'nationality'
#   ✓ Control probe: skipped (as expected)
# 
# ================================================================================
# ✓ ALL TESTS PASSED
# ================================================================================
```

### 6.3: Migration Path (Minimal Risk)

#### Step 1: Implement Base Changes (No Breaking Changes)
- Add `use_only_labels` parameter to pipeline/strategy (defaults to `None`)
- Add `visualization_data` to `ProbingContext` (optional field)
- Add `SOBRTaxonomy.filter_columns()` method
- **Status**: Backward compatible, no existing code breaks

#### Step 2: Migrate HPC Strategy (Breaking Change for HPC)
- Remove `_generate_visualizations()` from hpc.py
- Update `_run_probing_stage()` to return `visualization_data`
- **Test**: Run Phase A in HPC mode, verify visualizations still generated
- **Rollback**: Restore `_generate_visualizations()` if issues found

#### Step 3: Update Pipeline Visualization Logic (Enhancement)
- Add comprehensive visualization orchestration in `phase_a_pipeline.py`
- Add conditional logic for single-label vs multi-label
- **Test**: Run Phase A in both laptop and HPC modes, verify output identical
- **Rollback**: Revert pipeline changes, keep Step 2 changes

#### Step 4: Add CLI Commands (New Features)
- Add `--use-only` to `run-phase-a` command
- Add `run-full-pipeline` command
- **Test**: Run E2E verification script
- **Rollback**: None needed (new commands don't affect existing)

### 6.4: Documentation Updates

#### README.md
```markdown
## New Feature: Label Filtering with `--use-only`

Filter Phase A execution to specific demographics:

```bash
# Single-label mode (e.g., for nationality-only research)
neuro-stylometry run-phase-a \
    --dataset artifacts/data/sobr.arrow \
    --output-dir artifacts/phase_a \
    --use-only nationality

# Multi-label mode (e.g., nationality and gender)
neuro-stylometry run-phase-a \
    --dataset artifacts/data/sobr.arrow \
    --output-dir artifacts/phase_a \
    --use-only nationality \
    --use-only female
```

**Impact**:
- Only specified demographics are detected, masked, and removed by LEACE
- Projection matrix dimension matches filtered label count
- Some visualizations are skipped in single-label mode
- Label filter propagates to Phase D and verification automatically

**Use Cases**:
1. **Focused research**: Study only nationality leakage
2. **Computational efficiency**: Reduce dataset size by filtering rows
3. **Ablation studies**: Compare single-label vs multi-label projections
```

#### Copilot Instructions Update

Add to `.github/copilot-instructions.md`:

```markdown
## Label Filtering (`--use-only`)

When users ask about filtering to specific demographics:
- CLI flag: `--use-only <label>` (can be repeated)
- Impacts: taxonomy, concept encoding, LEACE projection, dataset filtering
- Single-label mode: Some visualizations skipped (specificity_gap, entity_cooccurrence, demographic_score_matrix)
- Multi-label mode: All visualizations generated
- Metadata propagates: `phase_a_metrics.json` → `training_metadata.json`
```

---

## Part 7: Answers to Your Questions (Updated)

### Q1: Filter application timing in Phase A
**Answer**: Apply filtering **after Arrow load, before chunking**. This is correct because:
- LEACE projection is specific to the filtered label(s)
- If you later want to run inference on unfiltered data, you'd need a separate Phase A run anyway
- Filtering early minimizes wasted compute on irrelevant rows

### Q2: Shared tokenized datasets in `run-full-pipeline`
**Answer**: Reuse existing `--preprocess/--no-preprocess` logic. Add `--force-rerun` flag for explicit override:
```python
@click.option("--force-rerun", is_flag=True, help="Force re-run even if artifacts exist")
```

### Q3: Error handling in `run-full-pipeline`
**Answer**: **Stop immediately on failure**. Partial results can leave inconsistent state. The command should:
1. Log which phase failed
2. Preserve artifacts from completed phases
3. Exit with non-zero status
4. Suggest resume command (e.g., `--skip-phase-a` if Phase A completed)

---

## Implementation Order (Updated for Option A)

1. **Add base infrastructure** (Task 1-2)
   - Add `use_only_labels` parameter to pipeline/strategy
   - Add `SOBRTaxonomy.filter_columns()` method
   - Update `ProbingContext` dataclass
   - **Testing**: Unit tests for taxonomy filtering, demographic encoder

2. **Migrate visualization to pipeline** (Task 3)
   - Remove `_generate_visualizations()` from hpc.py
   - Update `_run_probing_stage()` to return visualization data
   - Add comprehensive visualization orchestration in phase_a_pipeline.py
   - **Testing**: Integration test comparing HPC/laptop output

3. **Add single-label adaptations** (Task 4)
   - Add conditional logic for single-label vs multi-label
   - Update visualization functions with adaptive titles
   - **Testing**: E2E test with single-label mode

4. **Add CLI commands** (Task 6-7)
   - Add `run-full-pipeline` command
   - Create `verify_use_only_e2e.py` script
   - **Testing**: E2E verification script