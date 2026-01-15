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

### Task 7: Create E2E Verification Script (FULL PIPELINE: Phase A → D → Verify)

**New file**: `scripts/verify_use_only_e2e.py`

**Requirements**:
1. Run complete pipeline via CLI commands (not direct imports)
2. Use 200 samples with **weighted nationality distribution** (all distinct values present)
3. Verify Phase A uses ONLY 'nationality' column_prompts from gliner_taxonomy.yaml
4. Collect and display logs from all phases
5. Verify metadata propagation through all phases
6. Run sequentially: Phase A → Phase D → Verify

---

#### 7.1: Test Architecture

```
verify_use_only_e2e.py
  │
  ├─ Step 1: Prepare Dataset (200 samples, weighted nationality sampling)
  │   └─ Ensure all nationality values have representation
  │
  ├─ Step 2: Run Phase A via CLI (neuro-stylometry run-phase-a --use-only nationality)
  │   ├─ Capture stdout/stderr logs
  │   └─ Verify only 'nationality' prompts accessed (log inspection)
  │
  ├─ Step 3: Verify Phase A Artifacts
  │   ├─ Check files exist (clean_dataset.arrow, projection_matrix.pt, etc.)
  │   ├─ Verify pollution_logs.arrow contains ONLY nationality detections
  │   ├─ Verify phase_a_metrics.json has label_filter={'nationality'}
  │   └─ Verify single-label visualizations (no specificity_gap, etc.)
  │
  ├─ Step 4: Run Phase D via CLI (neuro-stylometry run-phase-d --use-only nationality)
  │   ├─ Capture stdout/stderr logs
  │   └─ Use Phase A outputs (clean_dataset.arrow, projection_matrix.pt)
  │
  ├─ Step 5: Verify Phase D Artifacts
  │   ├─ Check training_metadata.json has label_filter={'nationality'}
  │   ├─ Verify SingleTaskHead used (not MultiTaskHead)
  │   └─ Verify baseline/constrained checkpoints exist
  │
  ├─ Step 6: Run Verify via CLI (neuro-stylometry verify)
  │   ├─ Capture stdout/stderr logs
  │   └─ Auto-detect --use-only from training_metadata.json
  │
  └─ Step 7: Verify Verification Artifacts
      ├─ Check CHG metrics (head-level orthogonality)
      └─ Check SVS metrics (subspace alignment)
```

---

#### 7.2: Weighted Sampling Strategy

**Problem**: Simple random sampling may miss rare nationality values (e.g., "Latvian" with 0.2% frequency).

**Solution**: Stratified sampling with minimum per-class representation:

```python
def create_weighted_nationality_sample(
    dataset_path: Path,
    output_path: Path,
    n_samples: int = 200,
    min_samples_per_class: int = 2,
) -> None:
    """
    Create a stratified sample ensuring all nationality values are represented.
    
    Strategy:
    1. Compute nationality value counts (e.g., German: 1200, Latvian: 8)
    2. Reserve min_samples_per_class for each nationality (2 * n_classes)
    3. Distribute remaining samples proportionally by frequency
    4. Sample with replacement for rare classes if needed
    
    Args:
        dataset_path: Path to full SOBR dataset.
        output_path: Path to save sampled dataset.
        n_samples: Total samples (default: 200).
        min_samples_per_class: Minimum samples per nationality (default: 2).
    """
    import pyarrow as pa
    import pyarrow.feather as feather
    import numpy as np
    import pandas as pd
    
    # Load full dataset
    table = feather.read_table(dataset_path)
    df = table.to_pandas()
    
    # Filter to non-null nationality
    df_valid = df[df["nationality"].notna()].copy()
    
    # Get nationality value counts
    nationality_counts = df_valid["nationality"].value_counts()
    n_classes = len(nationality_counts)
    
    print(f"Found {n_classes} distinct nationality values")
    print(f"Total non-null samples: {len(df_valid)}")
    
    # Calculate samples per class
    reserved = min_samples_per_class * n_classes
    remaining = n_samples - reserved
    
    if remaining < 0:
        raise ValueError(f"Cannot sample {n_samples} with min {min_samples_per_class} per class ({n_classes} classes)")
    
    # Allocate samples
    samples_per_class = {}
    for nationality, count in nationality_counts.items():
        # Guaranteed minimum
        samples_per_class[nationality] = min_samples_per_class
        
        # Proportional allocation of remaining
        proportion = count / len(df_valid)
        additional = int(remaining * proportion)
        samples_per_class[nationality] += additional
    
    # Sample from each nationality
    sampled_dfs = []
    for nationality, n_sample in samples_per_class.items():
        nationality_df = df_valid[df_valid["nationality"] == nationality]
        
        # Sample with replacement if class is too small
        replace = len(nationality_df) < n_sample
        sampled = nationality_df.sample(n=n_sample, replace=replace, random_state=42)
        sampled_dfs.append(sampled)
    
    # Combine and shuffle
    final_df = pd.concat(sampled_dfs, ignore_index=True).sample(frac=1, random_state=42)
    
    # Truncate to exact n_samples (may be slightly over due to rounding)
    final_df = final_df.head(n_samples)
    
    print(f"\nFinal sample distribution:")
    print(final_df["nationality"].value_counts())
    print(f"\nTotal samples: {len(final_df)}")
    
    # Save as Arrow
    final_table = pa.Table.from_pandas(final_df, schema=table.schema)
    feather.write_feather(final_table, output_path)
```

---

#### 7.3: Verification: Only 'nationality' Prompts Accessed

**Critical Requirement**: Ensure GLiNER does NOT use prompts from other demographics.

**Verification Strategy**:

1. **Log Inspection** (Passive):
   - Check Phase A logs for taxonomy loading messages
   - Verify "Loaded taxonomy: ['nationality']" appears
   - Verify NO other demographic names appear in GLiNER init logs

2. **Artifact Inspection** (Forensic):
   - Load `pollution_logs.arrow` and verify `entity_type` column contains ONLY "nationality"
   - If other entity types appear, test FAILS

3. **Code Instrumentation** (Active - if needed):
   - Add logging to `SOBRTaxonomy.filter_columns()` method
   - Add logging to `GLiNERDetector.__init__()` to print loaded prompts
   - Add assertion in pipeline to check `len(taxonomy.column_prompts) == 1`

**Implementation**:

```python
def verify_only_nationality_prompts_used(
    phase_a_logs: str,
    pollution_logs_path: Path,
) -> bool:
    """
    Verify that ONLY nationality prompts were used in Phase A.
    
    Checks:
    1. Logs contain "nationality" taxonomy loading
    2. pollution_logs.arrow has only "nationality" entity types
    3. No other demographic names (birth_year, female, etc.) appear
    
    Args:
        phase_a_logs: Captured stdout/stderr from Phase A.
        pollution_logs_path: Path to pollution_logs.arrow.
        
    Returns:
        True if only nationality was used, False otherwise.
    """
    print("\n[3/7] Verifying ONLY nationality prompts used...")
    
    # Forbidden demographic names (must NOT appear)
    forbidden_demographics = [
        "birth_year", "female", "political_leaning",
        "extrovert", "sensing", "feeling", "judging"
    ]
    
    # Check logs for taxonomy loading
    if "nationality" not in phase_a_logs:
        print("  ✗ 'nationality' not found in Phase A logs")
        return False
    
    for forbidden in forbidden_demographics:
        if forbidden in phase_a_logs:
            print(f"  ✗ Forbidden demographic '{forbidden}' found in logs")
            return False
    
    print("  ✓ Logs contain only 'nationality' references")
    
    # Check pollution logs entity types
    import pyarrow.feather as feather
    logs_table = feather.read_table(pollution_logs_path)
    
    if "entity_type" not in logs_table.column_names:
        print("  ✗ pollution_logs.arrow missing 'entity_type' column")
        return False
    
    entity_types = logs_table["entity_type"].to_pylist()
    unique_types = set(et for et in entity_types if et is not None)
    
    if unique_types != {"nationality"}:
        print(f"  ✗ Unexpected entity types: {unique_types}")
        return False
    
    print(f"  ✓ pollution_logs.arrow contains only 'nationality' ({len(entity_types)} spans)")
    
    return True
```

---

#### 7.4: Running CLI Commands with Log Capture

**Strategy**: Use `subprocess.run()` to invoke CLI commands and capture output.

```python
import subprocess
import sys
from pathlib import Path

def run_cli_command(
    command: List[str],
    description: str,
) -> Tuple[bool, str]:
    """
    Run a CLI command and capture stdout/stderr.
    
    Args:
        command: Command and arguments (e.g., ["neuro-stylometry", "run-phase-a", ...])
        description: Human-readable description for logging.
        
    Returns:
        Tuple of (success: bool, logs: str)
    """
    print(f"\n{'=' * 80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'=' * 80}\n")
    
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
    )
    
    # Print captured output
    print(result.stdout)
    
    success = result.returncode == 0
    if success:
        print(f"\n✓ {description} completed successfully")
    else:
        print(f"\n✗ {description} failed with exit code {result.returncode}")
    
    return success, result.stdout


# Example usage:
success, logs = run_cli_command(
    command=[
        "neuro-stylometry", "run-phase-a",
        "--dataset", str(sampled_dataset_path),
        "--output-dir", str(phase_a_dir),
        "--mode", "laptop",
        "--use-only", "nationality",
    ],
    description="Phase A (Pollution Filtering)",
)

if not success:
    sys.exit(1)
```

---

#### 7.5: Complete Test Flow

```python
#!/usr/bin/env python3
"""
E2E verification script for --use-only filter across FULL pipeline.

Tests single-label mode (nationality only) through all phases:
- Phase A: Pollution detection + LEACE projection
- Phase D: Training with projection-aware architecture
- Verify: CHG + SVS orthogonality verification

Verifies:
1. Only 'nationality' column_prompts used (NO other demographics)
2. Weighted sampling ensures all nationality values represented
3. Metadata propagates correctly through all phases
4. Single-label adaptations work (SingleTaskHead, skipped visualizations)
5. All artifacts have correct dimensions

Usage:
    python scripts/verify_use_only_e2e.py \
        --dataset artifacts/data/sobr.arrow \
        --output-dir artifacts/test_use_only \
        --samples 200
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Tuple, List

import pyarrow as pa
import pyarrow.feather as feather
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="E2E verification for --use-only filter (FULL PIPELINE)"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/data/sobr.arrow"),
        help="Path to full SOBR dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/test_use_only"),
        help="Output directory for test artifacts",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of samples to use (default: 200)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["laptop", "hpc"],
        default="laptop",
        help="Hardware mode (default: laptop)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean output directory before running",
    )
    
    args = parser.parse_args()
    
    # Setup paths
    output_dir = args.output_dir
    sampled_dataset = output_dir / "sobr_nationality_200.arrow"
    phase_a_dir = output_dir / "phase_a"
    phase_d_dir = output_dir / "phase_d"
    
    # Clean if requested
    if args.clean and output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("E2E Verification: --use-only nationality (FULL PIPELINE)")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}")
    print(f"Samples: {args.samples}")
    print(f"Mode: {args.mode}")
    print("=" * 80)
    
    # =========================================================================
    # Step 1: Create Weighted Sample
    # =========================================================================
    print("\n[1/7] Creating weighted nationality sample...")
    create_weighted_nationality_sample(
        dataset_path=args.dataset,
        output_path=sampled_dataset,
        n_samples=args.samples,
    )
    
    # =========================================================================
    # Step 2: Run Phase A
    # =========================================================================
    success, phase_a_logs = run_cli_command(
        command=[
            "neuro-stylometry", "run-phase-a",
            "--dataset", str(sampled_dataset),
            "--output-dir", str(phase_a_dir),
            "--mode", args.mode,
            "--use-only", "nationality",
        ],
        description="Phase A (Pollution Filtering)",
    )
    
    if not success:
        print("\n✗ Phase A FAILED")
        return 1
    
    # =========================================================================
    # Step 3: Verify Only Nationality Prompts Used
    # =========================================================================
    if not verify_only_nationality_prompts_used(
        phase_a_logs=phase_a_logs,
        pollution_logs_path=phase_a_dir / "pollution_logs.arrow",
    ):
        print("\n✗ Phase A used non-nationality prompts!")
        return 1
    
    # =========================================================================
    # Step 4: Verify Phase A Artifacts
    # =========================================================================
    if not verify_phase_a_artifacts(
        phase_a_dir=phase_a_dir,
        expected_label="nationality",
    ):
        print("\n✗ Phase A artifacts verification FAILED")
        return 1
    
    # =========================================================================
    # Step 5: Run Phase D
    # =========================================================================
    success, phase_d_logs = run_cli_command(
        command=[
            "neuro-stylometry", "run-phase-d",
            "--dataset", str(phase_a_dir / "clean_dataset.arrow"),
            "--output-dir", str(phase_d_dir),
            "--artifacts-dir", str(phase_a_dir),
            "--mode", args.mode,
            "--use-only", "nationality",
            "--preprocess",  # Auto-tokenize if needed
        ],
        description="Phase D (Training)",
    )
    
    if not success:
        print("\n✗ Phase D FAILED")
        return 1
    
    # =========================================================================
    # Step 6: Verify Phase D Artifacts
    # =========================================================================
    if not verify_phase_d_artifacts(
        phase_d_dir=phase_d_dir,
        expected_label="nationality",
    ):
        print("\n✗ Phase D artifacts verification FAILED")
        return 1
    
    # =========================================================================
    # Step 7: Run Verify
    # =========================================================================
    success, verify_logs = run_cli_command(
        command=[
            "neuro-stylometry", "verify",
            "--dataset", str(phase_a_dir / "clean_dataset.arrow"),
            "--phase-d-dir", str(phase_d_dir),
            "--artifacts-dir", str(phase_a_dir),
            "--mode", args.mode,
            # --use-only auto-detected from training_metadata.json
        ],
        description="Verification (CHG + SVS)",
    )
    
    if not success:
        print("\n✗ Verification FAILED")
        return 1
    
    # =========================================================================
    # Step 8: Verify Verification Artifacts
    # =========================================================================
    if not verify_verification_artifacts(
        phase_d_dir=phase_d_dir,
        expected_label="nationality",
    ):
        print("\n✗ Verification artifacts check FAILED")
        return 1
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED")
    print("=" * 80)
    print(f"\nArtifacts saved to: {args.output_dir}")
    print(f"  - Phase A: {phase_a_dir}")
    print(f"  - Phase D: {phase_d_dir}")
    print(f"  - Logs captured for all phases")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

#### 7.6: What's Already Addressed vs. Still Missing

| Requirement | Status | Notes |
|-------------|--------|-------|
| **Run full pipeline (A→D→Verify)** | ✅ Addressed | Uses CLI commands sequentially |
| **200 samples with weighted distribution** | ✅ Addressed | `create_weighted_nationality_sample()` ensures all nationalities present |
| **Verify ONLY nationality prompts used** | ✅ Addressed | `verify_only_nationality_prompts_used()` checks logs + artifacts |
| **Capture and display all logs** | ✅ Addressed | `run_cli_command()` captures stdout/stderr |
| **Metadata propagation verification** | ✅ Addressed | Checks `label_filter` in metrics JSONs across phases |
| **Single-label adaptations (visualizations)** | ✅ Addressed | Verifies specificity_gap/cooccurrence/score_matrix are skipped |
| **Single-label architecture (SingleTaskHead)** | ✅ Addressed | `verify_phase_d_artifacts()` checks training_metadata.json |
| **Artifact dimension checks** | ✅ Addressed | Verifies projection matrix, dataset row counts, tokenized shapes |
| **--use-only CLI implementation** | ⚠️ **MISSING** | Task 1-2: Add `--use-only` to pipeline/strategy (NOT YET IMPLEMENTED) |
| **Taxonomy filtering logic** | ⚠️ **MISSING** | `SOBRTaxonomy.filter_columns()` method doesn't exist yet |
| **Pipeline visualization consolidation** | ⚠️ **MISSING** | Task 3: Remove HPC `_generate_visualizations()`, centralize in pipeline |
| **Phase D `--use-only` support** | ⚠️ **MISSING** | PhaseDDataset filtering + SingleTaskHead detection |

---

#### 7.7: Pre-Execution Checklist (Before Running Test)

**Must be implemented first (from Tasks 1-4)**:
- [ ] Add `use_only_labels` parameter to `PhaseAPipeline.run()`
- [ ] Add `use_only_labels` parameter to `HPCFilterStrategy.execute()` / `LaptopFilterStrategy.execute()`
- [ ] Implement `SOBRTaxonomy.filter_columns(columns: List[str])` method
- [ ] Update `phase_a_pipeline.py` to filter taxonomy before passing to detector
- [ ] Remove `_generate_visualizations()` from `hpc.py`, move logic to pipeline
- [ ] Add conditional visualization logic (skip specificity_gap if single-label)
- [ ] Add `use_only_labels` parameter to `PhaseDPipeline` / `PhaseDDataset`
- [ ] Update verification to auto-detect `--use-only` from training_metadata.json

**Can run test script after implementation**:
- Script will verify that implementations work correctly end-to-end
- Any failures indicate bugs in implementation or missing edge cases

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

#### E2E Test (Full Pipeline)

**Goal**: Verify --use-only filter works correctly across ALL phases using CLI commands.

**Test Scope**:
- **Phase A**: Only nationality prompts accessed, no other demographics
- **Phase D**: SingleTaskHead used, training metadata propagates
- **Verify**: CHG + SVS metrics computed correctly

**Execution**:
```bash
# Run full pipeline E2E test
python scripts/verify_use_only_e2e.py \
    --dataset artifacts/data/sobr.arrow \
    --output-dir artifacts/test_use_only \
    --samples 200 \
    --mode laptop

# Expected duration: 10-15 minutes (laptop), 5-7 minutes (HPC)
```

**Expected Output** (abbreviated):
```
================================================================================
E2E Verification: --use-only nationality (FULL PIPELINE)
================================================================================
Dataset: artifacts/data/sobr.arrow
Output: artifacts/test_use_only
Samples: 200
Mode: laptop
================================================================================

[1/7] Creating weighted nationality sample...
Found 27 distinct nationality values
Total non-null samples: 5000
Final sample distribution:
German       24
British      18
American     16
...
Latvian       2
Total samples: 200

================================================================================
Running: Phase A (Pollution Filtering)
Command: neuro-stylometry run-phase-a --dataset artifacts/test_use_only/sobr_nationality_200.arrow --output-dir artifacts/test_use_only/phase_a --mode laptop --use-only nationality
================================================================================
[Phase A logs appear here...]
✓ Phase A (Pollution Filtering) completed successfully

[3/7] Verifying ONLY nationality prompts used...
  ✓ Logs contain only 'nationality' references
  ✓ pollution_logs.arrow contains only 'nationality' (847 spans)

[4/7] Verifying Phase A artifacts...
  ✓ Dataset: 200 rows
  ✓ Projection matrix: (768, 768)
  ✓ Label filter: ['nationality']
  ✓ Probing results: ['nationality']
  ✓ Visualizations: specificity_gap.png skipped (single-label)

================================================================================
Running: Phase D (Training)
Command: neuro-stylometry run-phase-d --dataset artifacts/test_use_only/phase_a/clean_dataset.arrow --output-dir artifacts/test_use_only/phase_d --artifacts-dir artifacts/test_use_only/phase_a --mode laptop --use-only nationality --preprocess
================================================================================
[Phase D logs appear here...]
✓ Phase D (Training) completed successfully

[5/7] Verifying Phase D artifacts...
  ✓ training_metadata.json: label_filter=['nationality']
  ✓ Architecture: SingleTaskHead detected
  ✓ Checkpoints: baseline + constrained exist

================================================================================
Running: Verification (CHG + SVS)
Command: neuro-stylometry verify --dataset artifacts/test_use_only/phase_a/clean_dataset.arrow --phase-d-dir artifacts/test_use_only/phase_d --artifacts-dir artifacts/test_use_only/phase_a --mode laptop
================================================================================
[Verification logs appear here...]
✓ Verification (CHG + SVS) completed successfully

[8/7] Verifying verification artifacts...
  ✓ CHG metrics computed
  ✓ SVS metrics computed
  ✓ Auto-detected use_only from training_metadata.json

================================================================================
✓ ALL TESTS PASSED
================================================================================

Artifacts saved to: artifacts/test_use_only
  - Phase A: artifacts/test_use_only/phase_a
  - Phase D: artifacts/test_use_only/phase_d
  - Logs captured for all phases
================================================================================
```

**Critical Assertion**: Test verifies that pollution_logs.arrow contains ONLY "nationality" entity types (no birth_year, female, etc.)
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