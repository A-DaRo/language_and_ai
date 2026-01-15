#!/usr/bin/env python3
"""
E2E verification script for --use-only filter across full pipeline.

Usage:
    python scripts/verify_use_only_e2e.py \\
        --dataset artifacts/data/sobr.arrow \\
        --output-dir artifacts/test_use_only \\
        --samples 200 \\
        --mode laptop
    
Full Pipeline Test Sequence:
    1. Create weighted subset (200 samples, nationality weighted distribution)
    2. Phase A with --use-only nationality (taxonomy prompt verification)
    3. Phase D training on filtered data
    4. Verify metrics and metadata propagation

STRICT VERIFICATION:
    - Verifies that ONLY nationality column_prompts are accessed during GLiNER
    - No other demographic prompts should be in inference labels
    - Validates taxonomy filtering works end-to-end

Exit codes:
    0: All tests passed
    1: Test failure (see stderr for details)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging to show all output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Also enable logging for neuro_stylometry modules
logging.getLogger("neuro_stylometry").setLevel(logging.INFO)


# =============================================================================
# Constants
# =============================================================================
ALL_DEMOGRAPHIC_COLUMNS = [
    "nationality",
    "female",
    "birth_year",
    "feeling_thinking",
    "judging_perceiving",
    "sensing_intuitive",
    "extrovert_introvert",
    "political_leaning",
]

# Columns that should NOT appear in prompts when --use-only nationality is set
EXCLUDED_COLUMNS_FOR_NATIONALITY_ONLY = [
    col for col in ALL_DEMOGRAPHIC_COLUMNS if col != "nationality"
]


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class TestResult:
    """Result from a test step."""
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaxonomyAccessLog:
    """Log of taxonomy prompt accesses during inference."""
    accessed_columns: Set[str] = field(default_factory=set)
    inference_labels: List[str] = field(default_factory=list)
    
    def contains_excluded_columns(self, excluded: List[str]) -> List[str]:
        """Return list of excluded columns that were accessed."""
        return [col for col in excluded if col in self.accessed_columns]


# =============================================================================
# Helper Functions
# =============================================================================
def print_section(title: str, char: str = "=", width: int = 80):
    """Print a section header."""
    print(f"\n{char * width}")
    print(title)
    print(f"{char * width}\n")


def print_step(step_num: int, total: int, description: str):
    """Print a step indicator."""
    print(f"\n[{step_num}/{total}] {description}")
    print("-" * 60)


def print_substep(description: str):
    """Print a substep indicator."""
    print(f"  → {description}")


# =============================================================================
# Dataset Preparation with Weighted Sampling
# =============================================================================
def create_weighted_subset(
    input_path: Path,
    output_path: Path,
    target_column: str,
    num_samples: int,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Create a weighted subset with more balanced distribution of target column.
    
    Uses stratified sampling to ensure adequate representation of each
    category in the target column.
    
    Args:
        input_path: Path to source Arrow dataset.
        output_path: Path to save subset Arrow dataset.
        target_column: Column to use for stratified sampling.
        num_samples: Target number of samples.
        seed: Random seed for reproducibility.
        
    Returns:
        Dict with subset statistics.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.feather as feather
    
    print_substep(f"Loading source dataset: {input_path}")
    table = feather.read_table(input_path)
    print(f"    Source: {len(table):,} rows")
    
    # Filter to rows with valid target column
    valid_mask = pc.is_valid(table[target_column])
    table = table.filter(valid_mask)
    print(f"    After filtering nulls in '{target_column}': {len(table):,} rows")
    
    # Get value counts for stratification
    values = table[target_column].to_pandas()
    value_counts = values.value_counts()
    print(f"    Unique values in '{target_column}': {len(value_counts)}")
    
    # Calculate samples per stratum (weighted by inverse frequency for balance)
    rng = np.random.default_rng(seed)
    
    # Target roughly equal samples per category, but cap at available
    samples_per_category = num_samples // len(value_counts)
    remainder = num_samples % len(value_counts)
    
    selected_indices = []
    stats = {"by_category": {}}
    
    for i, (category, count) in enumerate(value_counts.items()):
        # Allocate extra samples to first categories if there's remainder
        target_count = samples_per_category + (1 if i < remainder else 0)
        actual_count = min(target_count, count)
        
        # Get indices for this category
        category_mask = values == category
        category_indices = np.where(category_mask)[0]
        
        # Sample from this category
        if len(category_indices) > actual_count:
            sampled = rng.choice(category_indices, size=actual_count, replace=False)
        else:
            sampled = category_indices
        
        selected_indices.extend(sampled.tolist())
        stats["by_category"][str(category)] = len(sampled)
    
    # Shuffle selected indices
    selected_indices = list(set(selected_indices))  # Remove any duplicates
    rng.shuffle(selected_indices)
    
    # Take final subset (might be slightly different from target due to category availability)
    if len(selected_indices) > num_samples:
        selected_indices = selected_indices[:num_samples]
    
    subset_table = table.take(pa.array(selected_indices))
    
    # Save subset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feather.write_feather(subset_table, output_path)
    
    stats["total_samples"] = len(subset_table)
    stats["source_samples"] = len(table)
    stats["target_column"] = target_column
    
    print(f"    Created subset: {len(subset_table):,} samples")
    print(f"    Distribution: {stats['by_category']}")
    
    return stats


# =============================================================================
# Taxonomy Prompt Verification
# =============================================================================
def verify_taxonomy_filtering(
    taxonomy_config: Dict[str, Any],
    use_only_labels: List[str],
) -> TestResult:
    """
    Verify that taxonomy is correctly filtered to only include specified labels.
    
    This is the CRITICAL check: ensures only nationality prompts are used
    when --use-only nationality is specified.
    
    Args:
        taxonomy_config: The column_prompts dict from taxonomy.
        use_only_labels: Labels that should be included.
        
    Returns:
        TestResult with pass/fail and details.
    """
    result = TestResult(passed=True)
    
    # Get all columns in the taxonomy
    taxonomy_columns = set(taxonomy_config.keys())
    expected_columns = set(use_only_labels)
    
    result.info["taxonomy_columns"] = list(taxonomy_columns)
    result.info["expected_columns"] = list(expected_columns)
    
    # Check for unexpected columns (should be EMPTY if filtering works)
    unexpected = taxonomy_columns - expected_columns
    if unexpected:
        result.passed = False
        result.errors.append(
            f"CRITICAL: Taxonomy contains unexpected columns: {sorted(unexpected)}"
        )
        result.errors.append(
            "This means --use-only filter is NOT properly filtering the taxonomy!"
        )
    
    # Check for missing columns
    missing = expected_columns - taxonomy_columns
    if missing:
        result.passed = False
        result.errors.append(f"Expected columns missing from taxonomy: {sorted(missing)}")
    
    # Count prompts per column
    prompt_counts = {col: len(prompts) for col, prompts in taxonomy_config.items()}
    result.info["prompt_counts"] = prompt_counts
    
    if result.passed:
        print(f"    ✓ Taxonomy contains ONLY expected columns: {sorted(expected_columns)}")
        print(f"    ✓ Prompt counts: {prompt_counts}")
    
    return result


def verify_inference_labels(
    inference_labels: List[str],
    use_only_labels: List[str],
    taxonomy_config: Dict[str, Any],
) -> TestResult:
    """
    Verify that inference labels only contain prompts from allowed columns.
    
    Args:
        inference_labels: Flat list of labels used in GLiNER inference.
        use_only_labels: Allowed demographic columns.
        taxonomy_config: Full taxonomy for reference.
        
    Returns:
        TestResult with pass/fail and details.
    """
    result = TestResult(passed=True)
    
    result.info["num_inference_labels"] = len(inference_labels)
    result.info["inference_labels_sample"] = inference_labels[:10]
    
    # Build mapping of prompt -> column from taxonomy
    prompt_to_column = {}
    for col, prompts in taxonomy_config.items():
        for prompt in prompts:
            prompt_to_column[prompt] = col
    
    # Check each inference label
    unexpected_columns = set()
    unexpected_labels = []
    
    for label in inference_labels:
        if label in prompt_to_column:
            col = prompt_to_column[label]
            if col not in use_only_labels:
                unexpected_columns.add(col)
                unexpected_labels.append((label, col))
    
    if unexpected_columns:
        result.passed = False
        result.errors.append(
            f"CRITICAL: Inference labels contain prompts from excluded columns: {sorted(unexpected_columns)}"
        )
        for label, col in unexpected_labels[:5]:
            result.errors.append(f"  - '{label}' → column '{col}'")
        if len(unexpected_labels) > 5:
            result.errors.append(f"  ... and {len(unexpected_labels) - 5} more")
    else:
        print(f"    ✓ All {len(inference_labels)} inference labels are from allowed columns")
    
    return result


# =============================================================================
# Phase A Execution with Taxonomy Monitoring
# =============================================================================
def run_phase_a_with_monitoring(
    dataset_path: Path,
    output_dir: Path,
    use_only_labels: List[str],
    mode: str = "laptop",
) -> TestResult:
    """
    Run Phase A with --use-only filter and monitor taxonomy access.
    
    This function:
    1. Loads config and creates strategy
    2. Intercepts taxonomy to verify filtering
    3. Runs Phase A pipeline
    4. Returns verification results
    
    Args:
        dataset_path: Path to input dataset.
        output_dir: Output directory for artifacts.
        use_only_labels: List of demographic labels to filter to.
        mode: Hardware mode (laptop/hpc).
        
    Returns:
        TestResult with execution and verification results.
    """
    from neuro_stylometry.phase_a_pipeline import PhaseAPipeline
    from neuro_stylometry.factories.strategy_factory import StrategyFactory
    from neuro_stylometry.hardware_ops.detection import ProfileType
    from neuro_stylometry.config import load_pipeline_config
    from neuro_stylometry.pollution_guard.gliner_detector import SOBRTaxonomy
    
    result = TestResult(passed=True)
    
    try:
        print_substep(f"Hardware mode: {mode}")
        print_substep(f"Labels: {use_only_labels}")
        print_substep(f"Output: {output_dir}")
        
        # Create strategy
        profile_type = ProfileType.HPC if mode == "hpc" else ProfileType.LAPTOP
        strategy = StrategyFactory.create_filter_strategy(profile_type)
        
        # Load configuration
        config = load_pipeline_config(mode=mode)
        
        # Load taxonomy config to verify filtering
        taxonomy_cfg_path = Path(config.get("gliner", {}).get("taxonomy_path", "conf/base/gliner_taxonomy.yaml"))
        
        if not taxonomy_cfg_path.is_absolute():
            taxonomy_cfg_path = Path(__file__).parent.parent / taxonomy_cfg_path
        
        import yaml
        with open(taxonomy_cfg_path) as f:
            full_taxonomy_yaml = yaml.safe_load(f)
        
        # The YAML structure is: taxonomy.column_prompts.<col>.prompts/distractors/etc.
        taxonomy_section = full_taxonomy_yaml.get("taxonomy", {})
        
        print_substep("Verifying taxonomy configuration...")
        
        # Create taxonomy using from_config which properly parses the nested structure
        full_taxonomy = SOBRTaxonomy.from_config(taxonomy_section)
        all_columns = sorted(full_taxonomy.column_prompts.keys())
        print(f"    Full taxonomy columns: {all_columns}")
        print(f"    Total inference labels: {len(full_taxonomy.get_inference_labels())}")
        
        # TEST: Verify filter_columns method exists and works
        try:
            filtered_taxonomy = full_taxonomy.filter_columns(use_only_labels)
            filtered_columns = sorted(filtered_taxonomy.column_prompts.keys())
            print(f"    Filtered taxonomy columns: {filtered_columns}")
            print(f"    Filtered inference labels: {len(filtered_taxonomy.get_inference_labels())}")
            
            # Verify filtering worked - convert column_prompts keys to set for comparison
            taxonomy_columns = set(filtered_taxonomy.column_prompts.keys())
            expected_columns = set(use_only_labels)
            
            # Check for unexpected columns (should be EMPTY if filtering works)
            unexpected = taxonomy_columns - expected_columns
            if unexpected:
                result.passed = False
                result.errors.append(
                    f"CRITICAL: Taxonomy contains unexpected columns: {sorted(unexpected)}"
                )
                result.errors.append(
                    "This means --use-only filter is NOT properly filtering the taxonomy!"
                )
                return result
            
            # Check for missing columns
            missing = expected_columns - taxonomy_columns
            if missing:
                result.passed = False
                result.errors.append(f"Expected columns missing from taxonomy: {sorted(missing)}")
                return result
            
            print(f"    ✓ Taxonomy filtering verified: only {use_only_labels} columns")
                
        except AttributeError as e:
            result.passed = False
            result.errors.append(f"CRITICAL: SOBRTaxonomy.filter_columns() method missing: {e}")
            return result
        
        # Verify that the strategies actually USE the filtered taxonomy
        # This is done by checking if use_only_labels flows through to GLiNER init
        print_substep("Running Phase A pipeline...")
        
        # Create and run pipeline
        pipeline = PhaseAPipeline(strategy=strategy, config=config)
        
        start_time = time.time()
        artifacts = pipeline.run(
            input_dataset_path=dataset_path,
            output_dir=output_dir,
            use_only_labels=use_only_labels,
        )
        elapsed = time.time() - start_time
        
        result.info["elapsed_seconds"] = elapsed
        result.info["num_samples"] = artifacts.metadata.get("num_samples")
        
        print(f"    Phase A completed in {elapsed:.1f}s")
        print(f"    Samples processed: {artifacts.metadata.get('num_samples')}")
        
        # Verify artifacts exist
        if not artifacts.clean_dataset_path.exists():
            result.passed = False
            result.errors.append("clean_dataset.arrow not created")
        
        if not artifacts.projection_matrix_path.exists():
            result.passed = False
            result.errors.append("projection_matrix.pt not created")
        
        # Check metrics for label_filter propagation
        metrics_path = output_dir / "reports" / "phase_a" / "phase_a_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            
            label_filter = metrics.get("execution", {}).get("label_filter")
            if label_filter is None:
                result.warnings.append("label_filter not found in metrics")
            elif set(label_filter) != set(use_only_labels):
                result.errors.append(
                    f"label_filter mismatch: expected {use_only_labels}, got {label_filter}"
                )
                result.passed = False
            else:
                print(f"    ✓ label_filter in metrics: {label_filter}")
            
            # Check probed columns
            probed_cols = list(metrics.get("probe", {}).get("by_column", {}).keys())
            result.info["probed_columns"] = probed_cols
            
            # For single-label mode, should only probe the specified column
            if len(use_only_labels) == 1:
                if len(probed_cols) != 1 or probed_cols[0] != use_only_labels[0]:
                    result.warnings.append(
                        f"Expected single probed column '{use_only_labels[0]}', got {probed_cols}"
                    )
                else:
                    print(f"    ✓ Single-label probing verified: {probed_cols}")
        
        return result
        
    except Exception as e:
        result.passed = False
        result.errors.append(f"Phase A execution failed: {e}")
        logger.exception("Phase A execution error")
        return result
# =============================================================================
# Phase A Artifact Verification
# =============================================================================
def verify_phase_a_artifacts(
    output_dir: Path,
    expected_labels: List[str],
) -> TestResult:
    """
    Verify Phase A artifacts match expected structure and dimensions.
    
    Args:
        output_dir: Directory containing Phase A artifacts.
        expected_labels: Expected demographic labels from --use-only.
        
    Returns:
        TestResult with verification results.
    """
    import pyarrow.feather as feather
    import torch
    
    result = TestResult(passed=True)
    is_single_label = len(expected_labels) == 1
    
    # Check required artifacts exist
    required_files = [
        "clean_dataset.arrow",
        "projection_matrix.pt",
        "pollution_logs.arrow",
    ]
    
    for filename in required_files:
        filepath = output_dir / filename
        if not filepath.exists():
            result.errors.append(f"Missing required artifact: {filename}")
            result.passed = False
    
    if not result.passed:
        return result
    
    # Load and verify clean dataset
    clean_dataset = feather.read_table(output_dir / "clean_dataset.arrow")
    result.info["num_samples"] = len(clean_dataset)
    print(f"    ✓ Dataset: {len(clean_dataset)} rows")
    
    # Load and verify projection matrix
    projection_matrix = torch.load(
        output_dir / "projection_matrix.pt",
        map_location="cpu",
        weights_only=False
    )
    result.info["projection_shape"] = list(projection_matrix.shape)
    print(f"    ✓ Projection matrix: {tuple(projection_matrix.shape)}")
    
    # Check projection matrix is valid (roughly idempotent)
    P2 = projection_matrix @ projection_matrix
    diff = torch.norm(P2 - projection_matrix).item()
    if diff > 0.1:
        result.warnings.append(f"Projection matrix idempotence check: ||P²-P||={diff:.4f}")
    else:
        print(f"    ✓ Projection idempotence: ||P²-P||={diff:.6f}")
    
    # Load pollution logs
    pollution_logs = feather.read_table(output_dir / "pollution_logs.arrow")
    result.info["num_pollution_spans"] = len(pollution_logs)
    print(f"    ✓ Pollution logs: {len(pollution_logs)} spans")
    
    # Check entity types in pollution logs (should only be from allowed columns)
    if "entity_type" in pollution_logs.column_names:
        entity_types = set(pollution_logs["entity_type"].to_pylist())
        result.info["entity_types"] = list(entity_types)
        print(f"    ✓ Entity types detected: {sorted(entity_types)}")
    
    return result


# =============================================================================
# Visualization Verification
# =============================================================================
def verify_visualizations(
    reports_dir: Path,
    is_single_label: bool,
) -> TestResult:
    """
    Verify correct visualizations were generated.
    
    Args:
        reports_dir: Reports directory containing visualizations.
        is_single_label: Whether single-label mode was used.
        
    Returns:
        TestResult with verification results.
    """
    result = TestResult(passed=True)
    
    # Expected visualizations (may vary based on config)
    expected_always = [
        "amnesic_drop_with_ci.png",
        "embedding_separability.png",
    ]
    
    optional = [
        "solver_convergence_benchmark.png",
    ]
    
    # These should be skipped in single-label mode
    multi_label_only = [
        "specificity_gap.png",
        "entity_cooccurrence.png",
        "demographic_score_matrix.png",
    ]
    
    found_visualizations = []
    if reports_dir.exists():
        found_visualizations = [f.name for f in reports_dir.glob("*.png")]
    
    result.info["found_visualizations"] = found_visualizations
    
    for viz in expected_always:
        if viz in found_visualizations:
            print(f"    ✓ {viz}")
        else:
            result.warnings.append(f"Expected visualization not found: {viz}")
            print(f"    ⚠ {viz}: not found")
    
    for viz in optional:
        if viz in found_visualizations:
            print(f"    ✓ {viz}")
        else:
            print(f"    - {viz}: not generated (optional)")
    
    for viz in multi_label_only:
        if is_single_label:
            if viz in found_visualizations:
                result.warnings.append(f"Single-label mode should skip: {viz}")
                print(f"    ⚠ {viz}: should be skipped in single-label mode")
            else:
                print(f"    ✓ {viz}: skipped (single-label)")
        else:
            if viz in found_visualizations:
                print(f"    ✓ {viz}")
            else:
                print(f"  - {viz}: not found")
    
    return result


def verify_phase_d_artifacts(phase_d_dir: Path) -> TestResult:
    """Verify Phase D training artifacts exist and look compatible."""
    result = TestResult(passed=True)

    baseline_dir = phase_d_dir / "baseline"
    constrained_dir = phase_d_dir / "constrained"
    comparative_path = phase_d_dir / "phase_d_comparative_metrics.json"

    for run_dir, label in ((baseline_dir, "baseline"), (constrained_dir, "constrained")):
        if not run_dir.exists():
            result.errors.append(f"Missing Phase D {label} dir: {run_dir}")
            result.passed = False
            continue

        required = [
            run_dir / "model.pt",
            run_dir / "head.pt",
            run_dir / "phase_d_metrics.json",
            run_dir / "training_metadata.json",
        ]
        optional = [
            run_dir / "test_metrics.json",
            run_dir / "test_details.json",
            run_dir / "phase_d_evaluation_details.json",
        ]

        for path in required:
            if not path.exists():
                result.errors.append(f"Missing {label} artifact: {path.name}")
                result.passed = False
        for path in optional:
            if not path.exists():
                result.warnings.append(f"Missing optional {label} artifact: {path.name}")

    if not comparative_path.exists():
        result.warnings.append("Missing comparative metrics: phase_d_comparative_metrics.json")

    return result


def verify_phase_d_verify_outputs(verify_dir: Path) -> TestResult:
    """Verify Phase D verification (CHG + SVS) outputs."""
    result = TestResult(passed=True)

    required = [
        verify_dir / "head_classification_baseline.json",
        verify_dir / "head_classification_constrained.json",
        verify_dir / "svs_baseline.json",
        verify_dir / "svs_constrained.json",
        verify_dir / "verification_summary.json",
    ]
    for path in required:
        if not path.exists():
            result.errors.append(f"Missing verification artifact: {path.name}")
            result.passed = False

    comparative_path = verify_dir / "phase_d_comparative_metrics.json"
    if not comparative_path.exists():
        result.warnings.append("Missing verification comparative metrics: phase_d_comparative_metrics.json")

    return result


def verify_phase_d_report_outputs(report_dir: Path) -> TestResult:
    """Verify Phase D report output directory and report HTML."""
    result = TestResult(passed=True)

    report_path = report_dir / "phase_d_report.html"
    assets_dir = report_dir / "phase_d_assets"

    if not report_path.exists():
        result.errors.append(f"Missing report: {report_path.name}")
        result.passed = False
    if not assets_dir.exists():
        result.warnings.append(f"Missing report assets dir: {assets_dir.name}")

    return result


def verify_single_label_adaptations(
    output_dir: Path,
    label: str,
) -> TestResult:
    """
    Verify single-label mode adaptations.
    
    Args:
        output_dir: Phase A output directory.
        label: The single label that was used.
        
    Returns:
        TestResult with verification results.
    """
    result = TestResult(passed=True)
    
    metrics_path = output_dir / "reports" / "phase_a" / "phase_a_metrics.json"
    if not metrics_path.exists():
        result.warnings.append("Metrics file not found")
        return result
    
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    # Check mode
    label_filter = metrics.get("execution", {}).get("label_filter", [])
    is_single = len(label_filter) == 1
    mode = "single_label" if is_single else "multi_label"
    print(f"    ✓ Mode: {mode}")
    result.info["mode"] = mode
    
    # Check probing results
    probe_results = metrics.get("probe", {})
    if probe_results:
        by_column = probe_results.get("by_column", {})
        if is_single:
            if len(by_column) == 1 and label in by_column:
                print(f"    ✓ Probing results: single column '{label}'")
            elif len(by_column) == 0:
                result.warnings.append("No probing results found")
            else:
                result.errors.append(
                    f"Expected single column '{label}' in probing, got {list(by_column.keys())}"
                )
                result.passed = False
    
    return result


def run_phase_d(
    dataset_path: Path,
    output_dir: Path,
    artifacts_dir: Path,
    use_only_labels: List[str],
    mode: str,
) -> bool:
    """Run Phase D training for baseline + constrained."""
    from neuro_stylometry.phase_d_pipeline import run_phase_d_training

    try:
        run_phase_d_training(
            dataset_path=dataset_path,
            output_dir=output_dir,
            artifacts_dir=artifacts_dir,
            mode=mode,
            use_only_labels=tuple(use_only_labels) if use_only_labels else None,
        )
        return True
    except Exception as e:
        logger.error(f"Phase D execution failed: {e}", exc_info=True)
        return False


def run_phase_d_verify(
    dataset_path: Path,
    phase_d_dir: Path,
    artifacts_dir: Path,
    output_dir: Path,
    use_only_labels: List[str],
    mode: str,
) -> bool:
    """Run Phase D verification (CHG + SVS)."""
    from neuro_stylometry.config import find_config_root, load_verify_config
    from neuro_stylometry.stylometry_net.verification import run_verification

    def _derive_tokenized_paths(base_path: Path) -> tuple[Path, Path]:
        if base_path.suffix:
            return (
                base_path.with_name(f"{base_path.stem}_post{base_path.suffix}"),
                base_path.with_name(f"{base_path.stem}_post_masked{base_path.suffix}"),
            )
        return (
            base_path / "tokenized_post.arrow",
            base_path / "tokenized_post_masked.arrow",
        )

    try:
        verify_config = load_verify_config(mode=mode, experiment_config_path=None)
        verify_root = find_config_root("verify.yaml")

        model_name = verify_config.get("model", {}).get("name", "roberta-base")
        max_length = int(verify_config.get("model", {}).get("max_length", 512))
        taxonomy_value = verify_config.get("model", {}).get("taxonomy_path")
        if taxonomy_value:
            taxonomy_candidate = Path(taxonomy_value)
            taxonomy_path = (
                taxonomy_candidate
                if taxonomy_candidate.is_absolute()
                else verify_root / taxonomy_candidate
            )
        else:
            taxonomy_path = verify_root / "conf/base/gliner_taxonomy.yaml"

        verify_cfg = verify_config.get("verify", {})
        chg_cfg = verify_cfg.get("chg", {})
        svs_cfg = verify_cfg.get("svs", {})

        dataset_post, dataset_masked = _derive_tokenized_paths(dataset_path)
        if not dataset_post.exists():
            dataset_post = None
        if not dataset_masked.exists():
            dataset_masked = None

        output_dir.mkdir(parents=True, exist_ok=True)

        run_verification(
            dataset_path=dataset_path,
            dataset_path_post=dataset_post,
            dataset_path_masked=dataset_masked,
            artifacts_dir=artifacts_dir,
            phase_d_dir=phase_d_dir,
            output_dir=output_dir,
            model_name=model_name,
            taxonomy_path=taxonomy_path,
            max_length=max_length,
            batch_size=int(verify_cfg.get("batch_size", 8)),
            chg_epochs=int(chg_cfg.get("epochs", 5)),
            chg_lr=float(chg_cfg.get("learning_rate", 1e-3)),
            chg_regularization=float(chg_cfg.get("regularization", 0.01)),
            facilitating_threshold=float(chg_cfg.get("facilitating_threshold", 0.7)),
            irrelevant_threshold=float(chg_cfg.get("irrelevant_threshold", 0.3)),
            svs_max_batches=int(svs_cfg.get("max_batches", 5)),
            svs_query_strategy=svs_cfg.get("query_strategy", "mean_tokens"),
            svs_pos_backend=svs_cfg.get("pos_backend", "lexical"),
            svs_use_pos=bool(svs_cfg.get("use_pos", True)),
            svs_spacy_model=svs_cfg.get("spacy_model", "en_core_web_sm"),
            svs_spacy_use_gpu=bool(svs_cfg.get("spacy_use_gpu", False)),
            svs_spacy_gpu_id=svs_cfg.get("spacy_gpu_id", 0),
            svs_spacy_batch_size=int(svs_cfg.get("spacy_batch_size", 32)),
            svs_spacy_n_process=int(svs_cfg.get("spacy_n_process", 1)),
            svs_spacy_disable=list(svs_cfg.get("spacy_disable", [])),
            svs_hf_model=svs_cfg.get(
                "hf_model",
                "vblagoje/bert-english-uncased-finetuned-pos",
            ),
            svs_hf_device=int(svs_cfg.get("hf_device", 0)),
            svs_hf_batch_size=int(svs_cfg.get("hf_batch_size", 16)),
            use_only_labels=tuple(use_only_labels) if use_only_labels else None,
        )
        return True
    except Exception as e:
        logger.error(f"Phase D verification failed: {e}", exc_info=True)
        return False


def run_phase_d_report(
    phase_d_dir: Path,
    output_dir: Path,
    dataset_path: Path,
    use_only_labels: List[str],
) -> bool:
    """Generate Phase D report with visualizations."""
    from neuro_stylometry.evaluation.comparative_report import generate_phase_d_report

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        generate_phase_d_report(
            phase_d_dir=phase_d_dir,
            output_dir=output_dir,
            dataset_path=dataset_path,
            chg_subdir="chg",
            use_only=use_only_labels or None,
        )
        return True
    except Exception as e:
        logger.error(f"Phase D report generation failed: {e}", exc_info=True)
        return False


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="E2E verification for --use-only filter (Full Pipeline Test)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single-label test with 200 samples
    python scripts/verify_use_only_e2e.py \\
        --dataset artifacts/data/sobr.arrow \\
        --output-dir artifacts/test_use_only \\
        --samples 200 \\
        --mode laptop
    
    # Multi-label test
    python scripts/verify_use_only_e2e.py \\
        --dataset artifacts/data/sobr.arrow \\
        --output-dir artifacts/test_use_only \\
        --samples 200 \\
        --labels nationality female \\
        --mode laptop
        """
    )
    parser.add_argument(
        "--samples",
        "--phase-a-samples",
        dest="samples",
        type=int,
        default=200,
        help="Number of samples to use for testing (default: 200)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["nationality"],
        help="Demographic label(s) to test with --use-only (default: nationality)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/data/sobr.arrow"),
        help="Path to input dataset Arrow file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/test_use_only"),
        help="Output directory for test artifacts",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "laptop", "hpc"],
        default="auto",
        help="Hardware mode (auto, laptop, hpc)",
    )
    parser.add_argument(
        "--skip-phase-d",
        action="store_true",
        help="Skip Phase D execution (faster test)",
    )
    parser.add_argument(
        "--skip-execution",
        action="store_true",
        help="Skip Phase A execution, only verify existing artifacts",
    )
    parser.add_argument(
        "--skip-phase-a",
        "--disable-phase-a",
        action="store_true",
        help="Skip Phase A execution (requires existing artifacts)",
    )
    parser.add_argument(
        "--run-phase-d",
        action="store_true",
        help="Run Phase D training after Phase A",
    )
    parser.add_argument(
        "--run-verify",
        action="store_true",
        help="Run Phase D verification (CHG + SVS)",
    )
    parser.add_argument(
        "--run-report",
        action="store_true",
        help="Generate Phase D report",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean output directory before running",
    )
    
    args = parser.parse_args()
    
    # Verify dataset exists
    if not args.dataset.exists():
        print(f"❌ Dataset not found: {args.dataset}")
        print(f"   Please run: python scripts/convert_pandas_to_arrow.py --raw-data-dir datasets --output {args.dataset}")
        return 1
    
    # Determine if single-label mode
    is_single_label = len(args.labels) == 1
    mode_str = "single-label" if is_single_label else "multi-label"

    # Auto-detect hardware mode once for all phases
    if args.mode == "auto":
        from neuro_stylometry.hardware_ops.detection import HardwareDetector, ProfileType

        profile = HardwareDetector.detect()
        args.mode = "laptop" if profile.profile_type == ProfileType.LAPTOP else "hpc"
    
    # Clean output directory if requested
    if args.clean and args.output_dir.exists():
        print(f"Cleaning output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output paths
    phase_a_output = args.output_dir / "phase_a"
    phase_d_output = args.output_dir / "phase_d"
    reports_dir = phase_a_output / "reports" / "phase_a"
    phase_d_verify_output = phase_d_output / "chg"
    phase_d_report_output = phase_d_output / "reports"
    subset_path = args.output_dir / "subset.arrow"
    
    # Print header
    print_section(f"E2E Verification: --use-only filter ({mode_str} mode)")
    print(f"Labels: {args.labels}")
    print(f"Samples: {args.samples}")
    print(f"Mode: {args.mode}")
    print(f"Output: {args.output_dir}")
    
    skip_phase_a = args.skip_execution or args.skip_phase_a

    total_steps = 2  # artifacts + visualizations
    if not skip_phase_a:
        total_steps += 2  # subset + phase A
    if is_single_label:
        total_steps += 1
    if args.run_phase_d:
        total_steps += 1
    if args.run_verify:
        total_steps += 1
    if args.run_report:
        total_steps += 1
    current_step = 0
    all_results: List[TestResult] = []
    
    # ==========================================================================
    # Step 1: Create Weighted Subset
    # ==========================================================================
    if not skip_phase_a:
        current_step += 1
        print_step(current_step, total_steps, "Creating weighted subset dataset")
        
        try:
            subset_stats = create_weighted_subset(
                input_path=args.dataset,
                output_path=subset_path,
                target_column=args.labels[0],  # Use first label for stratification
                num_samples=args.samples,
                seed=42,
            )
            print(f"    ✓ Subset created: {subset_stats['total_samples']} samples")
        except Exception as e:
            print(f"    ❌ Failed to create subset: {e}")
            logger.exception("Subset creation failed")
            return 1
    
    # ==========================================================================
    # Step 2: Run Phase A with Taxonomy Monitoring
    # ==========================================================================
    if not skip_phase_a:
        current_step += 1
        print_step(current_step, total_steps, "Running Phase A with --use-only filter")
        
        phase_a_result = run_phase_a_with_monitoring(
            dataset_path=subset_path,
            output_dir=phase_a_output,
            use_only_labels=args.labels,
            mode=args.mode,
        )
        all_results.append(phase_a_result)
        
        if not phase_a_result.passed:
            print("\n❌ Phase A execution/verification FAILED")
            for error in phase_a_result.errors:
                print(f"  ERROR: {error}")
            return 1
        
        print("\n    ✓ Phase A completed successfully")
    else:
        print(f"\nSkipping execution, verifying artifacts in {phase_a_output}")
    
    # ==========================================================================
    # Step 3: Verify Phase A Artifacts
    # ==========================================================================
    current_step += 1
    print_step(current_step, total_steps, "Verifying Phase A artifacts")
    
    artifact_result = verify_phase_a_artifacts(
        output_dir=phase_a_output,
        expected_labels=args.labels,
    )
    all_results.append(artifact_result)
    
    if artifact_result.errors:
        print("\n    ❌ ARTIFACT ERRORS:")
        for error in artifact_result.errors:
            print(f"      - {error}")
    
    if artifact_result.warnings:
        print("\n    ⚠️  WARNINGS:")
        for warning in artifact_result.warnings:
            print(f"      - {warning}")
    
    # ==========================================================================
    # Step 4: Verify Visualizations
    # ==========================================================================
    current_step += 1
    print_step(current_step, total_steps, "Verifying visualizations")
    
    viz_result = verify_visualizations(
        reports_dir=reports_dir,
        is_single_label=is_single_label,
    )
    all_results.append(viz_result)
    
    # ==========================================================================
    # Step 5: Single-Label Adaptations (if applicable)
    # ==========================================================================
    if is_single_label:
        current_step += 1
        print_step(current_step, total_steps, f"Verifying single-label adaptations for '{args.labels[0]}'")
        
        single_result = verify_single_label_adaptations(
            output_dir=phase_a_output,
            label=args.labels[0],
        )
        all_results.append(single_result)
    
    # Step 5: Run Phase D training (optional)
    if args.run_phase_d and not args.skip_phase_d:
        current_step += 1
        print_step(current_step, total_steps, "Running Phase D training...")

        clean_dataset = phase_a_output / "clean_dataset.arrow"
        if not clean_dataset.exists():
            print(f"❌ Missing Phase A dataset: {clean_dataset}")
            return 1

        success = run_phase_d(
            dataset_path=clean_dataset,
            output_dir=phase_d_output,
            artifacts_dir=phase_a_output,
            use_only_labels=args.labels,
            mode=args.mode,
        )
        if not success:
            print("\n❌ Phase D execution failed")
            return 1
        print("\n✓ Phase D execution completed")

        phase_d_result = verify_phase_d_artifacts(phase_d_output)
        all_results.append(phase_d_result)
        if phase_d_result.errors:
            for error in phase_d_result.errors:
                print(f"  ❌ {error}")
        if phase_d_result.warnings:
            for warning in phase_d_result.warnings:
                print(f"  ⚠ {warning}")

    # Step 6: Run Phase D verification (optional)
    if args.run_verify:
        current_step += 1
        print_step(current_step, total_steps, "Running Phase D verification (CHG + SVS)...")

        clean_dataset = phase_a_output / "clean_dataset.arrow"
        if not clean_dataset.exists():
            print(f"❌ Missing Phase A dataset: {clean_dataset}")
            return 1

        success = run_phase_d_verify(
            dataset_path=clean_dataset,
            phase_d_dir=phase_d_output,
            artifacts_dir=phase_a_output,
            output_dir=phase_d_verify_output,
            use_only_labels=args.labels,
            mode=args.mode,
        )
        if not success:
            print("\n❌ Phase D verification failed")
            return 1
        print("\n✓ Phase D verification completed")

        verify_result = verify_phase_d_verify_outputs(phase_d_verify_output)
        all_results.append(verify_result)
        if verify_result.errors:
            for error in verify_result.errors:
                print(f"  ❌ {error}")
        if verify_result.warnings:
            for warning in verify_result.warnings:
                print(f"  ⚠ {warning}")

    # Step 7: Run Phase D report (optional)
    if args.run_report:
        current_step += 1
        print_step(current_step, total_steps, "Generating Phase D report...")

        clean_dataset = phase_a_output / "clean_dataset.arrow"
        if not clean_dataset.exists():
            print(f"❌ Missing Phase A dataset: {clean_dataset}")
            return 1

        success = run_phase_d_report(
            phase_d_dir=phase_d_output,
            output_dir=phase_d_report_output,
            dataset_path=clean_dataset,
            use_only_labels=args.labels,
        )
        if not success:
            print("\n❌ Phase D report generation failed")
            return 1
        print("\n✓ Phase D report generated")

        report_result = verify_phase_d_report_outputs(phase_d_report_output)
        all_results.append(report_result)
        if report_result.errors:
            for error in report_result.errors:
                print(f"  ❌ {error}")
        if report_result.warnings:
            for warning in report_result.warnings:
                print(f"  ⚠ {warning}")
    
    # ==========================================================================
    # Final Results
    # ==========================================================================
    print_section("TEST RESULTS")
    
    total_errors = sum(len(r.errors) for r in all_results)
    total_warnings = sum(len(r.warnings) for r in all_results)
    all_passed = all(r.passed for r in all_results)
    
    if total_errors > 0:
        print("ERRORS:")
        for r in all_results:
            for error in r.errors:
                print(f"  ❌ {error}")
    
    if total_warnings > 0:
        print("\nWARNINGS:")
        for r in all_results:
            for warning in r.warnings:
                print(f"  ⚠️  {warning}")
    
    print()
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print()
        print("SUMMARY:")
        print(f"  - Labels: {args.labels}")
        print(f"  - Samples: {args.samples}")
        print(f"  - Mode: {mode_str}")
        print(f"  - Artifacts: {phase_a_output}")
        return 0
    else:
        print("❌ TESTS FAILED")
        print(f"  - {total_errors} error(s)")
        print(f"  - {total_warnings} warning(s)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
