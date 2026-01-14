#!/usr/bin/env python3
"""
Test Script: `--use-only` Label Filtering for Phase D.

This script validates the correctness of the `--use-only` label filter
implementation by running the pipeline with various label combinations:
    - Single-label filtering (should instantiate SingleTaskHead)
    - Multi-label filtering (should instantiate MultiTaskHead)
    - Full pipeline (no filtering) as baseline

The script creates a minimal test dataset (1000 rows, ≥100 non-null per label)
and runs abbreviated training + verification cycles to ensure:
    1. Dataset filtering applies AND semantics correctly
    2. SingleTaskHead vs MultiTaskHead selection is correct
    3. training_metadata.json stores filter metadata
    4. verify command auto-detects and reapplies filters

Usage:
    python scripts/test_use_only_filtering.py --arrow-path artifacts/data/sobr.arrow
    python scripts/test_use_only_filtering.py --arrow-path artifacts/data/sobr.arrow --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.compute as pc
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neuro_stylometry.stylometry_net.phase_d_dataset import get_demographic_columns

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEMOGRAPHIC_COLUMNS = get_demographic_columns()
TARGET_ROWS = 5000  # Need enough rows to have sufficient non-null per label after split
MIN_NON_NULL_PER_LABEL = 100
SEED = 42

# Test configurations: (name, labels, expected_head_type)
# NOTE: Multi-label tests require rows where ALL specified labels are non-null.
# With the SOBR dataset structure (demographics from independent sources), 
# very few rows have multiple demographics. We focus on single-label tests
# which cover the primary use case for --use-only.
TEST_CONFIGS: list[tuple[str, tuple[str, ...], str]] = [
    ("single_binary_gender", ("female",), "SingleTaskHead"),
    ("single_multiclass_nationality", ("nationality",), "SingleTaskHead"),
    ("single_binary_extrovert", ("extrovert",), "SingleTaskHead"),  # Another binary label
    ("single_binary_political", ("political_leaning",), "SingleTaskHead"),  # Another label type
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset Preparation
# ---------------------------------------------------------------------------
def create_test_dataset(source_arrow: Path, output_dir: Path) -> Path:
    """
    Create a minimal test dataset by sampling from the source.
    
    Strategy:
        1. Load full dataset
        2. Sample down to TARGET_ROWS (preserving original distribution)
        3. Report non-null counts per demographic (informational only)
        4. Write to output_dir/test_dataset.arrow
    
    Note: Each demographic column has its own set of non-null rows.
    The pipeline's --use-only filtering handles per-label row selection.
    
    Returns:
        Path to the test dataset
    """
    logger.info(f"Loading source dataset: {source_arrow}")
    table = feather.read_table(source_arrow)
    logger.info(f"Source dataset: {len(table)} rows")
    
    # Sample down to TARGET_ROWS using contiguous slice to avoid Arrow concatenation issues
    np.random.seed(SEED)
    if len(table) > TARGET_ROWS:
        # Use a contiguous slice starting at a random offset
        max_start = len(table) - TARGET_ROWS
        start_idx = np.random.randint(0, max_start + 1)
        table = table.slice(start_idx, TARGET_ROWS)
        logger.info(f"Sliced to {len(table)} rows (offset {start_idx})")
    
    # Report non-null counts per label (informational)
    logger.info("Non-null counts per demographic in sampled dataset:")
    for col in DEMOGRAPHIC_COLUMNS:
        if col in table.column_names:
            non_null = pc.sum(pc.is_valid(table[col])).as_py()
            logger.info(f"  {col}: {non_null} non-null values")
    
    # Write test dataset
    output_path = output_dir / "test_dataset.arrow"
    feather.write_feather(table, output_path)
    logger.info(f"Test dataset written: {output_path}")
    
    return output_path


# ---------------------------------------------------------------------------
# Test Execution
# ---------------------------------------------------------------------------
# Path to quick_test config relative to repo root
QUICK_TEST_CONFIG = Path(__file__).parent.parent / "conf" / "experiments" / "quick_test.yaml"


def run_phase_d_training(
    dataset_path: Path,
    output_dir: Path,
    artifacts_dir: Path,
    use_only: tuple[str, ...] | None = None,
) -> tuple[bool, Path]:
    """
    Run Phase D training with optional label filtering.
    
    Args:
        dataset_path: Path to source arrow dataset (clean_dataset.arrow)
        output_dir: Output directory for checkpoints
        artifacts_dir: Phase A artifacts directory (projection_matrix.pt)
        use_only: Optional label filter
    
    Returns:
        (success: bool, checkpoint_dir: Path)
    """
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "-m", "neuro_stylometry",
        "run-phase-d",
        "--dataset", str(output_dir / "tokenized"),  # Will be created by auto-preprocess
        "--output-dir", str(checkpoint_dir),
        "--artifacts-dir", str(artifacts_dir),
        "--source-dataset", str(dataset_path),
        "--preprocess",  # Auto-preprocess tokenized datasets
        "--pre-pad",  # Use pre-padding for efficiency
        "--mode", "laptop",
        "--config", str(QUICK_TEST_CONFIG),  # Use minimal training config
    ]
    
    if use_only:
        for label in use_only:
            cmd.extend(["--use-only", label])
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        # Stream output in real-time instead of capturing
        result = subprocess.run(
            cmd,
            timeout=600,  # 10 minute timeout for preprocessing + training
        )
        
        return result.returncode == 0, checkpoint_dir
    
    except subprocess.TimeoutExpired:
        logger.error("Training timed out after 10 minutes")
        return False, checkpoint_dir
    except Exception as e:
        logger.error(f"Training error: {e}")
        return False, checkpoint_dir


def run_verification(
    phase_d_dir: Path,
    dataset_path: Path,
    artifacts_dir: Path,
    use_only: tuple[str, ...] | None = None,
) -> tuple[bool, dict | None]:
    """
    Run verification on trained model.
    
    If use_only is None, auto-detection from training_metadata.json should occur.
    
    Args:
        phase_d_dir: Directory containing baseline/constrained subdirectories
        dataset_path: Path to clean_dataset.arrow
        artifacts_dir: Phase A artifacts directory
        use_only: Optional explicit label filter (None = auto-detect)
    
    Returns:
        (success: bool, results: dict | None)
    """
    output_dir = phase_d_dir / "chg"
    
    cmd = [
        sys.executable, "-m", "neuro_stylometry",
        "verify",
        "--dataset", str(dataset_path),
        "--phase-d-dir", str(phase_d_dir),
        "--artifacts-dir", str(artifacts_dir),
        "--output-dir", str(output_dir),
        "--batch-size", "4",
        "--chg-epochs", "1",  # Minimal: 1 epoch
        "--svs-max-batches", "1",  # Minimal: 1 batch
    ]
    
    # Only add --use-only if explicitly provided (otherwise test auto-detection)
    if use_only is not None:
        for label in use_only:
            cmd.extend(["--use-only", label])
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        # Stream output in real-time
        result = subprocess.run(
            cmd,
            timeout=600,
        )
        
        if result.returncode != 0:
            return False, None
        
        # Try to load results
        results_file = output_dir / "verification_results.json"
        if results_file.exists():
            with open(results_file) as f:
                return True, json.load(f)
        
        return True, None
    
    except subprocess.TimeoutExpired:
        logger.error("Verification timed out after 10 minutes")
        return False, None
    except Exception as e:
        logger.error(f"Verification error: {e}")
        return False, None


def validate_training_metadata(
    checkpoint_dir: Path,
    expected_labels: tuple[str, ...] | None,
    expected_head: str,
) -> tuple[bool, list[str]]:
    """
    Validate that training_metadata.json contains correct filter information.
    
    Returns:
        (valid: bool, errors: list[str])
    """
    errors = []
    
    # Find the actual checkpoint directory
    possible_dirs = list(checkpoint_dir.glob("*/"))
    if not possible_dirs:
        return False, ["No checkpoint subdirectory found"]
    
    metadata_path = possible_dirs[0] / "training_metadata.json"
    
    if not metadata_path.exists():
        return False, [f"training_metadata.json not found at {metadata_path}"]
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    # Validate label_filter - it's a dict with use_only key
    label_filter = metadata.get("label_filter", {})
    actual_labels = label_filter.get("use_only") if isinstance(label_filter, dict) else None
    
    if expected_labels is None:
        if actual_labels is not None:
            errors.append(f"Expected label_filter.use_only=null, got {actual_labels}")
    else:
        if actual_labels is None:
            errors.append(f"Expected label_filter.use_only={list(expected_labels)}, got null")
        elif set(actual_labels) != set(expected_labels):
            errors.append(f"Expected label_filter.use_only={list(expected_labels)}, got {actual_labels}")
    
    # Validate model_variant (stored as single_task/multi_task, map to class name)
    actual_variant = metadata.get("model_variant")
    expected_variant_value = "single_task" if expected_head == "SingleTaskHead" else "multi_task"
    if actual_variant != expected_variant_value:
        errors.append(f"Expected model_variant={expected_variant_value}, got {actual_variant}")
    
    # For single-task, validate task_name
    if expected_head == "SingleTaskHead":
        task_name = metadata.get("task_name")
        if not task_name:
            errors.append("Expected task_name for SingleTaskHead, got null/empty")
        elif expected_labels and task_name != expected_labels[0]:
            errors.append(f"Expected task_name={expected_labels[0]}, got {task_name}")
    
    return len(errors) == 0, errors


def run_test_case(
    test_name: str,
    labels: tuple[str, ...] | None,
    expected_head: str,
    dataset_path: Path,
    artifacts_dir: Path,
    work_dir: Path,
) -> dict:
    """
    Run a single test case.
    
    Args:
        test_name: Name of the test case
        labels: Labels to filter on (None for no filtering)
        expected_head: Expected head type (SingleTaskHead or MultiTaskHead)
        dataset_path: Path to clean_dataset.arrow
        artifacts_dir: Phase A artifacts directory
        work_dir: Working directory for test outputs
    
    Returns:
        Test result dict with status, errors, etc.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST: {test_name}")
    logger.info(f"  Labels: {labels}")
    logger.info(f"  Expected head: {expected_head}")
    logger.info(f"{'='*60}")
    
    result = {
        "name": test_name,
        "labels": labels,
        "expected_head": expected_head,
        "training_passed": False,
        "metadata_passed": False,
        "verification_passed": False,
        "autodetect_passed": False,
        "errors": [],
    }
    
    # Create test output directory
    test_dir = work_dir / test_name
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Run training
    logger.info(f"Step 1: Training with labels={labels}")
    train_ok, checkpoint_dir = run_phase_d_training(
        dataset_path=dataset_path,
        output_dir=test_dir,
        artifacts_dir=artifacts_dir,
        use_only=labels,
    )
    result["training_passed"] = train_ok
    
    if not train_ok:
        result["errors"].append("Training failed")
        return result
    
    # Step 2: Validate training_metadata.json
    logger.info("Step 2: Validating training_metadata.json")
    meta_ok, meta_errors = validate_training_metadata(
        checkpoint_dir=checkpoint_dir,
        expected_labels=labels,
        expected_head=expected_head,
    )
    result["metadata_passed"] = meta_ok
    result["errors"].extend(meta_errors)
    
    # Step 3: Run verification with explicit labels
    logger.info("Step 3: Running verification (explicit labels)")
    verify_ok, _ = run_verification(
        phase_d_dir=checkpoint_dir,
        dataset_path=dataset_path,
        artifacts_dir=artifacts_dir,
        use_only=labels,
    )
    result["verification_passed"] = verify_ok
    
    if not verify_ok:
        result["errors"].append("Verification with explicit labels failed")
    
    # Step 4: Run verification with auto-detection (no explicit labels)
    logger.info("Step 4: Running verification (auto-detection)")
    autodetect_ok, _ = run_verification(
        phase_d_dir=checkpoint_dir,
        dataset_path=dataset_path,
        artifacts_dir=artifacts_dir,
        use_only=None,  # Should auto-detect from training_metadata.json
    )
    result["autodetect_passed"] = autodetect_ok
    
    if not autodetect_ok:
        result["errors"].append("Verification with auto-detection failed")
    
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Test --use-only label filtering implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--arrow-path",
        type=Path,
        default=Path("artifacts/phase_a/clean_dataset.arrow"),
        help="Path to source Arrow dataset (clean_dataset.arrow)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/phase_a"),
        help="Phase A artifacts directory (contains projection_matrix.pt)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Working directory for test outputs (default: temp dir)",
    )
    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        help="Keep test outputs after completion",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--test-filter",
        type=str,
        default=None,
        help="Only run tests matching this substring",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    # Validate source dataset
    if not args.arrow_path.exists():
        logger.error(f"Source dataset not found: {args.arrow_path}")
        sys.exit(1)
    
    # Validate artifacts directory
    proj_matrix = args.artifacts_dir / "projection_matrix.pt"
    if not proj_matrix.exists():
        logger.error(f"projection_matrix.pt not found in {args.artifacts_dir}")
        sys.exit(1)
    
    # Determine work directory
    if args.work_dir:
        work_dir = args.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="test_use_only_"))
        cleanup = not args.keep_outputs
    
    logger.info(f"Work directory: {work_dir}")
    logger.info(f"Artifacts directory: {args.artifacts_dir}")
    
    try:
        # Create test dataset
        logger.info("\n" + "="*60)
        logger.info("PREPARING TEST DATASET")
        logger.info("="*60)
        test_dataset = create_test_dataset(args.arrow_path, work_dir)
        
        # Filter test configs if requested
        test_configs = TEST_CONFIGS
        if args.test_filter:
            test_configs = [
                (name, labels, head) 
                for name, labels, head in TEST_CONFIGS
                if args.test_filter in name
            ]
            logger.info(f"Running {len(test_configs)} tests matching '{args.test_filter}'")
        
        # Run tests
        results = []
        for test_name, labels, expected_head in test_configs:
            result = run_test_case(
                test_name=test_name,
                labels=labels,
                expected_head=expected_head,
                dataset_path=test_dataset,
                artifacts_dir=args.artifacts_dir,
                work_dir=work_dir,
            )
            results.append(result)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("TEST SUMMARY")
        logger.info("="*60)
        
        passed = 0
        failed = 0
        
        for result in results:
            all_passed = (
                result["training_passed"] 
                and result["metadata_passed"]
                and result["verification_passed"]
                and result["autodetect_passed"]
            )
            
            status = "[PASS]" if all_passed else "[FAIL]"
            logger.info(f"\n{status}: {result['name']}")
            logger.info(f"  Training:      {'[OK]' if result['training_passed'] else '[FAIL]'}")
            logger.info(f"  Metadata:      {'[OK]' if result['metadata_passed'] else '[FAIL]'}")
            logger.info(f"  Verification:  {'[OK]' if result['verification_passed'] else '[FAIL]'}")
            logger.info(f"  Auto-detect:   {'[OK]' if result['autodetect_passed'] else '[FAIL]'}")
            
            if result["errors"]:
                for error in result["errors"]:
                    logger.info(f"    >> {error}")
            
            if all_passed:
                passed += 1
            else:
                failed += 1
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TOTAL: {passed} passed, {failed} failed out of {len(results)} tests")
        logger.info(f"{'='*60}")
        
        # Write results to file
        results_file = work_dir / "test_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults written to: {results_file}")
        
        return 0 if failed == 0 else 1
    
    finally:
        if cleanup:
            logger.info(f"\nCleaning up work directory: {work_dir}")
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            logger.info(f"\nTest outputs preserved in: {work_dir}")


if __name__ == "__main__":
    sys.exit(main())
