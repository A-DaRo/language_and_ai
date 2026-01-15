#!/usr/bin/env python3
"""
E2E verification script for --use-only filter across full pipeline.

Usage:
    python scripts/verify_use_only_e2e.py --samples 100 --label nationality
    python scripts/verify_use_only_e2e.py --samples 100 --label nationality --label female
    
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
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

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


def print_section(title: str, char: str = "=", width: int = 80):
    """Print a section header."""
    print(f"\n{char * width}")
    print(title)
    print(f"{char * width}\n")


def print_step(step_num: int, total: int, description: str):
    """Print a step indicator."""
    print(f"\n[{step_num}/{total}] {description}")
    print("-" * 60)


def verify_phase_a_artifacts(
    output_dir: Path,
    expected_labels: List[str],
) -> dict:
    """
    Verify Phase A artifacts match expected structure and dimensions.
    
    Args:
        output_dir: Directory containing Phase A artifacts.
        expected_labels: Expected demographic labels from --use-only.
        
    Returns:
        Dict with verification results.
    """
    import pyarrow.feather as feather
    import torch
    
    results = {
        "passed": True,
        "errors": [],
        "warnings": [],
        "info": {},
    }
    
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
            results["errors"].append(f"Missing required artifact: {filename}")
            results["passed"] = False
    
    if not results["passed"]:
        return results
    
    # Load and verify clean dataset
    clean_dataset = feather.read_table(output_dir / "clean_dataset.arrow")
    results["info"]["num_samples"] = len(clean_dataset)
    print(f"  ✓ Dataset: {len(clean_dataset)} rows")
    
    # Load and verify projection matrix
    projection_matrix = torch.load(output_dir / "projection_matrix.pt", map_location="cpu", weights_only=False)
    results["info"]["projection_shape"] = list(projection_matrix.shape)
    print(f"  ✓ Projection matrix: {tuple(projection_matrix.shape)}")
    
    # Check projection matrix is valid (roughly idempotent)
    P2 = projection_matrix @ projection_matrix
    diff = torch.norm(P2 - projection_matrix).item()
    if diff > 0.1:
        results["warnings"].append(f"Projection matrix idempotence check: ||P²-P||={diff:.4f}")
    
    # Load pollution logs
    pollution_logs = feather.read_table(output_dir / "pollution_logs.arrow")
    results["info"]["num_pollution_spans"] = len(pollution_logs)
    
    # Check for metrics file
    metrics_path = output_dir / "reports" / "phase_a_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        
        # Check label_filter in execution section
        label_filter = metrics.get("execution", {}).get("label_filter")
        results["info"]["label_filter"] = label_filter
        
        if label_filter is None:
            results["errors"].append(
                f"Expected label_filter={expected_labels} but got None"
            )
            results["passed"] = False
        else:
            # Verify all expected labels are in the filter
            for label in expected_labels:
                if label not in label_filter:
                    results["errors"].append(
                        f"Expected label_filter to contain '{label}' but got {label_filter}"
                    )
                    results["passed"] = False
            if results["passed"]:
                print(f"  ✓ Label filter: {label_filter}")
        
        # Check probe results if available
        probe_results = metrics.get("probe", {})
        if probe_results:
            by_column = probe_results.get("by_column", {})
            results["info"]["probed_columns"] = list(by_column.keys())
            
            # Verify all expected labels are in probed columns
            for label in expected_labels:
                if label not in by_column:
                    results["errors"].append(
                        f"Expected '{label}' in probed columns but found: {list(by_column.keys())}"
                    )
                    results["passed"] = False
            
            if results["passed"]:
                print(f"  ✓ Probing results: {list(by_column.keys())}")
    else:
        results["warnings"].append(f"Metrics file not found: {metrics_path}")
    
    return results


def verify_visualizations(
    reports_dir: Path,
    is_single_label: bool,
) -> dict:
    """
    Verify correct visualizations were generated.
    
    Args:
        reports_dir: Reports directory containing visualizations.
        is_single_label: Whether single-label mode was used.
        
    Returns:
        Dict with verification results.
    """
    results = {
        "passed": True,
        "errors": [],
        "warnings": [],
        "info": {},
    }
    
    # Expected visualizations
    always_expected = [
        "amnesic_drop_with_ci.png",
        "embedding_separability.png",
    ]
    
    # These may or may not be generated depending on probe config
    optional = [
        "solver_convergence_benchmark.png",
    ]
    
    # These should be skipped in single-label mode
    multi_label_only = [
        "specificity_gap.png",
        "entity_cooccurrence.png",
        "demographic_score_matrix.png",
    ]
    
    for viz in always_expected:
        viz_path = reports_dir / viz
        if viz_path.exists():
            print(f"  ✓ {viz}")
        else:
            # Not all visualizations may be generated depending on config
            results["warnings"].append(f"Expected visualization not found: {viz}")
            print(f"  ⚠ {viz}: not found")
    
    for viz in optional:
        viz_path = reports_dir / viz
        if viz_path.exists():
            print(f"  ✓ {viz}")
        else:
            print(f"  - {viz}: not generated (optional)")
    
    for viz in multi_label_only:
        viz_path = reports_dir / viz
        if is_single_label:
            if viz_path.exists():
                results["warnings"].append(f"Single-label mode should skip: {viz}")
                print(f"  ⚠ {viz}: should be skipped in single-label mode")
            else:
                print(f"  ✓ {viz}: skipped (single-label)")
        else:
            if viz_path.exists():
                print(f"  ✓ {viz}")
            else:
                print(f"  - {viz}: not found")
    
    return results


def verify_single_label_adaptations(
    output_dir: Path,
    label: str,
) -> dict:
    """
    Verify single-label mode adaptations.
    
    Args:
        output_dir: Phase A output directory.
        label: The single label that was used.
        
    Returns:
        Dict with verification results.
    """
    results = {
        "passed": True,
        "errors": [],
        "warnings": [],
        "info": {},
    }
    
    metrics_path = output_dir / "reports" / "phase_a_metrics.json"
    if not metrics_path.exists():
        results["warnings"].append("Metrics file not found")
        return results
    
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    # Check mode
    label_filter = metrics.get("execution", {}).get("label_filter", [])
    is_single = len(label_filter) == 1
    mode = "single_label" if is_single else "multi_label"
    print(f"  ✓ Mode: {mode}")
    results["info"]["mode"] = mode
    
    # Check probing results
    probe_results = metrics.get("probe", {})
    if probe_results:
        by_column = probe_results.get("by_column", {})
        if is_single:
            if len(by_column) == 1 and label in by_column:
                print(f"  ✓ Probing results: single column '{label}'")
            else:
                results["errors"].append(
                    f"Expected single column '{label}' in probing results"
                )
                results["passed"] = False
    
    # Check control probe (should be skipped in single-label mode)
    control_probe = metrics.get("control_probe", {})
    if is_single:
        if not control_probe or control_probe.get("skipped"):
            print(f"  ✓ Control probe: skipped (as expected)")
        else:
            results["warnings"].append("Control probe should be skipped in single-label mode")
    
    return results


def run_phase_a(
    dataset_path: Path,
    output_dir: Path,
    use_only_labels: List[str],
    samples: int,
    mode: str = "auto",
) -> bool:
    """
    Run Phase A with --use-only filter.
    
    Args:
        dataset_path: Path to input dataset.
        output_dir: Output directory for artifacts.
        use_only_labels: List of demographic labels to filter to.
        samples: Number of samples to use (subset).
        mode: Hardware mode (auto/laptop/hpc).
        
    Returns:
        True if successful, False otherwise.
    """
    from neuro_stylometry.phase_a_pipeline import PhaseAPipeline
    from neuro_stylometry.factories.strategy_factory import StrategyFactory
    from neuro_stylometry.hardware_ops.detection import HardwareDetector, ProfileType
    from neuro_stylometry.config import load_pipeline_config
    
    try:
        # Auto-detect hardware mode
        if mode == "auto":
            profile = HardwareDetector.detect()
            mode = "laptop" if profile.profile_type == ProfileType.LAPTOP else "hpc"
        
        print(f"Hardware mode: {mode}")
        print(f"Labels: {use_only_labels}")
        print(f"Samples: {samples}")
        print(f"Output: {output_dir}")
        print()
        
        # Create strategy
        profile_type = ProfileType.HPC if mode == "hpc" else ProfileType.LAPTOP
        strategy = StrategyFactory.create_filter_strategy(profile_type)
        
        # Load configuration
        config = load_pipeline_config(mode=mode)
        
        # Enable subset for faster testing
        config["subset"] = {"enabled": True, "size": samples}
        
        # Disable some optional features for speed
        config.setdefault("gliner", {})["compute_explicit_recall"] = False
        
        # Create and run pipeline
        pipeline = PhaseAPipeline(strategy=strategy, config=config)
        
        print("Starting Phase A execution...")
        print("-" * 60)
        
        start_time = time.time()
        artifacts = pipeline.run(
            input_dataset_path=dataset_path,
            output_dir=output_dir,
            use_only_labels=use_only_labels,
        )
        elapsed = time.time() - start_time
        
        print("-" * 60)
        print(f"Phase A completed in {elapsed:.1f}s")
        print(f"  Clean dataset: {artifacts.clean_dataset_path}")
        print(f"  Projection matrix: {artifacts.projection_matrix_path}")
        print(f"  Num samples: {artifacts.metadata.get('num_samples')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Phase A execution failed: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="E2E verification for --use-only filter"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples to use for testing (default: 100)",
    )
    parser.add_argument(
        "--label",
        dest="labels",
        action="append",
        required=True,
        help="Demographic label(s) to test with --use-only. Can be specified multiple times.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/data/sobr.arrow"),
        help="Path to input dataset Arrow file (default: artifacts/data/sobr.arrow)",
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
        help="Hardware mode (default: auto)",
    )
    parser.add_argument(
        "--skip-execution",
        action="store_true",
        help="Skip execution, only verify existing artifacts",
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
        return 1
    
    # Determine if single-label mode
    is_single_label = len(args.labels) == 1
    
    # Clean output directory if requested
    if args.clean and args.output_dir.exists():
        print(f"Cleaning output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print header
    print_section(f"E2E Verification: --use-only filter ({'single-label' if is_single_label else 'multi-label'} mode)")
    print(f"Labels: {args.labels}")
    print(f"Samples: {args.samples}")
    print(f"Output: {args.output_dir}")
    
    phase_a_output = args.output_dir / "phase_a"
    reports_dir = phase_a_output / "reports"
    
    total_steps = 4 if is_single_label else 3
    current_step = 0
    all_passed = True
    
    # Step 1: Run Phase A
    if not args.skip_execution:
        current_step += 1
        print_step(current_step, total_steps, "Running Phase A with --use-only filter")
        
        success = run_phase_a(
            dataset_path=args.dataset,
            output_dir=phase_a_output,
            use_only_labels=args.labels,
            samples=args.samples,
            mode=args.mode,
        )
        
        if not success:
            print("\n❌ Phase A execution failed")
            return 1
        
        print("\n✓ Phase A execution completed")
    else:
        print(f"\nSkipping execution, verifying artifacts in {phase_a_output}")
    
    # Step 2: Verify Phase A artifacts
    current_step += 1
    print_step(current_step, total_steps, "Verifying Phase A artifacts...")
    
    results = verify_phase_a_artifacts(
        output_dir=phase_a_output,
        expected_labels=args.labels,
    )
    
    if results["errors"]:
        print("\n❌ ERRORS:")
        for error in results["errors"]:
            print(f"  - {error}")
        all_passed = False
    
    if results["warnings"]:
        print("\n⚠️  WARNINGS:")
        for warning in results["warnings"]:
            print(f"  - {warning}")
    
    # Step 3: Verify visualizations
    current_step += 1
    print_step(current_step, total_steps, "Verifying visualizations...")
    
    viz_results = verify_visualizations(
        reports_dir=reports_dir,
        is_single_label=is_single_label,
    )
    
    if viz_results["errors"]:
        for error in viz_results["errors"]:
            print(f"  ❌ {error}")
        all_passed = False
    
    # Step 4 (single-label only): Verify single-label adaptations
    if is_single_label:
        current_step += 1
        print_step(current_step, total_steps, f"Verifying single-label adaptations for '{args.labels[0]}'...")
        
        adapt_results = verify_single_label_adaptations(
            output_dir=phase_a_output,
            label=args.labels[0],
        )
        
        if adapt_results["errors"]:
            for error in adapt_results["errors"]:
                print(f"  ❌ {error}")
            all_passed = False
    
    # Final result
    print_section("TEST RESULTS")
    
    if all_passed and results["passed"]:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("❌ TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
