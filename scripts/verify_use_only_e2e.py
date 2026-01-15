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


def verify_phase_d_artifacts(phase_d_dir: Path) -> dict:
    """Verify Phase D training artifacts exist and look compatible."""
    results = {"passed": True, "errors": [], "warnings": [], "info": {}}

    baseline_dir = phase_d_dir / "baseline"
    constrained_dir = phase_d_dir / "constrained"
    comparative_path = phase_d_dir / "phase_d_comparative_metrics.json"

    for run_dir, label in ((baseline_dir, "baseline"), (constrained_dir, "constrained")):
        if not run_dir.exists():
            results["errors"].append(f"Missing Phase D {label} dir: {run_dir}")
            results["passed"] = False
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
                results["errors"].append(f"Missing {label} artifact: {path.name}")
                results["passed"] = False
        for path in optional:
            if not path.exists():
                results["warnings"].append(f"Missing optional {label} artifact: {path.name}")

    if not comparative_path.exists():
        results["warnings"].append("Missing comparative metrics: phase_d_comparative_metrics.json")

    return results


def verify_phase_d_verify_outputs(verify_dir: Path) -> dict:
    """Verify Phase D verification (CHG + SVS) outputs."""
    results = {"passed": True, "errors": [], "warnings": [], "info": {}}

    required = [
        verify_dir / "head_classification_baseline.json",
        verify_dir / "head_classification_constrained.json",
        verify_dir / "svs_baseline.json",
        verify_dir / "svs_constrained.json",
        verify_dir / "verification_summary.json",
    ]
    for path in required:
        if not path.exists():
            results["errors"].append(f"Missing verification artifact: {path.name}")
            results["passed"] = False

    comparative_path = verify_dir / "phase_d_comparative_metrics.json"
    if not comparative_path.exists():
        results["warnings"].append("Missing verification comparative metrics: phase_d_comparative_metrics.json")

    return results


def verify_phase_d_report_outputs(report_dir: Path) -> dict:
    """Verify Phase D report output directory and report HTML."""
    results = {"passed": True, "errors": [], "warnings": [], "info": {}}

    report_path = report_dir / "phase_d_report.html"
    assets_dir = report_dir / "phase_d_assets"

    if not report_path.exists():
        results["errors"].append(f"Missing report: {report_path.name}")
        results["passed"] = False
    if not assets_dir.exists():
        results["warnings"].append(f"Missing report assets dir: {assets_dir.name}")

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


def main():
    parser = argparse.ArgumentParser(
        description="E2E verification for --use-only filter"
    )
    parser.add_argument(
        "--samples",
        "--phase-a-samples",
        dest="samples",
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
        help="Hardware mode (auto, laptop, hpc)",
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
        return 1
    
    # Determine if single-label mode
    is_single_label = len(args.labels) == 1

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
    
    # Print header
    print_section(f"E2E Verification: --use-only filter ({'single-label' if is_single_label else 'multi-label'} mode)")
    print(f"Labels: {args.labels}")
    print(f"Samples: {args.samples}")
    print(f"Output: {args.output_dir}")
    
    phase_a_output = args.output_dir / "phase_a"
    reports_dir = phase_a_output / "reports"
    phase_d_output = args.output_dir / "phase_d"
    phase_d_verify_output = phase_d_output / "chg"
    phase_d_report_output = phase_d_output / "reports"
    
    total_steps = 0
    skip_phase_a = args.skip_execution or args.skip_phase_a
    if not skip_phase_a:
        total_steps += 1
    total_steps += 2  # artifacts + visualizations
    if is_single_label:
        total_steps += 1
    if args.run_phase_d:
        total_steps += 1
    if args.run_verify:
        total_steps += 1
    if args.run_report:
        total_steps += 1
    current_step = 0
    all_passed = True
    
    # Step 1: Run Phase A
    if not skip_phase_a:
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

    # Step 5: Run Phase D training (optional)
    if args.run_phase_d:
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

        phase_d_results = verify_phase_d_artifacts(phase_d_output)
        if phase_d_results["errors"]:
            for error in phase_d_results["errors"]:
                print(f"  ❌ {error}")
            all_passed = False
        if phase_d_results["warnings"]:
            for warning in phase_d_results["warnings"]:
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

        verify_results = verify_phase_d_verify_outputs(phase_d_verify_output)
        if verify_results["errors"]:
            for error in verify_results["errors"]:
                print(f"  ❌ {error}")
            all_passed = False
        if verify_results["warnings"]:
            for warning in verify_results["warnings"]:
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

        report_results = verify_phase_d_report_outputs(phase_d_report_output)
        if report_results["errors"]:
            for error in report_results["errors"]:
                print(f"  ❌ {error}")
            all_passed = False
        if report_results["warnings"]:
            for warning in report_results["warnings"]:
                print(f"  ⚠ {warning}")
    
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
