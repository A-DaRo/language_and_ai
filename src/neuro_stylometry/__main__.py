# src/neuro_stylometry/__main__.py
"""
Neuro-Symbolic Stylometry Pipeline CLI.

This module provides the command-line interface for the neuro-stylometry pipeline.
All commands use the PhaseAPipeline facade with strict YAML configuration authority.
"""

import click
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
def cli(verbose):
    """Neuro-Symbolic Stylometry Pipeline CLI."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command("run-phase-a")
@click.option(
    "--dataset", 
    type=click.Path(exists=True, path_type=Path), 
    required=True, 
    help="Path to dataset Arrow file"
)
@click.option(
    "--output-dir", 
    type=click.Path(path_type=Path), 
    required=True, 
    help="Output directory for artifacts"
)
@click.option(
    "--mode", 
    type=click.Choice(["auto", "laptop", "hpc"]), 
    default="auto", 
    help="Hardware mode (auto-detected if not specified)"
)
@click.option(
    "--config", 
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
    help="Optional experiment YAML config to merge on top of base+mode configs"
)
@click.option(
    "--dry-run", 
    is_flag=True, 
    help="Print resolved config without execution"
)
@click.option(
    "--skip-chunking",
    is_flag=True,
    help="Skip chunking stage (requires post_chunked column in dataset)"
)
@click.option(
    "--skip-inference",
    is_flag=True,
    help="Skip inference stage (requires inference_results.arrow artifact)"
)
@click.option(
    "--skip-leace",
    is_flag=True,
    help="Skip LEACE stage (requires projection_matrix.pt artifact)"
)
@click.option(
    "--skip-probing",
    is_flag=True,
    help="Skip probing stage (exit after LEACE completion)"
)
@click.option(
    "--force-skip",
    is_flag=True,
    help="Force-skip missing prerequisites (injects execution.force_skip=True in config)"
)
def run_phase_a(
    dataset: Path, 
    output_dir: Path, 
    mode: str, 
    config_path: Path | None, 
    dry_run: bool,
    skip_chunking: bool,
    skip_inference: bool,
    skip_leace: bool,
    skip_probing: bool,
    force_skip: bool,
):
    """Execute Phase A pollution detection and mitigation pipeline."""
    from .phase_a_pipeline import PhaseAPipeline
    from .factories.strategy_factory import StrategyFactory
    from .hardware_ops.detection import HardwareDetector, ProfileType
    from .config import load_pipeline_config
    
    try:
        # Auto-detect hardware mode if needed
        if mode == "auto":
            profile = HardwareDetector.detect()
            mode = "laptop" if profile.profile_type == ProfileType.LAPTOP else "hpc"
            logger.info(f"Auto-detected hardware mode: {mode}")
        
        # Create strategy based on mode
        profile_type = ProfileType.HPC if mode == "hpc" else ProfileType.LAPTOP
        strategy = StrategyFactory.create_filter_strategy(profile_type)
        
        # Load configuration from YAML (strict authority)
        config = load_pipeline_config(
            mode=mode, 
            experiment_config_path=config_path
        )
        
        # Inject skip flags into config (CLI overrides YAML)
        if any([skip_chunking, skip_inference, skip_leace, skip_probing]) or force_skip:
            if "execution" not in config:
                config["execution"] = {}
            if any([skip_chunking, skip_inference, skip_leace, skip_probing]):
                config["execution"]["skip_stages"] = {
                    "skip_chunking": skip_chunking,
                    "skip_inference": skip_inference,
                    "skip_leace": skip_leace,
                    "skip_probing": skip_probing,
                }
                logger.info(f"Skip flags enabled: chunking={skip_chunking}, inference={skip_inference}, "
                           f"leace={skip_leace}, probing={skip_probing}")
            if force_skip:
                config["execution"]["force_skip"] = True
                logger.warning("Force-skip enabled via CLI: missing prerequisites will be bypassed (warnings only)")
        
        # Dry run: print config and exit
        if dry_run:
            import yaml
            click.echo("Resolved Configuration:")
            click.echo("-" * 40)
            click.echo(yaml.dump(config, default_flow_style=False, sort_keys=False))
            return
        
        # Create pipeline with PhaseAPipeline facade
        pipeline = PhaseAPipeline(strategy=strategy, config=config)
        
        # Execute pipeline
        logger.info("Starting Phase A execution")
        artifacts = pipeline.run(
            input_dataset_path=dataset,
            output_dir=output_dir,
        )
        
        # Report results
        click.echo("\n" + "=" * 80)
        click.echo("PHASE A COMPLETE")
        click.echo("=" * 80)
        click.echo(f"Clean dataset: {artifacts.clean_dataset_path}")
        click.echo(f"Projection matrix: {artifacts.projection_matrix_path}")
        click.echo(f"Pollution logs: {artifacts.pollution_logs_path}")
        if artifacts.metrics_path:
            click.echo(f"Metrics: {artifacts.metrics_path}")
        if artifacts.reports_dir:
            click.echo(f"Reports directory: {artifacts.reports_dir}")
        click.echo(f"\nProcessed {artifacts.metadata['num_samples']} samples")
        click.echo(f"Detected {artifacts.metadata['num_pollution_spans']} pollution spans")
        
    except Exception as e:
        logger.error(f"Phase A execution failed: {e}", exc_info=True)
        raise click.ClickException(str(e))


@cli.command("config-show")
@click.option(
    "--mode", 
    type=click.Choice(["auto", "laptop", "hpc"]), 
    default="auto",
    help="Hardware mode"
)
@click.option(
    "--config", 
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    help="Optional experiment config to merge"
)
def config_show(mode: str, config_path: Path | None):
    """Display merged configuration for a given mode."""
    from .config import load_pipeline_config
    from .hardware_ops.detection import HardwareDetector, ProfileType
    import yaml
    
    # Auto-detect mode if needed
    if mode == "auto":
        profile = HardwareDetector.detect()
        mode = "laptop" if profile.profile_type == ProfileType.LAPTOP else "hpc"
        click.echo(f"Auto-detected mode: {mode}")
    
    config = load_pipeline_config(
        mode=mode,
        experiment_config_path=config_path,
    )
    
    click.echo(yaml.dump(config, default_flow_style=False, sort_keys=False))


@cli.command("validate-handover")
@click.option(
    "--artifacts-dir", 
    type=click.Path(exists=True, path_type=Path), 
    required=True,
    help="Phase A artifacts directory"
)
def validate_handover(artifacts_dir: Path):
    """Validate Phase A handover contract compliance."""
    import torch
    import pyarrow.feather as feather
    
    click.echo(f"Validating Phase A artifacts in: {artifacts_dir}")
    
    errors = []
    warnings = []
    
    # Check artifact existence
    clean_dataset_path = artifacts_dir / "clean_dataset.arrow"
    projection_matrix_path = artifacts_dir / "projection_matrix.pt"
    pollution_logs_path = artifacts_dir / "pollution_logs.arrow"
    
    if not clean_dataset_path.exists():
        errors.append(f"Missing: clean_dataset.arrow")
    if not projection_matrix_path.exists():
        errors.append(f"Missing: projection_matrix.pt")
    if not pollution_logs_path.exists():
        errors.append(f"Missing: pollution_logs.arrow")
    
    if errors:
        click.echo("✗ Artifact existence check failed:")
        for error in errors:
            click.echo(f"  - {error}")
        raise click.ClickException("Validation failed: missing artifacts")
    
    click.echo("✓ All required artifacts present")
    
    # Validate projection matrix
    try:
        P = torch.load(projection_matrix_path, map_location="cpu")
        
        # Check shape
        if P.shape != (768, 768):
            errors.append(f"Projection shape: expected (768, 768), got {P.shape}")
        
        # Check dtype
        if P.dtype != torch.float32:
            warnings.append(f"Projection dtype: expected float32, got {P.dtype}")
        
        # Check idempotence
        P2 = P @ P
        rel_error = torch.norm(P2 - P) / torch.norm(P)
        if rel_error >= 1e-5:
            errors.append(f"Idempotence violated: ||P²-P||/||P|| = {rel_error:.2e}")
        
        # Check finite
        if not torch.isfinite(P).all():
            errors.append("Projection contains non-finite values")
        
        click.echo(f"✓ Projection matrix valid (idempotence error: {rel_error:.2e})")
        
    except Exception as e:
        errors.append(f"Projection matrix load failed: {e}")
    
    # Validate clean dataset
    try:
        table = feather.read_table(clean_dataset_path)
        
        # Check required columns
        required = {"post_id", "author_id", "post", "post_masked"}
        actual = set(table.column_names)
        missing = required - actual
        if missing:
            errors.append(f"Clean dataset missing columns: {missing}")
        
        # Check post_masked populated
        if "post_masked" in actual:
            post_masked = table["post_masked"].to_pylist()
            if not all(isinstance(pm, str) for pm in post_masked):
                errors.append("post_masked contains non-string values")
            
            masked_count = sum(1 for pm in post_masked if "[MASK:" in pm)
            click.echo(f"✓ Clean dataset: {len(table)} rows, {masked_count} with masks")
        
    except Exception as e:
        errors.append(f"Clean dataset load failed: {e}")
    
    # Validate pollution logs
    try:
        logs = feather.read_table(pollution_logs_path)
        
        # Check schema
        required_log_cols = {"post_id", "span_start", "span_end", "span_text", "entity_type"}
        actual_log_cols = set(logs.column_names)
        missing_log = required_log_cols - actual_log_cols
        if missing_log:
            errors.append(f"Pollution logs missing columns: {missing_log}")
        
        click.echo(f"✓ Pollution logs: {len(logs)} entries")
        
    except Exception as e:
        errors.append(f"Pollution logs load failed: {e}")
    
    # Report results
    if warnings:
        click.echo("\nWarnings:")
        for warning in warnings:
            click.echo(f"  ⚠ {warning}")
    
    if errors:
        click.echo("\n✗ Handover contract violated:")
        for error in errors:
            click.echo(f"  - {error}")
        raise click.ClickException("Validation failed")
    
    click.echo("\n✓ Handover contract satisfied")


@cli.command("run-phase-d")
@click.option(
    "--dataset",
    type=click.Path(path_type=Path),
    required=True,
    help="Tokenized dataset path or base path for auto-preprocess outputs",
)
@click.option(
    "--dataset-post",
    type=click.Path(path_type=Path),
    default=None,
    help="Tokenized dataset for raw posts (post)",
)
@click.option(
    "--dataset-masked",
    type=click.Path(path_type=Path),
    default=None,
    help="Tokenized dataset for masked posts (post_masked)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for Phase D artifacts",
)
@click.option(
    "--artifacts-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Phase A artifacts directory (contains projection_matrix.pt)",
)
@click.option(
    "--source-dataset",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to clean_dataset.arrow for preprocessing (auto-detected from artifacts-dir if not set)",
)
@click.option(
    "--preprocess/--no-preprocess",
    default=True,
    show_default=True,
    help="Auto-preprocess tokenized datasets when missing",
)
@click.option(
    "--pre-pad/--no-pre-pad",
    default=False,
    show_default=True,
    help="Pre-pad sequences during auto-preprocess for zero-copy collation",
)
@click.option(
    "--mode",
    type=click.Choice(["laptop", "hpc"]),
    default="laptop",
    help="Hardware mode (laptop or hpc)",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional experiment config to override defaults",
)
@click.option(
    "--data-change",
    type=click.Choice(["yes", "no"]),
    default="no",
    show_default=True,
    help="Rewrite dataset split column for small test runs",
)
@click.option(
    "--graph-train",
    is_flag=True,
    default=False,
    help="Enable manual CUDA graph training path (static bucket capture).",
)
@click.option(
    "--use-only",
    "use_only_labels",
    multiple=True,
    type=click.Choice([
        "birth_year", "female", "nationality", "political_leaning",
        "extrovert", "sensing", "feeling", "judging"
    ]),
    help="Filter to rows with valid values for specified labels (AND semantics). "
         "Can be specified multiple times. Single label uses simplified SingleTaskHead.",
)
def run_phase_d(
    dataset: Path,
    dataset_post: Path | None,
    dataset_masked: Path | None,
    output_dir: Path,
    artifacts_dir: Path,
    source_dataset: Path | None,
    preprocess: bool,
    pre_pad: bool,
    mode: str,
    config_path: Path | None,
    data_change: str,
    graph_train: bool,
    use_only_labels: tuple[str, ...],
):
    """Train Phase D baseline and constrained models."""
    from .config import find_config_root, load_phase_d_config
    from .data_engine.tokenization import preprocess_dataset
    from .phase_d_pipeline import run_phase_d_training

    try:
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

        if preprocess:
            if dataset_post is None and dataset_masked is None:
                dataset_post, dataset_masked = _derive_tokenized_paths(dataset)
            else:
                dataset_post = dataset_post or dataset
                dataset_masked = dataset_masked or dataset
        else:
            dataset_post = dataset_post or dataset
            dataset_masked = dataset_masked or dataset

        if not dataset_post.exists() or not dataset_masked.exists():
            if not preprocess:
                missing = []
                if not dataset_post.exists():
                    missing.append(str(dataset_post))
                if not dataset_masked.exists():
                    missing.append(str(dataset_masked))
                raise click.ClickException(
                    "Tokenized dataset(s) not found:\n"
                    + "\n".join(f"  - {path}" for path in missing)
                    + "\nEnable --preprocess or provide existing datasets."
                )

            if source_dataset is None:
                source_dataset = artifacts_dir / "clean_dataset.arrow"
            if not source_dataset.exists():
                raise click.ClickException(
                    f"Source dataset not found: {source_dataset}\n"
                    f"Provide --source-dataset or ensure clean_dataset.arrow exists in artifacts-dir"
                )

            config = load_phase_d_config(mode=mode, experiment_config_path=config_path)
            model_name = config.get("model", {}).get("name", "roberta-base")
            max_length = config.get("model", {}).get("max_length", 512)

            click.echo(f"\n{'=' * 80}")
            click.echo("AUTO-PREPROCESSING: Creating tokenized datasets")
            click.echo(f"{'=' * 80}")
            click.echo(f"Source: {source_dataset}")
            click.echo(f"Model: {model_name}")
            click.echo(f"Max length: {max_length}")
            click.echo(f"Pre-pad: {pre_pad}")

            for text_field, output_path in (
                ("post", dataset_post),
                ("post_masked", dataset_masked),
            ):
                if output_path.exists():
                    click.echo(f"Skipping {text_field}: exists at {output_path}")
                    continue
                click.echo(f"\nTokenizing '{text_field}' -> {output_path}")
                stats = preprocess_dataset(
                    input_path=source_dataset,
                    output_path=output_path,
                    model_name=model_name,
                    max_length=max_length,
                    text_field=text_field,
                    num_workers=None,
                    pre_pad=pre_pad,
                )
                click.echo(
                    "Tokenization complete: "
                    f"{stats['num_rows']} rows in {stats['elapsed_seconds']:.1f}s "
                    f"(pre_pad={stats['pre_pad']})"
                )

            click.echo(f"{'=' * 80}\n")

        dataset_base = dataset if dataset.exists() else (dataset_post or dataset_masked or dataset)
        
        # Convert use_only_labels tuple to None if empty
        use_only = use_only_labels if use_only_labels else None
        if use_only:
            click.echo(f"Label filter: --use-only {' --use-only '.join(use_only)}")
            if len(use_only) == 1:
                click.echo(f"  -> Single-task mode enabled for '{use_only[0]}'")

        logger.info("Phase D config root: %s", find_config_root("phase_d.yaml"))
        results = run_phase_d_training(
            dataset_path=dataset_base,
            dataset_path_post=dataset_post,
            dataset_path_masked=dataset_masked,
            output_dir=output_dir,
            artifacts_dir=artifacts_dir,
            mode=mode,
            config_path=config_path,
            data_change=(data_change == "yes"),
            graph_train=graph_train,
            use_only_labels=use_only,
        )

        click.echo("\n" + "=" * 80)
        click.echo("PHASE D TRAINING COMPLETE")
        click.echo("=" * 80)
        click.echo(f"\nOutput directory: {output_dir}")
        click.echo(f"  - Baseline: {output_dir}/baseline")
        click.echo(f"  - Constrained: {output_dir}/constrained")
        click.echo(f"  - Comparative metrics: {output_dir}/phase_d_comparative_metrics.json")

    except Exception as e:
        logger.error(f"Phase D training failed: {e}", exc_info=True)
        raise click.ClickException(str(e))


@cli.command("verify")
@click.option(
    "--dataset",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Phase D dataset path (clean_dataset.arrow)",
)
@click.option(
    "--phase-d-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path("artifacts/phase_d"),
    show_default=True,
    help="Phase D artifacts directory with baseline/constrained runs",
)
@click.option(
    "--artifacts-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path("artifacts/phase_a"),
    show_default=True,
    help="Phase A artifacts directory (projection matrix)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("artifacts/phase_d/chg"),
    show_default=True,
    help="Output directory for CHG + SVS artifacts",
)
@click.option(
    "--model-name",
    default="roberta-base",
    show_default=True,
    help="Base Hugging Face model name",
)
@click.option(
    "--taxonomy-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("conf/base/gliner_taxonomy.yaml"),
    show_default=True,
    help="GLiNER taxonomy path for mask-token alignment",
)
@click.option(
    "--max-length",
    default=512,
    show_default=True,
    help="Tokenizer max length for Phase D",
)
@click.option(
    "--batch-size",
    default=8,
    show_default=True,
    help="Batch size for verification",
)
@click.option(
    "--chg-epochs",
    default=5,
    show_default=True,
    help="CHG gate training epochs",
)
@click.option(
    "--chg-lr",
    default=1e-3,
    show_default=True,
    help="CHG gate learning rate",
)
@click.option(
    "--chg-regularization",
    default=0.01,
    show_default=True,
    help="CHG gate L1 regularization",
)
@click.option(
    "--facilitating-threshold",
    default=0.7,
    show_default=True,
    help="Gate threshold for facilitating heads",
)
@click.option(
    "--irrelevant-threshold",
    default=0.3,
    show_default=True,
    help="Gate threshold for irrelevant heads",
)
@click.option(
    "--svs-max-batches",
    default=5,
    show_default=True,
    help="Max batches for SVS calculation",
)
@click.option(
    "--svs-query-strategy",
    type=click.Choice(["cls", "all_tokens", "mean_tokens"]),
    default="mean_tokens",
    show_default=True,
    help="Query aggregation strategy for SVS",
)
@click.option(
    "--use-only",
    "use_only_labels",
    multiple=True,
    type=click.Choice([
        "birth_year", "female", "nationality", "political_leaning",
        "extrovert", "sensing", "feeling", "judging"
    ]),
    help="Filter to specific labels (auto-detected from training_metadata.json if not provided). "
         "Can be specified multiple times.",
)
def verify(
    dataset: Path,
    phase_d_dir: Path,
    artifacts_dir: Path,
    output_dir: Path,
    model_name: str,
    taxonomy_path: Path,
    max_length: int,
    batch_size: int,
    chg_epochs: int,
    chg_lr: float,
    chg_regularization: float,
    facilitating_threshold: float,
    irrelevant_threshold: float,
    svs_max_batches: int,
    svs_query_strategy: str,
    use_only_labels: tuple[str, ...],
):
    """Run CHG + SVS verification for baseline vs constrained models."""
    import json as json_module
    from .stylometry_net.verification import run_verification
    
    # Auto-detect use_only_labels from training_metadata.json if not provided
    use_only = use_only_labels if use_only_labels else None
    
    if not use_only:
        # Try to auto-detect from baseline training_metadata.json
        baseline_meta_path = phase_d_dir / "baseline" / "training_metadata.json"
        constrained_meta_path = phase_d_dir / "constrained" / "training_metadata.json"
        
        baseline_labels = None
        constrained_labels = None
        
        if baseline_meta_path.exists():
            try:
                with open(baseline_meta_path) as f:
                    baseline_meta = json_module.load(f)
                    label_filter = baseline_meta.get("label_filter", {})
                    baseline_labels = label_filter.get("use_only")
            except Exception as e:
                logger.warning(f"Failed to load baseline metadata: {e}")
        
        if constrained_meta_path.exists():
            try:
                with open(constrained_meta_path) as f:
                    constrained_meta = json_module.load(f)
                    label_filter = constrained_meta.get("label_filter", {})
                    constrained_labels = label_filter.get("use_only")
            except Exception as e:
                logger.warning(f"Failed to load constrained metadata: {e}")
        
        # Check for consistency between baseline and constrained
        if baseline_labels and constrained_labels and baseline_labels != constrained_labels:
            logger.warning(
                f"⚠️ Baseline and constrained models were trained with different label filters:\n"
                f"  Baseline: {baseline_labels}\n"
                f"  Constrained: {constrained_labels}\n"
                "Using baseline filter for verification."
            )
        
        # Use detected labels (prefer baseline)
        use_only = tuple(baseline_labels) if baseline_labels else None
        
        if use_only:
            click.echo(f"Auto-detected label filter from training metadata: {list(use_only)}")
        else:
            click.echo("No label filter detected - using all demographic labels")

    run_verification(
        dataset_path=dataset,
        artifacts_dir=artifacts_dir,
        phase_d_dir=phase_d_dir,
        output_dir=output_dir,
        model_name=model_name,
        taxonomy_path=taxonomy_path,
        max_length=max_length,
        batch_size=batch_size,
        chg_epochs=chg_epochs,
        chg_lr=chg_lr,
        chg_regularization=chg_regularization,
        facilitating_threshold=facilitating_threshold,
        irrelevant_threshold=irrelevant_threshold,
        svs_max_batches=svs_max_batches,
        svs_query_strategy=svs_query_strategy,
        use_only_labels=use_only,
    )
    click.echo(f"Verification complete. Outputs in: {output_dir}")


@cli.command("report-phase-d")
@click.option(
    "--phase-d-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Phase D artifacts directory with baseline/constrained runs",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("artifacts/reports"),
    show_default=True,
    help="Output directory for Phase D report",
)
@click.option(
    "--dataset",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional dataset path for report metadata",
)
def report_phase_d(
    phase_d_dir: Path,
    output_dir: Path,
    dataset: Path | None,
):
    """Generate Phase D report from existing artifacts."""
    from .evaluation.comparative_report import generate_phase_d_report

    report_path = generate_phase_d_report(
        phase_d_dir=phase_d_dir,
        output_dir=output_dir,
        dataset_path=dataset,
    )
    click.echo(f"Report generated: {report_path}")


@cli.command("hardware-info")
def hardware_info():
    """Display detected hardware profile."""
    from .hardware_ops.detection import HardwareDetector
    
    profile = HardwareDetector.detect()
    click.echo(str(profile))


if __name__ == "__main__":
    cli()
