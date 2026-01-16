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
@click.option(
    "--use-only",
    "use_only_labels",
    type=str,
    multiple=True,
    help=(
        "Filter to specified demographic label(s). Restricts dataset rows and LEACE/probing "
        "to only the specified demographic columns. May be specified multiple times. "
        "Example: --use-only gender --use-only birth_year"
    )
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
    use_only_labels: tuple,
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
            if use_only_labels:
                click.echo(f"\n--use-only labels: {list(use_only_labels)}")
            return
        
        # Convert tuple to list for use_only_labels
        use_only_list = list(use_only_labels) if use_only_labels else None
        
        if use_only_list:
            logger.info(f"Single-label mode enabled: filtering to {use_only_list}")
        
        # Create pipeline with PhaseAPipeline facade
        pipeline = PhaseAPipeline(strategy=strategy, config=config)
        
        # Execute pipeline
        logger.info("Starting Phase A execution")
        artifacts = pipeline.run(
            input_dataset_path=dataset,
            output_dir=output_dir,
            use_only_labels=use_only_list,
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
    "--dataset-post",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Tokenized dataset for raw posts (post)",
)
@click.option(
    "--dataset-masked",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Tokenized dataset for masked posts (post_masked)",
)
@click.option(
    "--mode",
    type=click.Choice(["laptop", "hpc"]),
    default="laptop",
    help="Hardware mode for verify.yaml (laptop or hpc)",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
    help="Optional verify YAML config to merge on top of base+mode configs",
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
    default=None,
    help="Base Hugging Face model name (defaults to verify.yaml)",
)
@click.option(
    "--taxonomy-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="GLiNER taxonomy path for mask-token alignment (defaults to verify.yaml)",
)
@click.option(
    "--max-length",
    default=None,
    type=int,
    help="Tokenizer max length for Phase D (defaults to verify.yaml)",
)
@click.option(
    "--batch-size",
    default=None,
    type=int,
    help="Batch size for verification (defaults to verify.yaml)",
)
@click.option(
    "--chg-epochs",
    default=None,
    type=int,
    help="CHG gate training epochs (defaults to verify.yaml)",
)
@click.option(
    "--chg-lr",
    default=None,
    type=float,
    help="CHG gate learning rate (defaults to verify.yaml)",
)
@click.option(
    "--chg-regularization",
    default=None,
    type=float,
    help="CHG gate L1 regularization (defaults to verify.yaml)",
)
@click.option(
    "--facilitating-threshold",
    default=None,
    type=float,
    help="Gate threshold for facilitating heads (defaults to verify.yaml)",
)
@click.option(
    "--irrelevant-threshold",
    default=None,
    type=float,
    help="Gate threshold for irrelevant heads (defaults to verify.yaml)",
)
@click.option(
    "--svs-max-batches",
    default=None,
    type=int,
    help="Max batches for SVS calculation (defaults to verify.yaml)",
)
@click.option(
    "--svs-query-strategy",
    type=click.Choice(["cls", "all_tokens", "mean_tokens"]),
    default=None,
    help="Query aggregation strategy for SVS (defaults to verify.yaml)",
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
    dataset_post: Path | None,
    dataset_masked: Path | None,
    mode: str,
    config_path: Path | None,
    phase_d_dir: Path,
    artifacts_dir: Path,
    output_dir: Path,
    model_name: str | None,
    taxonomy_path: Path | None,
    max_length: int | None,
    batch_size: int | None,
    chg_epochs: int | None,
    chg_lr: float | None,
    chg_regularization: float | None,
    facilitating_threshold: float | None,
    irrelevant_threshold: float | None,
    svs_max_batches: int | None,
    svs_query_strategy: str | None,
    use_only_labels: tuple[str, ...],
):
    """Run CHG + SVS verification for baseline vs constrained models."""
    import json as json_module
    from .stylometry_net.verification import run_verification
    from .config import find_config_root, load_verify_config
    
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

    config = load_verify_config(mode=mode, experiment_config_path=config_path)
    config_root = find_config_root("verify.yaml")

    if model_name is None:
        model_name = config.get("model", {}).get("name", "roberta-base")
    if taxonomy_path is None:
        taxonomy_value = config.get("model", {}).get("taxonomy_path")
        if taxonomy_value:
            taxonomy_candidate = Path(taxonomy_value)
            taxonomy_path = (
                taxonomy_candidate
                if taxonomy_candidate.is_absolute()
                else config_root / taxonomy_candidate
            )
    if taxonomy_path is None:
        taxonomy_path = config_root / "conf/base/gliner_taxonomy.yaml"
    if not taxonomy_path.exists():
        raise click.ClickException(f"Taxonomy path not found: {taxonomy_path}")
    if max_length is None:
        max_length = int(config.get("model", {}).get("max_length", 512))

    verify_cfg = config.get("verify", {})
    chg_cfg = verify_cfg.get("chg", {})
    svs_cfg = verify_cfg.get("svs", {})
    if batch_size is None:
        batch_size = int(verify_cfg.get("batch_size", 8))
    if chg_epochs is None:
        chg_epochs = int(chg_cfg.get("epochs", 5))
    if chg_lr is None:
        chg_lr = float(chg_cfg.get("learning_rate", 1e-3))
    if chg_regularization is None:
        chg_regularization = float(chg_cfg.get("regularization", 0.01))
    if facilitating_threshold is None:
        facilitating_threshold = float(chg_cfg.get("facilitating_threshold", 0.7))
    if irrelevant_threshold is None:
        irrelevant_threshold = float(chg_cfg.get("irrelevant_threshold", 0.3))
    if svs_max_batches is None:
        svs_max_batches = int(svs_cfg.get("max_batches", 5))
    if svs_query_strategy is None:
        svs_query_strategy = svs_cfg.get("query_strategy", "mean_tokens")
    svs_pos_backend = svs_cfg.get("pos_backend", "lexical")
    svs_use_pos = bool(svs_cfg.get("use_pos", True))
    svs_spacy_model = svs_cfg.get("spacy_model", "en_core_web_sm")
    svs_spacy_use_gpu = bool(svs_cfg.get("spacy_use_gpu", False))
    svs_spacy_gpu_id = svs_cfg.get("spacy_gpu_id", 0)
    svs_spacy_batch_size = int(svs_cfg.get("spacy_batch_size", 32))
    svs_spacy_n_process = int(svs_cfg.get("spacy_n_process", 1))
    svs_spacy_disable = list(svs_cfg.get("spacy_disable", []))
    svs_hf_model = svs_cfg.get("hf_model", "vblagoje/bert-english-uncased-finetuned-pos")
    svs_hf_device = int(svs_cfg.get("hf_device", 0))
    svs_hf_batch_size = int(svs_cfg.get("hf_batch_size", 16))

    if dataset_post is None:
        derived_post, _ = _derive_tokenized_paths(dataset)
        if derived_post.exists():
            dataset_post = derived_post

    if dataset_masked is None:
        _, derived_masked = _derive_tokenized_paths(dataset)
        if derived_masked.exists():
            dataset_masked = derived_masked

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
        dataset_path_post=dataset_post,
        dataset_path_masked=dataset_masked,
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
        svs_pos_backend=svs_pos_backend,
        svs_use_pos=svs_use_pos,
        svs_spacy_model=svs_spacy_model,
        svs_spacy_use_gpu=svs_spacy_use_gpu,
        svs_spacy_gpu_id=svs_spacy_gpu_id,
        svs_spacy_batch_size=svs_spacy_batch_size,
        svs_spacy_n_process=svs_spacy_n_process,
        svs_spacy_disable=svs_spacy_disable,
        svs_hf_model=svs_hf_model,
        svs_hf_device=svs_hf_device,
        svs_hf_batch_size=svs_hf_batch_size,
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
@click.option(
    "--use-only",
    "use_only_labels",
    multiple=True,
    type=str,
    help="Filter to specific labels (auto-detected from training_metadata.json if not provided). "
         "Can be specified multiple times. Example: --use-only nationality",
)
@click.option(
    "--baseline-dir",
    type=str,
    default="baseline",
    show_default=True,
    help="Baseline subdirectory name within phase-d-dir",
)
@click.option(
    "--constrained-dir",
    type=str,
    default="constrained",
    show_default=True,
    help="Constrained subdirectory name within phase-d-dir",
)
@click.option(
    "--chg-dir",
    type=str,
    default=None,
    help="CHG/verification subdirectory name (defaults to 'chg' or 'chg-data' if found)",
)
def report_phase_d(
    phase_d_dir: Path,
    output_dir: Path,
    dataset: Path | None,
    use_only_labels: tuple[str, ...],
    baseline_dir: str,
    constrained_dir: str,
    chg_dir: str | None,
):
    """Generate Phase D comparative report with visualizations.

    This command generates an HTML report comparing baseline and constrained
    model performance. It auto-detects the tasks trained from training_metadata.json
    or can be explicitly filtered with --use-only.

    The report includes:

    \b
    - Executive summary with key metrics
    - Training loss curves (combined overlay)
    - Per-task accuracy and F1 comparisons
    - Per-class F1 bar charts (sorted by performance)
    - Top-N confusion matrices (zoomed to most frequent classes)
    - Confusion change analysis (baseline vs constrained)
    - Class distribution histograms
    - Performance vs sample count scatter plots
    - Precision/recall trade-off analysis
    - CHG head classification comparison
    - SVS verification scores with interpretation

    \b
    Examples:
      # Auto-detect tasks from metadata
      python -m neuro_stylometry report-phase-d --phase-d-dir artifacts/phase_d

      # Custom directory names (for non-standard layouts)
      python -m neuro_stylometry report-phase-d \\
        --phase-d-dir artifacts/phase-d-for-reports \\
        --baseline-dir baseline-data \\
        --constrained-dir constrained-data \\
        --chg-dir chg-data

      # Explicit single-task mode
      python -m neuro_stylometry report-phase-d \\
        --phase-d-dir artifacts/phase_d \\
        --use-only nationality
    """
    import json as json_module
    from .evaluation.comparative_report import generate_phase_d_report

    # Auto-detect use_only from training_metadata.json if not provided
    use_only = list(use_only_labels) if use_only_labels else None

    if not use_only:
        # Try to auto-detect from baseline training_metadata.json
        baseline_meta_path = phase_d_dir / baseline_dir / "training_metadata.json"

        if baseline_meta_path.exists():
            try:
                with open(baseline_meta_path) as f:
                    baseline_meta = json_module.load(f)
                    label_filter = baseline_meta.get("label_filter", {})
                    detected_labels = label_filter.get("use_only")
                    if detected_labels:
                        use_only = detected_labels
                        click.echo(f"Auto-detected label filter from metadata: {use_only}")
            except Exception as e:
                logger.warning(f"Failed to load baseline metadata: {e}")

    if use_only:
        click.echo(f"Generating report for task(s): {', '.join(use_only)}")
    else:
        click.echo("Generating report for all detected tasks")

    report_path = generate_phase_d_report(
        phase_d_dir=phase_d_dir,
        output_dir=output_dir,
        dataset_path=dataset,
        baseline_subdir=baseline_dir,
        constrained_subdir=constrained_dir,
        chg_subdir=chg_dir,
        use_only=use_only,
    )
    click.echo(f"Report generated: {report_path}")


@cli.command("run-full-pipeline")
@click.option(
    "--dataset",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to input dataset Arrow file",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for all pipeline artifacts",
)
@click.option(
    "--mode",
    type=click.Choice(["auto", "laptop", "hpc"]),
    default="auto",
    help="Hardware mode (auto-detected if not specified)",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
    help="Optional experiment YAML config to merge on top of base+mode configs",
)
@click.option(
    "--use-only",
    "use_only_labels",
    type=str,
    multiple=True,
    help=(
        "Filter to specified demographic label(s). Restricts Phase A and Phase D "
        "to only the specified demographic columns. May be specified multiple times. "
        "Example: --use-only gender --use-only birth_year"
    ),
)
@click.option(
    "--skip-phase-a",
    is_flag=True,
    help="Skip Phase A (requires existing Phase A artifacts in output-dir/phase_a)",
)
@click.option(
    "--skip-phase-d",
    is_flag=True,
    help="Skip Phase D (only run Phase A)",
)
@click.option(
    "--skip-verify",
    is_flag=True,
    help="Skip Phase D verification (CHG + SVS)",
)
@click.option(
    "--skip-report",
    is_flag=True,
    help="Skip Phase D report generation",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print resolved config and execution plan without running",
)
def run_full_pipeline(
    dataset: Path,
    output_dir: Path,
    mode: str,
    config_path: Path | None,
    use_only_labels: tuple,
    skip_phase_a: bool,
    skip_phase_d: bool,
    skip_verify: bool,
    skip_report: bool,
    dry_run: bool,
):
    """
    Execute full pipeline: Phase A → Phase D.
    
    This command orchestrates the complete neuro-stylometry pipeline:
    
    1. Phase A: Pollution detection and LEACE projection
       - Detects demographic pollution spans with GLiNER
       - Applies typed masking
       - Computes LEACE projection matrix
       - Outputs: clean_dataset.arrow, projection_matrix.pt, pollution_logs.arrow
    
    2. Phase D: Comparative fine-tuning
       - Runs baseline (no projection) and constrained (with projection) training
       - Computes comparative metrics
       - Outputs: tokenized_dataset.arrow, baseline/, constrained/
    
    Use --use-only to focus on specific demographic labels (single-label mode).
    Use --skip-phase-a to start from existing Phase A artifacts.
    Use --skip-phase-d to only run Phase A.
    """
    import json
    import yaml
    from .phase_a_pipeline import PhaseAPipeline
    from .factories.strategy_factory import StrategyFactory
    from .hardware_ops.detection import HardwareDetector, ProfileType
    from .config import load_pipeline_config, load_phase_d_config
    from .data_engine.tokenization import preprocess_dataset
    
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

        # Auto-detect hardware mode if needed
        if mode == "auto":
            profile = HardwareDetector.detect()
            mode = "laptop" if profile.profile_type == ProfileType.LAPTOP else "hpc"
            logger.info(f"Auto-detected hardware mode: {mode}")
        
        # Convert use_only_labels tuple to list
        use_only_list = list(use_only_labels) if use_only_labels else None
        
        # Create output directories
        output_dir = Path(output_dir)
        phase_a_dir = output_dir / "phase_a"
        phase_d_dir = output_dir / "phase_d"
        
        # Load configuration
        config = load_pipeline_config(mode=mode, experiment_config_path=config_path)
        
        # Dry run: print execution plan
        if dry_run:
            click.echo("=" * 80)
            click.echo("FULL PIPELINE EXECUTION PLAN")
            click.echo("=" * 80)
            click.echo(f"\nDataset: {dataset}")
            click.echo(f"Output directory: {output_dir}")
            click.echo(f"Hardware mode: {mode}")
            if use_only_list:
                click.echo(f"Label filter (--use-only): {use_only_list}")
            click.echo(f"\nPhase A: {'SKIP' if skip_phase_a else 'RUN'}")
            click.echo(f"  - Artifacts: {phase_a_dir}")
            click.echo(f"Phase D: {'SKIP' if skip_phase_d else 'RUN'}")
            click.echo(f"  - Artifacts: {phase_d_dir}")
            click.echo(f"Verify (CHG+SVS): {'SKIP' if skip_verify else 'RUN'}")
            click.echo(f"  - Artifacts: {phase_d_dir / 'chg'}")
            click.echo(f"Report: {'SKIP' if skip_report else 'RUN'}")
            click.echo(f"  - Output: {phase_d_dir / 'reports'}")
            click.echo(f"\nResolved Configuration:")
            click.echo("-" * 40)
            click.echo(yaml.dump(config, default_flow_style=False, sort_keys=False))
            return
        
        # =====================================================================
        # Phase A Execution
        # =====================================================================
        if skip_phase_a:
            logger.info("Skipping Phase A (--skip-phase-a)")
            # Validate Phase A artifacts exist
            required_artifacts = [
                phase_a_dir / "clean_dataset.arrow",
                phase_a_dir / "projection_matrix.pt",
            ]
            for artifact in required_artifacts:
                if not artifact.exists():
                    raise click.ClickException(
                        f"--skip-phase-a requires {artifact} to exist"
                    )
            logger.info(f"Phase A artifacts validated: {phase_a_dir}")
            phase_a_artifacts = None
        else:
            click.echo("\n" + "=" * 80)
            click.echo("PHASE A: Pollution Detection & LEACE Projection")
            click.echo("=" * 80 + "\n")
            
            # Create strategy based on mode
            profile_type = ProfileType.HPC if mode == "hpc" else ProfileType.LAPTOP
            strategy = StrategyFactory.create_filter_strategy(profile_type)
            
            # Create and run Phase A pipeline
            pipeline_a = PhaseAPipeline(strategy=strategy, config=config)
            phase_a_artifacts = pipeline_a.run(
                input_dataset_path=dataset,
                output_dir=phase_a_dir,
                use_only_labels=use_only_list,
            )
            
            click.echo(f"\nPhase A complete!")
            click.echo(f"  Clean dataset: {phase_a_artifacts.clean_dataset_path}")
            click.echo(f"  Projection matrix: {phase_a_artifacts.projection_matrix_path}")
            click.echo(f"  Pollution logs: {phase_a_artifacts.pollution_logs_path}")
        
        # =====================================================================
        # Phase D Execution
        # =====================================================================
        if skip_phase_d:
            logger.info("Skipping Phase D (--skip-phase-d)")
            click.echo("\n" + "=" * 80)
            click.echo("FULL PIPELINE COMPLETE (Phase D skipped)")
            click.echo("=" * 80)
            click.echo(f"\nPhase A artifacts: {phase_a_dir}")
            return
        
        click.echo("\n" + "=" * 80)
        click.echo("PHASE D: Comparative Fine-Tuning")
        click.echo("=" * 80 + "\n")
        
        # Import Phase D runner
        from .phase_d_pipeline import run_phase_d_training
        
        # Determine Phase A input paths
        clean_dataset_path = phase_a_dir / "clean_dataset.arrow"
        projection_matrix_path = phase_a_dir / "projection_matrix.pt"
        
        # Load Phase D config to check for AOT mode
        phase_d_config = load_phase_d_config(mode=mode, experiment_config_path=config_path)
        opt_cfg = phase_d_config.get('optimization', {})
        use_aot_mode = opt_cfg.get('use_aot_mode', False)
        
        # Determine tokenized dataset paths
        dataset_post, dataset_masked = _derive_tokenized_paths(clean_dataset_path)
        
        # Auto-preprocess tokenized datasets if AOT mode is enabled and they don't exist
        if use_aot_mode and (not dataset_post.exists() or not dataset_masked.exists()):
            model_name = phase_d_config.get("model", {}).get("name", "roberta-base")
            max_length = phase_d_config.get("model", {}).get("max_length", 512)
            pre_pad = True  # Default to pre-padding for AOT mode

            click.echo(f"\n{'=' * 80}")
            click.echo("AUTO-PREPROCESSING: Creating tokenized datasets for AOT mode")
            click.echo(f"{'=' * 80}")
            click.echo(f"Source: {clean_dataset_path}")
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
                    input_path=clean_dataset_path,
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
        
        # Set dataset paths if they exist, otherwise pass None
        dataset_post_arg = dataset_post if dataset_post.exists() else None
        dataset_masked_arg = dataset_masked if dataset_masked.exists() else None
        
        # Run Phase D
        phase_d_results = run_phase_d_training(
            dataset_path=clean_dataset_path,
            dataset_path_post=dataset_post_arg,
            dataset_path_masked=dataset_masked_arg,
            output_dir=phase_d_dir,
            artifacts_dir=phase_a_dir,
            mode=mode,
            use_only_labels=tuple(use_only_list) if use_only_list else None,
        )
        
        click.echo(f"\nPhase D complete!")
        click.echo(f"  Baseline results: {phase_d_dir / 'baseline'}")
        click.echo(f"  Constrained results: {phase_d_dir / 'constrained'}")

        # =====================================================================
        # Phase D Verification (CHG + SVS)
        # =====================================================================
        if not skip_verify:
            click.echo("\n" + "=" * 80)
            click.echo("PHASE D VERIFY: CHG + SVS")
            click.echo("=" * 80 + "\n")

            from .stylometry_net.verification import run_verification
            from .config import find_config_root, load_verify_config

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

            # Use tokenized datasets from Phase D (already preprocessed if needed)
            verify_dataset_post = dataset_post_arg
            verify_dataset_masked = dataset_masked_arg

            verify_output_dir = phase_d_dir / "chg"
            verify_output_dir.mkdir(parents=True, exist_ok=True)

            run_verification(
                dataset_path=clean_dataset_path,
                dataset_path_post=verify_dataset_post,
                dataset_path_masked=verify_dataset_masked,
                artifacts_dir=phase_a_dir,
                phase_d_dir=phase_d_dir,
                output_dir=verify_output_dir,
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
                use_only_labels=tuple(use_only_list) if use_only_list else None,
            )
            click.echo(f"Verification complete. Outputs in: {verify_output_dir}")

        # =====================================================================
        # Phase D Report
        # =====================================================================
        if not skip_report:
            click.echo("\n" + "=" * 80)
            click.echo("PHASE D REPORT")
            click.echo("=" * 80 + "\n")

            from .evaluation.comparative_report import generate_phase_d_report

            report_dir = phase_d_dir / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = generate_phase_d_report(
                phase_d_dir=phase_d_dir,
                output_dir=report_dir,
                dataset_path=clean_dataset_path,
                baseline_subdir="baseline",
                constrained_subdir="constrained",
                chg_subdir="chg",
                use_only=use_only_list,
            )
            click.echo(f"Report generated: {report_path}")
        
        # =====================================================================
        # Final Summary
        # =====================================================================
        click.echo("\n" + "=" * 80)
        click.echo("FULL PIPELINE COMPLETE")
        click.echo("=" * 80)
        click.echo(f"\nAll artifacts in: {output_dir}")
        click.echo(f"  Phase A: {phase_a_dir}")
        click.echo(f"  Phase D: {phase_d_dir}")
        
        if use_only_list:
            click.echo(f"\nLabel filter applied: {use_only_list}")
        
        # Save pipeline summary
        summary = {
            "dataset": str(dataset),
            "output_dir": str(output_dir),
            "mode": mode,
            "label_filter": use_only_list,
            "phase_a_dir": str(phase_a_dir),
            "phase_d_dir": str(phase_d_dir),
        }
        summary_path = output_dir / "pipeline_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        click.echo(f"\nPipeline summary: {summary_path}")
        
    except Exception as e:
        logger.error(f"Full pipeline execution failed: {e}", exc_info=True)
        raise click.ClickException(str(e))


@cli.command("hardware-info")
def hardware_info():
    """Display detected hardware profile."""
    from .hardware_ops.detection import HardwareDetector
    
    profile = HardwareDetector.detect()
    click.echo(str(profile))


if __name__ == "__main__":
    cli()
