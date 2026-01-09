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
def run_phase_a(dataset: Path, output_dir: Path, mode: str, config_path: Path | None, dry_run: bool):
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
def run_phase_d():
    """Phase D (Neural Stylometry) - Not yet implemented.
    
    Phase D training requires Phase A artifacts to be generated first.
    Run `neuro-stylometry run-phase-a` to generate the required artifacts.
    """
    raise click.ClickException(
        "Phase D is not yet implemented.\n"
        "Phase A artifacts can be generated with: neuro-stylometry run-phase-a\n"
        "Phase D will use artifacts from: artifacts/phase_a/"
    )


@cli.command("verify")
def verify():
    """Causal Head Gating verification - Not yet implemented."""
    raise click.ClickException(
        "Verification (Causal Head Gating) is not yet implemented.\n"
        "This command will compare dirty vs clean models after Phase D is complete."
    )


@cli.command("hardware-info")
def hardware_info():
    """Display detected hardware profile."""
    from .hardware_ops.detection import HardwareDetector
    
    profile = HardwareDetector.detect()
    click.echo(str(profile))


if __name__ == "__main__":
    cli()
