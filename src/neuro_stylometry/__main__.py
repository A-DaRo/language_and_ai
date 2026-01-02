# src/neuro_stylometry/__main__.py

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
def cli():
    """Neuro-Symbolic Stylometry Pipeline CLI."""
    pass


@cli.command()
@click.option("--dataset", type=click.Path(exists=True), required=True, help="Input SOBR dataset path")
@click.option("--output-dir", type=click.Path(), required=True, help="Output directory for artifacts")
@click.option("--mode", type=click.Choice(["auto", "laptop", "hpc"]), default="auto", help="Execution mode")
@click.option("--max-samples", type=int, default=None, help="Maximum samples to process (laptop mode)")
@click.option("--projection-label", type=str, default="nationality", help="Demographic label for projection")
def run_phase_a(dataset, output_dir, mode, max_samples, projection_label):
    """Execute Phase A pollution detection and mitigation."""
    from .phase_a_pipeline import PhaseAPipeline
    from .factories.strategy_factory import StrategyFactory
    from .hardware_ops.detection import ProfileType
    
    logger.info("Starting Phase A execution")
    
    # Determine profile type
    if mode == "laptop":
        profile_type = ProfileType.LAPTOP
    elif mode == "hpc":
        profile_type = ProfileType.HPC
    else:  # auto
        profile_type = None
    
    # Create strategy
    strategy = StrategyFactory.create_filter_strategy(profile_type)
    
    # Get recommended config and apply overrides
    if profile_type:
        config = StrategyFactory.get_recommended_config(profile_type)
    else:
        config = StrategyFactory.get_recommended_config(ProfileType.LAPTOP)
    
    # Apply CLI overrides
    config["projection_label"] = projection_label
    if max_samples:
        config["max_samples"] = max_samples
    
    # Create pipeline
    pipeline = PhaseAPipeline(strategy=strategy, config=config)
    
    # Execute
    artifacts = pipeline.run(
        input_dataset_path=Path(dataset),
        output_dir=Path(output_dir),
    )
    
    click.echo("\n" + "=" * 80)
    click.echo("PHASE A COMPLETE")
    click.echo("=" * 80)
    click.echo(f"Clean dataset: {artifacts.clean_dataset_path}")
    click.echo(f"Projection matrix: {artifacts.projection_matrix_path}")
    click.echo(f"Pollution logs: {artifacts.pollution_logs_path}")
    click.echo(f"\nProcessed {artifacts.metadata['num_samples']} samples")
    click.echo(f"Detected {artifacts.metadata['num_pollution_spans']} pollution spans")


@cli.command()
@click.option("--output-dir", type=click.Path(exists=True), required=True, help="Phase A output directory")
def validate_handover(output_dir):
    """Validate Phase A handover artifacts for Phase D."""
    from ...scripts.validate_phase_a_handover import validate_handover as validate_fn
    
    success = validate_fn(Path(output_dir))
    
    if success:
        click.echo("✓ Phase A handover validation PASSED")
    else:
        click.echo("✗ Phase A handover validation FAILED")
        exit(1)


if __name__ == "__main__":
    cli()
@click.pass_context
def run_phase_d(ctx, dataset, projection, output_dir):
    """Execute Phase D constrained training."""
    from .training.trainer import Trainer
    from .factories.trainer_factory import TrainerFactory
    import torch
    
    projection_matrix = torch.load(projection)
    trainer = TrainerFactory.create_trainer(
        config=ctx.obj["config"],
        projection_matrix=projection_matrix
    )
    trainer.train(Path(dataset), Path(output_dir))

@cli.command()
@click.option("--model-a", type=click.Path(exists=True), required=True, help="Dirty model path")
@click.option("--model-b", type=click.Path(exists=True), required=True, help="Clean model path")
@click.option("--output", type=click.Path(), required=True)
@click.pass_context
def verify(ctx, model_a, model_b, output):
    """Run Causal Head Gating verification and generate comparison report."""
    from .stylometry_net.chg_verifier import CHGVerifier
    from .evaluation.comparative_report import ReportGenerator
    
    verifier = CHGVerifier(ctx.obj["config"])
    
    gates_a = verifier.learn_gates(Path(model_a))
    gates_b = verifier.learn_gates(Path(model_b))
    
    report = ReportGenerator.generate(gates_a, gates_b, Path(output))
    click.echo(f"Verification report saved to {output}")

if __name__ == "__main__":
    cli()
