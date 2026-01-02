# src/neuro_stylometry/__main__.py

import click
from pathlib import Path
from .config import load_config

@click.group()
@click.option("--config", "-c", type=click.Path(exists=True), help="Experiment config path")
@click.option("--mode", type=click.Choice(["hpc", "laptop", "auto"]), default="auto")
@click.pass_context
def cli(ctx, config, mode):
    """Neuro-Symbolic Stylometry Pipeline CLI."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(
        Path(config) if config else Path("conf/base/pipeline.yaml"),
        mode=None if mode == "auto" else mode
    )

@cli.command()
@click.option("--input-dir", type=click.Path(exists=True), required=True)
@click.option("--output-path", type=click.Path(), required=True)
@click.pass_context
def convert_data(ctx, input_dir, output_path):
    """Convert Pandas DataFrames to unified Arrow format."""
    from .data_engine.converter import PandasToArrowConverter
    
    converter = PandasToArrowConverter(ctx.obj["config"])
    converter.convert(Path(input_dir), Path(output_path))

@cli.command()
@click.option("--dataset", type=click.Path(exists=True), required=True)
@click.option("--output-dir", type=click.Path(), required=True)
@click.pass_context
def run_phase_a(ctx, dataset, output_dir):
    """Execute Phase A pollution detection and mitigation."""
    from .pollution_guard import PhaseAPipeline
    from .factories.strategy_factory import StrategyFactory
    
    strategy = StrategyFactory.create_filter_strategy()
    pipeline = PhaseAPipeline(ctx.obj["config"], strategy)
    artifacts = pipeline.run(Path(dataset), Path(output_dir))
    
    click.echo(f"Phase A complete. Artifacts saved to {output_dir}")
    click.echo(f"  Explicit Recall: {artifacts.explicit_recall:.2%}")
    click.echo(f"  Amnesic Drop: {artifacts.amnesic_drop:.2%}")

@cli.command()
@click.option("--dataset", type=click.Path(exists=True), required=True)
@click.option("--projection", type=click.Path(exists=True), required=True)
@click.option("--output-dir", type=click.Path(), required=True)
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
