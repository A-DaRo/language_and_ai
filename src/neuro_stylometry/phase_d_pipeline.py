"""Phase D training pipeline orchestrator."""

from __future__ import annotations

import json
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict

from .config import load_phase_d_config
from .training.trainer import PhaseDTrainer, PhaseDTrainConfig

logger = logging.getLogger(__name__)


def convert_to_python_types(obj):
    """Recursively convert NumPy/PyTorch types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def run_phase_d_training(
    dataset_path: Path,
    output_dir: Path,
    artifacts_dir: Path,
    mode: str = "laptop",
    config_path: Path | None = None,
) -> Dict[str, Any]:
    """
    Run Phase D training: baseline + constrained models.

    Args:
        dataset_path: Path to clean_dataset.arrow from Phase A
        output_dir: Output directory for Phase D artifacts
        artifacts_dir: Phase A artifacts directory (for projection_matrix.pt)
        mode: Hardware mode (laptop/hpc)
        config_path: Optional experiment config override

    Returns:
        Dict with results for baseline and constrained models
    """
    # Load configuration
    config = load_phase_d_config(mode=mode, experiment_config_path=config_path)

    # Override paths from CLI
    config['data']['dataset_path'] = str(dataset_path)
    config['data']['artifacts_dir'] = str(artifacts_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ========== Train Baseline Model ==========
    logger.info("=" * 80)
    logger.info("Training BASELINE model (raw text, no Affine Guard)")
    logger.info("=" * 80)

    baseline_dir = output_dir / "baseline"
    baseline_config = PhaseDTrainConfig(
        dataset_path=Path(config['data']['dataset_path']),
        artifacts_dir=Path(config['data']['artifacts_dir']),
        output_dir=baseline_dir,
        model_name=config['model']['name'],
        taxonomy_path=Path(config['model']['taxonomy_path']),
        max_length=config['model']['max_length'],
        batch_size=config['training']['batch_size'],
        num_epochs=config['training']['num_epochs'],
        max_steps=config['training'].get('max_steps'),
        learning_rate=config['training']['learning_rate'],
        save_every_steps=config['training'].get('save_every_steps'),
        save_every_epochs=config['training'].get('save_every_epochs'),
        split_ratios=config['data'].get('split_ratios'),
    )

    baseline_trainer = PhaseDTrainer(baseline_config)

    logger.info("Building baseline dataloaders (text_field='post')")
    train_loader, label_maps = baseline_trainer._build_loader(
        text_field="post",
        label_maps=None,
        split="train",
    )
    baseline_trainer.label_maps = label_maps

    val_loader, _ = baseline_trainer._build_loader(
        text_field="post",
        label_maps=label_maps,
        split="val",
    )

    test_loader, _ = baseline_trainer._build_loader(
        text_field="post",
        label_maps=label_maps,
        split="test",
    )

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Build baseline model
    logger.info("Building baseline model (no Affine Guard)")
    model, head = baseline_trainer._build_model(use_affine_guard=False)

    # Train baseline
    logger.info("Starting baseline training...")
    baseline_trainer._train(
        model=model,
        head=head,
        loader=train_loader,
        eval_loader=val_loader,
        run_dir=baseline_dir,
    )

    # Evaluate baseline on test
    if config['evaluation']['eval_on_test']:
        logger.info("Evaluating baseline on test set...")
        test_metrics = baseline_trainer._evaluate(
            model=model,
            head=head,
            loader=test_loader,
            num_classes=label_maps.num_classes(),
        )
        results['baseline_test'] = test_metrics
        logger.info(f"Baseline Test Metrics: {test_metrics}")

        # Save test metrics
        test_metrics_path = baseline_dir / "test_metrics.json"
        with open(test_metrics_path, 'w') as f:
            json.dump(convert_to_python_types(test_metrics), f, indent=2)

    # ========== Train Constrained Model ==========
    logger.info("=" * 80)
    logger.info("Training CONSTRAINED model (masked text + Affine Guard)")
    logger.info("=" * 80)

    constrained_dir = output_dir / "constrained"
    constrained_config = PhaseDTrainConfig(
        dataset_path=Path(config['data']['dataset_path']),
        artifacts_dir=Path(config['data']['artifacts_dir']),
        output_dir=constrained_dir,
        model_name=config['model']['name'],
        taxonomy_path=Path(config['model']['taxonomy_path']),
        max_length=config['model']['max_length'],
        batch_size=config['training']['batch_size'],
        num_epochs=config['training']['num_epochs'],
        max_steps=config['training'].get('max_steps'),
        learning_rate=config['training']['learning_rate'],
        save_every_steps=config['training'].get('save_every_steps'),
        save_every_epochs=config['training'].get('save_every_epochs'),
        split_ratios=config['data'].get('split_ratios'),
    )

    constrained_trainer = PhaseDTrainer(constrained_config)
    constrained_trainer.label_maps = label_maps  # Reuse same label maps

    logger.info("Building constrained dataloaders (text_field='post_masked')")
    train_loader_c, _ = constrained_trainer._build_loader(
        text_field="post_masked",
        label_maps=label_maps,
        split="train",
    )

    val_loader_c, _ = constrained_trainer._build_loader(
        text_field="post_masked",
        label_maps=label_maps,
        split="val",
    )

    test_loader_c, _ = constrained_trainer._build_loader(
        text_field="post_masked",
        label_maps=label_maps,
        split="test",
    )

    # Build constrained model with Affine Guard
    logger.info("Building constrained model (with Affine Guard)")
    model_c, head_c = constrained_trainer._build_model(use_affine_guard=True)

    # Train constrained
    logger.info("Starting constrained training...")
    constrained_trainer._train(
        model=model_c,
        head=head_c,
        loader=train_loader_c,
        eval_loader=val_loader_c,
        run_dir=constrained_dir,
    )

    # Evaluate constrained on test
    if config['evaluation']['eval_on_test']:
        logger.info("Evaluating constrained on test set...")
        test_metrics_c = constrained_trainer._evaluate(
            model=model_c,
            head=head_c,
            loader=test_loader_c,
            num_classes=label_maps.num_classes(),
        )
        results['constrained_test'] = test_metrics_c
        logger.info(f"Constrained Test Metrics: {test_metrics_c}")

        # Save test metrics
        test_metrics_path_c = constrained_dir / "test_metrics.json"
        with open(test_metrics_path_c, 'w') as f:
            json.dump(convert_to_python_types(test_metrics_c), f, indent=2)

    # ========== Save Comparative Summary ==========
    _save_comparative_summary(results, output_dir, label_maps.num_classes())

    # ========== Optional Report Generation ==========
    if config.get("evaluation", {}).get("write_report", True):
        reports_dir = config.get("output", {}).get("reports_dir")
        reports_dir = Path(reports_dir) if reports_dir else output_dir / "reports"
        try:
            from .evaluation.comparative_report import generate_phase_d_report

            report_path = generate_phase_d_report(
                phase_d_dir=output_dir,
                output_dir=reports_dir,
                dataset_path=dataset_path,
            )
            logger.info(f"Phase D report written to {report_path}")
        except Exception as exc:
            logger.warning("Phase D report generation failed: %s", exc)

    logger.info("=" * 80)
    logger.info("Phase D Training Complete!")
    logger.info(f"Baseline artifacts: {baseline_dir}")
    logger.info(f"Constrained artifacts: {constrained_dir}")
    logger.info("=" * 80)

    return results


def _save_comparative_summary(
    results: Dict[str, Any],
    output_dir: Path,
    num_classes: Dict[str, int],
):
    """Save comparative metrics summary."""
    summary = {
        "baseline": {
            "test": results.get('baseline_test', {}),
        },
        "constrained": {
            "test": results.get('constrained_test', {}),
        },
    }

    # Compute differences (constrained - baseline) for each task
    summary["delta"] = {}
    if "baseline_test" in results and "constrained_test" in results:
        baseline_metrics = results["baseline_test"]
        constrained_metrics = results["constrained_test"]

        for task in num_classes.keys():
            if task in baseline_metrics and task in constrained_metrics:
                baseline_acc = baseline_metrics[task].get("accuracy", 0)
                constrained_acc = constrained_metrics[task].get("accuracy", 0)
                summary["delta"][task] = {
                    "accuracy_delta": constrained_acc - baseline_acc,
                    "baseline_accuracy": baseline_acc,
                    "constrained_accuracy": constrained_acc,
                }

    summary_path = output_dir / "phase_d_comparative_metrics.json"
    with open(summary_path, 'w') as f:
        json.dump(convert_to_python_types(summary), f, indent=2)

    logger.info(f"Saved comparative summary: {summary_path}")

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("COMPARATIVE RESULTS (Test Set)")
    logger.info("=" * 80)
    logger.info(f"{'Task':<20} {'Baseline Acc':<15} {'Constrained Acc':<15} {'Delta':<10}")
    logger.info("-" * 80)
    for task in sorted(summary["delta"].keys()):
        delta_info = summary["delta"][task]
        baseline_acc = delta_info["baseline_accuracy"]
        constrained_acc = delta_info["constrained_accuracy"]
        delta = delta_info["accuracy_delta"]
        logger.info(f"{task:<20} {baseline_acc:<15.4f} {constrained_acc:<15.4f} {delta:+.4f}")
    logger.info("=" * 80)
