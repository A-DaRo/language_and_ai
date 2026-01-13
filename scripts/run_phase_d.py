#!/usr/bin/env python3
# scripts/run_phase_d.py
# Phase D execution script

"""Execute Phase D baseline + constrained training with config overrides."""

from __future__ import annotations

import argparse
from pathlib import Path

from neuro_stylometry.phase_d_pipeline import run_phase_d_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase D training.")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to clean_dataset.arrow from Phase A",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for Phase D artifacts",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        help="Phase A artifacts directory (contains projection_matrix.pt)",
    )
    parser.add_argument(
        "--mode",
        choices=("laptop", "hpc"),
        default="laptop",
        help="Hardware mode",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional experiment config to override defaults",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_phase_d_training(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        artifacts_dir=args.artifacts_dir,
        mode=args.mode,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
