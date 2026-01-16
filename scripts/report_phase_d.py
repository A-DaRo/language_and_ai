#!/usr/bin/env python3
# scripts/report_phase_d.py
# Phase D report generator (CLI stub).

"""Generate Phase D report from existing artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from neuro_stylometry.evaluation.comparative_report import generate_phase_d_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase D report.")
    parser.add_argument(
        "--phase-d-dir",
        type=Path,
        required=True,
        help="Phase D artifacts directory with baseline/constrained runs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports"),
        help="Output directory for report output",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional dataset path for report metadata",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = generate_phase_d_report(
        phase_d_dir=args.phase_d_dir,
        output_dir=args.output_dir,
        dataset_path=args.dataset,
    )
    print(f"Report generated: {report_path}")


if __name__ == "__main__":
    main()
