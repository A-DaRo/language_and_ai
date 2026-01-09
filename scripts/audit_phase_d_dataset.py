#!/usr/bin/env python3
# scripts/audit_phase_d_dataset.py
# Dataset audit for Phase D readiness.

from __future__ import annotations

import argparse
from pathlib import Path

from neuro_stylometry.evaluation.dataset_audit import AuditConfig, audit_dataset, write_audit_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit SOBR Arrow dataset for Phase D.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to SOBR Arrow dataset (e.g., artifacts/data/sobr.arrow)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reports/dataset_audit.json"),
        help="Output path for JSON report",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Optional sample size for audit (0 = full dataset)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_size = args.sample_size if args.sample_size > 0 else None
    config = AuditConfig(sample_size=sample_size, seed=args.seed)
    summary = audit_dataset(args.input, config=config)
    write_audit_report(summary, args.output)
    print(f"Dataset audit written to {args.output}")


if __name__ == "__main__":
    main()
