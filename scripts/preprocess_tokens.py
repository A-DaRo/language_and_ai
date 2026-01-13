#!/usr/bin/env python3
"""
AOT (Ahead-Of-Time) Tokenization Preprocessor.

This script parallelizes tokenization across all CPU cores and materializes
the tokenized dataset to disk. This removes tokenization from the training
loop entirely, eliminating the JIT tokenization bottleneck.

Usage:
    python scripts/preprocess_tokens.py \
        --input artifacts/phase_a/clean_dataset.arrow \
        --output artifacts/phase_d/tokenized_dataset.arrow \
        --model roberta-base \
        --max-length 512 \
        --workers 8 \
        --pre-pad
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neuro_stylometry.data_engine.tokenization import preprocess_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess dataset with parallel AOT tokenization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input Arrow file path",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output Arrow file path",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="roberta-base",
        help="HuggingFace model name for tokenizer",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="post_masked",
        help="Column name containing text to tokenize",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count - 1)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Samples per worker chunk",
    )
    parser.add_argument(
        "--pre-pad",
        action="store_true",
        default=False,
        help="Pre-pad all sequences to max-length for zero-copy collation (HPC mode)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    preprocess_dataset(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        max_length=args.max_length,
        text_field=args.text_field,
        num_workers=args.workers,
        chunk_size=args.chunk_size,
        pre_pad=args.pre_pad,
    )

    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()
