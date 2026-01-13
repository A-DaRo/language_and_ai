#!/usr/bin/env python3
"""
AOT (Ahead-Of-Time) Tokenization Preprocessor.

This script parallelizes tokenization across all CPU cores and materializes
the tokenized dataset to disk. This removes tokenization from the training
loop entirely, eliminating the JIT tokenization bottleneck.

Architecture:
    1. Load Arrow dataset with memory mapping
    2. Distribute tokenization across worker processes
    3. Write tokenized results to a new Arrow file with SOBR_SCHEMA_TOKENIZED

Usage:
    python scripts/preprocess_tokens.py \
        --input artifacts/phase_a/clean_dataset.arrow \
        --output artifacts/phase_d/tokenized_dataset.arrow \
        --model roberta-base \
        --max-length 512 \
        --workers 8

Reference: Phase D Final Optimization Blueprint Section 2.1
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neuro_stylometry.data_engine.schemas import (
    SOBR_SCHEMA_TOKENIZED,
    TOKENIZED_FIELDS,
    get_demographic_columns,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer worker processes."""
    model_name: str
    max_length: int
    taxonomy_path: Optional[Path] = None


# Global tokenizer for worker processes (initialized once per worker)
_worker_tokenizer = None


def _init_worker(config: TokenizerConfig) -> None:
    """Initialize tokenizer in worker process."""
    global _worker_tokenizer
    
    # Import here to avoid loading in main process
    from transformers import AutoTokenizer
    
    _worker_tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        use_fast=True,  # Use Rust tokenizer for speed
    )
    
    # Configure for our schema
    _worker_tokenizer.model_max_length = config.max_length


def _tokenize_batch(
    batch: List[Tuple[int, str]],
) -> List[Tuple[int, List[int], List[int], int]]:
    """
    Tokenize a batch of texts in a worker process.
    
    Args:
        batch: List of (index, text) tuples.
        
    Returns:
        List of (index, input_ids, attention_mask, token_count) tuples.
    """
    global _worker_tokenizer
    
    if _worker_tokenizer is None:
        raise RuntimeError("Worker tokenizer not initialized")
    
    results = []
    
    for idx, text in batch:
        if text is None:
            text = ""
        
        # Tokenize with truncation
        encoding = _worker_tokenizer(
            text,
            padding=False,  # No padding - we'll pad in collator with quantized lengths
            truncation=True,
            max_length=_worker_tokenizer.model_max_length,
            return_attention_mask=True,
        )
        
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        token_count = len(input_ids)
        
        results.append((idx, input_ids, attention_mask, token_count))
    
    return results


def _chunk_iterator(
    table: pa.Table,
    text_field: str,
    chunk_size: int = 1000,
) -> Iterator[List[Tuple[int, str]]]:
    """
    Yield chunks of (index, text) tuples for parallel processing.
    
    Args:
        table: PyArrow table with text data.
        text_field: Name of the text column.
        chunk_size: Number of samples per chunk.
        
    Yields:
        Lists of (index, text) tuples.
    """
    text_col = table[text_field]
    num_rows = table.num_rows
    
    for start_idx in range(0, num_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, num_rows)
        chunk = []
        
        for i in range(start_idx, end_idx):
            text = text_col[i].as_py()
            chunk.append((i, text if text else ""))
        
        yield chunk


def preprocess_dataset(
    input_path: Path,
    output_path: Path,
    model_name: str,
    max_length: int,
    text_field: str = "post_masked",
    num_workers: Optional[int] = None,
    chunk_size: int = 500,
    taxonomy_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Preprocess dataset with parallel tokenization.
    
    Args:
        input_path: Path to input Arrow file.
        output_path: Path for output Arrow file.
        model_name: HuggingFace model name for tokenizer.
        max_length: Maximum sequence length.
        text_field: Column name containing text to tokenize.
        num_workers: Number of worker processes (default: CPU count).
        chunk_size: Samples per worker chunk.
        taxonomy_path: Optional taxonomy config path.
        
    Returns:
        Statistics dictionary with tokenization results.
    """
    start_time = time.time()
    
    # Determine worker count
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    logger.info(f"Loading input dataset: {input_path}")
    table = feather.read_table(input_path, memory_map=True)
    num_rows = table.num_rows
    
    logger.info(f"Dataset: {num_rows} rows, {len(table.column_names)} columns")
    logger.info(f"Text field: {text_field}")
    logger.info(f"Tokenizer: {model_name}, max_length={max_length}")
    logger.info(f"Workers: {num_workers}")
    
    # Validate text field exists
    if text_field not in table.column_names:
        available = ", ".join(table.column_names)
        raise ValueError(f"Text field '{text_field}' not found. Available: {available}")
    
    # Prepare tokenizer config
    config = TokenizerConfig(
        model_name=model_name,
        max_length=max_length,
        taxonomy_path=taxonomy_path,
    )
    
    # Pre-allocate result arrays
    all_input_ids: List[Optional[List[int]]] = [None] * num_rows
    all_attention_masks: List[Optional[List[int]]] = [None] * num_rows
    all_token_counts: List[int] = [0] * num_rows
    
    # Tokenize in parallel
    logger.info("Starting parallel tokenization...")
    
    chunks = list(_chunk_iterator(table, text_field, chunk_size))
    total_chunks = len(chunks)
    
    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(config,),
    ) as pool:
        results_iter = pool.imap(_tokenize_batch, chunks)
        
        with tqdm(total=num_rows, desc="Tokenizing") as pbar:
            for batch_results in results_iter:
                for idx, input_ids, attention_mask, token_count in batch_results:
                    all_input_ids[idx] = input_ids
                    all_attention_masks[idx] = attention_mask
                    all_token_counts[idx] = token_count
                    pbar.update(1)
    
    # Validate all rows were processed
    missing = sum(1 for ids in all_input_ids if ids is None)
    if missing > 0:
        raise RuntimeError(f"Failed to tokenize {missing} rows")
    
    # Build tokenized columns
    logger.info("Building Arrow columns...")
    
    # Convert to uint16/uint8 for memory efficiency
    input_ids_arrays = [
        np.array(ids, dtype=np.uint16) for ids in all_input_ids
    ]
    attention_mask_arrays = [
        np.array(mask, dtype=np.uint8) for mask in all_attention_masks
    ]
    
    # Create PyArrow arrays
    input_ids_col = pa.array(input_ids_arrays, type=pa.list_(pa.uint16()))
    attention_mask_col = pa.array(attention_mask_arrays, type=pa.list_(pa.uint8()))
    token_count_col = pa.array(all_token_counts, type=pa.int16())
    
    # Build output table with new columns
    columns_to_keep = [
        col for col in table.column_names
        if col not in ("input_ids", "attention_mask", "token_count")
    ]
    
    output_columns = {name: table[name] for name in columns_to_keep}
    output_columns["input_ids"] = input_ids_col
    output_columns["attention_mask"] = attention_mask_col
    output_columns["token_count"] = token_count_col
    # Store metadata about which text field was tokenized
    output_columns["tokenized_text_field"] = pa.array(
        [text_field] * num_rows, type=pa.string()
    )
    
    # Create output table
    output_table = pa.table(output_columns)
    
    # Write to disk
    logger.info(f"Writing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    feather.write_feather(
        output_table,
        output_path,
        compression="lz4",  # Fast compression for mmap access
    )
    
    # Compute statistics
    elapsed = time.time() - start_time
    token_counts = np.array(all_token_counts)
    
    stats = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "num_rows": num_rows,
        "model_name": model_name,
        "max_length": max_length,
        "num_workers": num_workers,
        "elapsed_seconds": round(elapsed, 2),
        "rows_per_second": round(num_rows / elapsed, 1),
        "token_stats": {
            "mean": round(float(token_counts.mean()), 1),
            "std": round(float(token_counts.std()), 1),
            "min": int(token_counts.min()),
            "max": int(token_counts.max()),
            "median": int(np.median(token_counts)),
            "total": int(token_counts.sum()),
        },
    }
    
    logger.info(f"Completed in {elapsed:.1f}s ({stats['rows_per_second']:.0f} rows/s)")
    logger.info(f"Token stats: mean={stats['token_stats']['mean']:.0f}, "
                f"max={stats['token_stats']['max']}")
    
    return stats


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess dataset with parallel AOT tokenization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input Arrow file path",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output Arrow file path",
    )
    parser.add_argument(
        "--model", "-m",
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
        "--workers", "-w",
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
    
    args = parser.parse_args()
    
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    stats = preprocess_dataset(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        max_length=args.max_length,
        text_field=args.text_field,
        num_workers=args.workers,
        chunk_size=args.chunk_size,
    )
    
    logger.info("Preprocessing complete!")
    

if __name__ == "__main__":
    main()
