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
    pre_pad: bool = False  # If True, pad all sequences to max_length for zero-copy collation


# Global tokenizer for worker processes (initialized once per worker)
_worker_tokenizer = None


def _init_worker(config: TokenizerConfig) -> None:
    """Initialize tokenizer in worker process."""
    global _worker_tokenizer, _worker_pre_pad
    
    # Import here to avoid loading in main process
    from transformers import AutoTokenizer
    
    _worker_tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        use_fast=True,  # Use Rust tokenizer for speed
    )
    
    # Configure for our schema
    _worker_tokenizer.model_max_length = config.max_length
    _worker_pre_pad = config.pre_pad


# Global flag for pre-padding mode
_worker_pre_pad: bool = False


def _tokenize_batch(
    batch: List[Tuple[int, str]],
) -> List[Tuple[int, np.ndarray, np.ndarray, int]]:
    """
    Tokenize a batch of texts in a worker process.
    
    Args:
        batch: List of (index, text) tuples.
        
    Returns:
        List of (index, input_ids, attention_mask, token_count) tuples.
        When pre_pad=True, arrays are fixed-length (max_length) for zero-copy collation.
    """
    global _worker_tokenizer, _worker_pre_pad
    
    if _worker_tokenizer is None:
        raise RuntimeError("Worker tokenizer not initialized")
    
    results = []
    max_length = _worker_tokenizer.model_max_length
    
    for idx, text in batch:
        if text is None:
            text = ""
        
        if _worker_pre_pad:
            # Pre-padded mode: pad to fixed max_length for zero-copy collation
            # This enables torch.stack() in collator without per-sample copying
            encoding = _worker_tokenizer(
                text,
                padding="max_length",  # Pad to fixed length
                truncation=True,
                max_length=max_length,
                return_attention_mask=True,
            )
            
            # Store as fixed-size numpy arrays (enables zero-copy in collator)
            # Use int32 for input_ids (tokenizer vocab can exceed uint16)
            # Use int8 for attention_mask (0/1 values)
            input_ids = np.array(encoding["input_ids"], dtype=np.int32)
            attention_mask = np.array(encoding["attention_mask"], dtype=np.int8)
            
            # token_count is the actual (non-padded) length
            token_count = int(attention_mask.sum())
        else:
            # Variable-length mode: no padding, collator handles dynamic padding
            encoding = _worker_tokenizer(
                text,
                padding=False,
                truncation=True,
                max_length=max_length,
                return_attention_mask=True,
            )
            
            input_ids = np.array(encoding["input_ids"], dtype=np.uint16)
            attention_mask = np.array(encoding["attention_mask"], dtype=np.uint8)
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
    pre_pad: bool = False,
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
    logger.info(f"Pre-padding mode: {pre_pad} (zero-copy collation)")
    
    # Validate text field exists
    if text_field not in table.column_names:
        available = ", ".join(table.column_names)
        raise ValueError(f"Text field '{text_field}' not found. Available: {available}")
    
    # Prepare tokenizer config
    config = TokenizerConfig(
        model_name=model_name,
        max_length=max_length,
        taxonomy_path=taxonomy_path,
        pre_pad=pre_pad,
    )
    
    # Pre-allocate result arrays
    all_input_ids: List[Optional[np.ndarray]] = [None] * num_rows
    all_attention_masks: List[Optional[np.ndarray]] = [None] * num_rows
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
    
    if pre_pad:
        # Pre-padded mode: store as fixed-size tensors for zero-copy collation
        # Use FixedSizeList for efficient memory layout and potential zero-copy
        input_ids_col = pa.FixedSizeListArray.from_arrays(
            pa.array(np.stack(all_input_ids).flatten(), type=pa.int32()),
            max_length,
        )
        attention_mask_col = pa.FixedSizeListArray.from_arrays(
            pa.array(np.stack(all_attention_masks).flatten(), type=pa.int8()),
            max_length,
        )
        logger.info(f"Pre-padded storage: {max_length} tokens/sample (zero-copy ready)")
    else:
        # Variable-length mode: convert to uint16/uint8 for memory efficiency
        input_ids_arrays = [
            ids.astype(np.uint16) if ids.dtype != np.uint16 else ids
            for ids in all_input_ids
        ]
        attention_mask_arrays = [
            mask.astype(np.uint8) if mask.dtype != np.uint8 else mask
            for mask in all_attention_masks
        ]
        
        # Create PyArrow arrays (variable-length lists)
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
    
    # Store pre-pad metadata for downstream zero-copy optimization
    output_columns["is_pre_padded"] = pa.array(
        [pre_pad] * num_rows, type=pa.bool_()
    )
    if pre_pad:
        output_columns["padded_length"] = pa.array(
            [max_length] * num_rows, type=pa.int16()
        )
    
    # Create output table
    output_table = pa.table(output_columns)
    
    # Write to disk
    logger.info(f"Writing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use uncompressed for pre-padded data to enable true zero-copy mmap
    # Use lz4 for variable-length data (better space efficiency)
    compression = None if pre_pad else "lz4"
    logger.info(f"Compression: {compression or 'none (zero-copy optimized)'}")
    
    feather.write_feather(
        output_table,
        output_path,
        compression=compression,
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
        "pre_pad": pre_pad,
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
    parser.add_argument(
        "--pre-pad",
        action="store_true",
        default=False,
        help="Pre-pad all sequences to max-length for zero-copy collation (HPC mode)",
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
        pre_pad=args.pre_pad,
    )
    
    logger.info("Preprocessing complete!")
    

if __name__ == "__main__":
    main()
