"""AOT tokenization utilities for Phase D preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
from tqdm import tqdm

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
_worker_pre_pad: bool = False


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


def _tokenize_batch(
    batch: List[Tuple[int, str]],
) -> List[Tuple[int, np.ndarray, np.ndarray, int]]:
    """
    Tokenize a batch of texts in a worker process.

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
            encoding = _worker_tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_attention_mask=True,
            )
            input_ids = np.array(encoding["input_ids"], dtype=np.int32)
            attention_mask = np.array(encoding["attention_mask"], dtype=np.int8)
            token_count = int(attention_mask.sum())
        else:
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
    """Yield chunks of (index, text) tuples for parallel processing."""
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

    Returns:
        Statistics dictionary with tokenization results.
    """
    start_time = time.time()

    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    logger.info("Loading input dataset: %s", input_path)
    table = feather.read_table(input_path, memory_map=True)
    num_rows = table.num_rows

    logger.info("Dataset: %d rows, %d columns", num_rows, len(table.column_names))
    logger.info("Text field: %s", text_field)
    logger.info("Tokenizer: %s, max_length=%d", model_name, max_length)
    logger.info("Workers: %d", num_workers)
    logger.info("Pre-padding mode: %s (zero-copy collation)", pre_pad)

    if text_field not in table.column_names:
        available = ", ".join(table.column_names)
        raise ValueError(f"Text field '{text_field}' not found. Available: {available}")

    config = TokenizerConfig(
        model_name=model_name,
        max_length=max_length,
        taxonomy_path=taxonomy_path,
        pre_pad=pre_pad,
    )

    all_input_ids: List[Optional[np.ndarray]] = [None] * num_rows
    all_attention_masks: List[Optional[np.ndarray]] = [None] * num_rows
    all_token_counts: List[int] = [0] * num_rows

    logger.info("Starting parallel tokenization...")
    chunks = list(_chunk_iterator(table, text_field, chunk_size))

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

    missing = sum(1 for ids in all_input_ids if ids is None)
    if missing > 0:
        raise RuntimeError(f"Failed to tokenize {missing} rows")

    logger.info("Building Arrow columns...")

    if pre_pad:
        input_ids_col = pa.FixedSizeListArray.from_arrays(
            pa.array(np.stack(all_input_ids).flatten(), type=pa.int32()),
            max_length,
        )
        attention_mask_col = pa.FixedSizeListArray.from_arrays(
            pa.array(np.stack(all_attention_masks).flatten(), type=pa.int8()),
            max_length,
        )
        logger.info("Pre-padded storage: %d tokens/sample (zero-copy ready)", max_length)
    else:
        input_ids_arrays = [
            ids.astype(np.uint16) if ids.dtype != np.uint16 else ids
            for ids in all_input_ids
        ]
        attention_mask_arrays = [
            mask.astype(np.uint8) if mask.dtype != np.uint8 else mask
            for mask in all_attention_masks
        ]
        input_ids_col = pa.array(input_ids_arrays, type=pa.list_(pa.uint16()))
        attention_mask_col = pa.array(attention_mask_arrays, type=pa.list_(pa.uint8()))

    token_count_col = pa.array(all_token_counts, type=pa.int16())

    columns_to_keep = [
        col for col in table.column_names
        if col not in ("input_ids", "attention_mask", "token_count")
    ]

    output_columns = {name: table[name] for name in columns_to_keep}
    output_columns["input_ids"] = input_ids_col
    output_columns["attention_mask"] = attention_mask_col
    output_columns["token_count"] = token_count_col
    output_columns["tokenized_text_field"] = pa.array(
        [text_field] * num_rows, type=pa.string()
    )
    output_columns["is_pre_padded"] = pa.array(
        [pre_pad] * num_rows, type=pa.bool_()
    )
    if pre_pad:
        output_columns["padded_length"] = pa.array(
            [max_length] * num_rows, type=pa.int16()
        )

    output_table = pa.table(output_columns)

    logger.info("Writing output: %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compression = None if pre_pad else "lz4"
    logger.info("Compression: %s", compression or "none (zero-copy optimized)")
    feather.write_feather(
        output_table,
        output_path,
        compression=compression,
    )

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

    logger.info("Completed in %.1fs (%.0f rows/s)", elapsed, stats["rows_per_second"])
    logger.info(
        "Token stats: mean=%.0f, max=%d",
        stats["token_stats"]["mean"],
        stats["token_stats"]["max"],
    )

    return stats
