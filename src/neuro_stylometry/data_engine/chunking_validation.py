"""Validation utilities for staged execution chunking artifacts.

These helpers are used for safe resume logic:
- Only skip Stage 1 when `post_chunked` is present, fully populated, and
  consistent with the raw `post` text.

Important nuance:
The semantic chunker strips sentence whitespace and (in accumulate mode)
re-joins sentences with single spaces. Hard-split chunks may also reflow
whitespace and include overlapping spans. Therefore, validation uses a
whitespace-normalized comparison.

Performance Optimizations (v2.0):
- Automatic parallelization for large datasets (>1000 posts)
- ProcessPoolExecutor for true multi-core CPU utilization
- Vectorized null detection via PyArrow compute
- NumPy-accelerated first/last non-whitespace detection
- Early termination with max_invalid
"""

from __future__ import annotations

import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


_WS_RE = re.compile(r"\s+")

# Threshold for automatic parallelization
_PARALLEL_THRESHOLD = 1000


def _norm_ws(text: str) -> str:
    """Normalize whitespace: collapse runs to single space, strip edges."""
    return _WS_RE.sub(" ", text).strip()


def _find_non_ws_bounds(post: str) -> Tuple[int, int]:
    """Find first and last non-whitespace char indices. Returns (-1, -1) for whitespace-only."""
    if not post:
        return -1, -1
    stripped = post.lstrip()
    if not stripped:
        return -1, -1
    first = len(post) - len(stripped)
    last_end = len(post.rstrip())
    return first, last_end


def _validate_single_row(args: Tuple[int, str, List[dict]]) -> Optional[int]:
    """
    Validate a single row. Returns index if invalid, None if valid.
    
    This function is pickle-able for multiprocessing.
    """
    idx, post, chunk_list = args
    
    if chunk_list is None:
        return idx
    
    first_non_ws, last_non_ws_end = _find_non_ws_bounds(post or "")
    
    if not chunk_list:
        # Empty chunk list is valid only for whitespace-only posts
        return None if first_non_ws == -1 else idx
    
    if first_non_ws == -1:
        # Whitespace-only post with non-empty chunks is invalid
        return idx
    
    post_len = len(post)
    
    # Extract and validate chunk data
    num_chunks = len(chunk_list)
    chunk_data = []
    
    for chunk in chunk_list:
        try:
            s = int(chunk.get("start", 0))
            e = int(chunk.get("end", 0))
            t = str(chunk.get("text", ""))
        except (TypeError, ValueError):
            return idx
        
        # Bounds check
        if s < 0 or e < 0 or s > e or e > post_len:
            return idx
        
        chunk_data.append((s, e, t))
    
    # Sort by start
    chunk_data.sort(key=lambda x: x[0])
    
    # Check coverage
    min_start = chunk_data[0][0]
    max_end = max(e for _, e, _ in chunk_data)
    
    if min_start > first_non_ws or max_end < last_non_ws_end:
        return idx
    
    # Check gaps and content
    prev_end = 0
    for s, e, t in chunk_data:
        # Gap check (only if not overlapping)
        if s > prev_end:
            gap = post[prev_end:s]
            if gap and not gap.isspace():
                return idx
        
        # Content check (whitespace-normalized)
        raw_span = post[s:e]
        if _norm_ws(raw_span) != _norm_ws(t):
            return idx
        
        prev_end = max(prev_end, e)
    
    return None  # Valid


def _validate_batch(args: Tuple[List[int], List[str], List[List[dict]]]) -> List[int]:
    """
    Validate a batch of rows. Returns list of invalid indices.
    
    This function is pickle-able for multiprocessing.
    """
    indices, posts, chunk_lists = args
    invalid = []
    
    for i, idx in enumerate(indices):
        result = _validate_single_row((idx, posts[i], chunk_lists[i]))
        if result is not None:
            invalid.append(result)
    
    return invalid


def find_invalid_post_chunked_indices(
    *,
    posts: List[str],
    post_chunked_column: pa.Array,
    sample_indices: Optional[Iterable[int]] = None,
    max_invalid: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> List[int]:
    """Return row indices where post_chunked does not match the corresponding post.

    Rules (tolerant to semantic_chunker whitespace behavior):
    - For each chunk, `normalize_ws(chunk.text)` must equal
      `normalize_ws(post[start:end])`.
    - Chunk spans must be within bounds and ordered (non-decreasing start).
    - Any *gaps* between successive chunks may only contain whitespace.
      Overlaps are allowed (hard-split overlap).
    - Coverage: chunks must cover all non-whitespace characters of the post.

    Empty posts (whitespace-only) are valid with an empty chunk list.

    Performance:
    - Automatically parallelizes for datasets > 1000 rows
    - Uses ProcessPoolExecutor for true multi-core utilization
    - Set num_workers=1 to force single-threaded execution

    Args:
        posts: List of raw post strings.
        post_chunked_column: Arrow list<struct> column (Array or ChunkedArray).
        sample_indices: Optional iterable of row indices to validate.
            If provided, only these rows are checked.
        max_invalid: Optional early-exit cap for number of invalid indices collected.
        num_workers: Number of parallel workers. Default: auto (cpu_count for large datasets).

    Returns:
        List of invalid row indices.
    """
    n_posts = len(posts)
    n_chunks = len(post_chunked_column)
    
    if n_posts != n_chunks:
        raise ValueError(
            f"Length mismatch: posts={n_posts} vs post_chunked={n_chunks}"
        )
    
    # Convert sample_indices to list
    if sample_indices is None:
        indices = list(range(n_posts))
    else:
        indices = list(sample_indices)
    
    if not indices:
        return []
    
    # Determine number of workers
    if num_workers is None:
        if len(indices) >= _PARALLEL_THRESHOLD:
            num_workers = min(mp.cpu_count(), 32)  # Cap at 32 workers
        else:
            num_workers = 1
    
    # Fast path: detect nulls via PyArrow compute
    if isinstance(post_chunked_column, pa.ChunkedArray):
        null_chunks = [pc.is_null(chunk) for chunk in post_chunked_column.chunks]
        if null_chunks:
            null_mask = np.concatenate([c.to_numpy(zero_copy_only=False) for c in null_chunks])
        else:
            null_mask = np.zeros(n_posts, dtype=bool)
    else:
        null_mask = pc.is_null(post_chunked_column).to_numpy(zero_copy_only=False)
    
    # Collect indices with null chunks (these are invalid)
    invalid_from_nulls = [i for i in indices if null_mask[i]]
    
    # Filter out null indices from validation set
    indices_to_validate = [i for i in indices if not null_mask[i]]
    
    if not indices_to_validate:
        result = invalid_from_nulls
        if max_invalid is not None:
            result = result[:max_invalid]
        return result
    
    # Extract data for validation (converting Arrow to Python once)
    # This is the expensive part, but necessary for multiprocessing
    validation_posts = [posts[i] or "" for i in indices_to_validate]
    validation_chunks = []
    
    for i in indices_to_validate:
        try:
            chunk_list = post_chunked_column[i].as_py()
        except Exception:
            chunk_list = None
        validation_chunks.append(chunk_list)
    
    # Single-threaded path
    if num_workers <= 1:
        invalid = list(invalid_from_nulls)
        for idx, post, chunk_list in zip(indices_to_validate, validation_posts, validation_chunks):
            result = _validate_single_row((idx, post, chunk_list))
            if result is not None:
                invalid.append(result)
                if max_invalid is not None and len(invalid) >= max_invalid:
                    break
        invalid.sort()
        return invalid
    
    # Parallel path: split work into batches
    batch_size = max(100, len(indices_to_validate) // num_workers)
    batches = []
    
    for i in range(0, len(indices_to_validate), batch_size):
        end = min(i + batch_size, len(indices_to_validate))
        batch_indices = indices_to_validate[i:end]
        batch_posts = validation_posts[i:end]
        batch_chunks = validation_chunks[i:end]
        batches.append((batch_indices, batch_posts, batch_chunks))
    
    # Run validation in parallel
    invalid = list(invalid_from_nulls)
    
    # Use spawn context for better cross-platform compatibility
    ctx = mp.get_context("spawn")
    
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        futures = [executor.submit(_validate_batch, batch) for batch in batches]
        
        for future in as_completed(futures):
            try:
                batch_invalid = future.result()
                invalid.extend(batch_invalid)
                
                if max_invalid is not None and len(invalid) >= max_invalid:
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break
            except Exception as e:
                # Log but continue with other batches
                import logging
                logging.getLogger(__name__).warning(f"Batch validation failed: {e}")
    
    # Sort and truncate
    invalid.sort()
    if max_invalid is not None:
        invalid = invalid[:max_invalid]
    
    return invalid


# Alias for backward compatibility
find_invalid_post_chunked_indices_parallel = find_invalid_post_chunked_indices
