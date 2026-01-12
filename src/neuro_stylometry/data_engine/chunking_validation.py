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
- Vectorized null detection via PyArrow compute
- Batched whitespace normalization
- NumPy-accelerated first/last non-whitespace detection
- Early termination with bitmap operations
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


_WS_RE = re.compile(r"\s+")


def _norm_ws(text: str) -> str:
    """Normalize whitespace: collapse runs to single space, strip edges."""
    return _WS_RE.sub(" ", text).strip()


def _find_non_ws_bounds_vectorized(posts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find first and last non-whitespace character indices for all posts.
    
    Returns:
        Tuple of (first_non_ws, last_non_ws_end) arrays.
        Values are -1 for whitespace-only posts.
    """
    n = len(posts)
    first_non_ws = np.full(n, -1, dtype=np.int32)
    last_non_ws_end = np.full(n, -1, dtype=np.int32)
    
    for i, post in enumerate(posts):
        if not post:
            continue
        # Use string methods which are C-optimized
        stripped = post.lstrip()
        if stripped:
            first_non_ws[i] = len(post) - len(stripped)
            rstripped = post.rstrip()
            last_non_ws_end[i] = len(rstripped)
    
    return first_non_ws, last_non_ws_end


def _batch_normalize_ws(texts: List[str]) -> List[str]:
    """Batch normalize whitespace for a list of texts."""
    return [_WS_RE.sub(" ", t).strip() if t else "" for t in texts]


def _validate_chunk_list_fast(
    post: str,
    chunk_list: List[dict],
    first_non_ws: int,
    last_non_ws_end: int,
) -> bool:
    """
    Fast validation of a single chunk list against its post.
    
    Optimized version with early exits and minimal allocations.
    """
    if not chunk_list:
        # Empty chunk list is valid only for whitespace-only posts
        return first_non_ws == -1
    
    if first_non_ws == -1:
        # Whitespace-only post with non-empty chunks is invalid
        return False
    
    post_len = len(post)
    
    # Extract and validate chunk data in one pass
    num_chunks = len(chunk_list)
    starts = np.empty(num_chunks, dtype=np.int32)
    ends = np.empty(num_chunks, dtype=np.int32)
    texts = []
    
    for j, chunk in enumerate(chunk_list):
        try:
            s = int(chunk.get("start", 0))
            e = int(chunk.get("end", 0))
            t = str(chunk.get("text", ""))
        except (TypeError, ValueError):
            return False
        
        # Bounds check
        if s < 0 or e < 0 or s > e or e > post_len:
            return False
        
        starts[j] = s
        ends[j] = e
        texts.append(t)
    
    # Sort by start (most chunks are already sorted)
    if num_chunks > 1:
        sort_idx = np.argsort(starts)
        starts = starts[sort_idx]
        ends = ends[sort_idx]
        texts = [texts[i] for i in sort_idx]
    
    # Check coverage: min_start <= first_non_ws, max_end >= last_non_ws_end
    min_start = int(starts[0])
    max_end = int(np.max(ends))
    
    if min_start > first_non_ws or max_end < last_non_ws_end:
        return False
    
    # Check gaps and content
    prev_end = 0
    for j in range(num_chunks):
        s, e = int(starts[j]), int(ends[j])
        
        # Gap check (only if not overlapping)
        if s > prev_end:
            gap = post[prev_end:s]
            # Fast whitespace-only check
            if gap and not gap.isspace():
                return False
        
        # Content check
        raw_span = post[s:e]
        if _norm_ws(raw_span) != _norm_ws(texts[j]):
            return False
        
        prev_end = max(prev_end, e)
    
    return True


def find_invalid_post_chunked_indices(
    *,
    posts: List[str],
    post_chunked_column: pa.Array,
    sample_indices: Optional[Iterable[int]] = None,
    max_invalid: Optional[int] = None,
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

    Performance Optimizations:
    - Vectorized null detection via PyArrow compute
    - Pre-computed non-whitespace bounds via string methods
    - NumPy-accelerated sorting and bounds checking
    - Early termination via max_invalid

    Args:
        posts: List of raw post strings.
        post_chunked_column: Arrow list<struct> column (Array or ChunkedArray).
        sample_indices: Optional iterable of row indices to validate.
            If provided, only these rows are checked.
        max_invalid: Optional early-exit cap for number of invalid indices collected.

    Returns:
        List of invalid row indices.
    """
    n_posts = len(posts)
    n_chunks = len(post_chunked_column)
    
    if n_posts != n_chunks:
        raise ValueError(
            f"Length mismatch: posts={n_posts} vs post_chunked={n_chunks}"
        )
    
    # Convert sample_indices to list for efficient indexing
    if sample_indices is None:
        indices = list(range(n_posts))
    else:
        indices = list(sample_indices)
    
    if not indices:
        return []
    
    # Fast path: detect null chunks via PyArrow compute
    # Handle ChunkedArray without combine_chunks() to avoid offset overflow
    # on large datasets with nested list structures
    if isinstance(post_chunked_column, pa.ChunkedArray):
        # Process chunks individually to avoid offset overflow
        # pc.is_null returns small boolean arrays that can be safely concatenated
        null_chunks = [pc.is_null(chunk) for chunk in post_chunked_column.chunks]
        if null_chunks:
            null_mask = np.concatenate([chunk.to_numpy() for chunk in null_chunks])
        else:
            null_mask = np.zeros(n_posts, dtype=bool)
    else:
        null_mask = pc.is_null(post_chunked_column).to_numpy()
    
    # Pre-compute non-whitespace bounds for all posts we'll check
    # Only compute for indices we'll actually validate
    indices_set = set(indices)
    posts_to_check = [posts[i] if i in indices_set else "" for i in range(n_posts)]
    first_non_ws, last_non_ws_end = _find_non_ws_bounds_vectorized(posts_to_check)
    
    invalid: List[int] = []
    
    for i in indices:
        # Fast null check via precomputed mask
        if null_mask[i]:
            invalid.append(i)
            if max_invalid is not None and len(invalid) >= max_invalid:
                break
            continue
        
        post = posts[i] or ""
        fnw = int(first_non_ws[i])
        lnw = int(last_non_ws_end[i])
        
        # Extract chunk list
        try:
            chunk_list = post_chunked_column[i].as_py()
        except Exception:
            invalid.append(i)
            if max_invalid is not None and len(invalid) >= max_invalid:
                break
            continue
        
        if chunk_list is None:
            invalid.append(i)
            if max_invalid is not None and len(invalid) >= max_invalid:
                break
            continue
        
        # Validate chunk list
        if not _validate_chunk_list_fast(post, chunk_list, fnw, lnw):
            invalid.append(i)
            if max_invalid is not None and len(invalid) >= max_invalid:
                break
    
    return invalid


def find_invalid_post_chunked_indices_parallel(
    *,
    posts: List[str],
    post_chunked_column: pa.Array,
    sample_indices: Optional[Iterable[int]] = None,
    max_invalid: Optional[int] = None,
    num_workers: int = 4,
) -> List[int]:
    """
    Parallel version of find_invalid_post_chunked_indices.
    
    Uses multiprocessing to validate chunks across multiple CPU cores.
    Recommended for large datasets (>10k posts).
    
    Args:
        posts: List of raw post strings.
        post_chunked_column: Arrow list<struct> column.
        sample_indices: Optional row indices to validate.
        max_invalid: Early-exit cap for invalid count.
        num_workers: Number of parallel workers.
        
    Returns:
        List of invalid row indices (sorted).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    n_posts = len(posts)
    
    if sample_indices is None:
        indices = list(range(n_posts))
    else:
        indices = list(sample_indices)
    
    if not indices or num_workers <= 1:
        return find_invalid_post_chunked_indices(
            posts=posts,
            post_chunked_column=post_chunked_column,
            sample_indices=sample_indices,
            max_invalid=max_invalid,
        )
    
    # Split indices into chunks for workers
    chunk_size = max(1, len(indices) // num_workers)
    index_chunks = [
        indices[i:i + chunk_size]
        for i in range(0, len(indices), chunk_size)
    ]
    
    # Pre-compute bounds for all posts
    first_non_ws, last_non_ws_end = _find_non_ws_bounds_vectorized(posts)
    
    # Pre-compute null mask without combine_chunks() to avoid offset overflow
    if isinstance(post_chunked_column, pa.ChunkedArray):
        null_chunks = [pc.is_null(chunk) for chunk in post_chunked_column.chunks]
        if null_chunks:
            null_mask = np.concatenate([chunk.to_numpy() for chunk in null_chunks])
        else:
            null_mask = np.zeros(len(posts), dtype=bool)
    else:
        null_mask = pc.is_null(post_chunked_column).to_numpy()
    
    all_invalid: List[int] = []
    
    def validate_chunk(idx_chunk: List[int]) -> List[int]:
        chunk_invalid = []
        for i in idx_chunk:
            if null_mask[i]:
                chunk_invalid.append(i)
                continue
            
            post = posts[i] or ""
            fnw = int(first_non_ws[i])
            lnw = int(last_non_ws_end[i])
            
            try:
                chunk_list = post_chunked_column[i].as_py()
            except Exception:
                chunk_invalid.append(i)
                continue
            
            if chunk_list is None or not _validate_chunk_list_fast(post, chunk_list, fnw, lnw):
                chunk_invalid.append(i)
        
        return chunk_invalid
    
    # Use threads (GIL released during Arrow operations)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(validate_chunk, chunk) for chunk in index_chunks]
        
        for future in as_completed(futures):
            chunk_invalid = future.result()
            all_invalid.extend(chunk_invalid)
            
            if max_invalid is not None and len(all_invalid) >= max_invalid:
                # Cancel remaining futures
                for f in futures:
                    f.cancel()
                break
    
    # Sort and truncate
    all_invalid.sort()
    if max_invalid is not None:
        all_invalid = all_invalid[:max_invalid]
    
    return all_invalid
