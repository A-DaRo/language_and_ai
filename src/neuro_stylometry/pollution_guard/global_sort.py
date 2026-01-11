"""
Global Sort Utilities for Staged Execution.

Implements the Scatter-Gather pattern for globally-sorted inference:
- Flatten: Convert nested post_chunked column to flat array with lineage tracking.
- Sort: Global argsort by token_count for minimal padding batches.
- Gather: Reconstruct per-document results after inference.

Reference: Technical Report Section 3 (The "Global Sort" Inference Engine)

Key Optimizations:
- Zero-copy token_count extraction via PyArrow child arrays.
- Vectorized argsort via NumPy (cache-friendly for millions of chunks).
- Efficient offset-based reconstruction using ListArray.offsets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa

logger = logging.getLogger(__name__)


@dataclass
class FlattenedChunks:
    """
    Result of flattening a nested post_chunked column.
    
    Attributes:
        texts: Flat list of all chunk texts in document order.
        token_counts: NumPy array of token counts (for sorting).
        doc_offsets: NumPy array of document boundary offsets.
            doc_offsets[i] is the start index of document i's chunks in texts.
            doc_offsets[-1] is len(texts) (sentinel).
        chunk_metadata: List of (doc_idx, chunk_idx_in_doc) for each flat chunk.
        sort_indices: Indices to sort chunks by token_count (computed lazily).
        inverse_indices: Inverse permutation to restore original order.
    """
    texts: List[str]
    token_counts: np.ndarray  # int16 for memory efficiency
    doc_offsets: np.ndarray   # int64 for large datasets
    chunk_metadata: List[Tuple[int, int]]  # (doc_idx, local_chunk_idx)
    sort_indices: Optional[np.ndarray] = None
    inverse_indices: Optional[np.ndarray] = None
    
    @property
    def num_chunks(self) -> int:
        return len(self.texts)
    
    @property
    def num_docs(self) -> int:
        return len(self.doc_offsets) - 1
    
    def compute_sort_indices(self) -> np.ndarray:
        """
        Compute global sort permutation by token_count (ascending).
        
        Sorts chunks so that consecutive chunks have similar lengths,
        minimizing padding waste in batched inference.
        
        Returns:
            Array of indices that would sort chunks by length.
        """
        if self.sort_indices is None:
            self.sort_indices = np.argsort(self.token_counts, kind="stable")
            # Compute inverse for reconstruction
            self.inverse_indices = np.argsort(self.sort_indices, kind="stable")
        return self.sort_indices
    
    def get_sorted_texts(self) -> List[str]:
        """Return chunk texts sorted by token_count."""
        sort_idx = self.compute_sort_indices()
        return [self.texts[i] for i in sort_idx]
    
    def get_sorted_token_counts(self) -> np.ndarray:
        """Return token counts in sorted order."""
        sort_idx = self.compute_sort_indices()
        return self.token_counts[sort_idx]


def flatten_chunks(
    post_chunked_column: pa.Array,
    extract_texts: bool = True,
) -> FlattenedChunks:
    """
    Flatten nested post_chunked column into flat arrays with lineage tracking.
    
    Uses PyArrow's zero-copy views for efficient access to nested data.
    The token_count child array is extracted as a contiguous NumPy array
    for O(1) access during sorting.
    
    Args:
        post_chunked_column: Arrow ListArray of chunk structs.
        extract_texts: If True, extract chunk texts (slower but needed for inference).
            If False, only extract metadata for planning/inspection.
            
    Returns:
        FlattenedChunks with flat texts, token_counts, and reconstruction info.
    """
    if not pa.types.is_list(post_chunked_column.type):
        raise TypeError(
            f"Expected ListArray for post_chunked, got {post_chunked_column.type}"
        )
    
    # Convert to ListArray for offset access
    if isinstance(post_chunked_column, pa.ChunkedArray):
        # Combine chunks for contiguous access
        post_chunked_column = post_chunked_column.combine_chunks()
    
    list_array = post_chunked_column
    
    # Get document offsets from ListArray
    # offsets[i] is the start of document i's chunks in the flattened array
    # offsets[i+1] - offsets[i] is the number of chunks in document i
    offsets = list_array.offsets.to_numpy()  # int32 or int64
    doc_offsets = offsets.astype(np.int64)
    
    # Get flattened struct array (all chunks across all documents)
    flat_structs = list_array.flatten()
    total_chunks = len(flat_structs)
    
    logger.debug(
        f"Flattening {len(doc_offsets) - 1} documents with {total_chunks} total chunks"
    )
    
    # Extract token_count as contiguous array (zero-copy if possible)
    try:
        token_count_array = flat_structs.field("token_count")
        token_counts = token_count_array.to_numpy(zero_copy_only=False).astype(np.int16)
    except Exception as exc:
        logger.warning(f"Failed to extract token_count field: {exc}. Using zeros.")
        token_counts = np.zeros(total_chunks, dtype=np.int16)
    
    # Extract texts if requested
    if extract_texts:
        try:
            text_array = flat_structs.field("text")
            texts = text_array.to_pylist()
        except Exception as exc:
            logger.warning(f"Failed to extract text field: {exc}. Using empty strings.")
            texts = [""] * total_chunks
    else:
        texts = []  # Caller will handle text extraction later
    
    # Build chunk metadata for reconstruction
    # chunk_metadata[flat_idx] = (doc_idx, local_chunk_idx)
    chunk_metadata: List[Tuple[int, int]] = []
    num_docs = len(doc_offsets) - 1
    
    for doc_idx in range(num_docs):
        start = doc_offsets[doc_idx]
        end = doc_offsets[doc_idx + 1]
        for local_idx, flat_idx in enumerate(range(start, end)):
            chunk_metadata.append((doc_idx, local_idx))
    
    return FlattenedChunks(
        texts=texts,
        token_counts=token_counts,
        doc_offsets=doc_offsets,
        chunk_metadata=chunk_metadata,
    )


def gather_results(
    flat_results: List[Any],
    flattened: FlattenedChunks,
    restore_sort_order: bool = True,
) -> List[List[Any]]:
    """
    Reconstruct per-document results from flat inference output.
    
    After inference on globally-sorted chunks, this function:
    1. Restores original (unsorted) order using inverse_indices.
    2. Groups results back into per-document lists using doc_offsets.
    
    Args:
        flat_results: Results from inference, one per chunk.
            If inference was on sorted chunks, these are in sorted order.
        flattened: FlattenedChunks with reconstruction metadata.
        restore_sort_order: If True, flat_results are in sorted order and
            need to be permuted back. If False, already in document order.
            
    Returns:
        List of lists: nested_results[doc_idx] = [results for doc's chunks].
    """
    if len(flat_results) != flattened.num_chunks:
        raise ValueError(
            f"Result count mismatch: {len(flat_results)} vs {flattened.num_chunks} chunks"
        )
    
    # Restore original order if results are from sorted inference
    if restore_sort_order and flattened.inverse_indices is not None:
        # Permute results back to document order
        restored_results = [None] * len(flat_results)
        for sorted_idx, orig_idx in enumerate(flattened.inverse_indices):
            restored_results[orig_idx] = flat_results[sorted_idx]
        flat_results = restored_results
    
    # Group by document using offsets
    nested_results: List[List[Any]] = []
    
    for doc_idx in range(flattened.num_docs):
        start = flattened.doc_offsets[doc_idx]
        end = flattened.doc_offsets[doc_idx + 1]
        doc_results = flat_results[start:end]
        nested_results.append(doc_results)
    
    return nested_results


def create_sorted_batches(
    flattened: FlattenedChunks,
    batch_size: int,
    max_tokens_per_batch: Optional[int] = None,
) -> List[List[int]]:
    """
    Create batches of chunk indices optimized for minimal padding.
    
    Chunks are sorted by token_count, then grouped into batches.
    Each batch contains chunks of similar length, minimizing padding waste.
    
    Args:
        flattened: FlattenedChunks with sort_indices computed.
        batch_size: Maximum chunks per batch.
        max_tokens_per_batch: Optional limit on total tokens per batch.
            If set, batches may be smaller to respect token budget.
            
    Returns:
        List of batches, where each batch is a list of sorted chunk indices.
    """
    sort_indices = flattened.compute_sort_indices()
    sorted_lengths = flattened.token_counts[sort_indices]
    
    batches: List[List[int]] = []
    current_batch: List[int] = []
    current_tokens = 0
    
    for i, sorted_idx in enumerate(sort_indices):
        chunk_len = sorted_lengths[i]
        
        # Check if adding this chunk exceeds limits
        would_exceed_batch_size = len(current_batch) >= batch_size
        would_exceed_tokens = (
            max_tokens_per_batch is not None
            and current_tokens + chunk_len > max_tokens_per_batch
        )
        
        if current_batch and (would_exceed_batch_size or would_exceed_tokens):
            # Commit current batch
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        
        current_batch.append(int(sorted_idx))
        current_tokens += chunk_len
    
    # Commit final batch
    if current_batch:
        batches.append(current_batch)
    
    logger.debug(
        f"Created {len(batches)} batches from {flattened.num_chunks} chunks "
        f"(batch_size={batch_size})"
    )
    
    return batches


def compute_padding_stats(flattened: FlattenedChunks, batch_size: int) -> Dict[str, float]:
    """
    Compute padding efficiency statistics for sorted batching.
    
    Compares padding waste between:
    - Unsorted batching (chunks in document order)
    - Sorted batching (chunks sorted by length)
    
    Args:
        flattened: FlattenedChunks with token counts.
        batch_size: Batch size to use for comparison.
        
    Returns:
        Dict with padding statistics:
        - unsorted_padding_ratio: Fraction of tokens wasted without sorting
        - sorted_padding_ratio: Fraction of tokens wasted with sorting
        - efficiency_gain: Reduction in padding (1.0 = 100% reduction)
    """
    if flattened.num_chunks == 0:
        return {
            "unsorted_padding_ratio": 0.0,
            "sorted_padding_ratio": 0.0,
            "efficiency_gain": 0.0,
        }
    
    token_counts = flattened.token_counts
    
    def compute_padding(indices: np.ndarray) -> float:
        """Compute total padding for batches of given indices."""
        total_padding = 0
        total_tokens = 0
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_lengths = token_counts[batch_indices]
            max_len = batch_lengths.max() if len(batch_lengths) > 0 else 0
            batch_padding = max_len * len(batch_lengths) - batch_lengths.sum()
            total_padding += batch_padding
            total_tokens += max_len * len(batch_lengths)
        
        return total_padding / total_tokens if total_tokens > 0 else 0.0
    
    # Unsorted (document order)
    unsorted_indices = np.arange(flattened.num_chunks)
    unsorted_ratio = compute_padding(unsorted_indices)
    
    # Sorted (by length)
    sort_indices = flattened.compute_sort_indices()
    sorted_ratio = compute_padding(sort_indices)
    
    efficiency_gain = (unsorted_ratio - sorted_ratio) / unsorted_ratio if unsorted_ratio > 0 else 0.0
    
    return {
        "unsorted_padding_ratio": unsorted_ratio,
        "sorted_padding_ratio": sorted_ratio,
        "efficiency_gain": efficiency_gain,
    }
