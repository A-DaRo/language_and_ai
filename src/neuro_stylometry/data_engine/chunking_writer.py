"""
Streaming Chunking Artifact Writer for Staged Execution.

Implements the Producer-Consumer pattern for safe parallel chunking with
streaming writes to disk, avoiding OOM for large datasets (>1M rows).

Reference: Technical Report Section 2 (Stage 1: Streaming Parallel Chunking)

Architecture:
- Main Process: Orchestrates multiprocessing pool and streams results to Arrow IPC.
- Workers: Accept raw texts, perform PySBD/Tokenization, return chunk dictionaries.
- Streaming: Micro-batches are flushed to disk immediately, keeping RAM constant.

Key Benefits:
- Safety: File I/O restricted to main process (no race conditions).
- Efficiency: Workers run at full CPU saturation via imap_unordered.
- Resilience: Chunks persisted to disk enable crash recovery.
"""

from __future__ import annotations

import gc
import logging
import multiprocessing as mp
import os
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from .schemas import CHUNK_STRUCT, POST_CHUNKED_FIELD

logger = logging.getLogger(__name__)


# Global chunker for multiprocessing workers
_WORKER_CHUNKER: Optional[Any] = None
_WORKER_TOKENIZER_NAME: Optional[str] = None


def _worker_init(
    tokenizer_name: str,
    budget_config_dict: Dict[str, Any],
    language: str,
    words_splitter_type: Optional[str],
) -> None:
    """Initialize chunker in worker process."""
    global _WORKER_CHUNKER, _WORKER_TOKENIZER_NAME
    
    # Prevent nested tokenizer parallelism
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    
    from transformers import AutoTokenizer
    from ..pollution_guard.semantic_chunker import SemanticChunker, BudgetConfig
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True,
            local_files_only=True,
            legacy=True,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True,
            legacy=True,
        )
    
    # Reconstruct BudgetConfig from dict
    budget_config = BudgetConfig(**budget_config_dict)
    # Disable nested parallelism in worker
    budget_config = replace(budget_config, parallel_chunking_workers=0)
    
    # Reconstruct words_splitter if type is known
    words_splitter = None
    if words_splitter_type:
        try:
            from gliner.data_processing.tokenizer import WordsSplitter
            words_splitter = WordsSplitter(splitter_type=words_splitter_type)
        except Exception as exc:
            logger.debug(f"Worker failed to create WordsSplitter: {exc}")
    
    _WORKER_CHUNKER = SemanticChunker(
        tokenizer=tokenizer,
        config=budget_config,
        language=language,
        words_splitter=words_splitter,
        words_splitter_type=words_splitter_type,
    )
    _WORKER_TOKENIZER_NAME = tokenizer_name


def _worker_chunk_batch(
    args: Tuple[List[int], List[str], List[Optional[List[str]]]],
) -> Tuple[List[int], List[List[Dict[str, Any]]]]:
    """
    Worker task: chunk a batch of documents and return Arrow-compatible dicts.
    
    Args:
        args: Tuple of (doc_indices, texts, labels_list)
        
    Returns:
        Tuple of (doc_indices, list_of_chunk_dicts_per_doc)
    """
    global _WORKER_CHUNKER
    
    doc_indices, texts, labels_list = args
    
    if _WORKER_CHUNKER is None:
        raise RuntimeError("Worker chunker not initialized")
    
    results: List[List[Dict[str, Any]]] = []
    
    for text, labels in zip(texts, labels_list):
        if not text or not text.strip() or not labels:
            results.append([])
            continue
        
        chunks = _WORKER_CHUNKER.chunk_text(text, labels)
        # Convert ChunkInfo to Arrow-compatible dicts
        chunk_dicts = [chunk.to_arrow_struct() for chunk in chunks]
        results.append(chunk_dicts)
    
    return doc_indices, results


def _worker_chunk_one(
    args: Tuple[int, str, Optional[List[str]]],
) -> Tuple[int, List[Dict[str, Any]]]:
    """Worker task: chunk a single document and return Arrow-compatible dicts."""
    global _WORKER_CHUNKER

    doc_idx, text, labels = args

    if _WORKER_CHUNKER is None:
        raise RuntimeError("Worker chunker not initialized")

    if not text or not text.strip() or not labels:
        return doc_idx, []

    chunks = _WORKER_CHUNKER.chunk_text(text, labels)
    chunk_dicts = [chunk.to_arrow_struct() for chunk in chunks]
    return doc_idx, chunk_dicts


class ChunkingArtifactWriter:
    """
    Streaming writer for chunked text artifacts.
    
    Implements producer-consumer pattern with memory-mapped streaming to avoid
    OOM on large datasets. Uses multiprocessing for CPU-bound chunking work
    while keeping file I/O on the main thread for safety.
    
    Usage:
        writer = ChunkingArtifactWriter(...)
        table_with_chunks = writer.process_and_save(
            posts=posts,
            labels_list=labels_list,
            post_ids=post_ids,
        )
    """
    
    def __init__(
        self,
        tokenizer_name: str,
        budget_config: Any,  # BudgetConfig
        words_splitter_type: Optional[str] = None,
        language: str = "en",
        num_workers: int = 0,
        micro_batch_size: int = 1000,
        progress_callback: Optional[Callable[[int], None]] = None,
    ):
        """
        Initialize the chunking writer.
        
        Args:
            tokenizer_name: HuggingFace tokenizer identifier.
            budget_config: BudgetConfig for semantic chunker.
            words_splitter_type: GLiNER words splitter type (e.g., 'whitespace').
            language: Language for PySBD segmenter.
            num_workers: Number of worker processes. 0 = sequential (single-process).
            micro_batch_size: Documents per micro-batch for streaming writes.
            progress_callback: Optional callback(num_docs_processed).
        """
        self.tokenizer_name = tokenizer_name
        self.budget_config = budget_config
        self.words_splitter_type = words_splitter_type
        self.language = language
        self.num_workers = num_workers
        self.micro_batch_size = micro_batch_size
        self.progress_callback = progress_callback
        
        # Convert BudgetConfig to dict for pickling
        self._budget_config_dict = {
            field: getattr(budget_config, field)
            for field in budget_config.__dataclass_fields__
        }
    
    def process_and_save(
        self,
        posts: List[str],
        labels_list: List[Optional[List[str]]],
        table: pa.Table,
        indices_to_recompute: Optional[List[int]] = None,
    ) -> pa.Table:
        """
        Chunk all posts and append post_chunked column to table.
        
        Implements the streaming writer pattern:
        1. Spawn worker pool for parallel chunking
        2. Stream results via imap_unordered
        3. Accumulate micro-batches and flush to temporary Arrow IPC
        4. Read back and attach as column to original table
        
        Args:
            posts: List of raw post texts.
            labels_list: Per-document labels for budget calculation.
            table: Original PyArrow table to augment.
            
        Returns:
            Table with post_chunked column appended.
        """
        n_docs = len(posts)
        
        if n_docs != len(labels_list):
            raise ValueError(f"Length mismatch: {n_docs} posts vs {len(labels_list)} labels")
        
        # If resuming, only process docs where post_chunked is NULL, unless
        # indices_to_recompute is explicitly provided.
        existing_col = table["post_chunked"] if "post_chunked" in table.column_names else None

        if indices_to_recompute is not None:
            indices_to_process = sorted(set(int(i) for i in indices_to_recompute))
            logger.info(
                f"ChunkingArtifactWriter: Recomputing {len(indices_to_process)} documents (forced)"
            )
        elif existing_col is not None:
            missing_indices: List[int] = []
            if isinstance(existing_col, pa.ChunkedArray):
                offset = 0
                for chunk in existing_col.iterchunks():
                    mask = chunk.is_null().to_numpy(zero_copy_only=False)
                    local_missing = np.flatnonzero(mask).astype(int)
                    missing_indices.extend((local_missing + offset).tolist())
                    offset += len(chunk)
            else:
                mask = existing_col.is_null().to_numpy(zero_copy_only=False)
                missing_indices = np.flatnonzero(mask).astype(int).tolist()

            if len(missing_indices) == 0:
                logger.info(
                    "ChunkingArtifactWriter: post_chunked already fully populated; skipping Stage 1"
                )
                return table

            indices_to_process = missing_indices
            logger.info(
                f"ChunkingArtifactWriter: Resume detected; processing {len(indices_to_process)} missing documents"
            )
        else:
            logger.info(f"ChunkingArtifactWriter: Processing {n_docs} documents")
            indices_to_process = list(range(n_docs))
        posts_to_process = [posts[i] for i in indices_to_process]
        labels_to_process = [labels_list[i] for i in indices_to_process]

        if self.num_workers <= 1 or len(indices_to_process) < 1000:
            chunk_dicts = self._compute_chunk_dicts_sequential(posts_to_process, labels_to_process)
        else:
            chunk_dicts = self._compute_chunk_dicts_parallel(indices_to_process, posts_to_process, labels_to_process)

        # No existing column: attach directly
        if existing_col is None:
            return self._attach_chunks_to_table(table, chunk_dicts)

        # Existing column: replace / fill specific indices, preserving chunking
        merged_col = self._merge_into_existing_column(
            existing_col=existing_col,
            indices_to_process=indices_to_process,
            computed_chunk_dicts=chunk_dicts,
        )

        col_idx = table.column_names.index("post_chunked")
        return table.set_column(col_idx, POST_CHUNKED_FIELD, merged_col)
    
    def _compute_chunk_dicts_sequential(
        self,
        posts: List[str],
        labels_list: List[Optional[List[str]]],
    ) -> List[List[Dict[str, Any]]]:
        """Sequential chunking. Returns list-of-chunk-dicts per document."""
        from ..pollution_guard.semantic_chunker import SemanticChunker, BudgetConfig
        from transformers import AutoTokenizer
        
        logger.info("Using sequential chunking (single-process)")
        
        # Initialize chunker
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                use_fast=True,
                local_files_only=True,
                legacy=True,
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                use_fast=True,
                legacy=True,
            )
        
        # Reconstruct words_splitter
        words_splitter = None
        if self.words_splitter_type:
            try:
                from gliner.data_processing.tokenizer import WordsSplitter
                words_splitter = WordsSplitter(splitter_type=self.words_splitter_type)
            except Exception:
                pass
        
        budget_config = BudgetConfig(**self._budget_config_dict)
        chunker = SemanticChunker(
            tokenizer=tokenizer,
            config=budget_config,
            language=self.language,
            words_splitter=words_splitter,
            words_splitter_type=self.words_splitter_type,
        )
        
        all_chunk_dicts: List[List[Dict[str, Any]]] = []

        for i, (text, labels) in enumerate(zip(posts, labels_list)):
            if not text or not text.strip() or not labels:
                all_chunk_dicts.append([])
            else:
                chunks = chunker.chunk_text(text, labels)
                all_chunk_dicts.append([c.to_arrow_struct() for c in chunks])

            if self.progress_callback:
                # Per-document progress (requested behavior)
                self.progress_callback(i + 1)

        return all_chunk_dicts
    
    def _compute_chunk_dicts_parallel(
        self,
        original_indices: List[int],
        posts: List[str],
        labels_list: List[Optional[List[str]]],
    ) -> List[List[Dict[str, Any]]]:
        """Parallel chunking with per-document progress updates."""
        n_docs = len(posts)
        max_workers = min(self.num_workers, os.cpu_count() or 1)
        
        logger.info(f"Using parallel chunking: {max_workers} workers")

        # Pre-allocate results in input order (which corresponds to original_indices order)
        results_in_order: List[Optional[List[Dict[str, Any]]]] = [None] * n_docs
        processed_count = 0

        index_to_pos = {orig_idx: pos for pos, orig_idx in enumerate(original_indices)}
        
        ctx = mp.get_context("spawn")
        
        try:
            with ctx.Pool(
                processes=max_workers,
                initializer=_worker_init,
                initargs=(
                    self.tokenizer_name,
                    self._budget_config_dict,
                    self.language,
                    self.words_splitter_type,
                ),
            ) as pool:
                tasks = zip(original_indices, posts, labels_list)

                # Per-document completion updates via imap_unordered
                for orig_idx, chunk_dicts in pool.imap_unordered(
                    _worker_chunk_one,
                    tasks,
                    chunksize=64,
                ):
                    pos = index_to_pos.get(orig_idx)
                    if pos is None:
                        continue
                    results_in_order[pos] = chunk_dicts

                    processed_count += 1
                    if self.progress_callback:
                        self.progress_callback(processed_count)
        
        except Exception as exc:
            logger.error(f"Parallel chunking failed: {exc}")
            logger.info("Falling back to sequential chunking")
            return self._compute_chunk_dicts_sequential(posts, labels_list)

        # Convert None to empty lists for any missed documents
        return [c if c is not None else [] for c in results_in_order]

    @staticmethod
    def _merge_into_existing_column(
        *,
        existing_col: pa.Array,
        indices_to_process: List[int],
        computed_chunk_dicts: List[List[Dict[str, Any]]],
    ) -> pa.Array:
        """Merge computed chunk dicts into an existing post_chunked column.

        This avoids combine_chunks()/concat_arrays() over huge string buffers by
        operating chunk-by-chunk.
        """
        if len(indices_to_process) != len(computed_chunk_dicts):
            raise ValueError(
                f"Index/result length mismatch: {len(indices_to_process)} vs {len(computed_chunk_dicts)}"
            )

        if not indices_to_process:
            return existing_col

        idx_to_value = {int(i): v for i, v in zip(indices_to_process, computed_chunk_dicts)}

        # If the existing column is not chunked, rebuild it once.
        if not isinstance(existing_col, pa.ChunkedArray):
            values = existing_col.to_pylist()
            for i, v in idx_to_value.items():
                values[i] = v
            return pa.array(values, type=pa.list_(CHUNK_STRUCT))

        # ChunkedArray path: rebuild each chunk independently.
        new_chunks: List[pa.Array] = []
        offset = 0
        for chunk in existing_col.iterchunks():
            chunk_len = len(chunk)
            chunk_values = chunk.to_pylist()
            for local in range(chunk_len):
                global_idx = offset + local
                if global_idx in idx_to_value:
                    chunk_values[local] = idx_to_value[global_idx]
            new_chunks.append(pa.array(chunk_values, type=pa.list_(CHUNK_STRUCT)))
            offset += chunk_len

        return pa.chunked_array(new_chunks, type=pa.list_(CHUNK_STRUCT))
    
    def _attach_chunks_to_table(
        self,
        table: pa.Table,
        all_chunk_dicts: List[List[Dict[str, Any]]],
    ) -> pa.Table:
        """
        Attach post_chunked column to table.
        
        Args:
            table: Original table.
            all_chunk_dicts: List of chunk dicts per document.
            
        Returns:
            Table with post_chunked column appended.
        """
        # Convert list of dicts to Arrow ListArray of structs
        # Each element is a list of chunk structs
        chunk_arrays = []
        
        for doc_chunks in all_chunk_dicts:
            if not doc_chunks:
                # Empty list for documents with no chunks
                chunk_arrays.append([])
            else:
                chunk_arrays.append(doc_chunks)
        
        # Build the Arrow column
        # PyArrow can infer nested list<struct> from Python dicts
        try:
            post_chunked_array = pa.array(
                chunk_arrays,
                type=pa.list_(CHUNK_STRUCT),
            )
        except Exception as exc:
            logger.warning(f"Failed to create typed array: {exc}. Trying untyped.")
            post_chunked_array = pa.array(chunk_arrays)
        
        # Append column to table
        if "post_chunked" in table.column_names:
            # Replace existing column
            col_idx = table.column_names.index("post_chunked")
            table = table.set_column(col_idx, POST_CHUNKED_FIELD, post_chunked_array)
        else:
            # Append new column
            table = table.append_column(POST_CHUNKED_FIELD, post_chunked_array)
        
        logger.info(f"Attached post_chunked column with {len(all_chunk_dicts)} entries")
        
        return table


def load_chunked_table(path: Path) -> pa.Table:
    """Load a table with post_chunked column from Arrow file."""
    import pyarrow.feather as feather
    return feather.read_table(path, memory_map=True)


def save_chunked_table(table: pa.Table, path: Path) -> None:
    """Save a table with post_chunked column to Arrow file."""
    import pyarrow.feather as feather
    feather.write_feather(table, path)
    logger.info(f"Saved chunked table to {path}")


def save_chunked_table_atomic(table: pa.Table, path: Path) -> None:
    """Atomically save a table (including post_chunked) to an Arrow/Feather file.

    Writes to a temporary file in the same directory and replaces the target.
    This ensures Stage 1 results are durably persisted and recoverable after crashes.
    """
    import pyarrow.feather as feather

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        feather.write_feather(table, tmp_path)
        os.replace(tmp_path, target)
        logger.info(f"Atomically saved chunked table to {target}")
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            # Best-effort cleanup
            pass
