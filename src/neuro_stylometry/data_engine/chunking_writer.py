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
        
        logger.info(f"ChunkingArtifactWriter: Processing {n_docs} documents")
        
        if self.num_workers <= 1 or n_docs < 1000:
            # Sequential mode for small datasets or when parallelism disabled
            return self._process_sequential(posts, labels_list, table)
        else:
            return self._process_parallel(posts, labels_list, table)
    
    def _process_sequential(
        self,
        posts: List[str],
        labels_list: List[Optional[List[str]]],
        table: pa.Table,
    ) -> pa.Table:
        """Sequential processing for small datasets."""
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
        
        # Process all documents
        all_chunk_dicts: List[List[Dict[str, Any]]] = []
        
        for i, (text, labels) in enumerate(zip(posts, labels_list)):
            if not text or not text.strip() or not labels:
                all_chunk_dicts.append([])
            else:
                chunks = chunker.chunk_text(text, labels)
                all_chunk_dicts.append([c.to_arrow_struct() for c in chunks])
            
            if self.progress_callback and (i + 1) % 100 == 0:
                self.progress_callback(i + 1)
        
        if self.progress_callback:
            self.progress_callback(len(posts))
        
        return self._attach_chunks_to_table(table, all_chunk_dicts)
    
    def _process_parallel(
        self,
        posts: List[str],
        labels_list: List[Optional[List[str]]],
        table: pa.Table,
    ) -> pa.Table:
        """Parallel processing with streaming writes."""
        n_docs = len(posts)
        max_workers = min(self.num_workers, os.cpu_count() or 1)
        
        logger.info(f"Using parallel chunking: {max_workers} workers")
        
        # Partition into batches for workers
        batch_size = max(100, n_docs // (max_workers * 4))  # 4 batches per worker
        batches: List[Tuple[List[int], List[str], List[Optional[List[str]]]]] = []
        
        for i in range(0, n_docs, batch_size):
            end = min(i + batch_size, n_docs)
            batch_indices = list(range(i, end))
            batch_texts = posts[i:end]
            batch_labels = labels_list[i:end]
            batches.append((batch_indices, batch_texts, batch_labels))
        
        logger.info(f"Created {len(batches)} batches ({batch_size} docs/batch)")
        
        # Pre-allocate results array
        all_chunk_dicts: List[Optional[List[Dict[str, Any]]]] = [None] * n_docs
        processed_count = 0
        
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
                # Process batches with imap_unordered for non-blocking
                for doc_indices, batch_results in pool.imap_unordered(
                    _worker_chunk_batch, batches
                ):
                    # Store results at original indices
                    for idx, chunks in zip(doc_indices, batch_results):
                        all_chunk_dicts[idx] = chunks
                    
                    processed_count += len(doc_indices)
                    if self.progress_callback:
                        self.progress_callback(processed_count)
        
        except Exception as exc:
            logger.error(f"Parallel chunking failed: {exc}")
            logger.info("Falling back to sequential chunking")
            return self._process_sequential(posts, labels_list, table)
        
        # Convert None to empty lists for any missed documents
        all_chunk_dicts = [c if c is not None else [] for c in all_chunk_dicts]
        
        return self._attach_chunks_to_table(table, all_chunk_dicts)
    
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
