"""
Async Result Storer for Scatter-Store-Gather Pattern.

Implements asynchronous, disk-backed storage for GLiNER inference results
to decouple GPU inference from result reconstruction. This enables:
- Constant memory footprint regardless of dataset size
- Crash recovery via persistent Arrow IPC storage
- GPU pipeline saturation by eliminating CPU-side blocking

Architecture:
- Producer (main thread): Submits GPU output tensors to queue
- Consumer (background thread): Processes and writes to Arrow IPC
- Crash Recovery: Checks existing IPC file, skips completed chunks

Reference: Technical Report Section 3 (The "Scatter-Store-Gather" Pattern)
"""

from __future__ import annotations

import logging
import queue
import threading
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema Definitions
# ---------------------------------------------------------------------------

# Entity struct schema for nested storage
ENTITY_STRUCT = pa.struct([
    ("text", pa.string()),
    ("label", pa.string()),
    ("score", pa.float32()),
    ("start", pa.int32()),
    ("end", pa.int32()),
])

# Chunk result schema for IPC storage
CHUNK_RESULT_SCHEMA = pa.schema([
    ("chunk_id", pa.int32()),
    ("entities", pa.list_(ENTITY_STRUCT)),
])


@dataclass
class AsyncStorerConfig:
    """
    Configuration for AsyncResultStorer.
    
    Attributes:
        queue_size: Max items in producer->consumer queue (backpressure control).
        flush_every_n: Flush to disk every N batches.
        timeout_seconds: Queue get timeout before checking done flag.
    """
    queue_size: int = 16
    flush_every_n: int = 10
    timeout_seconds: float = 0.5


@dataclass
class StorerStats:
    """Statistics from AsyncResultStorer operation."""
    chunks_written: int = 0
    batches_processed: int = 0
    chunks_recovered: int = 0
    errors: int = 0


class AsyncResultStorer:
    """
    Async disk-backed storage for inference results.
    
    Implements the "Scatter-Store-Gather" pattern:
    - Scatter: Results arrive out-of-order (global-sorted chunks)
    - Store: Persisted to Arrow IPC keyed by chunk_id
    - Gather: Reconstruction reads by doc_offsets ranges
    
    Thread Safety:
    - submit() is thread-safe (main thread)
    - Consumer runs in dedicated background thread
    - Uses queue for producer-consumer communication
    
    Crash Recovery:
    - On init, checks for existing IPC file
    - Extracts already-completed chunk_ids
    - Returns set of completed chunks to skip during inference
    
    Usage:
        storer = AsyncResultStorer(output_path, num_chunks, config)
        completed = storer.get_completed_chunk_ids()  # For crash recovery
        storer.start()
        
        for batch_indices, entities in inference_loop:
            if not all(idx in completed for idx in batch_indices):
                storer.submit(batch_indices, entities)
        
        storer.finish()
        final_results = storer.load_results()
    """
    
    def __init__(
        self,
        output_path: Path,
        num_chunks: int,
        config: Optional[AsyncStorerConfig] = None,
    ):
        """
        Initialize async result storer.
        
        Args:
            output_path: Path to Arrow IPC file for storing results.
            num_chunks: Total number of chunks expected.
            config: Storage configuration.
        """
        self._path = Path(output_path)
        self._num_chunks = num_chunks
        self._config = config or AsyncStorerConfig()
        
        # Producer-consumer queue
        self._queue: queue.Queue[Optional[Tuple[List[int], List[List[Dict[str, Any]]]]]] = queue.Queue(
            maxsize=self._config.queue_size
        )
        
        # Consumer thread
        self._consumer_thread: Optional[threading.Thread] = None
        self._done = threading.Event()
        self._error: Optional[Exception] = None
        
        # Statistics
        self._stats = StorerStats()
        self._lock = threading.Lock()
        
        # Writer (initialized on start)
        self._writer: Optional[ipc.RecordBatchFileWriter] = None
        self._temp_path: Optional[Path] = None
        
        # Completed chunks from crash recovery
        self._completed_chunk_ids: Set[int] = set()
        
        # Check for existing results (crash recovery)
        self._check_existing_results()
        
        logger.info(
            f"AsyncResultStorer initialized: path={self._path}, "
            f"num_chunks={num_chunks}, recovered={len(self._completed_chunk_ids)}"
        )
    
    def _check_existing_results(self) -> None:
        """Check for existing IPC file and extract completed chunk_ids."""
        if not self._path.exists():
            logger.debug(f"No existing results at {self._path}")
            return
        
        try:
            with pa.memory_map(str(self._path), 'r') as source:
                reader = ipc.open_file(source)
                table = reader.read_all()
            
            if "chunk_id" in table.column_names:
                chunk_ids = table["chunk_id"].to_pylist()
                self._completed_chunk_ids = set(chunk_ids)
                self._stats.chunks_recovered = len(self._completed_chunk_ids)
                
                logger.info(
                    f"Crash recovery: found {len(self._completed_chunk_ids)} "
                    f"completed chunks in {self._path}"
                )
        except Exception as e:
            logger.warning(f"Failed to read existing results for recovery: {e}")
            self._completed_chunk_ids = set()
    
    def get_completed_chunk_ids(self) -> Set[int]:
        """
        Get set of chunk_ids already completed (for crash recovery).
        
        Returns:
            Set of chunk indices that have already been processed.
        """
        return self._completed_chunk_ids.copy()
    
    def get_pending_chunk_ids(self) -> Set[int]:
        """
        Get set of chunk_ids that still need processing.
        
        Returns:
            Set of chunk indices not yet in the result store.
        """
        all_ids = set(range(self._num_chunks))
        return all_ids - self._completed_chunk_ids
    
    @property
    def num_completed(self) -> int:
        """Number of chunks already completed."""
        return len(self._completed_chunk_ids)
    
    @property
    def num_pending(self) -> int:
        """Number of chunks still pending."""
        return self._num_chunks - len(self._completed_chunk_ids)
    
    @property
    def is_complete(self) -> bool:
        """Check if all chunks have been processed."""
        return len(self._completed_chunk_ids) >= self._num_chunks
    
    def start(self) -> None:
        """
        Start the background consumer thread.
        
        Creates a temporary file for writing, which is atomically
        renamed on finish() to ensure crash safety.
        """
        if self._consumer_thread is not None and self._consumer_thread.is_alive():
            logger.warning("Consumer thread already running")
            return
        
        # Create temp file for atomic writes
        self._temp_path = self._path.with_suffix(".tmp")
        
        # If we're appending to existing results, we need to copy them first
        existing_batches: List[pa.RecordBatch] = []
        if self._completed_chunk_ids and self._path.exists():
            try:
                with pa.memory_map(str(self._path), 'r') as source:
                    reader = ipc.open_file(source)
                    for i in range(reader.num_record_batches):
                        existing_batches.append(reader.get_batch(i))
                logger.debug(f"Loaded {len(existing_batches)} existing batches for append")
            except Exception as e:
                logger.warning(f"Failed to load existing batches: {e}")
                existing_batches = []
        
        # Open writer
        self._writer = ipc.new_file(str(self._temp_path), CHUNK_RESULT_SCHEMA)
        
        # Write existing batches first
        for batch in existing_batches:
            self._writer.write_batch(batch)
        
        # Reset done flag
        self._done.clear()
        self._error = None
        
        # Start consumer thread
        self._consumer_thread = threading.Thread(
            target=self._consume_loop,
            name="AsyncResultStorer-Consumer",
            daemon=True,
        )
        self._consumer_thread.start()
        
        logger.debug("Consumer thread started")
    
    def submit(
        self,
        batch_indices: List[int],
        batch_entities: List[List[Dict[str, Any]]],
    ) -> None:
        """
        Submit a batch of results for async storage.
        
        Non-blocking unless queue is full (backpressure).
        
        Args:
            batch_indices: List of chunk indices (global flat indices).
            batch_entities: List of entity lists, one per chunk.
        """
        if self._error is not None:
            raise RuntimeError(f"Consumer thread failed: {self._error}")
        
        if len(batch_indices) != len(batch_entities):
            raise ValueError(
                f"Mismatch: {len(batch_indices)} indices vs {len(batch_entities)} entity lists"
            )
        
        # Put on queue (may block if full - backpressure)
        self._queue.put((batch_indices, batch_entities))
    
    def _consume_loop(self) -> None:
        """Background consumer loop - processes queue and writes to disk."""
        batches_since_flush = 0
        
        try:
            while not self._done.is_set() or not self._queue.empty():
                try:
                    item = self._queue.get(timeout=self._config.timeout_seconds)
                except queue.Empty:
                    continue
                
                if item is None:  # Poison pill
                    break
                
                batch_indices, batch_entities = item
                
                # Convert to Arrow record batch
                records = []
                for chunk_id, entities in zip(batch_indices, batch_entities):
                    # Convert entities to Arrow-compatible format
                    arrow_entities = [
                        {
                            "text": str(e.get("text", "")),
                            "label": str(e.get("label", "")),
                            "score": float(e.get("score", 0.0)),
                            "start": int(e.get("start", 0)),
                            "end": int(e.get("end", 0)),
                        }
                        for e in entities
                    ]
                    records.append({
                        "chunk_id": int(chunk_id),
                        "entities": arrow_entities,
                    })
                
                # Write batch
                batch = pa.RecordBatch.from_pylist(records, schema=CHUNK_RESULT_SCHEMA)
                self._writer.write_batch(batch)
                
                # Update stats
                with self._lock:
                    self._stats.chunks_written += len(batch_indices)
                    self._stats.batches_processed += 1
                    self._completed_chunk_ids.update(batch_indices)
                
                batches_since_flush += 1
                
                # Periodic flush (no explicit flush needed for IPC, but good for progress)
                if batches_since_flush >= self._config.flush_every_n:
                    batches_since_flush = 0
                
        except Exception as e:
            logger.error(f"Consumer thread error: {e}")
            self._error = e
            with self._lock:
                self._stats.errors += 1
    
    def finish(self) -> Path:
        """
        Signal completion and wait for consumer to finish.
        
        Atomically renames temp file to final path.
        
        Returns:
            Path to the final Arrow IPC file.
        """
        # Signal done
        self._done.set()
        
        # Send poison pill to unblock queue.get()
        try:
            self._queue.put(None, timeout=1.0)
        except queue.Full:
            pass
        
        # Wait for consumer thread
        if self._consumer_thread is not None:
            self._consumer_thread.join(timeout=30.0)
            if self._consumer_thread.is_alive():
                logger.warning("Consumer thread did not terminate cleanly")
        
        # Close writer
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        
        # Atomic rename
        if self._temp_path is not None and self._temp_path.exists():
            self._temp_path.replace(self._path)
            logger.info(f"Finalized results to {self._path}")
        
        # Check for errors
        if self._error is not None:
            raise RuntimeError(f"Consumer thread failed: {self._error}")
        
        return self._path
    
    def get_stats(self) -> StorerStats:
        """Get current statistics."""
        with self._lock:
            return StorerStats(
                chunks_written=self._stats.chunks_written,
                batches_processed=self._stats.batches_processed,
                chunks_recovered=self._stats.chunks_recovered,
                errors=self._stats.errors,
            )
    
    def load_results(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        Load all results from the IPC file.
        
        Returns:
            Dict mapping chunk_id -> list of entities.
        """
        if not self._path.exists():
            return {}
        
        try:
            with pa.memory_map(str(self._path), 'r') as source:
                reader = ipc.open_file(source)
                table = reader.read_all()
            
            result: Dict[int, List[Dict[str, Any]]] = {}
            
            chunk_ids = table["chunk_id"].to_pylist()
            entities_col = table["entities"].to_pylist()
            
            for chunk_id, entities in zip(chunk_ids, entities_col):
                # Convert back to dict format
                entity_dicts = [
                    {
                        "text": e["text"],
                        "label": e["label"],
                        "score": e["score"],
                        "start": e["start"],
                        "end": e["end"],
                    }
                    for e in (entities or [])
                ]
                result[int(chunk_id)] = entity_dicts
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            return {}
    
    def load_results_as_flat_list(self, num_chunks: int) -> List[List[Dict[str, Any]]]:
        """
        Load results as a flat list indexed by chunk_id.
        
        Args:
            num_chunks: Total number of chunks (for pre-allocation).
            
        Returns:
            List where result[chunk_id] = entities for that chunk.
        """
        results_dict = self.load_results()
        
        flat_results: List[List[Dict[str, Any]]] = [[] for _ in range(num_chunks)]
        
        for chunk_id, entities in results_dict.items():
            if 0 <= chunk_id < num_chunks:
                flat_results[chunk_id] = entities
        
        return flat_results


def create_async_storer_from_config(
    output_path: Path,
    num_chunks: int,
    config: Optional[Dict[str, Any]] = None,
) -> AsyncResultStorer:
    """
    Create AsyncResultStorer from pipeline configuration.
    
    Args:
        output_path: Path to Arrow IPC file.
        num_chunks: Total number of chunks.
        config: Pipeline config dict (reads execution.async_storage section).
        
    Returns:
        Configured AsyncResultStorer instance.
    """
    storer_config = AsyncStorerConfig()
    
    if config is not None:
        async_cfg = config.get("execution", {}).get("async_storage", {})
        storer_config = AsyncStorerConfig(
            queue_size=int(async_cfg.get("queue_size", 16)),
            flush_every_n=int(async_cfg.get("flush_every_n", 10)),
            timeout_seconds=float(async_cfg.get("timeout_seconds", 0.5)),
        )
    
    return AsyncResultStorer(output_path, num_chunks, storer_config)
