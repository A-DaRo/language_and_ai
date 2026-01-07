"""
Asynchronous Prefetching for Pipeline Parallelism.

Implements a Producer-Consumer pattern to decouple CPU-bound chunking from
GPU-bound inference, enabling pipeline parallelism.

Architecture:
- Producer Process (CPU): Runs SemanticChunker, tokenizes text, pushes to queue
- Consumer Process (GPU): Pulls batches from queue, executes inference

This pipelines the workload: while the GPU processes Batch N, the CPU prepares
Batch N+1, maximizing hardware utilization.

Configurable based on ProfileType:
- LAPTOP: Single-process mode with async batching (lighter overhead)
- HPC: Multi-process with dedicated chunker worker (higher throughput)

Reference: Performance Optimization Report - Asynchronous Prefetching Section
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
from queue import Empty, Full
import threading
import time
from enum import Enum

import torch

logger = logging.getLogger(__name__)


# Import ProfileType with fallback for testing
try:
    from .detection import ProfileType, HardwareDetector
except ImportError:
    class ProfileType(str, Enum):
        HPC = "HPC"
        LAPTOP = "LAPTOP"
    HardwareDetector = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AsyncPrefetchConfig:
    """
    Configuration for asynchronous prefetching.
    
    Attributes:
        enabled: Master switch for async prefetching.
        queue_size: Maximum items in prefetch queue.
        num_workers: Number of prefetch worker processes.
        timeout_seconds: Queue operation timeout.
        prefetch_factor: Items to prefetch per worker.
        use_multiprocessing: Use multiprocessing (vs threading).
        pin_memory: Pin tensors to memory for faster GPU transfer.
        profile_aware: Adjust settings based on ProfileType.
    """
    enabled: bool = False
    queue_size: int = 4
    num_workers: int = 1
    timeout_seconds: float = 30.0
    prefetch_factor: int = 2
    use_multiprocessing: bool = True
    pin_memory: bool = True
    profile_aware: bool = True


@dataclass
class PrefetchBatch:
    """
    A prefetched batch ready for inference.
    
    Attributes:
        batch_id: Unique identifier for this batch.
        texts: List of chunk texts.
        labels: Inference labels (same for all texts in batch).
        doc_indices: Document indices for result mapping.
        chunk_infos: Original ChunkInfo objects for offset projection.
        tokenized: Optional pre-tokenized tensors (input_ids, attention_mask).
        metadata: Additional batch metadata.
    """
    batch_id: int
    texts: List[str]
    labels: List[str]
    doc_indices: List[int]
    chunk_infos: List[Any]  # ChunkInfo objects
    tokenized: Optional[Dict[str, torch.Tensor]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class StopSignal:
    """Sentinel value to signal worker shutdown."""
    pass


# ---------------------------------------------------------------------------
# Producer (Chunker) Worker
# ---------------------------------------------------------------------------


def chunker_worker(
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    chunker_factory: Callable[[], Any],
    taxonomy_factory: Callable[[], Any],
    batch_size: int,
    config: AsyncPrefetchConfig,
):
    """
    Worker process that chunks texts and prepares batches.
    
    Runs in a separate process (or thread), reading raw texts from input_queue,
    applying semantic chunking, and pushing prepared batches to output_queue.
    
    Args:
        input_queue: Queue of (doc_idx, text, active_columns) tuples.
        output_queue: Queue for PrefetchBatch objects.
        chunker_factory: Factory function to create SemanticChunker.
        taxonomy_factory: Factory function to create SOBRTaxonomy.
        batch_size: Target batch size.
        config: Prefetch configuration.
    """
    logger.info("Chunker worker started")
    
    try:
        # Initialize components (must be done in worker process)
        chunker = chunker_factory()
        taxonomy = taxonomy_factory()
        
        # Batch accumulation buffer
        batch_buffer: List[Tuple[int, Any, List[str]]] = []  # (doc_idx, chunk_info, labels)
        batch_id = 0
        
        while True:
            try:
                item = input_queue.get(timeout=config.timeout_seconds)
                
                # Check for stop signal
                if isinstance(item, StopSignal):
                    # Flush remaining buffer
                    if batch_buffer:
                        _flush_batch(output_queue, batch_buffer, batch_id, config)
                    output_queue.put(StopSignal())
                    break
                
                doc_idx, text, active_columns = item
                
                # Determine labels for this document
                if active_columns:
                    labels = taxonomy.get_prompts_for_columns(active_columns)
                else:
                    labels = taxonomy.get_inference_labels()
                
                if not labels or not text or not text.strip():
                    continue
                
                # Chunk the text
                chunks = chunker.chunk_text(text, labels)
                
                for chunk_info in chunks:
                    batch_buffer.append((doc_idx, chunk_info, labels))
                    
                    # Flush when batch is full
                    if len(batch_buffer) >= batch_size:
                        _flush_batch(output_queue, batch_buffer, batch_id, config)
                        batch_buffer = []
                        batch_id += 1
                
            except Empty:
                # Timeout - flush partial batch
                if batch_buffer:
                    _flush_batch(output_queue, batch_buffer, batch_id, config)
                    batch_buffer = []
                    batch_id += 1
                continue
                
    except Exception as e:
        logger.error(f"Chunker worker error: {e}")
        output_queue.put(StopSignal())
        raise
    
    logger.info("Chunker worker stopped")


def _flush_batch(
    output_queue: mp.Queue,
    buffer: List[Tuple[int, Any, List[str]]],
    batch_id: int,
    config: AsyncPrefetchConfig,
):
    """Flush accumulated chunks to output queue as a PrefetchBatch."""
    if not buffer:
        return
    
    # Group by label set for efficient batching
    # For simplicity, use first item's labels (assumes homogeneous batches)
    texts = [item[1].text for item in buffer]
    labels = buffer[0][2]
    doc_indices = [item[0] for item in buffer]
    chunk_infos = [item[1] for item in buffer]
    
    batch = PrefetchBatch(
        batch_id=batch_id,
        texts=texts,
        labels=labels,
        doc_indices=doc_indices,
        chunk_infos=chunk_infos,
    )
    
    try:
        output_queue.put(batch, timeout=config.timeout_seconds)
    except Full:
        logger.warning(f"Output queue full, dropping batch {batch_id}")


# ---------------------------------------------------------------------------
# Async Prefetch Pipeline
# ---------------------------------------------------------------------------


class AsyncPrefetchPipeline:
    """
    Asynchronous prefetch pipeline for GLiNER detection.
    
    Implements a producer-consumer pattern:
    - Producer: Chunks texts and prepares batches (CPU-bound)
    - Consumer: Main thread pulls batches and runs inference (GPU-bound)
    
    This decouples the CPU chunking work from GPU inference, allowing
    the CPU to prepare the next batch while the GPU processes the current one.
    
    Usage:
        pipeline = AsyncPrefetchPipeline(config, chunker_factory, taxonomy_factory)
        pipeline.start()
        
        # Submit texts for processing
        for idx, text in enumerate(texts):
            pipeline.submit(idx, text, active_columns[idx])
        
        # Signal completion and get results
        pipeline.finish_submission()
        
        for batch in pipeline.iter_batches():
            results = model.inference(batch.texts, batch.labels)
            # Process results...
        
        pipeline.stop()
    """
    
    def __init__(
        self,
        config: Optional[AsyncPrefetchConfig] = None,
        chunker_factory: Optional[Callable[[], Any]] = None,
        taxonomy_factory: Optional[Callable[[], Any]] = None,
        batch_size: int = 32,
        profile: Optional[ProfileType] = None,
    ):
        """
        Initialize async prefetch pipeline.
        
        Args:
            config: Prefetch configuration.
            chunker_factory: Factory to create SemanticChunker in worker.
            taxonomy_factory: Factory to create SOBRTaxonomy in worker.
            batch_size: Target batch size for inference.
            profile: Hardware profile (auto-detect if None).
        """
        self.config = config or AsyncPrefetchConfig()
        self.chunker_factory = chunker_factory
        self.taxonomy_factory = taxonomy_factory
        self.batch_size = batch_size
        
        # Detect or use provided profile
        if profile is None and HardwareDetector is not None:
            profile = HardwareDetector.detect().profile_type
        self.profile = profile or ProfileType.LAPTOP
        
        # Apply profile-specific settings
        if self.config.profile_aware:
            self._apply_profile_settings()
        
        # Queue and worker state
        self._input_queue: Optional[mp.Queue] = None
        self._output_queue: Optional[mp.Queue] = None
        self._worker: Optional[Union[mp.Process, threading.Thread]] = None
        self._running = False
        
        # Statistics
        self._batches_produced = 0
        self._batches_consumed = 0
        self._submission_finished = False
    
    def _apply_profile_settings(self):
        """Adjust config based on hardware profile."""
        if self.profile == ProfileType.HPC:
            # HPC: More aggressive prefetching
            self.config.queue_size = max(self.config.queue_size, 8)
            self.config.num_workers = max(self.config.num_workers, 2)
            self.config.use_multiprocessing = True
            logger.debug("Applied HPC profile settings for async prefetch")
        else:
            # LAPTOP: Conservative settings
            self.config.queue_size = min(self.config.queue_size, 4)
            self.config.num_workers = 1
            # Use threading on laptop to avoid multiprocessing overhead
            self.config.use_multiprocessing = False
            logger.debug("Applied LAPTOP profile settings for async prefetch")
    
    def start(self):
        """Start the prefetch pipeline."""
        if self._running:
            logger.warning("Pipeline already running")
            return
        
        if not self.config.enabled:
            logger.info("Async prefetch disabled, running in synchronous mode")
            return
        
        if self.chunker_factory is None or self.taxonomy_factory is None:
            raise ValueError("chunker_factory and taxonomy_factory required")
        
        # Create queues
        if self.config.use_multiprocessing:
            self._input_queue = mp.Queue(maxsize=self.config.queue_size * 2)
            self._output_queue = mp.Queue(maxsize=self.config.queue_size)
            
            self._worker = mp.Process(
                target=chunker_worker,
                args=(
                    self._input_queue,
                    self._output_queue,
                    self.chunker_factory,
                    self.taxonomy_factory,
                    self.batch_size,
                    self.config,
                ),
                daemon=True,
            )
        else:
            # Use threading (lighter overhead for laptop)
            import queue
            self._input_queue = queue.Queue(maxsize=self.config.queue_size * 2)
            self._output_queue = queue.Queue(maxsize=self.config.queue_size)
            
            self._worker = threading.Thread(
                target=chunker_worker,
                args=(
                    self._input_queue,
                    self._output_queue,
                    self.chunker_factory,
                    self.taxonomy_factory,
                    self.batch_size,
                    self.config,
                ),
                daemon=True,
            )
        
        self._worker.start()
        self._running = True
        self._submission_finished = False
        
        logger.info(
            f"Async prefetch started: queue_size={self.config.queue_size}, "
            f"multiprocessing={self.config.use_multiprocessing}"
        )
    
    def submit(
        self,
        doc_idx: int,
        text: str,
        active_columns: Optional[List[str]] = None,
    ):
        """
        Submit a document for processing.
        
        Args:
            doc_idx: Document index for result mapping.
            text: Document text.
            active_columns: Optional list of active SOBR columns.
        """
        if not self._running or self._input_queue is None:
            raise RuntimeError("Pipeline not running. Call start() first.")
        
        if self._submission_finished:
            raise RuntimeError("Cannot submit after finish_submission() called")
        
        self._input_queue.put((doc_idx, text, active_columns))
    
    def finish_submission(self):
        """Signal that all documents have been submitted."""
        if not self._running or self._input_queue is None:
            return
        
        self._input_queue.put(StopSignal())
        self._submission_finished = True
    
    def iter_batches(self) -> Iterator[PrefetchBatch]:
        """
        Iterate over prefetched batches.
        
        Yields:
            PrefetchBatch objects ready for inference.
        """
        if not self._running or self._output_queue is None:
            return
        
        while True:
            try:
                item = self._output_queue.get(timeout=self.config.timeout_seconds)
                
                if isinstance(item, StopSignal):
                    break
                
                self._batches_consumed += 1
                yield item
                
            except Empty:
                if self._submission_finished:
                    # All submissions done, check if worker is still alive
                    if self._worker is not None and not self._worker.is_alive():
                        break
                continue
    
    def stop(self):
        """Stop the prefetch pipeline."""
        if not self._running:
            return
        
        # Send stop signal if not already sent
        if not self._submission_finished and self._input_queue is not None:
            try:
                self._input_queue.put(StopSignal(), timeout=1.0)
            except (Full, Exception):
                pass
        
        # Wait for worker to finish
        if self._worker is not None:
            self._worker.join(timeout=self.config.timeout_seconds)
            if self._worker.is_alive():
                if hasattr(self._worker, 'terminate'):
                    self._worker.terminate()
        
        self._running = False
        logger.info("Async prefetch stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "enabled": self.config.enabled,
            "running": self._running,
            "profile": str(self.profile),
            "batches_produced": self._batches_produced,
            "batches_consumed": self._batches_consumed,
            "use_multiprocessing": self.config.use_multiprocessing,
            "queue_size": self.config.queue_size,
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if not self._submission_finished:
            self.finish_submission()
        self.stop()
        return False


# ---------------------------------------------------------------------------
# Synchronous Fallback
# ---------------------------------------------------------------------------


class SyncPrefetchPipeline:
    """
    Synchronous fallback when async prefetch is disabled.
    
    Provides the same interface as AsyncPrefetchPipeline but processes
    everything in the main thread.
    """
    
    def __init__(
        self,
        chunker: Any,
        taxonomy: Any,
        batch_size: int = 32,
    ):
        """
        Initialize synchronous pipeline.
        
        Args:
            chunker: SemanticChunker instance.
            taxonomy: SOBRTaxonomy instance.
            batch_size: Target batch size.
        """
        self.chunker = chunker
        self.taxonomy = taxonomy
        self.batch_size = batch_size
        
        self._buffer: List[Tuple[int, str, Optional[List[str]]]] = []
        self._batches: List[PrefetchBatch] = []
    
    def submit(
        self,
        doc_idx: int,
        text: str,
        active_columns: Optional[List[str]] = None,
    ):
        """Submit a document."""
        self._buffer.append((doc_idx, text, active_columns))
    
    def process_all(self) -> List[PrefetchBatch]:
        """Process all submitted documents and return batches."""
        all_chunks: List[Tuple[int, Any, List[str]]] = []
        
        for doc_idx, text, active_columns in self._buffer:
            if not text or not text.strip():
                continue
            
            if active_columns:
                labels = self.taxonomy.get_prompts_for_columns(active_columns)
            else:
                labels = self.taxonomy.get_inference_labels()
            
            if not labels:
                continue
            
            chunks = self.chunker.chunk_text(text, labels)
            for chunk_info in chunks:
                all_chunks.append((doc_idx, chunk_info, labels))
        
        # Create batches
        batches = []
        batch_id = 0
        
        for i in range(0, len(all_chunks), self.batch_size):
            batch_items = all_chunks[i:i + self.batch_size]
            
            batch = PrefetchBatch(
                batch_id=batch_id,
                texts=[item[1].text for item in batch_items],
                labels=batch_items[0][2] if batch_items else [],
                doc_indices=[item[0] for item in batch_items],
                chunk_infos=[item[1] for item in batch_items],
            )
            batches.append(batch)
            batch_id += 1
        
        self._buffer.clear()
        return batches


# ---------------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------------


def create_prefetch_config_from_dict(
    config: Dict[str, Any],
    profile: Optional[ProfileType] = None,
) -> AsyncPrefetchConfig:
    """
    Create AsyncPrefetchConfig from pipeline configuration.
    
    Args:
        config: Pipeline config dict.
        profile: Hardware profile (optional).
        
    Returns:
        Configured AsyncPrefetchConfig.
    """
    execution_cfg = config.get("execution", {})
    prefetch_cfg = execution_cfg.get("async_prefetch", {})
    
    return AsyncPrefetchConfig(
        enabled=bool(prefetch_cfg.get("enabled", False)),
        queue_size=int(prefetch_cfg.get("queue_size", 4)),
        num_workers=int(prefetch_cfg.get("num_workers", 1)),
        timeout_seconds=float(prefetch_cfg.get("timeout_seconds", 30.0)),
        prefetch_factor=int(prefetch_cfg.get("prefetch_factor", 2)),
        use_multiprocessing=bool(prefetch_cfg.get("use_multiprocessing", True)),
        pin_memory=bool(prefetch_cfg.get("pin_memory", True)),
        profile_aware=bool(prefetch_cfg.get("profile_aware", True)),
    )


def create_prefetch_pipeline(
    config: Dict[str, Any],
    chunker_factory: Optional[Callable[[], Any]] = None,
    taxonomy_factory: Optional[Callable[[], Any]] = None,
    batch_size: int = 32,
    profile: Optional[ProfileType] = None,
) -> Union[AsyncPrefetchPipeline, None]:
    """
    Create prefetch pipeline from configuration.
    
    Returns AsyncPrefetchPipeline if enabled, None otherwise (use sync mode).
    
    Args:
        config: Pipeline config dict.
        chunker_factory: Factory for SemanticChunker.
        taxonomy_factory: Factory for SOBRTaxonomy.
        batch_size: Target batch size.
        profile: Hardware profile.
        
    Returns:
        AsyncPrefetchPipeline or None.
    """
    prefetch_config = create_prefetch_config_from_dict(config, profile)
    
    if not prefetch_config.enabled:
        return None
    
    return AsyncPrefetchPipeline(
        config=prefetch_config,
        chunker_factory=chunker_factory,
        taxonomy_factory=taxonomy_factory,
        batch_size=batch_size,
        profile=profile,
    )
