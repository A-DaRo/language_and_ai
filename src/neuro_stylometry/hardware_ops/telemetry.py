"""
Telemetry - GPU Performance Monitoring and Metrics Collection.

Provides tools for measuring and collecting GPU inference performance:
- @timed_cuda_block decorator for precise CUDA timing.
- TelemetryCollector singleton for aggregating metrics across batches.
- Integration with RuntimeController for feedback-based tuning.

Usage:
    # Decorator for timing functions
    @timed_cuda_block("gliner_inference")
    def run_inference(batch):
        return model(batch)
    
    # Manual timing
    with CUDATimer() as timer:
        result = model(batch)
    print(f"Elapsed: {timer.elapsed_ms:.2f}ms")
    
    # Metrics collection
    collector = TelemetryCollector.get_instance()
    collector.record_batch(
        batch_size=len(batch),
        tokens=total_tokens,
        elapsed_ms=timer.elapsed_ms,
    )
    summary = collector.get_summary()
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar

import torch

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CUDATimer:
    """
    Precise CUDA event-based timer for GPU operations.
    
    Uses CUDA events with synchronization for accurate timing
    of GPU operations, accounting for async execution.
    
    Usage:
        with CUDATimer() as timer:
            result = model(batch)
        print(f"Elapsed: {timer.elapsed_ms:.2f}ms")
    """
    
    def __init__(self, synchronize: bool = False):
        """
        Initialize CUDA timer.
        
        Args:
            synchronize: If True, synchronize CUDA before measuring.
                Set to False for overlapped timing (less accurate).
        """
        self.synchronize = synchronize
        self._start_event: Optional[torch.cuda.Event] = None
        self._end_event: Optional[torch.cuda.Event] = None
        self._elapsed_ms: Optional[float] = None
        self._use_cuda = torch.cuda.is_available()
        self._cpu_start: float = 0.0
        self._cpu_end: float = 0.0
    
    def __enter__(self) -> CUDATimer:
        self._cpu_start = time.perf_counter()
        if self._use_cuda:
            if self.synchronize:
                torch.cuda.synchronize()
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._cpu_end = time.perf_counter()
        if self._use_cuda:
            if self._end_event is not None:
                self._end_event.record()
            try:
                if self.synchronize:
                    torch.cuda.synchronize()
                if self._start_event is None or self._end_event is None:
                    raise RuntimeError("CUDA events not initialized")
                self._elapsed_ms = self._start_event.elapsed_time(self._end_event)
                return
            except Exception as exc:
                logger.debug("CUDA timer fallback to CPU timing: %s", exc)
        self._elapsed_ms = (self._cpu_end - self._cpu_start) * 1000.0
    
    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        if self._elapsed_ms is None:
            raise RuntimeError("Timer not yet completed")
        return self._elapsed_ms


@contextmanager
def cuda_timed_section(name: Optional[str] = None) -> Generator[CUDATimer, None, None]:
    """
    Context manager for timed CUDA sections with optional naming.
    
    Args:
        name: Optional section name for logging.
        
    Yields:
        CUDATimer instance.
    """
    timer = CUDATimer()
    with timer:
        yield timer
    
    if name and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"[{name}] {timer.elapsed_ms:.2f}ms")


def timed_cuda_block(
    name: Optional[str] = None,
    log_level: int = logging.DEBUG,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for timing CUDA-enabled functions.
    
    The decorated function will be timed using CUDA events (if available)
    and the elapsed time logged.
    
    Args:
        name: Name for the timed block (defaults to function name).
        log_level: Logging level for timing output.
        
    Returns:
        Decorator function.
        
    Usage:
        @timed_cuda_block("inference")
        def run_model(batch):
            return model(batch)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        block_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            timer = CUDATimer()
            with timer:
                result = func(*args, **kwargs)
            
            if logger.isEnabledFor(log_level):
                logger.log(log_level, f"[{block_name}] {timer.elapsed_ms:.2f}ms")
            
            return result
        return wrapper
    return decorator


@dataclass
class BatchMetrics:
    """Metrics for a single batch."""
    batch_size: int
    tokens: int
    elapsed_ms: float
    memory_mb: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    @property
    def throughput_tokens_per_sec(self) -> float:
        """Tokens per second throughput."""
        if self.elapsed_ms <= 0:
            return 0.0
        return (self.tokens / self.elapsed_ms) * 1000.0
    
    @property
    def throughput_samples_per_sec(self) -> float:
        """Samples per second throughput."""
        if self.elapsed_ms <= 0:
            return 0.0
        return (self.batch_size / self.elapsed_ms) * 1000.0


@dataclass
class TelemetrySummary:
    """Aggregated telemetry summary."""
    total_batches: int
    total_tokens: int
    total_samples: int
    total_time_ms: float
    avg_batch_time_ms: float
    avg_throughput_tokens_per_sec: float
    avg_throughput_samples_per_sec: float
    peak_memory_mb: float
    avg_memory_mb: float
    min_batch_time_ms: float
    max_batch_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "total_batches": self.total_batches,
            "total_tokens": self.total_tokens,
            "total_samples": self.total_samples,
            "total_time_ms": round(self.total_time_ms, 2),
            "avg_batch_time_ms": round(self.avg_batch_time_ms, 2),
            "avg_throughput_tokens_per_sec": round(self.avg_throughput_tokens_per_sec, 1),
            "avg_throughput_samples_per_sec": round(self.avg_throughput_samples_per_sec, 1),
            "peak_memory_mb": round(self.peak_memory_mb, 1),
            "avg_memory_mb": round(self.avg_memory_mb, 1),
            "min_batch_time_ms": round(self.min_batch_time_ms, 2),
            "max_batch_time_ms": round(self.max_batch_time_ms, 2),
        }


class TelemetryCollector:
    """
    Singleton collector for GPU telemetry metrics.
    
    Thread-safe collection of batch metrics with aggregation.
    Use get_instance() to access the singleton.
    
    Usage:
        collector = TelemetryCollector.get_instance()
        collector.record_batch(batch_size=32, tokens=4096, elapsed_ms=15.5)
        summary = collector.get_summary()
    """
    
    _instance: Optional[TelemetryCollector] = None
    _lock = threading.Lock()
    
    def __init__(self, max_history: int = 10000):
        """
        Initialize telemetry collector.
        
        Args:
            max_history: Maximum number of batch records to keep.
        """
        self._metrics: List[BatchMetrics] = []
        self._max_history = max_history
        self._metrics_lock = threading.Lock()
        self._start_time = time.time()
    
    @classmethod
    def get_instance(cls, max_history: int = 10000) -> TelemetryCollector:
        """Get the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(max_history=max_history)
        return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (useful for testing)."""
        with cls._lock:
            cls._instance = None
    
    def record_batch(
        self,
        batch_size: int,
        tokens: int,
        elapsed_ms: float,
        memory_mb: Optional[float] = None,
    ) -> None:
        """
        Record metrics for a completed batch.
        
        Args:
            batch_size: Number of samples in batch.
            tokens: Total tokens processed.
            elapsed_ms: Wall-clock time for batch (ms).
            memory_mb: GPU memory used (MB). Auto-detected if None.
        """
        if memory_mb is None:
            memory_mb = self._get_current_memory_mb()
        
        metrics = BatchMetrics(
            batch_size=batch_size,
            tokens=tokens,
            elapsed_ms=elapsed_ms,
            memory_mb=memory_mb,
        )
        
        with self._metrics_lock:
            self._metrics.append(metrics)
            if len(self._metrics) > self._max_history:
                # Remove oldest entries
                self._metrics = self._metrics[-self._max_history:]
    
    def _get_current_memory_mb(self) -> float:
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0.0
    
    def get_summary(self, last_n: Optional[int] = None) -> TelemetrySummary:
        """
        Get aggregated summary of collected metrics.
        
        Args:
            last_n: If provided, only summarize the last N batches.
            
        Returns:
            TelemetrySummary with aggregated statistics.
        """
        with self._metrics_lock:
            metrics = self._metrics.copy()
        
        if last_n is not None:
            metrics = metrics[-last_n:]
        
        if not metrics:
            return TelemetrySummary(
                total_batches=0,
                total_tokens=0,
                total_samples=0,
                total_time_ms=0.0,
                avg_batch_time_ms=0.0,
                avg_throughput_tokens_per_sec=0.0,
                avg_throughput_samples_per_sec=0.0,
                peak_memory_mb=0.0,
                avg_memory_mb=0.0,
                min_batch_time_ms=0.0,
                max_batch_time_ms=0.0,
            )
        
        total_batches = len(metrics)
        total_tokens = sum(m.tokens for m in metrics)
        total_samples = sum(m.batch_size for m in metrics)
        total_time_ms = sum(m.elapsed_ms for m in metrics)
        
        batch_times = [m.elapsed_ms for m in metrics]
        memory_values = [m.memory_mb for m in metrics]
        
        avg_batch_time_ms = total_time_ms / total_batches
        avg_throughput_tokens = (total_tokens / total_time_ms) * 1000.0 if total_time_ms > 0 else 0.0
        avg_throughput_samples = (total_samples / total_time_ms) * 1000.0 if total_time_ms > 0 else 0.0
        
        return TelemetrySummary(
            total_batches=total_batches,
            total_tokens=total_tokens,
            total_samples=total_samples,
            total_time_ms=total_time_ms,
            avg_batch_time_ms=avg_batch_time_ms,
            avg_throughput_tokens_per_sec=avg_throughput_tokens,
            avg_throughput_samples_per_sec=avg_throughput_samples,
            peak_memory_mb=max(memory_values) if memory_values else 0.0,
            avg_memory_mb=sum(memory_values) / len(memory_values) if memory_values else 0.0,
            min_batch_time_ms=min(batch_times) if batch_times else 0.0,
            max_batch_time_ms=max(batch_times) if batch_times else 0.0,
        )
    
    def get_recent_throughput(self, window: int = 10) -> float:
        """
        Get recent throughput (tokens/sec) over last N batches.
        
        Args:
            window: Number of recent batches to consider.
            
        Returns:
            Throughput in tokens/second.
        """
        summary = self.get_summary(last_n=window)
        return summary.avg_throughput_tokens_per_sec
    
    def clear(self) -> None:
        """Clear all collected metrics."""
        with self._metrics_lock:
            self._metrics.clear()
        self._start_time = time.time()
    
    def export_metrics(self) -> List[Dict[str, Any]]:
        """Export all metrics as list of dicts."""
        with self._metrics_lock:
            return [
                {
                    "batch_size": m.batch_size,
                    "tokens": m.tokens,
                    "elapsed_ms": m.elapsed_ms,
                    "memory_mb": m.memory_mb,
                    "timestamp": m.timestamp,
                    "throughput_tokens_per_sec": m.throughput_tokens_per_sec,
                }
                for m in self._metrics
            ]


def create_runtime_metrics(
    timer: CUDATimer,
    tokens: int,
    batch_size: int = 0,
) -> "RuntimeMetrics":
    """
    Create RuntimeMetrics from a CUDATimer.
    
    Helper to bridge telemetry with RuntimeController.
    
    Args:
        timer: Completed CUDATimer.
        tokens: Tokens processed in batch.
        batch_size: Number of sequences (optional).
        
    Returns:
        RuntimeMetrics suitable for controller.report_metrics().
    """
    from .runtime import RuntimeMetrics
    
    memory_mb = 0.0
    memory_peak_mb = None
    
    if torch.cuda.is_available():
        memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        # Peak is only available if memory stats are enabled
        try:
            stats = torch.cuda.memory_stats()
            memory_peak_mb = stats.get("active_bytes.all.peak", 0) / (1024 * 1024)
        except Exception:
            pass
    
    return RuntimeMetrics(
        tokens_processed=tokens,
        batch_time_ms=timer.elapsed_ms,
        memory_used_mb=memory_mb,
        memory_peak_mb=memory_peak_mb,
        batch_size=batch_size,
    )
