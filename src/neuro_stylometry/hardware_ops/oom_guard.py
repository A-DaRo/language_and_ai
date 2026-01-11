"""
OOMGuard - Out-of-Memory Protection for GPU Operations.

Provides robust OOM handling with automatic retry and budget adjustment.
Integrates with RuntimeController for feedback-based recovery.

Usage:
    controller = RuntimeController.from_config(config)
    
    # As a wrapper function
    result = execute_with_oom_protection(
        lambda: model(batch),
        controller=controller,
        retry_limit=3,
    )
    
    # As a context manager
    with oom_guarded_context(controller) as guard:
        result = model(batch)
        if guard.oom_occurred:
            # Handle partial results
            pass
"""

from __future__ import annotations

import functools
import gc
import logging
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Generator, Optional, TypeVar, Union

import torch

logger = logging.getLogger(__name__)

# Type variable for generic return types
T = TypeVar("T")


@dataclass
class OOMEvent:
    """
    Record of an OOM event for debugging and analysis.
    
    Attributes:
        budget_before: Token budget before OOM.
        budget_after: Token budget after recovery.
        memory_allocated_mb: Allocated memory at time of OOM.
        memory_reserved_mb: Reserved memory at time of OOM.
        retry_count: Number of retries so far.
        exception_msg: Original exception message.
        traceback_str: Truncated traceback for debugging.
    """
    budget_before: int
    budget_after: int
    memory_allocated_mb: float
    memory_reserved_mb: float
    retry_count: int
    exception_msg: str
    traceback_str: str


class OOMRecoveryError(RuntimeError):
    """Raised when OOM recovery fails after all retries."""
    
    def __init__(self, message: str, oom_events: list[OOMEvent]):
        super().__init__(message)
        self.oom_events = oom_events


def is_cuda_oom(exception: BaseException) -> bool:
    """
    Check if an exception is a CUDA out-of-memory error.
    
    Handles both RuntimeError and torch.cuda.OutOfMemoryError.
    
    Args:
        exception: The caught exception.
        
    Returns:
        True if this is a CUDA OOM error.
    """
    # Direct OOM error type (PyTorch >= 1.10)
    if hasattr(torch.cuda, "OutOfMemoryError"):
        if isinstance(exception, torch.cuda.OutOfMemoryError):
            return True
    
    # RuntimeError with OOM message
    if isinstance(exception, RuntimeError):
        msg = str(exception).lower()
        oom_indicators = [
            "out of memory",
            "cuda out of memory",
            "cuda error: out of memory",
            "cudnn error: cudnn_status_internal_error",  # Often caused by OOM
            "cublas error: cublas_status_alloc_failed",
        ]
        return any(indicator in msg for indicator in oom_indicators)
    
    return False


def clear_cuda_cache() -> None:
    """
    Aggressively clear CUDA cache and force garbage collection.
    
    This is called after OOM to free as much memory as possible
    before retrying with a reduced budget.
    """
    if not torch.cuda.is_available():
        return
    
    # Force Python garbage collection first
    gc.collect()
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Synchronize to ensure all operations complete
    torch.cuda.synchronize()
    
    # Second GC pass for any CUDA tensors freed by synchronization
    gc.collect()
    
    logger.debug("CUDA cache cleared and garbage collected")


def get_memory_snapshot() -> tuple[float, float]:
    """
    Get current CUDA memory usage.
    
    Returns:
        Tuple of (allocated_mb, reserved_mb).
    """
    if not torch.cuda.is_available():
        return (0.0, 0.0)
    
    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    return (allocated, reserved)


def execute_with_oom_protection(
    func: Callable[[], T],
    controller: Optional["RuntimeController"] = None,
    retry_limit: int = 3,
    on_oom: Optional[Callable[[OOMEvent], None]] = None,
    reraise_non_oom: bool = True,
) -> Optional[T]:
    """
    Execute a function with automatic OOM recovery.
    
    If a CUDA OOM error occurs:
    1. Record the event.
    2. Clear CUDA cache.
    3. If controller provided, reduce budget via handle_oom().
    4. Retry up to retry_limit times.
    5. If all retries fail, raise OOMRecoveryError.
    
    Args:
        func: Zero-argument callable to execute.
        controller: Optional RuntimeController for budget feedback.
        retry_limit: Maximum retry attempts after OOM.
        on_oom: Optional callback invoked on each OOM event.
        reraise_non_oom: If True, reraise non-OOM exceptions.
        
    Returns:
        Result of func() if successful, None if all retries exhausted.
        
    Raises:
        OOMRecoveryError: If recovery fails after all retries.
        Exception: Non-OOM exceptions if reraise_non_oom=True.
    """
    # Import here to avoid circular imports
    from .runtime import RuntimeController
    
    oom_events: list[OOMEvent] = []
    
    for attempt in range(retry_limit + 1):  # +1 for initial attempt
        try:
            return func()
            
        except Exception as e:
            if not is_cuda_oom(e):
                if reraise_non_oom:
                    raise
                logger.error(f"Non-OOM exception: {e}")
                return None
            
            # Record OOM event
            allocated, reserved = get_memory_snapshot()
            budget_before = controller.current_budget if controller else 0
            
            # Clear cache before handling
            clear_cuda_cache()
            
            # Let controller adjust budget
            budget_after = budget_before
            if controller:
                budget_after = controller.handle_oom()
            
            event = OOMEvent(
                budget_before=budget_before,
                budget_after=budget_after,
                memory_allocated_mb=allocated,
                memory_reserved_mb=reserved,
                retry_count=attempt,
                exception_msg=str(e),
                traceback_str=traceback.format_exc()[-500:],  # Last 500 chars
            )
            oom_events.append(event)
            
            # Invoke callback if provided
            if on_oom:
                try:
                    on_oom(event)
                except Exception as callback_err:
                    logger.warning(f"OOM callback error: {callback_err}")
            
            logger.warning(
                f"OOM event #{attempt + 1}/{retry_limit + 1}: "
                f"budget {budget_before} -> {budget_after}, "
                f"allocated={allocated:.0f}MB, reserved={reserved:.0f}MB"
            )
            
            if attempt >= retry_limit:
                # All retries exhausted
                raise OOMRecoveryError(
                    f"CUDA OOM after {retry_limit + 1} attempts. "
                    f"Final budget: {budget_after}",
                    oom_events,
                )
    
    # Should not reach here
    return None


@dataclass
class OOMGuardState:
    """State object for the oom_guarded_context context manager."""
    oom_occurred: bool = False
    oom_events: list[OOMEvent] = None
    final_budget: int = 0
    
    def __post_init__(self):
        if self.oom_events is None:
            self.oom_events = []


@contextmanager
def oom_guarded_context(
    controller: Optional["RuntimeController"] = None,
    clear_cache_on_exit: bool = False,
) -> Generator[OOMGuardState, None, None]:
    """
    Context manager for OOM-guarded code blocks.
    
    Unlike execute_with_oom_protection, this does NOT retry automatically.
    Instead, it catches OOM errors and allows the caller to decide how to proceed.
    
    Usage:
        with oom_guarded_context(controller) as guard:
            result = model(batch)
        
        if guard.oom_occurred:
            # Handle OOM, maybe reduce batch size and retry manually
            new_budget = guard.final_budget
    
    Args:
        controller: Optional RuntimeController for budget feedback.
        clear_cache_on_exit: Clear CUDA cache on normal exit (not just OOM).
        
    Yields:
        OOMGuardState with oom_occurred flag and event history.
    """
    from .runtime import RuntimeController
    
    state = OOMGuardState()
    state.final_budget = controller.current_budget if controller else 0
    
    try:
        yield state
        
    except Exception as e:
        if is_cuda_oom(e):
            state.oom_occurred = True
            
            allocated, reserved = get_memory_snapshot()
            budget_before = controller.current_budget if controller else 0
            
            clear_cuda_cache()
            
            budget_after = budget_before
            if controller:
                budget_after = controller.handle_oom()
            
            event = OOMEvent(
                budget_before=budget_before,
                budget_after=budget_after,
                memory_allocated_mb=allocated,
                memory_reserved_mb=reserved,
                retry_count=0,
                exception_msg=str(e),
                traceback_str=traceback.format_exc()[-500:],
            )
            state.oom_events.append(event)
            state.final_budget = budget_after
            
            logger.warning(
                f"OOM caught in guarded context: "
                f"budget {budget_before} -> {budget_after}"
            )
        else:
            # Re-raise non-OOM exceptions
            raise
    
    finally:
        if clear_cache_on_exit and torch.cuda.is_available():
            torch.cuda.empty_cache()


def oom_protected(
    retry_limit: int = 3,
    controller_attr: str = "_runtime_controller",
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for methods that should have OOM protection.
    
    The decorated method will be wrapped in execute_with_oom_protection.
    If the method's class has a RuntimeController at controller_attr,
    it will be used for budget feedback.
    
    Usage:
        class Inference:
            def __init__(self):
                self._runtime_controller = RuntimeController(...)
            
            @oom_protected(retry_limit=3)
            def run_batch(self, batch):
                return self.model(batch)
    
    Args:
        retry_limit: Maximum retry attempts.
        controller_attr: Attribute name for RuntimeController on self.
        
    Returns:
        Decorator function.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Try to get controller from self
            controller = getattr(self, controller_attr, None)
            
            return execute_with_oom_protection(
                lambda: func(self, *args, **kwargs),
                controller=controller,
                retry_limit=retry_limit,
            )
        return wrapper
    return decorator
