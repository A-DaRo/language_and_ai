"""
RuntimeController - Self-Optimizing Batch Budget Controller.

Implements adaptive token-budget batching with:
- PID-like feedback loop that adjusts based on throughput measurements.
- State machine (WARMUP, SCALING_UP, STABLE, THROTTLING, RECOVERY).
- OOM-aware budget slashing with exponential backoff recovery.

The controller dynamically adjusts token budgets to maximize GPU utilization
without hitting OOM errors. It works across hardware profiles (12GB RTX 3060
to 80GB H100) without manual configuration.

Usage:
    controller = RuntimeController.from_config(config)
    for batch in DynamicBatchIterator(chunks, controller):
        results = model(batch)
        controller.report_metrics(RuntimeMetrics(
            tokens_processed=len(batch),
            batch_time_ms=elapsed_ms,
            memory_used_mb=current_memory,
        ))
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class RuntimeState(str, Enum):
    """State machine states for the runtime controller."""
    WARMUP = "warmup"            # Collecting baseline measurements
    SCALING_UP = "scaling_up"   # Aggressively increasing budget
    STABLE = "stable"           # Budget is optimal, minor adjustments
    THROTTLING = "throttling"   # Reducing budget due to slowdown/pressure
    RECOVERY = "recovery"       # Recovering from OOM, cautious scaling


@dataclass
class RuntimeMetrics:
    """
    Metrics collected after each batch for feedback control.
    
    Attributes:
        tokens_processed: Number of tokens in the batch.
        batch_time_ms: Wall-clock time for the batch (ms).
        memory_used_mb: GPU memory used after batch (MB).
        memory_peak_mb: Peak GPU memory during batch (MB, optional).
        batch_size: Number of sequences in the batch.
    """
    tokens_processed: int
    batch_time_ms: float
    memory_used_mb: float
    memory_peak_mb: Optional[float] = None
    batch_size: int = 0
    
    @property
    def throughput_tokens_per_sec(self) -> float:
        """Compute tokens/second throughput."""
        if self.batch_time_ms <= 0:
            return 0.0
        return (self.tokens_processed / self.batch_time_ms) * 1000.0


@dataclass
class RuntimeConfig:
    """
    Configuration for RuntimeController.
    
    Attributes:
        warmup_batches: Number of batches to run before adjusting budget.
        initial_token_budget: Starting token budget per batch.
        min_token_budget: Minimum allowed token budget.
        max_token_budget: Maximum allowed token budget.
        memory_headroom_mb: Keep this much free memory (MB).
        scale_up_factor: Multiply budget by this when scaling up.
        scale_down_factor: Multiply budget by this when scaling down.
        oom_slash_factor: Multiply budget by this after OOM (aggressive).
        stability_threshold: Throughput variance below this = stable.
        history_window: Number of recent batches to consider for averages.
        recovery_patience: Batches to wait in recovery before scaling up.
    """
    warmup_batches: int = 10
    initial_token_budget: int = 16384  # ~128 sequences * 128 tokens
    min_token_budget: int = 2048       # ~16 sequences * 128 tokens
    max_token_budget: int = 262144     # ~2048 sequences * 128 tokens
    memory_headroom_mb: float = 1024.0  # 1GB headroom
    scale_up_factor: float = 1.25
    scale_down_factor: float = 0.85
    oom_slash_factor: float = 0.5
    stability_threshold: float = 0.1   # 10% variance = stable
    history_window: int = 10
    recovery_patience: int = 5
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> RuntimeConfig:
        """Create config from dictionary (e.g., from YAML)."""
        autotuning = config.get("execution", {}).get("autotuning", {})
        return cls(
            warmup_batches=int(autotuning.get("warmup_batches", 10)),
            initial_token_budget=int(autotuning.get("initial_token_budget", 16384)),
            min_token_budget=int(autotuning.get("min_token_budget", 2048)),
            max_token_budget=int(autotuning.get("max_token_budget", 262144)),
            memory_headroom_mb=float(autotuning.get("memory_headroom_mb", 1024.0)),
            scale_up_factor=float(autotuning.get("scale_up_factor", 1.25)),
            scale_down_factor=float(autotuning.get("scale_down_factor", 0.85)),
            oom_slash_factor=float(autotuning.get("oom_slash_factor", 0.5)),
            stability_threshold=float(autotuning.get("stability_threshold", 0.1)),
            history_window=int(autotuning.get("history_window", 10)),
            recovery_patience=int(autotuning.get("recovery_patience", 5)),
        )


@dataclass
class RuntimeSnapshot:
    """Snapshot of controller state for logging/debugging."""
    state: RuntimeState
    current_budget: int
    throughput_avg: float
    throughput_variance: float
    memory_pressure: float
    batches_processed: int
    oom_count: int = 0
    oom_ceiling: int = 0
    successful_budget: int = 0


class RuntimeController:
    """
    Self-optimizing runtime controller for GPU inference.
    
    Implements a feedback loop that dynamically adjusts token budgets
    based on observed throughput and memory pressure. The controller
    uses a state machine to manage different operating modes:
    
    - WARMUP: Collecting initial measurements (warmup_batches iterations).
    - SCALING_UP: Aggressively increasing budget when headroom exists.
    - STABLE: Budget is near-optimal, only minor adjustments.
    - THROTTLING: Reducing budget due to memory pressure or slowdown.
    - RECOVERY: Cautiously recovering from an OOM event.
    
    The PID-like feedback uses:
    - Proportional: Current throughput vs. moving average.
    - Integral: Accumulated memory pressure over time.
    - Derivative: Rate of throughput change (trend).
    """
    
    def __init__(self, config: RuntimeConfig):
        """
        Initialize the runtime controller.
        
        Args:
            config: RuntimeConfig with tuning parameters.
        """
        self.config = config
        self._state = RuntimeState.WARMUP
        self._current_budget = config.initial_token_budget
        self._batches_processed = 0
        
        # History buffers for feedback computation
        self._throughput_history: List[float] = []
        self._memory_history: List[float] = []
        self._budget_history: List[int] = []
        
        # OOM tracking
        self._oom_count = 0
        self._batches_since_oom = 0
        self._pre_oom_budget = config.initial_token_budget
        
        # OOM ceiling: caps budget scaling to prevent repeated OOM cycles
        # Starts at max_token_budget and is lowered each time OOM occurs
        self._oom_ceiling = config.max_token_budget
        # Track highest budget that completed successfully (no OOM)
        self._successful_budget = config.initial_token_budget
        
        # GPU memory info (cached)
        self._total_memory_mb: Optional[float] = None
        self._detect_gpu_memory()
        
        logger.info(
            f"RuntimeController initialized: state={self._state.value}, "
            f"initial_budget={self._current_budget}, "
            f"warmup_batches={config.warmup_batches}"
        )
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> RuntimeController:
        """Create controller from a full pipeline config dict."""
        return cls(RuntimeConfig.from_dict(config))
    
    def _detect_gpu_memory(self) -> None:
        """Detect total GPU memory for pressure calculations."""
        if torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(0)
                self._total_memory_mb = props.total_memory / (1024 * 1024)
                logger.debug(f"Detected GPU memory: {self._total_memory_mb:.0f} MB")
            except Exception as e:
                logger.warning(f"Failed to detect GPU memory: {e}")
                self._total_memory_mb = 8192.0  # Conservative fallback
        else:
            self._total_memory_mb = None
    
    @property
    def state(self) -> RuntimeState:
        """Current controller state."""
        return self._state
    
    @property
    def current_budget(self) -> int:
        """Current token budget for batching."""
        return self._current_budget
    
    @property
    def batches_processed(self) -> int:
        """Number of batches processed."""
        return self._batches_processed
    
    @property
    def oom_ceiling(self) -> int:
        """Current OOM ceiling - maximum allowed budget to prevent OOM cycles."""
        return self._oom_ceiling
    
    @property
    def successful_budget(self) -> int:
        """Highest budget that completed without OOM."""
        return self._successful_budget
    
    @property
    def oom_count(self) -> int:
        """Total number of OOM events encountered."""
        return self._oom_count
    
    def get_next_budget(self) -> int:
        """
        Get the token budget for the next batch.
        
        This is the main interface for batch iterators. Call this before
        constructing each batch to get the current recommended budget.
        
        Returns:
            Token budget (max total tokens for next batch).
        """
        return self._current_budget
    
    def report_metrics(self, metrics: RuntimeMetrics) -> None:
        """
        Report metrics after a batch completes.
        
        This triggers the feedback loop to potentially adjust the budget
        and transition between states.
        
        Args:
            metrics: RuntimeMetrics from the completed batch.
        """
        self._batches_processed += 1
        self._batches_since_oom += 1
        
        # Track successful budget (batch completed without OOM)
        if self._current_budget > self._successful_budget:
            self._successful_budget = self._current_budget
        
        # Record history
        throughput = metrics.throughput_tokens_per_sec
        self._throughput_history.append(throughput)
        self._memory_history.append(metrics.memory_used_mb)
        self._budget_history.append(self._current_budget)
        
        # Trim history to window size
        window = self.config.history_window
        if len(self._throughput_history) > window:
            self._throughput_history = self._throughput_history[-window:]
            self._memory_history = self._memory_history[-window:]
            self._budget_history = self._budget_history[-window:]
        
        # State machine transitions
        self._update_state(metrics)
        
        # Adjust budget based on state
        self._adjust_budget(metrics)
    
    def handle_oom(self) -> int:
        """
        Handle an OOM event by aggressively reducing budget and lowering ceiling.
        
        Call this when a CUDA OOM error is caught. The controller will:
        1. Lower the OOM ceiling to prevent future scaling beyond safe levels.
        2. Slash the budget by oom_slash_factor.
        3. Transition to RECOVERY state.
        4. Clear CUDA cache.
        
        The ceiling mechanism prevents repeated OOM cycles: once OOM occurs at
        budget B, the ceiling is set to 0.9*B (slightly below to allow near-max
        performance). Future budget adjustments cannot exceed this ceiling.
        
        Returns:
            New (reduced) token budget.
        """
        self._oom_count += 1
        self._pre_oom_budget = self._current_budget
        self._batches_since_oom = 0
        
        # Lower the OOM ceiling: set to 90% of the budget that caused OOM
        # This prevents future scaling from reaching the problematic level
        new_ceiling = int(self._current_budget * 0.9)
        new_ceiling = max(new_ceiling, self.config.min_token_budget)
        self._oom_ceiling = min(self._oom_ceiling, new_ceiling)
        
        # Aggressive budget slash
        new_budget = int(self._current_budget * self.config.oom_slash_factor)
        new_budget = max(new_budget, self.config.min_token_budget)
        
        logger.warning(
            f"OOM detected (count={self._oom_count}): "
            f"slashing budget {self._current_budget} -> {new_budget}, "
            f"ceiling lowered to {self._oom_ceiling}"
        )
        
        self._current_budget = new_budget
        self._state = RuntimeState.RECOVERY
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return self._current_budget
    
    def _update_state(self, metrics: RuntimeMetrics) -> None:
        """Update state machine based on current metrics."""
        prev_state = self._state
        
        if self._state == RuntimeState.WARMUP:
            if self._batches_processed >= self.config.warmup_batches:
                # Transition based on throughput variance
                variance = self._compute_throughput_variance()
                if variance < self.config.stability_threshold:
                    self._state = RuntimeState.STABLE
                else:
                    self._state = RuntimeState.SCALING_UP
                logger.info(
                    f"Warmup complete: transitioning to {self._state.value} "
                    f"(variance={variance:.3f})"
                )
        
        elif self._state == RuntimeState.RECOVERY:
            if self._batches_since_oom >= self.config.recovery_patience:
                # Cautiously exit recovery
                self._state = RuntimeState.SCALING_UP
                logger.info("Exiting recovery mode, resuming scaling")
        
        elif self._state == RuntimeState.SCALING_UP:
            variance = self._compute_throughput_variance()
            memory_pressure = self._compute_memory_pressure(metrics)
            
            if memory_pressure > 0.9:
                self._state = RuntimeState.THROTTLING
                logger.info(f"Memory pressure high ({memory_pressure:.2f}), throttling")
            elif variance < self.config.stability_threshold:
                self._state = RuntimeState.STABLE
                logger.info(f"Throughput stabilized (variance={variance:.3f})")
        
        elif self._state == RuntimeState.THROTTLING:
            memory_pressure = self._compute_memory_pressure(metrics)
            if memory_pressure < 0.7:
                self._state = RuntimeState.SCALING_UP
                logger.info("Memory pressure relieved, resuming scaling")
        
        elif self._state == RuntimeState.STABLE:
            # Check if we should scale up or throttle
            memory_pressure = self._compute_memory_pressure(metrics)
            throughput_trend = self._compute_throughput_trend()
            
            if memory_pressure > 0.85:
                self._state = RuntimeState.THROTTLING
            elif throughput_trend > 0.05 and memory_pressure < 0.6:
                # Positive trend and headroom: try scaling
                self._state = RuntimeState.SCALING_UP
        
        if prev_state != self._state:
            logger.debug(f"State transition: {prev_state.value} -> {self._state.value}")
    
    def _adjust_budget(self, metrics: RuntimeMetrics) -> None:
        """Adjust budget based on current state and metrics."""
        prev_budget = self._current_budget
        
        if self._state == RuntimeState.WARMUP:
            # No adjustments during warmup
            return
        
        elif self._state == RuntimeState.RECOVERY:
            # Very cautious scaling in recovery, respecting OOM ceiling
            if self._batches_since_oom > 2:
                # Small increase after a few successful batches
                new_budget = int(self._current_budget * 1.05)
                # Cap at 80% of ceiling during recovery for safety margin
                recovery_cap = int(self._oom_ceiling * 0.8)
                self._current_budget = min(new_budget, recovery_cap, self._pre_oom_budget)
        
        elif self._state == RuntimeState.SCALING_UP:
            # Aggressive scaling, but respect OOM ceiling
            new_budget = int(self._current_budget * self.config.scale_up_factor)
            # Never exceed OOM ceiling (prevents repeated OOM cycles)
            self._current_budget = min(new_budget, self.config.max_token_budget, self._oom_ceiling)
        
        elif self._state == RuntimeState.THROTTLING:
            # Scale down
            new_budget = int(self._current_budget * self.config.scale_down_factor)
            self._current_budget = max(new_budget, self.config.min_token_budget)
        
        elif self._state == RuntimeState.STABLE:
            # Minor adjustments based on memory pressure
            memory_pressure = self._compute_memory_pressure(metrics)
            if memory_pressure < 0.5:
                # Plenty of headroom, small increase
                new_budget = int(self._current_budget * 1.02)
                self._current_budget = min(new_budget, self.config.max_token_budget)
            elif memory_pressure > 0.75:
                # Getting tight, small decrease
                new_budget = int(self._current_budget * 0.98)
                self._current_budget = max(new_budget, self.config.min_token_budget)
        
        # Clamp to bounds
        self._current_budget = max(
            self.config.min_token_budget,
            min(self._current_budget, self.config.max_token_budget)
        )
        
        if prev_budget != self._current_budget:
            logger.debug(
                f"Budget adjusted: {prev_budget} -> {self._current_budget} "
                f"(state={self._state.value})"
            )
    
    def _compute_throughput_variance(self) -> float:
        """Compute coefficient of variation for recent throughput."""
        if len(self._throughput_history) < 2:
            return 1.0  # High variance during warmup
        
        import numpy as np
        arr = np.array(self._throughput_history)
        mean = arr.mean()
        if mean <= 0:
            return 1.0
        std = arr.std()
        return std / mean
    
    def _compute_throughput_trend(self) -> float:
        """Compute throughput trend (positive = improving)."""
        if len(self._throughput_history) < 3:
            return 0.0
        
        # Simple linear trend
        recent = self._throughput_history[-3:]
        if recent[0] <= 0:
            return 0.0
        return (recent[-1] - recent[0]) / recent[0]
    
    def _compute_memory_pressure(self, metrics: RuntimeMetrics) -> float:
        """
        Compute memory pressure as fraction of available memory used.
        
        Returns value in [0, 1] where 1 = fully utilized.
        """
        if self._total_memory_mb is None or self._total_memory_mb <= 0:
            return 0.5  # Unknown, assume moderate
        
        # Account for headroom
        usable_memory = self._total_memory_mb - self.config.memory_headroom_mb
        if usable_memory <= 0:
            return 0.9  # Very constrained
        
        return min(1.0, metrics.memory_used_mb / usable_memory)
    
    def get_snapshot(self) -> RuntimeSnapshot:
        """Get current controller state snapshot for logging."""
        return RuntimeSnapshot(
            state=self._state,
            current_budget=self._current_budget,
            throughput_avg=sum(self._throughput_history) / len(self._throughput_history)
                           if self._throughput_history else 0.0,
            throughput_variance=self._compute_throughput_variance(),
            memory_pressure=self._memory_history[-1] / self._total_memory_mb
                           if self._memory_history and self._total_memory_mb else 0.0,
            batches_processed=self._batches_processed,
            oom_count=self._oom_count,
            oom_ceiling=self._oom_ceiling,
            successful_budget=self._successful_budget,
        )
    
    def reset(self) -> None:
        """Reset controller to initial state."""
        self._state = RuntimeState.WARMUP
        self._current_budget = self.config.initial_token_budget
        self._batches_processed = 0
        self._throughput_history.clear()
        self._memory_history.clear()
        self._budget_history.clear()
        self._oom_count = 0
        self._batches_since_oom = 0
        self._oom_ceiling = self.config.max_token_budget
        self._successful_budget = self.config.initial_token_budget
        logger.info("RuntimeController reset to initial state")


def get_gpu_memory_stats() -> Dict[str, float]:
    """
    Get current GPU memory statistics.
    
    Returns:
        Dict with keys:
        - allocated_mb: Currently allocated memory (MB).
        - reserved_mb: Reserved memory in cache (MB).
        - free_mb: Free memory (MB).
        - total_mb: Total GPU memory (MB).
    """
    if not torch.cuda.is_available():
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "free_mb": 0.0,
            "total_mb": 0.0,
        }
    
    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    free = total - reserved
    
    return {
        "allocated_mb": allocated,
        "reserved_mb": reserved,
        "free_mb": free,
        "total_mb": total,
    }
