"""
CUDA Graph Capture and Replay Module.

Implements CUDA graph capture for static-shape inference batches to reduce
kernel launch overhead. CUDA graphs record a sequence of GPU operations once,
then replay them with minimal CPU involvement.

Key Benefits:
- Eliminates kernel launch overhead for repetitive inference calls
- Reduces CPU-GPU synchronization points
- Enables fused execution of multiple small kernels

Constraints:
- Input tensors must have fixed shape during replay
- Graph capture requires warmup to stabilize CUDA state
- Not compatible with dynamic control flow or shape-dependent operations

Gated by: execution.enable_cuda_graphs config flag

Reference: Performance Optimization Report - Graph Compilation Section
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Generic
import threading

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class GraphCacheConfig:
    """
    Configuration for CUDA graph caching.
    
    Attributes:
        enabled: Master switch for CUDA graph capture (respects config flag).
        warmup_iterations: Number of warmup runs before capture.
        max_cached_graphs: Maximum number of graphs to cache (LRU eviction).
        capture_pool_size_mb: Memory pool size for graph capture (MB).
        use_cuda_graph_memory_pool: Use CUDA graph-specific memory pool.
    """
    enabled: bool = False
    warmup_iterations: int = 3
    max_cached_graphs: int = 16
    capture_pool_size_mb: int = 256
    use_cuda_graph_memory_pool: bool = True


@dataclass
class CapturedGraph:
    """
    A captured CUDA graph with its associated buffers.
    
    Attributes:
        graph: The captured CUDA graph object.
        input_buffers: Pre-allocated input tensor buffers (shape-matched).
        output_buffers: Pre-allocated output tensor buffers.
        shape_key: The shape key this graph was captured for.
        capture_stream: CUDA stream used during capture.
    """
    graph: torch.cuda.CUDAGraph
    input_buffers: Dict[str, Tensor]
    output_buffers: Dict[str, Tensor]
    shape_key: Tuple[int, ...]
    capture_stream: Optional[torch.cuda.Stream] = None


class GraphCache:
    """
    CUDA Graph cache manager for static-shape inference batches.
    
    Captures and replays CUDA graphs for inference operations with fixed
    input shapes, eliminating kernel launch overhead for repetitive calls.
    
    Architecture:
    - Shape-keyed cache: Each unique input shape gets its own captured graph
    - LRU eviction: Oldest graphs evicted when cache is full
    - Thread-safe: Uses locks for concurrent access from multiple workers
    
    Usage:
        cache = GraphCache(config)
        
        # First call with new shape: capture graph
        # Subsequent calls: replay cached graph
        outputs = cache.run_with_graph(
            shape_key=(batch_size, seq_len),
            forward_fn=model.forward,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    
    Note: CUDA graphs require static shapes. Use bucketing to limit shape variety.
    """
    
    def __init__(self, config: Optional[GraphCacheConfig] = None):
        """
        Initialize CUDA graph cache.
        
        Args:
            config: Graph cache configuration. If None, uses defaults.
        """
        self.config = config or GraphCacheConfig()
        
        # Shape key -> CapturedGraph mapping
        self._cache: Dict[Tuple[int, ...], CapturedGraph] = {}
        
        # LRU tracking: list of shape keys in access order (oldest first)
        self._lru_order: List[Tuple[int, ...]] = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._captures = 0
        
        # Capture stream (reused across captures)
        self._capture_stream: Optional[torch.cuda.Stream] = None
        
        # Check CUDA availability
        self._cuda_available = torch.cuda.is_available()
        if self.config.enabled and not self._cuda_available:
            logger.warning("CUDA graphs requested but CUDA not available - disabling")
            self.config.enabled = False
        
        if self.config.enabled:
            logger.info(
                f"GraphCache initialized: max_graphs={self.config.max_cached_graphs}, "
                f"warmup={self.config.warmup_iterations}"
            )
    
    @property
    def is_enabled(self) -> bool:
        """Check if CUDA graphs are enabled and available."""
        return self.config.enabled and self._cuda_available
    
    def capture(
        self,
        shape_key: Tuple[int, ...],
        forward_fn: Callable[..., Dict[str, Tensor]],
        **input_tensors: Tensor,
    ) -> CapturedGraph:
        """
        Capture a CUDA graph for the given input shape.
        
        Performs warmup iterations, then captures the forward function
        execution into a replayable CUDA graph.
        
        Args:
            shape_key: Tuple identifying the input shape (e.g., (batch, seq_len)).
            forward_fn: Model forward function to capture.
            **input_tensors: Named input tensors (must be on CUDA).
            
        Returns:
            CapturedGraph containing the graph and pre-allocated buffers.
            
        Raises:
            RuntimeError: If capture fails or inputs aren't on CUDA.
        """
        if not self._cuda_available:
            raise RuntimeError("CUDA not available for graph capture")
        
        # Validate all inputs are on CUDA
        for name, tensor in input_tensors.items():
            if not tensor.is_cuda:
                raise RuntimeError(
                    f"Input tensor '{name}' must be on CUDA for graph capture, "
                    f"got device: {tensor.device}"
                )
        
        logger.debug(f"Capturing CUDA graph for shape_key={shape_key}")
        
        # Create capture stream if needed
        if self._capture_stream is None:
            self._capture_stream = torch.cuda.Stream()
        
        # Allocate input buffers (copy input shapes)
        input_buffers = {
            name: tensor.clone().detach()
            for name, tensor in input_tensors.items()
        }
        
        # Warmup iterations (stabilizes CUDA state)
        with torch.cuda.stream(self._capture_stream):
            for _ in range(self.config.warmup_iterations):
                _ = forward_fn(**input_buffers)
        
        # Synchronize before capture
        torch.cuda.current_stream().wait_stream(self._capture_stream)
        
        # Capture the graph
        graph = torch.cuda.CUDAGraph()
        
        with torch.cuda.graph(graph, stream=self._capture_stream):
            output_dict = forward_fn(**input_buffers)
        
        # Clone output buffers (these will hold results during replay)
        output_buffers = {
            name: tensor.clone().detach()
            for name, tensor in output_dict.items()
        }
        
        self._captures += 1
        
        captured = CapturedGraph(
            graph=graph,
            input_buffers=input_buffers,
            output_buffers=output_buffers,
            shape_key=shape_key,
            capture_stream=self._capture_stream,
        )
        
        logger.debug(f"CUDA graph captured: shape={shape_key}, outputs={list(output_buffers.keys())}")
        
        return captured
    
    def replay(
        self,
        captured: CapturedGraph,
        **input_tensors: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Replay a captured CUDA graph with new input data.
        
        Copies input data into pre-allocated buffers, replays the graph,
        and returns output tensors.
        
        Args:
            captured: Previously captured graph.
            **input_tensors: New input tensors (same shapes as capture).
            
        Returns:
            Dict of output tensors.
        """
        # Copy input data into graph's input buffers (in-place)
        for name, tensor in input_tensors.items():
            if name in captured.input_buffers:
                captured.input_buffers[name].copy_(tensor)
        
        # Replay the graph
        captured.graph.replay()
        
        # Return clones of output buffers (avoid returning internal buffers)
        return {
            name: tensor.clone()
            for name, tensor in captured.output_buffers.items()
        }
    
    def run_with_graph(
        self,
        shape_key: Tuple[int, ...],
        forward_fn: Callable[..., Dict[str, Tensor]],
        **input_tensors: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Run forward function with CUDA graph caching.
        
        On first call with a shape: captures graph and stores in cache.
        On subsequent calls: replays cached graph for minimal overhead.
        
        If CUDA graphs are disabled or unavailable, falls back to direct execution.
        
        Args:
            shape_key: Tuple identifying input shape for caching.
            forward_fn: Model forward function.
            **input_tensors: Named input tensors.
            
        Returns:
            Dict of output tensors.
        """
        # Fallback if disabled or not on CUDA
        if not self.is_enabled:
            return forward_fn(**input_tensors)
        
        with self._lock:
            if shape_key in self._cache:
                # Cache hit - replay existing graph
                self._hits += 1
                self._update_lru(shape_key)
                return self.replay(self._cache[shape_key], **input_tensors)
            else:
                # Cache miss - capture new graph
                self._misses += 1
                
                # Check cache capacity and evict if needed
                if len(self._cache) >= self.config.max_cached_graphs:
                    self._evict_oldest()
                
                # Capture new graph
                captured = self.capture(shape_key, forward_fn, **input_tensors)
                self._cache[shape_key] = captured
                self._lru_order.append(shape_key)
                
                # Return outputs from capture (already computed)
                return dict(captured.output_buffers)
    
    def _update_lru(self, shape_key: Tuple[int, ...]) -> None:
        """Move shape_key to end of LRU list (most recently used)."""
        if shape_key in self._lru_order:
            self._lru_order.remove(shape_key)
        self._lru_order.append(shape_key)
    
    def _evict_oldest(self) -> None:
        """Evict the least recently used cached graph."""
        if not self._lru_order:
            return
        
        oldest_key = self._lru_order.pop(0)
        if oldest_key in self._cache:
            del self._cache[oldest_key]
            logger.debug(f"Evicted CUDA graph for shape_key={oldest_key}")
    
    def clear(self) -> int:
        """
        Clear all cached graphs.
        
        Returns:
            Number of graphs cleared.
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._lru_order.clear()
            logger.debug(f"Cleared {count} cached CUDA graphs")
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with hits, misses, captures, and cached shapes.
        """
        with self._lock:
            hit_rate = self._hits / max(self._hits + self._misses, 1)
            return {
                "enabled": self.is_enabled,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "captures": self._captures,
                "cached_shapes": list(self._cache.keys()),
                "cache_size": len(self._cache),
                "max_cache_size": self.config.max_cached_graphs,
            }
    
    def __contains__(self, shape_key: Tuple[int, ...]) -> bool:
        """Check if a shape is cached."""
        with self._lock:
            return shape_key in self._cache
    
    def __len__(self) -> int:
        """Get number of cached graphs."""
        with self._lock:
            return len(self._cache)


def create_graph_cache_from_config(config: Dict[str, Any]) -> GraphCache:
    """
    Create GraphCache from pipeline configuration dict.
    
    Args:
        config: Pipeline config with execution.enable_cuda_graphs, etc.
        
    Returns:
        Configured GraphCache instance.
    """
    execution_cfg = config.get("execution", {})
    
    graph_cfg = GraphCacheConfig(
        enabled=bool(execution_cfg.get("enable_cuda_graphs", False)),
        warmup_iterations=int(execution_cfg.get("cuda_graph_warmup", 3)),
        max_cached_graphs=int(execution_cfg.get("cuda_graph_cache_size", 16)),
    )
    
    return GraphCache(graph_cfg)


# ---------------------------------------------------------------------------
# torch.compile Integration
# ---------------------------------------------------------------------------


@dataclass
class CompileConfig:
    """
    Configuration for torch.compile optimization.
    
    Attributes:
        enabled: Master switch for model compilation.
        mode: Compilation mode ("default", "reduce-overhead", "max-autotune").
        fullgraph: Attempt to capture entire model as single graph.
        dynamic: Enable dynamic shape support (slower but flexible).
        backend: Compilation backend ("inductor", "eager", "aot_eager").
    """
    enabled: bool = False
    mode: str = "reduce-overhead"
    fullgraph: bool = False
    dynamic: bool = False
    backend: str = "inductor"


def compile_model_if_enabled(
    model: torch.nn.Module,
    config: Optional[CompileConfig] = None,
) -> torch.nn.Module:
    """
    Apply torch.compile to model if enabled and available.
    
    Uses "reduce-overhead" mode by default for minimal kernel launch overhead,
    which is the primary bottleneck for small batch sizes.
    
    Args:
        model: PyTorch model to compile.
        config: Compilation configuration.
        
    Returns:
        Compiled model (or original if compilation disabled/unavailable).
    """
    config = config or CompileConfig()
    
    if not config.enabled:
        logger.debug("torch.compile disabled by config")
        return model
    
    # Check PyTorch version (requires 2.0+)
    torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
    if torch_version < (2, 0):
        logger.warning(
            f"torch.compile requires PyTorch 2.0+, got {torch.__version__}. "
            "Falling back to uncompiled model."
        )
        return model
    
    try:
        logger.info(
            f"Compiling model with torch.compile: mode={config.mode}, "
            f"backend={config.backend}, fullgraph={config.fullgraph}"
        )
        
        compiled = torch.compile(
            model,
            mode=config.mode,
            fullgraph=config.fullgraph,
            dynamic=config.dynamic,
            backend=config.backend,
        )
        
        logger.info("Model compiled successfully")
        return compiled
        
    except Exception as e:
        logger.warning(f"torch.compile failed: {e}. Using uncompiled model.")
        return model


def create_compile_config_from_dict(config: Dict[str, Any]) -> CompileConfig:
    """
    Create CompileConfig from pipeline configuration dict.
    
    Args:
        config: Pipeline config with execution.enable_torch_compile, etc.
        
    Returns:
        Configured CompileConfig instance.
    """
    execution_cfg = config.get("execution", {})
    
    return CompileConfig(
        enabled=bool(execution_cfg.get("enable_torch_compile", False)),
        mode=str(execution_cfg.get("torch_compile_mode", "reduce-overhead")),
        fullgraph=bool(execution_cfg.get("torch_compile_fullgraph", False)),
        dynamic=bool(execution_cfg.get("torch_compile_dynamic", False)),
        backend=str(execution_cfg.get("torch_compile_backend", "inductor")),
    )
