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


def _prepare_capture_buffer(tensor: Tensor) -> Tensor:
    clean = tensor.detach().clone()
    if tensor.requires_grad:
        clean.requires_grad_(True)
    return clean


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
        
        # CUDA graph memory pool for stable allocations across captures
        self._capture_pool: Optional[Tuple[int, int]] = None
        
        # Check CUDA availability
        self._cuda_available = torch.cuda.is_available()
        if self.config.enabled and not self._cuda_available:
            logger.warning("CUDA graphs requested but CUDA not available - disabling")
            self.config.enabled = False
        
        # Initialize memory pool if enabled
        if self.config.enabled and self.config.use_cuda_graph_memory_pool and self._cuda_available:
            try:
                # Get memory pool handle from CUDA allocator
                # This creates a dedicated pool for graph captures, reducing fragmentation
                self._capture_pool = torch.cuda.graph_pool_handle()
                logger.info(
                    f"CUDA graph memory pool initialized "
                    f"(hint: {self.config.capture_pool_size_mb}MB)"
                )
            except Exception as e:
                logger.warning(f"Failed to create CUDA graph pool: {e}")
                self._capture_pool = None
        
        if self.config.enabled:
            logger.info(
                f"GraphCache initialized: max_graphs={self.config.max_cached_graphs}, "
                f"warmup={self.config.warmup_iterations}, "
                f"memory_pool={'enabled' if self._capture_pool else 'disabled'}"
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
        
        # Ensure capture stream waits on any prior work before cloning buffers.
        self._capture_stream.wait_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()

        # Allocate input buffers (copy input shapes) on the capture stream.
        # Use detach().clone() to ensure clean buffer creation and preserve requires_grad.
        with torch.cuda.stream(self._capture_stream):
            input_buffers = {
                name: _prepare_capture_buffer(tensor)
                for name, tensor in input_tensors.items()
            }
        
        # Make capture stream wait for current stream processing?
        # torch.cuda.current_stream().synchronize() # Already covered by global sync
        
        # Warmup iterations (stabilizes CUDA state)
        with torch.cuda.stream(self._capture_stream):
            for _ in range(self.config.warmup_iterations):
                _ = forward_fn(**input_buffers)
        
        # Synchronize before capture to avoid cross-stream hazards.
        torch.cuda.synchronize()
        
        # Capture the graph with dedicated memory pool (if available)
        graph = torch.cuda.CUDAGraph()
        
        capture_kwargs = {"stream": self._capture_stream}
        if self._capture_pool is not None:
             capture_kwargs["pool"] = self._capture_pool
        
        # Ensure optimizer logic hasn't touched gradients in a way that confuses capture?
        # Since we can't control optimizer here, rely on user doing zero_grad() before run()
        
        with torch.cuda.graph(graph, **capture_kwargs):
            output_dict = forward_fn(**input_buffers)
        
        # Keep captured output tensors so replay writes into the same buffers.
        output_buffers = {name: tensor.detach() for name, tensor in output_dict.items()}
        
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


# ---------------------------------------------------------------------------
# Shape Bucketing for CUDA Graphs
# ---------------------------------------------------------------------------


@dataclass
class ShapeBucket:
    """
    A fixed-shape bucket for CUDA graph capture.
    
    Inputs are padded to the bucket's dimensions, enabling graph reuse.
    
    Attributes:
        batch_size: Fixed batch dimension for this bucket.
        seq_len: Fixed sequence length dimension for this bucket.
        usage_count: Number of times this bucket has been used.
    """
    batch_size: int
    seq_len: int
    usage_count: int = 0
    
    @property
    def shape_key(self) -> Tuple[int, int]:
        """Return the (batch_size, seq_len) tuple for graph cache lookup."""
        return (self.batch_size, self.seq_len)


@dataclass
class ShapeBucketConfig:
    """
    Configuration for shape bucketing.
    
    Attributes:
        batch_buckets: Predefined batch size buckets (powers of 2 recommended).
        seq_len_buckets: Predefined sequence length buckets.
        max_batch_size: Maximum batch size (inputs exceeding this are processed in multiple rounds).
        max_seq_len: Maximum sequence length (longer sequences are truncated).
        adaptive_buckets: If True, create new buckets for unseen shapes.
        min_bucket_usage: Minimum uses before capturing a graph for a bucket.
    """
    batch_buckets: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])
    seq_len_buckets: List[int] = field(default_factory=lambda: [128, 256, 384, 512, 768, 1024])
    max_batch_size: int = 64
    max_seq_len: int = 1024
    adaptive_buckets: bool = False
    min_bucket_usage: int = 2  # Capture graph after 2 uses (skip one-offs)


class ShapeBucketer:
    """
    Maps variable-shaped inputs to fixed-shape buckets for CUDA graph reuse.
    
    CUDA graphs require static input shapes. This class handles:
    1. Bucket assignment: Maps (actual_batch, actual_seq) -> (bucket_batch, bucket_seq)
    2. Padding: Pads inputs to bucket dimensions
    3. Unpadding: Extracts valid outputs after graph replay
    
    Architecture:
    - Buckets are defined by quantized (batch_size, seq_len) pairs
    - Inputs are padded to the smallest bucket that fits
    - Graph cache stores one graph per bucket
    
    Example:
        bucketer = ShapeBucketer(config)
        
        # Pad inputs to bucket shape
        bucket, padded_inputs = bucketer.pad_to_bucket(input_ids, attention_mask)
        
        # Run through CUDA graph
        outputs = graph_cache.run_with_graph(
            bucket.shape_key, model.forward, **padded_inputs
        )
        
        # Unpad outputs
        valid_outputs = bucketer.unpad_outputs(outputs, original_batch_size, original_seq_len)
    """
    
    def __init__(self, config: Optional[ShapeBucketConfig] = None):
        """
        Initialize shape bucketer.
        
        Args:
            config: Bucket configuration. If None, uses defaults.
        """
        self.config = config or ShapeBucketConfig()
        
        # Pre-compute all valid bucket combinations
        self._buckets: Dict[Tuple[int, int], ShapeBucket] = {}
        for batch in self.config.batch_buckets:
            for seq in self.config.seq_len_buckets:
                key = (batch, seq)
                self._buckets[key] = ShapeBucket(batch_size=batch, seq_len=seq)
        
        # Sorted bucket dimensions for fast lookup
        self._sorted_batches = sorted(self.config.batch_buckets)
        self._sorted_seqs = sorted(self.config.seq_len_buckets)
        
        # Statistics
        self._total_padded_tokens = 0
        self._total_actual_tokens = 0
        self._bucket_hits: Dict[Tuple[int, int], int] = {}
        
        logger.info(
            f"ShapeBucketer initialized: {len(self._sorted_batches)} batch buckets × "
            f"{len(self._sorted_seqs)} seq buckets = {len(self._buckets)} total buckets"
        )
    
    def find_bucket(self, batch_size: int, seq_len: int) -> Optional[ShapeBucket]:
        """
        Find the smallest bucket that fits the given dimensions.
        
        Args:
            batch_size: Actual batch size.
            seq_len: Actual sequence length.
            
        Returns:
            ShapeBucket if one fits, None if dimensions exceed max buckets.
        """
        # Find smallest fitting batch bucket
        target_batch = None
        for b in self._sorted_batches:
            if b >= batch_size:
                target_batch = b
                break
        
        if target_batch is None:
            if batch_size <= self.config.max_batch_size:
                target_batch = self.config.max_batch_size
            else:
                return None  # Batch too large
        
        # Find smallest fitting seq bucket
        target_seq = None
        for s in self._sorted_seqs:
            if s >= seq_len:
                target_seq = s
                break
        
        if target_seq is None:
            if seq_len <= self.config.max_seq_len:
                target_seq = self.config.max_seq_len
            else:
                return None  # Sequence too long
        
        key = (target_batch, target_seq)
        
        # Create bucket if not exists (adaptive mode or within existing bounds)
        if key not in self._buckets:
            if self.config.adaptive_buckets or (
                target_batch in self._sorted_batches and target_seq in self._sorted_seqs
            ):
                self._buckets[key] = ShapeBucket(batch_size=target_batch, seq_len=target_seq)
                logger.debug(f"Created new bucket: {key}")
            else:
                return None
        
        bucket = self._buckets[key]
        bucket.usage_count += 1
        
        # Track statistics
        self._bucket_hits[key] = self._bucket_hits.get(key, 0) + 1
        
        return bucket
    
    def should_capture_graph(self, bucket: ShapeBucket) -> bool:
        """
        Determine if a graph should be captured for this bucket.
        
        Args:
            bucket: The bucket to check.
            
        Returns:
            True if bucket has been used enough times to warrant graph capture.
        """
        return bucket.usage_count >= self.config.min_bucket_usage
    
    def pad_tensors(
        self,
        bucket: ShapeBucket,
        input_ids: Tensor,
        attention_mask: Tensor,
        pad_token_id: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Pad input tensors to bucket dimensions.
        
        Args:
            bucket: Target bucket for padding.
            input_ids: Original input IDs (batch, seq).
            attention_mask: Original attention mask (batch, seq).
            pad_token_id: Token ID for padding.
            
        Returns:
            Tuple of (padded_input_ids, padded_attention_mask).
        """
        actual_batch, actual_seq = input_ids.shape
        target_batch, target_seq = bucket.batch_size, bucket.seq_len
        
        # Track padding statistics
        self._total_actual_tokens += actual_batch * actual_seq
        self._total_padded_tokens += target_batch * target_seq
        
        # No padding needed if shapes match
        if actual_batch == target_batch and actual_seq == target_seq:
            return input_ids, attention_mask
        
        # Pad sequence dimension
        if actual_seq < target_seq:
            seq_pad = target_seq - actual_seq
            input_ids = torch.nn.functional.pad(
                input_ids, (0, seq_pad), value=pad_token_id
            )
            attention_mask = torch.nn.functional.pad(
                attention_mask, (0, seq_pad), value=0
            )
        
        # Pad batch dimension
        if actual_batch < target_batch:
            batch_pad = target_batch - actual_batch
            # Pad with zeros (will be masked out)
            input_ids = torch.nn.functional.pad(
                input_ids, (0, 0, 0, batch_pad), value=pad_token_id
            )
            attention_mask = torch.nn.functional.pad(
                attention_mask, (0, 0, 0, batch_pad), value=0
            )
        
        return input_ids, attention_mask
    
    def unpad_outputs(
        self,
        outputs: Dict[str, Tensor],
        original_batch: int,
        original_seq: int,
    ) -> Dict[str, Tensor]:
        """
        Remove padding from output tensors.
        
        Args:
            outputs: Padded output tensors.
            original_batch: Original batch size before padding.
            original_seq: Original sequence length before padding.
            
        Returns:
            Unpadded output tensors.
        """
        unpadded = {}
        for name, tensor in outputs.items():
            if tensor.dim() >= 2:
                # Assume (batch, seq, ...) or (batch, ...)
                if tensor.dim() >= 2 and tensor.shape[0] >= original_batch:
                    tensor = tensor[:original_batch]
                if tensor.dim() >= 2 and tensor.shape[1] >= original_seq:
                    tensor = tensor[:, :original_seq]
            unpadded[name] = tensor
        return unpadded
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get bucketing statistics.
        
        Returns:
            Dict with padding efficiency and bucket usage.
        """
        efficiency = (
            self._total_actual_tokens / max(self._total_padded_tokens, 1) * 100
        )
        
        return {
            "total_actual_tokens": self._total_actual_tokens,
            "total_padded_tokens": self._total_padded_tokens,
            "padding_efficiency_pct": efficiency,
            "bucket_hits": dict(self._bucket_hits),
            "num_active_buckets": len(self._bucket_hits),
        }
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._total_padded_tokens = 0
        self._total_actual_tokens = 0
        self._bucket_hits.clear()


class GraphAwareInference:
    """
    High-level API for CUDA graph-accelerated inference with shape bucketing.
    
    Combines ShapeBucketer and GraphCache to provide a seamless interface
    for graph-accelerated inference on variable-shaped inputs.
    
    Architecture:
    1. Inputs arrive with variable (batch, seq) shapes
    2. ShapeBucketer finds appropriate fixed-shape bucket
    3. Inputs are padded to bucket dimensions
    4. GraphCache captures or replays graph for bucket shape
    5. Outputs are unpadded and returned
    
    This class handles the common case where graph capture is beneficial
    (repeated inference with similar shapes) while gracefully falling back
    to direct execution for one-off shapes.
    """
    
    def __init__(
        self,
        graph_cache: GraphCache,
        bucket_config: Optional[ShapeBucketConfig] = None,
    ):
        """
        Initialize graph-aware inference.
        
        Args:
            graph_cache: Configured GraphCache instance.
            bucket_config: Shape bucketing configuration.
        """
        self.graph_cache = graph_cache
        self.bucketer = ShapeBucketer(bucket_config)
        
        # Track fallback stats
        self._direct_calls = 0
        self._graph_calls = 0
    
    @property
    def is_enabled(self) -> bool:
        """Check if CUDA graphs are enabled."""
        return self.graph_cache.is_enabled
    
    def run(
        self,
        forward_fn: Callable[..., Dict[str, Tensor]],
        input_ids: Tensor,
        attention_mask: Tensor,
        pad_token_id: int = 0,
        **extra_kwargs: Any,
    ) -> Dict[str, Tensor]:
        """
        Run forward function with automatic graph caching.
        
        Args:
            forward_fn: Model forward function.
            input_ids: Input token IDs (batch, seq).
            attention_mask: Attention mask (batch, seq).
            pad_token_id: Token ID for padding.
            **extra_kwargs: Additional arguments passed to forward_fn.
            
        Returns:
            Output tensors with padding removed.
        """
        if not self.is_enabled:
            self._direct_calls += 1
            return forward_fn(input_ids=input_ids, attention_mask=attention_mask, **extra_kwargs)
        
        original_batch, original_seq = input_ids.shape
        
        # Find bucket
        bucket = self.bucketer.find_bucket(original_batch, original_seq)
        
        if bucket is None:
            # No suitable bucket - fall back to direct execution
            self._direct_calls += 1
            logger.debug(
                f"No bucket for shape ({original_batch}, {original_seq}) - direct execution"
            )
            return forward_fn(input_ids=input_ids, attention_mask=attention_mask, **extra_kwargs)
        
        # Pad inputs to bucket shape
        padded_ids, padded_mask = self.bucketer.pad_tensors(
            bucket, input_ids, attention_mask, pad_token_id
        )
        
        # Check if graph should be captured for this bucket
        if not self.bucketer.should_capture_graph(bucket):
            # Not enough uses yet - run directly to warm up
            self._direct_calls += 1
            outputs = forward_fn(input_ids=padded_ids, attention_mask=padded_mask, **extra_kwargs)
        else:
            # Use graph cache
            self._graph_calls += 1
            outputs = self.graph_cache.run_with_graph(
                bucket.shape_key,
                forward_fn,
                input_ids=padded_ids,
                attention_mask=padded_mask,
                **extra_kwargs,
            )
        
        # Unpad outputs
        return self.bucketer.unpad_outputs(outputs, original_batch, original_seq)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics from graph cache and bucketer."""
        return {
            "graph_cache": self.graph_cache.get_stats(),
            "bucketer": self.bucketer.get_stats(),
            "direct_calls": self._direct_calls,
            "graph_calls": self._graph_calls,
            "graph_call_ratio": self._graph_calls / max(self._direct_calls + self._graph_calls, 1),
        }


class GraphAwareTraining:
    """
    CUDA graph-accelerated training with shape bucketing and static buffers.

    Captures forward + loss + backward for repeated (batch, seq) shapes.
    Optimizer step remains outside the graph.
    """

    def __init__(
        self,
        graph_cache: GraphCache,
        bucket_config: Optional[ShapeBucketConfig] = None,
        *,
        pad_token_id: int = 0,
        ignore_index: int = -1,
    ):
        self.graph_cache = graph_cache
        self.bucketer = ShapeBucketer(bucket_config)
        self.pad_token_id = int(pad_token_id)
        self.ignore_index = int(ignore_index)
        self._label_keys: Optional[list[str]] = None
        self._accum_steps: Optional[int] = None
        self._direct_calls = 0
        self._graph_calls = 0
        self._fallback_calls = 0

    @property
    def is_enabled(self) -> bool:
        return self.graph_cache.is_enabled

    def _pad_labels(
        self,
        labels: tuple[Tensor, ...],
        target_batch: int,
    ) -> tuple[Tensor, ...]:
        if not labels:
            return labels
        current_batch = labels[0].size(0)
        if current_batch >= target_batch:
            return labels
        pad = target_batch - current_batch
        padded = []
        for label in labels:
            padded.append(
                torch.nn.functional.pad(label, (0, pad), value=self.ignore_index)
            )
        return tuple(padded)

    def _train_step_fn(
        self,
        *,
        model: Any,
        head: Any,
        accum_steps: int,
        label_keys: list[str],
    ) -> Callable[..., Dict[str, Tensor]]:
        ignore_index = self.ignore_index

        def _step(input_ids: Tensor, attention_mask: Tensor, **label_inputs: Tensor):
            labels = tuple(label_inputs[key] for key in label_keys)
            cls_embedding = model.forward_cls(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = head.forward_compiled(cls_embedding)
            loss, valid_flag = head.compute_loss_compiled(
                logits,
                labels,
                ignore_index=ignore_index,
            )
            (loss / accum_steps).backward()
            return {"loss": loss, "valid": valid_flag}

        return _step

    def run(
        self,
        *,
        model: Any,
        head: Any,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: tuple[Tensor, ...] | Dict[str, Tensor],
        accum_steps: int,
        pad_token_id: Optional[int] = None,
        pre_capture_hook: Optional[Callable[[], bool]] = None,
    ) -> tuple[Tensor, Tensor, bool]:
        if self._accum_steps is None:
            self._accum_steps = int(accum_steps)
        elif int(accum_steps) != self._accum_steps:
            raise RuntimeError("accum_steps changed during CUDA graph training")

        if isinstance(labels, dict):
            label_keys = list(getattr(head, "task_order", labels.keys()))
            if self._label_keys is None:
                self._label_keys = label_keys
            elif self._label_keys != label_keys:
                raise RuntimeError("Label order changed during CUDA graph training")
            try:
                labels_tuple = tuple(labels[key] for key in self._label_keys)
            except KeyError as exc:
                raise RuntimeError(f"Missing label key for CUDA graph training: {exc}") from exc
        else:
            labels_tuple = labels
            if self._label_keys is None:
                if hasattr(head, "task_order"):
                    self._label_keys = list(head.task_order)
                else:
                    self._label_keys = [f"label_{i}" for i in range(len(labels_tuple))]
            if len(labels_tuple) != len(self._label_keys):
                raise RuntimeError("Label count mismatch during CUDA graph training")

        if not self.is_enabled:
            self._direct_calls += 1
            step_fn = self._train_step_fn(
                model=model,
                head=head,
                accum_steps=self._accum_steps,
                label_keys=self._label_keys,
            )
            outputs = step_fn(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **{k: v for k, v in zip(self._label_keys, labels_tuple)},
            )
            return outputs["loss"], outputs["valid"], False

        original_batch, original_seq = input_ids.shape
        bucket = self.bucketer.find_bucket(original_batch, original_seq)
        if bucket is None:
            self._fallback_calls += 1
            step_fn = self._train_step_fn(
                model=model,
                head=head,
                accum_steps=self._accum_steps,
                label_keys=self._label_keys,
            )
            outputs = step_fn(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **{k: v for k, v in zip(self._label_keys, labels_tuple)},
            )
            return outputs["loss"], outputs["valid"], False

        pad_token_id = (
            int(pad_token_id)
            if pad_token_id is not None
            else int(self.pad_token_id)
        )
        padded_ids, padded_mask = self.bucketer.pad_tensors(
            bucket,
            input_ids,
            attention_mask,
            pad_token_id=pad_token_id,
        )
        padded_labels = self._pad_labels(labels_tuple, bucket.batch_size)

        label_inputs = {k: v for k, v in zip(self._label_keys, padded_labels)}
        step_fn = self._train_step_fn(
            model=model,
            head=head,
            accum_steps=self._accum_steps,
            label_keys=self._label_keys,
        )

        if not self.bucketer.should_capture_graph(bucket):
            self._direct_calls += 1
            outputs = step_fn(
                input_ids=padded_ids,
                attention_mask=padded_mask,
                **label_inputs,
            )
            return outputs["loss"], outputs["valid"], False

        if bucket.shape_key not in self.graph_cache and pre_capture_hook is not None:
            if not bool(pre_capture_hook()):
                self._fallback_calls += 1
                outputs = step_fn(
                    input_ids=padded_ids,
                    attention_mask=padded_mask,
                    **label_inputs,
                )
                return outputs["loss"], outputs["valid"], False
        self._graph_calls += 1
        outputs = self.graph_cache.run_with_graph(
            bucket.shape_key,
            step_fn,
            input_ids=padded_ids,
            attention_mask=padded_mask,
            **label_inputs,
        )
        return outputs["loss"], outputs["valid"], True

    def get_stats(self) -> Dict[str, Any]:
        return {
            "graph_cache": self.graph_cache.get_stats(),
            "bucketer": self.bucketer.get_stats(),
            "direct_calls": self._direct_calls,
            "graph_calls": self._graph_calls,
            "fallback_calls": self._fallback_calls,
            "graph_call_ratio": self._graph_calls / max(self._direct_calls + self._graph_calls, 1),
        }


def create_graph_aware_inference_from_config(config: Dict[str, Any]) -> GraphAwareInference:
    """
    Create GraphAwareInference from pipeline configuration.
    
    Args:
        config: Pipeline config with execution.enable_cuda_graphs, cuda_graph_* settings.
        
    Returns:
        Configured GraphAwareInference instance.
    """
    execution_cfg = config.get("execution", {})
    
    # Create graph cache
    graph_cache = create_graph_cache_from_config(config)
    
    # Extract bucket config
    cuda_graph_cfg = execution_cfg.get("cuda_graphs", {})
    
    bucket_config = ShapeBucketConfig(
        batch_buckets=cuda_graph_cfg.get(
            "batch_buckets", [1, 2, 4, 8, 16, 32, 64]
        ),
        seq_len_buckets=cuda_graph_cfg.get(
            "seq_len_buckets", [128, 256, 384, 512, 768, 1024]
        ),
        max_batch_size=int(cuda_graph_cfg.get("max_batch_size", 64)),
        max_seq_len=int(cuda_graph_cfg.get("max_seq_len", 1024)),
        adaptive_buckets=bool(cuda_graph_cfg.get("adaptive_buckets", False)),
        min_bucket_usage=int(cuda_graph_cfg.get("min_bucket_usage", 2)),
    )
    
    return GraphAwareInference(graph_cache, bucket_config)


def create_graph_aware_training_from_config(
    config: Dict[str, Any],
    *,
    default_max_seq_len: int = 512,
) -> GraphAwareTraining:
    graph_cfg = config.get("cuda_graph_training", {})
    graph_cache = GraphCache(
        GraphCacheConfig(
            enabled=bool(graph_cfg.get("enabled", True)),
            warmup_iterations=int(graph_cfg.get("warmup_iterations", 0)),
            max_cached_graphs=int(graph_cfg.get("max_cached_graphs", 16)),
            capture_pool_size_mb=int(graph_cfg.get("capture_pool_size_mb", 256)),
            use_cuda_graph_memory_pool=bool(
                graph_cfg.get("use_cuda_graph_memory_pool", True)
            ),
        )
    )
    bucket_config = ShapeBucketConfig(
        batch_buckets=graph_cfg.get("batch_buckets", [8, 16, 32, 64]),
        seq_len_buckets=graph_cfg.get("seq_len_buckets", [128, 256, 384, 512]),
        max_batch_size=int(graph_cfg.get("max_batch_size", 64)),
        max_seq_len=int(graph_cfg.get("max_seq_len", default_max_seq_len)),
        adaptive_buckets=bool(graph_cfg.get("adaptive_buckets", False)),
        min_bucket_usage=int(graph_cfg.get("min_bucket_usage", 2)),
    )
    return GraphAwareTraining(
        graph_cache,
        bucket_config,
        pad_token_id=int(graph_cfg.get("pad_token_id", 0)),
        ignore_index=int(graph_cfg.get("ignore_index", -1)),
    )
