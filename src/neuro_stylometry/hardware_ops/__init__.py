"""
Hardware Operations module - Hardware-aware execution.

Provides hardware detection, CUDA graph caching, torch.compile integration,
sequence packing, GPU span filtering, async prefetching, and self-optimizing
runtime for optimized inference across laptop and HPC environments.
"""

from .detection import HardwareDetector, HardwareProfile, ProfileType
from .cuda_graphs import (
    GraphCache,
    GraphCacheConfig,
    CapturedGraph,
    GraphAwareTraining,
    CompileConfig,
    compile_model_if_enabled,
    create_graph_cache_from_config,
    create_graph_aware_training_from_config,
    create_compile_config_from_dict,
)
from .sequence_packing import (
    SequencePacker,
    SequencePackerConfig,
    PackedSequence,
    GPUSpanFilter,
    GPUSpanFilterConfig,
    create_span_filter_from_config,
    create_sequence_packer_from_config,
)
from .async_prefetch import (
    AsyncPrefetchPipeline,
    AsyncPrefetchConfig,
    PrefetchBatch,
    SyncPrefetchPipeline,
    create_prefetch_config_from_dict,
    create_prefetch_pipeline,
)
from .runtime import (
    RuntimeController,
    RuntimeConfig,
    RuntimeMetrics,
    RuntimeState,
    RuntimeSnapshot,
    get_gpu_memory_stats,
)
from .oom_guard import (
    OOMEvent,
    OOMRecoveryError,
    execute_with_oom_protection,
    oom_guarded_context,
    oom_protected,
    is_cuda_oom,
    clear_cuda_cache,
)
from .telemetry import (
    CUDATimer,
    TelemetryCollector,
    TelemetrySummary,
    BatchMetrics,
    timed_cuda_block,
    cuda_timed_section,
    create_runtime_metrics,
)

__all__ = [
    # Detection
    "HardwareDetector",
    "HardwareProfile", 
    "ProfileType",
    # CUDA Graphs & Compile
    "GraphCache",
    "GraphCacheConfig",
    "CapturedGraph",
    "GraphAwareTraining",
    "CompileConfig",
    "compile_model_if_enabled",
    "create_graph_cache_from_config",
    "create_graph_aware_training_from_config",
    "create_compile_config_from_dict",
    # Sequence Packing & GPU Filtering
    "SequencePacker",
    "SequencePackerConfig",
    "PackedSequence",
    "GPUSpanFilter",
    "GPUSpanFilterConfig",
    "create_span_filter_from_config",
    "create_sequence_packer_from_config",
    # Async Prefetch
    "AsyncPrefetchPipeline",
    "AsyncPrefetchConfig",
    "PrefetchBatch",
    "SyncPrefetchPipeline",
    "create_prefetch_config_from_dict",
    "create_prefetch_pipeline",
    # Runtime Controller (Self-Optimizing)
    "RuntimeController",
    "RuntimeConfig",
    "RuntimeMetrics",
    "RuntimeState",
    "RuntimeSnapshot",
    "get_gpu_memory_stats",
    # OOM Protection
    "OOMEvent",
    "OOMRecoveryError",
    "execute_with_oom_protection",
    "oom_guarded_context",
    "oom_protected",
    "is_cuda_oom",
    "clear_cuda_cache",
    # Telemetry
    "CUDATimer",
    "TelemetryCollector",
    "TelemetrySummary",
    "BatchMetrics",
    "timed_cuda_block",
    "cuda_timed_section",
    "create_runtime_metrics",
]
