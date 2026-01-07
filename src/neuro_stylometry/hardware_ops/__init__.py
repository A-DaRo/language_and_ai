"""
Hardware Operations module - Hardware-aware execution.

Provides hardware detection, CUDA graph caching, torch.compile integration,
sequence packing, GPU span filtering, and async prefetching for optimized
inference across laptop and HPC environments.
"""

from .detection import HardwareDetector, HardwareProfile, ProfileType
from .cuda_graphs import (
    GraphCache,
    GraphCacheConfig,
    CapturedGraph,
    CompileConfig,
    compile_model_if_enabled,
    create_graph_cache_from_config,
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

__all__ = [
    # Detection
    "HardwareDetector",
    "HardwareProfile", 
    "ProfileType",
    # CUDA Graphs & Compile
    "GraphCache",
    "GraphCacheConfig",
    "CapturedGraph",
    "CompileConfig",
    "compile_model_if_enabled",
    "create_graph_cache_from_config",
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
]
