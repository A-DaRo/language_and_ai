"""
Pollution Guard Module.

Exports pollution detection and mitigation components.

Autotuning Integration (v2.2):
- Both LaptopFilterStrategy and HPCFilterStrategy use RuntimeController
- DynamicBatchIterator for adaptive token-budget batching
- OOM protection with automatic retry and budget slashing
- Configure via execution.autotuning section in pipeline YAML
"""

from .gliner_detector import (
    GLiNERDetector,
    SOBRTaxonomy,
    EntityWidthConstraints,
    BatchInferenceConfig,
    auto_compute_bucket_count,
    compute_chunk_length_buckets,
)
from .semantic_chunker import (
    SemanticChunker,
    BudgetConfig,
    ChunkInfo,
    project_entity_offsets,
    deduplicate_entities,
)
from .masker import SpanMasker, MaskingResult
from .embedder import FrozenEmbedder
from .leace import LEACEComputer, CovarianceStats
from .probe import LinearProbe, compute_amnesic_drop

from .strategies.base import PollutionFilterStrategy
from .strategies.laptop import LaptopFilterStrategy
from .strategies.hpc import HPCFilterStrategy

# Global sort utilities with autotuning support
from .global_sort import (
    FlattenedChunks,
    flatten_chunks,
    gather_results,
    create_sorted_batches,
    DynamicBatchIterator,
    create_dynamic_batches,
)

__all__ = [
    # GLiNER Detection
    "GLiNERDetector",
    "SOBRTaxonomy",
    "EntityWidthConstraints",
    # Batch Inference Optimization (v2.0)
    "BatchInferenceConfig",
    "auto_compute_bucket_count",
    "compute_chunk_length_buckets",
    # Semantic Chunking
    "SemanticChunker",
    "BudgetConfig",
    "ChunkInfo",
    "project_entity_offsets",
    "deduplicate_entities",
    # Masking
    "SpanMasker",
    "MaskingResult",
    # Embeddings & LEACE
    "FrozenEmbedder",
    "LEACEComputer",
    "CovarianceStats",
    "LinearProbe",
    "compute_amnesic_drop",
    # Strategies
    "PollutionFilterStrategy",
    "LaptopFilterStrategy",
    "HPCFilterStrategy",
    # Global Sort & Autotuning
    "FlattenedChunks",
    "flatten_chunks",
    "gather_results",
    "create_sorted_batches",
    "DynamicBatchIterator",
    "create_dynamic_batches",
]
