"""
Pollution Guard Module.

Exports pollution detection and mitigation components.
"""

from .gliner_detector import GLiNERDetector, SOBRTaxonomy, EntityWidthConstraints
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

__all__ = [
    # GLiNER Detection
    "GLiNERDetector",
    "SOBRTaxonomy",
    "EntityWidthConstraints",
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
]
