"""
Pollution Guard Module.

Exports pollution detection and mitigation components.
"""

from .gliner_detector import GLiNERDetector, SOBRTaxonomy, EntityWidthConstraints
from .masker import SpanMasker, MaskingResult
from .embedder import FrozenEmbedder
from .leace import LEACEComputer, CovarianceStats
from .probe import LinearProbe, compute_amnesic_drop

from .strategies.base import PollutionFilterStrategy
from .strategies.laptop import LaptopFilterStrategy
from .strategies.hpc import HPCFilterStrategy

__all__ = [
    "GLiNERDetector",
    "SOBRTaxonomy",
    "EntityWidthConstraints",
    "SpanMasker",
    "MaskingResult",
    "FrozenEmbedder",
    "LEACEComputer",
    "CovarianceStats",
    "LinearProbe",
    "compute_amnesic_drop",
    "PollutionFilterStrategy",
    "LaptopFilterStrategy",
    "HPCFilterStrategy",
]
