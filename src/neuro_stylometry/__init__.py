"""
Neuro-Symbolic Stylometry Pipeline.

Phase A: Pollution Detection and Mitigation
- GLiNER-based symbolic pollution detection
- LEACE geometric projection
- Dual-mode execution (Laptop/HPC)
"""

__version__ = "0.1.0"

from .phase_a_pipeline import PhaseAPipeline, PhaseArtifacts
from .hardware_ops.detection import HardwareDetector, HardwareProfile, ProfileType
from .factories.strategy_factory import StrategyFactory

__all__ = [
    "PhaseAPipeline",
    "PhaseArtifacts",
    "HardwareDetector",
    "HardwareProfile",
    "ProfileType",
    "StrategyFactory",
]
