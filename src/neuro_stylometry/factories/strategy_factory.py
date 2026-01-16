"""
Strategy Factory for Hardware-Aware Strategy Selection.

Automatically selects and instantiates the appropriate strategy
based on detected hardware profile.

Implements: phaseA-D_implementation_plan.md Section 8.1
"""

import logging
from typing import Optional

from ..hardware_ops.detection import HardwareDetector, ProfileType
from ..config import load_pipeline_config
from ..pollution_guard.strategies.base import PollutionFilterStrategy
from ..pollution_guard.strategies.laptop import LaptopFilterStrategy
from ..pollution_guard.strategies.hpc import HPCFilterStrategy

logger = logging.getLogger(__name__)


class StrategyFactory:
    """
    Factory for creating hardware-appropriate execution strategies.
    
    Implements the Factory pattern for dual-mode execution.
    """
    
    @staticmethod
    def create_filter_strategy(
        profile_type: Optional[ProfileType | str] = None,
    ) -> PollutionFilterStrategy:
        """
        Create pollution filter strategy based on hardware profile.
        
        Args:
            profile_type: Explicit profile type (auto-detected if None).
            
        Returns:
            Concrete PollutionFilterStrategy instance.
        """
        if profile_type is None:
            # Auto-detect hardware profile
            profile = HardwareDetector.detect()
            profile_type = profile.profile_type
        elif isinstance(profile_type, str):
            normalized = profile_type.lower()
            if normalized not in {"hpc", "laptop"}:
                raise ValueError(
                    f"Unsupported profile_type '{profile_type}' (expected 'hpc'|'laptop')"
                )
            profile_type = ProfileType.HPC if normalized == "hpc" else ProfileType.LAPTOP
        
        logger.info(f"Creating filter strategy for profile: {profile_type.value}")
        
        if profile_type == ProfileType.HPC:
            return HPCFilterStrategy()
        else:  # ProfileType.LAPTOP
            return LaptopFilterStrategy()
    
    @staticmethod
    def get_recommended_config(profile_type: ProfileType) -> dict:
        """
        Get recommended configuration for a hardware profile.
        
        Args:
            profile_type: Hardware profile type.
            
        Returns:
            Configuration dictionary with recommended settings.
        """
        mode = "hpc" if profile_type == ProfileType.HPC else "laptop"
        return load_pipeline_config(mode=mode)
