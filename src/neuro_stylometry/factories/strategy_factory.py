"""
Strategy Factory for Hardware-Aware Strategy Selection.

Automatically selects and instantiates the appropriate strategy
based on detected hardware profile.

Implements: phaseA-D_implementation_plan.md Section 8.1
"""

import logging
from typing import Optional

from ..hardware_ops.detection import HardwareDetector, ProfileType
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
        profile_type: Optional[ProfileType] = None,
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
        if profile_type == ProfileType.HPC:
            return {
                "gliner_threshold": 0.85,
                "gliner_batch_size": 16,
                "embedder_batch_size": 32,
                "leace_regularization": 1e-5,
                "use_full_batch_leace": True,
            }
        else:  # LAPTOP
            return {
                "gliner_threshold": 0.85,
                "gliner_batch_size": 2,
                "embedder_batch_size": 4,
                "leace_regularization": 1e-5,
                "leace_batch_size": 50,
                "max_samples": 1000,  # Use subset
            }
