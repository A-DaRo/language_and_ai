"""
Abstract Strategy Interface for Phase A Pollution Filtering.

Defines the contract for dual-mode execution strategies.

Implements: phaseA-D_implementation_plan.md Section 7.1-7.2
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from pathlib import Path
import torch
import pyarrow as pa


class PollutionFilterStrategy(ABC):
    """
    Abstract strategy for Phase A pollution filtering.
    
    Concrete implementations:
    - LaptopFilterStrategy: CPU/Low-VRAM with batch accumulation
    - HPCFilterStrategy: GPU with full-batch computation
    """
    
    @abstractmethod
    def execute(
        self,
        input_dataset_path: Path,
        output_dataset_path: Path,
        projection_matrix_path: Path,
        pollution_logs_path: Path,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute Phase A pollution filtering pipeline.
        
        Args:
            input_dataset_path: Path to input Arrow dataset.
            output_dataset_path: Path to output cleaned dataset.
            projection_matrix_path: Path to save projection matrix.
            pollution_logs_path: Path to save pollution logs.
            config: Configuration dictionary.
            
        Returns:
            Execution metadata (timing, stats, etc.).
        """
        pass
    
    @abstractmethod
    def get_device(self) -> torch.device:
        """Get PyTorch device for computation."""
        pass
    
    @abstractmethod
    def get_batch_size(self) -> int:
        """Get batch size for this strategy."""
        pass
