"""
Hardware Detection Module for Dual-Mode Execution.

Implements FR-01: Auto-detection of HPC vs. Laptop profiles based on:
- CPU core count (>= 32 cores indicates HPC)
- GPU VRAM (>= 40GB indicates A100/H100)
- Available system RAM (>= 256GB indicates HPC)
"""

import logging
import platform
import psutil
import torch
from dataclasses import dataclass
from typing import Literal, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ProfileType(str, Enum):
    """Hardware profile types."""
    HPC = "HPC"
    LAPTOP = "LAPTOP"


@dataclass
class HardwareProfile:
    """
    Hardware capability profile for dual-mode execution.
    
    Attributes:
        profile_type: HPC or LAPTOP
        cpu_cores: Total CPU cores (physical + logical)
        ram_gb: Total system RAM in GB
        has_cuda: Whether CUDA is available
        gpu_name: GPU model name (if available)
        gpu_vram_gb: GPU VRAM in GB (if available)
        compute_capability: CUDA compute capability (if available)
    """
    profile_type: ProfileType
    cpu_cores: int
    ram_gb: float
    has_cuda: bool
    gpu_name: Optional[str] = None
    gpu_vram_gb: Optional[float] = None
    compute_capability: Optional[tuple] = None
    
    def __str__(self) -> str:
        lines = [
            f"Hardware Profile: {self.profile_type.value}",
            f"  CPU Cores: {self.cpu_cores}",
            f"  RAM: {self.ram_gb:.1f} GB",
            f"  CUDA Available: {self.has_cuda}",
        ]
        if self.has_cuda:
            lines.extend([
                f"  GPU: {self.gpu_name}",
                f"  VRAM: {self.gpu_vram_gb:.1f} GB",
                f"  Compute Capability: {self.compute_capability}",
            ])
        return "\n".join(lines)


class HardwareDetector:
    """
    Hardware detection and profiling for dual-mode execution.
    
    Implements the decision logic:
    - HPC: >= 32 CPU cores OR >= 40GB VRAM OR >= 256GB RAM
    - LAPTOP: Otherwise
    """
    
    # Detection thresholds (as per phaseA-D_implementation_plan.md Section 1.3)
    HPC_CPU_CORES = 32
    HPC_VRAM_GB = 40.0
    HPC_RAM_GB = 256.0
    
    @classmethod
    def detect(cls) -> HardwareProfile:
        """
        Auto-detect hardware profile.
        
        Returns:
            HardwareProfile with detected capabilities.
        """
        # CPU detection
        cpu_cores = psutil.cpu_count(logical=True)
        
        # RAM detection
        ram_bytes = psutil.virtual_memory().total
        ram_gb = ram_bytes / (1024 ** 3)
        
        # GPU detection
        has_cuda = torch.cuda.is_available()
        gpu_name = None
        gpu_vram_gb = None
        compute_capability = None
        
        if has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_vram_bytes = torch.cuda.get_device_properties(0).total_memory
            gpu_vram_gb = gpu_vram_bytes / (1024 ** 3)
            compute_capability = torch.cuda.get_device_capability(0)
        
        # Profile type decision (Implements FR-01)
        is_hpc = (
            cpu_cores >= cls.HPC_CPU_CORES
            or (gpu_vram_gb is not None and gpu_vram_gb >= cls.HPC_VRAM_GB)
            or ram_gb >= cls.HPC_RAM_GB
        )
        profile_type = ProfileType.HPC if is_hpc else ProfileType.LAPTOP
        
        profile = HardwareProfile(
            profile_type=profile_type,
            cpu_cores=cpu_cores,
            ram_gb=ram_gb,
            has_cuda=has_cuda,
            gpu_name=gpu_name,
            gpu_vram_gb=gpu_vram_gb,
            compute_capability=compute_capability,
        )
        
        logger.info(f"Hardware detection complete:\n{profile}")
        return profile
    
    @classmethod
    def get_device(cls) -> torch.device:
        """
        Get recommended PyTorch device.
        
        Returns:
            torch.device for computation.
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            logger.warning("CUDA not available, falling back to CPU")
            return torch.device("cpu")
