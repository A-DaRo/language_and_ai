# src/neuro_stylometry/config.py

from dataclasses import dataclass
from typing import Optional, Literal
from pathlib import Path
from omegaconf import OmegaConf

@dataclass
class PipelineConfig:
    """Pipeline configuration dataclass."""
    pass  # To be populated from OmegaConf

def load_config(
    config_path: Path,
    overrides: Optional[dict] = None,
    mode: Optional[Literal["hpc", "laptop"]] = None
) -> PipelineConfig:
    """
    Load and merge configuration from YAML files.
    
    Args:
        config_path: Path to experiment config or base config
        overrides: Dictionary of runtime overrides
        mode: Explicit mode selection (auto-detected if None)
    
    Returns:
        Validated PipelineConfig dataclass
    """
    # Load base config
    base_cfg = OmegaConf.load(Path(__file__).parent.parent.parent / "conf/base/pipeline.yaml")
    
    # Detect or use specified mode
    if mode is None:
        from .hardware_ops.detection import HardwareDetector
        profile = HardwareDetector.detect()
        mode = "hpc" if profile.profile_type == "HPC" else "laptop"
    
    # Load mode-specific overrides
    mode_cfg = OmegaConf.load(Path(__file__).parent.parent.parent / f"conf/{mode}/pipeline.yaml")
    
    # Load experiment config if provided
    experiment_cfg = OmegaConf.load(config_path) if config_path.exists() else OmegaConf.create()
    
    # Merge configurations (later overrides earlier)
    merged = OmegaConf.merge(base_cfg, mode_cfg, experiment_cfg)
    
    # Apply runtime overrides
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.create(overrides))
    
    # Resolve interpolations and convert to dataclass
    OmegaConf.resolve(merged)
    return OmegaConf.to_object(merged, PipelineConfig)
