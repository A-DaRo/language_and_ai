# src/neuro_stylometry/config.py
"""
Configuration loading and validation for the neuro-stylometry pipeline.

This module provides:
1. YAML-based configuration loading with mode-specific overrides
2. Pydantic-based schema validation for type safety
3. Configuration authority is strictly YAML-based (no runtime overrides)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

from omegaconf import OmegaConf


# ==============================================================================
# Configuration Schema Definitions (Pydantic-style validation)
# ==============================================================================

class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


@dataclass
class GLiNERConfig:
    """GLiNER detector configuration."""
    model: str = "urchade/gliner_large-v2.1"
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    batch_size: int = 32
    confidence_threshold: float = 0.85
    taxonomy_path: str = "conf/base/gliner_taxonomy.yaml"
    compute_explicit_recall: bool = True
    explicit_recall_threshold: float = 0.95
    # Bi-encoder enforcement and word-limit settings
    require_bi_encoder: bool = False
    gliner_max_words: int = 512
    tokens_per_word_ratio: float = 1.3
    # Nested blocks (v2.0 batching + chunking). Kept as dicts for forward-compat.
    batch_inference: Dict[str, Any] = field(default_factory=dict)
    chunking: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.device not in ("auto", "cpu", "cuda"):
            raise ConfigValidationError(
                f"gliner.device must be 'auto', 'cpu', or 'cuda', got '{self.device}'"
            )
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ConfigValidationError(
                f"gliner.confidence_threshold must be in [0, 1], got {self.confidence_threshold}"
            )
        if self.batch_size <= 0:
            raise ConfigValidationError(
                f"gliner.batch_size must be positive, got {self.batch_size}"
            )
        if not isinstance(self.batch_inference, dict):
            raise ConfigValidationError(
                f"gliner.batch_inference must be a mapping, got {type(self.batch_inference).__name__}"
            )
        if not isinstance(self.chunking, dict):
            raise ConfigValidationError(
                f"gliner.chunking must be a mapping, got {type(self.chunking).__name__}"
            )
        if self.gliner_max_words <= 0:
            raise ConfigValidationError(
                f"gliner.gliner_max_words must be positive, got {self.gliner_max_words}"
            )
        if self.tokens_per_word_ratio <= 0:
            raise ConfigValidationError(
                f"gliner.tokens_per_word_ratio must be positive, got {self.tokens_per_word_ratio}"
            )


@dataclass
class EncoderConfig:
    """Encoder configuration."""
    model: str = "roberta-base"
    hidden_dim: int = 768
    max_length: int = 512
    batch_size: int = 32
    device: str = "auto"
    output_device: Optional[str] = None
    
    def __post_init__(self):
        if self.device not in ("auto", "cpu", "cuda"):
            raise ConfigValidationError(
                f"encoder.device must be 'auto', 'cpu', or 'cuda', got '{self.device}'"
            )
        if self.batch_size <= 0:
            raise ConfigValidationError(
                f"encoder.batch_size must be positive, got {self.batch_size}"
            )
        if self.max_length <= 0:
            raise ConfigValidationError(
                f"encoder.max_length must be positive, got {self.max_length}"
            )


@dataclass
class LEACEConfig:
    """LEACE projection configuration."""
    device: str = "cpu"
    force_cpu: bool = True
    regularization: float = 1e-5
    batch_size: int = 100
    compute_dtype: str = "float64"
    
    def __post_init__(self):
        if self.device not in ("cpu", "cuda"):
            raise ConfigValidationError(
                f"leace.device must be 'cpu' or 'cuda', got '{self.device}'"
            )
        if self.regularization < 1e-10:
            raise ConfigValidationError(
                f"leace.regularization must be >= 1e-10, got {self.regularization}"
            )
        if self.batch_size <= 0:
            raise ConfigValidationError(
                f"leace.batch_size must be positive, got {self.batch_size}"
            )
        if self.compute_dtype not in ("float32", "float64"):
            raise ConfigValidationError(
                f"leace.compute_dtype must be 'float32' or 'float64', got '{self.compute_dtype}'"
            )


@dataclass
class SubsetConfig:
    """Dataset subset configuration."""
    enabled: bool = False
    size: Optional[int] = None
    
    def __post_init__(self):
        if self.enabled and self.size is not None and self.size <= 0:
            raise ConfigValidationError(
                f"subset.size must be positive when enabled, got {self.size}"
            )


@dataclass
class ProbeConfig:
    """Probe/amnesic drop configuration."""
    compute_amnesic_drop: bool = True
    amnesic_drop_threshold: float = 0.30
    train_split: float = 0.8
    max_samples: Optional[int] = 20000
    
    def __post_init__(self):
        if not 0.0 < self.train_split < 1.0:
            raise ConfigValidationError(
                f"probe.train_split must be in (0, 1), got {self.train_split}"
            )
        if not 0.0 <= self.amnesic_drop_threshold <= 1.0:
            raise ConfigValidationError(
                f"probe.amnesic_drop_threshold must be in [0, 1], got {self.amnesic_drop_threshold}"
            )


@dataclass
class QualityConfig:
    """Quality enforcement configuration."""
    enforce_thresholds: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    wandb_enabled: bool = False


# ==============================================================================
# Validation Functions
# ==============================================================================

def validate_pipeline_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate pipeline configuration dict against schema.
    
    Args:
        config: Configuration dictionary to validate.
        
    Returns:
        Validated configuration dictionary.
        
    Raises:
        ConfigValidationError: If validation fails.
    """
    errors = []
    
    # Validate seed
    seed = config.get("seed", 42)
    if not isinstance(seed, int) or seed < 0:
        errors.append(f"seed must be a non-negative integer, got {seed}")
    
    # Validate GLiNER config
    try:
        gliner_cfg = config.get("gliner", {})
        GLiNERConfig(**gliner_cfg)
    except (TypeError, ConfigValidationError) as e:
        errors.append(f"gliner: {e}")
    
    # Validate encoder config
    try:
        encoder_cfg = config.get("encoder", {})
        EncoderConfig(**encoder_cfg)
    except (TypeError, ConfigValidationError) as e:
        errors.append(f"encoder: {e}")
    
    # Validate LEACE config
    try:
        leace_cfg = config.get("leace", {})
        LEACEConfig(**leace_cfg)
    except (TypeError, ConfigValidationError) as e:
        errors.append(f"leace: {e}")
    
    # Validate subset config
    try:
        subset_cfg = config.get("subset", {})
        SubsetConfig(**subset_cfg)
    except (TypeError, ConfigValidationError) as e:
        errors.append(f"subset: {e}")
    
    # Validate probe config
    try:
        probe_cfg = config.get("probe", {})
        ProbeConfig(**probe_cfg)
    except (TypeError, ConfigValidationError) as e:
        errors.append(f"probe: {e}")
    
    if errors:
        raise ConfigValidationError(
            "Invalid pipeline configuration:\n" + "\n".join(f"  - {e}" for e in errors)
        )
    
    return config


# ==============================================================================
# Configuration Loading
# ==============================================================================

def load_pipeline_config(
    *,
    mode: Literal["auto", "laptop", "hpc"] = "auto",
    experiment_config_path: Optional[Path] = None,
    validate: bool = True,
) -> Dict[str, Any]:
    """Load Phase A pipeline configuration from YAML.

    Configuration authority is *strictly YAML-based*.

    - If `experiment_config_path` is provided and exists, it is merged on top of
      base + mode configs.
    - No runtime overrides are supported (by design). The caller must supply
      YAML if they want different parameters.

    Args:
        mode: "auto" (hardware-detected), "laptop", or "hpc".
        experiment_config_path: Optional experiment YAML to merge.
        validate: Whether to validate the configuration (default True).

    Returns:
        A resolved Python dict.
        
    Raises:
        ConfigValidationError: If validation is enabled and config is invalid.
    """
    repo_root = Path(__file__).resolve().parents[2]
    base_path = repo_root / "conf" / "base" / "pipeline.yaml"

    if mode == "auto":
        from .hardware_ops.detection import HardwareDetector

        profile = HardwareDetector.detect()
        mode = "hpc" if profile.profile_type.value.lower() == "hpc" else "laptop"

    mode_path = repo_root / "conf" / mode / "pipeline.yaml"

    base_cfg = OmegaConf.load(base_path)
    mode_cfg = OmegaConf.load(mode_path)

    experiment_cfg = OmegaConf.create()
    if experiment_config_path is not None:
        path = Path(experiment_config_path)
        if path.exists():
            experiment_cfg = OmegaConf.load(path)

    merged = OmegaConf.merge(base_cfg, mode_cfg, experiment_cfg)
    merged = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(merged, dict):
        raise TypeError("Loaded config did not resolve to a mapping")
    
    # Validate if requested
    if validate:
        validate_pipeline_config(merged)
    
    return merged
