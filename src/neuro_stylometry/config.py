# src/neuro_stylometry/config.py
"""
Configuration loading and validation for the neuro-stylometry pipeline.

This module provides:
1. YAML-based configuration loading with mode-specific overrides
2. Pydantic-based schema validation for type safety
3. Configuration authority is strictly YAML-based (no runtime overrides)
"""

from __future__ import annotations

import os
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
    device: Optional[str] = "cpu"
    force_cpu: bool = True
    regularization: float = 1e-5
    batch_size: int = 100
    compute_dtype: str = "float64"
    
    def __post_init__(self):
        if self.device is not None and self.device not in ("cpu", "cuda"):
            raise ConfigValidationError(
                f"leace.device must be None, 'cpu' or 'cuda', got '{self.device}'"
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
    backend: str = "auto"
    use_kfold: bool = True
    n_folds: int = 5
    pvalue_threshold: float = 0.05
    torch_lr: float = 0.01
    torch_epochs: int = 100
    torch_batch_size: int = 256
    torch_weight_decay: float = 0.0001
    use_class_weights: bool = True
    benchmark_solvers: bool = False

    def __post_init__(self):
        if self.backend not in ("auto", "sklearn", "cuml", "torch"):
            raise ConfigValidationError(
                f"probe.backend must be 'auto', 'sklearn', 'cuml', or 'torch', got '{self.backend}'"
            )
        if not 0.0 < self.train_split < 1.0:
            raise ConfigValidationError(
                f"probe.train_split must be in (0, 1), got {self.train_split}"
            )
        if self.n_folds < 2:
            raise ConfigValidationError(
                f"probe.n_folds must be at least 2, got {self.n_folds}"
            )
        if self.max_samples is not None and self.max_samples <= 0:
            raise ConfigValidationError(
                f"probe.max_samples must be positive, got {self.max_samples}"
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
class PathsConfig:
    """File path configuration."""
    raw_data: str = "./datasets"
    output: str = "./artifacts"
    phase_a: Dict[str, str] = field(default_factory=dict)


@dataclass
class DataLoaderConfig:
    """DataLoader configuration."""
    batch_size: int = 32
    num_workers: int = 4
    persistent_workers: bool = False
    prefetch_factor: int = 2
    pin_memory: bool = True


@dataclass
class SkipStagesConfig:
    """
    Stage skip configuration for Phase A pipeline.
    
    Enables entry-point and early-exit execution:
    - skip_chunking: Start at inference (assumes post_chunked exists)
    - skip_inference: Start at LEACE (assumes inference results exist)
    - skip_leace: Start at probing (assumes projection_matrix.pt exists)
    - skip_probing: Exit after LEACE (skip metrics/visualizations)
    
    Sequential dependency: Chunking → Inference → LEACE → Probing
    Skipping a later stage implies prerequisites are on disk.
    """
    skip_chunking: bool = False
    skip_inference: bool = False
    skip_leace: bool = False
    skip_probing: bool = False
    
    def __post_init__(self):
        # Validate logical consistency: can't skip LEACE but run probing without projection
        if self.skip_leace and not self.skip_probing:
            # This is allowed - probing can load existing projection matrix
            pass
        # If skipping inference, must also skip chunking (inference depends on chunks)
        if self.skip_inference and not self.skip_chunking:
            self.skip_chunking = True
        # If skipping LEACE, must also skip chunking and inference
        if self.skip_leace:
            self.skip_chunking = True
            self.skip_inference = True


@dataclass
class ExecutionConfig:
    """Advanced execution optimization configuration."""
    staged_execution: Dict[str, Any] = field(default_factory=dict)
    autotuning: Dict[str, Any] = field(default_factory=dict)
    cuda_graphs: Dict[str, Any] = field(default_factory=dict)
    async_storage: Dict[str, Any] = field(default_factory=dict)
    async_prefetch: Dict[str, Any] = field(default_factory=dict)
    gpu_span_filter: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    enable_cuda_graphs: bool = False
    cuda_graph_warmup: int = 3
    cuda_graph_cache_size: int = 16
    enable_torch_compile: bool = False
    torch_compile_mode: str = "reduce-overhead"
    torch_compile_fullgraph: bool = False
    torch_compile_dynamic: bool = False
    torch_compile_backend: str = "inductor"
    sequence_packing: Dict[str, Any] = field(default_factory=dict)
    numa_pinning: bool = False
    # Stage skip configuration for modular execution
    skip_stages: Dict[str, bool] = field(default_factory=dict)
    # When True, missing prerequisites will be bypassed with warnings instead of errors
    force_skip: bool = False
    
    def get_skip_stages_config(self) -> SkipStagesConfig:
        """Get validated SkipStagesConfig from skip_stages dict and top-level flag."""
        return SkipStagesConfig(
            skip_chunking=self.skip_stages.get("skip_chunking", False),
            skip_inference=self.skip_stages.get("skip_inference", False),
            skip_leace=self.skip_stages.get("skip_leace", False),
            skip_probing=self.skip_stages.get("skip_probing", False),
            force_skip=self.skip_stages.get("force_skip", self.force_skip),
        )


@dataclass
class PrecisionConfig:
    """Numerical precision configuration."""
    dtype: str = "float32"
    use_grad_scaler: bool = False


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    enabled: bool = True
    pca_max_samples: int = 1000
    embedding_batch_size: int = 32
    figure_dpi: int = 150
    figure_format: str = "png"
    color_palette: str = "husl"
    plots: Dict[str, bool] = field(default_factory=dict)


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    wandb_enabled: bool = False
    log_gpu_memory: bool = False
    log_batch_timing: bool = False


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
    
    # Validate paths config
    try:
        paths_cfg = config.get("paths", {})
        PathsConfig(**paths_cfg)
    except (TypeError, ConfigValidationError) as e:
        errors.append(f"paths: {e}")
    
    # Validate dataloader config
    try:
        dataloader_cfg = config.get("dataloader", {})
        DataLoaderConfig(**dataloader_cfg)
    except (TypeError, ConfigValidationError) as e:
        errors.append(f"dataloader: {e}")
    
    # Validate execution config
    try:
        execution_cfg = config.get("execution", {})
        ExecutionConfig(**execution_cfg)
    except (TypeError, ConfigValidationError) as e:
        errors.append(f"execution: {e}")
    
    # Validate precision config
    try:
        precision_cfg = config.get("precision", {})
        PrecisionConfig(**precision_cfg)
    except (TypeError, ConfigValidationError) as e:
        errors.append(f"precision: {e}")
    
    # Validate quality config
    try:
        quality_cfg = config.get("quality", {})
        QualityConfig(**quality_cfg)
    except (TypeError, ConfigValidationError) as e:
        errors.append(f"quality: {e}")
    
    # Validate visualization config
    try:
        viz_cfg = config.get("visualization", {})
        VisualizationConfig(**viz_cfg)
    except (TypeError, ConfigValidationError) as e:
        errors.append(f"visualization: {e}")
    
    # Validate logging config
    try:
        logging_cfg = config.get("logging", {})
        LoggingConfig(**logging_cfg)
    except (TypeError, ConfigValidationError) as e:
        errors.append(f"logging: {e}")
    
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
    base_filename: str = "pipeline.yaml",
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
    repo_root = _find_config_root(base_filename)
    base_path = repo_root / "conf" / "base" / base_filename

    if mode == "auto":
        from .hardware_ops.detection import HardwareDetector

        profile = HardwareDetector.detect()
        mode = "hpc" if profile.profile_type.value.lower() == "hpc" else "laptop"

    mode_path = repo_root / "conf" / mode / base_filename

    if not base_path.exists():
        raise FileNotFoundError(
            f"Base config not found: {base_path}. "
            "Set NEURO_STYLOMETRY_CONFIG_ROOT or run from the repo root."
        )

    if not mode_path.exists():
        raise FileNotFoundError(
            f"Mode config not found: {mode_path}. "
            "Set NEURO_STYLOMETRY_CONFIG_ROOT or run from the repo root."
        )

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


def _resolve_config_root_from_hint(hint: Path, config_name: str) -> Optional[Path]:
    candidate = hint / "conf" / "base" / config_name
    if candidate.exists():
        return hint

    candidate = hint / "base" / config_name
    if candidate.exists():
        return hint.parent if hint.name == "conf" else hint

    if hint.name == "base":
        candidate = hint / config_name
        if candidate.exists():
            return hint.parent.parent

    return None


def _find_config_root(config_name: str) -> Path:
    env_hint = os.getenv("NEURO_STYLOMETRY_CONFIG_ROOT") or os.getenv("NEURO_STYLOMETRY_ROOT")
    if env_hint:
        resolved = _resolve_config_root_from_hint(Path(env_hint).expanduser().resolve(), config_name)
        if resolved is not None:
            return resolved

    for parent in Path(__file__).resolve().parents:
        resolved = _resolve_config_root_from_hint(parent, config_name)
        if resolved is not None:
            return resolved

    cwd = Path.cwd()
    for parent in (cwd, *cwd.parents):
        resolved = _resolve_config_root_from_hint(parent, config_name)
        if resolved is not None:
            return resolved

    return cwd


def find_config_root(config_name: str) -> Path:
    """Public helper for resolving config roots (useful for CLI diagnostics)."""
    return _find_config_root(config_name)


def load_phase_d_config(
    *,
    mode: Literal["auto", "laptop", "hpc"] = "laptop",
    experiment_config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Load Phase D training configuration from YAML.

    Merge order: base/phase_d.yaml -> {mode}/phase_d.yaml -> experiment config

    Args:
        mode: Hardware mode ("auto", "laptop", or "hpc").
        experiment_config_path: Optional experiment YAML to merge.

    Returns:
        Resolved configuration dictionary.

    Raises:
        FileNotFoundError: If base Phase D config not found.
    """
    return load_pipeline_config(
        mode=mode,
        experiment_config_path=experiment_config_path,
        validate=False,
        base_filename="phase_d.yaml",
    )
