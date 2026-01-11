"""
Phase A Pipeline Orchestrator.

Coordinates the complete Phase A execution:
1. GLiNER pollution detection
2. Span masking
3. LEACE projection computation
4. Output artifact generation

Implements: phaseA-D_implementation_plan.md Section 9.1
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .pollution_guard.strategies.base import PollutionFilterStrategy
from .config import load_pipeline_config

from .pollution_guard.gliner_detector import GLiNERDetector
from .pollution_guard.masker import SpanMasker
from .pollution_guard.embedder import FrozenEmbedder
from .pollution_guard.leace import LEACEComputer

logger = logging.getLogger(__name__)


@dataclass
class PhaseArtifacts:
    """
    Phase A output artifacts.
    
    Attributes:
        clean_dataset_path: Path to cleaned dataset with post_masked.
        projection_matrix_path: Path to LEACE projection matrix.
        pollution_logs_path: Path to pollution detection logs.
        metrics_path: Path to phase_a_metrics.json.
        reports_dir: Directory containing visualizations.
        metadata: Execution metadata.
    """
    clean_dataset_path: Path
    projection_matrix_path: Path
    pollution_logs_path: Path
    metrics_path: Optional[Path] = None
    reports_dir: Optional[Path] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PhaseAPipeline:
    """
    Phase A pipeline orchestrator.
    
    Coordinates pollution detection, masking, and LEACE projection
    using a strategy pattern for hardware-aware execution.
    
    Implements: FR-13 (Pipeline Orchestration)
    """
    
    def __init__(
        self,
        strategy: PollutionFilterStrategy,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Phase A pipeline.
        
        Args:
            strategy: Execution strategy (laptop/HPC).
            config: Pipeline configuration.
        """
        self.strategy = strategy
        # Strict YAML config authority:
        # - If caller provides config, we will not apply any overrides.
        # - If config is None, load default YAML config based on hardware mode.
        self.config = config if config is not None else load_pipeline_config(mode="auto")
        self._strict_user_config = config is not None

        # Lazy component cache for notebook-style usage.
        self._detector: Optional[GLiNERDetector] = None
        self._masker: Optional[SpanMasker] = None
        self._embedder: Optional[FrozenEmbedder] = None
        self._leace: Optional[LEACEComputer] = None
        
        logger.info(f"PhaseAPipeline initialized with {strategy.__class__.__name__}")

    @classmethod
    def from_yaml(
        cls,
        *,
        strategy: PollutionFilterStrategy,
        mode: str = "auto",
        experiment_config_path: Optional[Path] = None,
    ) -> "PhaseAPipeline":
        """Create a PhaseAPipeline from YAML configuration.

        This is the preferred construction path when you want strict YAML authority.
        """
        cfg = load_pipeline_config(mode=mode, experiment_config_path=experiment_config_path)
        return cls(strategy=strategy, config=cfg)

    def _cfg_get(self, path: str) -> Any:
        """Get a nested config value by dot-path.

        If the config was supplied explicitly by the caller, missing keys are errors.
        If config was loaded from defaults, missing keys still error because defaults
        are expected to be complete.
        """
        cur: Any = self.config
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                raise KeyError(f"Missing required config key: '{path}'")
            cur = cur[part]
        return cur

    def _resolve_device(self, device_spec: str) -> str:
        """Resolve an explicit device spec from YAML."""
        import torch

        spec = str(device_spec).lower()
        if spec == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if spec in {"cpu", "cuda"}:
            if spec == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("Config requests CUDA but torch.cuda.is_available() is False")
            return spec
        raise ValueError(f"Unsupported device spec '{device_spec}' (expected 'auto'|'cpu'|'cuda')")

    def _load_taxonomy_bundle(self) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Load taxonomy config from YAML path."""
        from omegaconf import OmegaConf

        taxonomy_path = Path(self._cfg_get("gliner.taxonomy_path"))
        if not taxonomy_path.is_absolute():
            repo_root = Path(__file__).resolve().parents[2]
            taxonomy_path = repo_root / taxonomy_path

        if not taxonomy_path.exists():
            raise FileNotFoundError(f"GLiNER taxonomy config not found: {taxonomy_path}")

        cfg = OmegaConf.to_container(OmegaConf.load(taxonomy_path), resolve=True)
        if not isinstance(cfg, dict):
            raise TypeError("Taxonomy YAML did not resolve to a mapping")
        taxonomy_cfg = cfg.get("taxonomy", {})
        # Note: width_constraints are now embedded per-column in taxonomy_cfg
        return taxonomy_cfg, None

    # --- Facade component builders (no parameter overrides) ---
    def get_detector(self) -> GLiNERDetector:
        """Return a configured GLiNERDetector (lazy cached)."""
        if self._detector is not None:
            return self._detector

        taxonomy_cfg, _ = self._load_taxonomy_bundle()
        device = self._resolve_device(self._cfg_get("gliner.device"))

        from .pollution_guard.gliner_detector import (
            SOBRTaxonomy,
            EntityWidthConstraints,
            BatchInferenceConfig,
        )
        from .pollution_guard.semantic_chunker import BudgetConfig
        
        # Build taxonomy from config
        taxonomy = SOBRTaxonomy.from_config(taxonomy_cfg)
        
        # Build width constraints from taxonomy (column-driven)
        constraints = EntityWidthConstraints.from_taxonomy(taxonomy)
        
        # Build batch inference config from YAML (Performance Optimization v2.0)
        batch_inference_cfg = self.config.get("gliner", {}).get("batch_inference", {})
        batch_config = BatchInferenceConfig(
            enable_batching=batch_inference_cfg.get("enable_batching", True),
            batch_size=int(self._cfg_get("gliner.batch_size")),
            num_buckets=batch_inference_cfg.get("num_buckets"),  # None = auto-compute
            min_bucket_size=batch_inference_cfg.get("min_bucket_size", 4),
            enable_prompt_caching=batch_inference_cfg.get("enable_prompt_caching", True),
        )
        
        # Build chunking/budget config from YAML (includes word-aware budgeting)
        gliner_cfg = self.config.get("gliner", {})
        chunking_cfg = gliner_cfg.get("chunking", {})
        chunk_cache_path = chunking_cfg.get("checkpoint_path") or chunking_cfg.get(
            "chunk_cache_path"
        )
        budget_config = BudgetConfig(
            model_max_length=int(self._cfg_get("encoder.max_length")),
            mode=chunking_cfg.get("mode", "single_sentence"),
            legacy_sequential_mode=chunking_cfg.get("legacy_sequential_mode", False),
            parallel_chunking_workers=int(chunking_cfg.get("parallel_chunking_workers", 0)),
            parallel_chunking_min_texts=int(chunking_cfg.get("parallel_chunking_min_texts", 5000)),
            batch_size_per_worker=int(chunking_cfg.get("batch_size_per_worker", 0)),
            # Word-aware budget constraints (GLiNER truncates at WORDS, not tokens)
            gliner_max_words=int(gliner_cfg.get("gliner_max_words", 512)),
            tokens_per_word_ratio=float(gliner_cfg.get("tokens_per_word_ratio", 1.3)),
        )
        
        # Bi-encoder enforcement flag
        require_bi_encoder = bool(gliner_cfg.get("require_bi_encoder", False))

        self._detector = GLiNERDetector(
            model_name=self._cfg_get("gliner.model"),
            device=device,
            max_length=int(self._cfg_get("encoder.max_length")),
            confidence_threshold=float(self._cfg_get("gliner.confidence_threshold")),
            taxonomy=taxonomy,
            constraints=constraints,
            budget_config=budget_config,
            batch_inference_config=batch_config,
            require_bi_encoder=require_bi_encoder,
            chunk_cache_path=str(chunk_cache_path) if chunk_cache_path else None,
        )
        return self._detector

    def get_masker(self) -> SpanMasker:
        """Return a configured SpanMasker tied to the detector tokenizer."""
        if self._masker is not None:
            return self._masker

        detector = self.get_detector()
        self._masker = SpanMasker(
            tokenizer=detector.model.data_processor.transformer_tokenizer,
            entity_to_mask=detector.get_mask_tokens(),
        )
        return self._masker

    def get_embedder(self) -> FrozenEmbedder:
        """Return a configured FrozenEmbedder (lazy cached).
        
        Critical (Section 5.1 of LEACE Strategy Report):
        - The embedder MUST register the same typed mask tokens as GLiNERDetector.
        - This ensures "[MASK:AGE]" is encoded as a single token, not subword fragments.
        - Mask tokens are extracted from SOBRTaxonomy via the detector.
        """
        if self._embedder is not None:
            return self._embedder

        # Get mask tokens from detector for tokenizer alignment (Section 5.1)
        detector = self.get_detector()
        mask_tokens = list(dict.fromkeys(detector.get_mask_tokens().values()))

        device = self._resolve_device(self._cfg_get("encoder.device"))
        self._embedder = FrozenEmbedder(
            model_name=self._cfg_get("encoder.model"),
            device=device,
            max_length=int(self._cfg_get("encoder.max_length")),
            special_tokens=mask_tokens,  # Critical for tokenizer alignment
        )
        return self._embedder

    def get_leace(self) -> LEACEComputer:
        """Return a configured LEACEComputer (lazy cached)."""
        if self._leace is not None:
            return self._leace

        device = self._resolve_device(self._cfg_get("encoder.device"))
        self._leace = LEACEComputer(
            embedding_dim=int(self._cfg_get("encoder.hidden_dim")),
            regularization=float(self._cfg_get("leace.regularization")),
            device=device,
            force_cpu=bool(self._cfg_get("leace.force_cpu")),
        )
        return self._leace
    
    def run(
        self,
        input_dataset_path: Path,
        output_dir: Path,
    ) -> PhaseArtifacts:
        """
        Execute Phase A pipeline end-to-end.
        
        Args:
            input_dataset_path: Path to input SOBR dataset.
            output_dir: Directory for output artifacts.
            
        Returns:
            PhaseArtifacts with paths to all outputs.
        """
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output paths
        clean_dataset_path = output_dir / "clean_dataset.arrow"
        projection_matrix_path = output_dir / "projection_matrix.pt"
        pollution_logs_path = output_dir / "pollution_logs.arrow"
        
        logger.info("=" * 80)
        logger.info("PHASE A PIPELINE: POLLUTION DETECTION & MITIGATION")
        logger.info("=" * 80)
        logger.info(f"Input dataset: {input_dataset_path}")
        logger.info(f"Output directory: {output_dir}")
        
        # Execute strategy
        metadata = self.strategy.execute(
            input_dataset_path=input_dataset_path,
            output_dataset_path=clean_dataset_path,
            projection_matrix_path=projection_matrix_path,
            pollution_logs_path=pollution_logs_path,
            config=self.config,
        )
        
        # Initialize paths for reports
        reports_dir = None
        metrics_path = None
        
        # --- Reporting & Visualization (config-driven) ---
        viz_config = self.config.get("visualization", {})
        viz_enabled = viz_config.get("enabled", True)
        
        if viz_enabled:
            try:
                logger.info("Generating Phase A reports and visualizations...")
                reports_dir = output_dir / "reports" / "phase_a"
                reports_dir.mkdir(parents=True, exist_ok=True)

                # Load artifacts for reporting
                import pyarrow.feather as feather
                import torch
                import json
                clean_table = feather.read_table(clean_dataset_path)
                logs_table = feather.read_table(pollution_logs_path)
                projection_matrix = torch.load(projection_matrix_path, map_location="cpu")

                # Compute and save metrics
                from .evaluation.metrics import compute_phase_a_metrics, save_metrics
                metrics_data = compute_phase_a_metrics(
                    clean_table=clean_table,
                    logs_table=logs_table,
                    projection_matrix=projection_matrix,
                    strategy_metadata=metadata,
                    config=self.config,
                )
                metrics_path = reports_dir / "phase_a_metrics.json"
                save_metrics(metrics_data, metrics_path)

                # Generate visualizations
                from .evaluation.visualizations_phase_a import generate_phase_a_plots
                import numpy as np
                
                # Sample for pandas conversion based on config
                vis_sample_size = int(viz_config.get("pca_max_samples", 1000)) * 2  # 2x for viz sampling buffer
                if len(clean_table) > vis_sample_size:
                    indices = np.random.choice(len(clean_table), vis_sample_size, replace=False)
                    indices.sort()
                    clean_df_sample = clean_table.take(indices).to_pandas()
                else:
                    clean_df_sample = clean_table.to_pandas()
                
                logs_df = logs_table.to_pandas()

                generate_phase_a_plots(
                    clean_df_sample=clean_df_sample,
                    logs_df=logs_df,
                    projection_matrix=projection_matrix,
                    embedder=self.get_embedder(),
                    output_dir=reports_dir,
                    metrics=metrics_data,
                    config=self.config,
                )
                logger.info(f"Saved visualizations to {reports_dir}")

            except Exception as e:
                logger.warning(f"Reporting/Visualization failed: {e}", exc_info=True)
        else:
            logger.info("Visualization generation disabled in config")

        # Create artifacts
        artifacts = PhaseArtifacts(
            clean_dataset_path=clean_dataset_path,
            projection_matrix_path=projection_matrix_path,
            pollution_logs_path=pollution_logs_path,
            metrics_path=metrics_path,
            reports_dir=reports_dir,
            metadata=metadata,
        )

        self._check_quality_gates(metadata)

        logger.info("=" * 80)
        logger.info("PHASE A COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Clean dataset: {clean_dataset_path}")
        logger.info(f"Projection matrix: {projection_matrix_path}")
        logger.info(f"Pollution logs: {pollution_logs_path}")
        if metrics_path:
            logger.info(f"Metrics: {metrics_path}")
        if reports_dir:
            logger.info(f"Reports directory: {reports_dir}")
        
        return artifacts

    def _check_quality_gates(self, metadata: Dict[str, Any]) -> None:
        """Check explicit recall and amnesic drop against configured thresholds."""
        enforce = bool(self._cfg_get("quality.enforce_thresholds"))

        recall_threshold = float(self._cfg_get("gliner.explicit_recall_threshold"))
        recall = metadata.get("explicit_recall", {}).get("overall")
        if recall_threshold is not None and recall is not None:
            if recall < recall_threshold:
                msg = (
                    f"Explicit recall {recall:.2%} < "
                    f"threshold {recall_threshold:.2%} (FR-26)"
                )
                if enforce:
                    raise RuntimeError(msg)
                logger.warning(msg)

        probe_threshold = float(self._cfg_get("probe.amnesic_drop_threshold"))
        probe = metadata.get("probe", {})
        amnesic_drop = probe.get("amnesic_drop")
        if probe_threshold is not None and amnesic_drop is not None:
            if amnesic_drop < probe_threshold:
                msg = (
                    f"Amnesic drop {amnesic_drop:.2%} < "
                    f"threshold {probe_threshold:.2%} (PA-PROBE-03)"
                )
                if enforce:
                    raise RuntimeError(msg)
                logger.warning(msg)
