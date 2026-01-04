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
from typing import Dict, Any
from dataclasses import dataclass

from .pollution_guard.strategies.base import PollutionFilterStrategy

logger = logging.getLogger(__name__)


@dataclass
class PhaseArtifacts:
    """
    Phase A output artifacts.
    
    Attributes:
        clean_dataset_path: Path to cleaned dataset with post_masked.
        projection_matrix_path: Path to LEACE projection matrix.
        pollution_logs_path: Path to pollution detection logs.
        metadata: Execution metadata.
    """
    clean_dataset_path: Path
    projection_matrix_path: Path
    pollution_logs_path: Path
    metadata: Dict[str, Any]


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
        config: Dict[str, Any],
    ):
        """
        Initialize Phase A pipeline.
        
        Args:
            strategy: Execution strategy (laptop/HPC).
            config: Pipeline configuration.
        """
        self.strategy = strategy
        self.config = config
        
        logger.info(f"PhaseAPipeline initialized with {strategy.__class__.__name__}")
    
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
        
        # Create artifacts
        artifacts = PhaseArtifacts(
            clean_dataset_path=clean_dataset_path,
            projection_matrix_path=projection_matrix_path,
            pollution_logs_path=pollution_logs_path,
            metadata=metadata,
        )

        self._check_quality_gates(metadata)
        
        logger.info("=" * 80)
        logger.info("PHASE A COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Clean dataset: {clean_dataset_path}")
        logger.info(f"Projection matrix: {projection_matrix_path}")
        logger.info(f"Pollution logs: {pollution_logs_path}")
        
        return artifacts

    def _check_quality_gates(self, metadata: Dict[str, Any]) -> None:
        """Check explicit recall and amnesic drop against configured thresholds."""
        enforce = self.config.get("enforce_quality_thresholds", False)

        recall_threshold = self.config.get("gliner_explicit_recall_threshold")
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

        probe_threshold = self.config.get("probe_amnesic_drop_threshold")
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
