"""
Abstract Strategy Interface for Phase A Pollution Filtering.

Defines the contract for dual-mode execution strategies with stage-based decomposition.

Stage Architecture (Sequential):
1. Chunking Stage: Parallel chunking → post_chunked column
2. Inference Stage: GLiNER detection → entity spans
3. LEACE Stage: Masking + embedding + projection computation
4. Probing Stage: Amnesic drop metrics + visualizations

Implements: phaseA-D_implementation_plan.md Section 7.1-7.2
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
import torch
import pyarrow as pa


@dataclass
class ChunkingContext:
    """
    Context produced by the chunking stage.
    
    Contains the table with populated post_chunked column,
    ready for the inference stage.
    """
    table_with_chunks: pa.Table
    posts: List[str]
    post_ids: List[str]
    stage_skipped: bool = False
    invalid_rows_recomputed: int = 0


@dataclass
class InferenceContext:
    """
    Context produced by the inference stage.
    
    Contains detected entities per document, ready for masking.
    """
    entities_batch: List[List[Dict[str, Any]]]
    skipped_doc_indices: Set[int] = field(default_factory=set)
    oom_count: int = 0
    final_budget: int = 0
    stage_skipped: bool = False
    # HPC-specific: async storage path for crash recovery
    async_storage_path: Optional[Path] = None
    chunks_recovered: int = 0
    # Laptop-specific: inference results cache path
    inference_results_path: Optional[Path] = None
    retry_rounds: int = 0


@dataclass
class LEACEContext:
    """
    Context produced by the LEACE stage.
    
    Contains the projection matrix and masked texts, ready for probing.
    """
    projection_matrix: torch.Tensor
    masked_texts: List[str]
    pollution_logs: List[Dict[str, Any]]
    total_spans: int = 0
    stage_skipped: bool = False
    # For probing stage
    probe_embeddings: Optional[List[torch.Tensor]] = None
    table_for_probing: Optional[pa.Table] = None


@dataclass
class ProbingContext:
    """
    Context produced by the probing stage.
    
    Contains all computed metrics for the final metadata.
    """
    by_column: Dict[str, Any] = field(default_factory=dict)
    by_column_extended: Dict[str, Any] = field(default_factory=dict)
    min_amnesic_drop: float = 0.0
    mean_amnesic_drop: float = 0.0
    separability_before: Dict[str, Any] = field(default_factory=dict)
    separability_after: Dict[str, Any] = field(default_factory=dict)
    class_imbalance: Dict[str, Any] = field(default_factory=dict)
    control_probe_results: Dict[str, Any] = field(default_factory=dict)
    benchmark_results: Dict[str, Any] = field(default_factory=dict)
    reports_dir: Optional[Path] = None
    stage_skipped: bool = False
    # Visualization data for pipeline orchestration (returned to pipeline for centralized plotting)
    visualization_data: Optional[Dict[str, Any]] = None
    # Demographic columns used for probing (for single-label mode detection)
    demo_cols: Optional[List[str]] = None
    # Probe configuration for visualization
    probe_config: Optional[Any] = None


@dataclass
class SkipStagesConfig:
    """
    Stage skip flags for modular pipeline execution.
    
    Enables entry-point and early-exit execution patterns.
    """
    skip_chunking: bool = False
    skip_inference: bool = False
    skip_leace: bool = False
    skip_probing: bool = False
    # When True, missing prerequisites will be bypassed with warnings instead of errors
    force_skip: bool = False
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SkipStagesConfig":
        """Create from execution.skip_stages config dict and top-level flag."""
        exec_cfg = config.get("execution", {})
        skip_stages = exec_cfg.get("skip_stages", {})
        return cls(
            skip_chunking=bool(skip_stages.get("skip_chunking", False)),
            skip_inference=bool(skip_stages.get("skip_inference", False)),
            skip_leace=bool(skip_stages.get("skip_leace", False)),
            skip_probing=bool(skip_stages.get("skip_probing", False)),
            force_skip=bool(exec_cfg.get("force_skip", False))
        )
    
    def validate_for_entry(self) -> None:
        """
        Validate skip flags have logical consistency.
        
        Sequential dependency: Chunking → Inference → LEACE → Probing
        Skipping a later stage implies earlier stages completed.
        """
        # If skipping inference, must also skip chunking
        if self.skip_inference and not self.skip_chunking:
            self.skip_chunking = True
        # If skipping LEACE, must also skip chunking and inference
        if self.skip_leace:
            self.skip_chunking = True
            self.skip_inference = True


class PollutionFilterStrategy(ABC):
    """
    Abstract strategy for Phase A pollution filtering.
    
    Concrete implementations:
    - LaptopFilterStrategy: CPU/Low-VRAM with batch accumulation
    - HPCFilterStrategy: GPU with full-batch computation
    
    Stage Methods (to be implemented by subclasses):
    - _run_chunking_stage: Parallel chunking → post_chunked
    - _run_inference_stage: GLiNER detection → entities
    - _run_leace_stage: Masking + embedding + LEACE projection
    - _run_probing_stage: Amnesic drop metrics + visualizations
    """
    
    @abstractmethod
    def execute(
        self,
        input_dataset_path: Path,
        output_dataset_path: Path,
        projection_matrix_path: Path,
        pollution_logs_path: Path,
        config: Dict[str, Any],
        use_only_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute Phase A pollution filtering pipeline.
        
        Args:
            input_dataset_path: Path to input Arrow dataset.
            output_dataset_path: Path to output cleaned dataset.
            projection_matrix_path: Path to save projection matrix.
            pollution_logs_path: Path to save pollution logs.
            config: Configuration dictionary.
            use_only_labels: Optional list of demographic labels to filter to.
                If provided, only rows with valid values for ALL specified labels
                will be processed, and only those labels will be used for LEACE.
            
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

