"""
GLiNER Pollution Detector for SOBR Corpus.

Implements the Symbolic Layer of Phase A with semantic-aware context management:
- Dynamic Prompt-Aware Budgeting (calculates effective text budget after label costs)
- Sentence-Boundary Chunking via PySBD (preserves semantic context)
- Fallback hard-slicing for pathological run-on sentences
- Tokenizer-safe typed masks (registered as special tokens)
- Entity-specific width constraints
- Column-driven prompt selection (only prompts for non-null columns per row)

Performance Optimizations (v2.0):
- Batched inference via GLiNER.inference() API (4-8x speedup)
- Length-based bucketing to minimize padding waste
- Bi-encoder prompt embedding caching (when available)
- VRAM-aware bucket count auto-computation

Semantic Chunker Optimizations (v2.1):
- Binary search token counting: O(log N) bisect replaces O(N) linear scan
  (50-100x speedup on token counting operations)
- "Tokenize Once" optimization: Eliminates redundant tokenization by using
  offset mapping (combined with bisect for maximum efficiency)
- Batched parallel processing: Workers process large document batches instead
  of individual documents, reducing IPC overhead (2-4x speedup on multi-core)
  Configure via BudgetConfig.parallel_chunking_workers, batch_size_per_worker

Autotuning Integration (v2.2):
- RuntimeController feedback loop for adaptive token budget
- DynamicBatchIterator for variable-size batches
- OOM protection with automatic retry and budget slashing
- Telemetry collection for monitoring throughput and memory

Reference: GLiNER_ImpNementation_Strategy.md Sections 2.1-2.4
Implements: FR-05 (GLiNER Integration), FR-06 (Chunking), FR-07 (Precision Filters)
"""

from __future__ import annotations

import inspect
import logging
import pickle
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np
import pyarrow as pa
import torch
from gliner import GLiNER
from tqdm.auto import tqdm

from .semantic_chunker import (
    BudgetConfig,
    ChunkInfo,
    SemanticChunker,
    deduplicate_entities,
    project_entity_offsets,
)

if TYPE_CHECKING:
    from ..hardware_ops.runtime import RuntimeController, RuntimeMetrics
    from ..hardware_ops.cuda_graphs import GraphAwareInference

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Batch Inference Configuration & Helpers
# ---------------------------------------------------------------------------


@dataclass
class BatchInferenceConfig:
    """
    Configuration for batched GLiNER inference.
    
    Attributes:
        enable_batching: Whether to use batched inference (vs sequential).
        batch_size: Maximum texts per inference batch.
        num_buckets: Number of length buckets for smart batching.
            If None or "auto", computed from available VRAM.
        min_bucket_size: Minimum samples per bucket before merging.
        enable_prompt_caching: Cache label embeddings for bi-encoder models.
    """
    enable_batching: bool = True
    batch_size: int = 32
    num_buckets: Optional[int] = None  # None = auto-compute from VRAM
    min_bucket_size: int = 4
    enable_prompt_caching: bool = True


@dataclass
class InferenceResult:
    """
    Result from inference_from_manifest with completeness tracking.
    
    Attributes:
        entities: List of entity lists per document (nested, document-order).
        skipped_doc_indices: Set of document indices that were skipped due to OOM.
            Empty if all documents were processed successfully.
        oom_count: Number of OOM events encountered during inference.
        final_budget: Final token budget after all adjustments.
    """
    entities: List[List[Dict[str, Any]]]
    skipped_doc_indices: Set[int] = field(default_factory=set)
    oom_count: int = 0
    final_budget: int = 0


def auto_compute_bucket_count(vram_gb: Optional[float] = None) -> int:
    """
    Auto-compute optimal bucket count based on available VRAM.
    
    Heuristic: More VRAM allows finer bucketing (less padding waste).
    - <8 GB: 5 buckets (aggressive bucketing to fit batches)
    - 8-16 GB: 8 buckets (laptop/consumer GPU)
    - 16-40 GB: 12 buckets (workstation)
    - 40+ GB: 16 buckets (A100/H100)
    
    Args:
        vram_gb: GPU VRAM in GB. If None, auto-detected.
        
    Returns:
        Recommended number of length buckets.
    """
    if vram_gb is None:
        if torch.cuda.is_available():
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = vram_bytes / (1024 ** 3)
        else:
            vram_gb = 0.0  # CPU mode
    
    if vram_gb < 8:
        return 5
    elif vram_gb < 16:
        return 8
    elif vram_gb < 40:
        return 12
    else:
        return 16


def compute_chunk_length_buckets(
    chunk_lengths: List[int],
    num_buckets: int,
) -> Tuple[List[int], Dict[int, List[int]]]:
    """
    Compute length bucket boundaries and assign chunks to buckets.
    
    Uses quantile-based bucketing for balanced distribution.
    
    Args:
        chunk_lengths: List of token lengths for each chunk.
        num_buckets: Number of buckets to create.
        
    Returns:
        Tuple of:
            - boundaries: Sorted bucket boundary values
            - bucket_to_indices: Mapping from bucket_id to chunk indices
    """
    if not chunk_lengths:
        return [], {}
    
    lengths_arr = np.array(chunk_lengths)
    
    # Quantile-based boundaries
    boundaries = []
    for i in range(1, num_buckets):
        quantile = i / num_buckets
        boundary = int(np.quantile(lengths_arr, quantile))
        boundaries.append(boundary)
    boundaries = sorted(set(boundaries))
    
    # Assign to buckets
    bucket_to_indices: Dict[int, List[int]] = {i: [] for i in range(len(boundaries) + 1)}
    
    for idx, length in enumerate(chunk_lengths):
        bucket_id = 0
        for boundary_idx, boundary in enumerate(boundaries):
            if length >= boundary:
                bucket_id = boundary_idx + 1
            else:
                break
        bucket_to_indices[bucket_id].append(idx)
    
    return boundaries, bucket_to_indices


# ---------------------------------------------------------------------------
# Column-Driven Taxonomy
# ---------------------------------------------------------------------------


@dataclass
class ColumnConfig:
    """Configuration for a single SOBR demographic column."""

    prompts: List[str] = field(default_factory=list)
    distractors: List[str] = field(default_factory=list)
    mask_token: str = "[MASK:UNKNOWN]"
    width: int = 8
    reference_patterns: List[str] = field(default_factory=list)


@dataclass
class SOBRTaxonomy:
    """
    Column-driven SOBR taxonomy mapping.

    Maps SOBR Arrow columns (birth_year, female, nationality, political_leaning,
    extrovert, sensing, feeling, judging) to GLiNER prompts, distractors,
    mask tokens, width constraints, and reference patterns.

    All prompt/distractor selection is driven by per-row column presence.
    No default hard-coded labels; requires YAML config.

    Implements: GLiNER_Implementation_Strategy.md Section 2.1
    """

    # Column name -> ColumnConfig mapping (loaded from YAML)
    column_prompts: Dict[str, ColumnConfig] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> SOBRTaxonomy:
        """
        Create taxonomy from a configuration dictionary.

        Expected YAML structure:
            taxonomy:
              column_prompts:
                birth_year:
                  prompts: [...]
                  distractors: [...]
                  mask_token: "[MASK:AGE]"
                  width: 6
                  reference_patterns: [...]
                female:
                  ...
        """
        column_prompts_raw = config.get("column_prompts", {})
        column_prompts: Dict[str, ColumnConfig] = {}

        for column_name, col_config in column_prompts_raw.items():
            if isinstance(col_config, dict):
                column_prompts[column_name] = ColumnConfig(
                    prompts=col_config.get("prompts", []),
                    distractors=col_config.get("distractors", []),
                    mask_token=col_config.get("mask_token", f"[MASK:{column_name.upper()}]"),
                    width=col_config.get("width", 8),
                    reference_patterns=col_config.get("reference_patterns", []),
                )

        return cls(column_prompts=column_prompts)

    def get_all_columns(self) -> List[str]:
        """Return all configured column names."""
        return list(self.column_prompts.keys())

    def get_prompts_for_columns(self, columns: List[str]) -> List[str]:
        """
        Get combined prompts + distractors for the specified columns.

        Args:
            columns: List of SOBR column names with non-null values for this row.

        Returns:
            Deduplicated list of prompts + distractors for inference.
        """
        labels: List[str] = []
        for col in columns:
            if col in self.column_prompts:
                cfg = self.column_prompts[col]
                labels.extend(cfg.prompts)
                labels.extend(cfg.distractors)
        return list(dict.fromkeys(labels))

    def get_inference_labels(self) -> List[str]:
        """
        Get all prompts + distractors for all columns (full label set).

        Returns:
            Deduplicated list of all labels for inference.
        """
        return self.get_prompts_for_columns(self.get_all_columns())

    def get_mask_tokens(self) -> Dict[str, str]:
        """
        Return column -> mask token mapping.

        Note: Multiple columns may share the same mask token (e.g., MBTI dimensions).
        """
        return {col: cfg.mask_token for col, cfg in self.column_prompts.items()}

    def get_mask_token_for_column(self, column: str) -> str:
        """Get mask token for a specific column."""
        if column in self.column_prompts:
            return self.column_prompts[column].mask_token
        return f"[MASK:{column.upper()}]"

    def get_width_for_column(self, column: str) -> int:
        """Get width constraint for a specific column."""
        if column in self.column_prompts:
            return self.column_prompts[column].width
        return 8  # default

    def get_reference_patterns(self) -> Dict[str, List[str]]:
        """Return column -> reference patterns mapping."""
        return {col: cfg.reference_patterns for col, cfg in self.column_prompts.items()}

    def get_reference_patterns_for_columns(self, columns: List[str]) -> Dict[str, List[str]]:
        """Get reference patterns for specific columns."""
        return {col: self.column_prompts[col].reference_patterns
                for col in columns if col in self.column_prompts}

    def normalize_label(self, label: str) -> str:
        """
        Map a prompt label to its source column name.

        Args:
            label: A prompt or distractor string from GLiNER output.

        Returns:
            The column name this label belongs to, or the label itself if unknown.
        """
        # Check if it's already a column name
        if label in self.column_prompts:
            return label
        # Reverse lookup: find which column contains this prompt
        return self._prompt_to_column(label)

    def _prompt_to_column(self, prompt_label: str) -> str:
        """Reverse lookup: prompt string -> column name."""
        for column, cfg in self.column_prompts.items():
            if prompt_label in cfg.prompts:
                return column
        return prompt_label  # Unknown prompt, return as-is

    def is_distractor(self, label: str) -> bool:
        """Check if a label is a distractor (should be filtered out)."""
        for cfg in self.column_prompts.values():
            if label in cfg.distractors:
                return True
        return False

    def filter_distractors(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove distractor entities and normalize labels to column names.

        Args:
            entities: Raw GLiNER detection results.

        Returns:
            Filtered entities with labels normalized to column names.
        """
        result = []
        for e in entities:
            e = dict(e)
            raw_label = e.get("label", "")

            # Skip distractors
            if self.is_distractor(raw_label):
                continue

            # Normalize to column name
            e["label"] = self.normalize_label(raw_label)
            result.append(e)

        return result

    # -----------------------------------------------------------------------
    # Backward Compatibility Methods
    # -----------------------------------------------------------------------

    def get_target_labels(self) -> List[str]:
        """
        Return column names (backward compatibility).

        Previously returned internal label IDs; now returns SOBR column names.
        """
        return self.get_all_columns()

    @property
    def distractor_labels(self) -> List[str]:
        """
        Return all distractors across columns (backward compatibility).

        Previously was a flat list; now aggregates from all column configs.
        """
        all_distractors: List[str] = []
        for cfg in self.column_prompts.values():
            all_distractors.extend(cfg.distractors)
        return list(dict.fromkeys(all_distractors))


@dataclass
class EntityWidthConstraints:
    """
    Entity-specific token width constraints (column-driven).

    Implements: GLiNER_Implementation_Strategy.md Section 2.2
    """

    max_widths: Dict[str, int] = field(default_factory=dict)
    default_width: int = 8

    @classmethod
    def from_taxonomy(cls, taxonomy: SOBRTaxonomy) -> EntityWidthConstraints:
        """Create width constraints from a column-driven taxonomy."""
        max_widths = {col: cfg.width for col, cfg in taxonomy.column_prompts.items()}
        return cls(max_widths=max_widths)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> EntityWidthConstraints:
        """Create width constraints from a configuration dictionary."""
        default_width = config.pop("default", 8) if isinstance(config, dict) else 8
        max_widths = {k: int(v) for k, v in config.items()} if isinstance(config, dict) else {}
        return cls(max_widths=max_widths, default_width=default_width)

    def get_max_width(self, entity_type: str) -> int:
        """Get maximum token width for entity type (column name)."""
        return self.max_widths.get(entity_type, self.default_width)


# ---------------------------------------------------------------------------
# GLiNER Detector
# ---------------------------------------------------------------------------


class GLiNERDetector:
    """
    GLiNER-based pollution span detector with semantic-aware context management.

    Architecture:
    - Dynamic Prompt-Aware Budgeting: Calculates effective text budget after label costs
    - Sentence-Boundary Chunking: PySBD-based segmentation preserves semantic context
    - Fallback Hard-Slicing: Handles pathological run-on sentences
    - Precision Filtering: Width constraints + confidence threshold

    Performance Optimizations (v2.0):
    - Batched inference via GLiNER.inference() API
    - Length-based bucketing to minimize padding waste
    - Bi-encoder prompt embedding caching (when available)

    Implements: FR-05, FR-06, FR-07
    """

    # Class-level model cache to avoid reloading
    _MODEL_CACHE: Dict[tuple, GLiNER] = {}

    @staticmethod
    def is_bi_encoder_model(model_name: str) -> bool:
        """
        Check if a GLiNER model is bi-encoder without fully loading it.
        
        Bi-encoder models have `labels_encoder` in their config. This method
        loads only the config to check architecture type before full model load.
        
        Args:
            model_name: HuggingFace model identifier (e.g., "urchade/gliner_large-v2.1").
            
        Returns:
            True if model is bi-encoder (has labels_encoder config), False otherwise.
            
        Note:
            This downloads config.json but not model weights, so it's lightweight.
            For local models, this is nearly instant.
        """
        try:
            from huggingface_hub import hf_hub_download
            import json
            
            # Try to download just the config file
            config_path = hf_hub_download(
                repo_id=model_name,
                filename="gliner_config.json",
                local_files_only=False,
            )
            
            with open(config_path, "r") as f:
                config = json.load(f)
            
            # Bi-encoder models have labels_encoder defined
            labels_encoder = config.get("labels_encoder")
            return labels_encoder is not None
            
        except Exception as e:
            logger.warning(
                f"Could not pre-check bi-encoder status for '{model_name}': {e}. "
                f"Will check after model load."
            )
            return False

    def __init__(
        self,
        model_name: str = "urchade/gliner_large-v2.1",
        device: str = "cuda",
        max_length: int = 512,
        confidence_threshold: float = 0.85,
        taxonomy: Optional[SOBRTaxonomy] = None,
        constraints: Optional[EntityWidthConstraints] = None,
        taxonomy_config: Optional[Dict[str, Any]] = None,
        constraints_config: Optional[Dict[str, Any]] = None,
        budget_config: Optional[BudgetConfig] = None,
        batch_inference_config: Optional[BatchInferenceConfig] = None,
        require_bi_encoder: bool = False,
        chunk_cache_path: Optional[str] = None,
        execution_config: Optional[Dict[str, Any]] = None,
        # Legacy parameters (ignored but accepted for backward compatibility)
        center_window_keep: int = 100,
    ):
        """
        Initialize GLiNER detector with semantic-aware context management.

        Args:
            model_name: GLiNER model identifier.
            device: PyTorch device (cuda/cpu).
            max_length: Maximum sequence length for GLiNER.
            confidence_threshold: Minimum confidence for accepting spans (>= 0.85).
            taxonomy: SOBR taxonomy (defaults to SOBRTaxonomy()).
            constraints: Entity width constraints (defaults to EntityWidthConstraints()).
            taxonomy_config: Dict to construct taxonomy from config.
            constraints_config: Dict to construct constraints from config.
            budget_config: Configuration for dynamic budgeting.
            batch_inference_config: Configuration for batched inference optimization.
            require_bi_encoder: If True, raise error if loaded model is not bi-encoder.
                Bi-encoder models support encode_labels() for prompt caching.
            chunk_cache_path: Optional path to store/load chunking checkpoint.
            execution_config: Optional dict with torch.compile and CUDA graph settings.
                Expected keys: enable_torch_compile, torch_compile_mode, torch_compile_backend,
                torch_compile_fullgraph, torch_compile_dynamic, enable_cuda_graphs.
            center_window_keep: DEPRECATED - Ignored. Semantic chunking handles this.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.confidence_threshold = confidence_threshold
        self.model_name = model_name
        self.require_bi_encoder = require_bi_encoder
        self.chunk_cache_path = Path(chunk_cache_path) if chunk_cache_path else None
        self._torch_compiled = False  # Will be set by _apply_torch_compile()

        # Initialize taxonomy and constraints
        if taxonomy is None and taxonomy_config is not None:
            taxonomy = SOBRTaxonomy.from_config(taxonomy_config)
        if constraints is None and constraints_config is not None:
            constraints = EntityWidthConstraints.from_config(constraints_config)

        self.taxonomy = taxonomy or SOBRTaxonomy()
        self.constraints = constraints or EntityWidthConstraints()
        self._taxonomy_label_order = self.taxonomy.get_inference_labels()
        self._taxonomy_label_set = set(self._taxonomy_label_order)

        # Budget configuration for semantic chunker
        self.budget_config = budget_config or BudgetConfig(model_max_length=max_length)
        
        # Batch inference configuration
        self.batch_config = batch_inference_config or BatchInferenceConfig()
        auto_bucket_count = self.batch_config.num_buckets is None
        
        # Store execution config for JIT compilation
        self._execution_config = execution_config

        # Load GLiNER model
        self.model = self._load_model(model_name)
        
        # Apply torch.compile if enabled (Phase 1 optimization)
        self._apply_torch_compile()
        
        # Initialize CUDA Graph acceleration (Phase 3 optimization)
        self._graph_inference = self._init_cuda_graphs()

        # Auto-compute bucket count if not specified (after device fallback is resolved)
        if auto_bucket_count:
            self._configure_auto_bucket_count()
        
        # Sync GLiNER processor max_len with our configured max_length
        self._sync_processor_max_len()

        # Register typed mask tokens as special tokens
        self._register_mask_tokens()

        # Extract GLiNER's words_splitter for exact word counting (1:1 parity)
        words_splitter = self._get_words_splitter()

        # Extract GLiNER words_splitter_type so multiprocessing workers can rebuild
        # the same splitter without sharing the non-picklable instance.
        words_splitter_type = self._get_words_splitter_type()

        # Initialize semantic chunker with injected words_splitter
        self.chunker = SemanticChunker(
            tokenizer=self.tokenizer,
            config=self.budget_config,
            words_splitter=words_splitter,
            words_splitter_type=words_splitter_type,
        )
        
        # Bi-encoder prompt embedding cache
        # Key: tuple(labels), Value: precomputed embeddings tensor
        self._prompt_embedding_cache: Dict[Tuple[str, ...], torch.Tensor] = {}
        self._is_bi_encoder = self._check_bi_encoder_support()
        
        # Enforce bi-encoder requirement if configured
        if self.require_bi_encoder and not self._is_bi_encoder:
            raise ValueError(
                f"require_bi_encoder=True but model '{model_name}' is a uni-encoder. "
                f"Uni-encoder models do not support label embedding caching (encode_labels API). "
                f"Use a bi-encoder model (e.g., one with labels_encoder in config) or set "
                f"require_bi_encoder=False in pipeline.yaml."
            )
        
        if self._is_bi_encoder and self.batch_config.enable_prompt_caching:
            logger.info("Bi-encoder model detected - prompt embedding caching enabled")
        elif self.require_bi_encoder:
            logger.info("Bi-encoder model enforced via require_bi_encoder=True")

        logger.info(f"GLiNERDetector initialized on {self.device}")
        logger.info(f"Batch inference: {'enabled' if self.batch_config.enable_batching else 'disabled'} "
                   f"(batch_size={self.batch_config.batch_size}, buckets={self.batch_config.num_buckets})")
        logger.info(f"Words splitter: {type(words_splitter).__name__ if words_splitter else 'fallback regex'}")

    def _check_bi_encoder_support(self) -> bool:
        """
        Check if the loaded model supports bi-encoder embedding caching.
        
        Bi-encoder models expose encode_labels() and batch_predict_with_embeds().
        
        Returns:
            True if model supports precomputed label embeddings.
        """
        core_model = getattr(self.model, "model", self.model)
        has_encode = hasattr(core_model, "encode_labels") or hasattr(self.model, "encode_labels")
        has_batch_predict = hasattr(core_model, "batch_predict_with_embeds") or hasattr(self.model, "batch_predict_with_embeds")
        return has_encode and has_batch_predict

    def _get_words_splitter(self) -> Optional[Any]:
        """
        Extract GLiNER's WordsSplitter for 1:1 chunking parity.
        
        GLiNER uses a words_splitter (e.g., WhitespaceTokenSplitter) to convert
        text into atomic word units. This method extracts the exact splitter used
        by the loaded model so the SemanticChunker can count words identically.
        
        Returns:
            The GLiNER words_splitter instance, or None if not accessible.
            When None, SemanticChunker falls back to regex-based splitting.
        """
        data_processor = getattr(self.model, "data_processor", None)
        if data_processor is None:
            logger.warning(
                "GLiNER model has no data_processor - cannot extract words_splitter. "
                "Chunker will use fallback regex-based word splitting."
            )
            return None
        
        words_splitter = getattr(data_processor, "words_splitter", None)
        if words_splitter is None:
            logger.warning(
                "GLiNER data_processor has no words_splitter attribute. "
                "Chunker will use fallback regex-based word splitting."
            )
            return None
        
        logger.debug(f"Extracted GLiNER words_splitter: {type(words_splitter).__name__}")
        return words_splitter

    def _get_words_splitter_type(self) -> str:
        """Extract GLiNER's configured words_splitter_type (e.g., 'whitespace').

        This is preferred over attempting to pickle/share the words_splitter instance
        when using multiprocessing.
        
        Returns:
            The splitter type string, defaulting to 'whitespace' if not found.
            Most GLiNER models use whitespace splitting, so this is a safe default.
        """
        data_processor = getattr(self.model, "data_processor", None)
        if data_processor is None:
            return "whitespace"

        cfg = getattr(data_processor, "config", None)
        if cfg is None:
            return "whitespace"

        splitter_type = getattr(cfg, "words_splitter_type", None)
        return str(splitter_type) if splitter_type is not None else "whitespace"

    def _order_labels_by_taxonomy(self, labels: List[str]) -> List[str]:
        label_set = set(labels)
        ordered = [label for label in self._taxonomy_label_order if label in label_set]
        ordered.extend([label for label in labels if label not in self._taxonomy_label_set])
        return ordered

    def _get_device_vram_gb(self) -> float:
        """Get VRAM in GB for the active device (0.0 for CPU)."""
        if self.device.type != "cuda":
            return 0.0
        if not torch.cuda.is_available():
            return 0.0
        try:
            device_index = self.device.index if self.device.index is not None else 0
            vram_bytes = torch.cuda.get_device_properties(device_index).total_memory
            return vram_bytes / (1024 ** 3)
        except Exception as e:
            logger.debug(f"Failed to read CUDA device properties: {e}")
            return 0.0

    def _configure_auto_bucket_count(self) -> None:
        vram_gb = self._get_device_vram_gb()
        self.batch_config.num_buckets = auto_compute_bucket_count(vram_gb)
        if vram_gb:
            logger.info(
                f"Auto-computed bucket count: {self.batch_config.num_buckets} "
                f"(VRAM: {vram_gb:.1f}GB)"
            )
        else:
            logger.info(
                f"Auto-computed bucket count: {self.batch_config.num_buckets} (CPU mode)"
            )
    
    def _get_cached_prompt_embeddings(self, labels: List[str]) -> Optional[torch.Tensor]:
        """
        Get or compute cached prompt embeddings for bi-encoder models.
        
        Args:
            labels: List of inference labels.
            
        Returns:
            Cached embeddings tensor, or None if caching disabled or not bi-encoder.
        """
        if not self._is_bi_encoder or not self.batch_config.enable_prompt_caching:
            return None
            
        cache_key = tuple(labels)
        
        if cache_key in self._prompt_embedding_cache:
            return self._prompt_embedding_cache[cache_key]
        
        # Compute embeddings
        try:
            # Try model-level encode_labels first
            if hasattr(self.model, "encode_labels"):
                embeddings = self.model.encode_labels(labels)
            else:
                core = getattr(self.model, "model", None)
                if core and hasattr(core, "encode_labels"):
                    embeddings = core.encode_labels(labels)
                else:
                    return None
            
            self._prompt_embedding_cache[cache_key] = embeddings
            logger.debug(f"Cached prompt embeddings for {len(labels)} labels")
            return embeddings
            
        except Exception as e:
            logger.warning(f"Failed to cache prompt embeddings: {e}")
            return None

    @property
    def tokenizer(self):
        """Access the underlying transformer tokenizer."""
        return self.model.data_processor.transformer_tokenizer

    def _load_model(self, model_name: str) -> GLiNER:
        """
        Load GLiNER model with caching and device management.

        Args:
            model_name: HuggingFace model identifier.

        Returns:
            Loaded GLiNER model.
        """
        cache_key = (model_name, str(self.device), self.max_length)

        if cache_key in GLiNERDetector._MODEL_CACHE:
            return GLiNERDetector._MODEL_CACHE[cache_key]

        logger.info(f"Loading GLiNER model: {model_name}")

        # Load to CPU first to avoid OOM during from_pretrained
        model = GLiNER.from_pretrained(
            model_name,
            map_location="cpu",
            max_length=self.max_length,
        )

        # Move to requested device
        if self.device.type == "cuda":
            try:
                model = model.half()
            except Exception:
                pass

        try:
            model = model.to(self.device)
        except Exception:
            logger.warning("Failed to move model to CUDA, falling back to CPU")
            model = model.to("cpu")
            self.device = torch.device("cpu")

        model.eval()
        GLiNERDetector._MODEL_CACHE[cache_key] = model

        return model

    def _apply_torch_compile(self) -> None:
        """
        Apply torch.compile to the model's transformer layers if enabled.
        
        Reads configuration from self._execution_config to determine:
        - Whether to enable compilation
        - Compilation mode (default, reduce-overhead, max-autotune)
        - Backend (inductor, eager, etc.)
        - Whether to use fullgraph mode
        
        For RTX 5090 (Blackwell), max-autotune mode with Triton backend
        provides optimal kernel fusion and significant speedup.
        
        Note: torch.compile requires PyTorch 2.0+
        """
        if self._execution_config is None:
            logger.debug("No execution_config provided - skipping torch.compile")
            return
        
        if not self._execution_config.get("enable_torch_compile", False):
            logger.debug("torch.compile disabled in config")
            return
        
        # Check PyTorch version
        try:
            torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
            if torch_version < (2, 0):
                logger.warning(
                    f"torch.compile requires PyTorch 2.0+, got {torch.__version__}. "
                    "Skipping compilation."
                )
                return
        except Exception:
            logger.warning("Could not parse PyTorch version, skipping torch.compile")
            return
        
        # Extract compile parameters
        mode = str(self._execution_config.get("torch_compile_mode", "reduce-overhead"))
        backend = str(self._execution_config.get("torch_compile_backend", "inductor"))
        fullgraph = bool(self._execution_config.get("torch_compile_fullgraph", False))
        dynamic = bool(self._execution_config.get("torch_compile_dynamic", False))
        
        logger.info(
            f"Applying torch.compile: mode={mode}, backend={backend}, "
            f"fullgraph={fullgraph}, dynamic={dynamic}"
        )
        
        # Find the token representation layer (the actual transformer)
        # GLiNER models typically have: model.model.token_rep_layer
        token_rep_layer = None
        compile_target_name = "unknown"
        
        # Try common paths to the transformer encoder
        for path, name in [
            (lambda: self.model.model.token_rep_layer, "model.model.token_rep_layer"),
            (lambda: self.model.token_rep_layer, "model.token_rep_layer"),
            (lambda: self.model.model.encoder, "model.model.encoder"),
            (lambda: self.model.encoder, "model.encoder"),
        ]:
            try:
                candidate = path()
                if candidate is not None and hasattr(candidate, "forward"):
                    token_rep_layer = candidate
                    compile_target_name = name
                    break
            except (AttributeError, TypeError):
                continue
        
        if token_rep_layer is None:
            logger.warning(
                "Could not find suitable transformer layer for torch.compile. "
                "Model structure may differ from expected GLiNER architecture."
            )
            return
        
        try:
            compiled_layer = torch.compile(
                token_rep_layer,
                mode=mode,
                backend=backend,
                fullgraph=fullgraph,
                dynamic=dynamic,
            )
            
            # Replace the layer with the compiled version
            # Navigate to parent and set attribute
            if compile_target_name == "model.model.token_rep_layer":
                self.model.model.token_rep_layer = compiled_layer
            elif compile_target_name == "model.token_rep_layer":
                self.model.token_rep_layer = compiled_layer
            elif compile_target_name == "model.model.encoder":
                self.model.model.encoder = compiled_layer
            elif compile_target_name == "model.encoder":
                self.model.encoder = compiled_layer
            
            self._torch_compiled = True
            logger.info(f"Successfully compiled {compile_target_name} with torch.compile")
            
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}. Continuing without compilation.")
            self._torch_compiled = False

    def _init_cuda_graphs(self) -> Optional["GraphAwareInference"]:
        """
        Initialize CUDA Graph acceleration for inference if enabled.
        
        Reads configuration from self._execution_config to determine:
        - Whether to enable CUDA graphs
        - Shape bucketing configuration
        - Graph cache size and warmup settings
        
        CUDA graphs eliminate kernel launch overhead by capturing and replaying
        GPU operations. Combined with shape bucketing, this provides significant
        speedup for repetitive inference patterns.
        
        Returns:
            GraphAwareInference instance if enabled and on CUDA, None otherwise.
        """
        if self._execution_config is None:
            logger.debug("No execution_config provided - skipping CUDA graphs")
            return None
        
        if not self._execution_config.get("enable_cuda_graphs", False):
            logger.debug("CUDA graphs disabled in config")
            return None
        
        if self.device.type != "cuda":
            logger.debug("CUDA graphs require CUDA device - skipping")
            return None
        
        try:
            from ..hardware_ops.cuda_graphs import (
                create_graph_aware_inference_from_config,
                GraphAwareInference,
            )
            
            # Build config dict from execution_config
            config = {"execution": self._execution_config}
            graph_inference = create_graph_aware_inference_from_config(config)
            
            if graph_inference.is_enabled:
                logger.info(
                    f"CUDA Graph acceleration enabled: "
                    f"cache_size={graph_inference.graph_cache.config.max_cached_graphs}, "
                    f"warmup={graph_inference.graph_cache.config.warmup_iterations}"
                )
                return graph_inference
            else:
                logger.debug("GraphAwareInference created but not enabled")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to initialize CUDA graphs: {e}. Continuing without graph acceleration.")
            return None

    @property
    def is_cuda_graph_enabled(self) -> bool:
        """Check if CUDA graph acceleration is active."""
        return self._graph_inference is not None and self._graph_inference.is_enabled

    def get_cuda_graph_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get CUDA graph cache statistics.
        
        Returns:
            Dict with cache hits, misses, bucket usage, or None if disabled.
        """
        if self._graph_inference is None:
            return None
        return self._graph_inference.get_stats()

    def _sync_processor_max_len(self) -> None:
        """
        Synchronize GLiNER's internal processor config with our BudgetConfig.
        
        CRITICAL: GLiNER's data processor truncates at max_len WORDS (not tokens).
        The processor.preprocess_example() emits warnings like:
            "Sentence of length 778 has been truncated to 512"
        These are WORD counts, not subword tokens.
        
        This method:
        1. Reads max_len from GLiNER processor (the actual WORD limit)
        2. Updates our BudgetConfig.gliner_max_words to match if stricter
        3. Syncs the processor max_len to our configured value
        
        This ensures the SemanticChunker uses the same word limits as GLiNER.
        """
        data_processor = getattr(self.model, "data_processor", None)
        if data_processor is None:
            logger.warning("GLiNER model has no data_processor - cannot sync max_len")
            return
        
        # Get current processor max_len (this is the WORD limit)
        processor_max_len = getattr(data_processor, "max_len", None)
        
        if processor_max_len is not None:
            # Update BudgetConfig if processor has a stricter word limit
            if processor_max_len < self.budget_config.gliner_max_words:
                logger.info(
                    f"GLiNER processor max_len ({processor_max_len} words) is stricter than "
                    f"configured gliner_max_words ({self.budget_config.gliner_max_words}). "
                    f"Using processor limit for safety."
                )
                self.budget_config.gliner_max_words = processor_max_len
            
            if processor_max_len != self.max_length:
                logger.warning(
                    f"GLiNER processor max_len ({processor_max_len}) differs from "
                    f"configured max_length ({self.max_length}). "
                    f"Note: processor max_len is in WORDS, not tokens. "
                    f"Syncing processor to {self.max_length}."
                )
            # Set processor max_len to our configured value
            data_processor.max_len = self.max_length
            logger.debug(f"Synced GLiNER processor max_len to {self.max_length}")
        else:
            logger.debug("GLiNER processor has no max_len attribute")
        
        # Also sync tokenizer model_max_length with BudgetConfig
        tokenizer = self.tokenizer
        tokenizer_max_len = getattr(tokenizer, "model_max_length", None)
        if tokenizer_max_len is not None and tokenizer_max_len < self.budget_config.model_max_length:
            logger.info(
                f"Tokenizer model_max_length ({tokenizer_max_len}) is stricter than "
                f"configured model_max_length ({self.budget_config.model_max_length}). "
                f"Using tokenizer limit for safety."
            )
            self.budget_config.model_max_length = tokenizer_max_len

    def _register_mask_tokens(self) -> None:
        """
        Register typed mask tokens as special tokens in tokenizer.

        Critical: Prevents tokenizer fragmentation of mask strings.
        """
        mask_tokens = list(dict.fromkeys(self.taxonomy.get_mask_tokens().values()))
        tokenizer = self.tokenizer

        # Set explicit max_length
        try:
            tokenizer.model_max_length = int(self.max_length)
        except Exception:
            pass

        # Add special tokens
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": mask_tokens})

        if num_added > 0:
            logger.info(f"Registered {num_added} typed mask tokens as special tokens")
            if not self._resize_model_embeddings(len(tokenizer)):
                raise RuntimeError(
                    "Failed to resize GLiNER embeddings after adding special tokens"
                )

        # Verify single-token encoding
        for mask_token in mask_tokens:
            token_ids = tokenizer.encode(mask_token, add_special_tokens=False)
            if len(token_ids) != 1:
                logger.warning(
                    f"Mask token {mask_token} encoded as {len(token_ids)} tokens: {token_ids}"
                )

    def _resize_model_embeddings(self, new_vocab_size: int) -> bool:
        """Resize the underlying transformer token embeddings."""
        # Search common attribute paths
        for obj in (
            self.model,
            getattr(self.model, "model", None),
            getattr(getattr(self.model, "model", None), "token_rep_layer", None),
            getattr(self.model, "token_rep_layer", None),
            getattr(getattr(self.model, "data_processor", None), "transformer_model", None),
        ):
            if obj is None:
                continue
            fn = getattr(obj, "resize_token_embeddings", None)
            if callable(fn):
                try:
                    sig = inspect.signature(fn)
                    if "mean_resizing" in sig.parameters:
                        fn(new_vocab_size, mean_resizing=False)
                    else:
                        fn(new_vocab_size)
                    return True
                except Exception:
                    pass

        # BFS fallback for nested structures
        visited: set[int] = set()
        queue = deque([self.model])
        steps = 0

        while queue and steps < 500:
            steps += 1
            cur = queue.popleft()
            if cur is None or id(cur) in visited:
                continue
            visited.add(id(cur))

            fn = getattr(cur, "resize_token_embeddings", None)
            if callable(fn):
                try:
                    sig = inspect.signature(fn)
                    if "mean_resizing" in sig.parameters:
                        fn(new_vocab_size, mean_resizing=False)
                    else:
                        fn(new_vocab_size)
                    return True
                except Exception:
                    pass

            try:
                for v in vars(cur).values():
                    if v is not None and hasattr(v, "__dict__"):
                        queue.append(v)
            except Exception:
                continue

        return False

    def get_mask_tokens(self) -> Dict[str, str]:
        """Expose mask tokens for downstream masking."""
        return self.taxonomy.get_mask_tokens()

    def _batched_token_lengths(self, texts: List[str]) -> List[int]:
        """Compute token lengths efficiently for many texts.

        Tokenization/chunking is CPU-bound, but this reduces Python overhead by
        using the tokenizer's batched API (fast tokenizers) when possible.
        """
        if not texts:
            return []

        # Fast path: many HuggingFace fast tokenizers support return_length.
        try:
            encoded = self.tokenizer(
                texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_length=True,
            )
            lengths = encoded.get("length")
            if lengths is not None:
                return [int(x) for x in lengths]
        except Exception:
            pass

        # Fallback: per-text encode (slower but safe).
        return [len(self.tokenizer.encode(t, add_special_tokens=False)) for t in texts]

    def _load_chunk_cache(
        self,
        cache_path: Path,
        doc_count: int,
        total_chars: int,
    ) -> Optional[List[List[ChunkInfo]]]:
        if not cache_path.exists():
            return None

        try:
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:
            logger.warning("Failed to load chunk cache from %s: %s", cache_path, exc)
            return None

        if not isinstance(payload, dict):
            logger.warning("Invalid chunk cache format at %s", cache_path)
            return None

        if payload.get("doc_count") != doc_count or payload.get("total_chars") != total_chars:
            logger.warning("Chunk cache metadata mismatch; ignoring %s", cache_path)
            return None

        chunk_lists = payload.get("chunk_lists")
        if not isinstance(chunk_lists, list) or len(chunk_lists) != doc_count:
            logger.warning("Chunk cache content mismatch; ignoring %s", cache_path)
            return None

        logger.info("Loaded chunk cache from %s", cache_path)
        return chunk_lists

    def _save_chunk_cache(
        self,
        cache_path: Path,
        chunk_lists: List[List[ChunkInfo]],
        doc_count: int,
        total_chars: int,
    ) -> None:
        payload = {
            "version": 1,
            "doc_count": doc_count,
            "total_chars": total_chars,
            "chunk_lists": chunk_lists,
        }

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(f"{cache_path.name}.tmp")
        with tmp_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(cache_path)
        logger.info("Saved chunk cache to %s", cache_path)

    # -----------------------------------------------------------------------
    # Detection Methods
    # -----------------------------------------------------------------------

    def detect_spans(
        self,
        texts: List[str],
        active_columns: Optional[List[Optional[List[str]]]] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        inference_progress_callback: Optional[Any] = None,
        inference_total_callback: Optional[Any] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Detect pollution spans in texts with optional per-row column selection.

        Uses semantic-aware chunking with optimized batched inference.
        
        Performance Optimizations:
        - Batched inference via GLiNER.inference() API
        - Length-based bucketing to minimize padding waste
        - Bi-encoder prompt embedding caching (when available)

        Args:
            texts: List of input texts.
            active_columns: Optional per-text list of SOBR column names with non-null
                values. If provided, only prompts/distractors for those columns are
                used. None or empty list for a text means use all columns.
                Example: [["birth_year", "female"], ["nationality"], None, ...]
            batch_size: Batch size for inference. If None, uses config default.
            show_progress: Show progress bar.
            inference_progress_callback: Optional callback invoked per inference batch/text.
            inference_total_callback: Optional callback invoked with total batches/texts.

        Returns:
            List of detected spans per text. Each span dict contains:
                - text: Detected span text
                - label: Column name (e.g., "birth_year", "female")
                - score: Confidence score
                - start: Character start offset
                - end: Character end offset
        """
        # Use legacy sequential mode if configured or batching is disabled
        if self.budget_config.legacy_sequential_mode or not self.batch_config.enable_batching:
            return self._detect_spans_sequential(
                texts,
                active_columns,
                batch_size,
                show_progress,
                inference_progress_callback,
                inference_total_callback,
            )
        
        # Use optimized batched inference
        return self._detect_spans_batched(
            texts,
            active_columns,
            batch_size,
            show_progress,
            inference_progress_callback,
            inference_total_callback,
        )
    
    def _detect_spans_sequential(
        self,
        texts: List[str],
        active_columns: Optional[List[Optional[List[str]]]] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        inference_progress_callback: Optional[Any] = None,
        inference_total_callback: Optional[Any] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Original sequential detection (legacy mode for backward compatibility).
        
        Processes each text independently with single-chunk inference calls.
        """
        all_results: List[List[Dict[str, Any]]] = []

        iterator = enumerate(texts)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=len(texts),
                desc="GLiNER detection (sequential)",
                unit="text",
                dynamic_ncols=True,
            )
        elif inference_total_callback is not None:
            inference_total_callback(len(texts))

        for idx, text in iterator:
            if not text or not text.strip():
                all_results.append([])
                if inference_progress_callback is not None:
                    inference_progress_callback(1)
                continue

            # Determine active columns for this row
            row_columns: Optional[List[str]] = None
            if active_columns is not None and idx < len(active_columns):
                row_columns = active_columns[idx]

            # Get labels for this row (column-driven or full set)
            if row_columns:
                labels = self.taxonomy.get_prompts_for_columns(row_columns)
            else:
                labels = self.taxonomy.get_inference_labels()

            if not labels:
                all_results.append([])
                continue

            # Chunk text using semantic-aware strategy
            chunks = self.chunker.chunk_text(text, labels)

            if not chunks:
                all_results.append([])
                continue

            # Detect entities in each chunk
            doc_entities: List[Dict[str, Any]] = []

            for chunk_info in chunks:
                chunk_entities = self._detect_in_chunk(
                    chunk_info.text,
                    labels,
                    threshold=self.confidence_threshold,
                )

                # Project offsets to document-global coordinates
                for entity in chunk_entities:
                    projected = project_entity_offsets(entity, chunk_info)
                    doc_entities.append(projected)

            # Deduplicate entities from overlapping chunks
            doc_entities = deduplicate_entities(doc_entities)
            all_results.append(doc_entities)
            if inference_progress_callback is not None:
                inference_progress_callback(1)

        return all_results

    def _detect_spans_batched(
        self,
        texts: List[str],
        active_columns: Optional[List[Optional[List[str]]]] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = True,
        inference_progress_callback: Optional[Any] = None,
        inference_total_callback: Optional[Any] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Optimized batched detection with length bucketing.
        
        Architecture:
        1. Chunk all texts and collect metadata
        2. Group chunks by labels (same-label batching)
        3. Within label groups, bucket by length
        4. Run batched inference per bucket
        5. Map results back to original document coordinates
        """
        effective_batch_size = batch_size or self.batch_config.batch_size
        
        # Step 1: Chunk all texts and collect metadata
        # Structure: List of (doc_idx, chunk_info, labels)
        all_chunks: List[Tuple[int, ChunkInfo, List[str]]] = []
        empty_docs: Set[int] = set()

        chunk_pbar = None
        if show_progress:
            chunk_pbar = tqdm(
                total=len(texts),
                desc="GLiNER chunking",
                unit="text",
                dynamic_ncols=True,
                position=0,
                leave=True,
            )

        doc_labels: List[Optional[List[str]]] = []
        total_chars = 0
        for doc_idx in range(len(texts)):
            text = texts[doc_idx]
            if text:
                total_chars += len(text)
            if not text or not text.strip():
                doc_labels.append(None)
                continue

            row_columns: Optional[List[str]] = None
            if active_columns is not None and doc_idx < len(active_columns):
                row_columns = active_columns[doc_idx]

            if row_columns:
                labels = self.taxonomy.get_prompts_for_columns(row_columns)
            else:
                labels = self.taxonomy.get_inference_labels()

            doc_labels.append(labels if labels else None)

        def _tick_progress() -> None:
            if chunk_pbar:
                chunk_pbar.update(1)

        chunk_lists: Optional[List[List[ChunkInfo]]] = None
        if self.chunk_cache_path is not None:
            chunk_lists = self._load_chunk_cache(
                self.chunk_cache_path,
                len(texts),
                total_chars,
            )

        try:
            if chunk_lists is None:
                chunk_lists = self.chunker.chunk_texts(
                    texts=texts,
                    labels_list=doc_labels,
                    progress_callback=_tick_progress if show_progress else None,
                )
                if self.chunk_cache_path is not None:
                    try:
                        self._save_chunk_cache(
                            self.chunk_cache_path,
                            chunk_lists,
                            len(texts),
                            total_chars,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed to save chunk cache to %s: %s",
                            self.chunk_cache_path,
                            exc,
                        )
            elif chunk_pbar:
                chunk_pbar.update(len(texts))

            for doc_idx, chunks in enumerate(chunk_lists):
                labels = doc_labels[doc_idx]
                if not labels or not chunks:
                    empty_docs.add(doc_idx)
                    continue
                for chunk_info in chunks:
                    all_chunks.append((doc_idx, chunk_info, labels))
        finally:
            if chunk_pbar:
                chunk_pbar.close()
        
        if not all_chunks:
            return [[] for _ in texts]
        
        # Step 2: Group chunks by label set (for batch inference with same labels)
        label_groups: Dict[
            Tuple[str, ...],
            Dict[str, Any],
        ] = {}
        for chunk_idx, (doc_idx, chunk_info, labels) in enumerate(all_chunks):
            ordered_labels = self._order_labels_by_taxonomy(labels)
            label_key = tuple(sorted(ordered_labels))
            group = label_groups.get(label_key)
            if group is None:
                label_groups[label_key] = {
                    "labels": ordered_labels,
                    "chunks": [],
                }
                group = label_groups[label_key]
            elif group["labels"] != ordered_labels:
                logger.debug(
                    "Label order mismatch within label group; using taxonomy order"
                )
            group["chunks"].append((chunk_idx, doc_idx, chunk_info))
        
        # Step 3: Process each label group with length bucketing
        # Results indexed by chunk_idx
        chunk_results: Dict[int, List[Dict[str, Any]]] = {}

        total_chunks = len(all_chunks)
        total_label_groups = len(label_groups)

        bucket_plans: List[Dict[str, Any]] = []
        for group in label_groups.values():
            labels = list(group["labels"])
            group_chunks = group["chunks"]

            # Get cached prompt embeddings if available (bi-encoder only)
            cache_key = tuple(labels)
            cache_hit_before = cache_key in self._prompt_embedding_cache
            prompt_embeddings = self._get_cached_prompt_embeddings(labels)
            cache_hit = bool(cache_hit_before and prompt_embeddings is not None)

            # Compute chunk lengths for bucketing
            chunk_texts = [c[2].text for c in group_chunks]
            chunk_lengths = self._batched_token_lengths(chunk_texts)

            # Create length buckets
            min_bucket_size = max(1, int(self.batch_config.min_bucket_size))
            num_buckets_config = self.batch_config.num_buckets or 1
            max_buckets_by_size = max(1, len(group_chunks) // min_bucket_size)
            num_buckets = min(
                num_buckets_config,
                len(group_chunks),
                max_buckets_by_size,
            )
            if num_buckets < 2:
                # Too few chunks for bucketing, process directly
                bucket_to_indices = {0: list(range(len(group_chunks)))}
            else:
                _, bucket_to_indices = compute_chunk_length_buckets(
                    chunk_lengths, num_buckets
                )
                bucket_to_indices = self._merge_small_buckets(
                    bucket_to_indices,
                    min_bucket_size,
                )

            bucket_plans.append(
                {
                    "labels": labels,
                    "group_chunks": group_chunks,
                    "chunk_texts": chunk_texts,
                    "chunk_lengths": chunk_lengths,
                    "bucket_to_indices": bucket_to_indices,
                    "prompt_embeddings": prompt_embeddings,
                    "cache_hit": cache_hit,
                }
            )

        total_batches = sum(
            (len(bucket_indices) + effective_batch_size - 1) // effective_batch_size
            for plan in bucket_plans
            for bucket_indices in plan["bucket_to_indices"].values()
        )

        if inference_total_callback is not None:
            inference_total_callback(total_batches)

        pbar = None
        if show_progress:
            pbar = tqdm(
                total=total_batches,
                desc="GLiNER inference",
                unit="batch",
                dynamic_ncols=True,
            )
            pbar.set_postfix(
                {
                    "texts": len(texts),
                    "chunks": total_chunks,
                    "groups": total_label_groups,
                    "bs": effective_batch_size,
                },
                refresh=True,
            )

        try:
            for plan in bucket_plans:
                labels = plan["labels"]
                group_chunks = plan["group_chunks"]
                chunk_texts = plan["chunk_texts"]
                chunk_lengths = plan["chunk_lengths"]
                bucket_to_indices = plan["bucket_to_indices"]
                prompt_embeddings = plan["prompt_embeddings"]
                cache_hit = plan["cache_hit"]

                # Process each bucket
                for bucket_indices in bucket_to_indices.values():
                    if not bucket_indices:
                        continue

                    if pbar:
                        pbar.set_postfix(
                            {
                                "labels": len(labels),
                                "group": len(group_chunks),
                                "buckets": len(bucket_to_indices),
                                "cache": "hit" if cache_hit else "miss",
                                "bs": effective_batch_size,
                            },
                            refresh=False,
                        )

                    # Sort by length within bucket for optimal padding
                    sorted_indices = sorted(
                        bucket_indices,
                        key=lambda i: chunk_lengths[i],
                    )

                    # Process in batches
                    for batch_start in range(0, len(sorted_indices), effective_batch_size):
                        batch_indices = sorted_indices[batch_start:batch_start + effective_batch_size]
                        batch_texts = [chunk_texts[i] for i in batch_indices]

                        # Run batched inference
                        batch_entities = self._detect_batch(
                            batch_texts, labels, prompt_embeddings
                        )

                        # Store results with original indices
                        for local_idx, entities in enumerate(batch_entities):
                            group_idx = batch_indices[local_idx]
                            chunk_idx = group_chunks[group_idx][0]  # Original chunk index
                            chunk_results[chunk_idx] = entities

                        if pbar:
                            pbar.update(1)
                        if inference_progress_callback is not None:
                            inference_progress_callback(1)
        finally:
            if pbar:
                pbar.close()
        
        # Step 4: Map chunk results back to documents
        # Initialize result containers per document
        doc_entities: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(texts))}
        
        for chunk_idx, (doc_idx, chunk_info, _) in enumerate(all_chunks):
            entities = chunk_results.get(chunk_idx, [])
            
            # Project offsets to document-global coordinates
            for entity in entities:
                projected = project_entity_offsets(entity, chunk_info)
                doc_entities[doc_idx].append(projected)
        
        # Step 5: Deduplicate entities per document
        all_results: List[List[Dict[str, Any]]] = []
        for doc_idx in range(len(texts)):
            if doc_idx in empty_docs:
                all_results.append([])
            else:
                deduped = deduplicate_entities(doc_entities[doc_idx])
                all_results.append(deduped)
        
        return all_results
    
    def _detect_batch(
        self,
        texts: List[str],
        labels: List[str],
        prompt_embeddings: Optional[torch.Tensor] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Run batched GLiNER inference on multiple texts.
        
        Uses GLiNER.inference() API with optional precomputed prompt embeddings
        for bi-encoder models.
        
        Phase 3 Optimization Note:
            CUDA graphs are initialized via _init_cuda_graphs() but GLiNER's high-level
            inference() API performs Python-level data processing that prevents direct
            graph capture. The GraphAwareInference infrastructure is ready for when
            lower-level tensor hooks (e.g., batch_forward) become available.
            
            Current optimization path: torch.compile on token_rep_layer provides
            similar benefits via Triton kernel fusion with less capture complexity.
        
        Args:
            texts: List of chunk texts.
            labels: Inference labels (same for all texts in batch).
            prompt_embeddings: Precomputed label embeddings (bi-encoder only).
            
        Returns:
            List of entity lists, one per input text.
        """
        if not texts:
            return []
        
        # Build inference kwargs
        inference_kwargs: Dict[str, Any] = {
            "flat_ner": True,
            "threshold": self.confidence_threshold,
            "batch_size": len(texts),  # Process entire batch at once
        }
        
        # Add prompt embeddings for bi-encoder models
        if prompt_embeddings is not None:
            inference_kwargs["labels_embeddings"] = prompt_embeddings
        
        # Run batched inference via GLiNER.inference()
        # Note: CUDA graphs would apply at tensor level if we had direct access
        # to the model's batch_forward method. Currently, torch.compile on
        # token_rep_layer provides the primary compilation benefit.
        inference_kwargs = self._filter_inference_kwargs(inference_kwargs)
        try:
            all_entities = self.model.inference(
                texts,
                labels,
                **inference_kwargs,
            )
        except TypeError as e:
            if self._is_kwarg_typeerror(e):
                # Fallback if inference() doesn't accept certain kwargs
                logger.debug(
                    f"Batch inference kwargs rejected, using minimal args: {e}"
                )
                all_entities = self.model.inference(
                    texts,
                    labels,
                    flat_ner=True,
                    threshold=self.confidence_threshold,
                )
            else:
                raise
        
        # Post-process: filter distractors and apply width constraints
        processed_results: List[List[Dict[str, Any]]] = []
        
        for entities in all_entities:
            # Filter distractors
            entities = self.taxonomy.filter_distractors(entities)
            
            # Apply width constraints
            valid_entities = []
            for entity in entities:
                span_tokens = self.tokenizer.encode(entity["text"], add_special_tokens=False)
                token_count = len(span_tokens)
                max_width = self.constraints.get_max_width(entity["label"])

                if token_count <= max_width:
                    valid_entities.append(entity)
                else:
                    logger.debug(
                        f"Filtered span '{entity['text']}' ({entity['label']}): "
                        f"{token_count} tokens > max {max_width}"
                    )
            
            processed_results.append(valid_entities)
        
        return processed_results

    def _detect_in_chunk(
        self,
        text: str,
        labels: List[str],
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """
        Run GLiNER inference on a single chunk.

        Args:
            text: Chunk text (guaranteed to fit within budget).
            labels: Inference labels.
            threshold: Confidence threshold.

        Returns:
            List of detected entities with chunk-local offsets.
        """
        # Run inference
        entities = self.model.predict_entities(
            text,
            labels,
            threshold=threshold,
            max_length=self.max_length,
        )

        # Filter distractors
        entities = self.taxonomy.filter_distractors(entities)

        # Apply width constraints
        valid_entities = []
        for entity in entities:
            span_tokens = self.tokenizer.encode(entity["text"], add_special_tokens=False)
            token_count = len(span_tokens)
            max_width = self.constraints.get_max_width(entity["label"])

            if token_count <= max_width:
                valid_entities.append(entity)
            else:
                logger.debug(
                    f"Filtered span '{entity['text']}' ({entity['label']}): "
                    f"{token_count} tokens > max {max_width}"
                )

        return valid_entities

    def _run_with_cuda_graphs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        forward_fn: Optional[Any] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run tensor-level inference with optional CUDA graph acceleration.
        
        This method provides the low-level tensor interface for CUDA graphs.
        Use this when you have direct tensor access (e.g., custom inference loops).
        
        Architecture:
        1. If CUDA graphs enabled: bucket inputs → pad → graph replay → unpad
        2. Otherwise: direct forward pass
        
        Args:
            input_ids: Tokenized input IDs (batch, seq).
            attention_mask: Attention mask (batch, seq).
            forward_fn: Optional custom forward function. If None, uses
                        model.model.token_rep_layer.forward.
        
        Returns:
            Dict of output tensors (unpadded to original shape).
        
        Example:
            # For custom low-level inference
            outputs = detector._run_with_cuda_graphs(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
        """
        # Determine forward function
        if forward_fn is None:
            # Try to find the token representation layer
            for path in [
                lambda: self.model.model.token_rep_layer,
                lambda: self.model.token_rep_layer,
                lambda: self.model.model.encoder,
            ]:
                try:
                    layer = path()
                    if layer is not None and hasattr(layer, "forward"):
                        forward_fn = layer.forward
                        break
                except (AttributeError, TypeError):
                    continue
        
        if forward_fn is None:
            raise RuntimeError(
                "Could not find suitable forward function for CUDA graph inference. "
                "Provide forward_fn argument explicitly."
            )
        
        # Use CUDA graphs if enabled
        if self._graph_inference is not None and self._graph_inference.is_enabled:
            # Get pad token ID from tokenizer
            pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
            
            return self._graph_inference.run(
                forward_fn=forward_fn,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=pad_token_id,
            )
        else:
            # Direct execution
            return forward_fn(input_ids=input_ids, attention_mask=attention_mask)

    def detect_spans_long(
        self,
        texts: List[str],
        active_columns: Optional[List[Optional[List[str]]]] = None,
        batch_size: int = 8,
        show_progress: bool = False,
        inference_progress_callback: Optional[Any] = None,
        inference_total_callback: Optional[Any] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Detect pollution spans in long texts.

        This method is now an alias for detect_spans(), which handles all text
        lengths through semantic-aware chunking.

        Args:
            texts: List of input texts (any length).
            active_columns: Optional per-text list of SOBR column names.
            batch_size: Batch size for inference.
            show_progress: Show progress bar.

        Returns:
            List of detected spans per text.
        """
        return self.detect_spans(
            texts,
            active_columns=active_columns,
            batch_size=batch_size,
            show_progress=show_progress,
            inference_progress_callback=inference_progress_callback,
            inference_total_callback=inference_total_callback,
        )

    # -------------------------------------------------------------------------
    # Staged Execution API (Decoupled Chunking & Inference)
    # -------------------------------------------------------------------------

    def chunk_batch(
        self,
        texts: List[str],
        labels_list: Optional[List[Optional[List[str]]]] = None,
        show_progress: bool = False,
    ) -> List[List[ChunkInfo]]:
        """
        Chunk texts without running inference (Stage 1 of staged execution).
        
        Pure CPU operation. Generates ChunkInfo objects with pre-computed
        token_count for zero-copy global sorting in Stage 2.
        
        Args:
            texts: List of documents to chunk.
            labels_list: Per-document labels for budget calculation.
                If None, uses full taxonomy labels for all documents.
            show_progress: Show progress bar.
            
        Returns:
            List of ChunkInfo lists, one per document.
        """
        # Default to full taxonomy labels if not provided
        if labels_list is None:
            all_labels = self.taxonomy.get_inference_labels()
            labels_list = [all_labels for _ in texts]
        
        if len(texts) != len(labels_list):
            raise ValueError(
                f"texts and labels_list must have same length: "
                f"{len(texts)} vs {len(labels_list)}"
            )
        
        def _progress_callback() -> None:
            if show_progress and pbar is not None:
                pbar.update(1)
        
        pbar = None
        if show_progress:
            pbar = tqdm(
                total=len(texts),
                desc="Chunking (Stage 1)",
                unit="text",
                dynamic_ncols=True,
            )
        
        try:
            chunk_lists = self.chunker.chunk_texts(
                texts=texts,
                labels_list=labels_list,
                progress_callback=_progress_callback if show_progress else None,
            )
        finally:
            if pbar is not None:
                pbar.close()
        
        return chunk_lists

    def inference_from_manifest(
        self,
        post_chunked_column: "pa.Array",
        labels: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        cached_label_embeddings: Optional[torch.Tensor] = None,
        pin_memory: bool = False,
        show_progress: bool = True,
        runtime_controller: Optional["RuntimeController"] = None,
    ) -> InferenceResult:
        """
        Run inference from pre-computed chunks (Stage 2 of staged execution).
        
        Implements global sorting for optimal batching with optional autotuning:
        1. Flatten: Extract all chunks from nested Arrow column.
        2. Sort: Global argsort by token_count minimizes padding.
        3. Batch: Create optimal batches (dynamic if runtime_controller provided).
        4. Infer: Run GPU inference with OOM protection (if controller provided).
        5. Gather: Reconstruct per-document results.
        
        Args:
            post_chunked_column: Arrow ListArray from table["post_chunked"].
                Must be typed as list<struct<text, start, end, token_count, is_hard_split>>.
            labels: Inference labels. If None, uses full taxonomy.
            batch_size: Fallback batch size. Ignored if runtime_controller provided.
            cached_label_embeddings: Pre-computed label embeddings from strategy.
                Hoisted to strategy level for bi-encoder efficiency.
            pin_memory: Pin tensors for async GPU transfer (HPC mode).
            show_progress: Show progress bar.
            runtime_controller: Optional RuntimeController for autotuning.
                If provided, uses DynamicBatchIterator with adaptive token budgets
                and OOM protection. If None, uses fixed batch_size batching.
            
        Returns:
            InferenceResult containing:
            - entities: List of entity lists per document (nested, document-order).
            - skipped_doc_indices: Set of doc indices skipped due to OOM failures.
            - oom_count: Number of OOM events during inference.
            - final_budget: Final token budget after adjustments.
        """
        import pyarrow as pa
        from .global_sort import flatten_chunks, gather_results, create_sorted_batches, DynamicBatchIterator

        # Guardrail: Stage 2 requires fully-populated post_chunked (no NULLs).
        try:
            nulls = int(post_chunked_column.null_count)
        except Exception:
            nulls = 0
        if nulls > 0:
            raise ValueError(
                f"post_chunked contains {nulls} NULL rows; run Stage 1 chunking for remaining entries before Stage 2"
            )
        
        effective_batch_size = batch_size or self.batch_config.batch_size
        inference_labels = labels or self.taxonomy.get_inference_labels()
        
        # Use cached embeddings if provided, otherwise compute/retrieve from cache
        prompt_embeddings = cached_label_embeddings
        if prompt_embeddings is None:
            prompt_embeddings = self._get_cached_prompt_embeddings(inference_labels)
        
        # Step 1: Flatten chunks with lineage tracking
        logger.info("Stage 2.1: Flattening post_chunked column")
        flattened = flatten_chunks(
            post_chunked_column,
            extract_texts=True,
            include_chunk_metadata=False,
        )
        
        if flattened.num_chunks == 0:
            logger.warning("No chunks to process")
            return [[] for _ in range(flattened.num_docs)]
        
        logger.info(
            f"Flattened {flattened.num_docs} documents into {flattened.num_chunks} chunks"
        )
        
        # Step 2: Compute global sort permutation
        logger.info("Stage 2.2: Computing global sort permutation by token_count")
        sort_indices = flattened.compute_sort_indices()
        sorted_texts = flattened.get_sorted_texts()
        
        # Step 3: Create batches (dynamic or fixed)
        use_autotuning = runtime_controller is not None
        
        if use_autotuning:
            logger.info("Stage 2.3: Using DynamicBatchIterator with RuntimeController")
            batch_iterator = DynamicBatchIterator(
                flattened=flattened,
                controller=runtime_controller,
                min_batch_size=1,
                max_batch_size=effective_batch_size * 4,  # Upper bound
            )
        else:
            logger.info(f"Stage 2.3: Creating fixed batches (batch_size={effective_batch_size})")
            batches = create_sorted_batches(flattened, effective_batch_size)
            batch_iterator = iter(batches)
            logger.info(f"Created {len(batches)} batches for {flattened.num_chunks} chunks")
        
        # Step 4: Run inference on sorted batches
        # Pre-allocate results in sorted order
        flat_results_sorted: List[List[Dict[str, Any]]] = [[] for _ in range(flattened.num_chunks)]
        
        # Track which chunks were skipped due to unrecoverable OOM
        skipped_chunk_indices: Set[int] = set()
        
        pbar = None
        if show_progress:
            pbar = tqdm(
                total=flattened.num_chunks,
                desc="GLiNER inference (Stage 2)",
                unit="chunk",
                dynamic_ncols=True,
                leave=True,
            )
        
        # Import OOM protection if using autotuning
        if use_autotuning:
            from ..hardware_ops.oom_guard import execute_with_oom_protection, OOMRecoveryError
            from ..hardware_ops.runtime import RuntimeMetrics
            from ..hardware_ops.telemetry import CUDATimer
        
        try:
            for batch_indices in batch_iterator:
                batch_texts = [flattened.texts[i] for i in batch_indices]
                batch_tokens = sum(int(flattened.token_counts[i]) for i in batch_indices)
                
                if use_autotuning:
                    # Run batched inference with OOM protection and timing
                    with CUDATimer() as timer:
                        try:
                            batch_entities = execute_with_oom_protection(
                                lambda bt=batch_texts: self._detect_batch(
                                    bt,
                                    inference_labels,
                                    prompt_embeddings,
                                ),
                                controller=runtime_controller,
                                retry_limit=3,
                            )
                        except OOMRecoveryError:
                            logger.error(
                                f"OOM recovery failed for batch of {len(batch_indices)} chunks; marking as skipped"
                            )
                            # Track skipped chunks for retry
                            for flat_idx in batch_indices:
                                skipped_chunk_indices.add(flat_idx)
                                flat_results_sorted[flat_idx] = []
                            if pbar is not None:
                                pbar.update(len(batch_indices))
                            continue
                    
                    # Report metrics to controller for feedback loop
                    memory_mb = (
                        torch.cuda.memory_allocated() / (1024 ** 2)
                        if torch.cuda.is_available()
                        else 0.0
                    )
                    runtime_controller.report_metrics(RuntimeMetrics(
                        tokens_processed=batch_tokens,
                        batch_time_ms=timer.elapsed_ms,
                        memory_used_mb=memory_mb,
                        batch_size=len(batch_indices),
                    ))
                else:
                    # Run batched inference without autotuning
                    batch_entities = self._detect_batch(
                        batch_texts,
                        inference_labels,
                        prompt_embeddings,
                    )
                
                # Store results at their sorted positions
                for local_idx, entities in enumerate(batch_entities):
                    flat_idx = batch_indices[local_idx]
                    flat_results_sorted[flat_idx] = entities

                    if pbar is not None:
                        # Per-chunk progress update (requested behavior)
                        pbar.update(1)
        finally:
            if pbar is not None:
                pbar.close()
        
        # Log autotuning summary if used
        oom_count = 0
        final_budget = batch_size or self.batch_config.batch_size
        if use_autotuning:
            snapshot = runtime_controller.get_snapshot()
            oom_count = snapshot.oom_count
            final_budget = snapshot.current_budget
            logger.info(
                f"Autotuning summary: state={snapshot.state}, "
                f"final_budget={snapshot.current_budget:,}, "
                f"oom_count={snapshot.oom_count}, "
                f"oom_ceiling={snapshot.oom_ceiling:,}, "
                f"skipped_chunks={len(skipped_chunk_indices)}"
            )
        
        # Step 5: Gather results back to document order
        logger.info("Stage 2.5: Gathering results to document order")
        
        # Results are in flat (document) order, not sorted order
        # Need to project offsets for each chunk
        nested_entities = gather_results(
            flat_results_sorted,
            flattened,
            restore_sort_order=False,  # Results already in flat order
        )
        
        # Apply offset projection and deduplication per document
        # We need chunk metadata to project offsets
        all_results: List[List[Dict[str, Any]]] = []
        
        chunk_starts = flattened.chunk_starts
        if chunk_starts is None:
            chunk_starts = np.zeros(flattened.num_chunks, dtype=np.int32)
        
        for doc_idx, doc_chunk_entities in enumerate(nested_entities):
            doc_start = flattened.doc_offsets[doc_idx]
            doc_end = flattened.doc_offsets[doc_idx + 1]
            
            projected_entities: List[Dict[str, Any]] = []
            
            for local_chunk_idx, chunk_entities in enumerate(doc_chunk_entities):
                flat_idx = doc_start + local_chunk_idx
                
                chunk_start = int(chunk_starts[flat_idx])
                
                # Project each entity's offsets
                for entity in chunk_entities:
                    projected = dict(entity)
                    projected["start"] = projected.get("start", 0) + chunk_start
                    projected["end"] = projected.get("end", 0) + chunk_start
                    projected_entities.append(projected)
            
            # Deduplicate entities from overlapping chunks
            deduped = deduplicate_entities(projected_entities)
            all_results.append(deduped)
        
        # Compute skipped document indices from skipped chunk indices
        skipped_doc_indices: Set[int] = set()
        for flat_idx in skipped_chunk_indices:
            # Find which document this chunk belongs to
            for doc_idx in range(flattened.num_docs):
                doc_start = flattened.doc_offsets[doc_idx]
                doc_end = flattened.doc_offsets[doc_idx + 1]
                if doc_start <= flat_idx < doc_end:
                    skipped_doc_indices.add(doc_idx)
                    break
        
        if skipped_doc_indices:
            logger.warning(
                f"Inference incomplete: {len(skipped_doc_indices)} documents had chunks "
                f"skipped due to OOM ({len(skipped_chunk_indices)} total chunks)"
            )
        
        return InferenceResult(
            entities=all_results,
            skipped_doc_indices=skipped_doc_indices,
            oom_count=oom_count,
            final_budget=final_budget,
        )

    def get_cached_label_embeddings(
        self,
        labels: Optional[List[str]] = None,
    ) -> Optional[torch.Tensor]:
        """
        Get or compute cached label embeddings for bi-encoder models.
        
        This method is exposed for staged execution strategies to hoist
        label embedding computation to the strategy level.
        
        Args:
            labels: Labels to encode. If None, uses full taxonomy.
            
        Returns:
            Cached embeddings tensor, or None if not bi-encoder.
        """
        if labels is None:
            labels = self.taxonomy.get_inference_labels()
        return self._get_cached_prompt_embeddings(labels)

    def _merge_small_buckets(
        self,
        bucket_to_indices: Dict[int, List[int]],
        min_bucket_size: int,
    ) -> Dict[int, List[int]]:
        if min_bucket_size <= 1 or len(bucket_to_indices) <= 1:
            return bucket_to_indices

        merged: List[List[int]] = []
        pending: List[int] = []

        for bucket_id in sorted(bucket_to_indices.keys()):
            indices = bucket_to_indices[bucket_id]
            if not indices:
                continue
            if pending:
                indices = pending + indices
                pending = []
            if len(indices) < min_bucket_size:
                pending = indices
                continue
            merged.append(indices)

        if pending:
            if merged:
                merged[-1].extend(pending)
            else:
                merged.append(pending)

        return {idx: items for idx, items in enumerate(merged)}

    def get_effective_budget(self) -> int:
        """
        Get the current effective text budget.

        Returns:
            Number of text tokens that fit after accounting for prompt costs.
        """
        labels = self.taxonomy.get_inference_labels()
        return self.chunker.calculate_effective_budget(labels)

    def clear_prompt_cache(self) -> int:
        """
        Clear the prompt embedding cache.
        
        Returns:
            Number of cached entries cleared.
        """
        count = len(self._prompt_embedding_cache)
        self._prompt_embedding_cache.clear()
        logger.debug(f"Cleared {count} cached prompt embeddings")
        return count
    
    def get_batch_config(self) -> BatchInferenceConfig:
        """
        Get the current batch inference configuration.
        
        Returns:
            BatchInferenceConfig with current settings.
        """
        return self.batch_config
    
    def set_batch_config(self, config: BatchInferenceConfig) -> None:
        """
        Update batch inference configuration.
        
        Args:
            config: New batch inference configuration.
        """
        self.batch_config = config
        # Re-compute bucket count if set to auto
        if config.num_buckets is None:
            self._configure_auto_bucket_count()
        logger.info(f"Updated batch config: batch_size={config.batch_size}, "
                   f"buckets={self.batch_config.num_buckets}, "
                   f"batching={'enabled' if config.enable_batching else 'disabled'}")

    def _filter_inference_kwargs(self, inference_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            sig = inspect.signature(self.model.inference)
        except (TypeError, ValueError):
            return inference_kwargs

        for param in sig.parameters.values():
            if param.kind == param.VAR_KEYWORD:
                return inference_kwargs

        allowed = set(sig.parameters.keys())
        return {k: v for k, v in inference_kwargs.items() if k in allowed}

    @staticmethod
    def _is_kwarg_typeerror(error: TypeError) -> bool:
        message = str(error).lower()
        return (
            "unexpected keyword argument" in message
            or "multiple values for keyword argument" in message
        )
    
    @property
    def is_bi_encoder(self) -> bool:
        """Check if model supports bi-encoder prompt embedding caching."""
        return self._is_bi_encoder
    
    @property
    def is_torch_compiled(self) -> bool:
        """Check if torch.compile was successfully applied to the model."""
        return self._torch_compiled
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for debugging/monitoring.
        
        Returns:
            Dict with cache stats.
        """
        return {
            "prompt_cache_entries": len(self._prompt_embedding_cache),
            "prompt_cache_labels": [
                list(k) for k in self._prompt_embedding_cache.keys()
            ],
            "is_bi_encoder": self._is_bi_encoder,
            "prompt_caching_enabled": self.batch_config.enable_prompt_caching,
            "torch_compiled": self._torch_compiled,
        }
