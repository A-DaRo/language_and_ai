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

Reference: GLiNER_Implementation_Strategy.md Sections 2.1-2.4
Implements: FR-05 (GLiNER Integration), FR-06 (Chunking), FR-07 (Precision Filters)
"""

from __future__ import annotations

import inspect
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
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
            center_window_keep: DEPRECATED - Ignored. Semantic chunking handles this.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.confidence_threshold = confidence_threshold
        self.model_name = model_name
        self.require_bi_encoder = require_bi_encoder

        # Initialize taxonomy and constraints
        if taxonomy is None and taxonomy_config is not None:
            taxonomy = SOBRTaxonomy.from_config(taxonomy_config)
        if constraints is None and constraints_config is not None:
            constraints = EntityWidthConstraints.from_config(constraints_config)

        self.taxonomy = taxonomy or SOBRTaxonomy()
        self.constraints = constraints or EntityWidthConstraints()

        # Budget configuration for semantic chunker
        self.budget_config = budget_config or BudgetConfig(model_max_length=max_length)
        
        # Batch inference configuration
        self.batch_config = batch_inference_config or BatchInferenceConfig()
        
        # Auto-compute bucket count if not specified
        if self.batch_config.num_buckets is None:
            gpu_vram = None
            if torch.cuda.is_available():
                gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            self.batch_config.num_buckets = auto_compute_bucket_count(gpu_vram)
            logger.info(f"Auto-computed bucket count: {self.batch_config.num_buckets} (VRAM: {gpu_vram:.1f}GB)" if gpu_vram else f"Auto-computed bucket count: {self.batch_config.num_buckets} (CPU mode)")

        # Load GLiNER model
        self.model = self._load_model(model_name)
        
        # Sync GLiNER processor max_len with our configured max_length
        self._sync_processor_max_len()

        # Register typed mask tokens as special tokens
        self._register_mask_tokens()

        # Extract GLiNER's words_splitter for exact word counting (1:1 parity)
        words_splitter = self._get_words_splitter()

        # Initialize semantic chunker with injected words_splitter
        self.chunker = SemanticChunker(
            tokenizer=self.tokenizer,
            config=self.budget_config,
            words_splitter=words_splitter,
        )
        
        # Bi-encoder prompt embedding cache
        # Key: tuple(sorted(labels)), Value: precomputed embeddings tensor
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
            
        cache_key = tuple(sorted(labels))
        
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

    # -----------------------------------------------------------------------
    # Detection Methods
    # -----------------------------------------------------------------------

    def detect_spans(
        self,
        texts: List[str],
        active_columns: Optional[List[Optional[List[str]]]] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = False,
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

        Returns:
            List of detected spans per text. Each span dict contains:
                - text: Detected span text
                - label: Column name (e.g., "birth_year", "female")
                - score: Confidence score
                - start: Character start offset
                - end: Character end offset
        """
        # Use legacy sequential mode if configured
        if self.budget_config.legacy_sequential_mode:
            return self._detect_spans_sequential(
                texts, active_columns, batch_size, show_progress
            )
        
        # Use optimized batched inference
        return self._detect_spans_batched(
            texts, active_columns, batch_size, show_progress
        )
    
    def _detect_spans_sequential(
        self,
        texts: List[str],
        active_columns: Optional[List[Optional[List[str]]]] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = False,
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

        for idx, text in iterator:
            if not text or not text.strip():
                all_results.append([])
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

        return all_results

    def _detect_spans_batched(
        self,
        texts: List[str],
        active_columns: Optional[List[Optional[List[str]]]] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = True,
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
                leave=False,
            )

        doc_labels: List[Optional[List[str]]] = []
        for doc_idx in range(len(texts)):
            text = texts[doc_idx]
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

        try:
            chunk_lists = self.chunker.chunk_texts(
                texts=texts,
                labels_list=doc_labels,
                progress_callback=_tick_progress if show_progress else None,
            )

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
        label_groups: Dict[Tuple[str, ...], List[Tuple[int, int, ChunkInfo]]] = {}
        for chunk_idx, (doc_idx, chunk_info, labels) in enumerate(all_chunks):
            label_key = tuple(sorted(labels))
            if label_key not in label_groups:
                label_groups[label_key] = []
            label_groups[label_key].append((chunk_idx, doc_idx, chunk_info))
        
        # Step 3: Process each label group with length bucketing
        # Results indexed by chunk_idx
        chunk_results: Dict[int, List[Dict[str, Any]]] = {}
        
        total_batches = sum(
            (len(chunks) + effective_batch_size - 1) // effective_batch_size
            for chunks in label_groups.values()
        )

        total_chunks = len(all_chunks)
        total_label_groups = len(label_groups)
        
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
            for label_key, group_chunks in label_groups.items():
                labels = list(label_key)
                
                # Get cached prompt embeddings if available (bi-encoder only)
                cache_key = tuple(sorted(labels))
                cache_hit_before = cache_key in self._prompt_embedding_cache
                prompt_embeddings = self._get_cached_prompt_embeddings(labels)
                cache_hit = bool(cache_hit_before and prompt_embeddings is not None)
                
                # Compute chunk lengths for bucketing
                chunk_texts = [c[2].text for c in group_chunks]
                chunk_lengths = self._batched_token_lengths(chunk_texts)
                
                # Create length buckets
                num_buckets = min(self.batch_config.num_buckets, len(group_chunks))
                if num_buckets < 2:
                    # Too few chunks for bucketing, process directly
                    bucket_to_indices = {0: list(range(len(group_chunks)))}
                else:
                    _, bucket_to_indices = compute_chunk_length_buckets(
                        chunk_lengths, num_buckets
                    )
                
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
                        key=lambda i: chunk_lengths[i]
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
        try:
            all_entities = self.model.inference(
                texts,
                labels,
                **inference_kwargs,
            )
        except TypeError as e:
            # Fallback if inference() doesn't accept certain kwargs
            logger.debug(f"Batch inference kwargs rejected, using minimal args: {e}")
            all_entities = self.model.inference(
                texts,
                labels,
                flat_ner=True,
                threshold=self.confidence_threshold,
            )
        
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

    def detect_spans_long(
        self,
        texts: List[str],
        active_columns: Optional[List[Optional[List[str]]]] = None,
        batch_size: int = 8,
        show_progress: bool = False,
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
        )

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
            gpu_vram = None
            if torch.cuda.is_available():
                gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            self.batch_config.num_buckets = auto_compute_bucket_count(gpu_vram)
        logger.info(f"Updated batch config: batch_size={config.batch_size}, "
                   f"buckets={self.batch_config.num_buckets}, "
                   f"batching={'enabled' if config.enable_batching else 'disabled'}")
    
    @property
    def is_bi_encoder(self) -> bool:
        """Check if model supports bi-encoder prompt embedding caching."""
        return self._is_bi_encoder
    
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
        }
