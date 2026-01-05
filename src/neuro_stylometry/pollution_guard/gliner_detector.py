"""
GLiNER Pollution Detector for SOBR Corpus.

Implements the Symbolic Layer of Phase A with semantic-aware context management:
- Dynamic Prompt-Aware Budgeting (calculates effective text budget after label costs)
- Sentence-Boundary Chunking via PySBD (preserves semantic context)
- Fallback hard-slicing for pathological run-on sentences
- Tokenizer-safe typed masks (registered as special tokens)
- Entity-specific width constraints

Reference: GLiNER_Implementation_Strategy.md Sections 2.1-2.4
Implements: FR-05 (GLiNER Integration), FR-06 (Chunking), FR-07 (Precision Filters)
"""

from __future__ import annotations

import inspect
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from gliner import GLiNER
from tqdm import tqdm

from .semantic_chunker import (
    BudgetConfig,
    ChunkInfo,
    SemanticChunker,
    deduplicate_entities,
    project_entity_offsets,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Taxonomy & Constraints
# ---------------------------------------------------------------------------


@dataclass
class SOBRTaxonomy:
    """
    SOBR-specific taxonomy mapping.

    Maps SOBR schema concepts to semantically anchored GLiNER prompts.
    Critical: Do NOT use schema column names directly (e.g., "judging" triggers verbs).

    Implements: GLiNER_Implementation_Strategy.md Section 2.1
    """

    # Target labels (internal IDs) mapped to GLiNER prompt strings.
    target_label_prompts: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "age_statement": [
                "age statement",
                "self-identified age",
            ],
            "birth_year_statement": [
                "birth year statement",
                "born in year",
            ],
            "gender_indicator": [
                "gender self-identification",
                "self-identified gender",
            ],
            "nationality_statement": [
                "nationality statement",
                "self-identified nationality",
            ],
            "country_of_origin": [
                "country of origin",
                "place of origin",
            ],
            "demonym": [
                "demonym",
                "nationality adjective",
            ],
            "personality_type_identifier": [
                "personality type identifier",
                "self-identified personality type",
            ],
            "mbti_type": [
                "MBTI type",
                "MBTI identifier",
            ],
            "political_affiliation": [
                "political affiliation",
                "self-identified political affiliation",
            ],
            "ideology_self_id": [
                "political ideology",
                "self-identified political ideology",
            ],
        }
    )

    # Distractor labels: absorb non-self demographic mentions
    distractor_labels: List[str] = field(
        default_factory=lambda: [
            "third person reference",
            "third-person demographic mention",
            "quote attribution",
            "historical figure",
            "fictional character",
        ]
    )

    # Internal label -> typed mask token mapping (one-to-one)
    mask_tokens: Dict[str, str] = field(
        default_factory=lambda: {
            "age_statement": "[MASK:AGE]",
            "birth_year_statement": "[MASK:BIRTH_YEAR]",
            "gender_indicator": "[MASK:GENDER]",
            "nationality_statement": "[MASK:NATIONALITY]",
            "country_of_origin": "[MASK:COUNTRY]",
            "demonym": "[MASK:DEMONYM]",
            "personality_type_identifier": "[MASK:PERSONALITY]",
            "mbti_type": "[MASK:MBTI]",
            "political_affiliation": "[MASK:POLITICAL]",
            "ideology_self_id": "[MASK:IDEOLOGY]",
        }
    )

    # Internal label -> reference regex patterns (for explicit recall)
    reference_patterns: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "age_statement": [
                r"(?i)\bI\s*(?:am|'m)\s*\d{1,2}\b",
            ],
            "birth_year_statement": [
                r"(?i)\b(?:born\s+in|born\s+around)\s*\d{4}\b",
            ],
            "gender_indicator": [
                r"(?i)\bI\s*(?:am|'m)\s*(?:a\s+)?(?:man|woman|male|female)\b",
            ],
            "nationality_statement": [
                r"(?i)\bI\s*(?:am|'m)\s*(?:an?\s+)?[A-Z][a-z]+\b",
            ],
            "country_of_origin": [
                r"(?i)\bI\s*(?:am|'m)\s*from\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b",
            ],
            "mbti_type": [
                r"(?i)\b(?:INTJ|INTP|ENTJ|ENTP|INFJ|INFP|ENFJ|ENFP|ISTJ|ISFJ|ESTJ|ESFJ|ISTP|ISFP|ESTP|ESFP)\b",
            ],
        }
    )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> SOBRTaxonomy:
        """Create taxonomy from a configuration dictionary."""
        default = cls()
        return cls(
            target_label_prompts=config.get("target_label_prompts", default.target_label_prompts),
            distractor_labels=config.get("distractor_labels", default.distractor_labels),
            mask_tokens=config.get("mask_tokens", default.mask_tokens),
            reference_patterns=config.get("reference_patterns", default.reference_patterns),
        )

    def get_inference_labels(self) -> List[str]:
        """
        Get combined labels for GLiNER inference (targets + distractors).

        Returns:
            Deduplicated list of labels for inference.
        """
        labels: List[str] = []
        for prompts in self.target_label_prompts.values():
            labels.extend(prompts)
        labels.extend(self.distractor_labels)
        return list(dict.fromkeys(labels))

    def get_target_labels(self) -> List[str]:
        """Return internal target label IDs."""
        return list(self.target_label_prompts.keys())

    def get_mask_tokens(self) -> Dict[str, str]:
        """Return the internal label -> mask token mapping."""
        return dict(self.mask_tokens)

    def get_reference_patterns(self) -> Dict[str, List[str]]:
        """Return the internal label -> regex patterns mapping."""
        return dict(self.reference_patterns)

    def normalize_label(self, label: str) -> str:
        """Map a prompt label (or internal label) to the internal label ID."""
        if label in self.get_target_labels() or label in self.distractor_labels:
            return label
        return self._prompt_to_internal(label)

    def _prompt_to_internal(self, prompt_label: str) -> str:
        """Reverse lookup: prompt string -> internal label ID."""
        reverse = {}
        for internal, prompts in self.target_label_prompts.items():
            for prompt in prompts:
                reverse[prompt] = internal
        return reverse.get(prompt_label, prompt_label)

    def filter_distractors(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove distractor entities from detection results.

        Args:
            entities: Raw GLiNER detection results.

        Returns:
            Filtered entities containing only target labels.
        """
        normalized = []
        for e in entities:
            e = dict(e)
            e["label"] = self.normalize_label(e.get("label", ""))
            normalized.append(e)
        return [e for e in normalized if e["label"] in self.get_target_labels()]


@dataclass
class EntityWidthConstraints:
    """
    Entity-specific token width constraints.

    Implements: GLiNER_Implementation_Strategy.md Section 2.2
    """

    max_widths: Dict[str, int] = field(
        default_factory=lambda: {
            "age_statement": 5,
            "birth_year_statement": 6,
            "gender_indicator": 4,
            "nationality_statement": 5,
            "country_of_origin": 5,
            "demonym": 4,
            "personality_type_identifier": 6,
            "mbti_type": 4,
            "political_affiliation": 5,
            "ideology_self_id": 5,
            "default": 12,
        }
    )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> EntityWidthConstraints:
        """Create width constraints from a configuration dictionary."""
        default = cls()
        merged = dict(default.max_widths)
        merged.update({k: int(v) for k, v in config.items()})
        return cls(max_widths=merged)

    def get_max_width(self, entity_type: str) -> int:
        """Get maximum token width for entity type."""
        return self.max_widths.get(entity_type, self.max_widths["default"])


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

    Implements: FR-05, FR-06, FR-07
    """

    # Class-level model cache to avoid reloading
    _MODEL_CACHE: Dict[tuple, GLiNER] = {}

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
            center_window_keep: DEPRECATED - Ignored. Semantic chunking handles this.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.confidence_threshold = confidence_threshold

        # Initialize taxonomy and constraints
        if taxonomy is None and taxonomy_config is not None:
            taxonomy = SOBRTaxonomy.from_config(taxonomy_config)
        if constraints is None and constraints_config is not None:
            constraints = EntityWidthConstraints.from_config(constraints_config)

        self.taxonomy = taxonomy or SOBRTaxonomy()
        self.constraints = constraints or EntityWidthConstraints()

        # Budget configuration for semantic chunker
        self.budget_config = budget_config or BudgetConfig(model_max_length=max_length)

        # Load GLiNER model
        self.model = self._load_model(model_name)

        # Register typed mask tokens as special tokens
        self._register_mask_tokens()

        # Initialize semantic chunker
        self.chunker = SemanticChunker(
            tokenizer=self.tokenizer,
            config=self.budget_config,
        )

        logger.info(f"GLiNERDetector initialized on {self.device}")

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

    # -----------------------------------------------------------------------
    # Detection Methods
    # -----------------------------------------------------------------------

    def detect_spans(
        self,
        texts: List[str],
        batch_size: int = 8,
        show_progress: bool = False,
    ) -> List[List[Dict[str, Any]]]:
        """
        Detect pollution spans in texts (any length).

        Uses semantic-aware chunking for texts exceeding the token budget.

        Args:
            texts: List of input texts.
            batch_size: Batch size for inference.
            show_progress: Show progress bar.

        Returns:
            List of detected spans per text. Each span dict contains:
                - text: Detected span text
                - label: Entity type (internal label)
                - score: Confidence score
                - start: Character start offset
                - end: Character end offset
        """
        labels = self.taxonomy.get_inference_labels()
        all_results: List[List[Dict[str, Any]]] = []

        iterator = texts
        if show_progress:
            iterator = tqdm(texts, desc="GLiNER detection", unit="text")

        for text in iterator:
            if not text or not text.strip():
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
        batch_size: int = 8,
        show_progress: bool = False,
    ) -> List[List[Dict[str, Any]]]:
        """
        Detect pollution spans in long texts.

        This method is now an alias for detect_spans(), which handles all text
        lengths through semantic-aware chunking.

        Args:
            texts: List of input texts (any length).
            batch_size: Batch size for inference.
            show_progress: Show progress bar.

        Returns:
            List of detected spans per text.
        """
        return self.detect_spans(texts, batch_size=batch_size, show_progress=show_progress)

    def get_effective_budget(self) -> int:
        """
        Get the current effective text budget.

        Returns:
            Number of text tokens that fit after accounting for prompt costs.
        """
        labels = self.taxonomy.get_inference_labels()
        return self.chunker.calculate_effective_budget(labels)
