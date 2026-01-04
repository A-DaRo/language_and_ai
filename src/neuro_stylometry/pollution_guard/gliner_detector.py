"""
GLiNER Pollution Detector for SOBR Corpus.

Implements the Symbolic Layer of Phase A with critical SOBR-specific safeguards:
- Taxonomy decoupling (schema columns != prompt labels)
- Tokenizer-safe typed masks (registered as special tokens)
- Distractor prompts to absorb non-self mentions
- Center-window retention for long posts (>512 tokens)
- Entity-specific width constraints

Reference: GLiNER_Implementation_Strategy.md Sections 2.1-2.4
Implements: FR-05 (GLiNER Integration), FR-06 (Chunking), FR-07 (Precision Filters)
"""

import logging
import torch
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
from gliner import GLiNER
from collections import deque
from tqdm import tqdm
import inspect
import warnings

logger = logging.getLogger(__name__)


@dataclass
class SOBRTaxonomy:
    """
    SOBR-specific taxonomy mapping.
    
    Maps SOBR schema concepts to semantically anchored GLiNER prompts.
    Critical: Do NOT use schema column names directly (e.g., "judging" triggers verbs).
    
    Implements: GLiNER_Implementation_Strategy.md Section 2.1
    """

    # Target labels (internal IDs) mapped to GLiNER prompt strings.
    # Prompts are semantic anchors, not schema column names.
    target_label_prompts: Dict[str, List[str]] = field(default_factory=lambda: {
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
    })

    # Distractor labels: absorb non-self demographic mentions
    distractor_labels: List[str] = field(default_factory=lambda: [
        "third person reference",
        "third-person demographic mention",
        "quote attribution",
        "historical figure",
        "fictional character",
    ])

    # Internal label -> typed mask token mapping (one-to-one)
    mask_tokens: Dict[str, str] = field(default_factory=lambda: {
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
    })

    # Internal label -> reference regex patterns (for explicit recall)
    reference_patterns: Dict[str, List[str]] = field(default_factory=lambda: {
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
    })

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SOBRTaxonomy":
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
        # GLiNER is prompted with free-text label strings. For reliable detection,
        # we map our internal IDs to model-friendly prompts.
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
        # Fast-path: already internal
        if label in self.get_target_labels() or label in self.distractor_labels:
            return label
        return self._prompt_to_internal(label)

    def _prompt_to_internal(self, prompt_label: str) -> str:
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
    
    # Maximum token widths per entity type
    max_widths: Dict[str, int] = field(default_factory=lambda: {
        "age_statement": 5,          # "I am 25 years old"
        "birth_year_statement": 6,   # "I was born in 1995"
        "gender_indicator": 4,       # "As a woman"
        "nationality_statement": 5,  # "I'm German American"
        "country_of_origin": 5,      # "I'm from New Zealand"
        "demonym": 4,                # "I'm a Brit"
        "personality_type_identifier": 6,  # "I'm an INTP person"
        "mbti_type": 4,              # "I'm INTJ"
        "political_affiliation": 5,  # "I'm a liberal democrat"
        "ideology_self_id": 5,       # "I'm left-wing progressive"
        "default": 12,               # Fallback
    })

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EntityWidthConstraints":
        """Create width constraints from a configuration dictionary."""
        default = cls()
        merged = dict(default.max_widths)
        merged.update({k: int(v) for k, v in config.items()})
        return cls(max_widths=merged)
    
    def get_max_width(self, entity_type: str) -> int:
        """Get maximum token width for entity type."""
        return self.max_widths.get(entity_type, self.max_widths["default"])


class GLiNERDetector:
    """
    GLiNER-based pollution span detector with SOBR-specific safeguards.
    
    Key Features:
    - Tokenizer-safe mask registration (prevents fragmentation)
    - Center-window retention for long posts
    - Precision filtering (width constraints + confidence threshold)
    - Distractor-aware inference
    
    Implements: FR-05, FR-06, FR-07
    """
    
    def __init__(
        self,
        model_name: str = "urchade/gliner_large-v2.1",
        device: str = "cuda",
        max_length: int = 512,
        confidence_threshold: float = 0.85,
        center_window_keep: int = 100,  # Keep center 100 tokens for long posts
        taxonomy: Optional[SOBRTaxonomy] = None,
        constraints: Optional[EntityWidthConstraints] = None,
        taxonomy_config: Optional[Dict[str, Any]] = None,
        constraints_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize GLiNER detector with SOBR safeguards.
        
        Args:
            model_name: GLiNER model identifier.
            device: PyTorch device (cuda/cpu).
            max_length: Maximum sequence length for GLiNER.
            confidence_threshold: Minimum confidence for accepting spans (>= 0.85).
            center_window_keep: Number of center tokens to keep in long posts.
            taxonomy: SOBR taxonomy (defaults to SOBRTaxonomy()).
            constraints: Entity width constraints (defaults to EntityWidthConstraints()).
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.confidence_threshold = confidence_threshold
        self.center_window_keep = center_window_keep
        
        if taxonomy is None and taxonomy_config is not None:
            taxonomy = SOBRTaxonomy.from_config(taxonomy_config)
        if constraints is None and constraints_config is not None:
            constraints = EntityWidthConstraints.from_config(constraints_config)

        self.taxonomy = taxonomy or SOBRTaxonomy()
        self.constraints = constraints or EntityWidthConstraints()
        
        # Load GLiNER model (CPU first to avoid CUDA OOM during `from_pretrained` move)
        cache_key = (model_name, str(self.device), int(max_length))
        cached = getattr(GLiNERDetector, "_MODEL_CACHE", {}).get(cache_key)
        if cached is not None:
            self.model = cached
        else:
            logger.info(f"Loading GLiNER model: {model_name}")
            model = GLiNER.from_pretrained(
                model_name,
                map_location="cpu",
                max_length=max_length,
            )

            # Move to requested device (may still be CPU depending on availability)
            if self.device.type == "cuda":
                # Best-effort VRAM reduction
                try:
                    model = model.half()
                except Exception:
                    pass
            try:
                model = model.to(self.device)
            except Exception:
                # If moving fails for any reason, keep on CPU but keep detector consistent.
                model = model.to("cpu")
                self.device = torch.device("cpu")

            model.eval()
            self.model = model

            if not hasattr(GLiNERDetector, "_MODEL_CACHE"):
                GLiNERDetector._MODEL_CACHE = {}
            GLiNERDetector._MODEL_CACHE[cache_key] = self.model

        # Register typed mask tokens as special tokens (CRITICAL: prevents fragmentation)
        self._register_mask_tokens()
        
        logger.info(f"GLiNERDetector initialized on {self.device}")
    
    def _register_mask_tokens(self) -> None:
        """
        Register typed mask tokens as special tokens in tokenizer.
        
        Critical: Prevents tokenizer fragmentation of mask strings.
        Implements: GLiNER_Implementation_Strategy.md Section 1.2
        """
        # Define typed mask tokens (from taxonomy mapping)
        mask_tokens = list(dict.fromkeys(self.taxonomy.get_mask_tokens().values()))
        
        # Access the underlying transformer tokenizer
        tokenizer = self.model.data_processor.transformer_tokenizer

        # Ensure GLiNER/HF tokenization has an explicit max_length to avoid
        # "Default to no truncation" behavior on some tokenizers.
        try:
            tokenizer.model_max_length = int(self.max_length)
        except Exception:
            pass
        
        # Add special tokens (only if not already present)
        num_added = tokenizer.add_special_tokens({'additional_special_tokens': mask_tokens})
        
        if num_added > 0:
            logger.info(f"Registered {num_added} typed mask tokens as special tokens")
            # Resize model embeddings to accommodate new tokens (robustly)
            resized = self._resize_model_embeddings(len(tokenizer))
            if not resized:
                raise RuntimeError(
                    "Failed to resize GLiNER embeddings after adding special tokens; "
                    "mask tokens may cause out-of-range token IDs."
                )
        
        # Verify single-token encoding
        for mask_token in mask_tokens:
            token_ids = tokenizer.encode(mask_token, add_special_tokens=False)
            if len(token_ids) != 1:
                logger.warning(
                    f"Mask token {mask_token} encoded as {len(token_ids)} tokens: {token_ids}"
                )

    def get_mask_tokens(self) -> Dict[str, str]:
        """Expose mask tokens for downstream masking."""
        return self.taxonomy.get_mask_tokens()

    def _resize_model_embeddings(self, new_vocab_size: int) -> bool:
        """Resize the underlying transformer token embeddings.

        GLiNER wraps an underlying transformer/encoder. Depending on the GLiNER
        version/model class, `resize_token_embeddings` may live on different
        nested objects. We search dynamically rather than hardcoding depth.
        """

        # Fast-paths for common attribute layouts
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

        # Generic BFS over reachable attributes (bounded to avoid cycles)
        visited: set[int] = set()
        q = deque([self.model])
        steps = 0
        while q and steps < 500:
            steps += 1
            cur = q.popleft()
            if cur is None:
                continue
            cur_id = id(cur)
            if cur_id in visited:
                continue
            visited.add(cur_id)

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
                    # Keep searching; some wrappers may expose the method but fail.
                    pass

            # Enqueue children from __dict__ to avoid triggering properties.
            try:
                for v in vars(cur).values():
                    if v is None:
                        continue
                    # Traverse modules, processors, token layers, etc.
                    if hasattr(v, "__dict__") or isinstance(v, (list, tuple, dict)):
                        q.append(v)
                # Expand simple containers
                if isinstance(cur, dict):
                    for v in cur.values():
                        q.append(v)
                elif isinstance(cur, (list, tuple)):
                    for v in cur:
                        q.append(v)
            except Exception:
                continue

        return False
    
    def detect_spans(
        self,
        texts: List[str],
        batch_size: int = 8,
        show_progress: bool = False,
    ) -> List[List[Dict[str, Any]]]:
        """
        Detect pollution spans in texts (short texts, <= max_length).
        
        Args:
            texts: List of input texts.
            batch_size: Batch size for inference.
            
        Returns:
            List of detected spans per text. Each span is a dict with:
                - text: Detected span text
                - label: Entity type
                - score: Confidence score
                - start: Character start offset
                - end: Character end offset
        """
        # Get inference labels (targets + distractors)
        labels = self.taxonomy.get_inference_labels()
        
        # Run inference.
        # Note: GLiNER's `predict_entities` expects a single text (str) per call.
        all_entities: List[List[Dict[str, Any]]] = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="GLiNER detection", unit="batch")
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            for text in batch_texts:
                entities = self.model.predict_entities(
                    text,
                    labels,
                    threshold=self.confidence_threshold,
                    max_length=self.max_length,
                )
                all_entities.append(entities)
        
        # Post-process: filter distractors and apply width constraints
        filtered_entities = []
        tokenizer = self.model.data_processor.transformer_tokenizer
        
        for text, entities in zip(texts, all_entities):
            # Filter distractors
            entities = self.taxonomy.filter_distractors(entities)
            
            # Apply width constraints
            valid_entities = []
            for entity in entities:
                # Tokenize span to get token count
                span_tokens = tokenizer.encode(entity["text"], add_special_tokens=False)
                token_count = len(span_tokens)
                
                max_width = self.constraints.get_max_width(entity["label"])
                
                if token_count <= max_width:
                    valid_entities.append(entity)
                else:
                    logger.debug(
                        f"Filtered span '{entity['text']}' ({entity['label']}): "
                        f"{token_count} tokens > max {max_width}"
                    )
            
            filtered_entities.append(valid_entities)
        
        return filtered_entities
    
    def detect_spans_long(
        self,
        texts: List[str],
        batch_size: int = 8,
        show_progress: bool = False,
    ) -> List[List[Dict[str, Any]]]:
        """
        Detect pollution spans in long texts with center-window retention.
        
        For texts > max_length, uses overlapping chunking with center-window retention
        to mitigate edge effects.
        
        Implements: GLiNER_Implementation_Strategy.md Section 2.4
        
        Args:
            texts: List of input texts (can be > max_length).
            batch_size: Batch size for inference.
            
        Returns:
            List of detected spans per text (deduplicated across chunks).
        """
        tokenizer = self.model.data_processor.transformer_tokenizer
        all_results = []
        
        text_iter = texts
        if show_progress:
            text_iter = tqdm(texts, desc="GLiNER long-text detection", unit="text")
        for text in text_iter:
            # Tokenize to check length
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Token indices sequence length is longer than the specified maximum sequence length",
                )
                tokens = tokenizer.encode(text, add_special_tokens=False)
            
            if len(tokens) <= self.max_length:
                # Short text: use standard detection
                result = self.detect_spans(
                    [text],
                    batch_size=batch_size,
                    show_progress=False,
                )[0]
                all_results.append(result)
            else:
                # Long text: use chunking with center-window retention
                result = self._detect_with_chunking(
                    text,
                    tokens,
                    tokenizer,
                    batch_size,
                    threshold=self.confidence_threshold,
                )
                # Some models degrade confidence on long/repetitive inputs. If we
                # get zero spans, retry with a lower long-text-only threshold.
                if not result and self.confidence_threshold > 0.5:
                    result = self._detect_with_chunking(
                        text,
                        tokens,
                        tokenizer,
                        batch_size,
                        threshold=0.5,
                    )
                all_results.append(result)
        
        return all_results
    
    def _detect_with_chunking(
        self,
        text: str,
        tokens: List[int],
        tokenizer,
        batch_size: int,
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """
        Detect spans in long text using overlapping chunking.
        
        Args:
            text: Input text.
            tokens: Pre-tokenized token IDs.
            tokenizer: HuggingFace tokenizer.
            batch_size: Batch size.
            
        Returns:
            Deduplicated list of detected spans.
        """
        chunk_size = self.max_length
        if chunk_size <= 0:
            return []

        if self.center_window_keep >= chunk_size or self.center_window_keep <= 0:
            overlap = 0
        else:
            overlap = (chunk_size - self.center_window_keep) // 2

        # Token offsets for the original text (used to map chunk -> original offsets)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Token indices sequence length is longer than the specified maximum sequence length",
                )
                encoded = tokenizer(
                    text,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
            offsets = encoded["offset_mapping"]
        except Exception:
            offsets = None

        if not offsets:
            # Fallback: no offset mapping available; use decode-based chunking
            logger.warning("Tokenizer does not provide offsets; chunk offsets may be imprecise")
            offsets = [(0, 0)] * len(tokens)

        chunks = []
        chunk_char_offsets = []

        # Create overlapping chunks in token space
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            char_start = offsets[start][0]
            char_end = offsets[end - 1][1] if end - 1 < len(offsets) else len(text)
            chunk_text = text[char_start:char_end]

            chunks.append(chunk_text)
            chunk_char_offsets.append(char_start)

            if end >= len(tokens):
                break

            start += max(1, chunk_size - overlap)

        # Detect spans in all chunks
        all_chunk_entities = self._detect_spans_with_threshold(
            chunks,
            batch_size=batch_size,
            threshold=threshold,
        )

        collected_entities = []

        for chunk_idx, (chunk_text, entities, char_offset) in enumerate(
            zip(chunks, all_chunk_entities, chunk_char_offsets)
        ):
            is_first = (chunk_idx == 0)
            is_last = (chunk_idx == len(chunks) - 1)

            # Build token offsets for the chunk to do center-window filtering
            try:
                chunk_encoded = tokenizer(
                    chunk_text,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                chunk_offsets = chunk_encoded["offset_mapping"]
            except Exception:
                chunk_offsets = []

            if not chunk_offsets or overlap == 0 or is_first or is_last:
                # Keep all entities; adjust offsets to original text
                for entity in entities:
                    entity = dict(entity)
                    entity["start"] += char_offset
                    entity["end"] += char_offset
                    collected_entities.append(entity)
                continue

            left_edge = overlap // 2
            right_edge = len(chunk_offsets) - (overlap // 2)
            if right_edge < left_edge:
                left_edge = 0
                right_edge = len(chunk_offsets)

            if left_edge < len(chunk_offsets):
                char_left = chunk_offsets[left_edge][0]
            else:
                char_left = 0

            if right_edge - 1 < len(chunk_offsets) and right_edge > 0:
                char_right = chunk_offsets[right_edge - 1][1]
            else:
                char_right = len(chunk_text)

            for entity in entities:
                span_start = entity["start"]
                if char_left <= span_start < char_right:
                    entity = dict(entity)
                    entity["start"] += char_offset
                    entity["end"] += char_offset
                    collected_entities.append(entity)

        # Deduplicate by (start, end, label), keeping highest score
        dedup = {}
        for entity in collected_entities:
            key = (entity["start"], entity["end"], entity["label"])
            prev = dedup.get(key)
            if prev is None or entity.get("score", 0.0) > prev.get("score", 0.0):
                dedup[key] = entity

        return list(dedup.values())

    def _detect_spans_with_threshold(
        self,
        texts: List[str],
        batch_size: int,
        threshold: float,
    ) -> List[List[Dict[str, Any]]]:
        """Internal helper to run detection with an explicit threshold."""
        labels = self.taxonomy.get_inference_labels()

        all_entities: List[List[Dict[str, Any]]] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            for text in batch_texts:
                entities = self.model.predict_entities(
                    text,
                    labels,
                    threshold=threshold,
                    max_length=self.max_length,
                )
                all_entities.append(entities)

        # Post-process: filter distractors and apply width constraints
        filtered_entities = []
        tokenizer = self.model.data_processor.transformer_tokenizer

        for text, entities in zip(texts, all_entities):
            entities = self.taxonomy.filter_distractors(entities)

            valid_entities = []
            for entity in entities:
                span_tokens = tokenizer.encode(entity["text"], add_special_tokens=False)
                token_count = len(span_tokens)
                max_width = self.constraints.get_max_width(entity["label"])

                if token_count <= max_width:
                    valid_entities.append(entity)
                else:
                    logger.debug(
                        f"Filtered span '{entity['text']}' ({entity['label']}): "
                        f"{token_count} tokens > max {max_width}"
                    )

            filtered_entities.append(valid_entities)

        return filtered_entities
