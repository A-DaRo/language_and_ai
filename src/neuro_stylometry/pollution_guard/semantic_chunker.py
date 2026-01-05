"""
Semantic-Aware Context Management for GLiNER.

Implements Dynamic Prompt-Aware Budgeting + Sentence-Boundary Chunking to ensure
GLiNER inputs never exceed max_length while preserving semantic context.

Key Features:
- Dynamic budget calculation accounting for prompt/label token costs
- PySBD-based sentence boundary detection for semantic integrity
- Fallback hard-slicing for pathological run-on sentences
- Offset tracking for global coordinate reconstruction

Reference: GLiNER_Implementation_Strategy.md, Architectural Proposal for Semantic-Aware Context
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any

import pysbd

logger = logging.getLogger(__name__)


@dataclass
class ChunkInfo:
    """
    Metadata for a text chunk with offset tracking.

    Attributes:
        text: The chunk text content.
        char_start: Character offset of chunk start in original document.
        char_end: Character offset of chunk end in original document.
        is_hard_split: True if this chunk resulted from fallback hard-slicing.
    """

    text: str
    char_start: int
    char_end: int
    is_hard_split: bool = False


@dataclass
class BudgetConfig:
    """
    Configuration for dynamic prompt-aware budgeting.

    Attributes:
        model_max_length: Absolute maximum sequence length (e.g., 512).
        system_overhead: Reserved tokens for [CLS], [SEP], and safety buffer.
        ent_marker_cost: Token cost per label for GLiNER's internal [ENT] markers.
        hard_split_overlap: Token overlap when falling back to hard-slicing.
        min_budget_floor: Minimum effective budget to prevent degenerate chunking.
    """

    model_max_length: int = 512
    system_overhead: int = 5
    ent_marker_cost: int = 1
    hard_split_overlap: int = 50
    min_budget_floor: int = 50


class SemanticChunker:
    """
    Implements Dynamic Prompt-Aware Budgeting + Sentence-Boundary Chunking.

    This chunker ensures GLiNER inputs never exceed max_length while preserving
    semantic context through sentence-aware segmentation.

    Architecture:
    1. Phase A (Budget): Calculate effective text budget after prompt costs.
    2. Phase B (Accumulate): Fill chunks sentence-by-sentence up to budget.
    3. Phase C (Fallback): Hard-split pathological run-on sentences.
    4. Phase D (Offsets): Track character offsets for coordinate reconstruction.
    """

    def __init__(
        self,
        tokenizer: Any,
        config: Optional[BudgetConfig] = None,
        language: str = "en",
    ):
        """
        Initialize the semantic chunker.

        Args:
            tokenizer: HuggingFace tokenizer (from GLiNER's data processor).
            config: Budget configuration. Defaults to BudgetConfig().
            language: Language code for PySBD segmenter.
        """
        self.tokenizer = tokenizer
        self.config = config or BudgetConfig()

        # Initialize PySBD (Rule-based sentence segmenter)
        # clean=False preserves original whitespace and formatting
        self.segmenter = pysbd.Segmenter(language=language, clean=False)

        # Cache for label token costs (computed once per label set)
        self._label_cost_cache: dict[tuple[str, ...], int] = {}

    def calculate_effective_budget(self, labels: List[str]) -> int:
        """
        Phase A: Calculate the hard token limit for text after accounting for prompts.

        GLiNER shares its context window between prompts (labels) and text:
            [ENT] label1 [ENT] label2 ... [SEP] text_tokens

        Formula:
            Effective_Budget = Max_Length - (Label_Tokens + ENT_Markers + System_Overhead)

        Args:
            labels: List of inference labels (targets + distractors).

        Returns:
            Maximum number of text tokens that can safely fit in the model window.
        """
        # Use cache if available
        cache_key = tuple(sorted(labels))
        if cache_key in self._label_cost_cache:
            total_label_cost = self._label_cost_cache[cache_key]
        else:
            # Compute token cost for all labels
            total_label_cost = 0
            for label in labels:
                label_tokens = self.tokenizer.encode(label, add_special_tokens=False)
                # Cost = label tokens + [ENT] marker
                total_label_cost += len(label_tokens) + self.config.ent_marker_cost

            self._label_cost_cache[cache_key] = total_label_cost

        # Calculate effective budget
        total_overhead = total_label_cost + self.config.system_overhead
        effective_budget = self.config.model_max_length - total_overhead

        if effective_budget < self.config.min_budget_floor:
            logger.warning(
                f"Effective text budget is dangerously low ({effective_budget} tokens). "
                f"Consider reducing label count. Using minimum floor: {self.config.min_budget_floor}"
            )
            return self.config.min_budget_floor

        logger.debug(
            f"Budget calculation: max={self.config.model_max_length}, "
            f"label_cost={total_label_cost}, overhead={self.config.system_overhead}, "
            f"effective={effective_budget}"
        )

        return effective_budget

    def chunk_text(
        self,
        text: str,
        labels: List[str],
    ) -> List[ChunkInfo]:
        """
        Segment text into semantically coherent chunks that fit within the token budget.

        Strategy:
        1. Segment text into sentences using PySBD.
        2. Accumulate sentences until budget is reached.
        3. Fall back to hard-slicing for pathological run-on sentences.

        Args:
            text: Input text to chunk.
            labels: Inference labels for budget calculation.

        Returns:
            List of ChunkInfo objects with text and offset metadata.
        """
        if not text or not text.strip():
            return []

        budget = self.calculate_effective_budget(labels)

        # Phase 1: Semantic segmentation via PySBD
        sentences = self.segmenter.segment(text)

        if not sentences:
            return []

        # Track sentence positions in original text for offset mapping
        sentence_positions = self._compute_sentence_positions(text, sentences)

        chunks: List[ChunkInfo] = []
        current_sentences: List[Tuple[str, int, int]] = []  # (text, char_start, char_end)
        current_token_count = 0

        for sent_text, char_start, char_end in sentence_positions:
            sent_token_count = len(self.tokenizer.encode(sent_text, add_special_tokens=False))

            # Phase C: Pathological case - single sentence exceeds budget
            if sent_token_count > budget:
                # Flush any accumulated sentences first
                if current_sentences:
                    chunks.append(self._create_chunk(current_sentences, is_hard_split=False))
                    current_sentences = []
                    current_token_count = 0

                # Hard-split the oversized sentence
                hard_chunks = self._hard_split_sentence(
                    sent_text, char_start, budget, self.config.hard_split_overlap
                )
                chunks.extend(hard_chunks)
                continue

            # Phase B: Standard accumulation
            if current_token_count + sent_token_count > budget:
                # Budget exceeded - commit current chunk
                if current_sentences:
                    chunks.append(self._create_chunk(current_sentences, is_hard_split=False))

                # Start new chunk with current sentence
                current_sentences = [(sent_text, char_start, char_end)]
                current_token_count = sent_token_count
            else:
                # Add sentence to current chunk
                current_sentences.append((sent_text, char_start, char_end))
                current_token_count += sent_token_count

        # Commit final buffer
        if current_sentences:
            chunks.append(self._create_chunk(current_sentences, is_hard_split=False))

        return chunks

    def _compute_sentence_positions(
        self,
        text: str,
        sentences: List[str],
    ) -> List[Tuple[str, int, int]]:
        """
        Compute character positions for each sentence in the original text.

        PySBD returns sentence strings but not positions. We reconstruct positions
        by finding each sentence in the original text sequentially.

        Args:
            text: Original text.
            sentences: List of sentence strings from PySBD.

        Returns:
            List of (sentence_text, char_start, char_end) tuples.
        """
        positions = []
        search_start = 0

        for sentence in sentences:
            # Find sentence in text (handle potential whitespace variations)
            sentence_stripped = sentence.strip()
            if not sentence_stripped:
                continue

            # Search for the sentence content
            idx = text.find(sentence_stripped, search_start)
            if idx == -1:
                # Fallback: try original sentence with whitespace
                idx = text.find(sentence, search_start)
                if idx == -1:
                    # Last resort: use current position
                    logger.debug(f"Could not locate sentence position: '{sentence[:50]}...'")
                    idx = search_start

            char_start = idx
            char_end = idx + len(sentence_stripped)

            positions.append((sentence_stripped, char_start, char_end))
            search_start = char_end

        return positions

    def _create_chunk(
        self,
        sentences: List[Tuple[str, int, int]],
        is_hard_split: bool,
    ) -> ChunkInfo:
        """
        Create a ChunkInfo from accumulated sentences.

        Args:
            sentences: List of (text, char_start, char_end) tuples.
            is_hard_split: Whether this chunk resulted from hard-slicing.

        Returns:
            ChunkInfo with concatenated text and span offsets.
        """
        if not sentences:
            return ChunkInfo(text="", char_start=0, char_end=0, is_hard_split=is_hard_split)

        # Join sentences with space (preserving readable text)
        text = " ".join(sent[0] for sent in sentences)
        char_start = sentences[0][1]
        char_end = sentences[-1][2]

        return ChunkInfo(
            text=text,
            char_start=char_start,
            char_end=char_end,
            is_hard_split=is_hard_split,
        )

    def _hard_split_sentence(
        self,
        sentence: str,
        char_offset: int,
        budget: int,
        overlap: int,
    ) -> List[ChunkInfo]:
        """
        Phase C Fallback: Hard-split an oversized sentence by token count.

        This is invoked when a single sentence exceeds the token budget.
        Uses overlapping windows to mitigate context loss at boundaries.

        Args:
            sentence: The oversized sentence text.
            char_offset: Character offset of sentence start in original document.
            budget: Token budget per chunk.
            overlap: Token overlap between consecutive chunks.

        Returns:
            List of ChunkInfo objects from hard-slicing.
        """
        logger.debug(
            f"Hard-splitting pathological sentence "
            f"({len(self.tokenizer.encode(sentence, add_special_tokens=False))} tokens)"
        )

        # Tokenize with offset mapping for precise character boundaries
        try:
            encoded = self.tokenizer(
                sentence,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            token_ids = encoded["input_ids"]
            offsets = encoded["offset_mapping"]
        except Exception:
            # Fallback: simple tokenization without offsets
            token_ids = self.tokenizer.encode(sentence, add_special_tokens=False)
            offsets = None

        if not token_ids:
            return []

        chunks = []
        stride = max(1, budget - overlap)
        start_tok = 0

        while start_tok < len(token_ids):
            end_tok = min(start_tok + budget, len(token_ids))

            # Decode chunk text
            chunk_token_ids = token_ids[start_tok:end_tok]
            chunk_text = self.tokenizer.decode(chunk_token_ids, skip_special_tokens=True)

            # Compute character offsets
            if offsets and start_tok < len(offsets) and end_tok - 1 < len(offsets):
                local_char_start = offsets[start_tok][0]
                local_char_end = offsets[end_tok - 1][1]
            else:
                # Estimate offsets from token positions
                local_char_start = 0
                local_char_end = len(sentence)

            global_char_start = char_offset + local_char_start
            global_char_end = char_offset + local_char_end

            chunks.append(
                ChunkInfo(
                    text=chunk_text,
                    char_start=global_char_start,
                    char_end=global_char_end,
                    is_hard_split=True,
                )
            )

            if end_tok >= len(token_ids):
                break

            start_tok += stride

        return chunks


def project_entity_offsets(
    entity: dict,
    chunk_info: ChunkInfo,
) -> dict:
    """
    Project entity offsets from chunk-local to document-global coordinates.

    Args:
        entity: Entity dict with 'start' and 'end' keys (chunk-local).
        chunk_info: ChunkInfo with character offset metadata.

    Returns:
        New entity dict with global 'start' and 'end' offsets.
    """
    projected = dict(entity)
    projected["start"] = chunk_info.char_start + entity["start"]
    projected["end"] = chunk_info.char_start + entity["end"]
    return projected


def deduplicate_entities(entities: List[dict]) -> List[dict]:
    """
    Deduplicate entities by (start, end, label), keeping highest confidence.

    Overlapping chunks may detect the same entity multiple times.
    This function merges duplicates, retaining the detection with the highest score.

    Args:
        entities: List of entity dicts with 'start', 'end', 'label', 'score' keys.

    Returns:
        Deduplicated list of entities.
    """
    dedup: dict[Tuple[int, int, str], dict] = {}

    for entity in entities:
        key = (entity["start"], entity["end"], entity["label"])
        existing = dedup.get(key)

        if existing is None or entity.get("score", 0.0) > existing.get("score", 0.0):
            dedup[key] = entity

    return list(dedup.values())
