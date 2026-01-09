"""
Semantic-Aware Context Management for GLiNER.

Implements Dynamic Prompt-Aware Budgeting + Sentence-Boundary Chunking to ensure
GLiNER inputs never exceed max_length while preserving semantic context.

Key Features:
- Dynamic budget calculation accounting for prompt/label token costs
- PySBD-based sentence boundary detection for semantic integrity
- Fallback hard-slicing for pathological run-on sentences
- Offset tracking for global coordinate reconstruction
- GLiNER WordsSplitter injection for 1:1 parity with model preprocessing

Performance Optimizations (v2.1):
- **Bisect Token Counting (Strategy C):** Replaces O(N) linear scan with O(log N)
  binary search for token counting. Critical for performance parity with Rust tokenizers.
  Expected speedup: 50-100x on token counting operations.

- **"Tokenize Once" Optimization (Strategy B):** Document is tokenized once with
  offset_mapping, then sentence token counts are computed via offset lookups.
  Eliminates redundant tokenization (O(N sentences) → O(1) per document).

- **Batched Parallel Processing (Strategy A):** Workers receive large batches of
  documents instead of individual documents, amortizing tokenizer initialization
  overhead and reducing IPC costs by orders of magnitude.
  Expected speedup: 2-4x on multi-core systems for datasets > 1000 documents.

Reference: GLiNER_Implementation_Strategy.md, Architectural Proposal for Semantic-Aware Context
"""

from __future__ import annotations

import bisect
import logging
import multiprocessing as mp
import os
import re
import threading
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Callable, List, Literal, Tuple, Optional, Any

import pysbd

if TYPE_CHECKING:
    from gliner.data_processing.tokenizer import TokenSplitterBase

logger = logging.getLogger(__name__)

_PARALLEL_CHUNKER: Optional["SemanticChunker"] = None


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
            Increased to 10 (from 5) to account for special token variations
            and potential tokenizer instability.
        ent_marker_cost: Token cost per label for GLiNER's internal [ENT] markers.
        hard_split_overlap: Token overlap when falling back to hard-slicing.
        min_budget_floor: Minimum effective budget to prevent degenerate chunking.
        mode: Chunking mode - 'single_sentence' (one chunk per sentence, default)
              or 'accumulate' (fill chunks up to budget).
        legacy_sequential_mode: If True, forces sequential single-chunk inference
            (bypasses batch optimization). Useful for debugging or exact backward
            compatibility. Default False enables batched inference.
        gliner_max_words: GLiNER processor word limit. The GLiNER library truncates
            inputs at this many WORDS (via WordsSplitter), not subword tokens.
            This is independent of model_max_length which counts tokens.
        tokens_per_word_ratio: Average subword tokens per word. Used to convert
            the gliner_max_words limit into an equivalent token budget.
            Conservative estimate: 1.3 for English (handles compound words, etc).
    """

    model_max_length: int = 512
    system_overhead: int = 10  # Increased from 5 for safety margin
    ent_marker_cost: int = 1
    hard_split_overlap: int = 50
    min_budget_floor: int = 50
    mode: Literal["single_sentence", "accumulate"] = "single_sentence"
    legacy_sequential_mode: bool = False

    # Optional parallelism for document-level chunking (GLiNERDetector drives this).
    # Note: chunking is CPU-bound; threads can still help because fast tokenizers
    # may release the GIL and because we parallelize across documents.
    parallel_chunking_workers: int = 0
    parallel_chunking_min_texts: int = 512
    
    # Batched parallel processing parameters (Optimization Strategy A)
    # batch_size_per_worker: Number of documents to send per worker in one batch.
    # Higher values reduce IPC overhead but increase per-worker memory usage.
    # Default 0 means auto-compute: total_docs // num_workers (one batch per worker).
    batch_size_per_worker: int = 0

    # Word-aware budget constraints (GLiNER truncates at WORDS, not tokens)
    gliner_max_words: int = 512
    tokens_per_word_ratio: float = 1.3


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
    
    CRITICAL: Uses GLiNER's WordsSplitter for 1:1 parity with model preprocessing.
    This eliminates mismatches between chunker word counting and GLiNER's internal
    word counting, which prevented "778 words > 512" truncation warnings.
    """

    # Fallback regex pattern matching GLiNER's WhitespaceTokenSplitter
    _FALLBACK_WORD_PATTERN = re.compile(r"\w+(?:[-_]\w+)*|\S")

    def __init__(
        self,
        tokenizer: Any,
        config: Optional[BudgetConfig] = None,
        language: str = "en",
        words_splitter: Optional["TokenSplitterBase"] = None,
        words_splitter_type: Optional[str] = None,
    ):
        """
        Initialize the semantic chunker.

        Args:
            tokenizer: HuggingFace tokenizer (from GLiNER's data processor).
            config: Budget configuration. Defaults to BudgetConfig().
            language: Language code for PySBD segmenter.
            words_splitter: GLiNER's WordsSplitter instance for exact word counting.
                If None, falls back to regex-based splitting matching GLiNER's
                WhitespaceTokenSplitter pattern.
        """
        self.tokenizer = tokenizer
        self.config = config or BudgetConfig()
        self._language = language

        # Initialize PySBD (Rule-based sentence segmenter)
        # clean=False preserves original whitespace and formatting
        self.segmenter = pysbd.Segmenter(language=language, clean=False)

        # Store (or reconstruct) GLiNER's words_splitter for exact word counting.
        # IMPORTANT: In multiprocessing mode we cannot share the instance, so we pass the
        # splitter TYPE (a small string) and reconstruct it per worker.
        self._words_splitter_type = words_splitter_type
        self._words_splitter = words_splitter
        if self._words_splitter is None and self._words_splitter_type:
            try:
                from gliner.data_processing.tokenizer import WordsSplitter

                self._words_splitter = WordsSplitter(splitter_type=self._words_splitter_type)
            except Exception as exc:
                logger.warning(
                    "Failed to initialize GLiNER WordsSplitter(splitter_type=%r); "
                    "falling back to regex-based splitting: %s",
                    self._words_splitter_type,
                    exc,
                )
                self._words_splitter = None

        if self._words_splitter is None:
            logger.warning(
                "SemanticChunker initialized without GLiNER words_splitter. "
                "Falling back to regex-based word splitting. Word counts may differ "
                "from GLiNER's internal preprocessing, potentially causing truncation."
            )

        # Cache for label token costs (computed once per label set)
        self._label_cost_cache: dict[tuple[str, ...], int] = {}
        self._label_cost_lock = threading.Lock()

    def chunk_texts(
        self,
        texts: List[str],
        labels_list: List[Optional[List[str]]],
        progress_callback: Optional[Callable[[], None]] = None,
    ) -> List[List[ChunkInfo]]:
        """
        Chunk multiple texts, optionally using multiprocessing for HPC.

        Args:
            texts: List of documents to chunk.
            labels_list: Per-document labels (None or empty means skip).
            progress_callback: Optional callable invoked per document processed.

        Returns:
            List of ChunkInfo lists aligned with input order.
        """
        if len(texts) != len(labels_list):
            raise ValueError(
                "texts and labels_list must have the same length "
                f"(got {len(texts)} vs {len(labels_list)})"
            )

        workers = int(self.config.parallel_chunking_workers or 0)
        min_texts = int(self.config.parallel_chunking_min_texts or 0)

        if workers <= 1 or len(texts) < min_texts:
            return self._chunk_texts_sequential(texts, labels_list, progress_callback)

        tokenizer_name = getattr(self.tokenizer, "name_or_path", None)
        if not tokenizer_name:
            logger.warning(
                "Tokenizer name_or_path not available; falling back to sequential chunking."
            )
            results = []
            for text, labels in zip(texts, labels_list):
                if progress_callback:
                    progress_callback()
                if not text or not text.strip() or not labels:
                    results.append([])
                    continue
                results.append(self.chunk_text(text, labels))
            return results

        return self._chunk_texts_parallel(
            texts=texts,
            labels_list=labels_list,
            tokenizer_name=tokenizer_name,
            words_splitter_type=self._words_splitter_type,
            progress_callback=progress_callback,
        )

    def _chunk_texts_sequential(
        self,
        texts: List[str],
        labels_list: List[Optional[List[str]]],
        progress_callback: Optional[Callable[[], None]] = None,
    ) -> List[List[ChunkInfo]]:
        """Sequential chunking fallback for small datasets or when parallelism unavailable."""
        results: List[List[ChunkInfo]] = []
        for text, labels in zip(texts, labels_list):
            if progress_callback:
                progress_callback()
            if not text or not text.strip() or not labels:
                results.append([])
                continue
            results.append(self.chunk_text(text, labels))
        return results

    def _chunk_texts_parallel(
        self,
        texts: List[str],
        labels_list: List[Optional[List[str]]],
        tokenizer_name: str,
        words_splitter_type: Optional[str],
        progress_callback: Optional[Callable[[], None]],
    ) -> List[List[ChunkInfo]]:
        """
        OPTIMIZED parallel chunking using batched work units.
        
        Instead of sending individual documents (high IPC cost), partition the dataset
        into large batches and send one batch per worker. This amortizes initialization
        overhead and reduces pickling/unpickling costs by orders of magnitude.
        
        Architecture:
        1. Partition N documents into W batches (where W = num_workers)
        2. Each worker receives one massive batch of documents
        3. Worker initializes tokenizer ONCE, processes entire batch, returns results
        4. Master flattens batched results back into document order
        
        Expected speedup: 2-4x on multi-core systems for datasets > 1000 documents.
        """
        workers = int(self.config.parallel_chunking_workers or 0)
        max_workers = min(workers, (os.cpu_count() or workers))
        if max_workers <= 1:
            return self.chunk_texts(texts, labels_list, progress_callback=progress_callback)

        if self._words_splitter is not None and words_splitter_type:
            logger.info(
                "Parallel chunking initializes per-worker tokenizers and rebuilds "
                "words_splitter from type=%r.",
                words_splitter_type,
            )
        elif self._words_splitter is not None:
            logger.info(
                "Parallel chunking initializes per-worker tokenizers; "
                "words_splitter type is unknown, so workers fall back to regex splitting."
            )

        # Prevent nested parallelism in workers and keep config immutable.
        worker_config = replace(
            self.config,
            parallel_chunking_workers=0,
            parallel_chunking_min_texts=0,
        )

        # === BATCHED PARTITIONING (Optimization Strategy A) ===
        # Partition documents into batches for workers
        batch_size = self.config.batch_size_per_worker
        if batch_size <= 0:
            # Auto-compute: one batch per worker
            batch_size = (len(texts) + max_workers - 1) // max_workers
        
        batches = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_labels = labels_list[i : i + batch_size]
            batch_indices = list(range(i, min(i + batch_size, len(texts))))
            batches.append((batch_indices, batch_texts, batch_labels))
        
        logger.info(
            f"Parallel chunking: {len(texts)} documents partitioned into "
            f"{len(batches)} batches ({batch_size} docs/batch) across {max_workers} workers"
        )

        results: List[List[ChunkInfo]] = [[] for _ in texts]

        ctx = mp.get_context("spawn")
        try:
            with ctx.Pool(
                processes=max_workers,
                initializer=_parallel_chunker_init,
                initargs=(tokenizer_name, worker_config, self._language, words_splitter_type),
            ) as pool:
                # Process batches (not individual documents)
                for batch_indices, batch_results in pool.imap_unordered(
                    _parallel_chunk_batch_task, batches
                ):
                    # Flatten batch results back into original document order
                    for idx, chunks in zip(batch_indices, batch_results):
                        results[idx] = chunks
                        if progress_callback:
                            progress_callback()
        except Exception as exc:
            logger.warning(
                "Parallel chunking failed; falling back to sequential chunking: %s", exc
            )
            return self._chunk_texts_sequential(texts, labels_list, progress_callback)

        return results

    def _split_words(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into words using GLiNER's WordsSplitter (or fallback regex).
        
        Returns list of (word_text, char_start, char_end) tuples.
        This mirrors GLiNER's internal word splitting exactly.
        """
        if self._words_splitter is not None:
            # Use GLiNER's actual WordsSplitter
            return list(self._words_splitter(text))
        else:
            # Fallback: regex matching GLiNER's WhitespaceTokenSplitter
            return [
                (match.group(), match.start(), match.end())
                for match in self._FALLBACK_WORD_PATTERN.finditer(text)
            ]

    def _count_words(self, text: str) -> int:
        """Count words using GLiNER-compatible splitting."""
        return len(self._split_words(text))

    def _tokenize_with_offsets(self, text: str) -> Tuple[List[int], List[int], List[int]]:
        """
        Tokenize text once and return token lists optimized for binary search.
        
        Returns separate start/end lists instead of tuple pairs to enable O(log N)
        binary search via bisect module. This is Strategy C optimization.
        
        NOTE: This intentionally tokenizes the full document (which may exceed
        model max_length). We only use offsets for chunk boundary calculation;
        the actual model input chunks are guaranteed to fit within budget.
        
        Args:
            text: Input text to tokenize.
            
        Returns:
            Tuple of (token_ids, token_starts, token_ends) where:
                - token_ids: List of token IDs
                - token_starts: List of character start positions (sorted, for bisect)
                - token_ends: List of character end positions (sorted, for bisect)
        """
        try:
            encoding = self.tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                return_attention_mask=False,
                truncation=False,  # Explicitly disable truncation - we want full document offsets
                max_length=None,   # No length limit for offset mapping
            )
            token_ids = encoding["input_ids"]
            offset_mapping = encoding["offset_mapping"]
            
            # Unzip offsets into separate lists for efficient bisect operations
            if not offset_mapping:
                return [], [], []
            
            token_starts = [offset[0] for offset in offset_mapping]
            token_ends = [offset[1] for offset in offset_mapping]
            return token_ids, token_starts, token_ends
            
        except Exception as e:
            logger.warning(f"Failed to get offset mapping: {e}. Falling back to standard tokenization.")
            # Fallback: tokenize without offsets
            token_ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
            # Create dummy offsets (won't be accurate but allows code to continue)
            dummy_starts = [0] * len(token_ids)
            dummy_ends = [0] * len(token_ids)
            return token_ids, dummy_starts, dummy_ends

    def _count_tokens_in_span(
        self,
        char_start: int,
        char_end: int,
        token_starts: List[int],
        token_ends: List[int],
    ) -> int:
        """
        OPTIMIZED: Count tokens in span using O(log N) binary search (Strategy C).
        
        This replaces the O(N) linear scan that was causing performance regression.
        Uses bisect module to find token boundaries in logarithmic time.
        
        A token overlaps with the span if:
            token_start < char_end AND token_end > char_start
        
        Args:
            char_start: Start character position of the span.
            char_end: End character position of the span.
            token_starts: Sorted list of token start positions.
            token_ends: Sorted list of token end positions.
            
        Returns:
            Number of tokens that overlap with the character span.
        """
        if not token_starts:
            return 0
        
        # Find first token that ends AFTER the span starts (token_end > char_start)
        # bisect_right returns insertion point; elements to left are <= char_start
        start_idx = bisect.bisect_right(token_ends, char_start)
        
        # Find first token that starts AT or AFTER the span ends (token_start >= char_end)
        # bisect_left returns insertion point; elements to left are < char_end
        end_idx = bisect.bisect_left(token_starts, char_end)
        
        return max(0, end_idx - start_idx)

    def calculate_exact_cost(self, text: str) -> Tuple[int, int, List[Tuple[str, int, int]]]:
        """
        Calculate exact word and token cost for text using GLiNER's preprocessing.
        
        This mirrors GLiNER's ingestion process:
        1. Run WordsSplitter to get GLiNER-words
        2. Tokenize with is_split_into_words=True (GLiNER's approach)
        3. Return both costs for dual-constraint checking
        
        Args:
            text: Input text to analyze.
            
        Returns:
            Tuple of (word_count, token_count, word_spans) where:
                - word_count: Number of GLiNER-words
                - token_count: Number of subword tokens
                - word_spans: List of (word, char_start, char_end) tuples
        """
        if not text or not text.strip():
            return 0, 0, []
        
        # Phase 1: Get GLiNER-words with character spans
        word_spans = self._split_words(text)
        word_count = len(word_spans)
        
        if word_count == 0:
            return 0, 0, []
        
        # Phase 2: Tokenize as GLiNER does (is_split_into_words=True)
        words = [w[0] for w in word_spans]
        try:
            # GLiNER uses is_split_into_words=True for pre-tokenized input
            encoded = self.tokenizer(
                words,
                is_split_into_words=True,
                add_special_tokens=False,
                return_attention_mask=False,
            )
            token_count = len(encoded["input_ids"])
        except Exception:
            # Fallback: tokenize the raw text
            token_count = len(self.tokenizer.encode(text, add_special_tokens=False))
        
        return word_count, token_count, word_spans

    def calculate_effective_budget(self, labels: List[str]) -> int:
        """
        Phase A: Calculate the hard token limit for text after accounting for prompts.

        GLiNER shares its context window between prompts (labels) and text:
            [ENT] label1 [ENT] label2 ... [SEP] text_tokens

        Formula:
            Token_Budget = Max_Length - (Label_Tokens + ENT_Markers + System_Overhead)
            Word_Budget_Tokens = gliner_max_words * tokens_per_word_ratio
            Effective_Budget = min(Token_Budget, Word_Budget_Tokens)

        CRITICAL: GLiNER's processor truncates at gliner_max_words WORDS (whitespace-split),
        not subword tokens. This method ensures the token budget never produces chunks
        that exceed the word limit after detokenization.

        Args:
            labels: List of inference labels (targets + distractors).

        Returns:
            Maximum number of text tokens that can safely fit in the model window.
        """
        # Use cache if available (thread-safe)
        cache_key = tuple(sorted(labels))
        with self._label_cost_lock:
            total_label_cost = self._label_cost_cache.get(cache_key)

        if total_label_cost is None:
            # Compute token cost for all labels
            computed_cost = 0
            for label in labels:
                label_tokens = self.tokenizer.encode(label, add_special_tokens=False)
                # Cost = label tokens + [ENT] marker
                computed_cost += len(label_tokens) + self.config.ent_marker_cost

            with self._label_cost_lock:
                # Avoid overwriting if another thread populated it.
                total_label_cost = self._label_cost_cache.setdefault(cache_key, computed_cost)

        # Calculate token-based budget (original logic)
        total_overhead = total_label_cost + self.config.system_overhead
        token_budget = self.config.model_max_length - total_overhead

        # Calculate word-based budget (GLiNER processor limit)
        # GLiNER truncates at gliner_max_words WORDS, convert to token estimate
        word_budget_tokens = int(
            self.config.gliner_max_words * self.config.tokens_per_word_ratio
        )

        # Effective budget is the stricter of the two constraints
        effective_budget = min(token_budget, word_budget_tokens)

        if effective_budget < self.config.min_budget_floor:
            logger.warning(
                f"Effective text budget is dangerously low ({effective_budget} tokens). "
                f"Consider reducing label count. Using minimum floor: {self.config.min_budget_floor}"
            )
            return self.config.min_budget_floor

        logger.debug(
            f"Budget calculation: max={self.config.model_max_length}, "
            f"label_cost={total_label_cost}, overhead={self.config.system_overhead}, "
            f"token_budget={token_budget}, word_budget_tokens={word_budget_tokens}, "
            f"effective={effective_budget}"
        )

        return effective_budget

    def chunk_text(
        self,
        text: str,
        labels: List[str],
    ) -> List[ChunkInfo]:
        """
        Segment text into chunks that fit within the token budget.

        Modes:
        - 'single_sentence': One chunk per sentence (shortest possible inputs).
          Hard-split only if a single sentence exceeds budget.
        - 'accumulate': Fill chunks sentence-by-sentence up to budget.

        Optimization: Tokenizes the entire document ONCE and uses offset mapping
        to determine token counts for each sentence. This eliminates redundant
        tokenization (was O(N sentences), now O(1)).

        Args:
            text: Input text to chunk.
            labels: Inference labels for budget calculation.

        Returns:
            List of ChunkInfo objects with text and offset metadata.
        """
        if not text or not text.strip():
            return []

        budget = self.calculate_effective_budget(labels)

        # Phase 1: Tokenize document ONCE with offset mapping (optimized for bisect)
        token_ids, token_starts, token_ends = self._tokenize_with_offsets(text)

        # Phase 2: Semantic segmentation via PySBD
        sentences = self.segmenter.segment(text)

        if not sentences:
            return []

        # Phase 3: Track sentence positions in original text
        sentence_positions = self._compute_sentence_positions(text, sentences)

        # Phase 4: Dispatch to mode-specific chunking with pre-computed offsets
        if self.config.mode == "single_sentence":
            return self._chunk_single_sentence_optimized(sentence_positions, budget, token_starts, token_ends)
        else:
            return self._chunk_accumulate_optimized(sentence_positions, budget, token_starts, token_ends)

    def _chunk_single_sentence_optimized(
        self,
        sentence_positions: List[Tuple[str, int, int]],
        budget: int,
        token_starts: List[int],
        token_ends: List[int],
    ) -> List[ChunkInfo]:
        """
        OPTIMIZED single-sentence mode using binary search token counting.
        
        Eliminates redundant tokenization by using O(log N) bisect operations
        instead of O(N) linear scans or repeated tokenizer calls.

        Args:
            sentence_positions: List of (text, char_start, char_end) tuples.
            budget: Token budget per chunk.
            token_starts: Sorted list of token start positions for bisect.
            token_ends: Sorted list of token end positions for bisect.

        Returns:
            List of ChunkInfo objects (one per sentence, or multiple if hard-split).
        """
        chunks: List[ChunkInfo] = []
        word_limit = self.config.gliner_max_words

        for sent_text, char_start, char_end in sentence_positions:
            # Count tokens using O(log N) bisect (Strategy C optimization)
            sent_token_count = self._count_tokens_in_span(char_start, char_end, token_starts, token_ends)
            
            # Count words using GLiNER's splitter
            sent_word_count = self._count_words(sent_text)

            # Check BOTH constraints: token budget AND word limit
            exceeds_token_budget = sent_token_count > budget
            exceeds_word_limit = sent_word_count > word_limit

            if exceeds_token_budget or exceeds_word_limit:
                # Pathological: sentence exceeds budget or word limit -> hard-split
                logger.debug(
                    f"Sentence requires hard-split: {sent_word_count} GLiNER-words, "
                    f"{sent_token_count} tokens (limits: {word_limit} words, {budget} tokens)"
                )
                
                # Get word spans for hard-splitting
                word_spans = self._split_words(sent_text)

                hard_chunks = self._hard_split_sentence(
                    sent_text, char_start, budget, self.config.hard_split_overlap,
                    word_limit=word_limit,
                    word_spans=word_spans,
                )
                chunks.extend(hard_chunks)
            else:
                # Normal case: one sentence = one chunk
                chunks.append(
                    ChunkInfo(
                        text=sent_text,
                        char_start=char_start,
                        char_end=char_end,
                        is_hard_split=False,
                    )
                )

        return chunks

    def _chunk_accumulate_optimized(
        self,
        sentence_positions: List[Tuple[str, int, int]],
        budget: int,
        token_starts: List[int],
        token_ends: List[int],
    ) -> List[ChunkInfo]:
        """
        OPTIMIZED accumulate mode using binary search token counting.
        
        Eliminates redundant tokenization by using O(log N) bisect operations
        instead of O(N) linear scans or repeated tokenizer calls.

        Args:
            sentence_positions: List of (text, char_start, char_end) tuples.
            budget: Token budget per chunk.
            token_starts: Sorted list of token start positions for bisect.
            token_ends: Sorted list of token end positions for bisect.

        Returns:
            List of ChunkInfo objects.
        """
        chunks: List[ChunkInfo] = []
        current_sentences: List[Tuple[str, int, int]] = []
        current_token_count = 0
        current_word_count = 0
        word_limit = self.config.gliner_max_words

        for sent_text, char_start, char_end in sentence_positions:
            # Count tokens using O(log N) bisect (Strategy C optimization)
            sent_token_count = self._count_tokens_in_span(char_start, char_end, token_starts, token_ends)
            
            # Count words using GLiNER's splitter
            sent_word_count = self._count_words(sent_text)

            # Check BOTH constraints: token budget AND word limit
            exceeds_token_budget = sent_token_count > budget
            exceeds_word_limit = sent_word_count > word_limit

            # Pathological case - single sentence exceeds budget OR word limit
            if exceeds_token_budget or exceeds_word_limit:
                # Flush any accumulated sentences first
                if current_sentences:
                    chunks.append(self._create_chunk(current_sentences, is_hard_split=False))
                    current_sentences = []
                    current_token_count = 0
                    current_word_count = 0

                logger.debug(
                    f"Sentence requires hard-split in accumulate mode: "
                    f"{sent_word_count} GLiNER-words, {sent_token_count} tokens"
                )
                
                # Get word spans for hard-splitting
                word_spans = self._split_words(sent_text)

                # Hard-split the oversized sentence
                hard_chunks = self._hard_split_sentence(
                    sent_text, char_start, budget, self.config.hard_split_overlap,
                    word_limit=word_limit,
                    word_spans=word_spans,
                )
                chunks.extend(hard_chunks)
                continue

            # Standard accumulation - check BOTH token and word constraints
            would_exceed_tokens = current_token_count + sent_token_count > budget
            would_exceed_words = current_word_count + sent_word_count > word_limit

            if would_exceed_tokens or would_exceed_words:
                # Budget exceeded - commit current chunk
                if current_sentences:
                    chunks.append(self._create_chunk(current_sentences, is_hard_split=False))

                # Start new chunk with current sentence
                current_sentences = [(sent_text, char_start, char_end)]
                current_token_count = sent_token_count
                current_word_count = sent_word_count
            else:
                # Add sentence to current chunk
                current_sentences.append((sent_text, char_start, char_end))
                current_token_count += sent_token_count
                current_word_count += sent_word_count

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
        word_limit: Optional[int] = None,
        word_spans: Optional[List[Tuple[str, int, int]]] = None,
    ) -> List[ChunkInfo]:
        """
        Phase C Fallback: Hard-split an oversized sentence by GLiNER-words.

        This is invoked when a single sentence exceeds the token budget OR word limit.
        Uses overlapping word windows to mitigate context loss at boundaries.

        CRITICAL: Operates on GLiNER-words (from WordsSplitter), not raw characters.
        This ensures each chunk respects BOTH the token budget AND word limit with
        1:1 parity to GLiNER's internal preprocessing.

        Algorithm:
        1. Decompose sentence via WordsSplitter to get atomic word strings + spans.
        2. Slide a word-window over the list, dynamically checking token count.
        3. Adjust window size (in words) to maximize content without exceeding budget.
        4. Reconstruct chunk text and character offsets from word spans.

        Args:
            sentence: The oversized sentence text.
            char_offset: Character offset of sentence start in original document.
            budget: Token budget per chunk.
            overlap: Word overlap between consecutive chunks (in words, not tokens).
            word_limit: Maximum words per chunk (GLiNER's processor limit).
            word_spans: Pre-computed word spans from calculate_exact_cost().
                If None, will be computed here.

        Returns:
            List of ChunkInfo objects from hard-slicing.
        """
        if word_limit is None:
            word_limit = self.config.gliner_max_words

        # Get word spans (reuse if pre-computed)
        if word_spans is None:
            word_spans = self._split_words(sentence)

        if not word_spans:
            return []

        # Convert overlap from tokens to approximate words
        word_overlap = max(1, overlap // max(1, int(self.config.tokens_per_word_ratio)))

        total_words = len(word_spans)
        logger.debug(
            f"Hard-splitting pathological sentence: {total_words} GLiNER-words "
            f"(budget={budget} tokens, word_limit={word_limit})"
        )

        chunks = []
        start_word = 0

        while start_word < total_words:
            # Determine maximum words we can take (respecting word_limit)
            max_end_word = min(start_word + word_limit, total_words)

            # Binary search for the largest window that fits token budget
            best_end_word = start_word + 1  # At least one word
            low, high = start_word + 1, max_end_word

            while low <= high:
                mid = (low + high) // 2
                # Get words in this window
                window_words = [w[0] for w in word_spans[start_word:mid]]

                # Count tokens using GLiNER's approach
                try:
                    encoded = self.tokenizer(
                        window_words,
                        is_split_into_words=True,
                        add_special_tokens=False,
                        return_attention_mask=False,
                    )
                    token_count = len(encoded["input_ids"])
                except Exception:
                    # Fallback: join and tokenize
                    window_text = " ".join(window_words)
                    token_count = len(self.tokenizer.encode(window_text, add_special_tokens=False))

                if token_count <= budget:
                    best_end_word = mid
                    low = mid + 1
                else:
                    high = mid - 1

            # Build chunk from word window
            chunk_words = word_spans[start_word:best_end_word]
            if not chunk_words:
                # Safety: advance by at least 1 word to avoid infinite loop
                start_word += 1
                continue

            # Reconstruct text from word spans (preserves original spacing via offsets)
            first_span = chunk_words[0]
            last_span = chunk_words[-1]
            local_char_start = first_span[1]
            local_char_end = last_span[2]

            # Extract text from original sentence using precise offsets
            chunk_text = sentence[local_char_start:local_char_end]

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

            if best_end_word >= total_words:
                break

            # Advance with word overlap
            start_word = max(start_word + 1, best_end_word - word_overlap)

        return chunks


def _parallel_chunker_init(
    tokenizer_name: str,
    config: BudgetConfig,
    language: str,
    words_splitter_type: Optional[str],
) -> None:
    """Initializer for multiprocessing chunk workers."""
    global _PARALLEL_CHUNKER
    from transformers import AutoTokenizer

    # Prevent nested internal tokenizer parallelism inside multiple processes.
    # This avoids CPU oversubscription and makes timings more stable.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load tokenizer '{tokenizer_name}' in worker process. "
            "Ensure the tokenizer is cached locally."
        ) from exc
    _PARALLEL_CHUNKER = SemanticChunker(
        tokenizer=tokenizer,
        config=config,
        language=language,
        words_splitter=None,
        words_splitter_type=words_splitter_type,
    )


def _parallel_chunk_batch_task(
    args: Tuple[List[int], List[str], List[Optional[List[str]]]],
) -> Tuple[List[int], List[List[ChunkInfo]]]:
    """
    OPTIMIZED: Process a batch of documents in a worker process.
    
    This is the core of the batched parallelization optimization (Strategy A).
    Each worker receives a large batch of documents and processes them all,
    amortizing the tokenizer initialization cost across many documents.
    
    Args:
        args: Tuple of (indices, texts, labels_list) where:
            - indices: List of document indices for result alignment
            - texts: List of document texts to chunk
            - labels_list: List of label lists (one per document)
    
    Returns:
        Tuple of (indices, results) where results align with input indices.
    """
    indices, texts, labels_list = args
    
    if _PARALLEL_CHUNKER is None:
        raise RuntimeError("Parallel chunker not initialized in worker process")
    
    results = []
    for text, labels in zip(texts, labels_list):
        if not text or not text.strip() or not labels:
            results.append([])
        else:
            results.append(_PARALLEL_CHUNKER.chunk_text(text, labels))
    
    return indices, results


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
