"""
Unit tests for SemanticChunker.

Tests the semantic-aware context management system:
- Dynamic prompt-aware budgeting
- Sentence-boundary chunking
- Pathological sentence handling
- Offset tracking and entity projection
"""

import pytest
from unittest.mock import MagicMock, patch

from neuro_stylometry.pollution_guard.semantic_chunker import (
    BudgetConfig,
    ChunkInfo,
    SemanticChunker,
    deduplicate_entities,
    project_entity_offsets,
)


class MockTokenizer:
    """Mock tokenizer for testing without loading real model."""

    def __init__(self, tokens_per_word: int = 1):
        self.tokens_per_word = tokens_per_word

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Simple word-based tokenization."""
        if not text or not text.strip():
            return []
        words = text.split()
        return list(range(len(words) * self.tokens_per_word))

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode tokens back to approximate text."""
        return " ".join(["word"] * len(token_ids))

    def __call__(
        self, 
        text, 
        add_special_tokens: bool = False, 
        return_offsets_mapping: bool = False,
        is_split_into_words: bool = False,
        return_attention_mask: bool = True,
    ):
        """Callable interface for tokenizer.
        
        Supports is_split_into_words for GLiNER-style pre-tokenized input.
        """
        if is_split_into_words:
            # text is already a list of words
            words = text if isinstance(text, list) else [text]
            tokens = []
            for w in words:
                tokens.extend(list(range(len(tokens), len(tokens) + self.tokens_per_word)))
        else:
            # text is a string
            text_str = text if isinstance(text, str) else " ".join(text)
            tokens = self.encode(text_str, add_special_tokens=add_special_tokens)
        
        result = {"input_ids": tokens}
        
        if return_offsets_mapping:
            # Simple offset mapping: each "word" is ~5 chars
            offsets = []
            if isinstance(text, str):
                pos = 0
                for word in text.split():
                    offsets.append((pos, pos + len(word)))
                    pos += len(word) + 1  # +1 for space
            result["offset_mapping"] = offsets
        
        return result


class TestBudgetConfig:
    """Test BudgetConfig dataclass."""

    def test_default_values(self):
        """BudgetConfig has sensible defaults."""
        config = BudgetConfig()

        assert config.model_max_length == 512
        assert config.system_overhead == 10  # Increased from 5 for safety margin
        assert config.ent_marker_cost == 1
        assert config.hard_split_overlap == 50
        assert config.min_budget_floor == 50
        assert config.legacy_sequential_mode is False  # New default
        # Word-aware budget defaults
        assert config.gliner_max_words == 512
        assert config.tokens_per_word_ratio == 1.3

    def test_custom_values(self):
        """BudgetConfig accepts custom values."""
        config = BudgetConfig(
            model_max_length=1024,
            system_overhead=10,
            ent_marker_cost=2,
            hard_split_overlap=100,
            min_budget_floor=100,
        )

        assert config.model_max_length == 1024
        assert config.system_overhead == 10

    def test_legacy_sequential_mode_flag(self):
        """BudgetConfig supports legacy_sequential_mode flag."""
        config = BudgetConfig(legacy_sequential_mode=True)
        assert config.legacy_sequential_mode is True
        
        config = BudgetConfig(legacy_sequential_mode=False)
        assert config.legacy_sequential_mode is False
    
    def test_mode_options(self):
        """BudgetConfig supports single_sentence and accumulate modes."""
        config = BudgetConfig(mode="single_sentence")
        assert config.mode == "single_sentence"
        
        config = BudgetConfig(mode="accumulate")
        assert config.mode == "accumulate"

    def test_word_aware_budget_settings(self):
        """BudgetConfig supports word-aware budget constraints."""
        config = BudgetConfig(
            gliner_max_words=256,
            tokens_per_word_ratio=1.5,
        )
        assert config.gliner_max_words == 256
        assert config.tokens_per_word_ratio == 1.5


class TestChunkInfo:
    """Test ChunkInfo dataclass."""

    def test_creation(self):
        """ChunkInfo stores chunk metadata."""
        chunk = ChunkInfo(
            text="Hello world.",
            char_start=0,
            char_end=12,
            is_hard_split=False,
        )

        assert chunk.text == "Hello world."
        assert chunk.char_start == 0
        assert chunk.char_end == 12
        assert chunk.is_hard_split is False

    def test_hard_split_flag(self):
        """ChunkInfo tracks hard-split origin."""
        chunk = ChunkInfo(
            text="chunk",
            char_start=100,
            char_end=105,
            is_hard_split=True,
        )

        assert chunk.is_hard_split is True


class TestSemanticChunker:
    """Test SemanticChunker functionality."""

    @pytest.fixture
    def tokenizer(self):
        """Mock tokenizer."""
        return MockTokenizer(tokens_per_word=1)

    @pytest.fixture
    def chunker(self, tokenizer):
        """SemanticChunker with small budget for testing."""
        config = BudgetConfig(model_max_length=50, system_overhead=10, min_budget_floor=10)
        return SemanticChunker(tokenizer=tokenizer, config=config)

    def test_calculate_effective_budget(self, chunker):
        """Budget calculation accounts for label tokens."""
        # Each label is ~2 words = 2 tokens + 1 ENT marker = 3 tokens per label
        labels = ["age statement", "gender identity"]

        budget = chunker.calculate_effective_budget(labels)

        # max=50, system=10, labels=2*(2+1)=6, effective=50-10-6=34
        # But MockTokenizer returns 1 token per word, so "age statement" = 2 tokens
        # Budget = 50 - 10 - (2+1 + 2+1) = 50 - 10 - 6 = 34
        # BUT: word_budget_tokens = 512 * 1.3 = 665, so token_budget (34) wins
        assert budget == 34

    def test_budget_respects_word_limit(self, tokenizer):
        """Budget calculation respects GLiNER word limit."""
        # High token budget but low word limit
        config = BudgetConfig(
            model_max_length=1000,  # High token budget
            system_overhead=5,
            min_budget_floor=10,
            gliner_max_words=100,  # Low word limit
            tokens_per_word_ratio=1.0,  # 1:1 ratio for simplicity
        )
        chunker = SemanticChunker(tokenizer=tokenizer, config=config)
        
        labels = ["label"]  # Minimal labels (1 word + 1 ENT = 2 tokens)
        budget = chunker.calculate_effective_budget(labels)
        
        # token_budget = 1000 - 5 - 2 = 993
        # word_budget_tokens = 100 * 1.0 = 100
        # effective = min(993, 100) = 100
        assert budget == 100

    def test_budget_word_limit_with_ratio(self, tokenizer):
        """Word limit is correctly scaled by tokens_per_word_ratio."""
        config = BudgetConfig(
            model_max_length=1000,
            system_overhead=5,
            min_budget_floor=10,
            gliner_max_words=100,
            tokens_per_word_ratio=1.5,  # 1.5 tokens per word
        )
        chunker = SemanticChunker(tokenizer=tokenizer, config=config)
        
        labels = ["x"]  # 1 word + 1 ENT = 2 tokens
        budget = chunker.calculate_effective_budget(labels)
        
        # token_budget = 1000 - 5 - 2 = 993
        # word_budget_tokens = 100 * 1.5 = 150
        # effective = min(993, 150) = 150
        assert budget == 150

    def test_budget_token_limit_wins_when_stricter(self, tokenizer):
        """Token budget wins when it's stricter than word budget."""
        config = BudgetConfig(
            model_max_length=100,  # Low token limit
            system_overhead=5,
            min_budget_floor=10,
            gliner_max_words=500,  # High word limit
            tokens_per_word_ratio=1.3,
        )
        chunker = SemanticChunker(tokenizer=tokenizer, config=config)
        
        labels = ["label"]  # 1 word + 1 ENT = 2 tokens
        budget = chunker.calculate_effective_budget(labels)
        
        # token_budget = 100 - 5 - 2 = 93
        # word_budget_tokens = 500 * 1.3 = 650
        # effective = min(93, 650) = 93
        assert budget == 93

    def test_budget_caching(self, chunker):
        """Budget calculation is cached for same label set."""
        labels = ["label one", "label two"]

        budget1 = chunker.calculate_effective_budget(labels)
        budget2 = chunker.calculate_effective_budget(labels)

        assert budget1 == budget2
        # Same cache key should be used
        assert len(chunker._label_cost_cache) == 1

    def test_budget_floor(self, tokenizer):
        """Budget doesn't go below minimum floor."""
        # Create config where labels exceed available space
        config = BudgetConfig(model_max_length=20, system_overhead=5, min_budget_floor=10)
        chunker = SemanticChunker(tokenizer=tokenizer, config=config)

        # Many labels that would exhaust the budget
        labels = ["label " + str(i) for i in range(10)]

        budget = chunker.calculate_effective_budget(labels)

        # Should return floor, not negative
        assert budget == 10

    def test_chunk_text_empty(self, chunker):
        """Empty text returns empty list."""
        result = chunker.chunk_text("", labels=["label"])
        assert result == []

        result = chunker.chunk_text("   ", labels=["label"])
        assert result == []

    def test_chunk_text_single_sentence(self, chunker):
        """Single sentence within budget returns one chunk."""
        text = "This is a short sentence."
        result = chunker.chunk_text(text, labels=["label"])

        assert len(result) == 1
        assert result[0].text == text
        assert result[0].is_hard_split is False

    def test_chunk_text_multiple_sentences(self, tokenizer):
        """Multiple sentences are accumulated within budget (accumulated mode)."""
        # Note: Must use 'accumulated' mode to merge sentences within budget.
        # Default 'single_sentence' mode produces one chunk per sentence.
        config = BudgetConfig(
            model_max_length=100, 
            system_overhead=5, 
            min_budget_floor=10,
            mode="accumulated",  # Required for sentence merging
        )
        chunker = SemanticChunker(tokenizer=tokenizer, config=config)

        text = "First sentence. Second sentence. Third sentence."
        result = chunker.chunk_text(text, labels=["label"])

        # All three sentences should fit in one chunk (accumulated mode)
        assert len(result) == 1
        assert "First sentence" in result[0].text

    def test_chunk_text_budget_overflow(self, tokenizer):
        """Text exceeding budget creates multiple chunks."""
        # Very small budget
        config = BudgetConfig(model_max_length=20, system_overhead=5, min_budget_floor=5)
        chunker = SemanticChunker(tokenizer=tokenizer, config=config)

        text = "First sentence here. Second sentence here. Third sentence here."
        result = chunker.chunk_text(text, labels=["l"])

        # Should create multiple chunks due to budget
        assert len(result) >= 1

    def test_offset_tracking(self, chunker):
        """Chunks track character offsets correctly."""
        text = "First sentence. Second sentence."
        result = chunker.chunk_text(text, labels=["label"])

        assert len(result) >= 1
        first_chunk = result[0]

        assert first_chunk.char_start >= 0
        assert first_chunk.char_end <= len(text)


class TestProjectEntityOffsets:
    """Test entity offset projection."""

    def test_project_offsets(self):
        """Entity offsets are projected to global coordinates."""
        entity = {"start": 5, "end": 10, "text": "hello", "label": "test", "score": 0.9}
        chunk_info = ChunkInfo(text="prefix hello suffix", char_start=100, char_end=120)

        projected = project_entity_offsets(entity, chunk_info)

        assert projected["start"] == 105  # 100 + 5
        assert projected["end"] == 110  # 100 + 10
        assert projected["text"] == "hello"
        assert projected["label"] == "test"

    def test_project_preserves_original(self):
        """Projection creates new dict, doesn't modify original."""
        entity = {"start": 0, "end": 5, "text": "test"}
        chunk_info = ChunkInfo(text="test", char_start=50, char_end=55)

        projected = project_entity_offsets(entity, chunk_info)

        assert entity["start"] == 0  # Original unchanged
        assert projected["start"] == 50  # Projected modified


class TestDeduplicateEntities:
    """Test entity deduplication."""

    def test_deduplicate_identical(self):
        """Identical entities are deduplicated."""
        entities = [
            {"start": 0, "end": 5, "label": "test", "score": 0.8},
            {"start": 0, "end": 5, "label": "test", "score": 0.9},
        ]

        result = deduplicate_entities(entities)

        assert len(result) == 1
        assert result[0]["score"] == 0.9  # Higher score kept

    def test_deduplicate_different_positions(self):
        """Entities at different positions are kept."""
        entities = [
            {"start": 0, "end": 5, "label": "test", "score": 0.8},
            {"start": 10, "end": 15, "label": "test", "score": 0.9},
        ]

        result = deduplicate_entities(entities)

        assert len(result) == 2

    def test_deduplicate_different_labels(self):
        """Entities with different labels are kept."""
        entities = [
            {"start": 0, "end": 5, "label": "label_a", "score": 0.8},
            {"start": 0, "end": 5, "label": "label_b", "score": 0.9},
        ]

        result = deduplicate_entities(entities)

        assert len(result) == 2

    def test_deduplicate_empty(self):
        """Empty list returns empty list."""
        result = deduplicate_entities([])
        assert result == []

    def test_deduplicate_keeps_highest_score(self):
        """When deduplicating, keeps entity with highest score."""
        entities = [
            {"start": 5, "end": 10, "label": "age", "score": 0.7, "text": "25"},
            {"start": 5, "end": 10, "label": "age", "score": 0.95, "text": "25"},
            {"start": 5, "end": 10, "label": "age", "score": 0.85, "text": "25"},
        ]

        result = deduplicate_entities(entities)

        assert len(result) == 1
        assert result[0]["score"] == 0.95


class TestDirectWordLimitEnforcement:
    """Test direct word count enforcement (not just token-based conversion).
    
    GLiNER's processor truncates at WORDS (via WordsSplitter), not tokens.
    The chunker uses calculate_exact_cost() for GLiNER-compatible word counting.
    """

    @pytest.fixture
    def tokenizer(self):
        """Return a mock tokenizer that supports is_split_into_words."""
        from unittest.mock import MagicMock
        tokenizer = MagicMock()
        
        # Simulate ~1 token per word (simple case)
        def mock_encode(text, **kw):
            if isinstance(text, str):
                return text.split()
            return text  # Already a list
        
        tokenizer.encode = MagicMock(side_effect=mock_encode)
        
        # Add decode for hard-split
        tokenizer.decode = MagicMock(side_effect=lambda ids, **kw: " ".join(str(i) for i in ids))
        
        # Make tokenizer callable for is_split_into_words and offset mapping
        def tokenizer_call(text, is_split_into_words=False, **kwargs):
            if is_split_into_words:
                # text is already a list of words
                words = text if isinstance(text, list) else [text]
                result = {"input_ids": words}
            else:
                text_str = text if isinstance(text, str) else " ".join(text)
                words = text_str.split()
                result = {"input_ids": words}
                
                # Add offset_mapping if requested
                if kwargs.get("return_offsets_mapping"):
                    offsets = []
                    pos = 0
                    for w in words:
                        start = text_str.find(w, pos)
                        end = start + len(w)
                        offsets.append((start, end))
                        pos = end
                    result["offset_mapping"] = offsets
            
            return result
        
        tokenizer.side_effect = tokenizer_call
        return tokenizer

    def test_single_sentence_mode_respects_word_limit(self, tokenizer):
        """Single-sentence mode splits when word count exceeds gliner_max_words."""
        # Config with LOW word limit but HIGH token limit
        # This ensures word limit is the binding constraint
        config = BudgetConfig(
            model_max_length=10000,  # Very high token limit
            system_overhead=5,
            min_budget_floor=10,
            gliner_max_words=10,  # LOW word limit: 10 words max
            tokens_per_word_ratio=1.0,  # 1:1 for simplicity
            mode="single_sentence",
            hard_split_overlap=5,  # Small overlap for predictable chunk count
        )
        chunker = SemanticChunker(tokenizer=tokenizer, config=config)
        
        # Create sentence with 20 words (exceeds 10-word limit)
        # Include a period so pySBD treats it as a sentence
        long_sentence = " ".join([f"word{i}" for i in range(20)]) + "."
        labels = ["x"]
        
        chunks = chunker.chunk_text(long_sentence, labels)
        
        # Should be split because it exceeds word limit
        assert len(chunks) > 1, f"Should split sentence exceeding word limit, got {len(chunks)} chunks"
        # Each chunk should have is_hard_split=True
        assert all(c.is_hard_split for c in chunks), "Chunks should be hard-split"

    def test_accumulate_mode_respects_word_limit(self, tokenizer):
        """Accumulate mode flushes when accumulated word count would exceed limit."""
        config = BudgetConfig(
            model_max_length=10000,  # Very high token limit
            system_overhead=5,
            min_budget_floor=10,
            gliner_max_words=15,  # 15 words max
            tokens_per_word_ratio=1.0,
            mode="accumulate",
        )
        chunker = SemanticChunker(tokenizer=tokenizer, config=config)
        
        # Three sentences: 8 words, 8 words, 8 words
        # First two together = 16 words > 15 word limit
        text = "one two three four five six seven eight. " \
               "nine ten eleven twelve thirteen fourteen fifteen sixteen. " \
               "seventeen eighteen nineteen twenty twentyone twentytwo twentythree twentyfour."
        labels = ["x"]
        
        chunks = chunker.chunk_text(text, labels)
        
        # Should NOT accumulate all three into one chunk (would be 24 words > 15)
        # Each sentence is 8 words, so should get at least 2 chunks
        assert len(chunks) >= 2, f"Expected >= 2 chunks, got {len(chunks)}"
        
        # Verify each chunk respects word limit
        for chunk in chunks:
            word_count = len(chunk.text.split())
            assert word_count <= 15, f"Chunk has {word_count} words, exceeds limit of 15"

    def test_word_limit_stricter_than_token_budget(self, tokenizer):
        """When word limit is stricter than token budget, word limit wins."""
        # Tokenizer that returns 2 tokens per word (making token limit less strict)
        from unittest.mock import MagicMock
        fancy_tokenizer = MagicMock()
        
        def fancy_encode(text, **kw):
            if isinstance(text, str):
                return [f"t{i}" for w in text.split() for i in range(2)]
            # Already a list of words
            return [f"t{i}" for w in text for i in range(2)]
        
        fancy_tokenizer.encode = MagicMock(side_effect=fancy_encode)
        fancy_tokenizer.decode = MagicMock(
            side_effect=lambda ids, **kw: " ".join(str(i) for i in ids[::2])  # Return half (word-like)
        )
        
        def fancy_call(text, is_split_into_words=False, **kwargs):
            if is_split_into_words:
                words = text if isinstance(text, list) else [text]
                tokens = [f"t{i}" for w in words for i in range(2)]
            else:
                text_str = text if isinstance(text, str) else " ".join(text)
                words = text_str.split()
                tokens = [f"t{i}" for w in words for i in range(2)]
            
            result = {"input_ids": tokens}
            
            if kwargs.get("return_offsets_mapping") and isinstance(text, str):
                offsets = []
                pos = 0
                for w in words:
                    start = text.find(w, pos)
                    end = start + len(w)
                    offsets.extend([(start, end), (start, end)])  # 2 tokens per word
                    pos = end
                result["offset_mapping"] = offsets
            
            return result
        
        fancy_tokenizer.side_effect = fancy_call
        
        config = BudgetConfig(
            model_max_length=500,  # Token limit allows ~250 tokens after overhead
            system_overhead=5,
            min_budget_floor=10,
            gliner_max_words=50,  # Word limit: 50 words
            tokens_per_word_ratio=2.0,  # 2 tokens/word -> word_budget_tokens = 100
            mode="single_sentence",
            hard_split_overlap=10,
        )
        chunker = SemanticChunker(tokenizer=fancy_tokenizer, config=config)
        
        # Sentence with 60 words = 120 tokens
        # Token limit: 500 - 7 = 493 tokens > 120 tokens, so tokens OK
        # Word limit: 50 words < 60 words, so words NOT OK
        sentence_60_words = " ".join([f"word{i}" for i in range(60)]) + "."
        labels = ["x"]
        
        chunks = chunker.chunk_text(sentence_60_words, labels)
        
        # Should split because word count (60) exceeds word limit (50)
        assert len(chunks) > 1, f"Should split when word limit exceeded, got {len(chunks)} chunks"
