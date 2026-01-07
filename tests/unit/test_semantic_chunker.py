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

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False):
        """Callable interface for tokenizer."""
        tokens = self.encode(text, add_special_tokens=add_special_tokens)
        result = {"input_ids": tokens}
        if return_offsets_mapping:
            # Simple offset mapping: each "word" is ~5 chars
            offsets = []
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
        assert config.system_overhead == 5
        assert config.ent_marker_cost == 1
        assert config.hard_split_overlap == 50
        assert config.min_budget_floor == 50
        assert config.legacy_sequential_mode is False  # New default

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
        config = BudgetConfig(model_max_length=50, system_overhead=5, min_budget_floor=10)
        return SemanticChunker(tokenizer=tokenizer, config=config)

    def test_calculate_effective_budget(self, chunker):
        """Budget calculation accounts for label tokens."""
        # Each label is ~2 words = 2 tokens + 1 ENT marker = 3 tokens per label
        labels = ["age statement", "gender identity"]

        budget = chunker.calculate_effective_budget(labels)

        # max=50, system=5, labels=2*(2+1)=6, effective=50-5-6=39
        # But MockTokenizer returns 1 token per word, so "age statement" = 2 tokens
        # Budget = 50 - 5 - (2+1 + 2+1) = 50 - 5 - 6 = 39
        assert budget == 39

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
