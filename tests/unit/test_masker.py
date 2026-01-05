# tests/unit/test_masker.py
"""
Unit tests for SpanMasker functionality.

These tests validate:
1. Masking offset consistency and round-trip reconstruction
2. Overlapping span resolution
3. Edge cases and boundary conditions
4. Typed mask token handling

Implements: Testing Plan Step 5 (Functional Correctness)
"""

import pytest
from typing import List, Dict, Any

from neuro_stylometry.pollution_guard.masker import SpanMasker, MaskingResult


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def span_masker():
    """Create a SpanMasker without tokenizer validation."""
    return SpanMasker(tokenizer=None)


@pytest.fixture
def span_masker_with_custom_mapping():
    """Create a SpanMasker with custom entity-to-mask mapping."""
    custom_mapping = {
        "age_statement": "[MASK:AGE]",
        "nationality_statement": "[MASK:NATIONALITY]",
        "custom_type": "[MASK:CUSTOM]",
    }
    return SpanMasker(tokenizer=None, entity_to_mask=custom_mapping)


# ==============================================================================
# Offset and Round-Trip Tests
# ==============================================================================

@pytest.mark.unit
class TestMaskingOffsets:
    """Test masking offset handling and round-trip consistency."""
    
    def test_masking_offsets_round_trip_consistency(self, span_masker):
        """Masked text offsets allow reconstruction of original."""
        original_text = "I am 25 years old and live in Berlin, Germany."
        spans = [
            {"start": 5, "end": 7, "text": "25", "label": "age_statement", "score": 0.9},
            {"start": 30, "end": 36, "text": "Berlin", "label": "nationality_statement", "score": 0.95},
        ]
        
        result = span_masker.mask_spans(original_text, spans, "test_post")
        
        # Verify masks were applied
        assert "[MASK:AGE]" in result.masked_text
        assert "[MASK:NATIONALITY]" in result.masked_text
        assert "25" not in result.masked_text
        assert "Berlin" not in result.masked_text
        
        # Verify log structure
        assert len(result.mask_log) == 2
        for log in result.mask_log:
            assert "span_start" in log
            assert "span_end" in log
            assert "span_text" in log
            assert "mask_token" in log
    
    def test_masking_multiple_spans_same_type(self, span_masker):
        """Multiple spans of same type get same mask token."""
        text = "I am 25 years old and my friend is 30."
        spans = [
            {"start": 5, "end": 7, "text": "25", "label": "age_statement", "score": 0.9},
            {"start": 35, "end": 37, "text": "30", "label": "age_statement", "score": 0.9},
        ]
        
        result = span_masker.mask_spans(text, spans, "test_post")
        
        # Both should use same mask token
        mask_count = result.masked_text.count("[MASK:AGE]")
        assert mask_count == 2, f"Expected 2 AGE masks, got {mask_count}"
    
    def test_masking_preserves_surrounding_text(self, span_masker):
        """Masking preserves text around spans."""
        text = "Hello, I am 25 years old today."
        spans = [
            {"start": 12, "end": 14, "text": "25", "label": "age_statement", "score": 0.9},
        ]
        
        result = span_masker.mask_spans(text, spans, "test_post")
        
        assert result.masked_text.startswith("Hello, I am ")
        assert result.masked_text.endswith(" years old today.")
        assert "[MASK:AGE]" in result.masked_text


# ==============================================================================
# Overlapping Span Tests
# ==============================================================================

@pytest.mark.unit
class TestOverlappingSpans:
    """Test overlapping span resolution."""
    
    def test_masking_overlapping_spans_resolution(self, span_masker):
        """Overlapping spans are resolved consistently (longest wins)."""
        text = "I am a 25-year-old engineer from the USA."
        
        # Overlapping spans: "25" and "25-year-old"
        spans = [
            {"start": 7, "end": 9, "text": "25", "label": "age_statement", "score": 0.85},
            {"start": 7, "end": 18, "text": "25-year-old", "label": "age_statement", "score": 0.9},
        ]
        
        result = span_masker.mask_spans(text, spans, "test_post")
        
        # Should resolve to one mask (no double-masking)
        mask_count = result.masked_text.count("[MASK:AGE]")
        assert mask_count == 1, f"Expected 1 mask token, got {mask_count}"
    
    def test_masking_nested_spans(self, span_masker):
        """Nested spans are resolved (outer wins)."""
        text = "The city of Berlin, Germany is nice."
        
        # "Berlin" is nested within "Berlin, Germany"
        spans = [
            {"start": 12, "end": 18, "text": "Berlin", "label": "nationality_statement", "score": 0.8},
            {"start": 12, "end": 27, "text": "Berlin, Germany", "label": "nationality_statement", "score": 0.9},
        ]
        
        result = span_masker.mask_spans(text, spans, "test_post")
        
        # Should have one mask
        mask_count = result.masked_text.count("[MASK:NATIONALITY]")
        assert mask_count == 1, f"Expected 1 mask for nested spans, got {mask_count}"
    
    def test_masking_adjacent_non_overlapping_spans(self, span_masker):
        """Adjacent non-overlapping spans are both masked."""
        text = "Age: 25, Country: USA"
        
        spans = [
            {"start": 5, "end": 7, "text": "25", "label": "age_statement", "score": 0.9},
            {"start": 18, "end": 21, "text": "USA", "label": "nationality_statement", "score": 0.9},
        ]
        
        result = span_masker.mask_spans(text, spans, "test_post")
        
        assert "[MASK:AGE]" in result.masked_text
        assert "[MASK:NATIONALITY]" in result.masked_text


# ==============================================================================
# Edge Cases
# ==============================================================================

@pytest.mark.unit
class TestMaskerEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_masking_empty_text(self, span_masker):
        """Masking empty text returns empty result."""
        result = span_masker.mask_spans("", [], "test_post")
        
        assert result.masked_text == ""
        assert result.mask_log == []
    
    def test_masking_no_spans(self, span_masker):
        """Masking with no spans returns original text."""
        text = "This is a clean text with no pollution."
        
        result = span_masker.mask_spans(text, [], "test_post")
        
        assert result.masked_text == text
        assert result.mask_log == []
    
    def test_masking_span_at_start(self, span_masker):
        """Masking span at start of text."""
        text = "25 years is my age."
        spans = [
            {"start": 0, "end": 2, "text": "25", "label": "age_statement", "score": 0.9},
        ]
        
        result = span_masker.mask_spans(text, spans, "test_post")
        
        assert result.masked_text.startswith("[MASK:AGE]")
        assert "25" not in result.masked_text
    
    def test_masking_span_at_end(self, span_masker):
        """Masking span at end of text."""
        text = "My age is 25"
        spans = [
            {"start": 10, "end": 12, "text": "25", "label": "age_statement", "score": 0.9},
        ]
        
        result = span_masker.mask_spans(text, spans, "test_post")
        
        assert result.masked_text.endswith("[MASK:AGE]")
        assert "25" not in result.masked_text
    
    def test_masking_entire_text(self, span_masker):
        """Masking entire text as single span."""
        text = "25"
        spans = [
            {"start": 0, "end": 2, "text": "25", "label": "age_statement", "score": 0.9},
        ]
        
        result = span_masker.mask_spans(text, spans, "test_post")
        
        assert result.masked_text == "[MASK:AGE]"
    
    def test_masking_unicode_text(self, span_masker):
        """Masking works with Unicode characters."""
        text = "I'm 25 and from 北京"
        spans = [
            {"start": 4, "end": 6, "text": "25", "label": "age_statement", "score": 0.9},
        ]
        
        result = span_masker.mask_spans(text, spans, "test_post")
        
        assert "[MASK:AGE]" in result.masked_text
        assert "北京" in result.masked_text  # Preserved
    
    def test_masking_with_newlines(self, span_masker):
        """Masking preserves newlines in text."""
        text = "First line.\nI am 25 years old.\nThird line."
        spans = [
            {"start": 17, "end": 19, "text": "25", "label": "age_statement", "score": 0.9},
        ]
        
        result = span_masker.mask_spans(text, spans, "test_post")
        
        assert "\n" in result.masked_text
        assert "[MASK:AGE]" in result.masked_text


# ==============================================================================
# Typed Mask Token Tests
# ==============================================================================

@pytest.mark.unit
class TestTypedMaskTokens:
    """Test typed mask token handling."""
    
    def test_correct_mask_token_for_entity_type(self, span_masker):
        """Correct mask token is used for each entity type."""
        text = "I am 25 from Canada and identify as INTJ"
        spans = [
            {"start": 5, "end": 7, "text": "25", "label": "age_statement", "score": 0.9},
            {"start": 13, "end": 19, "text": "Canada", "label": "country_of_origin", "score": 0.9},
            {"start": 36, "end": 40, "text": "INTJ", "label": "mbti_type", "score": 0.9},
        ]
        
        result = span_masker.mask_spans(text, spans, "test_post")
        
        assert "[MASK:AGE]" in result.masked_text
        assert "[MASK:COUNTRY]" in result.masked_text
        assert "[MASK:MBTI]" in result.masked_text
    
    def test_unknown_entity_type_handling(self, span_masker):
        """Unknown entity types get generic mask or are handled gracefully."""
        text = "Some unknown pollution here"
        spans = [
            {"start": 5, "end": 12, "text": "unknown", "label": "unknown_type", "score": 0.9},
        ]
        
        # Should not raise, should handle gracefully
        result = span_masker.mask_spans(text, spans, "test_post")
        
        # Either masked with generic token or original text preserved
        assert "unknown" not in result.masked_text or len(result.mask_log) == 0
    
    def test_custom_mask_mapping(self, span_masker_with_custom_mapping):
        """Custom entity-to-mask mapping is respected."""
        text = "Custom pollution detected"
        spans = [
            {"start": 0, "end": 6, "text": "Custom", "label": "custom_type", "score": 0.9},
        ]
        
        result = span_masker_with_custom_mapping.mask_spans(text, spans, "test_post")
        
        assert "[MASK:CUSTOM]" in result.masked_text


# ==============================================================================
# Batch Masking Tests
# ==============================================================================

@pytest.mark.unit
class TestBatchMasking:
    """Test batch masking functionality."""
    
    def test_batch_masking_multiple_texts(self, span_masker):
        """Batch masking processes multiple texts correctly."""
        texts = [
            "I am 25 years old.",
            "She is from Canada.",
            "Clean text here.",
        ]
        entities_batch = [
            [{"start": 5, "end": 7, "text": "25", "label": "age_statement", "score": 0.9}],
            [{"start": 12, "end": 18, "text": "Canada", "label": "country_of_origin", "score": 0.9}],
            [],
        ]
        post_ids = ["post_1", "post_2", "post_3"]
        
        masked_texts, all_logs = span_masker.mask_batch(texts, entities_batch, post_ids)
        
        assert len(masked_texts) == 3
        assert "[MASK:AGE]" in masked_texts[0]
        assert "[MASK:COUNTRY]" in masked_texts[1]
        assert masked_texts[2] == texts[2]  # No masking
        
        # Check logs
        assert len(all_logs) == 2  # Two texts had entities
    
    def test_batch_masking_empty_batch(self, span_masker):
        """Batch masking handles empty input."""
        masked_texts, all_logs = span_masker.mask_batch([], [], [])
        
        assert masked_texts == []
        assert all_logs == []
    
    def test_batch_masking_all_empty_entities(self, span_masker):
        """Batch masking with no entities returns original texts."""
        texts = ["Text one.", "Text two.", "Text three."]
        entities_batch = [[], [], []]
        post_ids = ["p1", "p2", "p3"]
        
        masked_texts, all_logs = span_masker.mask_batch(texts, entities_batch, post_ids)
        
        assert masked_texts == texts
        assert all_logs == []


# ==============================================================================
# Log Structure Tests
# ==============================================================================

@pytest.mark.unit
class TestMaskLogStructure:
    """Test mask log structure for POLLUTION_LOG_SCHEMA compliance."""
    
    def test_log_contains_required_fields(self, span_masker):
        """Mask log contains all required fields."""
        text = "I am 25 years old."
        spans = [
            {"start": 5, "end": 7, "text": "25", "label": "age_statement", "score": 0.92},
        ]
        
        result = span_masker.mask_spans(text, spans, "test_post_id")
        
        assert len(result.mask_log) == 1
        log = result.mask_log[0]
        
        # Check required fields
        assert log["post_id"] == "test_post_id"
        assert log["span_start"] == 5
        assert log["span_end"] == 7
        assert log["span_text"] == "25"
        assert log["entity_type"] == "age_statement"
        assert log["confidence"] == pytest.approx(0.92)
        assert log["mask_token"] == "[MASK:AGE]"
    
    def test_log_preserves_entity_scores(self, span_masker):
        """Mask log preserves original entity confidence scores."""
        text = "Age 25, Country USA"
        spans = [
            {"start": 4, "end": 6, "text": "25", "label": "age_statement", "score": 0.88},
            {"start": 16, "end": 19, "text": "USA", "label": "nationality_statement", "score": 0.95},
        ]
        
        result = span_masker.mask_spans(text, spans, "test")
        
        scores = {log["entity_type"]: log["confidence"] for log in result.mask_log}
        
        assert scores.get("age_statement") == pytest.approx(0.88)
        assert scores.get("nationality_statement") == pytest.approx(0.95)
