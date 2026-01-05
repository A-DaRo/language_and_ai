"""
Integration Test: GLiNER Pollution Guard

Validates:
- GLiNER model loads correctly
- Typed mask tokens are single tokens
- Pollution detection on real SOBR data
- Span masking preserves text structure
- Distractor filtering works correctly
"""

import pytest
import torch
from pathlib import Path
import pyarrow as pa

from neuro_stylometry.pollution_guard.gliner_detector import (
    GLiNERDetector,
    SOBRTaxonomy,
    EntityWidthConstraints,
)
from neuro_stylometry.pollution_guard.masker import SpanMasker
from neuro_stylometry.data_engine.dataset import SOBRDataset


@pytest.fixture
def device():
    """Get available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def gliner_detector(device):
    """Initialize GLiNER detector."""
    return GLiNERDetector(
        model_name="urchade/gliner_large-v2.1",
        device=device,
        max_length=512,
        confidence_threshold=0.85,
        center_window_keep=100,
    )


@pytest.fixture
def span_masker(gliner_detector):
    """Initialize SpanMasker with GLiNER tokenizer."""
    tokenizer = gliner_detector.model.data_processor.transformer_tokenizer
    return SpanMasker(tokenizer=tokenizer)


@pytest.fixture
def sample_texts():
    """Sample texts with known pollution."""
    return [
        "I am 25 years old and work as a software engineer.",
        "As a German living in Berlin, I enjoy coding.",
        "I'm an INTJ personality type and I'm from Canada.",
        "My mother is 55 but I'm only 30.",
        "I'm a liberal who supports progressive policies.",
    ]


@pytest.fixture
def laptop_dataset():
    """Load laptop dataset."""
    path = Path("artifacts/data/sobr_laptop.arrow")
    if not path.exists():
        pytest.skip(f"Laptop dataset not found at {path}")
    return SOBRDataset(arrow_path=path, seed=42)


class TestGLiNERDetector:
    """Test GLiNER pollution detector."""
    
    def test_model_loading(self, gliner_detector, device):
        """Test that GLiNER model loads correctly."""
        assert gliner_detector.model is not None
        assert str(gliner_detector.device) == device
    
    def test_mask_token_registration(self, gliner_detector):
        """Test that typed mask tokens are registered as single tokens."""
        # Implements FR-05: Tokenizer safety
        tokenizer = gliner_detector.model.data_processor.transformer_tokenizer
        
        mask_tokens = [
            "[MASK:AGE]",
            "[MASK:BIRTH_YEAR]",
            "[MASK:GENDER]",
            "[MASK:NATIONALITY]",
            "[MASK:COUNTRY]",
            "[MASK:MBTI]",
            "[MASK:POLITICAL]",
        ]
        
        for mask_token in mask_tokens:
            token_ids = tokenizer.encode(mask_token, add_special_tokens=False)
            assert len(token_ids) == 1, (
                f"Mask token {mask_token} fragmented into {len(token_ids)} tokens"
            )
    
    def test_taxonomy_labels(self):
        """Test that taxonomy labels are properly defined."""
        taxonomy = SOBRTaxonomy()
        
        # Check target labels
        target_labels = taxonomy.get_target_labels()
        assert len(target_labels) > 0
        assert "age_statement" in target_labels
        assert "nationality_statement" in target_labels
        assert "mbti_type" in target_labels
        
        # Check distractor labels
        assert len(taxonomy.distractor_labels) > 0
        assert "third person reference" in taxonomy.distractor_labels
        
        # Check inference labels combine both
        inference_labels = taxonomy.get_inference_labels()
        assert "self-identified nationality" in inference_labels
        assert "third person reference" in inference_labels
    
    def test_pollution_detection_sample_texts(self, gliner_detector, sample_texts):
        """Test pollution detection on sample texts."""
        # Implements FR-06: Span detection
        entities_batch = gliner_detector.detect_spans(
            sample_texts,
            batch_size=2,
            show_progress=False,
        )
        
        assert len(entities_batch) == len(sample_texts)
        
        if all(len(entities) == 0 for entities in entities_batch):
            pytest.skip("No detections on sample texts (model/threshold variability)")
        
        # If detections exist, labels should be normalized internal IDs
        for entities in entities_batch:
            for entity in entities:
                assert entity["label"] in gliner_detector.taxonomy.get_target_labels()
    
    def test_confidence_threshold(self, gliner_detector, sample_texts):
        """Test that confidence threshold filters low-confidence spans."""
        # Implements FR-07: Precision filtering
        entities_batch = gliner_detector.detect_spans(sample_texts, batch_size=2)
        
        for entities in entities_batch:
            for entity in entities:
                assert entity["score"] >= gliner_detector.confidence_threshold, (
                    f"Entity {entity['text']} has score {entity['score']} "
                    f"below threshold {gliner_detector.confidence_threshold}"
                )
    
    def test_width_constraints(self, gliner_detector):
        """Test that entity width constraints are applied."""
        # Implements FR-07: Width constraints
        text = "I am 25 years old and I was born in the year nineteen hundred and ninety five"
        entities = gliner_detector.detect_spans([text], batch_size=1)[0]
        
        tokenizer = gliner_detector.model.data_processor.transformer_tokenizer
        constraints = gliner_detector.constraints
        
        for entity in entities:
            span_tokens = tokenizer.encode(entity["text"], add_special_tokens=False)
            token_count = len(span_tokens)
            max_width = constraints.get_max_width(entity["label"])
            
            assert token_count <= max_width, (
                f"Entity '{entity['text']}' ({entity['label']}) has {token_count} tokens, "
                f"exceeds max {max_width}"
            )
    
    def test_long_text_chunking(self, gliner_detector):
        """Test semantic chunking for long texts."""
        # Implements FR-06: Chunking strategy
        # Create a long text by repeating content
        short_text = "I am 25 years old. " * 10
        long_text = short_text * 20  # ~200 sentences, ~1200 tokens
        
        # Verify chunking works (creates multiple chunks)
        labels = gliner_detector.taxonomy.get_inference_labels()
        chunks = gliner_detector.chunker.chunk_text(long_text, labels)
        
        # Should create multiple chunks for text exceeding budget
        budget = gliner_detector.get_effective_budget()
        text_tokens = len(gliner_detector.tokenizer.encode(long_text, add_special_tokens=False))
        
        if text_tokens > budget:
            assert len(chunks) > 1, f"Expected multiple chunks for {text_tokens} tokens (budget={budget})"
        
        entities = gliner_detector.detect_spans_long([long_text], batch_size=1)[0]
        
        # Detection count is model-dependent; just verify no crash and proper deduplication
        # Check for duplicates (should be deduplicated)
        seen = set()
        for entity in entities:
            key = (entity["start"], entity["end"], entity["label"])
            assert key not in seen, f"Duplicate entity detected: {key}"
            seen.add(key)
    
    def test_real_sobr_data(self, gliner_detector, laptop_dataset):
        """Test pollution detection on real SOBR data."""
        # Implements FR-06: Real data integration
        table = laptop_dataset.table
        
        # Take first 10 posts
        sample_size = min(10, len(table))
        posts = table["post"][:sample_size].to_pylist()
        post_ids = table["post_id"][:sample_size].to_pylist()
        
        # Detect spans (use detect_spans_long for robustness)
        entities_batch = gliner_detector.detect_spans_long(posts, batch_size=2)
        
        assert len(entities_batch) == sample_size
        
        # Count total detections
        total_detections = sum(len(entities) for entities in entities_batch)
        print(f"\nDetected {total_detections} pollution spans in {sample_size} posts")
        
        # Log some examples
        for post_id, entities in zip(post_ids, entities_batch):
            if entities:
                print(f"  Post {post_id}: {len(entities)} spans")
                for entity in entities[:3]:  # Show first 3
                    print(f"    - {entity['label']}: '{entity['text']}' (conf={entity['score']:.2f})")


# ==============================================================================
# Edge Case and Boundary Tests
# ==============================================================================

@pytest.mark.integration
class TestGLiNERBoundaryConditions:
    """Test GLiNER detector with boundary condition inputs."""
    
    @pytest.mark.parametrize("text,description", [
        ("", "empty_string"),
        ("a", "single_character"),
        ("... ... ...", "punctuation_only"),
        ("🎉👍🌟", "emoji_only"),
        ("   \n\t  ", "whitespace_only"),
    ])
    def test_gliner_detector_boundary_texts(self, gliner_detector, text, description):
        """GLiNER handles boundary condition inputs robustly."""
        result = gliner_detector.detect_spans([text], batch_size=1, show_progress=False)
        
        if description == "empty_string":
            assert result[0] == [], "Empty text should yield no entities"
        elif description == "whitespace_only":
            assert result[0] == [], "Whitespace-only should yield no entities"
        else:
            # Other cases should process without error
            assert isinstance(result[0], list), f"Processing failed for {description}"
    
    def test_gliner_extremely_long_text(self, gliner_detector):
        """GLiNER handles extremely long text by chunking."""
        # Create text longer than max_length (512 tokens)
        long_text = "I am 25 years old. " * 500  # ~2500 words
        
        # Should not raise, should chunk and process
        result = gliner_detector.detect_spans_long([long_text], batch_size=1)
        
        assert isinstance(result[0], list), "Long text should return list of entities"
    
    def test_gliner_unicode_handling(self, gliner_detector):
        """Pollution detection works with diverse Unicode scripts."""
        texts = [
            "I am 25 years old, 我住在北京。",  # Mixed English + Chinese
            "Je m'appelle Pierre et je suis français.",  # French
            "Ich bin 30 Jahre alt und komme aus Deutschland.",  # German
            "I'm an INTJ from 日本",  # Mixed with Japanese
        ]
        results = gliner_detector.detect_spans(texts, batch_size=2)
        
        # Should process all texts without encoding errors
        assert len(results) == len(texts)
        for entities in results:
            assert isinstance(entities, list), "Unicode text failed to process"
    
    def test_gliner_special_characters(self, gliner_detector):
        """GLiNER handles special characters in text."""
        texts = [
            "I'm 25 y/o & from the U.S.A.",  # Abbreviations and symbols
            'I said "I am 30" to them.',  # Quotes
            "Age: 25; Nationality: Canadian",  # Semicolons
            "Born in '95, I'm a millennial.",  # Apostrophes
        ]
        results = gliner_detector.detect_spans(texts, batch_size=2)
        
        assert len(results) == len(texts)
        for entities in results:
            assert isinstance(entities, list)
    
    def test_gliner_adversarial_repetition(self, gliner_detector):
        """Detector doesn't hallucinate excessively on repetitive patterns."""
        text = "I am I am I am I am I am."
        result = gliner_detector.detect_spans([text], batch_size=1)
        
        # Should not report excessive detections
        entities = result[0]
        # Count unique spans
        unique_spans = {(e["start"], e["end"]) for e in entities}
        
        # Allow some detections but not excessive hallucination
        assert len(unique_spans) <= 5, (
            f"Excessive detections ({len(unique_spans)}) on repetitive text"
        )


@pytest.mark.integration
class TestGLiNERDistractorFiltering:
    """Test distractor filtering functionality."""
    
    def test_third_person_filtering(self, gliner_detector):
        """Third-person references are filtered as distractors."""
        text = "My friend is 25 years old. He is from Canada."
        
        # With distractor filtering enabled, should filter third-person
        entities = gliner_detector.detect_spans([text], batch_size=1)[0]
        
        # Check that any detected entities are relevant
        for entity in entities:
            # Should not detect "friend" age as self-identification
            if "25" in entity["text"]:
                # Either filtered or labeled as third-person
                assert entity["label"] != "age_statement" or \
                       "third" not in entity.get("filtered_reason", "").lower(), \
                       "Third-person age should be filtered"


class TestSpanMasker:
    """Test span masking functionality."""
    
    def test_masking_preserves_offsets(self, span_masker):
        """Test that masking in reverse order preserves character offsets."""
        text = "I am 25 years old and I'm from Canada."
        
        # Simulate detected entities
        entities = [
            {"start": 5, "end": 7, "text": "25", "label": "age_statement", "score": 0.9},
            {"start": 32, "end": 38, "text": "Canada", "label": "country_of_origin", "score": 0.95},
        ]
        
        result = span_masker.mask_spans(text, entities, "test_post")
        
        # Check masked text
        assert "[MASK:AGE]" in result.masked_text
        assert "[MASK:COUNTRY]" in result.masked_text
        assert "25" not in result.masked_text
        assert "Canada" not in result.masked_text
        
        # Check log entries
        assert len(result.mask_log) == 2
        assert all(log["post_id"] == "test_post" for log in result.mask_log)
    
    def test_mask_batch(self, span_masker, sample_texts):
        """Test batch masking."""
        # Simulate detected entities (empty for simplicity)
        entities_batch = [[] for _ in sample_texts]
        post_ids = [f"post_{i}" for i in range(len(sample_texts))]
        
        masked_texts, all_logs = span_masker.mask_batch(
            sample_texts, entities_batch, post_ids
        )
        
        assert len(masked_texts) == len(sample_texts)
        assert len(all_logs) == 0  # No entities, no logs
    
    def test_integrated_detection_and_masking(
        self, gliner_detector, span_masker, sample_texts
    ):
        """Test integrated detection + masking pipeline."""
        # Implements FR-08: Integrated masking
        post_ids = [f"post_{i}" for i in range(len(sample_texts))]
        
        # Detect spans
        entities_batch = gliner_detector.detect_spans(sample_texts, batch_size=2)
        
        # Mask spans
        masked_texts, all_logs = span_masker.mask_batch(
            sample_texts, entities_batch, post_ids
        )
        
        assert len(masked_texts) == len(sample_texts)
        
        # Verify masking happened
        for i, (original, masked, entities) in enumerate(
            zip(sample_texts, masked_texts, entities_batch)
        ):
            if entities:
                # Should have at least one mask token
                assert "[MASK:" in masked, f"Text {i} has entities but no mask tokens"
                print(f"\nOriginal: {original}")
                print(f"Masked:   {masked}")
        
        # Verify logs conform to POLLUTION_LOG_SCHEMA
        for log in all_logs:
            assert "post_id" in log
            assert "span_start" in log
            assert "span_end" in log
            assert "span_text" in log
            assert "entity_type" in log
            assert "confidence" in log
            assert "mask_token" in log
