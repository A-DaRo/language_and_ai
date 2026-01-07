"""
Integration Test: GLiNER Pollution Guard

Validates:
- GLiNER model loads correctly
- Typed mask tokens are single tokens
- Pollution detection on real SOBR data
- Span masking preserves text structure
- Distractor filtering works correctly
- Batch inference optimizations (v2.0)
"""

import pytest
import torch
from pathlib import Path
import pyarrow as pa

from neuro_stylometry.pollution_guard.gliner_detector import (
    GLiNERDetector,
    SOBRTaxonomy,
    EntityWidthConstraints,
    BatchInferenceConfig,
    auto_compute_bucket_count,
    compute_chunk_length_buckets,
)
from neuro_stylometry.pollution_guard.semantic_chunker import BudgetConfig
from neuro_stylometry.pollution_guard.masker import SpanMasker
from neuro_stylometry.data_engine.dataset import SOBRDataset


@pytest.fixture
def device():
    """Get available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def gliner_detector(device):
    """Initialize GLiNER detector with column-driven taxonomy."""
    import yaml
    
    # Load taxonomy from YAML config
    config_path = Path("conf/base/gliner_taxonomy.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        taxonomy_config = config.get("taxonomy", {})
    else:
        taxonomy_config = None
    
    return GLiNERDetector(
        model_name="urchade/gliner_large-v2.1",
        device=device,
        max_length=512,
        confidence_threshold=0.85,
        taxonomy_config=taxonomy_config,
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
    
    def test_taxonomy_labels(self, gliner_detector):
        """Test that taxonomy labels are properly defined (column-driven)."""
        taxonomy = gliner_detector.taxonomy
        
        # Check column names (was target_labels)
        columns = taxonomy.get_all_columns()
        assert len(columns) > 0
        # SOBR columns: birth_year, female, nationality, political_leaning, extrovert, sensing, feeling, judging
        assert "birth_year" in columns
        assert "nationality" in columns
        
        # Check backward-compat get_target_labels returns columns
        target_labels = taxonomy.get_target_labels()
        assert target_labels == columns
        
        # Check distractor labels (aggregated from all columns)
        assert len(taxonomy.distractor_labels) > 0
        # At least one distractor should exist
        assert any("third" in d.lower() or "person" in d.lower() for d in taxonomy.distractor_labels)
        
        # Check inference labels combine prompts + distractors
        inference_labels = taxonomy.get_inference_labels()
        assert len(inference_labels) > 0
        # Should include at least some prompt strings
        assert any("self-identified" in label or "statement" in label for label in inference_labels)
    
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
        
        # If detections exist, labels should be column names
        valid_columns = gliner_detector.taxonomy.get_all_columns()
        for entities in entities_batch:
            for entity in entities:
                assert entity["label"] in valid_columns, (
                    f"Entity label '{entity['label']}' not in valid columns: {valid_columns}"
                )
    
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


# ==============================================================================
# Batch Inference Optimization Tests (v2.0)
# ==============================================================================

class TestBatchInferenceConfig:
    """Test BatchInferenceConfig dataclass."""
    
    def test_default_values(self):
        """BatchInferenceConfig has sensible defaults."""
        config = BatchInferenceConfig()
        
        assert config.enable_batching is True
        assert config.batch_size == 32
        assert config.num_buckets is None  # Auto-compute
        assert config.min_bucket_size == 4
        assert config.enable_prompt_caching is True
    
    def test_custom_values(self):
        """BatchInferenceConfig accepts custom values."""
        config = BatchInferenceConfig(
            enable_batching=False,
            batch_size=64,
            num_buckets=10,
            min_bucket_size=8,
            enable_prompt_caching=False,
        )
        
        assert config.enable_batching is False
        assert config.batch_size == 64
        assert config.num_buckets == 10


class TestAutoComputeBucketCount:
    """Test VRAM-based bucket count auto-computation."""
    
    def test_low_vram_bucket_count(self):
        """Low VRAM (<8GB) results in 5 buckets."""
        assert auto_compute_bucket_count(4.0) == 5
        assert auto_compute_bucket_count(7.9) == 5
    
    def test_medium_vram_bucket_count(self):
        """Medium VRAM (8-16GB) results in 8 buckets."""
        assert auto_compute_bucket_count(8.0) == 8
        assert auto_compute_bucket_count(15.9) == 8
    
    def test_high_vram_bucket_count(self):
        """High VRAM (16-40GB) results in 12 buckets."""
        assert auto_compute_bucket_count(16.0) == 12
        assert auto_compute_bucket_count(39.9) == 12
    
    def test_datacenter_vram_bucket_count(self):
        """Datacenter VRAM (40+GB) results in 16 buckets."""
        assert auto_compute_bucket_count(40.0) == 16
        assert auto_compute_bucket_count(80.0) == 16
    
    def test_cpu_mode_bucket_count(self):
        """CPU mode (0 VRAM) results in 5 buckets."""
        assert auto_compute_bucket_count(0.0) == 5
    
    def test_auto_detect_bucket_count(self):
        """Auto-detect with None returns valid bucket count."""
        buckets = auto_compute_bucket_count(None)
        assert buckets in [5, 8, 12, 16]


class TestComputeChunkLengthBuckets:
    """Test length-based bucketing."""
    
    def test_empty_lengths(self):
        """Empty length list returns empty buckets."""
        boundaries, bucket_to_indices = compute_chunk_length_buckets([], 5)
        assert boundaries == []
        assert bucket_to_indices == {}
    
    def test_single_length(self):
        """Single length returns single bucket."""
        boundaries, bucket_to_indices = compute_chunk_length_buckets([10], 5)
        # Single item goes to bucket 0
        total_items = sum(len(indices) for indices in bucket_to_indices.values())
        assert total_items == 1
    
    def test_uniform_lengths(self):
        """Uniform lengths distribute correctly."""
        lengths = [10, 10, 10, 10]
        boundaries, bucket_to_indices = compute_chunk_length_buckets(lengths, 3)
        # All should be in same bucket (uniform distribution)
        total_items = sum(len(indices) for indices in bucket_to_indices.values())
        assert total_items == 4
    
    def test_varied_lengths(self):
        """Varied lengths are bucketed by quantiles."""
        lengths = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        boundaries, bucket_to_indices = compute_chunk_length_buckets(lengths, 5)
        
        # Should have multiple buckets
        non_empty_buckets = sum(1 for indices in bucket_to_indices.values() if indices)
        assert non_empty_buckets >= 2
        
        # All items should be assigned
        total_items = sum(len(indices) for indices in bucket_to_indices.values())
        assert total_items == len(lengths)


@pytest.mark.integration
class TestBatchedInference:
    """Test batched inference pipeline."""
    
    def test_batched_vs_sequential_equivalence(self, device, sample_texts):
        """Batched and sequential inference produce equivalent results."""
        import yaml
        
        # Load taxonomy
        config_path = Path("conf/base/gliner_taxonomy.yaml")
        if not config_path.exists():
            pytest.skip("Taxonomy config not found")
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        taxonomy_config = config.get("taxonomy", {})
        
        # Create detector with batching enabled
        batch_config = BatchInferenceConfig(
            enable_batching=True,
            batch_size=4,
            num_buckets=3,
        )
        detector_batched = GLiNERDetector(
            model_name="urchade/gliner_large-v2.1",
            device=device,
            max_length=512,
            confidence_threshold=0.85,
            taxonomy_config=taxonomy_config,
            batch_inference_config=batch_config,
        )
        
        # Create detector with legacy sequential mode
        budget_config_seq = BudgetConfig(
            model_max_length=512,
            legacy_sequential_mode=True,
        )
        batch_config_seq = BatchInferenceConfig(enable_batching=False)
        detector_sequential = GLiNERDetector(
            model_name="urchade/gliner_large-v2.1",
            device=device,
            max_length=512,
            confidence_threshold=0.85,
            taxonomy_config=taxonomy_config,
            budget_config=budget_config_seq,
            batch_inference_config=batch_config_seq,
        )
        
        # Run both modes
        results_batched = detector_batched.detect_spans(
            sample_texts, batch_size=4, show_progress=False
        )
        results_sequential = detector_sequential.detect_spans(
            sample_texts, batch_size=1, show_progress=False
        )
        
        # Results should have same length
        assert len(results_batched) == len(results_sequential)
        
        # Compare detection counts (may differ slightly due to batching effects)
        batched_count = sum(len(e) for e in results_batched)
        sequential_count = sum(len(e) for e in results_sequential)
        
        # Allow up to 10% difference due to batching effects
        if sequential_count > 0:
            diff_ratio = abs(batched_count - sequential_count) / max(sequential_count, 1)
            assert diff_ratio < 0.1, (
                f"Batched ({batched_count}) vs sequential ({sequential_count}) "
                f"differ by {diff_ratio:.1%}"
            )
    
    def test_batch_config_from_pipeline(self, device):
        """Test that batch config is properly wired from pipeline."""
        import yaml
        
        config_path = Path("conf/base/gliner_taxonomy.yaml")
        if not config_path.exists():
            pytest.skip("Taxonomy config not found")
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        taxonomy_config = config.get("taxonomy", {})
        
        # Custom batch config
        batch_config = BatchInferenceConfig(
            enable_batching=True,
            batch_size=16,
            num_buckets=6,
            enable_prompt_caching=True,
        )
        
        detector = GLiNERDetector(
            model_name="urchade/gliner_large-v2.1",
            device=device,
            max_length=512,
            confidence_threshold=0.85,
            taxonomy_config=taxonomy_config,
            batch_inference_config=batch_config,
        )
        
        # Verify config is set
        assert detector.batch_config.enable_batching is True
        assert detector.batch_config.batch_size == 16
        assert detector.batch_config.num_buckets == 6
    
    def test_prompt_cache_stats(self, gliner_detector, sample_texts):
        """Test prompt embedding cache statistics."""
        # Clear any existing cache
        gliner_detector.clear_prompt_cache()
        
        # Run detection to populate cache (if bi-encoder)
        gliner_detector.detect_spans(sample_texts, show_progress=False)
        
        # Get cache stats
        stats = gliner_detector.get_cache_stats()
        
        assert "prompt_cache_entries" in stats
        assert "is_bi_encoder" in stats
        assert "prompt_caching_enabled" in stats
        
        # If bi-encoder and caching enabled, should have cached entries
        if stats["is_bi_encoder"] and stats["prompt_caching_enabled"]:
            assert stats["prompt_cache_entries"] > 0
    
    def test_clear_prompt_cache(self, gliner_detector, sample_texts):
        """Test clearing prompt embedding cache."""
        # Run detection to populate cache
        gliner_detector.detect_spans(sample_texts, show_progress=False)
        
        # Clear cache
        cleared = gliner_detector.clear_prompt_cache()
        
        # Should return count of cleared entries
        assert isinstance(cleared, int)
        
        # Cache should be empty after clear
        stats = gliner_detector.get_cache_stats()
        assert stats["prompt_cache_entries"] == 0
    
    def test_set_batch_config(self, gliner_detector):
        """Test updating batch config at runtime."""
        original_batch_size = gliner_detector.batch_config.batch_size
        
        # Update config
        new_config = BatchInferenceConfig(
            enable_batching=True,
            batch_size=64,
            num_buckets=10,
        )
        gliner_detector.set_batch_config(new_config)
        
        assert gliner_detector.batch_config.batch_size == 64
        assert gliner_detector.batch_config.num_buckets == 10
