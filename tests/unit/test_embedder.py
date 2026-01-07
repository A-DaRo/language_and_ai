"""
Unit tests for FrozenEmbedder.

Tests:
1. Basic initialization and embedding extraction
2. Special token registration for tokenizer alignment (Section 5.1)
3. Single-token encoding verification for mask tokens
4. Frozen weights verification
"""

import pytest
import torch

from neuro_stylometry.pollution_guard.embedder import FrozenEmbedder


# Default mask tokens from SOBR taxonomy (matching conf/base/gliner_taxonomy.yaml)
SOBR_MASK_TOKENS = [
    "[MASK:AGE]",
    "[MASK:GENDER]",
    "[MASK:NATIONALITY]",
    "[MASK:POLITICAL]",
    "[MASK:MBTI]",
]


class TestFrozenEmbedderSpecialTokens:
    """Test special token registration for tokenizer alignment (Section 5.1)."""

    def test_special_tokens_registered_as_single_tokens(self, require_real_models):
        """Verify that mask tokens are encoded as single tokens after registration."""
        embedder = FrozenEmbedder(
            model_name="roberta-base",
            device="cpu",
            max_length=512,
            special_tokens=SOBR_MASK_TOKENS,
        )

        for token in SOBR_MASK_TOKENS:
            token_ids = embedder.tokenizer.encode(token, add_special_tokens=False)
            assert len(token_ids) == 1, (
                f"Special token '{token}' should encode as 1 token, "
                f"got {len(token_ids)}: {token_ids}"
            )

    def test_special_tokens_not_registered_causes_fragmentation(self, require_real_models):
        """Verify that without registration, mask tokens are fragmented."""
        embedder = FrozenEmbedder(
            model_name="roberta-base",
            device="cpu",
            max_length=512,
            special_tokens=None,  # No special tokens registered
        )

        # Without registration, "[MASK:AGE]" should be fragmented
        token = "[MASK:AGE]"
        token_ids = embedder.tokenizer.encode(token, add_special_tokens=False)
        # RoBERTa tokenizer will fragment this into multiple subwords
        assert len(token_ids) > 1, (
            f"Expected '{token}' to be fragmented without registration, "
            f"but got {len(token_ids)} token(s)"
        )

    def test_embedding_masked_text_with_special_tokens(self, require_real_models):
        """Test embedding extraction on masked text preserves special tokens."""
        embedder = FrozenEmbedder(
            model_name="roberta-base",
            device="cpu",
            max_length=512,
            special_tokens=SOBR_MASK_TOKENS,
        )

        # Simulate masked text from GLiNER
        masked_texts = [
            "I am [MASK:AGE] years old.",
            "As a [MASK:GENDER] from [MASK:NATIONALITY], I support [MASK:POLITICAL].",
            "My personality type is [MASK:MBTI].",
        ]

        embeddings = embedder.embed_texts(masked_texts, batch_size=2, show_progress=False)

        assert embeddings.shape == (len(masked_texts), 768)
        assert embeddings.dtype == torch.float32
        assert not torch.allclose(embeddings, torch.zeros_like(embeddings))

    def test_embeddings_differ_with_vs_without_special_tokens(self, require_real_models):
        """Verify that tokenizer alignment affects embedding values."""
        text_with_mask = "Hello [MASK:AGE] world"

        # Embedder WITH special tokens
        embedder_with = FrozenEmbedder(
            model_name="roberta-base",
            device="cpu",
            max_length=512,
            special_tokens=SOBR_MASK_TOKENS,
        )

        # Embedder WITHOUT special tokens
        embedder_without = FrozenEmbedder(
            model_name="roberta-base",
            device="cpu",
            max_length=512,
            special_tokens=None,
        )

        emb_with = embedder_with.embed_texts([text_with_mask], batch_size=1, show_progress=False)
        emb_without = embedder_without.embed_texts([text_with_mask], batch_size=1, show_progress=False)

        # Embeddings should differ due to different tokenization
        assert not torch.allclose(emb_with, emb_without, rtol=1e-3, atol=1e-5), (
            "Embeddings should differ when special tokens are registered vs not"
        )


class TestFrozenEmbedderBasicFunctionality:
    """Test basic embedding functionality."""

    def test_initialization(self, require_real_models):
        """Test that embedder initializes correctly."""
        embedder = FrozenEmbedder(
            model_name="roberta-base",
            device="cpu",
            max_length=512,
        )

        assert embedder.embedding_dim == 768
        assert embedder.max_length == 512

    def test_embedding_extraction_shape(self, require_real_models):
        """Test CLS embedding extraction produces correct shape."""
        embedder = FrozenEmbedder(
            model_name="roberta-base",
            device="cpu",
            max_length=512,
        )

        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming technology.",
        ]

        embeddings = embedder.embed_texts(texts, batch_size=2, show_progress=False)

        assert embeddings.shape == (len(texts), 768)
        assert embeddings.dtype == torch.float32

    def test_frozen_weights(self, require_real_models):
        """Test that model weights are frozen (no gradients)."""
        embedder = FrozenEmbedder(
            model_name="roberta-base",
            device="cpu",
            max_length=512,
        )

        for param in embedder.model.parameters():
            assert not param.requires_grad, "Model weights should be frozen"

    def test_output_device_cpu(self, require_real_models):
        """Test that output_device='cpu' moves embeddings to CPU."""
        embedder = FrozenEmbedder(
            model_name="roberta-base",
            device="cpu",
            max_length=512,
            output_device="cpu",
        )

        texts = ["Test text"]
        embeddings = embedder.embed_texts(texts, batch_size=1, show_progress=False)

        assert embeddings.device == torch.device("cpu")


class TestFrozenEmbedderTokenizerAlignment:
    """Test tokenizer alignment requirements from Section 5.1."""

    def test_model_embeddings_resized_for_special_tokens(self, require_real_models):
        """Verify model embeddings are resized when special tokens are added."""
        # Get original vocab size
        from transformers import AutoTokenizer
        original_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        original_vocab_size = len(original_tokenizer)

        # Create embedder with special tokens
        embedder = FrozenEmbedder(
            model_name="roberta-base",
            device="cpu",
            max_length=512,
            special_tokens=SOBR_MASK_TOKENS,
        )

        # Verify vocab size increased
        new_vocab_size = len(embedder.tokenizer)
        assert new_vocab_size > original_vocab_size, (
            f"Vocab size should increase after adding special tokens: "
            f"{original_vocab_size} -> {new_vocab_size}"
        )

        # Verify model embeddings match new vocab size
        model_embed_size = embedder.model.embeddings.word_embeddings.num_embeddings
        assert model_embed_size == new_vocab_size, (
            f"Model embedding size ({model_embed_size}) should match "
            f"tokenizer vocab size ({new_vocab_size})"
        )

    def test_duplicate_special_tokens_handled(self, require_real_models):
        """Test that duplicate special tokens are deduplicated."""
        # Include duplicates (e.g., multiple columns may map to [MASK:MBTI])
        tokens_with_dupes = [
            "[MASK:AGE]",
            "[MASK:MBTI]",
            "[MASK:AGE]",  # duplicate
            "[MASK:MBTI]",  # duplicate
            "[MASK:GENDER]",
        ]

        embedder = FrozenEmbedder(
            model_name="roberta-base",
            device="cpu",
            max_length=512,
            special_tokens=tokens_with_dupes,
        )

        # Should not raise, and unique tokens should be registered
        for token in ["[MASK:AGE]", "[MASK:MBTI]", "[MASK:GENDER]"]:
            token_ids = embedder.tokenizer.encode(token, add_special_tokens=False)
            assert len(token_ids) == 1
