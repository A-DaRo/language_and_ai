"""
Unit tests for hardware operations modules.

Tests CUDA graph caching, torch.compile integration, sequence packing,
GPU span filtering, and async prefetching.
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock

# Import modules under test
from neuro_stylometry.hardware_ops.cuda_graphs import (
    GraphCache,
    GraphCacheConfig,
    CapturedGraph,
    CompileConfig,
    compile_model_if_enabled,
    create_graph_cache_from_config,
    create_compile_config_from_dict,
)
from neuro_stylometry.hardware_ops.sequence_packing import (
    SequencePacker,
    SequencePackerConfig,
    PackedSequence,
    GPUSpanFilter,
    GPUSpanFilterConfig,
    create_span_filter_from_config,
    create_sequence_packer_from_config,
)
from neuro_stylometry.hardware_ops.async_prefetch import (
    AsyncPrefetchPipeline,
    AsyncPrefetchConfig,
    PrefetchBatch,
    SyncPrefetchPipeline,
    create_prefetch_config_from_dict,
)
from neuro_stylometry.hardware_ops.detection import ProfileType


# ---------------------------------------------------------------------------
# GraphCache Tests
# ---------------------------------------------------------------------------


class TestGraphCacheConfig:
    """Tests for GraphCacheConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = GraphCacheConfig()
        
        assert config.enabled is False
        assert config.warmup_iterations == 3
        assert config.max_cached_graphs == 16
        assert config.capture_pool_size_mb == 256
        assert config.use_cuda_graph_memory_pool is True
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = GraphCacheConfig(
            enabled=True,
            warmup_iterations=5,
            max_cached_graphs=32,
        )
        
        assert config.enabled is True
        assert config.warmup_iterations == 5
        assert config.max_cached_graphs == 32


class TestGraphCache:
    """Tests for GraphCache CUDA graph manager."""
    
    def test_initialization_disabled(self):
        """Test GraphCache initializes correctly when disabled."""
        config = GraphCacheConfig(enabled=False)
        cache = GraphCache(config)
        
        assert cache.is_enabled is False
        assert len(cache) == 0
    
    def test_initialization_no_cuda(self):
        """Test GraphCache handles missing CUDA gracefully."""
        config = GraphCacheConfig(enabled=True)
        
        with patch('torch.cuda.is_available', return_value=False):
            cache = GraphCache(config)
            # Should auto-disable when CUDA not available
            assert cache.is_enabled is False
    
    def test_fallback_when_disabled(self):
        """Test run_with_graph falls back to direct execution when disabled."""
        config = GraphCacheConfig(enabled=False)
        cache = GraphCache(config)
        
        # Create mock forward function
        mock_output = {"logits": torch.randn(2, 10)}
        forward_fn = Mock(return_value=mock_output)
        
        # Run with graph (should fall back to direct call)
        input_tensor = torch.randn(2, 5)
        result = cache.run_with_graph(
            shape_key=(2, 5),
            forward_fn=forward_fn,
            input=input_tensor,
        )
        
        # Verify direct call was made
        forward_fn.assert_called_once()
        assert "logits" in result
    
    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        config = GraphCacheConfig(enabled=False, max_cached_graphs=3)
        cache = GraphCache(config)
        
        # Manually add entries to test LRU
        cache._cache[(1,)] = Mock()
        cache._lru_order.append((1,))
        cache._cache[(2,)] = Mock()
        cache._lru_order.append((2,))
        cache._cache[(3,)] = Mock()
        cache._lru_order.append((3,))
        
        assert len(cache) == 3
        
        # Evict oldest
        cache._evict_oldest()
        
        assert len(cache) == 2
        assert (1,) not in cache
        assert (2,) in cache
        assert (3,) in cache
    
    def test_clear(self):
        """Test clearing the cache."""
        config = GraphCacheConfig(enabled=False)
        cache = GraphCache(config)
        
        # Add some entries
        cache._cache[(1,)] = Mock()
        cache._cache[(2,)] = Mock()
        
        count = cache.clear()
        
        assert count == 2
        assert len(cache) == 0
    
    def test_get_stats(self):
        """Test cache statistics."""
        config = GraphCacheConfig(enabled=False, max_cached_graphs=10)
        cache = GraphCache(config)
        
        cache._hits = 5
        cache._misses = 2
        cache._captures = 2
        
        stats = cache.get_stats()
        
        assert stats["hits"] == 5
        assert stats["misses"] == 2
        assert stats["captures"] == 2
        assert stats["hit_rate"] == pytest.approx(5 / 7)
        assert stats["max_cache_size"] == 10


class TestCreateGraphCacheFromConfig:
    """Tests for create_graph_cache_from_config factory."""
    
    def test_creates_from_config_dict(self):
        """Test creating GraphCache from config dict."""
        config = {
            "execution": {
                "enable_cuda_graphs": True,
                "cuda_graph_warmup": 5,
                "cuda_graph_cache_size": 32,
            }
        }
        
        with patch('torch.cuda.is_available', return_value=False):
            cache = create_graph_cache_from_config(config)
        
        assert cache.config.warmup_iterations == 5
        assert cache.config.max_cached_graphs == 32


# ---------------------------------------------------------------------------
# CompileConfig Tests
# ---------------------------------------------------------------------------


class TestCompileConfig:
    """Tests for CompileConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration."""
        config = CompileConfig()
        
        assert config.enabled is False
        assert config.mode == "reduce-overhead"
        assert config.fullgraph is False
        assert config.dynamic is False
        assert config.backend == "inductor"
    
    def test_custom_values(self):
        """Test custom configuration."""
        config = CompileConfig(
            enabled=True,
            mode="max-autotune",
            fullgraph=True,
        )
        
        assert config.enabled is True
        assert config.mode == "max-autotune"
        assert config.fullgraph is True


class TestCompileModelIfEnabled:
    """Tests for compile_model_if_enabled function."""
    
    def test_returns_original_when_disabled(self):
        """Test model returned unchanged when compilation disabled."""
        model = Mock(spec=torch.nn.Module)
        config = CompileConfig(enabled=False)
        
        result = compile_model_if_enabled(model, config)
        
        assert result is model
    
    def test_handles_old_pytorch_version(self):
        """Test graceful fallback for old PyTorch versions."""
        model = Mock(spec=torch.nn.Module)
        config = CompileConfig(enabled=True)
        
        with patch('torch.__version__', '1.13.0'):
            result = compile_model_if_enabled(model, config)
        
        assert result is model


class TestCreateCompileConfigFromDict:
    """Tests for create_compile_config_from_dict factory."""
    
    def test_creates_from_config_dict(self):
        """Test creating CompileConfig from config dict."""
        config = {
            "execution": {
                "enable_torch_compile": True,
                "torch_compile_mode": "max-autotune",
                "torch_compile_fullgraph": True,
                "torch_compile_dynamic": True,
                "torch_compile_backend": "eager",
            }
        }
        
        compile_cfg = create_compile_config_from_dict(config)
        
        assert compile_cfg.enabled is True
        assert compile_cfg.mode == "max-autotune"
        assert compile_cfg.fullgraph is True
        assert compile_cfg.dynamic is True
        assert compile_cfg.backend == "eager"


# ---------------------------------------------------------------------------
# SequencePacker Tests
# ---------------------------------------------------------------------------


class TestSequencePackerConfig:
    """Tests for SequencePackerConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration."""
        config = SequencePackerConfig()
        
        assert config.enabled is False
        assert config.max_packed_length == 2048
        assert config.min_sequences_to_pack == 2
        assert config.use_block_diagonal_mask is True
        assert config.pack_by_similarity is True


class TestSequencePacker:
    """Tests for SequencePacker class."""
    
    def test_can_pack_disabled(self):
        """Test can_pack returns False when disabled."""
        config = SequencePackerConfig(enabled=False)
        packer = SequencePacker(config)
        
        assert packer.can_pack([10, 20, 30]) is False
    
    def test_can_pack_too_few_sequences(self):
        """Test can_pack returns False with too few sequences."""
        config = SequencePackerConfig(enabled=True, min_sequences_to_pack=3)
        packer = SequencePacker(config)
        
        assert packer.can_pack([10, 20]) is False
        assert packer.can_pack([10, 20, 30]) is True
    
    def test_can_pack_exceeds_max_length(self):
        """Test can_pack returns False when total exceeds max."""
        config = SequencePackerConfig(enabled=True, max_packed_length=100)
        packer = SequencePacker(config)
        
        assert packer.can_pack([50, 60]) is False
        assert packer.can_pack([40, 50]) is True
    
    def test_pack_tokenized(self):
        """Test packing pre-tokenized sequences."""
        config = SequencePackerConfig(enabled=True)
        packer = SequencePacker(config)
        
        # Create test sequences
        seq1 = torch.tensor([1, 2, 3, 4, 5])
        seq2 = torch.tensor([10, 11, 12])
        seq3 = torch.tensor([20, 21])
        
        packed = packer.pack_tokenized([seq1, seq2, seq3])
        
        assert isinstance(packed, PackedSequence)
        assert packed.original_count == 3
        assert packed.sequence_lengths == [5, 3, 2]
        assert packed.sequence_boundaries == [0, 5, 8]
        assert packed.input_ids.shape == (10,)
        assert torch.equal(packed.input_ids[:5], seq1)
        assert torch.equal(packed.input_ids[5:8], seq2)
        assert torch.equal(packed.input_ids[8:], seq3)
    
    def test_block_diagonal_mask(self):
        """Test block-diagonal attention mask generation."""
        config = SequencePackerConfig(enabled=True, use_block_diagonal_mask=True)
        packer = SequencePacker(config)
        
        seq1 = torch.tensor([1, 2, 3])  # Length 3
        seq2 = torch.tensor([10, 11])   # Length 2
        
        packed = packer.pack_tokenized([seq1, seq2])
        
        # Check mask shape
        assert packed.attention_mask.shape == (5, 5)
        
        # Check block diagonal structure
        mask = packed.attention_mask
        # First block (seq1: positions 0-2)
        assert mask[0, 0] == 1.0
        assert mask[0, 2] == 1.0
        assert mask[2, 0] == 1.0
        # Second block (seq2: positions 3-4)
        assert mask[3, 3] == 1.0
        assert mask[3, 4] == 1.0
        assert mask[4, 3] == 1.0
        # Cross-sequence should be zero
        assert mask[0, 3] == 0.0
        assert mask[3, 0] == 0.0
    
    def test_unpack_hidden_states(self):
        """Test unpacking hidden states back to sequences."""
        config = SequencePackerConfig(enabled=True)
        packer = SequencePacker(config)
        
        seq1 = torch.tensor([1, 2, 3])
        seq2 = torch.tensor([10, 11])
        
        packed = packer.pack_tokenized([seq1, seq2])
        
        # Simulate hidden states output
        hidden_states = torch.randn(5, 768)
        
        unpacked = packer.unpack_hidden_states(hidden_states, packed)
        
        assert len(unpacked) == 2
        assert unpacked[0].shape == (3, 768)
        assert unpacked[1].shape == (2, 768)
    
    def test_get_stats(self):
        """Test packer statistics."""
        config = SequencePackerConfig(enabled=True)
        packer = SequencePacker(config)
        
        # Pack some sequences
        packer.pack_tokenized([torch.tensor([1, 2]), torch.tensor([3, 4, 5])])
        packer.pack_tokenized([torch.tensor([1]), torch.tensor([2])])
        
        stats = packer.get_stats()
        
        assert stats["enabled"] is True
        assert stats["pack_count"] == 2
        assert stats["total_sequences_packed"] == 4


# ---------------------------------------------------------------------------
# GPUSpanFilter Tests
# ---------------------------------------------------------------------------


class TestGPUSpanFilterConfig:
    """Tests for GPUSpanFilterConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration."""
        config = GPUSpanFilterConfig()
        
        assert config.enabled is True
        assert config.confidence_threshold == 0.85
        assert config.max_spans_per_sequence == 100
        assert config.use_topk is False
        assert config.topk_k == 50


class TestGPUSpanFilter:
    """Tests for GPUSpanFilter class."""
    
    def test_filter_below_threshold(self):
        """Test filtering spans below threshold."""
        config = GPUSpanFilterConfig(confidence_threshold=0.5)
        filter = GPUSpanFilter(config)
        
        # Create logits that will produce probabilities above/below threshold
        # Sigmoid(2) ≈ 0.88, Sigmoid(-2) ≈ 0.12
        logits = torch.zeros(4, 4, 3)
        logits[0, 1, 0] = 2.0  # Above threshold
        logits[1, 2, 1] = -2.0  # Below threshold
        logits[2, 3, 2] = 3.0  # Above threshold
        
        spans, scores, mask = filter.filter_span_logits(logits, threshold=0.5)
        
        # Should have 2 valid spans
        assert spans.shape[0] >= 2
        assert all(scores >= 0.5)
    
    def test_filter_empty_result(self):
        """Test filtering with no valid spans."""
        config = GPUSpanFilterConfig(confidence_threshold=0.99)
        filter = GPUSpanFilter(config)
        
        # All logits will produce low probabilities
        logits = torch.full((4, 4, 3), -10.0)
        
        spans, scores, mask = filter.filter_span_logits(logits)
        
        assert spans.shape[0] == 0
        assert scores.shape[0] == 0
    
    def test_max_spans_limit(self):
        """Test max spans per sequence limit."""
        config = GPUSpanFilterConfig(
            confidence_threshold=0.1,
            max_spans_per_sequence=5,
        )
        filter = GPUSpanFilter(config)
        
        # Create many valid spans
        logits = torch.full((10, 10, 3), 5.0)  # All high confidence
        
        spans, scores, _ = filter.filter_span_logits(logits)
        
        assert spans.shape[0] <= 5
    
    def test_topk_selection(self):
        """Test top-k span selection."""
        config = GPUSpanFilterConfig()
        filter = GPUSpanFilter(config)
        
        # Create varied logits
        logits = torch.randn(8, 8, 3)
        
        spans, scores = filter.topk_spans(logits, k=10)
        
        assert spans.shape[0] == 10
        assert scores.shape[0] == 10
        # Scores should be sorted descending
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    
    def test_nms_removes_overlapping(self):
        """Test non-maximum suppression removes overlapping spans."""
        filter = GPUSpanFilter()
        
        # Create overlapping spans with different scores
        spans = torch.tensor([
            [0, 5, 0],  # Higher score, should be kept
            [1, 4, 0],  # Overlaps, lower score, should be suppressed
            [10, 15, 1],  # Non-overlapping, should be kept
        ])
        scores = torch.tensor([0.9, 0.7, 0.8])
        
        filtered_spans, filtered_scores = filter.non_maximum_suppression(
            spans, scores, overlap_threshold=0.3
        )
        
        # First and third should remain, second suppressed
        assert filtered_spans.shape[0] == 2
    
    def test_get_stats(self):
        """Test filter statistics."""
        config = GPUSpanFilterConfig(confidence_threshold=0.5)
        filter = GPUSpanFilter(config)
        
        # Run some filtering
        logits = torch.randn(5, 5, 3)
        filter.filter_span_logits(logits)
        filter.filter_span_logits(logits)
        
        stats = filter.get_stats()
        
        assert stats["filter_calls"] == 2
        assert "filter_rate" in stats


# ---------------------------------------------------------------------------
# AsyncPrefetch Tests
# ---------------------------------------------------------------------------


class TestAsyncPrefetchConfig:
    """Tests for AsyncPrefetchConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration."""
        config = AsyncPrefetchConfig()
        
        assert config.enabled is False
        assert config.queue_size == 4
        assert config.num_workers == 1
        assert config.timeout_seconds == 30.0
        assert config.use_multiprocessing is True
        assert config.pin_memory is True
        assert config.profile_aware is True


class TestPrefetchBatch:
    """Tests for PrefetchBatch dataclass."""
    
    def test_creation(self):
        """Test creating PrefetchBatch."""
        batch = PrefetchBatch(
            batch_id=0,
            texts=["Hello", "World"],
            labels=["label1", "label2"],
            doc_indices=[0, 1],
            chunk_infos=[Mock(), Mock()],
        )
        
        assert batch.batch_id == 0
        assert len(batch.texts) == 2
        assert len(batch.labels) == 2


class TestSyncPrefetchPipeline:
    """Tests for synchronous fallback pipeline."""
    
    def test_submit_and_process(self):
        """Test submitting documents and processing."""
        # Create mock chunker and taxonomy
        mock_chunker = Mock()
        mock_chunk = Mock()
        mock_chunk.text = "chunk text"
        mock_chunker.chunk_text.return_value = [mock_chunk]
        
        mock_taxonomy = Mock()
        mock_taxonomy.get_inference_labels.return_value = ["label1"]
        mock_taxonomy.get_prompts_for_columns.return_value = ["label1"]
        
        pipeline = SyncPrefetchPipeline(
            chunker=mock_chunker,
            taxonomy=mock_taxonomy,
            batch_size=2,
        )
        
        # Submit documents
        pipeline.submit(0, "Document 1", None)
        pipeline.submit(1, "Document 2", ["col1"])
        pipeline.submit(2, "Document 3", None)
        
        # Process
        batches = pipeline.process_all()
        
        assert len(batches) >= 1
        assert all(isinstance(b, PrefetchBatch) for b in batches)


class TestAsyncPrefetchPipeline:
    """Tests for async prefetch pipeline."""
    
    def test_initialization_disabled(self):
        """Test pipeline with prefetch disabled."""
        config = AsyncPrefetchConfig(enabled=False)
        pipeline = AsyncPrefetchPipeline(config=config)
        
        assert pipeline.config.enabled is False
    
    def test_profile_aware_settings_laptop(self):
        """Test profile-aware settings for LAPTOP."""
        config = AsyncPrefetchConfig(
            enabled=True,
            profile_aware=True,
            queue_size=8,
            num_workers=2,
        )
        pipeline = AsyncPrefetchPipeline(
            config=config,
            profile=ProfileType.LAPTOP,
        )
        
        # Laptop should use threading and smaller queue
        assert pipeline.config.use_multiprocessing is False
        assert pipeline.config.queue_size <= 4
        assert pipeline.config.num_workers == 1
    
    def test_profile_aware_settings_hpc(self):
        """Test profile-aware settings for HPC."""
        config = AsyncPrefetchConfig(
            enabled=True,
            profile_aware=True,
            queue_size=4,
            num_workers=1,
        )
        pipeline = AsyncPrefetchPipeline(
            config=config,
            profile=ProfileType.HPC,
        )
        
        # HPC should use multiprocessing and larger queue
        assert pipeline.config.use_multiprocessing is True
        assert pipeline.config.queue_size >= 8
        assert pipeline.config.num_workers >= 2
    
    def test_get_stats(self):
        """Test pipeline statistics."""
        config = AsyncPrefetchConfig(enabled=False)
        pipeline = AsyncPrefetchPipeline(config=config, profile=ProfileType.LAPTOP)
        
        stats = pipeline.get_stats()
        
        assert "enabled" in stats
        assert "running" in stats
        assert "profile" in stats


class TestCreatePrefetchConfigFromDict:
    """Tests for create_prefetch_config_from_dict factory."""
    
    def test_creates_from_config_dict(self):
        """Test creating config from dict."""
        config = {
            "execution": {
                "async_prefetch": {
                    "enabled": True,
                    "queue_size": 8,
                    "num_workers": 2,
                    "timeout_seconds": 60.0,
                    "use_multiprocessing": False,
                }
            }
        }
        
        prefetch_cfg = create_prefetch_config_from_dict(config)
        
        assert prefetch_cfg.enabled is True
        assert prefetch_cfg.queue_size == 8
        assert prefetch_cfg.num_workers == 2
        assert prefetch_cfg.timeout_seconds == 60.0
        assert prefetch_cfg.use_multiprocessing is False


# ---------------------------------------------------------------------------
# Factory Function Tests
# ---------------------------------------------------------------------------


class TestCreateSpanFilterFromConfig:
    """Tests for create_span_filter_from_config factory."""
    
    def test_creates_from_config(self):
        """Test creating GPUSpanFilter from config."""
        config = {
            "gliner": {
                "confidence_threshold": 0.9,
                "gpu_span_filter": {
                    "enabled": True,
                    "max_spans_per_sequence": 50,
                    "use_topk": True,
                    "topk_k": 25,
                }
            }
        }
        
        filter = create_span_filter_from_config(config)
        
        assert filter.config.enabled is True
        assert filter.config.confidence_threshold == 0.9
        assert filter.config.max_spans_per_sequence == 50
        assert filter.config.use_topk is True
        assert filter.config.topk_k == 25


class TestCreateSequencePackerFromConfig:
    """Tests for create_sequence_packer_from_config factory."""
    
    def test_creates_from_config(self):
        """Test creating SequencePacker from config."""
        config = {
            "execution": {
                "sequence_packing": {
                    "enabled": True,
                    "max_packed_length": 1024,
                    "min_sequences_to_pack": 3,
                    "use_block_diagonal_mask": False,
                }
            }
        }
        
        packer = create_sequence_packer_from_config(config)
        
        assert packer.config.enabled is True
        assert packer.config.max_packed_length == 1024
        assert packer.config.min_sequences_to_pack == 3
        assert packer.config.use_block_diagonal_mask is False
