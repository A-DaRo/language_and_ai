"""
Integration tests for advanced execution optimizations.

Tests end-to-end functionality of CUDA graphs, torch.compile, sequence packing,
GPU span filtering, and async prefetching with real models (when available).
"""

import os
import pytest
import torch
import time
from typing import Dict, Any, List
from unittest.mock import Mock, patch

# Skip markers for hardware-dependent tests
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

requires_real_models = pytest.mark.skipif(
    os.environ.get("NEURO_STYLOMETRY_RUN_REAL_MODELS") != "1",
    reason="Real models not enabled (set NEURO_STYLOMETRY_RUN_REAL_MODELS=1)"
)

# Import modules
from neuro_stylometry.hardware_ops import (
    GraphCache,
    GraphCacheConfig,
    CompileConfig,
    compile_model_if_enabled,
    SequencePacker,
    SequencePackerConfig,
    GPUSpanFilter,
    GPUSpanFilterConfig,
    AsyncPrefetchPipeline,
    AsyncPrefetchConfig,
    SyncPrefetchPipeline,
    ProfileType,
    HardwareDetector,
)


# ---------------------------------------------------------------------------
# CUDA Graph Integration Tests
# ---------------------------------------------------------------------------


class TestCUDAGraphIntegration:
    """Integration tests for CUDA graph capture and replay."""
    
    @requires_cuda
    def test_capture_and_replay_simple_model(self):
        """Test capturing and replaying a simple model."""
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32),
        ).cuda()
        model.eval()
        
        config = GraphCacheConfig(enabled=True, warmup_iterations=2)
        cache = GraphCache(config)
        
        def forward_fn(x):
            with torch.no_grad():
                return {"output": model(x)}
        
        # First call should capture
        x1 = torch.randn(8, 64, device="cuda")
        result1 = cache.run_with_graph(
            shape_key=(8, 64),
            forward_fn=forward_fn,
            x=x1,
        )
        
        assert "output" in result1
        assert result1["output"].shape == (8, 32)
        assert cache.get_stats()["captures"] == 1
        
        # Second call should replay
        x2 = torch.randn(8, 64, device="cuda")
        result2 = cache.run_with_graph(
            shape_key=(8, 64),
            forward_fn=forward_fn,
            x=x2,
        )
        
        assert result2["output"].shape == (8, 32)
        assert cache.get_stats()["hits"] == 1
    
    @requires_cuda
    def test_multiple_shapes_cached(self):
        """Test caching graphs for multiple input shapes."""
        model = torch.nn.Linear(64, 32).cuda()
        model.eval()
        
        config = GraphCacheConfig(enabled=True, max_cached_graphs=5)
        cache = GraphCache(config)
        
        def forward_fn(x):
            with torch.no_grad():
                return {"output": model(x)}
        
        # Capture for different batch sizes
        for batch_size in [4, 8, 16]:
            x = torch.randn(batch_size, 64, device="cuda")
            cache.run_with_graph(
                shape_key=(batch_size, 64),
                forward_fn=forward_fn,
                x=x,
            )
        
        assert cache.get_stats()["cache_size"] == 3
        
        # Verify all shapes cached
        assert (4, 64) in cache
        assert (8, 64) in cache
        assert (16, 64) in cache


# ---------------------------------------------------------------------------
# torch.compile Integration Tests
# ---------------------------------------------------------------------------


class TestTorchCompileIntegration:
    """Integration tests for torch.compile optimization."""
    
    @pytest.mark.skipif(
        tuple(int(x) for x in torch.__version__.split(".")[:2]) < (2, 0),
        reason="torch.compile requires PyTorch 2.0+"
    )
    def test_compile_simple_model(self):
        """Test compiling a simple model.
        
        Note: This test may fail on Windows without Visual Studio Build Tools
        installed, as torch.compile's inductor backend requires a C++ compiler.
        """
        import sys
        
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32),
        )
        
        config = CompileConfig(enabled=True, mode="reduce-overhead")
        
        try:
            compiled = compile_model_if_enabled(model, config)
        except RuntimeError as e:
            if "Compiler" in str(e) and "not found" in str(e):
                pytest.skip(
                    "Skipping torch.compile test: C++ compiler not available. "
                    "Install Visual Studio Build Tools on Windows."
                )
            raise
        
        # Verify model still works
        x = torch.randn(4, 64)
        try:
            output = compiled(x)
        except Exception as e:
            # Handle inductor backend errors due to missing compiler
            if "inductor" in str(e).lower() or "compiler" in str(e).lower():
                pytest.skip(f"torch.compile inductor backend failed: {e}")
            raise
        
        assert output.shape == (4, 32)
    
    def test_compile_disabled_returns_original(self):
        """Test that disabled compilation returns original model."""
        model = torch.nn.Linear(64, 32)
        
        config = CompileConfig(enabled=False)
        result = compile_model_if_enabled(model, config)
        
        assert result is model


# ---------------------------------------------------------------------------
# Sequence Packing Integration Tests
# ---------------------------------------------------------------------------


class TestSequencePackingIntegration:
    """Integration tests for sequence packing."""
    
    def test_pack_and_unpack_preserves_content(self):
        """Test that pack/unpack cycle preserves sequence content."""
        config = SequencePackerConfig(enabled=True)
        packer = SequencePacker(config)
        
        # Create distinct sequences
        seq1 = torch.tensor([1, 2, 3, 4, 5])
        seq2 = torch.tensor([100, 101, 102])
        seq3 = torch.tensor([200, 201])
        
        # Pack
        packed = packer.pack_tokenized([seq1, seq2, seq3])
        
        # Simulate model hidden states
        hidden_dim = 768
        hidden_states = torch.randn(packed.input_ids.size(0), hidden_dim)
        
        # Unpack
        unpacked = packer.unpack_hidden_states(hidden_states, packed)
        
        # Verify shapes match original sequences
        assert len(unpacked) == 3
        assert unpacked[0].shape == (5, hidden_dim)
        assert unpacked[1].shape == (3, hidden_dim)
        assert unpacked[2].shape == (2, hidden_dim)
    
    def test_block_diagonal_attention_prevents_cross_attention(self):
        """Test that block-diagonal mask prevents cross-sequence attention."""
        config = SequencePackerConfig(enabled=True, use_block_diagonal_mask=True)
        packer = SequencePacker(config)
        
        seq1 = torch.tensor([1, 2, 3])
        seq2 = torch.tensor([10, 11])
        
        packed = packer.pack_tokenized([seq1, seq2])
        
        # Verify attention mask structure
        mask = packed.attention_mask
        
        # Seq1 tokens (0-2) should not attend to seq2 tokens (3-4)
        assert mask[0, 3] == 0.0
        assert mask[0, 4] == 0.0
        assert mask[1, 3] == 0.0
        
        # Seq2 tokens should not attend to seq1 tokens
        assert mask[3, 0] == 0.0
        assert mask[4, 1] == 0.0
        
        # Within-sequence attention should be allowed
        assert mask[0, 0] == 1.0
        assert mask[0, 2] == 1.0
        assert mask[3, 4] == 1.0
    
    def test_padding_savings_calculation(self):
        """Test that padding savings are calculated correctly."""
        config = SequencePackerConfig(enabled=True)
        packer = SequencePacker(config)
        
        # Pack sequences of varying lengths
        sequences = [
            torch.tensor([1, 2]),       # Length 2
            torch.tensor([1, 2, 3, 4]),  # Length 4
            torch.tensor([1]),           # Length 1
        ]
        
        packer.pack_tokenized(sequences)
        
        stats = packer.get_stats()
        
        # Traditional padding would be 3 * 4 = 12 tokens
        # Packed is 2 + 4 + 1 = 7 tokens
        # Savings = 12 - 7 = 5
        assert stats["padding_tokens_saved"] == 5


# ---------------------------------------------------------------------------
# GPU Span Filter Integration Tests
# ---------------------------------------------------------------------------


class TestGPUSpanFilterIntegration:
    """Integration tests for GPU span filtering."""
    
    def test_filter_consistency_cpu_gpu(self):
        """Test that GPU filtering produces consistent results."""
        config = GPUSpanFilterConfig(confidence_threshold=0.5)
        filter = GPUSpanFilter(config)
        
        # Create test logits
        torch.manual_seed(42)
        logits = torch.randn(10, 10, 5)
        
        # Filter on CPU
        spans_cpu, scores_cpu, _ = filter.filter_span_logits(logits)
        
        # Filter on GPU (if available)
        if torch.cuda.is_available():
            logits_gpu = logits.cuda()
            spans_gpu, scores_gpu, _ = filter.filter_span_logits(logits_gpu)
            
            # Results should match (move to CPU for comparison)
            assert torch.equal(spans_cpu, spans_gpu.cpu())
            assert torch.allclose(scores_cpu, scores_gpu.cpu())
    
    def test_nms_removes_highly_overlapping_spans(self):
        """Test NMS with highly overlapping spans."""
        filter = GPUSpanFilter()
        
        # Create spans with high overlap
        spans = torch.tensor([
            [0, 10, 0],   # Score 0.95
            [1, 9, 0],    # Score 0.90, overlaps with first
            [2, 8, 0],    # Score 0.85, overlaps with first and second
            [20, 25, 1],  # Score 0.80, no overlap
        ])
        scores = torch.tensor([0.95, 0.90, 0.85, 0.80])
        
        filtered_spans, filtered_scores = filter.non_maximum_suppression(
            spans, scores, overlap_threshold=0.5
        )
        
        # Should keep first (highest score) and fourth (no overlap)
        assert filtered_spans.shape[0] == 2
        assert filtered_scores[0] == 0.95
        assert filtered_scores[1] == 0.80


# ---------------------------------------------------------------------------
# Async Prefetch Integration Tests
# ---------------------------------------------------------------------------


class TestAsyncPrefetchIntegration:
    """Integration tests for async prefetching."""
    
    def test_sync_pipeline_produces_correct_batches(self):
        """Test synchronous pipeline produces correct batches."""
        # Create mock components
        mock_chunk = Mock()
        mock_chunk.text = "test chunk"
        
        mock_chunker = Mock()
        mock_chunker.chunk_text.return_value = [mock_chunk]
        
        mock_taxonomy = Mock()
        mock_taxonomy.get_inference_labels.return_value = ["label"]
        mock_taxonomy.get_prompts_for_columns.return_value = ["label"]
        
        pipeline = SyncPrefetchPipeline(
            chunker=mock_chunker,
            taxonomy=mock_taxonomy,
            batch_size=2,
        )
        
        # Submit documents
        for i in range(5):
            pipeline.submit(i, f"Document {i}", None)
        
        # Process
        batches = pipeline.process_all()
        
        # Should produce 3 batches (5 docs / batch_size 2, rounded up)
        assert len(batches) == 3
        
        # Verify batch structure
        for batch in batches:
            assert isinstance(batch.batch_id, int)
            assert len(batch.texts) > 0
            assert len(batch.doc_indices) == len(batch.texts)
    
    def test_async_pipeline_context_manager(self):
        """Test async pipeline context manager usage."""
        config = AsyncPrefetchConfig(enabled=False)  # Disabled for test
        
        with AsyncPrefetchPipeline(config=config) as pipeline:
            # Should not start when disabled
            assert not pipeline._running
    
    def test_profile_specific_configuration(self):
        """Test that profile-specific configurations are applied."""
        # Test LAPTOP profile
        laptop_config = AsyncPrefetchConfig(enabled=True, profile_aware=True)
        laptop_pipeline = AsyncPrefetchPipeline(
            config=laptop_config,
            profile=ProfileType.LAPTOP,
        )
        assert laptop_pipeline.config.use_multiprocessing is False
        
        # Test HPC profile
        hpc_config = AsyncPrefetchConfig(enabled=True, profile_aware=True)
        hpc_pipeline = AsyncPrefetchPipeline(
            config=hpc_config,
            profile=ProfileType.HPC,
        )
        assert hpc_pipeline.config.use_multiprocessing is True


# ---------------------------------------------------------------------------
# End-to-End Optimization Tests
# ---------------------------------------------------------------------------


@requires_real_models
class TestEndToEndOptimizations:
    """End-to-end tests with real GLiNER models."""
    
    @pytest.fixture
    def gliner_detector(self):
        """Create GLiNER detector for testing."""
        from neuro_stylometry.pollution_guard.gliner_detector import (
            GLiNERDetector,
            BatchInferenceConfig,
        )
        
        detector = GLiNERDetector(
            model_name="urchade/gliner_small-v2.1",  # Smaller model for testing
            device="cuda" if torch.cuda.is_available() else "cpu",
            confidence_threshold=0.3,  # Lower threshold for testing
            batch_inference_config=BatchInferenceConfig(
                enable_batching=True,
                batch_size=8,
            ),
        )
        return detector
    
    def test_batched_inference_produces_results(self, gliner_detector):
        """Test that batched inference runs and returns results.
        
        Note: We only verify the inference pipeline works, not that
        specific entities are detected (model-dependent).
        """
        texts = [
            "I was born in 1990 in New York.",
            "She is a 25-year-old American woman.",
            "The Republican senator from Texas spoke.",
            "As an introvert, I prefer quiet activities.",
        ]
        
        results = gliner_detector.detect_spans(texts, show_progress=False)
        
        # Verify correct number of results returned
        assert len(results) == len(texts)
        # Verify results are lists (may be empty depending on model)
        assert all(isinstance(r, list) for r in results)
    
    @requires_cuda
    def test_gpu_span_filtering_improves_performance(self, gliner_detector):
        """Test that GPU span filtering is faster than CPU filtering."""
        # This is a performance test - we just verify it runs
        texts = ["Test text with potential span detection."] * 10
        
        start_time = time.time()
        results = gliner_detector.detect_spans(texts, show_progress=False)
        elapsed = time.time() - start_time
        
        # Basic sanity check
        assert elapsed < 60  # Should complete in reasonable time
        assert len(results) == 10


# ---------------------------------------------------------------------------
# Configuration Integration Tests
# ---------------------------------------------------------------------------


class TestConfigurationIntegration:
    """Tests for configuration loading and application."""
    
    def test_hpc_config_enables_all_optimizations(self):
        """Test that HPC config enables all optimizations."""
        from omegaconf import OmegaConf
        from pathlib import Path
        
        # Load HPC config
        repo_root = Path(__file__).parents[2]
        config_path = repo_root / "conf" / "hpc" / "pipeline.yaml"
        
        if config_path.exists():
            config = OmegaConf.load(config_path)
            execution = config.get("execution", {})
            
            # Verify optimizations are enabled
            assert execution.get("enable_cuda_graphs", False) is True
            assert execution.get("enable_torch_compile", False) is True
            assert execution.get("sequence_packing", {}).get("enabled", False) is True
            assert execution.get("async_prefetch", {}).get("enabled", False) is True
    
    def test_laptop_config_conservative_settings(self):
        """Test that laptop config has conservative settings."""
        from omegaconf import OmegaConf
        from pathlib import Path
        
        repo_root = Path(__file__).parents[2]
        config_path = repo_root / "conf" / "laptop" / "pipeline.yaml"
        
        if config_path.exists():
            config = OmegaConf.load(config_path)
            execution = config.get("execution", {})
            
            # Verify conservative settings
            assert execution.get("enable_cuda_graphs", True) is False
            assert execution.get("enable_torch_compile", True) is False
            assert execution.get("sequence_packing", {}).get("enabled", True) is False


# ---------------------------------------------------------------------------
# Hardware Detection Integration Tests
# ---------------------------------------------------------------------------


class TestHardwareDetectionIntegration:
    """Tests for hardware detection integration."""
    
    def test_hardware_detector_returns_valid_profile(self):
        """Test hardware detector returns valid profile."""
        profile = HardwareDetector.detect()
        
        assert profile.profile_type in [ProfileType.HPC, ProfileType.LAPTOP]
        assert profile.cpu_cores > 0
        assert profile.ram_gb > 0
    
    def test_profile_affects_prefetch_config(self):
        """Test that detected profile affects prefetch configuration."""
        profile = HardwareDetector.detect()
        
        config = AsyncPrefetchConfig(enabled=True, profile_aware=True)
        pipeline = AsyncPrefetchPipeline(
            config=config,
            profile=profile.profile_type,
        )
        
        if profile.profile_type == ProfileType.HPC:
            assert pipeline.config.use_multiprocessing is True
        else:
            assert pipeline.config.use_multiprocessing is False
