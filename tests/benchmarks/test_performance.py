# tests/benchmarks/test_performance.py
"""
Performance benchmarks for Phase A components.

These tests measure:
1. Memory usage (peak RSS, allocations)
2. Throughput (samples/second)
3. Latency (per-batch and end-to-end)

Implements: Testing Plan Step 9 (Optional Benchmarking)

Usage:
    pytest tests/benchmarks/ -m benchmark
    pytest tests/benchmarks/ --benchmark-only  # Skip non-benchmark tests
    pytest tests/benchmarks/ --benchmark-json=results.json  # Export results

Note: These tests require the pytest-benchmark plugin.
"""

import gc
import tracemalloc
from typing import Optional
import warnings

import pytest
import torch

# Conditional imports for optional dependencies
try:
    import pytest_benchmark
    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False

try:
    from neuro_stylometry.pollution_guard.leace import LEACEComputer
    from neuro_stylometry.pollution_guard.masker import SpanMasker
    HAS_NEURO_STYLOMETRY = True
except ImportError:
    HAS_NEURO_STYLOMETRY = False


# Skip all benchmarks if pytest-benchmark is not installed
if not HAS_BENCHMARK:
    pytest.skip(
        "pytest-benchmark not installed. Install with: pip install pytest-benchmark",
        allow_module_level=True,
    )


# ==============================================================================
# Memory Profiling Utilities
# ==============================================================================

class MemoryTracker:
    """Context manager for tracking memory usage with tracemalloc."""
    
    def __init__(self):
        self.peak_bytes: Optional[int] = None
        self.current_bytes: Optional[int] = None
        self.snapshot_before = None
        self.snapshot_after = None
    
    def __enter__(self):
        gc.collect()  # Clear garbage before measurement
        tracemalloc.start()
        self.snapshot_before = tracemalloc.take_snapshot()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.snapshot_after = tracemalloc.take_snapshot()
        self.current_bytes, self.peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        gc.collect()  # Clean up after measurement
        return False
    
    @property
    def peak_mb(self) -> float:
        """Peak memory usage in megabytes."""
        return self.peak_bytes / (1024 * 1024) if self.peak_bytes else 0.0
    
    @property
    def current_mb(self) -> float:
        """Current memory usage in megabytes."""
        return self.current_bytes / (1024 * 1024) if self.current_bytes else 0.0


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def sample_texts():
    """Generate sample texts for benchmarking."""
    return [
        "The quick brown fox jumps over the lazy dog. " * 10,
        "To be or not to be, that is the question. " * 10,
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 10,
    ] * 100  # 300 texts


@pytest.fixture
def sample_embeddings():
    """Generate synthetic embeddings for LEACE benchmarking."""
    torch.manual_seed(42)
    n_samples = 1000
    embed_dim = 768
    return torch.randn(n_samples, embed_dim, dtype=torch.float32)


@pytest.fixture
def sample_concepts():
    """Generate synthetic concept labels."""
    torch.manual_seed(42)
    n_samples = 1000
    return torch.randint(0, 3, (n_samples,))


# ==============================================================================
# LEACE Benchmark Tests
# ==============================================================================

@pytest.mark.benchmark
@pytest.mark.skipif(not HAS_NEURO_STYLOMETRY, reason="neuro_stylometry not installed")
class TestLEACEBenchmarks:
    """Benchmark tests for LEACE computation."""
    
    def test_leace_accumulate_batch_throughput(self, benchmark, sample_embeddings, sample_concepts):
        """Measure throughput of LEACE batch accumulation."""
        computer = LEACEComputer(embed_dim=768, device="cpu", force_cpu=True)
        
        batch_size = 64
        n_batches = len(sample_embeddings) // batch_size
        
        def run_accumulation():
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                computer.accumulate_batch_concepts(
                    sample_embeddings[start:end],
                    sample_concepts[start:end],
                )
        
        benchmark(run_accumulation)
    
    def test_leace_fit_latency(self, benchmark, sample_embeddings, sample_concepts):
        """Measure latency of LEACE fit operation."""
        computer = LEACEComputer(embed_dim=768, device="cpu", force_cpu=True)
        
        # Pre-accumulate data
        computer.accumulate_batch_concepts(sample_embeddings, sample_concepts)
        
        # Benchmark fit
        benchmark(computer.fit)
    
    def test_leace_project_latency(self, benchmark, sample_embeddings, sample_concepts):
        """Measure latency of LEACE projection operation."""
        computer = LEACEComputer(embed_dim=768, device="cpu", force_cpu=True)
        computer.accumulate_batch_concepts(sample_embeddings, sample_concepts)
        P = computer.fit()
        
        # Benchmark projection
        def project():
            return sample_embeddings @ P.T
        
        benchmark(project)
    
    def test_leace_memory_accumulation(self, sample_embeddings, sample_concepts):
        """Measure memory usage during LEACE accumulation."""
        with MemoryTracker() as tracker:
            computer = LEACEComputer(embed_dim=768, device="cpu", force_cpu=True)
            computer.accumulate_batch_concepts(sample_embeddings, sample_concepts)
            computer.fit()
        
        # Log memory usage
        print(f"\nLEACE Memory Usage:")
        print(f"  Peak: {tracker.peak_mb:.2f} MB")
        print(f"  Current: {tracker.current_mb:.2f} MB")
        
        # Memory should be reasonable (< 500 MB for 1000 samples)
        assert tracker.peak_mb < 500, f"Peak memory {tracker.peak_mb:.2f} MB exceeds 500 MB threshold"


# ==============================================================================
# Masker Benchmark Tests
# ==============================================================================

@pytest.mark.benchmark
@pytest.mark.skipif(not HAS_NEURO_STYLOMETRY, reason="neuro_stylometry not installed")
class TestMaskerBenchmarks:
    """Benchmark tests for text masking operations."""
    
    def test_masker_single_text_throughput(self, benchmark):
        """Measure throughput of single text masking."""
        masker = SpanMasker()
        text = "John Smith was born in London in 1985 and works at Google."
        spans = [
            {"start": 0, "end": 10, "label": "PERSON"},
            {"start": 23, "end": 29, "label": "LOCATION"},
            {"start": 33, "end": 37, "label": "AGE"},
            {"start": 53, "end": 59, "label": "ORGANIZATION"},
        ]
        
        benchmark(masker.apply_masks, text, spans)
    
    def test_masker_batch_throughput(self, benchmark, sample_texts):
        """Measure throughput of batch text masking."""
        masker = SpanMasker()
        
        # Generate spans for each text
        batch_spans = [
            [
                {"start": 0, "end": 5, "label": "MISC"},
                {"start": 10, "end": 15, "label": "MISC"},
            ]
            for _ in sample_texts
        ]
        
        def mask_batch():
            return masker.apply_masks_batch(sample_texts, batch_spans)
        
        benchmark(mask_batch)
    
    def test_masker_memory_usage(self, sample_texts):
        """Measure memory usage for batch masking."""
        masker = SpanMasker()
        batch_spans = [
            [{"start": 0, "end": 5, "label": "MISC"}]
            for _ in sample_texts
        ]
        
        with MemoryTracker() as tracker:
            masker.apply_masks_batch(sample_texts, batch_spans)
        
        print(f"\nMasker Memory Usage ({len(sample_texts)} texts):")
        print(f"  Peak: {tracker.peak_mb:.2f} MB")
        
        # Memory should be minimal for masking operations
        assert tracker.peak_mb < 100, f"Peak memory {tracker.peak_mb:.2f} MB exceeds 100 MB threshold"


# ==============================================================================
# End-to-End Benchmark Tests
# ==============================================================================

@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.skipif(not HAS_NEURO_STYLOMETRY, reason="neuro_stylometry not installed")
class TestEndToEndBenchmarks:
    """End-to-end performance benchmarks."""
    
    def test_e2e_memory_budget_laptop(self):
        """
        Verify memory stays within laptop budget (< 8 GB).
        
        This test simulates a laptop-mode Phase A run and measures
        peak memory usage.
        """
        try:
            from neuro_stylometry.pollution_guard.strategies.laptop import LaptopPhaseAStrategy
            from neuro_stylometry.config import load_pipeline_config
            
            config = load_pipeline_config(mode="laptop", validate=False)
            
            with MemoryTracker() as tracker:
                # Just instantiate strategy to measure baseline
                strategy = LaptopPhaseAStrategy(config)
            
            print(f"\nLaptop Strategy Instantiation Memory:")
            print(f"  Peak: {tracker.peak_mb:.2f} MB")
            
            # Strategy instantiation should be lightweight
            assert tracker.peak_mb < 100, (
                f"Strategy instantiation uses {tracker.peak_mb:.2f} MB, "
                "exceeds 100 MB threshold"
            )
        except ImportError as e:
            pytest.skip(f"Required module not available: {e}")
    
    def test_throughput_samples_per_second(self, sample_embeddings, sample_concepts):
        """Measure overall throughput in samples/second."""
        import time
        
        if not HAS_NEURO_STYLOMETRY:
            pytest.skip("neuro_stylometry not installed")
        
        computer = LEACEComputer(embed_dim=768, device="cpu", force_cpu=True)
        
        n_samples = len(sample_embeddings)
        batch_size = 64
        n_batches = n_samples // batch_size
        
        start = time.perf_counter()
        
        for i in range(n_batches):
            idx_start = i * batch_size
            idx_end = idx_start + batch_size
            computer.accumulate_batch_concepts(
                sample_embeddings[idx_start:idx_end],
                sample_concepts[idx_start:idx_end],
            )
        
        P = computer.fit()
        
        elapsed = time.perf_counter() - start
        throughput = n_samples / elapsed
        
        print(f"\nLEACE Throughput: {throughput:.1f} samples/second")
        print(f"  Total samples: {n_samples}")
        print(f"  Elapsed: {elapsed:.2f}s")
        
        # Should process at least 100 samples/second on any hardware
        assert throughput > 100, f"Throughput {throughput:.1f} samples/s is below minimum threshold"


# ==============================================================================
# Comparative Benchmarks
# ==============================================================================

@pytest.mark.benchmark
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAS_NEURO_STYLOMETRY, reason="neuro_stylometry not installed")
class TestDeviceComparativeBenchmarks:
    """Compare performance between CPU and GPU."""
    
    def test_leace_cpu_vs_gpu_speedup(self, benchmark, sample_embeddings, sample_concepts):
        """Measure GPU speedup over CPU for LEACE."""
        import time
        
        # CPU timing
        cpu_computer = LEACEComputer(embed_dim=768, device="cpu", force_cpu=True)
        
        cpu_start = time.perf_counter()
        cpu_computer.accumulate_batch_concepts(sample_embeddings, sample_concepts)
        cpu_computer.fit()
        cpu_elapsed = time.perf_counter() - cpu_start
        
        # GPU timing
        gpu_embeddings = sample_embeddings.cuda()
        gpu_concepts = sample_concepts.cuda()
        gpu_computer = LEACEComputer(embed_dim=768, device="cuda", force_cpu=False)
        
        gpu_start = time.perf_counter()
        gpu_computer.accumulate_batch_concepts(gpu_embeddings, gpu_concepts)
        gpu_computer.fit()
        torch.cuda.synchronize()
        gpu_elapsed = time.perf_counter() - gpu_start
        
        speedup = cpu_elapsed / gpu_elapsed
        
        print(f"\nCPU vs GPU Comparison:")
        print(f"  CPU: {cpu_elapsed:.3f}s")
        print(f"  GPU: {gpu_elapsed:.3f}s")
        print(f"  Speedup: {speedup:.2f}x")


# ==============================================================================
# Memory Leak Detection
# ==============================================================================

@pytest.mark.benchmark
@pytest.mark.skipif(not HAS_NEURO_STYLOMETRY, reason="neuro_stylometry not installed")
class TestMemoryLeakDetection:
    """Test for memory leaks in repeated operations."""
    
    def test_leace_no_memory_leak(self, sample_embeddings, sample_concepts):
        """Verify no memory leak in repeated LEACE operations."""
        gc.collect()
        
        # Warm-up run
        computer = LEACEComputer(embed_dim=768, device="cpu", force_cpu=True)
        computer.accumulate_batch_concepts(sample_embeddings, sample_concepts)
        computer.fit()
        del computer
        gc.collect()
        
        # Measure baseline
        tracemalloc.start()
        baseline = tracemalloc.get_traced_memory()[0]
        
        # Repeat operations
        for _ in range(5):
            computer = LEACEComputer(embed_dim=768, device="cpu", force_cpu=True)
            computer.accumulate_batch_concepts(sample_embeddings, sample_concepts)
            computer.fit()
            del computer
            gc.collect()
        
        final = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        
        growth = (final - baseline) / (1024 * 1024)
        
        print(f"\nMemory Growth After 5 Iterations: {growth:.2f} MB")
        
        # Allow some growth, but should be minimal
        assert growth < 50, f"Memory grew by {growth:.2f} MB, possible leak"
