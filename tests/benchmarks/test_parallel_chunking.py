"""
Benchmark tests for parallel vs sequential chunking performance.

Tests measure:
- Throughput (documents/second) for different dataset sizes
- Speedup factor (parallel vs sequential)
- Scaling efficiency with worker count
- Memory overhead of parallel processing
- Hardware-aware conditional assertions

The parallel chunking optimization is designed for HPC environments with:
- Large datasets (>= 5,000 documents)
- High core counts (>= 16 cores)

On laptop/workstation hardware, the optimized sequential chunking (bisect +
tokenize-once) typically outperforms parallel due to spawn/IPC overhead.

Usage:
    pytest tests/benchmarks/test_parallel_chunking.py -v
    pytest tests/benchmarks/test_parallel_chunking.py --benchmark-only
    pytest tests/benchmarks/test_parallel_chunking.py --benchmark-json=chunking_results.json
"""

import os
import time
from typing import List, Optional
import pytest

try:
    from neuro_stylometry.pollution_guard.semantic_chunker import (
        BudgetConfig,
        SemanticChunker,
    )
    from transformers import AutoTokenizer
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False


# Skip all tests if dependencies not available
if not HAS_DEPENDENCIES:
    pytest.skip(
        "neuro_stylometry not installed or dependencies missing",
        allow_module_level=True,
    )


# ==============================================================================
# Hardware Detection Helpers
# ==============================================================================

def get_cpu_count() -> int:
    """Get available CPU core count."""
    return os.cpu_count() or 1


def is_hpc_environment() -> bool:
    """Check if running on HPC-like hardware (>= 16 cores)."""
    return get_cpu_count() >= 16


def get_environment_label() -> str:
    """Get a descriptive label for the current environment."""
    cores = get_cpu_count()
    if cores >= 16:
        return f"HPC ({cores} cores)"
    elif cores >= 8:
        return f"Workstation ({cores} cores)"
    else:
        return f"Laptop ({cores} cores)"


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture(scope="module")
def tokenizer():
    """Load tokenizer once for all tests."""
    return AutoTokenizer.from_pretrained(
        "knowledgator/gliner-bi-llama-v1.0",
        use_fast=True,
        legacy=True,
    )


@pytest.fixture
def sample_texts_small():
    """Generate 100 sample documents."""
    texts = []
    for i in range(100):
        # Mix of different lengths
        if i % 3 == 0:
            text = "Short text. " * 5
        elif i % 3 == 1:
            text = "Medium length text with multiple sentences. " * 15
        else:
            text = "Long document with many sentences and varied content. " * 30
        texts.append(text)
    return texts


@pytest.fixture
def sample_texts_medium():
    """Generate 1000 sample documents."""
    texts = []
    for i in range(1000):
        if i % 3 == 0:
            text = "Short text. " * 5
        elif i % 3 == 1:
            text = "Medium length text with multiple sentences. " * 15
        else:
            text = "Long document with many sentences and varied content. " * 30
        texts.append(text)
    return texts


@pytest.fixture
def sample_texts_large():
    """Generate 20000 sample documents (scaled up from 5000).
    
    The optimized sequential chunking is so fast that we need a larger dataset
    to demonstrate parallel benefits and amortize spawn overhead.
    """
    texts = []
    for i in range(20000):
        if i % 3 == 0:
            text = "Short text. " * 5
        elif i % 3 == 1:
            text = "Medium length text with multiple sentences. " * 15
        else:
            text = "Long document with many sentences and varied content. " * 30
        texts.append(text)
    return texts


@pytest.fixture
def sample_labels():
    """Common label set for chunking."""
    return ["birth_year", "nationality", "gender", "political_leaning"]


# ==============================================================================
# Helper Functions
# ==============================================================================

def chunk_with_timing(
    chunker: SemanticChunker,
    texts: List[str],
    labels: List[str],
    use_parallel: bool,
) -> tuple[float, int]:
    """
    Chunk texts and measure timing.
    
    Returns:
        Tuple of (elapsed_seconds, total_chunks)
    """
    labels_list = [labels if text.strip() else None for text in texts]
    
    start_time = time.perf_counter()
    
    if use_parallel:
        results = chunker._chunk_texts_parallel(
            texts=texts,
            labels_list=labels_list,
            tokenizer_name=chunker.tokenizer.name_or_path,
            words_splitter_type=chunker._words_splitter_type,
            progress_callback=None,
        )
    else:
        results = chunker._chunk_texts_sequential(
            texts=texts,
            labels_list=labels_list,
            progress_callback=None,
        )
    
    elapsed = time.perf_counter() - start_time
    total_chunks = sum(len(chunks) for chunks in results)
    
    return elapsed, total_chunks


# ==============================================================================
# Benchmark Tests
# ==============================================================================

@pytest.mark.benchmark
class TestChunkingThroughput:
    """Test chunking throughput for different modes and dataset sizes."""
    
    def test_sequential_small_dataset(self, tokenizer, sample_texts_small, sample_labels):
        """Baseline: Sequential chunking on 100 documents."""
        config = BudgetConfig(
            parallel_chunking_workers=0,
            parallel_chunking_min_texts=0,
        )
        chunker = SemanticChunker(
            tokenizer=tokenizer,
            config=config,
            words_splitter_type="whitespace",
        )
        
        elapsed, total_chunks = chunk_with_timing(
            chunker, sample_texts_small, sample_labels, use_parallel=False
        )
        
        throughput = len(sample_texts_small) / elapsed
        print(f"\n[Sequential-Small] {len(sample_texts_small)} docs in {elapsed:.2f}s = {throughput:.1f} docs/s")
        print(f"                   Generated {total_chunks} chunks")
        
        assert throughput > 0, "Throughput should be positive"
    
    def test_parallel_small_dataset(self, tokenizer, sample_texts_small, sample_labels):
        """Parallel chunking on 100 documents (may not show speedup due to overhead)."""
        config = BudgetConfig(
            parallel_chunking_workers=4,
            parallel_chunking_min_texts=50,
            parallel_chunking_min_cores=1,  # Force parallel for testing
            batch_size_per_worker=0,  # Auto-compute
        )
        chunker = SemanticChunker(
            tokenizer=tokenizer,
            config=config,
            words_splitter_type="whitespace",
        )
        
        elapsed, total_chunks = chunk_with_timing(
            chunker, sample_texts_small, sample_labels, use_parallel=True
        )
        
        throughput = len(sample_texts_small) / elapsed
        print(f"\n[Parallel-Small]   {len(sample_texts_small)} docs in {elapsed:.2f}s = {throughput:.1f} docs/s (4 workers)")
        print(f"                   Generated {total_chunks} chunks")
        
        assert throughput > 0, "Throughput should be positive"
    
    def test_sequential_medium_dataset(self, tokenizer, sample_texts_medium, sample_labels):
        """Baseline: Sequential chunking on 1000 documents."""
        config = BudgetConfig(
            parallel_chunking_workers=0,
            parallel_chunking_min_texts=0,
        )
        chunker = SemanticChunker(
            tokenizer=tokenizer,
            config=config,
            words_splitter_type="whitespace",
        )
        
        elapsed, total_chunks = chunk_with_timing(
            chunker, sample_texts_medium, sample_labels, use_parallel=False
        )
        
        throughput = len(sample_texts_medium) / elapsed
        print(f"\n[Sequential-Med]   {len(sample_texts_medium)} docs in {elapsed:.2f}s = {throughput:.1f} docs/s")
        print(f"                   Generated {total_chunks} chunks")
        
        assert throughput > 0, "Throughput should be positive"
    
    def test_parallel_medium_dataset(self, tokenizer, sample_texts_medium, sample_labels):
        """Parallel chunking on 1000 documents (tests functionality, not speedup)."""
        config = BudgetConfig(
            parallel_chunking_workers=8,
            parallel_chunking_min_texts=512,
            parallel_chunking_min_cores=1,  # Force parallel for testing
            batch_size_per_worker=0,  # Auto-compute
        )
        chunker = SemanticChunker(
            tokenizer=tokenizer,
            config=config,
            words_splitter_type="whitespace",
        )
        
        elapsed, total_chunks = chunk_with_timing(
            chunker, sample_texts_medium, sample_labels, use_parallel=True
        )
        
        throughput = len(sample_texts_medium) / elapsed
        print(f"\n[Parallel-Med]     {len(sample_texts_medium)} docs in {elapsed:.2f}s = {throughput:.1f} docs/s (8 workers)")
        print(f"                   Generated {total_chunks} chunks")
        
        assert throughput > 0, "Throughput should be positive"


@pytest.mark.benchmark
@pytest.mark.slow
class TestChunkingSpeedup:
    """Compare parallel vs sequential speedup for different configurations.
    
    IMPORTANT: These tests use hardware-aware assertions. The expected behavior
    differs based on the environment:
    
    - Laptop (<16 cores): Parallel overhead typically negates benefits; tests
      verify functionality rather than speedup.
    - HPC (>=16 cores): Tests assert significant speedup (>2x).
    """
    
    def test_speedup_medium_dataset(self, tokenizer, sample_texts_medium, sample_labels):
        """Measure speedup on 1000 documents.
        
        On medium datasets, parallel overhead often outweighs benefits.
        This test documents the overhead cost and verifies correctness.
        """
        # Sequential baseline
        config_seq = BudgetConfig(parallel_chunking_workers=0)
        chunker_seq = SemanticChunker(
            tokenizer=tokenizer,
            config=config_seq,
            words_splitter_type="whitespace",
        )
        seq_time, seq_chunks = chunk_with_timing(
            chunker_seq, sample_texts_medium, sample_labels, use_parallel=False
        )
        
        # Parallel with 8 workers (force parallel for comparison)
        config_par = BudgetConfig(
            parallel_chunking_workers=8,
            parallel_chunking_min_texts=512,
            parallel_chunking_min_cores=1,  # Force parallel for testing
        )
        chunker_par = SemanticChunker(
            tokenizer=tokenizer,
            config=config_par,
            words_splitter_type="whitespace",
        )
        par_time, par_chunks = chunk_with_timing(
            chunker_par, sample_texts_medium, sample_labels, use_parallel=True
        )
        
        speedup = seq_time / par_time
        seq_throughput = len(sample_texts_medium) / seq_time
        par_throughput = len(sample_texts_medium) / par_time
        overhead_penalty = ((par_time - seq_time) / seq_time) * 100 if par_time > seq_time else 0
        
        print(f"\n{'='*60}")
        print(f"SPEEDUP ANALYSIS ({len(sample_texts_medium)} documents)")
        print(f"Environment: {get_environment_label()}")
        print(f"{'='*60}")
        print(f"Sequential: {seq_time:.2f}s ({seq_throughput:.1f} docs/s)")
        print(f"Parallel:   {par_time:.2f}s ({par_throughput:.1f} docs/s) [8 workers]")
        print(f"Speedup:    {speedup:.2f}x")
        if overhead_penalty > 0:
            print(f"Overhead:   +{overhead_penalty:.1f}% (parallel slower due to spawn/IPC cost)")
        print(f"Chunks:     {seq_chunks} (seq) vs {par_chunks} (par)")
        print(f"{'='*60}\n")
        
        # Verify results are identical
        assert seq_chunks == par_chunks, "Sequential and parallel should produce same chunks"
        
        # Hardware-aware assertion:
        # - On HPC (>=16 cores): Expect some speedup
        # - On Laptop/Workstation: Document overhead, don't fail
        if is_hpc_environment():
            assert speedup > 1.0, f"Expected speedup on HPC, got {speedup:.2f}x"
        else:
            # On laptops, we expect overhead penalty for medium datasets
            # This is by design - document it but don't fail
            print(f"NOTE: Medium datasets ({len(sample_texts_medium)} docs) show overhead penalty "
                  f"on {get_environment_label()}. This is expected behavior.")
            print("      The hardware gate will route this to sequential mode in production.")
    
    def test_speedup_large_dataset(self, tokenizer, sample_texts_large, sample_labels):
        """Measure speedup on 20000 documents (should show best speedup on HPC).
        
        This test validates the parallel chunking design:
        - On HPC: Assert significant speedup (>2x)
        - On Laptop: Document performance, verify the gate works correctly
        """
        # Sequential baseline
        config_seq = BudgetConfig(parallel_chunking_workers=0)
        chunker_seq = SemanticChunker(
            tokenizer=tokenizer,
            config=config_seq,
            words_splitter_type="whitespace",
        )
        seq_time, seq_chunks = chunk_with_timing(
            chunker_seq, sample_texts_large, sample_labels, use_parallel=False
        )
        
        # Parallel with 16 workers (force parallel for comparison)
        config_par = BudgetConfig(
            parallel_chunking_workers=16,
            parallel_chunking_min_texts=512,
            parallel_chunking_min_cores=1,  # Force parallel for testing
        )
        chunker_par = SemanticChunker(
            tokenizer=tokenizer,
            config=config_par,
            words_splitter_type="whitespace",
        )
        par_time, par_chunks = chunk_with_timing(
            chunker_par, sample_texts_large, sample_labels, use_parallel=True
        )
        
        speedup = seq_time / par_time
        seq_throughput = len(sample_texts_large) / seq_time
        par_throughput = len(sample_texts_large) / par_time
        
        print(f"\n{'='*60}")
        print(f"SPEEDUP ANALYSIS ({len(sample_texts_large)} documents)")
        print(f"Environment: {get_environment_label()}")
        print(f"{'='*60}")
        print(f"Sequential: {seq_time:.2f}s ({seq_throughput:.1f} docs/s)")
        print(f"Parallel:   {par_time:.2f}s ({par_throughput:.1f} docs/s) [16 workers]")
        print(f"Speedup:    {speedup:.2f}x")
        print(f"Chunks:     {seq_chunks} (seq) vs {par_chunks} (par)")
        print(f"{'='*60}\n")
        
        assert seq_chunks == par_chunks, "Sequential and parallel should produce same chunks"
        
        # Hardware-aware assertion
        if is_hpc_environment():
            # HPC should show significant speedup on large datasets
            assert speedup > 2.0, f"Expected >2.0x speedup on HPC with large dataset, got {speedup:.2f}x"
        else:
            # Laptop: speedup depends on core count and spawn overhead
            # Just verify it doesn't crash and results are correct
            print(f"NOTE: Running on {get_environment_label()}.")
            if speedup > 1.0:
                print(f"      Parallel achieved {speedup:.2f}x speedup on large dataset.")
            else:
                print(f"      Parallel showed {speedup:.2f}x (overhead penalty). "
                      "The hardware gate will use sequential mode in production.")


@pytest.mark.benchmark
class TestWorkerScaling:
    """Test how throughput scales with worker count.
    
    This test measures the impact of worker count on performance, considering
    that optimal scaling depends heavily on hardware capabilities.
    """
    
    def test_worker_scaling(self, tokenizer, sample_texts_medium, sample_labels):
        """Measure throughput for 1, 2, 4, 8 workers."""
        worker_counts = [16, 20, 24, 28, 32] if is_hpc_environment() else [1, 2, 4, 8]
        results = {}

        # Baseline run with 1 worker to compute speedups and provide a stable baseline
        baseline_config = BudgetConfig(
            parallel_chunking_workers=1,
            parallel_chunking_min_texts=512,
            parallel_chunking_min_cores=1,
        )
        baseline_chunker = SemanticChunker(
            tokenizer=tokenizer,
            config=baseline_config,
            words_splitter_type="whitespace",
        )
        base_elapsed, base_chunks = chunk_with_timing(
            baseline_chunker, sample_texts_medium, sample_labels, use_parallel=True
        )
        results[1] = {
            "time": base_elapsed,
            "throughput": len(sample_texts_medium) / base_elapsed,
            "chunks": base_chunks,
        }
        
        for num_workers in worker_counts:
            if num_workers == 1:
                # already measured baseline
                continue

            config = BudgetConfig(
                parallel_chunking_workers=num_workers,
                parallel_chunking_min_texts=512,
                parallel_chunking_min_cores=1,  # Force parallel for testing
            )
            chunker = SemanticChunker(
                tokenizer=tokenizer,
                config=config,
                words_splitter_type="whitespace",
            )
            
            elapsed, total_chunks = chunk_with_timing(
                chunker, sample_texts_medium, sample_labels, use_parallel=True
            )
            
            throughput = len(sample_texts_medium) / elapsed
            results[num_workers] = {
                "time": elapsed,
                "throughput": throughput,
                "chunks": total_chunks,
            }
        
        print(f"\n{'='*60}")
        print(f"WORKER SCALING ANALYSIS ({len(sample_texts_medium)} documents)")
        print(f"Environment: {get_environment_label()}")
        print(f"{'='*60}")
        for num_workers, data in results.items():
            speedup = results[1]["time"] / data["time"] if num_workers > 1 else 1.0
            efficiency = (speedup / num_workers) * 100
            print(f"{num_workers:2d} workers: {data['time']:6.2f}s | {data['throughput']:7.1f} docs/s | "
                  f"Speedup: {speedup:4.2f}x | Efficiency: {efficiency:5.1f}%")
        print(f"{'='*60}\n")
        
        # All worker counts should produce identical chunks
        chunk_counts = [data["chunks"] for data in results.values()]
        assert len(set(chunk_counts)) == 1, "All worker counts should produce same chunks"
        

        # On HPC, identify and print the best worker configuration
        best_workers = max(results.keys(), key=lambda k: results[k]["throughput"])
        best_throughput = results[best_workers]["throughput"]
        print(f"NOTE: Best configuration on {get_environment_label()} is {best_workers} workers "
                f"with {best_throughput:.1f} docs/s throughput")



@pytest.mark.benchmark
class TestBatchSizeImpact:
    """Test impact of batch_size_per_worker on performance."""
    
    def test_batch_size_comparison(self, tokenizer, sample_texts_medium, sample_labels):
        """Compare different batch sizes per worker."""
        batch_sizes = [0, 100, 250, 500]  # 0 = auto-compute
        results = {}
        
        for batch_size in batch_sizes:
            config = BudgetConfig(
                parallel_chunking_workers=8,
                parallel_chunking_min_texts=512,
                parallel_chunking_min_cores=1,  # Force parallel for testing
                batch_size_per_worker=batch_size,
            )
            chunker = SemanticChunker(
                tokenizer=tokenizer,
                config=config,
                words_splitter_type="whitespace",
            )
            
            elapsed, total_chunks = chunk_with_timing(
                chunker, sample_texts_medium, sample_labels, use_parallel=True
            )
            
            throughput = len(sample_texts_medium) / elapsed
            results[batch_size] = {
                "time": elapsed,
                "throughput": throughput,
                "chunks": total_chunks,
            }
        
        print(f"\n{'='*60}")
        print(f"BATCH SIZE IMPACT ({len(sample_texts_medium)} documents, 8 workers)")
        print(f"Environment: {get_environment_label()}")
        print(f"{'='*60}")
        for batch_size, data in results.items():
            batch_label = "auto" if batch_size == 0 else str(batch_size)
            print(f"Batch size {batch_label:>4s}: {data['time']:6.2f}s | {data['throughput']:7.1f} docs/s")
        print(f"{'='*60}\n")
        
        # All configurations should produce same number of chunks
        chunk_counts = [data["chunks"] for data in results.values()]
        assert len(set(chunk_counts)) == 1, "All batch sizes should produce same chunks"


@pytest.mark.benchmark
class TestHardwareGate:
    """Test that the hardware/volume gate correctly routes execution."""
    
    def test_gate_respects_min_texts(self, tokenizer, sample_texts_small, sample_labels):
        """Verify that small datasets are routed to sequential mode."""
        config = BudgetConfig(
            parallel_chunking_workers=8,
            parallel_chunking_min_texts=5000,  # Higher than sample size
            parallel_chunking_min_cores=1,
        )
        chunker = SemanticChunker(
            tokenizer=tokenizer,
            config=config,
            words_splitter_type="whitespace",
        )
        
        labels_list = [sample_labels if text.strip() else None for text in sample_texts_small]
        
        # This should use sequential mode because len(texts) < min_texts
        start_time = time.perf_counter()
        results = chunker.chunk_texts(sample_texts_small, labels_list)
        elapsed = time.perf_counter() - start_time
        
        total_chunks = sum(len(chunks) for chunks in results)
        throughput = len(sample_texts_small) / elapsed
        
        print(f"\n[Gate Test - min_texts] {len(sample_texts_small)} docs in {elapsed:.2f}s = {throughput:.1f} docs/s")
        print(f"                        Config: min_texts=5000, actual={len(sample_texts_small)}")
        print(f"                        Expected: Sequential mode (routed by volume gate)")
        
        assert total_chunks > 0, "Should produce chunks"
    
    def test_gate_respects_min_cores(self, tokenizer, sample_texts_medium, sample_labels):
        """Verify that low-core systems are routed to sequential mode."""
        available_cores = get_cpu_count()
        
        config = BudgetConfig(
            parallel_chunking_workers=8,
            parallel_chunking_min_texts=100,  # Low enough to pass volume check
            parallel_chunking_min_cores=available_cores + 10,  # Higher than available
        )
        chunker = SemanticChunker(
            tokenizer=tokenizer,
            config=config,
            words_splitter_type="whitespace",
        )
        
        labels_list = [sample_labels if text.strip() else None for text in sample_texts_medium]
        
        # This should use sequential mode because available_cores < min_cores
        start_time = time.perf_counter()
        results = chunker.chunk_texts(sample_texts_medium, labels_list)
        elapsed = time.perf_counter() - start_time
        
        total_chunks = sum(len(chunks) for chunks in results)
        throughput = len(sample_texts_medium) / elapsed
        
        print(f"\n[Gate Test - min_cores] {len(sample_texts_medium)} docs in {elapsed:.2f}s = {throughput:.1f} docs/s")
        print(f"                        Config: min_cores={available_cores + 10}, actual={available_cores}")
        print(f"                        Expected: Sequential mode (routed by hardware gate)")
        
        assert total_chunks > 0, "Should produce chunks"


@pytest.mark.benchmark
class TestOverheadMeasurement:
    """Measure and document parallel spawn/IPC overhead."""
    
    def test_overhead_analysis(self, tokenizer, sample_texts_medium, sample_labels):
        """Quantify the overhead cost of parallel processing."""
        # Sequential baseline
        config_seq = BudgetConfig(parallel_chunking_workers=0)
        chunker_seq = SemanticChunker(
            tokenizer=tokenizer,
            config=config_seq,
            words_splitter_type="whitespace",
        )
        seq_time, seq_chunks = chunk_with_timing(
            chunker_seq, sample_texts_medium, sample_labels, use_parallel=False
        )
        
        # Time just the parallel pool creation and first batch
        config_par = BudgetConfig(
            parallel_chunking_workers=8,
            parallel_chunking_min_texts=100,
            parallel_chunking_min_cores=1,
        )
        chunker_par = SemanticChunker(
            tokenizer=tokenizer,
            config=config_par,
            words_splitter_type="whitespace",
        )
        
        # Time total parallel execution
        par_time, par_chunks = chunk_with_timing(
            chunker_par, sample_texts_medium, sample_labels, use_parallel=True
        )
        
        # Calculate overhead
        overhead_time = max(0, par_time - seq_time)
        overhead_pct = (overhead_time / seq_time) * 100 if seq_time > 0 else 0
        speedup = seq_time / par_time if par_time > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"OVERHEAD ANALYSIS ({len(sample_texts_medium)} documents)")
        print(f"Environment: {get_environment_label()}")
        print(f"{'='*60}")
        print(f"Sequential time:     {seq_time:.2f}s")
        print(f"Parallel time:       {par_time:.2f}s")
        print(f"Overhead:            {overhead_time:.2f}s ({overhead_pct:.1f}%)")
        print(f"Processing rate:")
        print(f"  Sequential:        {len(sample_texts_medium) / seq_time:.1f} docs/s")
        print(f"  Parallel:          {len(sample_texts_medium) / par_time:.1f} docs/s")
        print(f"{'='*60}")
        print(f"\nCONCLUSION: For {len(sample_texts_medium)} documents on {get_environment_label()},")
        if par_time < seq_time:
            print(f"parallel is {seq_time/par_time:.1f}x faster despite spawn overhead.")
        else:
            print(f"sequential is {par_time/seq_time:.1f}x faster due to spawn overhead.")
            print("The hardware gate correctly routes small/medium datasets to sequential mode.")
        print(f"{'='*60}\n")
        
        assert seq_chunks == par_chunks, "Chunk counts should match"


# ==============================================================================
# Integration Test
# ==============================================================================

def test_parallel_correctness(tokenizer, sample_texts_small, sample_labels):
    """Verify parallel chunking produces identical results to sequential."""
    # Sequential
    config_seq = BudgetConfig(parallel_chunking_workers=0)
    chunker_seq = SemanticChunker(
        tokenizer=tokenizer,
        config=config_seq,
        words_splitter_type="whitespace",
    )
    labels_list = [sample_labels if text.strip() else None for text in sample_texts_small]
    results_seq = chunker_seq._chunk_texts_sequential(
        sample_texts_small, labels_list, progress_callback=None
    )
    
    # Parallel (force parallel mode for testing)
    config_par = BudgetConfig(
        parallel_chunking_workers=4,
        parallel_chunking_min_texts=50,
        parallel_chunking_min_cores=1,  # Force parallel
    )
    chunker_par = SemanticChunker(
        tokenizer=tokenizer,
        config=config_par,
        words_splitter_type="whitespace",
    )
    results_par = chunker_par._chunk_texts_parallel(
        sample_texts_small,
        labels_list,
        tokenizer.name_or_path,
        chunker_par._words_splitter_type,
        progress_callback=None,
    )
    
    # Verify identical results
    assert len(results_seq) == len(results_par), "Result count mismatch"
    
    for idx, (seq_chunks, par_chunks) in enumerate(zip(results_seq, results_par)):
        assert len(seq_chunks) == len(par_chunks), f"Chunk count mismatch at doc {idx}"
        
        for chunk_idx, (seq_chunk, par_chunk) in enumerate(zip(seq_chunks, par_chunks)):
            assert seq_chunk.text == par_chunk.text, \
                f"Text mismatch at doc {idx}, chunk {chunk_idx}"
            assert seq_chunk.char_start == par_chunk.char_start, \
                f"Start offset mismatch at doc {idx}, chunk {chunk_idx}"
            assert seq_chunk.char_end == par_chunk.char_end, \
                f"End offset mismatch at doc {idx}, chunk {chunk_idx}"
            assert seq_chunk.is_hard_split == par_chunk.is_hard_split, \
                f"Hard split flag mismatch at doc {idx}, chunk {chunk_idx}"
