"""
Benchmark tests for parallel vs sequential chunking performance.

Tests measure:
- Throughput (documents/second) for different dataset sizes
- Speedup factor (parallel vs sequential)
- Scaling efficiency with worker count
- Memory overhead of parallel processing

Usage:
    pytest tests/benchmarks/test_parallel_chunking.py -v
    pytest tests/benchmarks/test_parallel_chunking.py --benchmark-only
    pytest tests/benchmarks/test_parallel_chunking.py --benchmark-json=chunking_results.json
"""

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
    """Generate 5000 sample documents."""
    texts = []
    for i in range(5000):
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
        chunker = SemanticChunker(tokenizer=tokenizer, config=config)
        
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
            batch_size_per_worker=0,  # Auto-compute
        )
        chunker = SemanticChunker(tokenizer=tokenizer, config=config)
        
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
        chunker = SemanticChunker(tokenizer=tokenizer, config=config)
        
        elapsed, total_chunks = chunk_with_timing(
            chunker, sample_texts_medium, sample_labels, use_parallel=False
        )
        
        throughput = len(sample_texts_medium) / elapsed
        print(f"\n[Sequential-Med]   {len(sample_texts_medium)} docs in {elapsed:.2f}s = {throughput:.1f} docs/s")
        print(f"                   Generated {total_chunks} chunks")
        
        assert throughput > 0, "Throughput should be positive"
    
    def test_parallel_medium_dataset(self, tokenizer, sample_texts_medium, sample_labels):
        """Parallel chunking on 1000 documents (should show 2-4x speedup)."""
        config = BudgetConfig(
            parallel_chunking_workers=8,
            parallel_chunking_min_texts=512,
            batch_size_per_worker=0,  # Auto-compute
        )
        chunker = SemanticChunker(tokenizer=tokenizer, config=config)
        
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
    """Compare parallel vs sequential speedup for different configurations."""
    
    def test_speedup_medium_dataset(self, tokenizer, sample_texts_medium, sample_labels):
        """Measure speedup on 1000 documents."""
        # Sequential baseline
        config_seq = BudgetConfig(parallel_chunking_workers=0)
        chunker_seq = SemanticChunker(tokenizer=tokenizer, config=config_seq)
        seq_time, seq_chunks = chunk_with_timing(
            chunker_seq, sample_texts_medium, sample_labels, use_parallel=False
        )
        
        # Parallel with 8 workers
        config_par = BudgetConfig(
            parallel_chunking_workers=8,
            parallel_chunking_min_texts=512,
        )
        chunker_par = SemanticChunker(tokenizer=tokenizer, config=config_par)
        par_time, par_chunks = chunk_with_timing(
            chunker_par, sample_texts_medium, sample_labels, use_parallel=True
        )
        
        speedup = seq_time / par_time
        seq_throughput = len(sample_texts_medium) / seq_time
        par_throughput = len(sample_texts_medium) / par_time
        
        print(f"\n{'='*60}")
        print(f"SPEEDUP ANALYSIS ({len(sample_texts_medium)} documents)")
        print(f"{'='*60}")
        print(f"Sequential: {seq_time:.2f}s ({seq_throughput:.1f} docs/s)")
        print(f"Parallel:   {par_time:.2f}s ({par_throughput:.1f} docs/s) [8 workers]")
        print(f"Speedup:    {speedup:.2f}x")
        print(f"Chunks:     {seq_chunks} (seq) vs {par_chunks} (par)")
        print(f"{'='*60}\n")
        
        # Verify results are identical
        assert seq_chunks == par_chunks, "Sequential and parallel should produce same chunks"
        
        # Speedup should be > 1.5x for medium datasets (conservative threshold)
        assert speedup > 1.5, f"Expected >1.5x speedup, got {speedup:.2f}x"
    
    def test_speedup_large_dataset(self, tokenizer, sample_texts_large, sample_labels):
        """Measure speedup on 5000 documents (should show best speedup)."""
        # Sequential baseline
        config_seq = BudgetConfig(parallel_chunking_workers=0)
        chunker_seq = SemanticChunker(tokenizer=tokenizer, config=config_seq)
        seq_time, seq_chunks = chunk_with_timing(
            chunker_seq, sample_texts_large, sample_labels, use_parallel=False
        )
        
        # Parallel with 16 workers
        config_par = BudgetConfig(
            parallel_chunking_workers=16,
            parallel_chunking_min_texts=512,
        )
        chunker_par = SemanticChunker(tokenizer=tokenizer, config=config_par)
        par_time, par_chunks = chunk_with_timing(
            chunker_par, sample_texts_large, sample_labels, use_parallel=True
        )
        
        speedup = seq_time / par_time
        seq_throughput = len(sample_texts_large) / seq_time
        par_throughput = len(sample_texts_large) / par_time
        
        print(f"\n{'='*60}")
        print(f"SPEEDUP ANALYSIS ({len(sample_texts_large)} documents)")
        print(f"{'='*60}")
        print(f"Sequential: {seq_time:.2f}s ({seq_throughput:.1f} docs/s)")
        print(f"Parallel:   {par_time:.2f}s ({par_throughput:.1f} docs/s) [16 workers]")
        print(f"Speedup:    {speedup:.2f}x")
        print(f"Chunks:     {seq_chunks} (seq) vs {par_chunks} (par)")
        print(f"{'='*60}\n")
        
        assert seq_chunks == par_chunks, "Sequential and parallel should produce same chunks"
        
        # Large datasets should show >2x speedup
        assert speedup > 2.0, f"Expected >2.0x speedup on large dataset, got {speedup:.2f}x"


@pytest.mark.benchmark
class TestWorkerScaling:
    """Test how throughput scales with worker count."""
    
    def test_worker_scaling(self, tokenizer, sample_texts_medium, sample_labels):
        """Measure throughput for 1, 2, 4, 8 workers."""
        worker_counts = [1, 2, 4, 8]
        results = {}
        
        for num_workers in worker_counts:
            config = BudgetConfig(
                parallel_chunking_workers=num_workers,
                parallel_chunking_min_texts=512,
            )
            chunker = SemanticChunker(tokenizer=tokenizer, config=config)
            
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
        print(f"{'='*60}")
        for num_workers, data in results.items():
            speedup = results[1]["time"] / data["time"] if num_workers > 1 else 1.0
            efficiency = (speedup / num_workers) * 100
            print(f"{num_workers:2d} workers: {data['time']:6.2f}s | {data['throughput']:7.1f} docs/s | "
                  f"Speedup: {speedup:4.2f}x | Efficiency: {efficiency:5.1f}%")
        print(f"{'='*60}\n")
        
        # Throughput should increase with worker count
        assert results[8]["throughput"] > results[1]["throughput"], \
            "8 workers should have higher throughput than 1 worker"


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
                batch_size_per_worker=batch_size,
            )
            chunker = SemanticChunker(tokenizer=tokenizer, config=config)
            
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
        print(f"{'='*60}")
        for batch_size, data in results.items():
            batch_label = "auto" if batch_size == 0 else str(batch_size)
            print(f"Batch size {batch_label:>4s}: {data['time']:6.2f}s | {data['throughput']:7.1f} docs/s")
        print(f"{'='*60}\n")
        
        # All configurations should produce same number of chunks
        chunk_counts = [data["chunks"] for data in results.values()]
        assert len(set(chunk_counts)) == 1, "All batch sizes should produce same chunks"


# ==============================================================================
# Integration Test
# ==============================================================================

def test_parallel_correctness(tokenizer, sample_texts_small, sample_labels):
    """Verify parallel chunking produces identical results to sequential."""
    # Sequential
    config_seq = BudgetConfig(parallel_chunking_workers=0)
    chunker_seq = SemanticChunker(tokenizer=tokenizer, config=config_seq)
    labels_list = [sample_labels if text.strip() else None for text in sample_texts_small]
    results_seq = chunker_seq._chunk_texts_sequential(
        sample_texts_small, labels_list, progress_callback=None
    )
    
    # Parallel
    config_par = BudgetConfig(parallel_chunking_workers=4, parallel_chunking_min_texts=50)
    chunker_par = SemanticChunker(tokenizer=tokenizer, config=config_par)
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
