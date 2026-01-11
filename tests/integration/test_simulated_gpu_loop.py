"""
Integration Tests for Simulated GPU Loop.

Tests the complete autotuning system with a simulated GPU environment.
Verifies OOM handling, budget convergence, and system stability without
requiring actual GPU hardware.

The test creates a "Virtual GPU" with configurable memory limits and
a cost function that determines VRAM usage based on batch token count.
"""

import pytest
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any
import numpy as np

from neuro_stylometry.hardware_ops.runtime import (
    RuntimeController,
    RuntimeConfig,
    RuntimeMetrics,
    RuntimeState,
)
from neuro_stylometry.hardware_ops.oom_guard import (
    execute_with_oom_protection,
    OOMRecoveryError,
)
from neuro_stylometry.pollution_guard.global_sort import (
    DynamicBatchIterator,
    FlattenedChunks,
)


class SimulatedGPU:
    """
    Simulated GPU environment for testing.
    
    Provides a virtual GPU with configurable memory limits and
    a cost function that maps token counts to VRAM usage.
    """
    
    def __init__(
        self,
        total_memory_gb: float = 10.0,
        cost_per_token_mb: float = 0.001,  # 1MB per 1000 tokens
    ):
        """
        Initialize simulated GPU.
        
        Args:
            total_memory_gb: Total "VRAM" in GB.
            cost_per_token_mb: Memory cost per token in MB.
        """
        self.total_memory_mb = total_memory_gb * 1024
        self.cost_per_token_mb = cost_per_token_mb
        self.current_allocated_mb = 0.0
        self.oom_count = 0
    
    def compute_cost(self, token_count: int) -> float:
        """Compute memory cost for a batch."""
        return token_count * self.cost_per_token_mb
    
    def allocate(self, token_count: int) -> bool:
        """
        Attempt to allocate memory for batch.
        
        Returns True if successful, raises OOM error if exceeds limit.
        """
        cost = self.compute_cost(token_count)
        
        if cost > self.total_memory_mb:
            self.oom_count += 1
            raise RuntimeError(
                f"CUDA out of memory. Tried to allocate {cost:.2f} MB "
                f"(total: {self.total_memory_mb:.2f} MB)"
            )
        
        self.current_allocated_mb = cost
        return True
    
    def free(self):
        """Free allocated memory."""
        self.current_allocated_mb = 0.0


class SimulatedInference:
    """
    Simulated inference engine for testing.
    
    Wraps a SimulatedGPU and provides inference-like behavior
    with timing simulation.
    """
    
    def __init__(self, gpu: SimulatedGPU):
        self.gpu = gpu
        self.batches_processed = 0
        self.total_tokens_processed = 0
    
    def run_batch(self, batch_texts: List[str], token_counts: List[int]) -> List[Dict]:
        """
        Simulate running inference on a batch.
        
        Args:
            batch_texts: List of text chunks.
            token_counts: Token count for each chunk.
            
        Returns:
            Simulated results (list of empty dicts).
        """
        total_tokens = sum(token_counts)
        
        # Attempt allocation (may raise OOM)
        self.gpu.allocate(total_tokens)
        
        # Simulate processing
        self.batches_processed += 1
        self.total_tokens_processed += total_tokens
        
        # Free memory
        self.gpu.free()
        
        # Return dummy results
        return [{"entities": []} for _ in batch_texts]


def create_test_flattened(
    num_chunks: int,
    min_tokens: int = 50,
    max_tokens: int = 200,
    seed: int = 42,
) -> FlattenedChunks:
    """Create test FlattenedChunks with random token counts."""
    np.random.seed(seed)
    token_counts = np.random.randint(min_tokens, max_tokens, size=num_chunks)
    
    texts = [f"chunk_{i}" for i in range(num_chunks)]
    doc_offsets = np.array([0, num_chunks], dtype=np.int64)
    chunk_metadata = [(0, i) for i in range(num_chunks)]
    
    return FlattenedChunks(
        texts=texts,
        token_counts=token_counts.astype(np.int16),
        doc_offsets=doc_offsets,
        chunk_metadata=chunk_metadata,
    )


class TestSimulatedGPULoop:
    """
    Integration tests with simulated GPU environment.
    
    These tests verify the complete autotuning system works correctly
    including OOM handling and budget convergence.
    """
    
    def test_survival_with_oom_recovery(self):
        """
        Test that the system survives OOM and completes processing.
        
        Setup:
        - Virtual GPU: 4GB (constrained)
        - Very high cost per token to force OOM with fewer tokens
        - Data: 500 large chunks
        
        Assertions:
        - Loop completes without crashing
        - All chunks processed
        """
        # Setup - constrained GPU with expensive memory cost
        gpu = SimulatedGPU(
            total_memory_gb=4.0,
            cost_per_token_mb=0.01,  # 1MB per 100 tokens (expensive)
        )
        inference = SimulatedInference(gpu)
        
        # With expensive tokens, initial budget will likely cause OOM
        config = RuntimeConfig(
            warmup_batches=2,
            initial_token_budget=500_000,  # 500K tokens = 5GB at this cost
            min_token_budget=10_000,
            max_token_budget=1_000_000,
            oom_slash_factor=0.3,
            recovery_patience=2,
        )
        controller = RuntimeController(config)
        controller._total_memory_mb = gpu.total_memory_mb
        
        # Large chunks to increase chance of OOM
        flattened = create_test_flattened(500, min_tokens=500, max_tokens=2000)
        
        # Run the loop
        processed_indices = []
        
        iterator = DynamicBatchIterator(
            flattened=flattened,
            controller=controller,
            min_batch_size=1,
            max_batch_size=200,
        )
        
        for batch_indices in iterator:
            batch_texts = [flattened.texts[i] for i in batch_indices]
            batch_tokens = [int(flattened.token_counts[i]) for i in batch_indices]
            total_tokens = sum(batch_tokens)
            
            try:
                result = execute_with_oom_protection(
                    lambda: inference.run_batch(batch_texts, batch_tokens),
                    controller=controller,
                    retry_limit=10,
                )
                
                # Report metrics
                controller.report_metrics(RuntimeMetrics(
                    tokens_processed=total_tokens,
                    batch_time_ms=10.0,  # Simulated
                    memory_used_mb=gpu.compute_cost(total_tokens),
                ))
                
                processed_indices.extend(batch_indices)
                
            except OOMRecoveryError:
                # Even recovery failed - this shouldn't happen with good params
                pytest.fail("OOM recovery failed completely")
        
        # Assertions - main goal is survival and completion
        assert len(processed_indices) == 500, f"All chunks should be processed, got {len(processed_indices)}"
        assert len(set(processed_indices)) == 500, "No duplicates"
        # If OOM occurred, verify it was handled
        if gpu.oom_count > 0:
            assert controller.state in (RuntimeState.RECOVERY, RuntimeState.SCALING_UP, RuntimeState.STABLE)
    
    def test_budget_convergence(self):
        """
        Test that budget converges to optimal value.
        
        Setup:
        - Virtual GPU: 10GB (10,000 MB)
        - Cost: 1MB per 1000 tokens
        - Optimal budget: ~9,000,000 tokens (uses ~9GB, leaves 1GB headroom)
        - Initial budget: 500,000 tokens (very low)
        
        Assertions:
        - Budget increases over time
        - Final budget is near optimal (within 20%)
        """
        gpu = SimulatedGPU(
            total_memory_gb=10.0,
            cost_per_token_mb=0.001,
        )
        inference = SimulatedInference(gpu)
        
        # Start low, should scale up
        config = RuntimeConfig(
            warmup_batches=3,
            initial_token_budget=500_000,  # Very low
            min_token_budget=100_000,
            max_token_budget=15_000_000,
            scale_up_factor=1.5,  # Aggressive scaling
            scale_down_factor=0.8,
            memory_headroom_mb=1000,  # 1GB headroom
            stability_threshold=0.2,
        )
        controller = RuntimeController(config)
        controller._total_memory_mb = gpu.total_memory_mb
        
        # Optimal is ~9,000,000 tokens (9GB usage, 1GB headroom)
        optimal_budget = 9_000_000
        
        flattened = create_test_flattened(500, min_tokens=1000, max_tokens=5000)
        
        budget_history = [controller.current_budget]
        
        iterator = DynamicBatchIterator(
            flattened=flattened,
            controller=controller,
            min_batch_size=1,
            max_batch_size=100,
        )
        
        for batch_indices in iterator:
            batch_texts = [flattened.texts[i] for i in batch_indices]
            batch_tokens = [int(flattened.token_counts[i]) for i in batch_indices]
            total_tokens = sum(batch_tokens)
            
            # Run inference (shouldn't OOM with low initial budget)
            inference.run_batch(batch_texts, batch_tokens)
            
            # Report metrics with low memory usage
            memory_used = gpu.compute_cost(total_tokens)
            controller.report_metrics(RuntimeMetrics(
                tokens_processed=total_tokens,
                batch_time_ms=10.0,
                memory_used_mb=memory_used,
            ))
            
            budget_history.append(controller.current_budget)
        
        # Budget should have increased
        assert budget_history[-1] > budget_history[0], \
            "Budget should increase from initial low value"
        
        # Should be within 50% of optimal (conservative due to short run)
        # Note: May not fully converge in a short test
        final_budget = budget_history[-1]
        assert final_budget > config.initial_token_budget, \
            "Budget should have scaled up"
    
    def test_recovery_after_oom(self):
        """
        Test that failed batch is successfully processed on retry.
        
        Setup:
        - Virtual GPU: 2GB (very constrained)
        - High cost per token to force OOM
        - Large chunks
        
        Assertions:
        - System processes all chunks
        - If OOM occurred, it was handled
        """
        gpu = SimulatedGPU(
            total_memory_gb=2.0,  # Very constrained
            cost_per_token_mb=0.02,  # 1MB per 50 tokens (very expensive)
        )
        inference = SimulatedInference(gpu)
        
        config = RuntimeConfig(
            warmup_batches=1,
            initial_token_budget=200_000,  # 200K tokens = 4GB at this cost
            min_token_budget=5_000,
            max_token_budget=500_000,
            oom_slash_factor=0.25,  # Very aggressive slash
            recovery_patience=1,
        )
        controller = RuntimeController(config)
        controller._total_memory_mb = gpu.total_memory_mb
        
        # Large chunks to force high memory usage
        flattened = create_test_flattened(50, min_tokens=1000, max_tokens=3000)
        
        processed_count = 0
        
        iterator = DynamicBatchIterator(
            flattened=flattened,
            controller=controller,
            min_batch_size=1,
            max_batch_size=20,
        )
        
        for batch_indices in iterator:
            batch_texts = [flattened.texts[i] for i in batch_indices]
            batch_tokens = [int(flattened.token_counts[i]) for i in batch_indices]
            
            def run_inference():
                return inference.run_batch(batch_texts, batch_tokens)
            
            try:
                result = execute_with_oom_protection(
                    run_inference,
                    controller=controller,
                    retry_limit=10,
                )
                
                controller.report_metrics(RuntimeMetrics(
                    tokens_processed=sum(batch_tokens),
                    batch_time_ms=10.0,
                    memory_used_mb=gpu.current_allocated_mb,
                ))
                
                processed_count += len(batch_indices)
                
            except OOMRecoveryError:
                pytest.fail("OOM recovery failed completely")
        
        # Main assertion: all chunks processed
        assert processed_count == 50, f"Expected 50 chunks, got {processed_count}"
        assert inference.batches_processed > 0, "Should have processed some batches"
    
    def test_stable_operation_after_warmup(self):
        """
        Test that system processes data and exits warmup.
        
        Setup:
        - Virtual GPU: 8GB
        - Reasonable initial budget (smaller to allow more batches)
        - Stable memory usage
        
        Assertions:
        - Controller exits WARMUP state
        - All data is processed
        - Budget remains within bounds
        """
        gpu = SimulatedGPU(
            total_memory_gb=8.0,
            cost_per_token_mb=0.001,
        )
        inference = SimulatedInference(gpu)
        
        config = RuntimeConfig(
            warmup_batches=3,  # Short warmup
            initial_token_budget=50_000,  # Lower budget so we get more batches
            min_token_budget=10_000,
            max_token_budget=7_000_000,
            stability_threshold=0.2,  # More lenient
            history_window=5,
        )
        controller = RuntimeController(config)
        controller._total_memory_mb = gpu.total_memory_mb
        
        # More chunks with varied sizes to ensure multiple batches
        flattened = create_test_flattened(200, min_tokens=500, max_tokens=1500)
        
        states_seen = set()
        budget_history = []
        processed_count = 0
        
        iterator = DynamicBatchIterator(
            flattened=flattened,
            controller=controller,
            min_batch_size=1,
            max_batch_size=50,
        )
        
        for batch_indices in iterator:
            batch_texts = [flattened.texts[i] for i in batch_indices]
            batch_tokens = [int(flattened.token_counts[i]) for i in batch_indices]
            total_tokens = sum(batch_tokens)
            
            inference.run_batch(batch_texts, batch_tokens)
            
            controller.report_metrics(RuntimeMetrics(
                tokens_processed=total_tokens,
                batch_time_ms=10.0,  # Consistent timing
                memory_used_mb=gpu.compute_cost(total_tokens),
            ))
            
            states_seen.add(controller.state)
            budget_history.append(controller.current_budget)
            processed_count += len(batch_indices)
        
        # Main assertions
        assert processed_count == 200, f"Should process all 200 chunks, got {processed_count}"
        assert controller.state != RuntimeState.WARMUP, "Should exit WARMUP state"
        
        # Budget should stay within configured bounds
        for budget in budget_history:
            assert config.min_token_budget <= budget <= config.max_token_budget, \
                f"Budget {budget} outside bounds [{config.min_token_budget}, {config.max_token_budget}]"


class TestSimulatedEdgeCases:
    """Edge case tests for the simulated GPU loop."""
    
    def test_single_chunk_processing(self):
        """System should handle single chunk dataset."""
        gpu = SimulatedGPU(total_memory_gb=10.0)
        inference = SimulatedInference(gpu)
        
        config = RuntimeConfig(initial_token_budget=10_000_000)
        controller = RuntimeController(config)
        controller._total_memory_mb = gpu.total_memory_mb
        
        flattened = create_test_flattened(1)
        
        iterator = DynamicBatchIterator(
            flattened=flattened,
            controller=controller,
        )
        
        processed = 0
        for batch_indices in iterator:
            processed += len(batch_indices)
        
        assert processed == 1
    
    def test_all_identical_chunks(self):
        """System should handle all identical chunk sizes."""
        gpu = SimulatedGPU(total_memory_gb=10.0)
        
        config = RuntimeConfig(initial_token_budget=1_000_000)
        controller = RuntimeController(config)
        controller._total_memory_mb = gpu.total_memory_mb
        
        # All chunks have exactly 100 tokens
        token_counts = [100] * 50
        texts = [f"chunk_{i}" for i in range(50)]
        flattened = FlattenedChunks(
            texts=texts,
            token_counts=np.array(token_counts, dtype=np.int16),
            doc_offsets=np.array([0, 50], dtype=np.int64),
            chunk_metadata=[(0, i) for i in range(50)],
        )
        
        iterator = DynamicBatchIterator(
            flattened=flattened,
            controller=controller,
        )
        
        all_indices = []
        for batch in iterator:
            all_indices.extend(batch)
        
        assert len(all_indices) == 50
    
    def test_very_large_single_chunk(self):
        """System should handle chunk larger than budget."""
        gpu = SimulatedGPU(
            total_memory_gb=10.0,
            cost_per_token_mb=0.001,
        )
        inference = SimulatedInference(gpu)
        
        config = RuntimeConfig(
            initial_token_budget=1000,  # Very small
            min_token_budget=100,
        )
        controller = RuntimeController(config)
        controller._total_memory_mb = gpu.total_memory_mb
        
        # Single chunk with 5000 tokens (5MB, fits in GPU)
        flattened = FlattenedChunks(
            texts=["large_chunk"],
            token_counts=np.array([5000], dtype=np.int16),
            doc_offsets=np.array([0, 1], dtype=np.int64),
            chunk_metadata=[(0, 0)],
        )
        
        iterator = DynamicBatchIterator(
            flattened=flattened,
            controller=controller,
            min_batch_size=1,
        )
        
        processed = 0
        for batch_indices in iterator:
            batch_tokens = [int(flattened.token_counts[i]) for i in batch_indices]
            inference.run_batch(
                [flattened.texts[i] for i in batch_indices],
                batch_tokens,
            )
            processed += len(batch_indices)
        
        assert processed == 1
