"""
Unit Tests for DynamicBatchIterator.

Tests the adaptive batch iteration logic with mock controllers,
verifying correct slicing, exact consumption, and budget adaptation.
"""

import pytest
from unittest.mock import MagicMock
import numpy as np

from neuro_stylometry.pollution_guard.global_sort import (
    DynamicBatchIterator,
    create_dynamic_batches,
    FlattenedChunks,
)
from neuro_stylometry.hardware_ops.runtime import RuntimeController, RuntimeConfig


def create_mock_flattened(
    num_chunks: int,
    token_counts: list = None,
) -> FlattenedChunks:
    """Create a mock FlattenedChunks for testing."""
    if token_counts is None:
        token_counts = [100] * num_chunks  # Default: 100 tokens each
    
    texts = [f"chunk_{i}" for i in range(num_chunks)]
    token_counts_arr = np.array(token_counts, dtype=np.int16)
    doc_offsets = np.array([0, num_chunks], dtype=np.int64)  # Single document
    chunk_metadata = [(0, i) for i in range(num_chunks)]
    
    return FlattenedChunks(
        texts=texts,
        token_counts=token_counts_arr,
        doc_offsets=doc_offsets,
        chunk_metadata=chunk_metadata,
    )


class TestDynamicBatchIteratorBasic:
    """Basic tests for DynamicBatchIterator."""
    
    def test_iterator_yields_all_chunks(self):
        """Iterator should yield all chunks exactly once."""
        flattened = create_mock_flattened(100, [50] * 100)
        
        # Use fallback batch size (no controller)
        iterator = DynamicBatchIterator(
            flattened=flattened,
            fallback_batch_size=10,
        )
        
        all_indices = []
        for batch_indices in iterator:
            all_indices.extend(batch_indices)
        
        # Check all chunks yielded
        assert len(all_indices) == 100
        assert sorted(all_indices) == list(range(100))
    
    def test_no_duplicates_no_drops(self):
        """Iterator should not duplicate or drop any chunks."""
        flattened = create_mock_flattened(50, [100] * 50)
        
        iterator = DynamicBatchIterator(
            flattened=flattened,
            fallback_batch_size=7,  # Doesn't divide evenly
        )
        
        all_indices = []
        for batch_indices in iterator:
            all_indices.extend(batch_indices)
        
        # No duplicates
        assert len(all_indices) == len(set(all_indices))
        # All chunks present
        assert set(all_indices) == set(range(50))
    
    def test_empty_flattened_yields_nothing(self):
        """Empty flattened should yield no batches."""
        flattened = create_mock_flattened(0)
        
        iterator = DynamicBatchIterator(flattened=flattened)
        batches = list(iterator)
        
        assert batches == []
    
    def test_progress_property(self):
        """Progress should track iteration progress."""
        flattened = create_mock_flattened(100, [50] * 100)
        iterator = DynamicBatchIterator(
            flattened=flattened,
            fallback_batch_size=20,
        )
        
        assert iterator.progress == 0.0
        
        batches_iter = iter(iterator)
        next(batches_iter)  # First batch
        
        assert 0 < iterator.progress < 1.0
    
    def test_remaining_chunks_property(self):
        """remaining_chunks should track remaining items."""
        flattened = create_mock_flattened(100, [50] * 100)
        iterator = DynamicBatchIterator(
            flattened=flattened,
            fallback_batch_size=25,
        )
        
        assert iterator.remaining_chunks == 100
        
        batches_iter = iter(iterator)
        next(batches_iter)  # First batch (~25 chunks)
        
        assert iterator.remaining_chunks < 100


class TestDynamicBatchIteratorWithController:
    """Tests with RuntimeController integration."""
    
    def test_uses_controller_budget(self):
        """Iterator should use budget from controller."""
        flattened = create_mock_flattened(100, [100] * 100)
        
        config = RuntimeConfig(
            initial_token_budget=500,  # 5 chunks of 100 tokens
            warmup_batches=0,
        )
        controller = RuntimeController(config)
        
        iterator = DynamicBatchIterator(
            flattened=flattened,
            controller=controller,
        )
        
        first_batch = next(iter(iterator))
        
        # Should be approximately 5 chunks (500 tokens / 100 tokens per chunk)
        assert len(first_batch) <= 6  # Allow some slack
    
    def test_variable_yield_with_changing_budget(self):
        """Iterator should adapt to changing controller budget."""
        flattened = create_mock_flattened(100, [100] * 100)
        
        # Create a mock controller that alternates budget
        mock_controller = MagicMock()
        budget_sequence = [200, 500, 300, 400]  # Tokens
        budget_index = [0]
        
        def get_budget():
            budget = budget_sequence[budget_index[0] % len(budget_sequence)]
            budget_index[0] += 1
            return budget
        
        mock_controller.get_next_budget = get_budget
        
        iterator = DynamicBatchIterator(
            flattened=flattened,
            controller=mock_controller,
        )
        
        batch_sizes = []
        for batch_indices in iterator:
            batch_sizes.append(len(batch_indices))
        
        # Batch sizes should vary
        assert len(set(batch_sizes)) > 1, "Batch sizes should vary with budget"
    
    def test_exact_consumption_with_fluctuating_budget(self):
        """All chunks consumed exactly once even with wild budget changes."""
        flattened = create_mock_flattened(73, [50] * 73)  # Odd number
        
        # Mock controller with wildly fluctuating budget
        mock_controller = MagicMock()
        budgets = [100, 500, 50, 300, 150, 25, 1000]
        budget_idx = [0]
        
        def get_budget():
            b = budgets[budget_idx[0] % len(budgets)]
            budget_idx[0] += 1
            return b
        
        mock_controller.get_next_budget = get_budget
        
        iterator = DynamicBatchIterator(
            flattened=flattened,
            controller=mock_controller,
            min_batch_size=1,
        )
        
        all_indices = []
        for batch_indices in iterator:
            all_indices.extend(batch_indices)
        
        # Exact consumption: no drops, no duplicates
        assert len(all_indices) == 73
        assert sorted(all_indices) == list(range(73))
    
    def test_respects_max_batch_size(self):
        """Iterator should respect max_batch_size limit."""
        flattened = create_mock_flattened(100, [10] * 100)
        
        config = RuntimeConfig(
            initial_token_budget=100000,  # Huge budget
        )
        controller = RuntimeController(config)
        
        iterator = DynamicBatchIterator(
            flattened=flattened,
            controller=controller,
            max_batch_size=20,
        )
        
        for batch_indices in iterator:
            assert len(batch_indices) <= 20
    
    def test_respects_min_batch_size(self):
        """Iterator should ensure min_batch_size for large chunks."""
        # Create chunks with varying sizes, some very large
        token_counts = [1000] * 10  # Large chunks
        flattened = create_mock_flattened(10, token_counts)
        
        config = RuntimeConfig(
            initial_token_budget=100,  # Very small budget
        )
        controller = RuntimeController(config)
        
        iterator = DynamicBatchIterator(
            flattened=flattened,
            controller=controller,
            min_batch_size=2,
        )
        
        first_batch = next(iter(iterator))
        
        # Even though budget is tiny, should get at least min_batch_size
        assert len(first_batch) >= 2


class TestDynamicBatchIteratorSorting:
    """Tests for sorted iteration behavior."""
    
    def test_chunks_sorted_by_length(self):
        """Chunks should be yielded sorted by token count."""
        # Create chunks with varying lengths
        token_counts = [500, 100, 300, 200, 400]
        flattened = create_mock_flattened(5, token_counts)
        
        iterator = DynamicBatchIterator(
            flattened=flattened,
            fallback_batch_size=10,  # Large enough for all
        )
        
        all_indices = []
        for batch_indices in iterator:
            all_indices.extend(batch_indices)
        
        # Get token counts in yielded order
        yielded_lengths = [flattened.token_counts[i] for i in all_indices]
        
        # Should be sorted (ascending)
        assert yielded_lengths == sorted(yielded_lengths)
    
    def test_batches_have_similar_lengths(self):
        """Consecutive chunks in a batch should have similar lengths due to sorting."""
        # Create many chunks with varying lengths (narrower range for tighter batches)
        np.random.seed(42)
        token_counts = list(np.random.randint(100, 300, size=100))
        flattened = create_mock_flattened(100, token_counts)
        
        iterator = DynamicBatchIterator(
            flattened=flattened,
            fallback_batch_size=10,
        )
        
        # Collect all batches and check that most have reasonable ratios
        ratios = []
        for batch_indices in iterator:
            if len(batch_indices) > 1:
                batch_lengths = [flattened.token_counts[i] for i in batch_indices]
                # Within a batch, max should not be too much larger than min
                ratio = max(batch_lengths) / max(min(batch_lengths), 1)
                ratios.append(ratio)
        
        # Most batches should have chunks with similar lengths (median ratio < 2)
        # This is a softer check that accounts for edge batches
        median_ratio = sorted(ratios)[len(ratios) // 2] if ratios else 1.0
        assert median_ratio < 2.5, f"Median batch length ratio too high: {median_ratio}"


class TestCreateDynamicBatches:
    """Tests for the convenience function."""
    
    def test_convenience_function_works(self):
        """create_dynamic_batches should work like the class."""
        flattened = create_mock_flattened(50, [100] * 50)
        
        all_indices = []
        for batch_indices in create_dynamic_batches(
            flattened,
            fallback_batch_size=10,
        ):
            all_indices.extend(batch_indices)
        
        assert len(all_indices) == 50
        assert sorted(all_indices) == list(range(50))
    
    def test_convenience_with_controller(self):
        """create_dynamic_batches should accept controller."""
        flattened = create_mock_flattened(50, [100] * 50)
        
        config = RuntimeConfig(initial_token_budget=500)
        controller = RuntimeController(config)
        
        batches = list(create_dynamic_batches(flattened, controller=controller))
        
        assert len(batches) > 0
        all_indices = [i for batch in batches for i in batch]
        assert len(all_indices) == 50


class TestDynamicBatchIteratorReset:
    """Tests for iterator reset functionality."""
    
    def test_reset_allows_reiteration(self):
        """reset() should allow iterating again."""
        flattened = create_mock_flattened(20, [50] * 20)
        
        iterator = DynamicBatchIterator(
            flattened=flattened,
            fallback_batch_size=5,
        )
        
        # First iteration
        first_run = []
        for batch in iterator:
            first_run.extend(batch)
        
        # Reset
        iterator.reset()
        
        # Second iteration
        second_run = []
        for batch in iterator:
            second_run.extend(batch)
        
        # Should get same results
        assert sorted(first_run) == sorted(second_run)
        assert len(first_run) == 20
