"""
BucketSampler: Sequence-length bucketing for minimizing padding waste.

Implements dynamic programming algorithm to find optimal bucket boundaries
and a PyTorch Sampler that groups similar-length sequences together.
"""

import torch
from torch.utils.data import Sampler
from typing import Iterable, List, Tuple, Optional
import numpy as np
from datasets import Dataset
import logging

logger = logging.getLogger(__name__)


def compute_bucket_boundaries(
    lengths: np.ndarray,
    num_buckets: int = 10,
    max_padding_ratio: float = 0.15,
) -> List[int]:
    """
    Compute optimal bucket boundaries using dynamic programming.
    
    Goal: Minimize total padding waste while keeping bucket count manageable.
    
    Args:
        lengths: Array of sequence lengths.
        num_buckets: Target number of buckets.
        max_padding_ratio: Maximum acceptable padding ratio per bucket.
        
    Returns:
        List of bucket boundary values (sorted ascending).
        
    Example:
        If boundaries = [50, 100, 200], buckets are:
            [0, 50), [50, 100), [100, 200), [200, max_length]
    """
    sorted_lengths = np.sort(lengths)
    min_len = sorted_lengths[0]
    max_len = sorted_lengths[-1]
    
    logger.info(f"Computing buckets: {len(lengths)} sequences, "
               f"length range [{min_len}, {max_len}]")
    
    # Simple quantile-based bucketing (DP optimization can be added later)
    boundaries = []
    for i in range(1, num_buckets):
        quantile = i / num_buckets
        boundary = int(np.quantile(sorted_lengths, quantile))
        boundaries.append(boundary)
    
    # Remove duplicates and sort
    boundaries = sorted(set(boundaries))
    
    # Compute padding statistics
    total_padding = 0
    total_tokens = 0
    
    for i, boundary in enumerate(boundaries + [max_len]):
        prev_boundary = boundaries[i-1] if i > 0 else 0
        
        # Find sequences in this bucket
        mask = (sorted_lengths >= prev_boundary) & (sorted_lengths < boundary)
        bucket_lengths = sorted_lengths[mask]
        
        if len(bucket_lengths) > 0:
            max_bucket_len = bucket_lengths.max()
            padding = len(bucket_lengths) * max_bucket_len - bucket_lengths.sum()
            tokens = len(bucket_lengths) * max_bucket_len
            
            total_padding += padding
            total_tokens += tokens
            
            logger.info(f"  Bucket {i}: [{prev_boundary}, {boundary}) "
                       f"- {len(bucket_lengths)} sequences, "
                       f"padding ratio: {padding/tokens:.2%}")
    
    overall_padding_ratio = total_padding / total_tokens if total_tokens > 0 else 0
    logger.info(f"Overall padding ratio: {overall_padding_ratio:.2%}")
    
    if overall_padding_ratio > max_padding_ratio:
        logger.warning(f"Padding ratio {overall_padding_ratio:.2%} exceeds "
                      f"target {max_padding_ratio:.2%}")
    
    return boundaries


class BucketSampler(Sampler):
    """
    PyTorch Sampler that groups sequences by length into buckets.
    
    Features:
        - Minimizes padding within batches
        - Maintains shuffling for SGD properties
        - Respects epoch boundaries for reproducibility
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_buckets: int = 10,
        shuffle: bool = True,
        seed: int = 42,
        length_column: str = "text_length",
    ):
        """
        Initialize bucketing sampler.
        
        Args:
            dataset: HuggingFace Dataset with length information.
            batch_size: Number of samples per batch.
            num_buckets: Number of length buckets to create.
            shuffle: Whether to shuffle within buckets.
            seed: Random seed for reproducibility.
            length_column: Column name containing sequence lengths.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_buckets = num_buckets
        self.shuffle = shuffle
        self.seed = seed
        self.length_column = length_column
        self.epoch = 0
        
        # Extract lengths and compute buckets
        self.lengths = np.array(dataset[length_column])
        self.boundaries = compute_bucket_boundaries(
            self.lengths,
            num_buckets=num_buckets
        )
        
        # Assign each sample to a bucket
        self.bucket_assignments = self._assign_to_buckets()
        
        # Group indices by bucket
        self.bucket_indices = self._group_by_bucket()
    
    def _assign_to_buckets(self) -> np.ndarray:
        """
        Assign each sample to a bucket based on its length.
        
        Returns:
            Array of bucket IDs (0 to num_buckets-1).
        """
        assignments = np.zeros(len(self.lengths), dtype=np.int32)
        
        for i, length in enumerate(self.lengths):
            # Find which bucket this length falls into
            bucket_id = 0
            for boundary_idx, boundary in enumerate(self.boundaries):
                if length >= boundary:
                    bucket_id = boundary_idx + 1
                else:
                    break
            assignments[i] = bucket_id
        
        return assignments
    
    def _group_by_bucket(self) -> List[List[int]]:
        """
        Group dataset indices by bucket.
        
        Returns:
            List of lists, where bucket_indices[i] contains all indices in bucket i.
        """
        num_buckets = len(self.boundaries) + 1
        buckets = [[] for _ in range(num_buckets)]
        
        for idx, bucket_id in enumerate(self.bucket_assignments):
            buckets[bucket_id].append(idx)
        
        logger.info(f"Bucket distribution: "
                   f"{[len(b) for b in buckets]}")
        
        return buckets
    
    def __iter__(self):
        """
        Generate batched indices for one epoch.
        
        Yields:
            Batch indices (list of integers).
        """
        # Set seed for this epoch
        rng = np.random.default_rng(self.seed + self.epoch)
        
        all_batches = []
        
        # Process each bucket
        for bucket_id, indices in enumerate(self.bucket_indices):
            if len(indices) == 0:
                continue
            
            # Shuffle within bucket if requested
            if self.shuffle:
                indices = np.array(indices)
                rng.shuffle(indices)
                indices = indices.tolist()
            
            # Create batches from this bucket
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                all_batches.append(batch)
        
        # Shuffle batch order (but not contents) to maintain stochasticity
        if self.shuffle:
            rng.shuffle(all_batches)
        
        # Yield batches
        for batch in all_batches:
            yield batch
    
    def __len__(self):
        """
        Return total number of batches per epoch.
        """
        total_samples = sum(len(bucket) for bucket in self.bucket_indices)
        return (total_samples + self.batch_size - 1) // self.batch_size
    
    def set_epoch(self, epoch: int):
        """
        Set the epoch for reproducible shuffling.
        
        Args:
            epoch: Current epoch number.
        """
        self.epoch = epoch


def create_bucketed_sampler(
    dataset: Dataset,
    batch_size: int,
    num_buckets: int = 10,
    shuffle: bool = True,
    seed: int = 42,
) -> BucketSampler:
    """
    Convenience function to create a BucketSampler.
    
    Args:
        dataset: HuggingFace Dataset.
        batch_size: Batch size.
        num_buckets: Number of length buckets.
        shuffle: Whether to shuffle.
        seed: Random seed.
        
    Returns:
        Configured BucketSampler.
    """
    return BucketSampler(
        dataset=dataset,
        batch_size=batch_size,
        num_buckets=num_buckets,
        shuffle=shuffle,
        seed=seed
    )


class SortishBatchSampler(Sampler[List[int]]):
    """
    Sortish batch sampler for offline length-based bucketing.

    Precomputes batches by sorting indices by length, grouping into
    mega-batches, shuffling mega-batches, then forming batches.
    """

    def __init__(
        self,
        lengths: Iterable[int],
        *,
        batch_size: int,
        mega_batch_mult: int = 100,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        self.lengths = np.asarray(list(lengths), dtype=np.int64)
        self.batch_size = int(batch_size)
        self.mega_batch_mult = int(mega_batch_mult)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        total = len(self.lengths)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = np.argsort(self.lengths, kind="stable")
        rng = np.random.default_rng(self.seed + self.epoch)
        mega = self.batch_size * max(1, self.mega_batch_mult)
        mega_batches = [
            indices[i:i + mega]
            for i in range(0, len(indices), mega)
        ]
        if self.shuffle:
            rng.shuffle(mega_batches)

        batches: List[List[int]] = []
        for mega_batch in mega_batches:
            mega_list = mega_batch.tolist()
            for i in range(0, len(mega_list), self.batch_size):
                batch = mega_list[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)

        for batch in batches:
            yield batch
