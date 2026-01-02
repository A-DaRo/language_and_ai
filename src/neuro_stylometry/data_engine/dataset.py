"""
SOBRDataset: Memory-mapped Arrow dataset with author-stratified splitting.

Ensures zero-copy access and prevents author leakage across train/val/test splits.
"""

import pyarrow as pa
import pyarrow.feather as feather
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
import logging

from .schemas import SOBR_SCHEMA, validate_schema

logger = logging.getLogger(__name__)


class SOBRDataset:
    """
    Wrapper for memory-mapped SOBR Arrow dataset with author-stratified splits.
    
    Key Features:
        - Memory mapping for zero-copy access
        - Author-stratified train/val/test splitting (no author leakage)
        - Preserves dictionary encoding for memory efficiency
    """
    
    def __init__(
        self,
        arrow_path: Path,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
    ):
        """
        Initialize dataset from Arrow file.
        
        Args:
            arrow_path: Path to the Arrow/Feather file.
            split_ratios: (train, val, test) proportions (must sum to 1.0).
            seed: Random seed for reproducible splitting.
            
        Raises:
            ValueError: If split_ratios don't sum to 1.0.
        """
        if not np.isclose(sum(split_ratios), 1.0):
            raise ValueError(f"split_ratios must sum to 1.0, got {sum(split_ratios)}")
        
        self.arrow_path = Path(arrow_path)
        self.split_ratios = split_ratios
        self.seed = seed
        
        # Load and validate
        self.table = self._load_with_mmap()
        validate_schema(self.table, SOBR_SCHEMA)
        logger.info(f"Loaded {len(self.table)} rows from {arrow_path}")
        
        # Create stratified splits
        self.splits = self._create_author_stratified_splits()
    
    def _load_with_mmap(self) -> pa.Table:
        """
        Load Arrow Table with memory mapping enabled.
        
        Returns:
            PyArrow Table with mmap=True for zero-copy access.
        """
        table = feather.read_table(
            self.arrow_path,
            memory_map=True  # Critical: enables zero-copy access
        )
        return table
    
    def _create_author_stratified_splits(self) -> Dict[str, List[int]]:
        """
        Create train/val/test splits stratified by author_id.
        
        This ensures all posts from Author X appear in exactly ONE split,
        preventing data leakage.
        
        Returns:
            Dict mapping split name -> list of row indices.
        """
        # Extract author_id column and compute integer codes robustly
        author_col = self.table['author_id']
        author_series = author_col.to_pandas()
        author_indices = pd.factorize(author_series)[0]
        
        # Get unique author codes
        unique_authors = np.unique(author_indices)
        logger.info(f"  Found {len(unique_authors)} unique authors")
        
        # Shuffle authors deterministically
        rng = np.random.default_rng(self.seed)
        rng.shuffle(unique_authors)
        
        # Split authors into train/val/test
        n_authors = len(unique_authors)
        train_end = int(n_authors * self.split_ratios[0])
        val_end = train_end + int(n_authors * self.split_ratios[1])
        
        train_authors = set(unique_authors[:train_end])
        val_authors = set(unique_authors[train_end:val_end])
        test_authors = set(unique_authors[val_end:])
        
        logger.info(f"  Split authors: {len(train_authors)} train, "
                   f"{len(val_authors)} val, {len(test_authors)} test")
        
        # Map authors back to row indices
        train_indices = np.where(np.isin(author_indices, list(train_authors)))[0].tolist()
        val_indices = np.where(np.isin(author_indices, list(val_authors)))[0].tolist()
        test_indices = np.where(np.isin(author_indices, list(test_authors)))[0].tolist()
        
        logger.info(f"  Split rows: {len(train_indices)} train, "
                   f"{len(val_indices)} val, {len(test_indices)} test")
        
        # Verify no overlap
        assert len(set(train_indices) & set(val_indices)) == 0, "Train/Val overlap detected!"
        assert len(set(train_indices) & set(test_indices)) == 0, "Train/Test overlap detected!"
        assert len(set(val_indices) & set(test_indices)) == 0, "Val/Test overlap detected!"
        
        return {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices,
        }
    
    def get_huggingface_dataset(self) -> DatasetDict:
        """
        Convert to HuggingFace DatasetDict for easy integration.
        
        Returns:
            DatasetDict with 'train', 'validation', 'test' splits.
        """
        # Convert Arrow Table to pandas (efficient for indexed selection)
        df = self.table.to_pandas()
        
        # Create datasets for each split
        datasets = {}
        for split_name, indices in self.splits.items():
            split_df = df.iloc[indices].reset_index(drop=True)
            
            # Convert back to HuggingFace Dataset
            hf_name = 'validation' if split_name == 'val' else split_name
            datasets[hf_name] = Dataset.from_pandas(split_df, preserve_index=False)
        
        return DatasetDict(datasets)
    
    def get_split_indices(self, split: str) -> List[int]:
        """
        Get row indices for a specific split.
        
        Args:
            split: One of 'train', 'val', 'test'.
            
        Returns:
            List of integer row indices.
        """
        if split not in self.splits:
            raise ValueError(f"Unknown split '{split}', must be one of {list(self.splits.keys())}")
        return self.splits[split]
    
    def get_author_statistics(self) -> Dict[str, any]:
        """
        Compute statistics about author distribution across splits.
        
        Returns:
            Dictionary with author counts, post counts per split.
        """
        author_col = self.table['author_id']
        author_series = author_col.to_pandas()
        author_indices = pd.factorize(author_series)[0]
        
        stats = {}
        for split_name, row_indices in self.splits.items():
            split_authors = author_indices[row_indices]
            unique_authors = np.unique(split_authors)
            
            stats[split_name] = {
                'num_authors': len(unique_authors),
                'num_posts': len(row_indices),
                'avg_posts_per_author': len(row_indices) / len(unique_authors),
            }
        
        return stats


def load_arrow_dataset(
    arrow_path: Path,
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> SOBRDataset:
    """
    Convenience function to load a SOBR dataset from Arrow file.
    
    Args:
        arrow_path: Path to Arrow/Feather file.
        split_ratios: (train, val, test) proportions.
        seed: Random seed for splitting.
        
    Returns:
        SOBRDataset instance with stratified splits.
    """
    return SOBRDataset(arrow_path, split_ratios, seed)
