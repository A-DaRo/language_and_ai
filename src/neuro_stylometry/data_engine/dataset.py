"""
SOBRDataset: Memory-mapped Arrow dataset with author-stratified splitting.

Ensures zero-copy access and prevents author leakage across train/val/test splits.
Supports schema migration for staged execution (post_chunked column).
"""

import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.compute as pc
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
import logging

from .schemas import (
    SOBR_SCHEMA,
    POST_CHUNKED_FIELD,
    CHUNK_STRUCT,
    validate_schema,
    validate_schema_flexible,
    has_post_chunked_column,
)

logger = logging.getLogger(__name__)


class SOBRDataset:
    """
    Wrapper for memory-mapped SOBR Arrow dataset with author-stratified splits.

    Key Features:
        - Memory mapping for zero-copy access
        - Author-stratified train/val/test splitting (no author leakage)
        - Preserves dictionary encoding for memory efficiency
        - Schema migration support for staged execution (post_chunked column)
    """

    def __init__(
        self,
        arrow_path: Path,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        strict_schema: bool = False,
        memory_map: bool = True,
    ):
        """
        Initialize dataset from Arrow file.

        Args:
            arrow_path: Path to the Arrow/Feather file.
            split_ratios: (train, val, test) proportions (must sum to 1.0).
            seed: Random seed for reproducible splitting.
            strict_schema: If True, enforce exact schema match. If False (default),
                allow additional columns like post_chunked for staged execution.

        Raises:
            ValueError: If split_ratios don't sum to 1.0.
        """
        if not np.isclose(sum(split_ratios), 1.0):
            raise ValueError(f"split_ratios must sum to 1.0, got {sum(split_ratios)}")

        self.arrow_path = Path(arrow_path)
        self.split_ratios = split_ratios
        self.seed = seed
        self.memory_map = bool(memory_map)

        # Load and validate (flexible by default to support staged execution)
        self.table = self._load_with_mmap()
        if strict_schema:
            validate_schema(self.table, SOBR_SCHEMA)
        else:
            validate_schema_flexible(self.table, SOBR_SCHEMA)
        logger.info(f"Loaded {len(self.table)} rows from {arrow_path}")

        # Log if post_chunked exists (staged execution mode)
        if has_post_chunked_column(self.table):
            logger.info("  Dataset has post_chunked column (staged execution ready)")

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
            memory_map=self.memory_map
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

    @staticmethod
    def _sanitize_arrow_table(table: pa.Table) -> pa.Table:
        """Sanitize Arrow table for Hugging Face `datasets` conversion.

        The `datasets` library can fail when encountering dictionary-encoded columns
        whose dictionary value type is `null` (e.g., `dictionary<values=null,...>`),
        which commonly arises when a split has an all-null dictionary column.

        Strategy:
        - For dictionary columns with a `null` value_type, cast to `string`.
          This preserves missingness while avoiding invalid `null` dictionary values.
        """

        if table.num_rows == 0:
            return table

        updated_columns = []
        updated_names = []

        for name in table.column_names:
            col = table[name]
            col_type = col.type

            if pa.types.is_dictionary(col_type) and (
                pa.types.is_null(col_type.value_type) or col.null_count == table.num_rows
            ):
                # Cast dictionary<null> (or all-null dictionary) -> string; keep nulls.
                try:
                    # Prefer compute.cast because it handles chunked arrays well.
                    casted = pc.cast(col, pa.string())
                except Exception:
                    casted = col.cast(pa.string())

                updated_columns.append(casted)
                updated_names.append(name)
                continue

            updated_columns.append(col)
            updated_names.append(name)

        return pa.table(updated_columns, names=updated_names)

    def get_huggingface_dataset(self) -> DatasetDict:
        """
        Convert to HuggingFace DatasetDict for easy integration.

        Returns:
            DatasetDict with 'train', 'validation', 'test' splits.
        """
        # Create datasets for each split (slice as Arrow first, then sanitize)
        datasets = {}
        for split_name, indices in self.splits.items():
            hf_name = 'validation' if split_name == 'val' else split_name

            split_table = self.table.take(pa.array(indices, type=pa.int64()))
            split_table = self._sanitize_arrow_table(split_table)

            # Avoid pandas -> arrow conversion issues with all-null categoricals.
            data_dict = split_table.to_pydict()
            datasets[hf_name] = Dataset.from_dict(data_dict)

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

    def ensure_post_chunked_column(self) -> pa.Table:
        """
        Ensure the table has a post_chunked column for staged execution.

        If the column is missing, adds it as an all-null column with the correct
        CHUNK_STRUCT schema. This enables loading older datasets into the staged
        execution pipeline (Stage 1 will then populate the column).

        Returns:
            Table with post_chunked column (original or extended).
        """
        if has_post_chunked_column(self.table):
            logger.debug("post_chunked column already exists")
            return self.table

        # Create all-null column with correct schema
        null_chunks = pa.nulls(len(self.table), type=pa.list_(CHUNK_STRUCT))

        # Append column to table
        self.table = self.table.append_column(POST_CHUNKED_FIELD, null_chunks)
        logger.info(f"Added post_chunked column ({len(self.table)} null entries)")

        return self.table

    def has_post_chunked(self) -> bool:
        """
        Check if dataset has post_chunked column populated (not all-null).

        Returns:
            True if post_chunked column exists and has at least one non-null value.
        """
        if not has_post_chunked_column(self.table):
            return False

        # Check if all values are null (column exists but not populated)
        col = self.table["post_chunked"]
        return col.null_count < len(self.table)


def load_arrow_dataset(
    arrow_path: Path,
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    strict_schema: bool = False,
) -> SOBRDataset:
    """
    Convenience function to load a SOBR dataset from Arrow file.

    Args:
        arrow_path: Path to Arrow/Feather file.
        split_ratios: (train, val, test) proportions.
        seed: Random seed for splitting.
        strict_schema: If True, enforce exact schema match.

    Returns:
        SOBRDataset instance with stratified splits.
    """
    return SOBRDataset(arrow_path, split_ratios, seed, strict_schema)
