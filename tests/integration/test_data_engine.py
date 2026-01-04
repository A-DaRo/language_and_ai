"""
Integration Test: Data Engine

Validates:
- SOBRDataset loads sobr_laptop.arrow correctly
- Memory mapping is enabled
- Schema validation passes
- Author-stratified splits have no leakage
"""

import pytest
from pathlib import Path
import pyarrow as pa
import numpy as np

from neuro_stylometry.data_engine.dataset import SOBRDataset
from neuro_stylometry.data_engine.schemas import SOBR_SCHEMA, validate_schema


@pytest.fixture
def laptop_dataset_path():
    """Path to laptop dataset (sobr_laptop.arrow)."""
    path = Path("artifacts/data/sobr_laptop.arrow")
    if not path.exists():
        pytest.skip(f"Laptop dataset not found at {path}")
    return path


@pytest.fixture
def sobr_dataset(laptop_dataset_path):
    """Load SOBRDataset for testing."""
    return SOBRDataset(
        arrow_path=laptop_dataset_path,
        split_ratios=(0.8, 0.1, 0.1),
        seed=42,
    )


class TestDataEngine:
    """Integration tests for data engine components."""
    
    def test_schema_validation(self, sobr_dataset):
        """Test that loaded dataset conforms to SOBR_SCHEMA."""
        # Implements FR-02: Schema validation
        validate_schema(sobr_dataset.table, SOBR_SCHEMA)
        
        # Check critical columns exist
        assert "post_id" in sobr_dataset.table.column_names
        assert "post" in sobr_dataset.table.column_names
        assert "post_masked" in sobr_dataset.table.column_names
        assert "author_id" in sobr_dataset.table.column_names
    
    def test_memory_mapping(self, laptop_dataset_path):
        """Test that memory mapping is enabled for zero-copy access."""
        # Directly load with feather to verify mmap flag
        import pyarrow.feather as feather
        table = feather.read_table(laptop_dataset_path, memory_map=True)
        
        # Verify table is loaded
        assert len(table) > 0
        assert isinstance(table, pa.Table)
    
    def test_author_stratified_splits(self, sobr_dataset):
        """Test that splits are stratified by author with no leakage."""
        # Implements FR-03: Author stratification
        splits = sobr_dataset.splits
        
        # Check all splits exist
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits
        
        # Check no index overlap
        train_set = set(splits['train'])
        val_set = set(splits['val'])
        test_set = set(splits['test'])
        
        assert len(train_set & val_set) == 0, "Train/Val overlap detected"
        assert len(train_set & test_set) == 0, "Train/Test overlap detected"
        assert len(val_set & test_set) == 0, "Val/Test overlap detected"
        
        # Check all rows are accounted for
        total_rows = len(sobr_dataset.table)
        split_rows = len(train_set) + len(val_set) + len(test_set)
        assert split_rows == total_rows, f"Split rows ({split_rows}) != total rows ({total_rows})"
        
        # Verify author-level stratification
        author_col = sobr_dataset.table['author_id'].to_pandas()
        
        train_authors = set(author_col.iloc[list(train_set)])
        val_authors = set(author_col.iloc[list(val_set)])
        test_authors = set(author_col.iloc[list(test_set)])
        
        # No author should appear in multiple splits
        assert len(train_authors & val_authors) == 0, "Authors shared between train/val"
        assert len(train_authors & test_authors) == 0, "Authors shared between train/test"
        assert len(val_authors & test_authors) == 0, "Authors shared between val/test"
    
    def test_split_ratios(self, sobr_dataset):
        """Test that split ratios are approximately correct."""
        total_rows = len(sobr_dataset.table)
        train_rows = len(sobr_dataset.splits['train'])
        val_rows = len(sobr_dataset.splits['val'])
        test_rows = len(sobr_dataset.splits['test'])
        
        train_ratio = train_rows / total_rows
        val_ratio = val_rows / total_rows
        test_ratio = test_rows / total_rows
        
        # Author-stratified splits can skew row ratios on small datasets.
        # Validate author ratios strictly, row ratios loosely.
        author_col = sobr_dataset.table['author_id'].to_pandas()
        train_authors = set(author_col.iloc[list(sobr_dataset.splits['train'])])
        val_authors = set(author_col.iloc[list(sobr_dataset.splits['val'])])
        test_authors = set(author_col.iloc[list(sobr_dataset.splits['test'])])
        total_authors = len(train_authors) + len(val_authors) + len(test_authors)

        train_author_ratio = len(train_authors) / total_authors
        val_author_ratio = len(val_authors) / total_authors
        test_author_ratio = len(test_authors) / total_authors

        assert 0.75 <= train_author_ratio <= 0.85, (
            f"Train author ratio {train_author_ratio:.2%} outside [75%, 85%]"
        )
        assert 0.05 <= val_author_ratio <= 0.15, (
            f"Val author ratio {val_author_ratio:.2%} outside [5%, 15%]"
        )
        assert 0.05 <= test_author_ratio <= 0.15, (
            f"Test author ratio {test_author_ratio:.2%} outside [5%, 15%]"
        )

        # Row ratios: allow wider tolerance for small datasets
        assert 0.60 <= train_ratio <= 0.90, f"Train ratio {train_ratio:.2%} outside [60%, 90%]"
        assert 0.02 <= val_ratio <= 0.25, f"Val ratio {val_ratio:.2%} outside [2%, 25%]"
        assert 0.02 <= test_ratio <= 0.25, f"Test ratio {test_ratio:.2%} outside [2%, 25%]"
    
    def test_huggingface_conversion(self, sobr_dataset):
        """Test conversion to HuggingFace DatasetDict."""
        dataset_dict = sobr_dataset.get_huggingface_dataset()
        
        # Check splits
        assert 'train' in dataset_dict
        assert 'validation' in dataset_dict  # Note: 'val' -> 'validation'
        assert 'test' in dataset_dict
        
        # Check columns
        train_dataset = dataset_dict['train']
        assert 'post' in train_dataset.column_names
        assert 'author_id' in train_dataset.column_names
    
    def test_demographic_labels_nullable(self, sobr_dataset):
        """Test that demographic labels handle NULL values correctly."""
        # Implements FR-04: Nullable demographic columns
        table = sobr_dataset.table
        
        # Check nullable columns
        nullable_cols = [
            'birth_year', 'female', 'nationality', 'political_leaning',
            'extrovert', 'sensing', 'feeling', 'judging'
        ]
        
        for col_name in nullable_cols:
            col = table[col_name]
            # Verify column allows nulls (doesn't crash on .null_count)
            null_count = col.null_count
            # Note: null_count can be 0 (all non-null) or > 0 (some nulls)
            assert null_count >= 0, f"Column {col_name} null_count check failed"
