"""
Unit tests for SOBRDataset schema migration helpers.

Tests:
- ensure_post_chunked_column() adds column when missing
- has_post_chunked() detects populated vs empty columns
- Flexible schema validation accepts extra columns
"""

import pytest
import pyarrow as pa
import pyarrow.feather as feather
import tempfile
from pathlib import Path

from neuro_stylometry.data_engine.dataset import SOBRDataset, load_arrow_dataset
from neuro_stylometry.data_engine.schemas import (
    SOBR_SCHEMA,
    CHUNK_STRUCT,
    has_post_chunked_column,
)


@pytest.fixture
def minimal_sobr_table():
    """Create a minimal valid SOBR table for testing (matches SOBR_SCHEMA)."""
    return pa.table({
        "post_id": pa.array(["p1", "p2", "p3"], type=pa.string()),
        "author_id": pa.array(
            ["a1", "a1", "a2"],
            type=pa.dictionary(pa.int32(), pa.string())
        ),
        "post": pa.array(["Hello world", "Test post", "Another one"], type=pa.string()),
        "post_masked": pa.array([None, None, None], type=pa.string()),
        "birth_year": pa.array([None, None, None], type=pa.int16()),
        "female": pa.array([None, None, None], type=pa.int8()),
        "nationality": pa.array([None, None, None], type=pa.string()),
        "political_leaning": pa.array([None, None, None], type=pa.string()),
        "extrovert": pa.array([None, None, None], type=pa.int8()),
        "sensing": pa.array([None, None, None], type=pa.int8()),
        "feeling": pa.array([None, None, None], type=pa.int8()),
        "judging": pa.array([None, None, None], type=pa.int8()),
        "split": pa.array(
            ["train", "train", "val"],
            type=pa.dictionary(pa.int8(), pa.string())
        ),
        "text_length": pa.array([11, 9, 11], type=pa.int32()),
    })


@pytest.fixture
def temp_arrow_file(minimal_sobr_table):
    """Create a temporary Arrow file with minimal SOBR data."""
    with tempfile.NamedTemporaryFile(suffix=".arrow", delete=False) as f:
        feather.write_feather(minimal_sobr_table, f.name)
        yield Path(f.name)
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


class TestSOBRDatasetMigration:
    """Tests for SOBRDataset migration helpers."""

    def test_load_without_post_chunked(self, temp_arrow_file):
        """Dataset loads successfully without post_chunked column."""
        dataset = SOBRDataset(temp_arrow_file)
        assert len(dataset.table) == 3
        assert "post_chunked" not in dataset.table.column_names

    def test_ensure_post_chunked_adds_column(self, temp_arrow_file):
        """ensure_post_chunked_column() adds column when missing."""
        dataset = SOBRDataset(temp_arrow_file)

        # Column should not exist initially
        assert not has_post_chunked_column(dataset.table)

        # Add the column
        table = dataset.ensure_post_chunked_column()

        # Column should now exist
        assert has_post_chunked_column(table)
        assert has_post_chunked_column(dataset.table)

        # All values should be null
        col = dataset.table["post_chunked"]
        assert col.null_count == len(dataset.table)

    def test_ensure_post_chunked_idempotent(self, temp_arrow_file):
        """ensure_post_chunked_column() is idempotent."""
        dataset = SOBRDataset(temp_arrow_file)

        # Call twice
        dataset.ensure_post_chunked_column()
        dataset.ensure_post_chunked_column()

        # Should still have exactly one post_chunked column
        count = sum(1 for name in dataset.table.column_names if name == "post_chunked")
        assert count == 1

    def test_has_post_chunked_false_when_all_null(self, temp_arrow_file):
        """has_post_chunked() returns False when column exists but is all-null."""
        dataset = SOBRDataset(temp_arrow_file)
        dataset.ensure_post_chunked_column()

        # Column exists but not populated
        assert not dataset.has_post_chunked()

    def test_has_post_chunked_true_when_populated(self, minimal_sobr_table):
        """has_post_chunked() returns True when column has non-null values."""
        # Create table with populated post_chunked
        chunks = [
            {"text": "chunk", "start": 0, "end": 5, "token_count": 2, "is_hard_split": False}
        ]
        post_chunked = pa.array(
            [chunks, None, None],
            type=pa.list_(CHUNK_STRUCT),
        )

        table_with_chunks = minimal_sobr_table.append_column("post_chunked", post_chunked)

        with tempfile.NamedTemporaryFile(suffix=".arrow", delete=False) as f:
            feather.write_feather(table_with_chunks, f.name)
            tmp_path = Path(f.name)

        dataset = SOBRDataset(tmp_path, memory_map=False)

        # Column exists and has at least one non-null value
        assert dataset.has_post_chunked()

        tmp_path.unlink(missing_ok=True)

    def test_strict_schema_rejects_extra_columns(self, minimal_sobr_table):
        """strict_schema=True rejects tables with extra columns."""
        # Add an unexpected column
        extra_table = minimal_sobr_table.append_column(
            "unexpected_column",
            pa.array([1, 2, 3])
        )

        with tempfile.NamedTemporaryFile(suffix=".arrow", delete=False) as f:
            feather.write_feather(extra_table, f.name)
            tmp_path = Path(f.name)

        # strict_schema=True should reject
        with pytest.raises(ValueError, match="unexpected_column"):
            SOBRDataset(tmp_path, strict_schema=True, memory_map=False)

        tmp_path.unlink(missing_ok=True)

    def test_flexible_schema_accepts_post_chunked(self, minimal_sobr_table):
        """Flexible schema (default) accepts post_chunked column."""
        # Add post_chunked column
        null_chunks = pa.nulls(3, type=pa.list_(CHUNK_STRUCT))
        table_with_chunks = minimal_sobr_table.append_column("post_chunked", null_chunks)

        with tempfile.NamedTemporaryFile(suffix=".arrow", delete=False) as f:
            feather.write_feather(table_with_chunks, f.name)
            tmp_path = Path(f.name)

        # Default (flexible) should accept
        dataset = SOBRDataset(tmp_path, memory_map=False)
        assert has_post_chunked_column(dataset.table)

        tmp_path.unlink(missing_ok=True)


class TestLoadArrowDataset:
    """Tests for load_arrow_dataset convenience function."""

    def test_load_with_strict_schema(self, temp_arrow_file):
        """load_arrow_dataset passes strict_schema to SOBRDataset."""
        # Should work with strict_schema=False (default)
        dataset = load_arrow_dataset(temp_arrow_file)
        assert len(dataset.table) == 3

    def test_load_arrow_dataset_signature(self, temp_arrow_file):
        """load_arrow_dataset accepts strict_schema parameter."""
        # This should not raise
        dataset = load_arrow_dataset(temp_arrow_file, strict_schema=False)
        assert dataset is not None
