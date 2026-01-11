"""
Unit tests for staged execution architecture components.

Tests:
- Schema extensions (CHUNK_STRUCT, post_chunked column)
- ChunkInfo serialization to/from Arrow
- Global sort utilities (flatten_chunks, gather_results, create_sorted_batches)
- SOBRDataset migration helpers
- ChunkingArtifactWriter streaming
"""

import pytest
import pyarrow as pa
import numpy as np
import tempfile
from pathlib import Path
from typing import List, Optional

# Schema imports
from neuro_stylometry.data_engine.schemas import (
    CHUNK_STRUCT,
    POST_CHUNKED_FIELD,
    SOBR_SCHEMA,
    SOBR_SCHEMA_STAGED,
    has_post_chunked_column,
    validate_schema_flexible,
)

# ChunkInfo imports
from neuro_stylometry.pollution_guard.semantic_chunker import ChunkInfo, BudgetConfig

# Global sort imports
from neuro_stylometry.pollution_guard.global_sort import (
    FlattenedChunks,
    flatten_chunks,
    gather_results,
    create_sorted_batches,
    compute_padding_stats,
)

from neuro_stylometry.data_engine.chunking_validation import find_invalid_post_chunked_indices


class TestChunkStruct:
    """Tests for CHUNK_STRUCT schema definition."""

    def test_chunk_struct_fields(self):
        """Verify CHUNK_STRUCT has all required fields."""
        field_names = [f.name for f in CHUNK_STRUCT]
        assert "text" in field_names
        assert "start" in field_names
        assert "end" in field_names
        assert "token_count" in field_names
        assert "is_hard_split" in field_names

    def test_chunk_struct_types(self):
        """Verify CHUNK_STRUCT field types."""
        fields = {f.name: f.type for f in CHUNK_STRUCT}
        assert fields["text"] == pa.string()
        assert fields["start"] == pa.int32()
        assert fields["end"] == pa.int32()
        assert fields["token_count"] == pa.int16()
        assert fields["is_hard_split"] == pa.bool_()

    def test_post_chunked_field(self):
        """Verify POST_CHUNKED_FIELD is a list of CHUNK_STRUCT."""
        assert POST_CHUNKED_FIELD.name == "post_chunked"
        assert POST_CHUNKED_FIELD.type == pa.list_(CHUNK_STRUCT)
        assert POST_CHUNKED_FIELD.nullable is True


class TestSobrSchemaStaged:
    """Tests for SOBR_SCHEMA_STAGED with post_chunked column."""

    def test_staged_schema_includes_post_chunked(self):
        """Verify SOBR_SCHEMA_STAGED has post_chunked column."""
        field_names = [f.name for f in SOBR_SCHEMA_STAGED]
        assert "post_chunked" in field_names

    def test_staged_schema_is_superset(self):
        """Verify SOBR_SCHEMA_STAGED includes all base schema fields."""
        base_fields = {f.name for f in SOBR_SCHEMA}
        staged_fields = {f.name for f in SOBR_SCHEMA_STAGED}
        assert base_fields.issubset(staged_fields)


class TestHasPostChunkedColumn:
    """Tests for has_post_chunked_column utility."""

    def test_table_without_post_chunked(self):
        """Table without post_chunked column returns False."""
        table = pa.table({"col1": [1, 2, 3]})
        assert has_post_chunked_column(table) is False

    def test_table_with_post_chunked(self):
        """Table with post_chunked column returns True."""
        # Create table with post_chunked column (all nulls)
        null_chunks = pa.nulls(3, type=pa.list_(CHUNK_STRUCT))
        table = pa.table({
            "col1": [1, 2, 3],
            "post_chunked": null_chunks,
        })
        assert has_post_chunked_column(table) is True


class TestValidateSchemaFlexible:
    """Tests for validate_schema_flexible utility."""

    def test_exact_match_passes(self):
        """Table matching expected schema passes validation."""
        # Create minimal table matching SOBR_SCHEMA (use actual column names)
        table = pa.table({
            "post_id": pa.array(["p1"], type=pa.string()),
            "author_id": pa.array(["a1"], type=pa.dictionary(pa.int32(), pa.string())),
            "post": pa.array(["text"], type=pa.string()),
            "post_masked": pa.array([None], type=pa.string()),
            "birth_year": pa.array([None], type=pa.int16()),
            "female": pa.array([None], type=pa.int8()),
            "nationality": pa.array([None], type=pa.string()),
            "political_leaning": pa.array([None], type=pa.string()),
            "extrovert": pa.array([None], type=pa.int8()),
            "sensing": pa.array([None], type=pa.int8()),
            "feeling": pa.array([None], type=pa.int8()),
            "judging": pa.array([None], type=pa.int8()),
            "split": pa.array([None], type=pa.dictionary(pa.int8(), pa.string())),
            "text_length": pa.array([None], type=pa.int32()),
        })
        # Should not raise
        validate_schema_flexible(table, SOBR_SCHEMA)

    def test_extra_column_allowed(self):
        """Table with extra columns passes flexible validation."""
        table = pa.table({
            "post_id": pa.array(["p1"], type=pa.string()),
            "author_id": pa.array(["a1"], type=pa.dictionary(pa.int32(), pa.string())),
            "post": pa.array(["text"], type=pa.string()),
            "post_masked": pa.array([None], type=pa.string()),
            "birth_year": pa.array([None], type=pa.int16()),
            "female": pa.array([None], type=pa.int8()),
            "nationality": pa.array([None], type=pa.string()),
            "political_leaning": pa.array([None], type=pa.string()),
            "extrovert": pa.array([None], type=pa.int8()),
            "sensing": pa.array([None], type=pa.int8()),
            "feeling": pa.array([None], type=pa.int8()),
            "judging": pa.array([None], type=pa.int8()),
            "split": pa.array([None], type=pa.dictionary(pa.int8(), pa.string())),
            "text_length": pa.array([None], type=pa.int32()),
            # Extra column
            "post_chunked": pa.nulls(1, type=pa.list_(CHUNK_STRUCT)),
        })
        # Should not raise
        validate_schema_flexible(table, SOBR_SCHEMA)


class TestChunkInfoSerialization:
    """Tests for ChunkInfo Arrow serialization."""

    def test_to_arrow_struct(self):
        """ChunkInfo can be serialized to Arrow struct."""
        chunk = ChunkInfo(
            text="Hello world",
            char_start=0,
            char_end=11,
            token_count=3,
            is_hard_split=False,
        )
        result = chunk.to_arrow_struct()
        assert result["text"] == "Hello world"
        assert result["start"] == 0
        assert result["end"] == 11
        assert result["token_count"] == 3
        assert result["is_hard_split"] is False

    def test_from_arrow_struct(self):
        """ChunkInfo can be deserialized from Arrow struct."""
        arrow_data = {
            "text": "Test chunk",
            "start": 5,
            "end": 15,
            "token_count": 4,
            "is_hard_split": True,
        }
        chunk = ChunkInfo.from_arrow_struct(arrow_data)
        assert chunk.text == "Test chunk"
        assert chunk.char_start == 5
        assert chunk.char_end == 15
        assert chunk.token_count == 4
        assert chunk.is_hard_split is True

    def test_roundtrip_serialization(self):
        """ChunkInfo survives roundtrip serialization."""
        original = ChunkInfo(
            text="Round trip test",
            char_start=10,
            char_end=25,
            token_count=5,
            is_hard_split=False,
        )
        arrow_data = original.to_arrow_struct()
        restored = ChunkInfo.from_arrow_struct(arrow_data)
        assert restored.text == original.text
        assert restored.char_start == original.char_start
        assert restored.char_end == original.char_end
        assert restored.token_count == original.token_count
        assert restored.is_hard_split == original.is_hard_split


class TestFlattenedChunks:
    """Tests for FlattenedChunks dataclass."""

    def test_init(self):
        """FlattenedChunks stores all required fields."""
        texts = ["chunk1", "chunk2", "chunk3"]
        token_counts = np.array([10, 20, 15], dtype=np.int16)
        doc_offsets = np.array([0, 2, 3], dtype=np.int64)
        chunk_metadata = [(0, 0), (0, 1), (1, 0)]  # (doc_idx, local_chunk_idx)

        flat = FlattenedChunks(
            texts=texts,
            token_counts=token_counts,
            doc_offsets=doc_offsets,
            chunk_metadata=chunk_metadata,
        )

        assert flat.texts == texts
        assert flat.num_chunks == 3
        assert flat.num_docs == 2

    def test_sort_indices_computed_lazily(self):
        """sort_indices are computed on first access."""
        token_counts = np.array([30, 10, 20], dtype=np.int16)
        flat = FlattenedChunks(
            texts=["a", "b", "c"],
            token_counts=token_counts,
            doc_offsets=np.array([0, 1, 2, 3]),
            chunk_metadata=[(0, 0), (1, 0), (2, 0)],
        )

        # First access computes sort indices
        sort_idx = flat.compute_sort_indices()
        # Should sort by token_count: [10, 20, 30] -> indices [1, 2, 0]
        assert list(sort_idx) == [1, 2, 0]


class TestFlattenChunks:
    """Tests for flatten_chunks utility."""

    def test_flatten_simple(self):
        """Flatten a simple post_chunked column."""
        # Create a post_chunked column with 2 documents
        chunks_doc1 = [
            {"text": "chunk1", "start": 0, "end": 6, "token_count": 2, "is_hard_split": False},
            {"text": "chunk2", "start": 6, "end": 12, "token_count": 3, "is_hard_split": False},
        ]
        chunks_doc2 = [
            {"text": "chunk3", "start": 0, "end": 6, "token_count": 4, "is_hard_split": True},
        ]

        post_chunked = pa.array(
            [chunks_doc1, chunks_doc2],
            type=pa.list_(CHUNK_STRUCT),
        )

        flat = flatten_chunks(post_chunked)

        assert flat.num_chunks == 3
        assert flat.texts == ["chunk1", "chunk2", "chunk3"]
        assert list(flat.token_counts) == [2, 3, 4]
        assert flat.num_docs == 2
        # Check chunk_metadata: (doc_idx, local_chunk_idx)
        assert flat.chunk_metadata == [(0, 0), (0, 1), (1, 0)]

    def test_flatten_empty_doc(self):
        """Flatten handles documents with no chunks."""
        chunks_doc1 = [
            {"text": "only", "start": 0, "end": 4, "token_count": 1, "is_hard_split": False},
        ]
        # Empty list for doc2
        chunks_doc2: List = []

        post_chunked = pa.array(
            [chunks_doc1, chunks_doc2],
            type=pa.list_(CHUNK_STRUCT),
        )

        flat = flatten_chunks(post_chunked)

        assert flat.num_chunks == 1
        assert flat.texts == ["only"]
        assert flat.num_docs == 2


class TestGatherResults:
    """Tests for gather_results utility."""

    def test_gather_simple(self):
        """Gather results back to per-document lists."""
        # Simulated flat results (3 chunks from 2 docs)
        flat_results = [
            [{"text": "r1", "label": "L1"}],
            [{"text": "r2", "label": "L2"}],
            [{"text": "r3", "label": "L3"}],
        ]

        # Create FlattenedChunks structure
        flat = FlattenedChunks(
            texts=["c1", "c2", "c3"],
            token_counts=np.array([10, 20, 15], dtype=np.int16),
            doc_offsets=np.array([0, 2, 3], dtype=np.int64),
            chunk_metadata=[(0, 0), (0, 1), (1, 0)],
        )

        gathered = gather_results(flat_results, flat, restore_sort_order=False)

        assert len(gathered) == 2  # 2 documents
        assert len(gathered[0]) == 2  # Doc 0 has 2 results (r1, r2)
        assert len(gathered[1]) == 1  # Doc 1 has 1 result (r3)


class TestCreateSortedBatches:
    """Tests for create_sorted_batches utility."""

    def test_batches_sorted_by_length(self):
        """Batches are created in sorted order by token count."""
        flat = FlattenedChunks(
            texts=["long", "short", "medium"],
            token_counts=np.array([100, 10, 50], dtype=np.int16),
            doc_offsets=np.array([0, 1, 2, 3], dtype=np.int64),
            chunk_metadata=[(0, 0), (1, 0), (2, 0)],
        )

        batches = create_sorted_batches(flat, batch_size=2)

        # First batch: 2 shortest chunks (indices in sorted order)
        assert len(batches) == 2
        # Sorted order: short (10) -> idx 1, medium (50) -> idx 2, long (100) -> idx 0
        # First batch has indices [1, 2], second has [0]
        assert batches[0] == [1, 2]  # short and medium
        assert batches[1] == [0]     # long


class TestComputePaddingStats:
    """Tests for compute_padding_stats utility."""

    def test_padding_stats(self):
        """Compute padding statistics for sorted vs unsorted batching."""
        flat = FlattenedChunks(
            texts=["a", "b", "c"],
            token_counts=np.array([10, 15, 20], dtype=np.int16),
            doc_offsets=np.array([0, 1, 2, 3], dtype=np.int64),
            chunk_metadata=[(0, 0), (1, 0), (2, 0)],
        )
        
        stats = compute_padding_stats(flat, batch_size=3)

        # All in one batch - max is 20
        # Total tokens: 10 + 15 + 20 = 45
        # Padded total: 20 * 3 = 60
        # Padding: 60 - 45 = 15
        # Unsorted and sorted are same when all in one batch
        assert "unsorted_padding_ratio" in stats
        assert "sorted_padding_ratio" in stats
        assert "efficiency_gain" in stats
        # With batch_size=3, all chunks in one batch, ratios should be equal
        assert stats["unsorted_padding_ratio"] == pytest.approx(stats["sorted_padding_ratio"])


class TestChunkingResumeValidation:
    def test_valid_when_whitespace_reflowed(self):
        posts = ["Hello\nworld", "  "]
        post_chunked = pa.array(
            [
                [
                    {
                        "text": "Hello world",
                        "start": 0,
                        "end": 11,
                        "token_count": 2,
                        "is_hard_split": False,
                    }
                ],
                [],
            ],
            type=pa.list_(CHUNK_STRUCT),
        )

        invalid = find_invalid_post_chunked_indices(posts=posts, post_chunked_column=post_chunked)
        assert invalid == []

    def test_invalid_on_non_whitespace_gap(self):
        posts = ["HelloXworld"]
        post_chunked = pa.array(
            [
                [
                    {"text": "Hello", "start": 0, "end": 5, "token_count": 1, "is_hard_split": False},
                    {"text": "world", "start": 6, "end": 11, "token_count": 1, "is_hard_split": False},
                ]
            ],
            type=pa.list_(CHUNK_STRUCT),
        )

        invalid = find_invalid_post_chunked_indices(posts=posts, post_chunked_column=post_chunked)
        assert invalid == [0]

    def test_invalid_on_content_mismatch(self):
        posts = ["Hello world"]
        post_chunked = pa.array(
            [
                [
                    {"text": "Hello wurld", "start": 0, "end": 11, "token_count": 2, "is_hard_split": False},
                ]
            ],
            type=pa.list_(CHUNK_STRUCT),
        )

        invalid = find_invalid_post_chunked_indices(posts=posts, post_chunked_column=post_chunked)
        assert invalid == [0]
