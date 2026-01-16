"""
Arrow Schema Definitions for SOBR Dataset and Pollution Logs.

This module serves as the source of truth for all data type contracts,
ensuring type safety before data reaches tensor computation.

Staged Execution Architecture:
- post_chunked: Persisted chunks enable CPU/GPU decoupling and crash recovery.
- CHUNK_STRUCT: Nested struct with pre-computed token_count for zero-copy global sorting.
"""

import pyarrow as pa
from typing import List, Optional


# ==============================================================================
# Chunk Struct Definition (Staged Execution)
# ==============================================================================
# Elevates chunks from ephemeral Python objects to first-class schema citizens.
# The token_count field enables O(1) length lookup for global argsort without
# deserializing text content (zero-copy inspection).
#
# Reference: Technical Report Section 1.1 (Enhanced Arrow Schema)
# ==============================================================================

CHUNK_STRUCT = pa.struct([
    ("text", pa.string()),           # The chunk text content
    ("start", pa.int32()),           # Global character start offset (for reconstruction)
    ("end", pa.int32()),             # Global character end offset (for reconstruction)
    ("token_count", pa.int16()),     # Pre-computed token length (critical for global sorting)
    ("is_hard_split", pa.bool_()),   # True if resulted from fallback hard-slicing
])

# Field definition for the chunked column (list of chunk structs per document)
POST_CHUNKED_FIELD = pa.field("post_chunked", pa.list_(CHUNK_STRUCT), nullable=True)


# ==============================================================================
# AOT (Ahead-Of-Time) Tokenization Schema (Phase D Optimization)
# ==============================================================================
# Pre-computed token tensors for zero-JIT training loops.
# Using uint16 for input_ids reduces PCIe bus pressure by 4x vs int64.
# uint8 for attention_mask since values are binary (0/1).
#
# Reference: Phase D Final Optimization Blueprint Section 2.1
# ==============================================================================

TOKENIZED_FIELDS = [
    # Pre-computed token IDs (vocab size < 65536 for all major tokenizers)
    ("input_ids", pa.list_(pa.uint16())),
    # Binary attention mask (1=attend, 0=ignore/pad)
    ("attention_mask", pa.list_(pa.uint8())),
    # Exact token count for O(1) bucketing (max 32767 tokens per sequence)
    ("token_count", pa.int16()),
]


# ==============================================================================
# SOBR Unified Dataset Schema
# ==============================================================================

# Base schema fields (without optional staged execution columns)
_SOBR_BASE_FIELDS = [
    # Primary Key and Content
    ("post_id", pa.string()),  # Unique identifier for each post
    ("author_id", pa.dictionary(pa.int32(), pa.string())),  # Dictionary-encoded for memory efficiency
    ("post", pa.string()),  # Raw post text
    ("post_masked", pa.string()),  # GLiNER-masked text (populated in Phase A)
    
    # Demographic Labels (All Nullable - Outer Join may produce NULLs)
    ("birth_year", pa.int16()),  # Nullable: Year of birth (e.g., 1985)
    ("female", pa.int8()),  # Nullable: Binary gender (0/1 or -1 for NULL)
    ("nationality", pa.string()),  # Nullable: Nationality as string
    ("political_leaning", pa.string()),  # Nullable: Political leaning as string
    
    # Myers-Briggs Type Indicator (4 binary dimensions, nullable)
    ("extrovert", pa.int8()),  # Nullable: E/I dimension (binary)
    ("sensing", pa.int8()),    # Nullable: S/N dimension (binary)
    ("feeling", pa.int8()),     # Nullable: F/T dimension (binary)
    ("judging", pa.int8()),   # Nullable: J/P dimension (binary)
    
    # Metadata
    ("split", pa.dictionary(pa.int8(), pa.string())),  # train/val/test (assigned during splitting)
    ("text_length", pa.int32()),  # Cached character count for bucketing
]

SOBR_SCHEMA = pa.schema(_SOBR_BASE_FIELDS)

# Extended schema including post_chunked for staged execution
SOBR_SCHEMA_STAGED = pa.schema(_SOBR_BASE_FIELDS + [POST_CHUNKED_FIELD])

# AOT-tokenized schema for Phase D high-throughput training (includes pre-computed tensors)
SOBR_SCHEMA_TOKENIZED = pa.schema(_SOBR_BASE_FIELDS + TOKENIZED_FIELDS)


# ==============================================================================
# Pollution Detection Log Schema
# ==============================================================================

POLLUTION_LOG_SCHEMA = pa.schema([
    # Reference
    ("post_id", pa.string()),  # Links back to SOBR_SCHEMA
    
    # Detected Span
    ("span_start", pa.int32()),  # Character offset start
    ("span_end", pa.int32()),    # Character offset end
    ("span_text", pa.string()),  # Original text of the detected span
    
    # GLiNER Detection Metadata
    ("entity_type", pa.string()),  # e.g., "BIRTH_YEAR", "NATIONALITY", "GENDER"
    ("confidence", pa.float32()),  # GLiNER detection confidence [0.0, 1.0]
    ("mask_token", pa.string()),   # The typed mask token applied (e.g., "[MASK:BIRTH_YEAR]")
])


# ==============================================================================
# Schema Validation Utilities
# ==============================================================================

def validate_schema(table: pa.Table, expected_schema: pa.Schema) -> bool:
    """
    Validate that a PyArrow Table conforms to the expected schema.
    
    Args:
        table: The PyArrow Table to validate.
        expected_schema: The expected schema.
        
    Returns:
        True if validation passes.
        
    Raises:
        ValueError: If schema validation fails with detailed error message.
    """
    actual_schema = table.schema
    
    # Check column count
    if len(actual_schema) != len(expected_schema):
        expected_names = [f.name for f in expected_schema]
        actual_names = [f.name for f in actual_schema]
        missing = [n for n in expected_names if n not in actual_names]
        extra = [n for n in actual_names if n not in expected_names]

        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")

        suffix = f" ({', '.join(details)})" if details else ""
        raise ValueError(
            f"Schema mismatch: Expected {len(expected_schema)} columns, "
            f"got {len(actual_schema)}{suffix}"
        )
    
    # Check column names and types
    mismatches = []
    for expected_field, actual_field in zip(expected_schema, actual_schema):
        if expected_field.name != actual_field.name:
            mismatches.append(
                f"Column name mismatch: Expected '{expected_field.name}', "
                f"got '{actual_field.name}'"
            )
        elif not actual_field.type.equals(expected_field.type):
            mismatches.append(
                f"Type mismatch for column '{expected_field.name}': "
                f"Expected {expected_field.type}, got {actual_field.type}"
            )
    
    if mismatches:
        raise ValueError("Schema validation failed:\n" + "\n".join(mismatches))
    
    return True


def get_categorical_columns() -> List[str]:
    """
    Return list of columns that should use dictionary encoding.
    
    This is used by the converter to ensure efficient memory usage
    and by the dataset splitter to identify the author_id column.
    """
    return [
        "author_id",
        "split",
    ]


def get_demographic_columns() -> List[str]:
    """
    Return list of all demographic label columns.
    
    Used by pollution guard strategies to identify which columns
    need to be projected away by LEACE.
    """
    return [
        "birth_year",
        "female",
        "nationality",
        "political_leaning",
        "extrovert",
        "sensing",
        "feeling",
        "judging",
    ]


def get_nullable_columns() -> List[str]:
    """
    Return list of columns that may contain NULL values.
    
    This is critical for the DataLoader collator to handle missing
    demographic labels correctly (e.g., map to ignore_index=-100).
    """
    return get_demographic_columns()  # All demographic labels are nullable


def has_post_chunked_column(table: pa.Table) -> bool:
    """
    Check if a table has the post_chunked column for staged execution.
    
    Used for resume logic: if post_chunked exists, skip Stage 1 (chunking).
    
    Args:
        table: PyArrow Table to check.
        
    Returns:
        True if post_chunked column exists and is properly typed.
    """
    if "post_chunked" not in table.column_names:
        return False
    
    col_type = table.schema.field("post_chunked").type
    # Verify it's a list of structs (not just any column named post_chunked)
    return pa.types.is_list(col_type) and pa.types.is_struct(col_type.value_type)


def count_missing_post_chunked(table: pa.Table) -> int:
    """Count how many rows have NULL post_chunked.

    Empty lists are treated as valid (computed) values; only Arrow NULLs count as missing.
    """
    if not has_post_chunked_column(table):
        return len(table)

    col = table["post_chunked"]
    return int(col.null_count)


def is_post_chunked_fully_populated(table: pa.Table) -> bool:
    """True iff post_chunked exists and has no NULLs."""
    if not has_post_chunked_column(table):
        return False
    return count_missing_post_chunked(table) == 0


def validate_schema_flexible(
    table: pa.Table,
    expected_schema: pa.Schema,
    allow_extra_columns: bool = True,
) -> bool:
    """
    Flexible schema validation that allows extra columns (e.g., post_chunked).
    
    Used for staged execution where post_chunked may or may not be present.
    
    Args:
        table: The PyArrow Table to validate.
        expected_schema: The base schema to validate against.
        allow_extra_columns: If True, extra columns are permitted.
        
    Returns:
        True if validation passes.
        
    Raises:
        ValueError: If schema validation fails.
    """
    actual_names = set(table.column_names)
    expected_names = set(f.name for f in expected_schema)
    
    # Check all expected columns exist
    missing = expected_names - actual_names
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Check types of expected columns
    mismatches = []
    for expected_field in expected_schema:
        actual_field = table.schema.field(expected_field.name)
        if not actual_field.type.equals(expected_field.type):
            mismatches.append(
                f"Type mismatch for '{expected_field.name}': "
                f"Expected {expected_field.type}, got {actual_field.type}"
            )
    
    if mismatches:
        raise ValueError("Schema validation failed:\n" + "\n".join(mismatches))
    
    # Check for unexpected columns
    if not allow_extra_columns:
        extra = actual_names - expected_names
        if extra:
            raise ValueError(f"Unexpected columns: {extra}")
    
    return True
