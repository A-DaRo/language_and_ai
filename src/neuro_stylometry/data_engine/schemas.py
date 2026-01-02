"""
Arrow Schema Definitions for SOBR Dataset and Pollution Logs.

This module serves as the source of truth for all data type contracts,
ensuring type safety before data reaches tensor computation.
"""

import pyarrow as pa
from typing import List


# ==============================================================================
# SOBR Unified Dataset Schema
# ==============================================================================

SOBR_SCHEMA = pa.schema([
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
])


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
        raise ValueError(
            f"Schema mismatch: Expected {len(expected_schema)} columns, "
            f"got {len(actual_schema)}"
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
