"""
Dataset audit utilities for Phase D evaluation readiness.

Focus: label coverage, class balance, text length statistics, and mask-token usage.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

from ..data_engine.schemas import SOBR_SCHEMA, get_demographic_columns, validate_schema_flexible

logger = logging.getLogger(__name__)


_MASK_PATTERN = re.compile(r"\[MASK:[A-Z_]+\]")


@dataclass(frozen=True)
class AuditConfig:
    """Configuration for dataset audit."""

    sample_size: Optional[int] = None
    seed: int = 42
    text_fields: Tuple[str, ...] = ("post", "post_masked")
    max_value_counts: int = 20
    high_missingness_threshold: float = 0.5


def audit_dataset(arrow_path: Path, config: Optional[AuditConfig] = None) -> Dict[str, Any]:
    """Run a dataset audit and return a JSON-serializable summary."""
    cfg = config or AuditConfig()
    arrow_path = Path(arrow_path)

    table = feather.read_table(arrow_path, memory_map=True)

    schema_ok = True
    schema_error = None
    try:
        validate_schema_flexible(table, SOBR_SCHEMA)
    except Exception as exc:
        schema_ok = False
        schema_error = str(exc)

    total_rows = table.num_rows
    sample_table = _sample_table(table, cfg.sample_size, cfg.seed)
    sampled_rows = sample_table.num_rows

    summary: Dict[str, Any] = {
        "dataset_path": str(arrow_path),
        "rows_total": int(total_rows),
        "rows_sampled": int(sampled_rows),
        "sample_fraction": float(sampled_rows / total_rows) if total_rows else 0.0,
        "schema_ok": schema_ok,
        "schema_error": schema_error,
        "columns_present": list(sample_table.column_names),
        "warnings": [],
    }

    summary["id_stats"] = _compute_id_stats(sample_table)
    summary["text_stats"] = _compute_text_stats(sample_table, cfg.text_fields)
    summary["label_stats"] = _compute_label_stats(
        sample_table,
        demographic_columns=get_demographic_columns(),
        max_values=cfg.max_value_counts,
    )

    summary["warnings"].extend(
        _missingness_warnings(
            summary["label_stats"],
            threshold=cfg.high_missingness_threshold,
        )
    )

    return summary


def write_audit_report(summary: Dict[str, Any], output_path: Path) -> None:
    """Write audit summary to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def _sample_table(table: pa.Table, sample_size: Optional[int], seed: int) -> pa.Table:
    if sample_size is None or sample_size <= 0 or sample_size >= table.num_rows:
        return table

    rng = np.random.default_rng(seed)
    indices = rng.choice(table.num_rows, size=int(sample_size), replace=False)
    return table.take(pa.array(indices, type=pa.int64()))


def _compute_id_stats(table: pa.Table) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}

    for col in ("post_id", "author_id"):
        if col not in table.column_names:
            continue
        distinct = _count_distinct_safe(table[col])
        stats[f"{col}_distinct"] = int(distinct)

    if "post_id" in table.column_names:
        distinct = stats.get("post_id_distinct", 0)
        stats["post_id_duplicates"] = int(table.num_rows - distinct)

    return stats


def _compute_text_stats(table: pa.Table, text_fields: Iterable[str]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    for field in text_fields:
        if field not in table.column_names:
            continue

        values = table[field].to_pylist()
        lengths = [len(v) for v in values if isinstance(v, str) and v]
        if not lengths:
            stats[field] = {"count": 0}
            continue

        lengths_arr = np.array(lengths)
        stats[field] = {
            "count": int(len(lengths)),
            "mean_chars": float(lengths_arr.mean()),
            "p50_chars": float(np.percentile(lengths_arr, 50)),
            "p95_chars": float(np.percentile(lengths_arr, 95)),
            "max_chars": int(lengths_arr.max()),
        }

        if field == "post_masked":
            mask_counts = [len(_MASK_PATTERN.findall(v)) for v in values if isinstance(v, str)]
            if mask_counts:
                mask_arr = np.array(mask_counts)
                stats[field]["mask_rows_pct"] = float(
                    np.count_nonzero(mask_arr) / len(mask_arr)
                )
                stats[field]["mask_tokens_mean"] = float(mask_arr.mean())
                stats[field]["mask_tokens_p95"] = float(np.percentile(mask_arr, 95))

    if "text_length" in table.column_names and "post" in table.column_names:
        cached_lengths = table["text_length"].to_pylist()
        post_values = table["post"].to_pylist()
        mismatch = 0
        checked = 0
        for cached, text in zip(cached_lengths, post_values):
            if text is None:
                continue
            if cached is None:
                mismatch += 1
                checked += 1
                continue
            checked += 1
            if len(text) != int(cached):
                mismatch += 1
        stats["text_length_mismatch_pct"] = float(mismatch / checked) if checked else 0.0

    return stats


def _compute_label_stats(
    table: pa.Table,
    demographic_columns: Iterable[str],
    max_values: int,
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    for col in demographic_columns:
        if col not in table.column_names:
            continue

        arr = table[col]
        null_count = arr.null_count
        non_null = table.num_rows - null_count

        column_stats: Dict[str, Any] = {
            "nulls": int(null_count),
            "non_nulls": int(non_null),
            "null_pct": float(null_count / table.num_rows) if table.num_rows else 0.0,
        }

        if non_null > 0:
            try:
                column_stats["distinct_non_null"] = int(
                    _count_distinct_safe(arr, skip_nulls=True)
                )
            except Exception:
                column_stats["distinct_non_null"] = None

            try:
                series = arr.to_pandas()
                counts = series.value_counts(dropna=True).head(max_values)
                column_stats["top_values"] = {
                    str(idx): int(val) for idx, val in counts.items()
                }
            except Exception:
                column_stats["top_values"] = None

            if pa.types.is_integer(arr.type):
                try:
                    min_max = pc.min_max(arr)
                    column_stats["min"] = int(min_max["min"].as_py())
                    column_stats["max"] = int(min_max["max"].as_py())
                except Exception:
                    column_stats["min"] = None
                    column_stats["max"] = None

        stats[col] = column_stats

    return stats


def _count_distinct_safe(arr: pa.Array, skip_nulls: bool = True) -> int:
    """Count distinct values across Arrow types, including dictionary-encoded arrays."""
    try:
        return int(pc.count_distinct(arr, skip_nulls=skip_nulls).as_py())
    except Exception:
        try:
            casted = pc.cast(arr, pa.string())
            return int(pc.count_distinct(casted, skip_nulls=skip_nulls).as_py())
        except Exception:
            series = arr.to_pandas()
            return int(series.nunique(dropna=skip_nulls))


def _missingness_warnings(
    label_stats: Dict[str, Any],
    threshold: float,
) -> List[str]:
    warnings: List[str] = []
    for col, info in label_stats.items():
        if info.get("null_pct", 0.0) >= threshold:
            warnings.append(
                f"High missingness in '{col}': {info.get('null_pct', 0.0):.2%}"
            )
    return warnings
