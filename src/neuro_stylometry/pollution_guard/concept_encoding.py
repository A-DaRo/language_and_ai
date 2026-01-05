"""Demographic concept encoding for multi-label LEACE.

This module builds a consistent, global concept design matrix Z from all
SOBR demographic columns.

- Numeric/binary columns: standardized (z-score) with missing indicator.
- Categorical string columns: one-hot with missing indicator.

The intent is to provide a stable, batch-consistent encoding for streaming
(Laptop) strategies where Z must have fixed dimensionality across batches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa

from ..data_engine.schemas import get_demographic_columns


def _is_missing_numeric(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"i", "u"}:
        # Some upstream code may use -1 as a sentinel for missing binary labels.
        return values < 0
    return np.isnan(values)


def _safe_float(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"i", "u"}:
        return values.astype(np.float32)
    return values.astype(np.float32, copy=False)


@dataclass(frozen=True)
class NumericFeatureSpec:
    mean: float
    std: float


@dataclass(frozen=True)
class CategoricalFeatureSpec:
    categories: Tuple[str, ...]
    index: Dict[str, int]


class DemographicEncoder:
    """Fit-once, transform-many encoder for demographic concept matrices."""

    def __init__(
        self,
        demographic_columns: Optional[List[str]] = None,
    ) -> None:
        self.demographic_columns = demographic_columns or get_demographic_columns()
        self.numeric_specs: Dict[str, NumericFeatureSpec] = {}
        self.categorical_specs: Dict[str, CategoricalFeatureSpec] = {}
        self._feature_names: List[str] = []

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    @property
    def feature_dim(self) -> int:
        return len(self._feature_names)

    def fit(self, table: pa.Table) -> "DemographicEncoder":
        """Fit global encoding parameters from a PyArrow table."""
        feature_names: List[str] = []

        for col in self.demographic_columns:
            if col not in table.column_names:
                raise KeyError(f"Missing demographic column '{col}' in table")

            arr = table[col]
            # Treat dictionary-encoded string as categorical.
            if pa.types.is_string(arr.type) or pa.types.is_dictionary(arr.type):
                series = arr.to_pandas()
                # Normalize missing to NaN; keep strings as str.
                non_null = series.dropna().astype(str)
                cats = tuple(sorted(non_null.unique().tolist()))
                index = {c: i for i, c in enumerate(cats)}
                self.categorical_specs[col] = CategoricalFeatureSpec(categories=cats, index=index)
                for c in cats:
                    feature_names.append(f"{col}__{c}")
                feature_names.append(f"{col}__MISSING")
            else:
                series = arr.to_pandas()
                values = series.to_numpy()
                values_f = _safe_float(values)
                missing = _is_missing_numeric(values_f)

                observed = values_f[~missing]
                if observed.size == 0:
                    mean = 0.0
                    std = 1.0
                else:
                    mean = float(observed.mean())
                    std = float(observed.std())
                    if not np.isfinite(std) or std <= 1e-12:
                        std = 1.0

                self.numeric_specs[col] = NumericFeatureSpec(mean=mean, std=std)
                feature_names.append(f"{col}__VALUE")
                feature_names.append(f"{col}__MISSING")

        self._feature_names = feature_names
        return self

    def transform(self, table: pa.Table) -> np.ndarray:
        """Transform a table slice into a dense concept matrix (n, k)."""
        if not self._feature_names:
            raise RuntimeError("DemographicEncoder must be fit() before transform()")

        n = len(table)
        k = self.feature_dim
        Z = np.zeros((n, k), dtype=np.float32)

        offset = 0
        for col in self.demographic_columns:
            if col in self.categorical_specs:
                spec = self.categorical_specs[col]
                series = table[col].to_pandas()
                # Fill missing indicator.
                missing = series.isna().to_numpy()

                # One-hot observed.
                if len(spec.categories) > 0:
                    vals = series.fillna("").astype(str).to_numpy()
                    for i, v in enumerate(vals):
                        if missing[i]:
                            continue
                        j = spec.index.get(v)
                        if j is None:
                            continue
                        Z[i, offset + j] = 1.0

                Z[:, offset + len(spec.categories)] = missing.astype(np.float32)
                offset += len(spec.categories) + 1
            else:
                spec = self.numeric_specs[col]
                series = table[col].to_pandas()
                values_f = _safe_float(series.to_numpy())
                missing = _is_missing_numeric(values_f)

                standardized = (values_f - spec.mean) / spec.std
                standardized = np.where(missing, 0.0, standardized).astype(np.float32)

                Z[:, offset] = standardized
                Z[:, offset + 1] = missing.astype(np.float32)
                offset += 2

        return Z


def extract_probe_labels(table: pa.Table, column: str) -> np.ndarray:
    """Extract a deterministic 1D label vector for probing.

    Returns int64 labels with -1 for missing.
    - Numeric: cast to int, missing if NULL/NaN or <0.
    - String: stable factorization using sorted unique categories.
    """

    if column not in table.column_names:
        raise KeyError(f"Missing column '{column}'")

    series = table[column].to_pandas()
    if series.dtype == object:
        non_null = series.dropna().astype(str)
        cats = sorted(non_null.unique().tolist())
        index = {c: i for i, c in enumerate(cats)}

        labels = np.full(len(series), -1, dtype=np.int64)
        for i, v in enumerate(series):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            j = index.get(str(v))
            if j is None:
                continue
            labels[i] = j
        return labels

    values = series.to_numpy()
    values_f = _safe_float(values)
    missing = _is_missing_numeric(values_f)
    labels = np.full(len(values_f), -1, dtype=np.int64)
    if (~missing).any():
        labels[~missing] = values_f[~missing].astype(np.int64)
    return labels
