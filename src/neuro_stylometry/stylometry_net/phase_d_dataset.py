"""Phase D Arrow dataset loading and collation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import torch
from torch.utils.data import Dataset

from ..data_engine.schemas import get_demographic_columns
from .tokenizer import PhaseDTokenizer


def _unique_values(array: pa.Array) -> List:
    if pa.types.is_dictionary(array.type):
        values = array.dictionary
        return [v.as_py() for v in values]
    unique = pc.unique(array)
    return [v.as_py() for v in unique]


def _build_label_map(array: pa.Array) -> Dict:
    values = [v for v in _unique_values(array) if v is not None]
    # Stable ordering for determinism
    try:
        values = sorted(values)
    except Exception:
        pass
    return {value: idx for idx, value in enumerate(values)}


@dataclass
class PhaseDLabelMaps:
    maps: Dict[str, Dict]

    def num_classes(self) -> Dict[str, int]:
        return {key: len(value) for key, value in self.maps.items()}


class PhaseDDataset(Dataset):
    """Arrow-backed dataset for Phase D."""

    def __init__(
        self,
        arrow_path: str | Path,
        *,
        text_field: str,
        label_fields: Optional[Iterable[str]] = None,
        split: Optional[str] = None,
        label_maps: Optional[PhaseDLabelMaps] = None,
    ) -> None:
        self.arrow_path = Path(arrow_path)
        self.text_field = text_field
        self.label_fields = list(label_fields or get_demographic_columns())

        table = feather.read_table(self.arrow_path, memory_map=True)
        if split is not None and "split" in table.column_names:
            mask = pc.equal(table["split"], split)
            table = table.filter(mask)
        self.table = table

        if self.text_field not in self.table.column_names:
            raise ValueError(f"Missing text field '{self.text_field}' in dataset")

        for field in self.label_fields:
            if field not in self.table.column_names:
                raise ValueError(f"Missing label field '{field}' in dataset")

        if label_maps is None:
            maps: Dict[str, Dict] = {}
            for field in self.label_fields:
                maps[field] = _build_label_map(self.table[field])
            self.label_maps = PhaseDLabelMaps(maps=maps)
        else:
            self.label_maps = label_maps

        self._text_col = self.table[self.text_field]
        self._label_cols = {field: self.table[field] for field in self.label_fields}

    def __len__(self) -> int:
        return self.table.num_rows

    def _encode_label(self, field: str, value) -> int:
        if value is None:
            return -1
        label_map = self.label_maps.maps.get(field, {})
        if value in label_map:
            return label_map[value]
        return -1

    def __getitem__(self, index: int) -> Dict:
        text = self._text_col[index].as_py()
        labels: Dict[str, int] = {}
        for field, col in self._label_cols.items():
            value = col[index].as_py()
            labels[field] = self._encode_label(field, value)
        post_id = self.table["post_id"][index].as_py() if "post_id" in self.table.column_names else None
        return {"text": text or "", "labels": labels, "post_id": post_id}


class PhaseDCollator:
    """Tokenizing collator for Phase D."""

    def __init__(self, tokenizer: PhaseDTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor | Dict]:
        texts = [sample["text"] for sample in batch]
        encoding = self.tokenizer.encode_batch(texts)

        labels: Dict[str, torch.Tensor] = {}
        if batch:
            for field in batch[0]["labels"].keys():
                labels[field] = torch.tensor(
                    [sample["labels"][field] for sample in batch],
                    dtype=torch.long,
                )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
            "post_id": [sample.get("post_id") for sample in batch],
        }
