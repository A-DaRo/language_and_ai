"""Phase D Arrow dataset loading and collation.

Provides two collation strategies:
1. PhaseDCollator: JIT tokenization (legacy, CPU-bound)
2. FastCollator: AOT tensor stacking (optimized, zero tokenization in hot loop)

And async prefetching for GPU pipeline parallelism:
- DevicePrefetcher: Overlaps H2D transfers with GPU compute via CUDA streams
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
import random
from typing import Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import torch
from torch.utils.data import Dataset, Sampler, DataLoader

from ..data_engine.schemas import get_demographic_columns
from ..data_engine.bucketing import snap_to_grid, DEFAULT_QUANTIZE_STEP
from .tokenizer import PhaseDTokenizer

logger = logging.getLogger(__name__)

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
    """
    Arrow-backed dataset for Phase D training.
    
    Supports two modes:
    1. JIT mode: Returns text for tokenization in collator (legacy)
    2. AOT mode: Returns pre-tokenized tensors for FastCollator (optimized)
    
    AOT mode is auto-detected when 'input_ids' column exists in the Arrow file.
    """

    def __init__(
        self,
        arrow_path: str | Path,
        *,
        text_field: str,
        label_fields: Optional[Iterable[str]] = None,
        split: Optional[str] = None,
        label_maps: Optional[PhaseDLabelMaps] = None,
        split_seed: int = 42,
        split_ratios: Optional[Dict[str, float]] = None,
        use_aot_tokens: Optional[bool] = None,  # Auto-detect if None
    ) -> None:
        self.arrow_path = Path(arrow_path)
        self.text_field = text_field
        self.label_fields = list(label_fields or get_demographic_columns())

        table = feather.read_table(self.arrow_path, memory_map=True)
        if split is not None:
            # Check if split column exists and has valid values
            has_valid_split = False
            if "split" in table.column_names:
                split_values = table["split"]
                non_null_count = pc.sum(pc.is_valid(split_values)).as_py()
                has_valid_split = non_null_count > 0

            if has_valid_split:
                mask = pc.equal(table["split"], split)
                table = table.filter(mask)
            elif "author_id" in table.column_names:
                ratios = split_ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
                table = _filter_by_author_split(
                    table,
                    split=split,
                    seed=split_seed,
                    ratios=ratios,
                )
        self.table = table

        if self.text_field not in self.table.column_names:
            raise ValueError(f"Missing text field '{self.text_field}' in dataset")

        for field in self.label_fields:
            if field not in self.table.column_names:
                raise ValueError(f"Missing label field '{field}' in dataset")

        if label_maps is None:
            self.label_maps = load_label_maps(self.table, self.label_fields)
        else:
            self.label_maps = label_maps

        self._text_col = self.table[self.text_field]
        self._label_cols = {field: self.table[field] for field in self.label_fields}
        self._length_cols = {
            name: self.table[name]
            for name in ("text_length", "token_length", "masked_text_length", "token_count")
            if name in self.table.column_names
        }
        
        # AOT mode: pre-tokenized data available
        # AOT is only valid if the tokenized field matches the requested text_field
        has_aot_columns = "input_ids" in self.table.column_names
        aot_field_matches = False
        tokenized_field = None
        
        if has_aot_columns and "tokenized_text_field" in self.table.column_names:
            # Check if at least one row has matching tokenized field
            tokenized_field = self.table["tokenized_text_field"][0].as_py()
            aot_field_matches = (tokenized_field == text_field)
        elif has_aot_columns:
            # Legacy: no tokenized_text_field column, assume match for backward compat
            aot_field_matches = True
        
        if use_aot_tokens is None:
            # Auto-detect: enable AOT only if columns exist AND field matches
            self._aot_mode = has_aot_columns and aot_field_matches
            if has_aot_columns and not aot_field_matches:
                logger.debug(
                    f"AOT columns found but tokenized field mismatch: "
                    f"requested '{text_field}', dataset has '{tokenized_field}'. "
                    f"Falling back to JIT tokenization."
                )
        else:
            self._aot_mode = use_aot_tokens
        
        if self._aot_mode:
            if "input_ids" not in self.table.column_names:
                raise ValueError(
                    "AOT mode enabled but 'input_ids' column not found. "
                    "Run preprocess_tokens.py first."
                )
            self._input_ids_col = self.table["input_ids"]
            self._attention_mask_col = self.table["attention_mask"]
            # Prefer token_count for length if available (exact token length)
            if "token_count" in self._length_cols:
                self._token_count_col = self._length_cols["token_count"]
            else:
                self._token_count_col = None

    @property
    def is_aot_mode(self) -> bool:
        """Whether dataset provides pre-tokenized data."""
        return self._aot_mode

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
        """
        Get a single sample.
        
        Returns different structure based on mode:
        - JIT mode: {"text": str, "labels": dict, "post_id": str}
        - AOT mode: {"input_ids": array, "attention_mask": array, "labels": dict, ...}
        """
        labels: Dict[str, int] = {}
        for field, col in self._label_cols.items():
            value = col[index].as_py()
            labels[field] = self._encode_label(field, value)
        
        post_id = (
            self.table["post_id"][index].as_py()
            if "post_id" in self.table.column_names
            else None
        )
        
        if self._aot_mode:
            # AOT mode: return pre-tokenized tensors
            input_ids = self._input_ids_col[index].as_py()
            attention_mask = self._attention_mask_col[index].as_py()
            
            # Convert to numpy arrays for efficient collation
            input_ids = np.array(input_ids, dtype=np.uint16)
            attention_mask = np.array(attention_mask, dtype=np.uint8)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "post_id": post_id,
            }
        else:
            # JIT mode: return text for tokenization in collator
            text = self._text_col[index].as_py()
            return {"text": text or "", "labels": labels, "post_id": post_id}

    def get_text(self, index: int) -> str:
        """Get raw text for a sample (always available)."""
        text = self._text_col[index].as_py()
        return text or ""

    def get_length(self, index: int, column: str = "text_length") -> Optional[int]:
        """Get cached length for a sample."""
        if column not in self._length_cols:
            return None
        value = self._length_cols[column][index].as_py()
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    
    def get_token_count(self, index: int) -> Optional[int]:
        """
        Get exact token count for a sample (AOT mode only).
        
        Returns None if token_count column not available.
        """
        if not self._aot_mode or self._token_count_col is None:
            return None
        value = self._token_count_col[index].as_py()
        return int(value) if value is not None else None
    
    def get_all_token_counts(self) -> Optional[np.ndarray]:
        """
        Get all token counts as numpy array (for sampler initialization).
        
        Returns None if token_count column not available.
        """
        if not self._aot_mode or "token_count" not in self._length_cols:
            return None
        return np.array(
            self._length_cols["token_count"].to_pylist(),
            dtype=np.int32,
        )


class PhaseDCollator:
    """Tokenizing collator for Phase D (legacy JIT mode)."""

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


# ==============================================================================
# FastCollator: Zero-tokenization AOT collator for high-throughput training
# ==============================================================================

class FastCollator:
    """
    High-throughput collator for pre-tokenized (AOT) datasets.
    
    This collator performs NO tokenization - it simply stacks pre-computed
    tensors from the Arrow dataset. This removes Python from the critical
    path and enables true GPU saturation.
    
    Features:
        - Zero tokenization overhead in training loop
        - Quantized padding for CUDA Graph stability
        - Pinned memory allocation for async H2D transfers
        - Direct numpy -> torch conversion (no Python list intermediates)
    
    Requirements:
        Dataset must have pre-computed columns:
        - input_ids: List[uint16] (from preprocess_tokens.py)
        - attention_mask: List[uint8]
        - token_count: int16
    
    Reference: Phase D Final Optimization Blueprint Section 2.2
    """
    
    def __init__(
        self,
        max_length: int = 512,
        pad_token_id: int = 1,  # RoBERTa pad token
        quantize_step: int = DEFAULT_QUANTIZE_STEP,
        use_pinned_memory: bool = True,
    ) -> None:
        """
        Initialize fast collator.
        
        Args:
            max_length: Maximum sequence length (for safety clamping).
            pad_token_id: Token ID to use for padding.
            quantize_step: Snap-to-Grid step for stable tensor shapes.
            use_pinned_memory: Whether to use pinned memory for async transfers.
        """
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.quantize_step = quantize_step
        self.use_pinned_memory = use_pinned_memory and torch.cuda.is_available()
    
    def __call__(
        self,
        batch: List[Dict],
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], List]]:
        """
        Collate pre-tokenized samples with quantized padding.
        
        Args:
            batch: List of sample dicts with 'input_ids', 'attention_mask',
                   'labels', and optionally 'token_count'.
        
        Returns:
            Collated batch dict with stacked tensors.
        """
        batch_size = len(batch)
        
        # Extract pre-tokenized sequences
        input_ids_list = []
        attention_mask_list = []
        lengths = []
        
        for sample in batch:
            # Get pre-computed tokens (already numpy arrays or lists)
            ids = sample.get("input_ids")
            mask = sample.get("attention_mask")
            
            if ids is None or mask is None:
                raise ValueError(
                    "FastCollator requires pre-tokenized data. "
                    "Run preprocess_tokens.py first or use PhaseDCollator."
                )
            
            # Convert to numpy if needed
            if not isinstance(ids, np.ndarray):
                ids = np.array(ids, dtype=np.int64)
            else:
                ids = ids.astype(np.int64)
            
            if not isinstance(mask, np.ndarray):
                mask = np.array(mask, dtype=np.int64)
            else:
                mask = mask.astype(np.int64)
            
            input_ids_list.append(ids)
            attention_mask_list.append(mask)
            lengths.append(len(ids))
        
        # Compute quantized max length (Snap-to-Grid)
        max_len_in_batch = max(lengths)
        padded_length = min(
            snap_to_grid(max_len_in_batch, self.quantize_step),
            self.max_length
        )
        
        # Pre-allocate tensors with pinned memory for async H2D
        if self.use_pinned_memory:
            input_ids = torch.zeros(
                (batch_size, padded_length),
                dtype=torch.long,
                pin_memory=True,
            )
            attention_mask = torch.zeros(
                (batch_size, padded_length),
                dtype=torch.long,
                pin_memory=True,
            )
        else:
            input_ids = torch.zeros((batch_size, padded_length), dtype=torch.long)
            attention_mask = torch.zeros((batch_size, padded_length), dtype=torch.long)
        
        # Fill tensors (pad on right with pad_token_id / 0)
        for i, (ids, mask) in enumerate(zip(input_ids_list, attention_mask_list)):
            seq_len = min(len(ids), padded_length)
            input_ids[i, :seq_len] = torch.from_numpy(ids[:seq_len])
            attention_mask[i, :seq_len] = torch.from_numpy(mask[:seq_len])
            # Padding positions already 0 from zeros initialization
        
        # Collate labels
        labels: Dict[str, torch.Tensor] = {}
        if batch and "labels" in batch[0]:
            label_keys = batch[0]["labels"].keys()
            for field in label_keys:
                label_values = [sample["labels"][field] for sample in batch]
                if self.use_pinned_memory:
                    labels[field] = torch.tensor(
                        label_values, dtype=torch.long, pin_memory=True
                    )
                else:
                    labels[field] = torch.tensor(label_values, dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "post_id": [sample.get("post_id") for sample in batch],
            "padded_length": padded_length,  # For telemetry
        }


# ==============================================================================
# DevicePrefetcher: Async H2D pipeline for GPU saturation
# ==============================================================================

class DevicePrefetcher:
    """
    Asynchronous Host-to-Device prefetcher using CUDA streams.
    
    This class wraps a DataLoader iterator and overlaps the H2D memory
    transfer of the next batch with the GPU computation of the current
    batch. This eliminates transfer latency from the critical path.
    
    Architecture:
        - Uses a dedicated CUDA stream for async transfers
        - Prefetches one batch ahead
        - Waits on stream synchronization only when yielding
    
    Usage:
        loader = DataLoader(dataset, collate_fn=FastCollator(...))
        prefetcher = DevicePrefetcher(loader, device)
        
        for batch in prefetcher:
            # batch is already on GPU, transfer overlapped with prev compute
            outputs = model(batch["input_ids"], batch["attention_mask"])
    
    Reference: Phase D Final Optimization Blueprint Section 2.2
    """
    
    def __init__(
        self,
        loader: DataLoader,
        device: torch.device,
        non_blocking: bool = True,
    ) -> None:
        """
        Initialize device prefetcher.
        
        Args:
            loader: PyTorch DataLoader to wrap.
            device: Target device (should be CUDA).
            non_blocking: Whether to use non-blocking transfers.
        """
        self.loader = loader
        self.device = device
        self.non_blocking = non_blocking
        
        self._stream: Optional[torch.cuda.Stream] = None
        self._iterator: Optional[Iterable] = None
        self._next_batch: Optional[Dict] = None
        
        # Only use async prefetching on CUDA
        self._use_async = device.type == "cuda" and torch.cuda.is_available()
        
        if self._use_async:
            self._stream = torch.cuda.Stream(device=device)
    
    def __iter__(self) -> "DevicePrefetcher":
        """Start iteration with prefetching."""
        self._iterator = iter(self.loader)
        self._preload()
        return self
    
    def __next__(self) -> Dict[str, Union[torch.Tensor, Dict, List]]:
        """
        Get next batch (already on device).
        
        Returns:
            Batch dict with tensors on target device.
        """
        if self._use_async and self._stream is not None:
            # Wait for the prefetch transfer to complete
            torch.cuda.current_stream(self.device).wait_stream(self._stream)
        
        batch = self._next_batch
        
        if batch is None:
            raise StopIteration
        
        # Start prefetching next batch
        self._preload()
        
        return batch
    
    def _preload(self) -> None:
        """Prefetch next batch to device asynchronously."""
        try:
            batch = next(self._iterator)  # type: ignore
        except StopIteration:
            self._next_batch = None
            return
        
        if self._use_async and self._stream is not None:
            # Transfer on dedicated stream (non-blocking)
            with torch.cuda.stream(self._stream):
                self._next_batch = self._to_device(batch)
        else:
            # Synchronous transfer (CPU or no CUDA)
            self._next_batch = self._to_device(batch)
    
    def _to_device(
        self,
        batch: Dict,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], List]]:
        """Move batch tensors to device."""
        result = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(
                    self.device,
                    non_blocking=self.non_blocking,
                )
            elif isinstance(value, dict):
                # Handle nested dicts (e.g., labels)
                result[key] = {
                    k: v.to(self.device, non_blocking=self.non_blocking)
                    if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                }
            else:
                # Pass through non-tensor values (e.g., post_id list)
                result[key] = value
        
        return result
    
    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.loader)


class PhaseDBudgetedBatchSampler(Sampler[List[int]]):
    """Batch sampler that respects a dynamic token budget."""

    def __init__(
        self,
        dataset: PhaseDDataset,
        *,
        length_fn: Callable[[int], int],
        budget_provider: Callable[[], int],
        max_batch_size: int,
        min_batch_size: int = 1,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.length_fn = length_fn
        self.budget_provider = budget_provider
        self.max_batch_size = max(1, int(max_batch_size))
        self.min_batch_size = max(1, int(min_batch_size))
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0
        self._length_cache = [-1] * len(dataset)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return max(1, (len(self.dataset) + self.max_batch_size - 1) // self.max_batch_size)

    def _get_length(self, index: int) -> int:
        cached = self._length_cache[index]
        if cached >= 0:
            return cached
        length = max(1, int(self.length_fn(index)))
        self._length_cache[index] = length
        return length

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng = random.Random(self.seed + self._epoch)
            rng.shuffle(indices)

        batch: List[int] = []
        budget = max(1, int(self.budget_provider()))
        total_tokens = 0

        for idx in indices:
            length = self._get_length(idx)

            if batch and (total_tokens + length > budget or len(batch) >= self.max_batch_size):
                if len(batch) >= self.min_batch_size or not self.drop_last:
                    yield batch
                batch = []
                budget = max(1, int(self.budget_provider()))
                total_tokens = 0

            batch.append(idx)
            total_tokens += length

        if batch and (len(batch) >= self.min_batch_size or not self.drop_last):
            yield batch


def load_label_maps(
    table_or_path: pa.Table | str | Path,
    label_fields: Iterable[str],
) -> PhaseDLabelMaps:
    if isinstance(table_or_path, pa.Table):
        table = table_or_path
    else:
        table = feather.read_table(Path(table_or_path), memory_map=True)
    maps: Dict[str, Dict] = {}
    for field in label_fields:
        maps[field] = _build_label_map(table[field])
    return PhaseDLabelMaps(maps=maps)


def _author_bucket(author_id: str, seed: int) -> int:
    payload = f"{seed}:{author_id}".encode("utf-8")
    digest = hashlib.md5(payload).hexdigest()
    return int(digest, 16) % 100


def _filter_by_author_split(
    table: pa.Table,
    *,
    split: str,
    seed: int,
    ratios: Dict[str, float],
) -> pa.Table:
    if split not in ratios:
        raise ValueError(f"Unknown split '{split}' for ratios {ratios}")

    train_cut = int(ratios.get("train", 0.8) * 100)
    val_cut = train_cut + int(ratios.get("val", 0.1) * 100)

    author_ids = table["author_id"].to_pylist()
    buckets = [_author_bucket(str(author_id), seed) for author_id in author_ids]

    if split == "train":
        mask = [bucket < train_cut for bucket in buckets]
    elif split == "val":
        mask = [train_cut <= bucket < val_cut for bucket in buckets]
    else:
        mask = [bucket >= val_cut for bucket in buckets]

    return table.filter(pa.array(mask))
