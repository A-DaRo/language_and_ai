"""Phase D checkpointing utilities."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(
    path: Path,
    *,
    model_state: Dict[str, Any],
    head_state: Dict[str, Any],
    optimizer_state: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model_state,
            "head_state": head_state,
            "optimizer_state": optimizer_state,
            "metadata": metadata,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )


def save_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def config_to_metadata(config) -> Dict[str, Any]:
    if hasattr(config, "__dataclass_fields__"):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    return {"config": str(config)}
