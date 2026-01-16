"""Attention analysis helpers for Phase D verification outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional


def summarize_head_classification(path: Path) -> Dict[str, int]:
    """
    Summarize CHG head classifications.

    Returns a dict with counts for facilitating/irrelevant/neutral.
    """
    path = Path(path)
    if not path.exists():
        return {"facilitating": 0, "irrelevant": 0, "neutral": 0}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    summary = {}
    for key in ("facilitating", "irrelevant", "neutral"):
        summary[key] = len(data.get(key, []))
    return summary


def load_svs_result(path: Path) -> Optional[Dict[str, float]]:
    """Load SVS result JSON if present."""
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        "svs": float(payload.get("svs", 0.0)),
        "function_mass": float(payload.get("function_mass", 0.0)),
        "content_mass": float(payload.get("content_mass", 0.0)),
    }
