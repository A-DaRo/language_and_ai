"""
Explicit Recall computation for GLiNER detections.

Compares detected spans against regex-based reference spans.
Implements: FR-26 (Explicit Recall Validation)
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any
import re


def _collect_reference_spans(
    text: str,
    reference_patterns: Dict[str, List[str]],
) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    for label, patterns in reference_patterns.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                spans.append((match.start(), match.end(), label))
    return spans


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def compute_explicit_recall(
    texts: List[str],
    detected_entities: List[List[Dict[str, Any]]],
    reference_patterns: Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    Compute explicit recall against regex reference patterns.

    Args:
        texts: Raw input texts.
        detected_entities: GLiNER detections (per text) with label/start/end.
        reference_patterns: Mapping of internal label -> list of regex patterns.

    Returns:
        Dict with overall recall and per-label recall.
    """
    total_ref = 0
    matched_ref = 0
    per_label = {}

    for text, entities in zip(texts, detected_entities):
        ref_spans = _collect_reference_spans(text, reference_patterns)
        total_ref += len(ref_spans)

        for start, end, label in ref_spans:
            detected = [
                e for e in entities
                if e.get("label") == label and _overlaps(start, end, e["start"], e["end"])
            ]
            if detected:
                matched_ref += 1
                per_label.setdefault(label, {"matched": 0, "total": 0})
                per_label[label]["matched"] += 1
            per_label.setdefault(label, {"matched": 0, "total": 0})
            per_label[label]["total"] += 1

    overall = (matched_ref / total_ref) if total_ref > 0 else None
    per_label_recall = {
        label: (counts["matched"] / counts["total"]) if counts["total"] > 0 else None
        for label, counts in per_label.items()
    }

    return {
        "overall": overall,
        "total_reference_spans": total_ref,
        "matched_reference_spans": matched_ref,
        "per_label": per_label_recall,
    }
