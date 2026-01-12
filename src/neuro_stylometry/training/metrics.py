"""Phase D evaluation metrics utilities."""

from __future__ import annotations

from typing import Dict, List

import torch


def compute_accuracy(preds: torch.Tensor, labels: torch.Tensor, ignore_index: int = -1) -> float:
    mask = labels != ignore_index
    if mask.sum().item() == 0:
        return 0.0
    correct = (preds[mask] == labels[mask]).float().mean().item()
    return float(correct)


def compute_macro_f1(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int = -1,
) -> float:
    mask = labels != ignore_index
    if mask.sum().item() == 0:
        return 0.0
    preds = preds[mask]
    labels = labels[mask]

    f1s: List[float] = []
    for cls in range(num_classes):
        tp = ((preds == cls) & (labels == cls)).sum().item()
        fp = ((preds == cls) & (labels != cls)).sum().item()
        fn = ((preds != cls) & (labels == cls)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1s.append(float(f1))

    return float(sum(f1s) / len(f1s)) if f1s else 0.0


def compute_task_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int = -1,
) -> Dict[str, float]:
    preds = torch.argmax(logits, dim=-1)
    return {
        "accuracy": compute_accuracy(preds, labels, ignore_index=ignore_index),
        "f1_macro": compute_macro_f1(
            preds,
            labels,
            num_classes=num_classes,
            ignore_index=ignore_index,
        ),
    }
