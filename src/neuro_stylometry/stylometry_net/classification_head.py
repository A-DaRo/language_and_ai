"""Multi-task classification heads for Phase D."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskHead(nn.Module):
    """Per-task linear heads with masked loss support."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_labels_per_task: Dict[str, int],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict(
            {task: nn.Linear(hidden_dim, n) for task, n in num_labels_per_task.items()}
        )

    def forward(self, cls_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        cls_embedding = self.dropout(cls_embedding)
        return {task: head(cls_embedding) for task, head in self.heads.items()}

    def compute_loss(
        self,
        logits: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]] = None,
        ignore_index: int = -1,
    ) -> Optional[torch.Tensor]:
        if not logits:
            return None
        weights = weights or {task: 1.0 for task in logits.keys()}
        total_loss = None
        for task, task_logits in logits.items():
            task_labels = labels.get(task)
            if task_labels is None:
                continue

            valid_mask = task_labels != ignore_index
            if valid_mask.sum().item() == 0:
                continue

            task_loss = F.cross_entropy(
                task_logits[valid_mask],
                task_labels[valid_mask],
            )
            weighted = weights.get(task, 1.0) * task_loss
            total_loss = weighted if total_loss is None else total_loss + weighted

        return total_loss
