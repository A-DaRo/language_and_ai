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
        self.task_order = tuple(num_labels_per_task.keys())

    def forward(self, cls_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        cls_embedding = self.dropout(cls_embedding)
        return {task: head(cls_embedding) for task, head in self.heads.items()}

    def forward_compiled(self, cls_embedding: torch.Tensor) -> tuple[torch.Tensor, ...]:
        cls_embedding = self.dropout(cls_embedding)
        logits = []
        for task in self.task_order:
            logits.append(self.heads[task](cls_embedding))
        return tuple(logits)

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

    def compute_loss_compiled(
        self,
        logits: tuple[torch.Tensor, ...],
        labels: tuple[torch.Tensor, ...],
        ignore_index: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        first_logits = logits[0]
        total_loss = torch.zeros((), device=first_logits.device, dtype=first_logits.dtype)
        valid_flag = torch.zeros((), device=first_logits.device, dtype=torch.int32)
        for task_logits, task_labels in zip(logits, labels):
            valid_mask = task_labels != ignore_index
            valid_count = valid_mask.sum()
            per_sample = F.cross_entropy(
                task_logits,
                task_labels,
                reduction="none",
                ignore_index=ignore_index,
            )
            loss_sum = (per_sample * valid_mask).sum()
            denom = valid_count.clamp(min=1)
            task_loss = loss_sum / denom
            total_loss = total_loss + task_loss
            valid_flag = valid_flag + (valid_count > 0).to(valid_flag.dtype)

        valid_flag = (valid_flag > 0).to(valid_flag.dtype)
        return total_loss, valid_flag
