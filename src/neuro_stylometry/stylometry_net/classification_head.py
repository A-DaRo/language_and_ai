"""Multi-task and single-task classification heads for Phase D."""

from __future__ import annotations

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleTaskHead(nn.Module):
    """Single-task classification head for filtered label training.
    
    Use this head when training on a single demographic label via --use-only.
    Returns a single logits tensor instead of a dict, simplifying the training loop.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_classes: int,
        task_name: str,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.num_classes = num_classes
        self.task_name = task_name

    @property
    def is_binary(self) -> bool:
        """Whether this is a binary classification task."""
        return self.num_classes == 2

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        """Forward pass returning single logits tensor.
        
        Args:
            cls_embedding: [batch_size, hidden_dim] CLS token embeddings
            
        Returns:
            Logits tensor of shape [batch_size, num_classes]
        """
        cls_embedding = self.dropout(cls_embedding)
        return self.classifier(cls_embedding)

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -1,
    ) -> Optional[torch.Tensor]:
        """Compute cross-entropy loss for single task.
        
        Args:
            logits: [batch_size, num_classes] model predictions
            labels: [batch_size] ground truth labels
            ignore_index: Label value to ignore in loss computation
            
        Returns:
            Scalar loss tensor, or None if all labels are ignored
        """
        loss = F.cross_entropy(logits, labels, ignore_index=ignore_index)
        # Handle all-ignored case (returns nan)
        loss = torch.nan_to_num(loss, nan=0.0)
        return loss if loss.item() != 0.0 else None


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
        # cls_embedding = self.dropout(cls_embedding) # DEBUG: Disable dropout
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

            # Avoid GPU sync: use cross_entropy with ignore_index directly
            # instead of checking valid_mask.sum().item() == 0
            # PyTorch's cross_entropy handles ignore_index efficiently
            task_loss = F.cross_entropy(
                task_logits,
                task_labels,
                ignore_index=ignore_index,
            )
            
            # Skip NaN losses (happens when all labels are ignore_index)
            # if torch.isnan(task_loss):
            #     continue
            task_loss = torch.nan_to_num(task_loss, nan=0.0)
                
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
