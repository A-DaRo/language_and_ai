"""Optimizer and scheduler utilities for Phase D training.

Includes support for fused AdamW optimizer which collapses the parameter
update loop into a single CUDA kernel, eliminating Python iteration overhead.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Tuple

import torch
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


def build_param_groups(
    *,
    model: torch.nn.Module,
    head: torch.nn.Module,
    base_lr: float,
    layerwise_lr_decay: float,
) -> List[dict]:
    """Create param groups with optional layer-wise LR decay."""
    layerwise_lr_decay = float(layerwise_lr_decay)
    if layerwise_lr_decay <= 0.0 or layerwise_lr_decay > 1.0:
        layerwise_lr_decay = 1.0

    param_groups: List[dict] = []

    encoder = getattr(model, "model", model)
    encoder_layers = None
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        encoder_layers = list(encoder.encoder.layer)

    if encoder_layers:
        num_layers = len(encoder_layers)
        for idx, layer in enumerate(encoder_layers):
            lr = base_lr * (layerwise_lr_decay ** (num_layers - 1 - idx))
            param_groups.append(
                {"params": [p for p in layer.parameters() if p.requires_grad], "lr": lr}
            )

        embeddings = getattr(encoder, "embeddings", None)
        if embeddings is not None:
            emb_lr = base_lr * (layerwise_lr_decay ** num_layers)
            param_groups.append(
                {"params": [p for p in embeddings.parameters() if p.requires_grad], "lr": emb_lr}
            )
    else:
        param_groups.append(
            {"params": [p for p in encoder.parameters() if p.requires_grad], "lr": base_lr}
        )

    param_groups.append(
        {"params": [p for p in head.parameters() if p.requires_grad], "lr": base_lr}
    )
    return param_groups


def _supports_fused_adamw() -> bool:
    """Check if fused AdamW is available (PyTorch 2.0+ with CUDA)."""
    if not torch.cuda.is_available():
        return False
    # fused=True requires CUDA and PyTorch 2.0+
    try:
        # Check PyTorch version supports fused
        import inspect
        sig = inspect.signature(torch.optim.AdamW.__init__)
        return "fused" in sig.parameters
    except Exception:
        return False


def build_optimizer(
    *,
    param_groups: Iterable[dict],
    base_lr: float,
    weight_decay: float = 0.01,
    use_fused: bool = True,
) -> torch.optim.Optimizer:
    """
    Build AdamW optimizer with optional fused kernel.
    
    Fused AdamW collapses the parameter update loop into a single CUDA
    kernel launch, eliminating Python iteration overhead over model
    parameters. This can provide significant speedups on large models.
    
    Args:
        param_groups: Parameter groups with learning rates.
        base_lr: Base learning rate.
        weight_decay: L2 regularization weight.
        use_fused: Whether to use fused kernel (requires CUDA + PyTorch 2.0+).
        
    Returns:
        Configured AdamW optimizer.
    """
    fused = use_fused and _supports_fused_adamw()
    
    if fused:
        logger.info("Using fused AdamW optimizer (single CUDA kernel)")
    else:
        if use_fused:
            logger.debug("Fused AdamW not available, using standard optimizer")
    
    return torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        weight_decay=weight_decay,
        fused=fused,
    )


def build_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    num_warmup_steps: int,
    total_training_steps: int,
):
    name = scheduler_name.lower().strip()
    if total_training_steps <= 0:
        return None
    if name == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_training_steps,
        )
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps,
    )
