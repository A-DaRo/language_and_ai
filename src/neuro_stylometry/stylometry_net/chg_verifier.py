"""Causal Head Gating (CHG) verification utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class HeadGates:
    """Learned gate values per attention head."""

    gates: torch.Tensor  # [num_layers, num_heads]
    layer_gate_means: torch.Tensor  # [num_layers]

    def classify_heads(
        self,
        facilitating_threshold: float = 0.7,
        irrelevant_threshold: float = 0.3,
    ) -> Dict[str, List[Tuple[int, int]]]:
        facilitating = []
        irrelevant = []
        neutral = []

        num_layers, num_heads = self.gates.shape
        for layer in range(num_layers):
            for head in range(num_heads):
                g = float(self.gates[layer, head].item())
                if g > facilitating_threshold:
                    facilitating.append((layer, head))
                elif g < irrelevant_threshold:
                    irrelevant.append((layer, head))
                else:
                    neutral.append((layer, head))

        return {
            "facilitating": facilitating,
            "irrelevant": irrelevant,
            "neutral": neutral,
        }


class CHGVerifier:
    """
    Learn per-head gates for attention heads.

    This is a lightweight proxy implementation: gates scale loss via mean gate
    value with L1 sparsity regularization. It does not re-run the forward pass
    with gated attention outputs.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_heads: int,
        learning_rate: float = 1e-3,
        num_epochs: int = 10,
        gate_init: float = 0.5,
        regularization: float = 0.01,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.gate_init = gate_init
        self.regularization = regularization

    def learn_gates(
        self,
        *,
        model: nn.Module,
        head: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> HeadGates:
        gates = nn.Parameter(
            torch.full((self.num_layers, self.num_heads), float(self.gate_init), device=device)
        )

        for param in model.parameters():
            param.requires_grad = False
        for param in head.parameters():
            param.requires_grad = False

        model.eval()
        head.eval()

        optimizer = torch.optim.Adam([gates], lr=self.learning_rate)

        for _ in range(self.num_epochs):
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = {k: v.to(device) for k, v in batch["labels"].items()}

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                )
                logits = head(outputs["cls_embedding"])
                base_loss = head.compute_loss(logits, labels)
                if base_loss is None:
                    continue

                gate_values = torch.sigmoid(gates)
                loss = base_loss * gate_values.mean()
                loss = loss + self.regularization * gate_values.abs().mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        final_gates = torch.sigmoid(gates).detach()
        layer_means = final_gates.mean(dim=1)
        return HeadGates(gates=final_gates, layer_gate_means=layer_means)
