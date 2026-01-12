"""Affine Guard layer for Phase D projection enforcement."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class AffineGuard(nn.Module):
    """
    Frozen projection layer applying the Phase A LEACE matrix.

    Notes:
        - The projection matrix is registered as a buffer (no gradient updates).
        - Forward uses h @ P.T to align with Phase A visualization usage.
    """

    def __init__(self, projection_matrix: torch.Tensor, freeze: bool = True) -> None:
        super().__init__()

        if projection_matrix.ndim != 2 or projection_matrix.shape[0] != projection_matrix.shape[1]:
            raise ValueError(
                "projection_matrix must be square with shape [hidden_dim, hidden_dim]"
            )

        self.hidden_dim = int(projection_matrix.shape[0])
        self.frozen = bool(freeze)

        if freeze:
            self.register_buffer("P", projection_matrix)
        else:
            self.P = nn.Parameter(projection_matrix)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: Optional[torch.device | str] = None,
        freeze: bool = True,
    ) -> "AffineGuard":
        projection_matrix = torch.load(path, map_location=device or "cpu")
        if not isinstance(projection_matrix, torch.Tensor):
            raise TypeError("Projection matrix checkpoint did not contain a torch.Tensor")
        return cls(projection_matrix=projection_matrix, freeze=freeze)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Hidden dimension mismatch: expected {self.hidden_dim}, "
                f"got {hidden_states.shape[-1]}"
            )

        P = self.P.to(dtype=hidden_states.dtype)
        # hidden_states: [..., D], P: [D, D] -> [..., D]
        return torch.matmul(hidden_states, P.transpose(0, 1))

    def extra_repr(self) -> str:
        return f"hidden_dim={self.hidden_dim}, frozen={self.frozen}"
