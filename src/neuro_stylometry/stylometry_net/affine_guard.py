"""Affine Guard layer for Phase D projection enforcement."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AffineGuard(nn.Module):
    """
    Frozen projection layer applying the Phase A LEACE matrix.

    Notes:
        - The projection matrix is registered as a buffer (no gradient updates).
        - Forward uses h @ P.T to align with Phase A visualization usage.
        - Supports torch.compile for fusing matmul operations (~25% overhead reduction).
    """

    def __init__(
        self,
        projection_matrix: torch.Tensor,
        freeze: bool = True,
        use_compile: bool = False,
    ) -> None:
        super().__init__()

        if projection_matrix.ndim != 2 or projection_matrix.shape[0] != projection_matrix.shape[1]:
            raise ValueError(
                "projection_matrix must be square with shape [hidden_dim, hidden_dim]"
            )

        self.hidden_dim = int(projection_matrix.shape[0])
        self.frozen = bool(freeze)
        self._use_compile = bool(use_compile)
        self._compiled_forward: Optional[callable] = None

        if freeze:
            self.register_buffer("P", projection_matrix)
        else:
            self.P = nn.Parameter(projection_matrix)
        
        # Enable torch.compile for matmul fusion (reduces ~25% overhead for constrained runs)
        if self._use_compile and torch.cuda.is_available():
            self._setup_compiled_forward()
    
    def _setup_compiled_forward(self) -> None:
        """Setup compiled forward pass for optimized matmul fusion."""
        try:
            # Compile the projection operation with default mode (stable, ~1.5-2x speedup)
            # Using 'default' mode instead of 'reduce-overhead' for stability
            @torch.compile(mode="default", fullgraph=True)
            def compiled_projection(hidden_states: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
                P_cast = P.to(dtype=hidden_states.dtype)
                return torch.matmul(hidden_states, P_cast.transpose(0, 1))
            
            self._compiled_forward = compiled_projection
            logger.info(
                f"AffineGuard: torch.compile enabled for matmul fusion "
                f"(hidden_dim={self.hidden_dim})"
            )
        except Exception as e:
            logger.warning(f"AffineGuard: torch.compile failed, using eager mode: {e}")
            self._compiled_forward = None

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: Optional[torch.device | str] = None,
        freeze: bool = True,
        use_compile: bool = False,
    ) -> "AffineGuard":
        """Load AffineGuard from a checkpoint file.
        
        Args:
            path: Path to projection matrix checkpoint.
            device: Device to load the tensor to.
            freeze: If True, projection is a frozen buffer (no gradients).
            use_compile: If True, enable torch.compile for matmul fusion.
            
        Returns:
            Initialized AffineGuard instance.
        """
        projection_matrix = torch.load(path, map_location=device or "cpu")
        if not isinstance(projection_matrix, torch.Tensor):
            raise TypeError("Projection matrix checkpoint did not contain a torch.Tensor")
        return cls(projection_matrix=projection_matrix, freeze=freeze, use_compile=use_compile)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply LEACE projection to hidden states.
        
        Args:
            hidden_states: Tensor of shape [..., hidden_dim].
            
        Returns:
            Projected tensor of same shape.
        """
        if hidden_states.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Hidden dimension mismatch: expected {self.hidden_dim}, "
                f"got {hidden_states.shape[-1]}"
            )
        
        # Use compiled forward if available (fused matmul, ~25% faster)
        if self._compiled_forward is not None:
            return self._compiled_forward(hidden_states, self.P)
        
        # Eager fallback
        P = self.P.to(dtype=hidden_states.dtype)
        # hidden_states: [..., D], P: [D, D] -> [..., D]
        return torch.matmul(hidden_states, P.transpose(0, 1))

    def extra_repr(self) -> str:
        compiled_str = ", compiled=True" if self._compiled_forward is not None else ""
        return f"hidden_dim={self.hidden_dim}, frozen={self.frozen}{compiled_str}"
