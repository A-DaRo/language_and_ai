"""Stylometric Validity Score (SVS) computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import PreTrainedTokenizer


DEFAULT_FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that",
    "to", "of", "in", "on", "at", "by", "for", "with", "from", "as",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "have", "has", "had", "will", "would", "shall", "should", "can", "could",
    "may", "might", "must", "this", "these", "those", "here", "there",
    "not", "no", "nor", "so", "because", "while", "when", "where", "which",
}


@dataclass
class SVSResult:
    svs: float
    function_mass: float
    content_mass: float


class SVSCalculator:
    """Compute function-word vs content-word attention ratios."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        *,
        function_words: Optional[Iterable[str]] = None,
        exclude_tokens: Optional[Iterable[str]] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.function_words = set(function_words or DEFAULT_FUNCTION_WORDS)
        if exclude_tokens:
            self.function_words -= set(exclude_tokens)

    def compute_svs(
        self,
        *,
        attentions: Sequence[torch.Tensor],
        input_ids: torch.Tensor,
        facilitating_heads: Sequence[Tuple[int, int]],
        attention_mask: Optional[torch.Tensor] = None,
        query_position: int = 0,
    ) -> SVSResult:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        tokens = [
            self.tokenizer.convert_ids_to_tokens(row.tolist())
            for row in input_ids.cpu()
        ]

        function_mass = 0.0
        content_mass = 0.0

        key_masks = attention_mask.bool().cpu()

        for layer_idx, head_idx in facilitating_heads:
            if layer_idx >= len(attentions):
                continue
            layer_attn = attentions[layer_idx]  # [B, H, S, S]
            if head_idx >= layer_attn.shape[1]:
                continue
            head_attn = layer_attn[:, head_idx, :, :]  # [B, S, S]

            for batch_idx, token_row in enumerate(tokens):
                key_mask = key_masks[batch_idx]
                attn_row = head_attn[batch_idx]
                if query_position >= attn_row.shape[0]:
                    continue

                attn_weights = attn_row[query_position].detach().cpu()
                for pos, token in enumerate(token_row):
                    if not bool(key_mask[pos]):
                        continue
                    norm = self._normalize_token(token)
                    if not norm:
                        continue
                    weight = float(attn_weights[pos].item())
                    if norm in self.function_words:
                        function_mass += weight
                    else:
                        content_mass += weight

        svs = function_mass / content_mass if content_mass > 0 else 0.0
        return SVSResult(svs=svs, function_mass=function_mass, content_mass=content_mass)

    @staticmethod
    def _normalize_token(token: str) -> str:
        token = token.lstrip("Ġ▁")
        token = token.strip(".,!?;:\"'`-()[]{}")
        token = token.lower()
        if not token.isalpha():
            return ""
        return token
