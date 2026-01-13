"""Stylometric Validity Score (SVS) computation."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

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
        use_pos: bool = True,
        spacy_model: str = "en_core_web_sm",
    ) -> None:
        self.tokenizer = tokenizer
        self.function_words = set(function_words or DEFAULT_FUNCTION_WORDS)
        if exclude_tokens:
            self.function_words -= set(exclude_tokens)
        self.use_pos = bool(use_pos)
        self.spacy_model = spacy_model
        self._nlp = None

        if self.use_pos:
            self._nlp = self._load_spacy_model(spacy_model)
            if self._nlp is None:
                self.use_pos = False

    def compute_svs(
        self,
        *,
        attentions: Sequence[torch.Tensor],
        input_ids: torch.Tensor,
        facilitating_heads: Sequence[Tuple[int, int]],
        attention_mask: Optional[torch.Tensor] = None,
        query_position: int = 0,
        query_strategy: str = "cls",
    ) -> SVSResult:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        tokens = [
            self.tokenizer.convert_ids_to_tokens(row.tolist())
            for row in input_ids.cpu()
        ]
        function_sets = self._build_function_sets(input_ids)

        function_mass = 0.0
        content_mass = 0.0

        key_masks = attention_mask.bool().cpu()

        strategy = query_strategy.lower().strip()
        if strategy not in {"cls", "all_tokens", "mean_tokens"}:
            logger.warning("Unknown SVS query strategy '%s'; defaulting to mean_tokens.", strategy)
            strategy = "mean_tokens"
        for layer_idx, head_idx in facilitating_heads:
            if layer_idx >= len(attentions):
                continue
            layer_attn = attentions[layer_idx]  # [B, H, S, S]
            if head_idx >= layer_attn.shape[1]:
                continue
            head_attn = layer_attn[:, head_idx, :, :]  # [B, S, S]

            for batch_idx, token_row in enumerate(tokens):
                key_mask = key_masks[batch_idx]
                function_words = function_sets[batch_idx]
                attn_row = head_attn[batch_idx]
                if strategy == "all_tokens":
                    query_indices = list(range(attn_row.shape[0]))
                elif strategy == "mean_tokens":
                    query_indices = list(range(attn_row.shape[0]))
                else:
                    if query_position >= attn_row.shape[0]:
                        continue
                    query_indices = [query_position]

                if not query_indices:
                    continue

                for q_idx in query_indices:
                    attn_weights = attn_row[q_idx].detach().cpu()
                    for pos, token in enumerate(token_row):
                        if not bool(key_mask[pos]):
                            continue
                        norm = self._normalize_token(token)
                        if not norm:
                            continue
                        weight = float(attn_weights[pos].item())
                        if strategy == "mean_tokens":
                            weight = weight / float(len(query_indices))
                        if norm in function_words:
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

    def _build_function_sets(self, input_ids: torch.Tensor) -> List[set[str]]:
        decoded = self.tokenizer.batch_decode(
            input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        if not self.use_pos or self._nlp is None:
            return [set(self.function_words) for _ in decoded]

        function_sets: List[set[str]] = []
        for text in decoded:
            words = set(self.function_words)
            try:
                doc = self._nlp(text)
                for token in doc:
                    if token.pos_ in {
                        "ADP",
                        "AUX",
                        "CCONJ",
                        "DET",
                        "PART",
                        "PRON",
                        "SCONJ",
                    }:
                        words.add(token.text.lower())
            except Exception:
                logger.warning("SVS POS tagging failed; falling back to lexical list.")
            function_sets.append(words)
        return function_sets

    @staticmethod
    def _load_spacy_model(model_name: str):
        try:
            import spacy
        except Exception:
            logger.warning("spaCy not available; SVS POS tagging disabled.")
            return None
        try:
            return spacy.load(model_name)
        except Exception:
            logger.warning("spaCy model '%s' not found; SVS POS tagging disabled.", model_name)
            return None
