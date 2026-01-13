"""Phase D tokenizer alignment utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

from omegaconf import OmegaConf
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_TAXONOMY_PATH = "conf/base/gliner_taxonomy.yaml"


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / candidate


def load_mask_tokens_from_taxonomy(taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH) -> List[str]:
    resolved = _resolve_repo_path(taxonomy_path)
    cfg = OmegaConf.to_container(OmegaConf.load(resolved), resolve=True)
    taxonomy = (cfg or {}).get("taxonomy", {})
    column_prompts = taxonomy.get("column_prompts", {})
    mask_tokens: List[str] = []
    for col_cfg in column_prompts.values():
        if isinstance(col_cfg, dict):
            token = col_cfg.get("mask_token")
            if token:
                mask_tokens.append(str(token))
    return list(dict.fromkeys(mask_tokens))


class PhaseDTokenizer:
    """Tokenizer wrapper that aligns typed mask tokens for Phase D."""

    def __init__(
        self,
        model_name: str = "roberta-base",
        max_length: int = 512,
        mask_tokens: Optional[Iterable[str]] = None,
        taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
        enforce_single_token: bool = True,
        legacy: bool = True,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=legacy)

        try:
            self.tokenizer.model_max_length = int(max_length)
        except Exception:
            pass

        if mask_tokens is None:
            mask_tokens = load_mask_tokens_from_taxonomy(taxonomy_path)

        self.mask_tokens = list(dict.fromkeys(mask_tokens or []))
        self.num_added = 0
        if self.mask_tokens:
            self.num_added = self.register_special_tokens(
                self.mask_tokens,
                enforce_single_token=enforce_single_token,
            )

    def register_special_tokens(
        self,
        tokens: Iterable[str],
        *,
        enforce_single_token: bool = True,
    ) -> int:
        unique_tokens = list(dict.fromkeys(tokens))
        num_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": unique_tokens}
        )
        if num_added > 0:
            logger.info(
                "Registered %d Phase D mask tokens: %s",
                num_added,
                unique_tokens[:3],
            )
        if enforce_single_token:
            self._assert_single_token(unique_tokens)
        return num_added

    def _assert_single_token(self, tokens: Iterable[str]) -> None:
        for token in tokens:
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(
                    f"Mask token '{token}' encoded as {len(token_ids)} tokens: {token_ids}"
                )

    def resize_model_embeddings(self, model) -> None:
        if self.num_added > 0:
            model.resize_token_embeddings(len(self.tokenizer))

    def encode_batch(
        self,
        texts: List[str],
        *,
        padding: bool | str = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ):
        return self.tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length,
            return_tensors=return_tensors,
        )

    def estimate_length(self, text: str) -> int:
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=True))
        except Exception:
            return max(1, len(text.split()))
