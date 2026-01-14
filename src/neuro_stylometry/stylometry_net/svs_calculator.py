"""Stylometric Validity Score (SVS) computation."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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

    _HF_PIPELINE_CACHE: Dict[tuple, object] = {}
    _NLP_CACHE: Dict[tuple, object] = {}

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        *,
        function_words: Optional[Iterable[str]] = None,
        exclude_tokens: Optional[Iterable[str]] = None,
        use_pos: bool = True,
        pos_backend: str = "spacy",
        spacy_model: str = "en_core_web_sm",
        spacy_use_gpu: bool = False,
        spacy_gpu_id: Optional[int] = 0,
        spacy_batch_size: int = 32,
        spacy_n_process: int = 1,
        spacy_disable: Optional[Iterable[str]] = None,
        hf_model: str = "vblagoje/bert-english-uncased-finetuned-pos",
        hf_device: int = 0,
        hf_batch_size: int = 16,
    ) -> None:
        self.tokenizer = tokenizer
        self.function_words = set(function_words or DEFAULT_FUNCTION_WORDS)
        if exclude_tokens:
            self.function_words -= set(exclude_tokens)
        self.use_pos = bool(use_pos)
        self.pos_backend = (pos_backend or "spacy").lower().strip()
        self.spacy_model = spacy_model
        self.spacy_use_gpu = bool(spacy_use_gpu)
        self.spacy_gpu_id = spacy_gpu_id
        self.spacy_batch_size = int(spacy_batch_size)
        self.spacy_n_process = int(spacy_n_process)
        self.spacy_disable = list(spacy_disable) if spacy_disable else []
        self.hf_model = hf_model
        self.hf_device = int(hf_device)
        self.hf_batch_size = int(hf_batch_size)
        self._nlp = None
        self._hf_pipeline = None

        if self.use_pos and self.pos_backend == "spacy":
            self._nlp = self._load_spacy_model(
                spacy_model,
                use_gpu=self.spacy_use_gpu,
                gpu_id=self.spacy_gpu_id,
                disable=self.spacy_disable,
            )
            if self._nlp is None:
                self.use_pos = False
        elif self.use_pos and self.pos_backend == "hf":
            self._hf_pipeline = self._load_hf_pipeline(
                model_name=self.hf_model,
                device=self.hf_device,
            )
            if self._hf_pipeline is None:
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

        input_ids_cpu = input_ids.cpu()
        tokens = [
            self.tokenizer.convert_ids_to_tokens(row.tolist())
            for row in input_ids_cpu
        ]
        function_sets = self._build_function_sets(input_ids)

        function_mass = 0.0
        content_mass = 0.0

        key_masks = attention_mask.bool().cpu()
        function_mask, content_mask = self._build_token_masks(
            tokens=tokens,
            function_sets=function_sets,
            key_masks=key_masks,
        )

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
            head_attn = layer_attn[:, head_idx, :, :].detach().cpu()  # [B, S, S]
            if strategy == "all_tokens":
                attn_weights = head_attn.sum(dim=1)
            elif strategy == "mean_tokens":
                attn_weights = head_attn.mean(dim=1)
            else:
                if query_position >= head_attn.shape[1]:
                    continue
                attn_weights = head_attn[:, query_position, :]

            function_mass += float((attn_weights * function_mask).sum().item())
            content_mass += float((attn_weights * content_mask).sum().item())

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
        if not self.use_pos or self.pos_backend == "lexical":
            return [set(self.function_words) for _ in decoded]

        if self.pos_backend == "hf":
            return self._build_function_sets_hf(decoded)

        if self.pos_backend == "spacy":
            if self._nlp is None:
                return [set(self.function_words) for _ in decoded]
            function_sets: List[set[str]] = []
            n_process = self.spacy_n_process
            if self.spacy_use_gpu and n_process != 1:
                logger.warning("spaCy GPU mode requires n_process=1; overriding.")
                n_process = 1

            try:
                docs = self._nlp.pipe(
                    decoded,
                    batch_size=max(1, self.spacy_batch_size),
                    n_process=max(1, n_process),
                )
                for doc in docs:
                    words = set(self.function_words)
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
                    function_sets.append(words)
            except Exception:
                logger.warning("SVS POS tagging failed; falling back to lexical list.")
                return [set(self.function_words) for _ in decoded]

            return function_sets

        return [set(self.function_words) for _ in decoded]

    def _build_function_sets_hf(self, decoded: Sequence[str]) -> List[set[str]]:
        if self._hf_pipeline is None:
            return [set(self.function_words) for _ in decoded]

        try:
            results = self._hf_pipeline(
                list(decoded),
                batch_size=max(1, self.hf_batch_size),
                aggregation_strategy="simple",
            )
        except Exception:
            logger.warning("SVS HF POS tagging failed; falling back to lexical list.")
            return [set(self.function_words) for _ in decoded]

        if results and isinstance(results, list) and isinstance(results[0], dict):
            results = [results]

        function_sets: List[set[str]] = []
        for token_preds in results:
            words = set(self.function_words)
            for pred in token_preds:
                tag = (
                    pred.get("entity_group")
                    or pred.get("entity")
                    or pred.get("tag")
                    or ""
                )
                tag = tag.replace("B-", "").replace("I-", "")
                if not tag:
                    continue
                if self._is_function_pos_tag(tag):
                    token_text = pred.get("word") or ""
                    norm = self._normalize_token(token_text)
                    if norm:
                        words.add(norm)
            function_sets.append(words)
        if len(function_sets) != len(decoded):
            return [set(self.function_words) for _ in decoded]
        return function_sets

    @staticmethod
    def _is_function_pos_tag(tag: str) -> bool:
        tag = tag.upper()
        universal = {
            "ADP",
            "AUX",
            "CCONJ",
            "DET",
            "PART",
            "PRON",
            "SCONJ",
        }
        ptb = {
            "IN",
            "TO",
            "DT",
            "CC",
            "PRP",
            "PRP$",
            "MD",
            "WDT",
            "WP",
            "WP$",
            "WRB",
            "PDT",
            "RP",
            "EX",
            "UH",
            "POS",
        }
        return tag in universal or tag in ptb

    def _build_token_masks(
        self,
        *,
        tokens: Sequence[Sequence[str]],
        function_sets: Sequence[set[str]],
        key_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = key_masks.shape
        function_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        content_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)

        for batch_idx, token_row in enumerate(tokens):
            function_words = function_sets[batch_idx]
            key_mask = key_masks[batch_idx]
            for pos, token in enumerate(token_row):
                if pos >= seq_len:
                    break
                if not bool(key_mask[pos]):
                    continue
                norm = self._normalize_token(token)
                if not norm:
                    continue
                if norm in function_words:
                    function_mask[batch_idx, pos] = True
                else:
                    content_mask[batch_idx, pos] = True

        return function_mask, content_mask

    @staticmethod
    def _load_spacy_model(
        model_name: str,
        *,
        use_gpu: bool = False,
        gpu_id: Optional[int] = None,
        disable: Optional[Iterable[str]] = None,
    ):
        cache_key = (model_name, bool(use_gpu), gpu_id, tuple(disable or []))
        cached = SVSCalculator._NLP_CACHE.get(cache_key)
        if cached is not None:
            return cached
        try:
            import spacy
        except Exception:
            logger.warning("spaCy not available; SVS POS tagging disabled.")
            return None
        try:
            if use_gpu:
                try:
                    if gpu_id is None:
                        spacy.require_gpu()
                    else:
                        spacy.require_gpu(gpu_id)
                except Exception as exc:
                    logger.warning("spaCy GPU requested but unavailable: %s. Falling back to CPU.", exc)
            nlp = spacy.load(model_name, disable=list(disable or []))
            SVSCalculator._NLP_CACHE[cache_key] = nlp
            return nlp
        except Exception:
            logger.warning("spaCy model '%s' not found; SVS POS tagging disabled.", model_name)
            return None

    @staticmethod
    def _load_hf_pipeline(
        *,
        model_name: str,
        device: int,
    ):
        cache_key = (model_name, int(device))
        cached = SVSCalculator._HF_PIPELINE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        try:
            from transformers import pipeline
        except Exception:
            logger.warning("Transformers not available; SVS HF POS tagging disabled.")
            return None
        try:
            pos_pipeline = pipeline(
                "token-classification",
                model=model_name,
                tokenizer=model_name,
                device=device,
            )
            SVSCalculator._HF_PIPELINE_CACHE[cache_key] = pos_pipeline
            return pos_pipeline
        except Exception:
            logger.warning("HF POS model '%s' not found; SVS POS tagging disabled.", model_name)
            return None
