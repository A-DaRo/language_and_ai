"""Phase D transformer with optional Affine Guard injection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from .roberta_local import RobertaModel
from .affine_guard import AffineGuard
from .tokenizer import PhaseDTokenizer

logger = logging.getLogger(__name__)


class AffineGuardTransformer(nn.Module):
    """
    Transformer wrapper that injects the Affine Guard after embeddings.

    This aligns with Phase A artifacts:
        - projection_matrix.pt from artifacts/phase_a
        - typed mask tokens from gliner taxonomy config
    """

    def __init__(
        self,
        *,
        model_name: str = "roberta-base",
        projection_matrix_path: Optional[str | Path] = None,
        mask_tokens: Optional[Iterable[str]] = None,
        taxonomy_path: str | Path = "conf/base/gliner_taxonomy.yaml",
        max_length: int = 512,
        freeze_projection: bool = True,
        enforce_single_token_masks: bool = True,
    ) -> None:
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name)

        # Enable Scaled Dot-Product Attention (SDPA) with Flash backend
        # Requires: transformers >= 4.36, torch >= 2.0
        try:
            # Use local RobertaModel implementation for debugging
            self.model = RobertaModel.from_pretrained(
                model_name,
                attn_implementation="sdpa", # Try SDPA again with local model
            )
            logger.info(f"Using local RobertaModel with SDPA for {model_name}")
        except Exception as e:
            logger.warning(f"Failed to init local RobertaModel: {e}")
            self.model = RobertaModel.from_pretrained(model_name)

        if not hasattr(self.model, "embeddings") or not hasattr(self.model, "encoder"):
            raise TypeError(
                f"Base model {model_name} does not expose embeddings/encoder attributes"
            )

        # self.tokenizer = PhaseDTokenizer(
        #     model_name=model_name,
        #     max_length=max_length,
        #     mask_tokens=mask_tokens,
        #     taxonomy_path=taxonomy_path,
        #     enforce_single_token=enforce_single_token_masks,
        # )
        # self.tokenizer.resize_model_embeddings(self.model)

        self.affine_guard: Optional[AffineGuard]
        if projection_matrix_path is not None:
            self.affine_guard = AffineGuard.from_checkpoint(
                projection_matrix_path,
                freeze=freeze_projection,
            )
        else:
            self.affine_guard = None

    @classmethod
    def from_phase_a(
        cls,
        artifacts_dir: str | Path,
        *,
        model_name: str = "roberta-base",
        taxonomy_path: str | Path = "conf/base/gliner_taxonomy.yaml",
        max_length: int = 512,
        freeze_projection: bool = True,
        enforce_single_token_masks: bool = True,
    ) -> "AffineGuardTransformer":
        artifacts_dir = Path(artifacts_dir)
        projection_path = artifacts_dir / "projection_matrix.pt"
        if not projection_path.exists():
            raise FileNotFoundError(f"Missing Phase A projection: {projection_path}")

        return cls(
            model_name=model_name,
            projection_matrix_path=projection_path,
            taxonomy_path=taxonomy_path,
            max_length=max_length,
            freeze_projection=freeze_projection,
            enforce_single_token_masks=enforce_single_token_masks,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> dict:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        embedding_output = self.model.embeddings(input_ids)
        if self.affine_guard is not None:
            embedding_output = self.affine_guard(embedding_output)

        extended_attention_mask = self.model.get_extended_attention_mask(
            attention_mask, input_ids.shape
        )
        if head_mask is not None:
            head_mask = self.model.get_head_mask(
                head_mask, self.model.config.num_hidden_layers
            )

        encoder_outputs = self.model.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        sequence_output = encoder_outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]

        output = {
            "last_hidden_state": sequence_output,
            "cls_embedding": cls_output,
        }

        if output_attentions:
            output["attentions"] = encoder_outputs.attentions
        if output_hidden_states:
            output["hidden_states"] = encoder_outputs.hidden_states

        return output

    def forward_cls(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Optimized forward for CUDA graph capture - returns only CLS embedding.
        
        Excludes optional outputs to ensure static graph structure. This method
        is required for GraphAwareTraining which captures CUDA graphs for the
        forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            CLS embedding tensor [batch_size, hidden_dim]
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        embedding_output = self.model.embeddings(input_ids)
        if self.affine_guard is not None:
            embedding_output = self.affine_guard(embedding_output)

        # Manual attention mask expansion for CUDA Graph safety
        # Avoids HF get_extended_attention_mask which may trigger CPU syncs
        # Target shape: [batch_size, 1, 1, seq_len]
        # logic: (1.0 - mask) * min_value
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=embedding_output.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(embedding_output.dtype).min
        
        # encoder_outputs = self.model.encoder(
        #     embedding_output,
        #     attention_mask=extended_attention_mask,
        #     head_mask=None,
        #     output_attentions=False,
        #     output_hidden_states=False,
        #     return_dict=True,
        # )

        # return encoder_outputs.last_hidden_state[:, 0, :]  # CLS only
        return embedding_output[:, 0, :] # Fake CLS from embeddings for debugging
