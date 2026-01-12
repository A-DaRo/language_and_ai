"""
Frozen Embedder for LEACE Projection.

Extracts CLS token embeddings from a frozen RoBERTa model.
Used to compute covariance statistics for LEACE.

Device Pipeline (Laptop Mode):
1. Text → GLiNER on GPU (config: gliner.device="cuda")
2. Embeddings extracted on GPU
3. Embeddings → CPU (explicit output_device="cpu")
4. LEACE accumulation on CPU (force_cpu=true)
5. P matrix computed on CPU (FP64 for stability)
6. P saved to disk (FP32 for downstream compatibility)

Critical Constraint (Section 5.1 of LEACE Strategy Report):
- FrozenEmbedder MUST register the same typed mask tokens as GLiNERDetector.
- Without this, mask tokens like "[MASK:AGE]" are fragmented into subwords,
  destroying the single-token assumption required for precise concept erasure.
- Use `special_tokens` parameter to pass mask tokens from SOBRTaxonomy.

Implements: FR-09 (Frozen Embedding)
"""

import logging
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional, Sequence
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FrozenEmbedder(nn.Module):
    """
    Frozen encoder for extracting CLS embeddings.
    
    Uses RoBERTa-base (or similar) with frozen weights.
    Extracts [CLS] token embeddings for LEACE covariance computation.
    
    Attributes:
        output_device: Device to move final embeddings to. If "cpu" is specified,
                       embeddings are moved from GPU to CPU after extraction.
                       Recommended for laptop mode to reduce VRAM fragmentation.
        special_tokens: List of typed mask tokens (e.g., ["[MASK:AGE]", "[MASK:GENDER]"])
                        that must be registered for tokenizer alignment with GLiNER.
    
    Critical Constraint (FR-09, Section 5.1):
        When processing masked text (`post_masked`), the tokenizer must encode
        mask tokens like "[MASK:AGE]" as single tokens, not subword fragments.
        Pass the mask tokens from SOBRTaxonomy via `special_tokens` parameter.
    
    Implements: FR-09
    """
    
    def __init__(
        self,
        model_name: str = "roberta-base",
        device: str = "cuda",
        max_length: int = 512,
        output_device: Optional[str] = None,
        special_tokens: Optional[Sequence[str]] = None,
    ):
        """
        Initialize frozen embedder.
        
        Args:
            model_name: HuggingFace model identifier.
            device: PyTorch device for model inference.
            max_length: Maximum sequence length.
            output_device: Device to move embeddings to after extraction.
                           If "cpu", embeddings are explicitly moved to CPU.
                           Recommended for laptop mode to reduce VRAM fragmentation.
            special_tokens: List of typed mask tokens to register (e.g., from
                            SOBRTaxonomy.get_mask_tokens().values()). CRITICAL for
                            tokenizer alignment with GLiNER-masked text. Without this,
                            "[MASK:AGE]" becomes ["[", "MASK", ":", "AGE", "]"].
        """
        super().__init__()
        
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.output_device = output_device
        
        # Load tokenizer and model
        logger.info(f"Loading frozen embedder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=True)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Register special tokens if provided (Section 5.1 requirement)
        if special_tokens:
            self._register_special_tokens(list(special_tokens))
        
        # Move to device and freeze
        self.model.to(self.device)
        self.model.eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size
        
        device_info = f"model_device={self.device}"
        if self.output_device:
            device_info += f", output_device={self.output_device}"
        
        logger.info(
            f"FrozenEmbedder initialized: dim={self.embedding_dim}, {device_info}"
        )

    def _register_special_tokens(self, special_tokens: List[str]) -> None:
        """
        Register typed mask tokens as additional special tokens.
        
        This ensures that tokens like "[MASK:AGE]" are encoded as single tokens,
        not fragmented into subwords. Critical for LEACE on masked text.
        
        Args:
            special_tokens: List of mask tokens (e.g., ["[MASK:AGE]", "[MASK:GENDER]"]).
        """
        # Deduplicate while preserving order
        unique_tokens = list(dict.fromkeys(special_tokens))
        
        # Add as additional special tokens
        num_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": unique_tokens}
        )
        
        if num_added > 0:
            # Resize model embeddings to accommodate new tokens
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info(
                f"Registered {num_added} special tokens for tokenizer alignment: "
                f"{unique_tokens[:3]}{'...' if len(unique_tokens) > 3 else ''}"
            )
            
            # Verify single-token encoding
            for token in unique_tokens:
                token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                if len(token_ids) != 1:
                    logger.warning(
                        f"Special token '{token}' encoded as {len(token_ids)} tokens "
                        f"(expected 1): {token_ids}"
                    )
    
    @torch.no_grad()
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
        output_device: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Extract CLS embeddings from texts.
        
        Args:
            texts: List of input texts.
            batch_size: Batch size for inference.
            show_progress: Whether to show progress bar.
            output_device: Override for output device. If provided, embeddings
                           are moved to this device before returning.
                           Useful for laptop mode: output_device='cpu' recommended
                           to reduce VRAM fragmentation during LEACE accumulation.
            
        Returns:
            Tensor of shape (num_texts, embedding_dim) with CLS embeddings.
            Located on output_device if specified, otherwise on model device.
        """
        # Determine final output device
        final_output_device = output_device or self.output_device
        
        all_embeddings = []
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding texts")
        
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encodings = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            
            # Move to device
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            # Extract CLS embeddings (first token)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_dim)
            
            # Move to output device if specified, otherwise keep on model device
            # (previous behavior defaulted to CPU, but GPU retention is better for HPC)
            if final_output_device:
                cls_embeddings = cls_embeddings.to(final_output_device)
            # else: keep on self.device (model device)
            
            all_embeddings.append(cls_embeddings)
        
        # Concatenate all batches
        embeddings = torch.cat(all_embeddings, dim=0)
        
        device_msg = f"device={embeddings.device}"
        if final_output_device:
            device_msg = f"output_device={final_output_device}"
        logger.info(f"Embedded {len(texts)} texts -> {embeddings.shape} ({device_msg})")
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        return self.embedding_dim
