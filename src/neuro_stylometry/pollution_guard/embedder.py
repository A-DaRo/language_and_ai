"""
Frozen Embedder for LEACE Projection.

Extracts CLS token embeddings from a frozen RoBERTa model.
Used to compute covariance statistics for LEACE.

Implements: FR-09 (Frozen Embedding)
"""

import logging
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FrozenEmbedder(nn.Module):
    """
    Frozen encoder for extracting CLS embeddings.
    
    Uses RoBERTa-base (or similar) with frozen weights.
    Extracts [CLS] token embeddings for LEACE covariance computation.
    
    Implements: FR-09
    """
    
    def __init__(
        self,
        model_name: str = "roberta-base",
        device: str = "cuda",
        max_length: int = 512,
    ):
        """
        Initialize frozen embedder.
        
        Args:
            model_name: HuggingFace model identifier.
            device: PyTorch device.
            max_length: Maximum sequence length.
        """
        super().__init__()
        
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        
        # Load tokenizer and model
        logger.info(f"Loading frozen embedder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Move to device and freeze
        self.model.to(self.device)
        self.model.eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size
        
        logger.info(
            f"FrozenEmbedder initialized: dim={self.embedding_dim}, device={self.device}"
        )
    
    @torch.no_grad()
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """
        Extract CLS embeddings from texts.
        
        Args:
            texts: List of input texts.
            batch_size: Batch size for inference.
            show_progress: Whether to show progress bar.
            
        Returns:
            Tensor of shape (num_texts, embedding_dim) with CLS embeddings.
        """
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
            
            all_embeddings.append(cls_embeddings.cpu())
        
        # Concatenate all batches
        embeddings = torch.cat(all_embeddings, dim=0)
        
        logger.info(f"Embedded {len(texts)} texts -> {embeddings.shape}")
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        return self.embedding_dim
