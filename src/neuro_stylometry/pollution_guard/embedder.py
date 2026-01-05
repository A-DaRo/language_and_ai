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
    
    Attributes:
        output_device: Device to move final embeddings to. If "cpu" is specified,
                       embeddings are moved from GPU to CPU after extraction.
                       Recommended for laptop mode to reduce VRAM fragmentation.
    
    Implements: FR-09
    """
    
    def __init__(
        self,
        model_name: str = "roberta-base",
        device: str = "cuda",
        max_length: int = 512,
        output_device: Optional[str] = None,
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
        """
        super().__init__()
        
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.output_device = output_device
        
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
        
        device_info = f"model_device={self.device}"
        if self.output_device:
            device_info += f", output_device={self.output_device}"
        
        logger.info(
            f"FrozenEmbedder initialized: dim={self.embedding_dim}, {device_info}"
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
            
            # Move to output device if specified
            if final_output_device:
                cls_embeddings = cls_embeddings.to(final_output_device)
            else:
                cls_embeddings = cls_embeddings.cpu()
            
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
