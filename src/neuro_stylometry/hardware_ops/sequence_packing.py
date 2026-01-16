"""
Sequence Packing and GPU-Accelerated Span Filtering.

Implements advanced inference optimizations for GLiNER span detection:

1. Sequence Packing: Concatenates multiple sentences into a single tensor with
   block-diagonal attention masks, achieving 100% compute density (zero padding).

2. GPU Span Filtering: Performs confidence thresholding on-device using tensor
   operations (torch.where/boolean masking) before transferring results to CPU.

Key Benefits:
- Sequence Packing: Eliminates padding waste, processes N sentences as 1 document
- GPU Filtering: Reduces CPU-GPU data transfer, keeps hot path on accelerator

Reference: Performance Optimization Report - Sequence Packing & Span Fusion Sections
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sequence Packing (Flash Attention Logic)
# ---------------------------------------------------------------------------


@dataclass
class PackedSequence:
    """
    A packed sequence of multiple texts concatenated into single tensors.
    
    Attributes:
        input_ids: Concatenated token IDs (total_tokens,).
        attention_mask: Block-diagonal attention mask (total_tokens, total_tokens).
        sequence_boundaries: Start indices for each original sequence.
        sequence_lengths: Length of each original sequence.
        original_count: Number of sequences packed.
        padding_token_id: Token ID used for any residual padding.
    """
    input_ids: Tensor
    attention_mask: Tensor
    sequence_boundaries: List[int]
    sequence_lengths: List[int]
    original_count: int
    padding_token_id: int = 0


@dataclass
class SequencePackerConfig:
    """
    Configuration for sequence packing.
    
    Attributes:
        enabled: Master switch for sequence packing.
        max_packed_length: Maximum total length for packed sequences.
        min_sequences_to_pack: Minimum sequences before packing activates.
        use_block_diagonal_mask: Use block-diagonal attention mask.
        pack_by_similarity: Try to pack sequences of similar length together.
    """
    enabled: bool = False
    max_packed_length: int = 2048
    min_sequences_to_pack: int = 2
    use_block_diagonal_mask: bool = True
    pack_by_similarity: bool = True


class SequencePacker:
    """
    Packs multiple short sequences into single tensors for efficient inference.
    
    Instead of padding each sequence to max length independently, this packer:
    1. Concatenates sequences into a single 1D input_ids tensor
    2. Creates a block-diagonal attention mask preventing cross-sequence attention
    3. Tracks boundaries for unpacking results back to original sequences
    
    This achieves 100% compute density (zero padding) while processing multiple
    sequences simultaneously, effectively simulating batch_size=1 with N sentences.
    
    Example:
        packer = SequencePacker(config, tokenizer)
        
        # Pack multiple short texts
        packed = packer.pack(["Hello world", "How are you?", "Fine thanks"])
        
        # Run inference on packed tensor (single forward pass)
        outputs = model(packed.input_ids, packed.attention_mask)
        
        # Unpack results back to individual sequences
        results = packer.unpack(outputs, packed)
    """
    
    def __init__(
        self,
        config: Optional[SequencePackerConfig] = None,
        tokenizer: Optional[Any] = None,
        pad_token_id: int = 0,
    ):
        """
        Initialize sequence packer.
        
        Args:
            config: Packing configuration.
            tokenizer: Tokenizer for encoding texts (optional, for text input).
            pad_token_id: Padding token ID for residual padding.
        """
        self.config = config or SequencePackerConfig()
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        
        # Statistics
        self._pack_count = 0
        self._total_sequences_packed = 0
        self._padding_saved = 0  # Tokens of padding avoided
    
    def can_pack(self, sequence_lengths: List[int]) -> bool:
        """
        Check if sequences can be packed together.
        
        Args:
            sequence_lengths: Lengths of sequences to potentially pack.
            
        Returns:
            True if packing is beneficial and possible.
        """
        if not self.config.enabled:
            return False
        
        if len(sequence_lengths) < self.config.min_sequences_to_pack:
            return False
        
        total_length = sum(sequence_lengths)
        if total_length > self.config.max_packed_length:
            return False
        
        return True
    
    def pack_tokenized(
        self,
        token_sequences: List[Tensor],
        device: Optional[torch.device] = None,
    ) -> PackedSequence:
        """
        Pack pre-tokenized sequences into a single tensor.
        
        Args:
            token_sequences: List of 1D token ID tensors.
            device: Target device for packed tensors.
            
        Returns:
            PackedSequence with concatenated inputs and block-diagonal mask.
        """
        if not token_sequences:
            raise ValueError("Cannot pack empty sequence list")
        
        device = device or token_sequences[0].device
        
        # Get sequence lengths
        seq_lengths = [seq.size(0) for seq in token_sequences]
        total_length = sum(seq_lengths)
        
        # Concatenate token IDs
        input_ids = torch.cat(token_sequences, dim=0).to(device)
        
        # Build block-diagonal attention mask
        attention_mask = self._build_block_diagonal_mask(
            seq_lengths, total_length, device
        )
        
        # Track boundaries for unpacking
        boundaries = []
        pos = 0
        for length in seq_lengths:
            boundaries.append(pos)
            pos += length
        
        # Update statistics
        self._pack_count += 1
        self._total_sequences_packed += len(token_sequences)
        
        # Calculate padding saved (vs. padding each to max)
        max_len = max(seq_lengths)
        traditional_tokens = len(token_sequences) * max_len
        self._padding_saved += traditional_tokens - total_length
        
        return PackedSequence(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sequence_boundaries=boundaries,
            sequence_lengths=seq_lengths,
            original_count=len(token_sequences),
            padding_token_id=self.pad_token_id,
        )
    
    def pack_texts(
        self,
        texts: List[str],
        device: Optional[torch.device] = None,
    ) -> PackedSequence:
        """
        Tokenize and pack text sequences.
        
        Args:
            texts: List of text strings to pack.
            device: Target device for packed tensors.
            
        Returns:
            PackedSequence with concatenated inputs.
            
        Raises:
            ValueError: If tokenizer not provided.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for text packing")
        
        # Tokenize each text
        token_sequences = []
        for text in texts:
            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=True,
                return_tensors="pt",
            )
            token_sequences.append(tokens.squeeze(0))
        
        return self.pack_tokenized(token_sequences, device)
    
    def _build_block_diagonal_mask(
        self,
        seq_lengths: List[int],
        total_length: int,
        device: torch.device,
    ) -> Tensor:
        """
        Build block-diagonal attention mask.
        
        Creates a mask where each sequence can only attend to itself,
        preventing token i in sequence A from attending to token j in sequence B.
        
        Args:
            seq_lengths: Length of each sequence.
            total_length: Total packed length.
            device: Target device.
            
        Returns:
            Block-diagonal attention mask (total_length, total_length).
        """
        if not self.config.use_block_diagonal_mask:
            # Return full attention mask (all ones)
            return torch.ones(total_length, total_length, device=device)
        
        # Initialize mask with zeros (no attention)
        mask = torch.zeros(total_length, total_length, device=device)
        
        # Fill diagonal blocks for each sequence
        pos = 0
        for length in seq_lengths:
            # Each sequence attends only to itself
            mask[pos:pos + length, pos:pos + length] = 1.0
            pos += length
        
        return mask
    
    def unpack_hidden_states(
        self,
        hidden_states: Tensor,
        packed: PackedSequence,
    ) -> List[Tensor]:
        """
        Unpack hidden states back to individual sequences.
        
        Args:
            hidden_states: Model output hidden states (total_length, hidden_dim).
            packed: PackedSequence with boundary information.
            
        Returns:
            List of hidden state tensors, one per original sequence.
        """
        results = []
        for i, (start, length) in enumerate(
            zip(packed.sequence_boundaries, packed.sequence_lengths)
        ):
            seq_hidden = hidden_states[start:start + length]
            results.append(seq_hidden)
        
        return results
    
    def unpack_span_logits(
        self,
        span_logits: Tensor,
        packed: PackedSequence,
    ) -> List[Tensor]:
        """
        Unpack span classification logits back to individual sequences.
        
        Handles 2D span representations (start, end) by masking cross-sequence spans.
        
        Args:
            span_logits: Span classification logits (total_length, total_length, ...).
            packed: PackedSequence with boundary information.
            
        Returns:
            List of span logit tensors, one per original sequence.
        """
        results = []
        for start, length in zip(
            packed.sequence_boundaries, packed.sequence_lengths
        ):
            # Extract this sequence's span logits
            seq_logits = span_logits[start:start + length, start:start + length]
            results.append(seq_logits)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get packing statistics.
        
        Returns:
            Dict with pack count, sequences packed, padding saved.
        """
        return {
            "enabled": self.config.enabled,
            "pack_count": self._pack_count,
            "total_sequences_packed": self._total_sequences_packed,
            "padding_tokens_saved": self._padding_saved,
            "avg_sequences_per_pack": (
                self._total_sequences_packed / max(self._pack_count, 1)
            ),
        }


# ---------------------------------------------------------------------------
# GPU-Accelerated Span Filtering
# ---------------------------------------------------------------------------


@dataclass
class GPUSpanFilterConfig:
    """
    Configuration for GPU-accelerated span filtering.
    
    Attributes:
        enabled: Enable GPU-side filtering (vs CPU filtering).
        confidence_threshold: Minimum confidence for span acceptance.
        max_spans_per_sequence: Maximum spans to return per sequence.
        use_topk: Use top-k selection instead of threshold filtering.
        topk_k: Number of top spans to keep (if use_topk=True).
    """
    enabled: bool = True
    confidence_threshold: float = 0.85
    max_spans_per_sequence: int = 100
    use_topk: bool = False
    topk_k: int = 50


class GPUSpanFilter:
    """
    GPU-accelerated span classification and filtering.
    
    Performs confidence thresholding on-device using tensor operations:
    - torch.where for conditional masking
    - Boolean indexing for efficient selection
    - Non-maximum suppression for overlapping spans
    
    This keeps the "hot path" on the GPU, minimizing CPU-GPU data transfer
    by only moving the final filtered span indices and scores to CPU.
    
    Architecture:
    1. Receive span logits tensor on GPU
    2. Apply sigmoid/softmax to get confidence scores
    3. Use torch.where to zero out below-threshold spans
    4. Extract indices of valid predictions
    5. Transfer only valid span data to CPU
    """
    
    def __init__(self, config: Optional[GPUSpanFilterConfig] = None):
        """
        Initialize GPU span filter.
        
        Args:
            config: Filter configuration.
        """
        self.config = config or GPUSpanFilterConfig()
        
        # Statistics
        self._filter_calls = 0
        self._total_spans_input = 0
        self._total_spans_output = 0
    
    def filter_span_logits(
        self,
        span_logits: Tensor,
        threshold: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Filter span logits on GPU using confidence threshold.
        
        Args:
            span_logits: Raw span logits (seq_len, seq_len, num_classes).
            threshold: Confidence threshold (uses config default if None).
            
        Returns:
            Tuple of:
                - valid_spans: (N, 3) tensor of (start, end, class_idx)
                - valid_scores: (N,) tensor of confidence scores
                - span_mask: Boolean mask of valid positions
        """
        threshold = threshold or self.config.confidence_threshold
        device = span_logits.device
        
        # Convert logits to probabilities
        probs = torch.sigmoid(span_logits)
        
        # Get maximum confidence across classes
        max_probs, class_indices = probs.max(dim=-1)
        
        # Create threshold mask on GPU
        span_mask = max_probs >= threshold
        
        # Zero out below-threshold spans (keeps tensor shape for debugging)
        filtered_probs = torch.where(
            span_mask,
            max_probs,
            torch.zeros_like(max_probs),
        )
        
        # Extract valid span indices
        valid_indices = torch.nonzero(span_mask, as_tuple=False)
        
        if valid_indices.numel() == 0:
            # No valid spans
            empty_spans = torch.empty(0, 3, dtype=torch.long, device=device)
            empty_scores = torch.empty(0, dtype=probs.dtype, device=device)
            return empty_spans, empty_scores, span_mask
        
        # Gather valid scores and class indices
        start_indices = valid_indices[:, 0]
        end_indices = valid_indices[:, 1]
        valid_scores = max_probs[start_indices, end_indices]
        valid_classes = class_indices[start_indices, end_indices]
        
        # Stack into (N, 3) tensor: (start, end, class)
        valid_spans = torch.stack([start_indices, end_indices, valid_classes], dim=1)
        
        # Apply max spans limit if needed
        if valid_spans.size(0) > self.config.max_spans_per_sequence:
            # Keep top-k by confidence
            topk_scores, topk_indices = valid_scores.topk(
                self.config.max_spans_per_sequence
            )
            valid_spans = valid_spans[topk_indices]
            valid_scores = topk_scores
        
        # Update statistics
        self._filter_calls += 1
        self._total_spans_input += span_mask.numel()
        self._total_spans_output += valid_spans.size(0)
        
        return valid_spans, valid_scores, span_mask
    
    def filter_batch_logits(
        self,
        batch_logits: Tensor,
        threshold: Optional[float] = None,
    ) -> List[Tuple[Tensor, Tensor]]:
        """
        Filter span logits for a batch of sequences.
        
        Args:
            batch_logits: Batched span logits (batch, seq_len, seq_len, num_classes).
            threshold: Confidence threshold.
            
        Returns:
            List of (valid_spans, valid_scores) tuples per batch item.
        """
        batch_size = batch_logits.size(0)
        results = []
        
        for i in range(batch_size):
            spans, scores, _ = self.filter_span_logits(batch_logits[i], threshold)
            results.append((spans, scores))
        
        return results
    
    def topk_spans(
        self,
        span_logits: Tensor,
        k: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Select top-k spans by confidence (alternative to threshold filtering).
        
        Args:
            span_logits: Raw span logits (seq_len, seq_len, num_classes).
            k: Number of top spans to select (uses config default if None).
            
        Returns:
            Tuple of (top_spans, top_scores).
        """
        k = k or self.config.topk_k
        
        # Convert to probabilities and get max per span
        probs = torch.sigmoid(span_logits)
        max_probs, class_indices = probs.max(dim=-1)
        
        # Flatten for top-k selection
        flat_probs = max_probs.view(-1)
        flat_classes = class_indices.view(-1)
        
        # Get top-k
        k = min(k, flat_probs.numel())
        topk_scores, topk_flat_indices = flat_probs.topk(k)
        
        # Convert flat indices back to (start, end)
        seq_len = span_logits.size(0)
        start_indices = topk_flat_indices // seq_len
        end_indices = topk_flat_indices % seq_len
        topk_classes = flat_classes[topk_flat_indices]
        
        # Stack into (k, 3) tensor
        top_spans = torch.stack([start_indices, end_indices, topk_classes], dim=1)
        
        return top_spans, topk_scores
    
    def non_maximum_suppression(
        self,
        spans: Tensor,
        scores: Tensor,
        overlap_threshold: float = 0.5,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply non-maximum suppression to remove overlapping spans.
        
        Keeps the highest-scoring span when multiple spans overlap.
        
        Args:
            spans: Span tensor (N, 3) with (start, end, class).
            scores: Confidence scores (N,).
            overlap_threshold: IoU threshold for suppression.
            
        Returns:
            Tuple of filtered (spans, scores).
        """
        if spans.size(0) == 0:
            return spans, scores
        
        # Sort by score descending
        sorted_indices = scores.argsort(descending=True)
        sorted_spans = spans[sorted_indices]
        sorted_scores = scores[sorted_indices]
        
        # Greedy NMS
        keep_mask = torch.ones(sorted_spans.size(0), dtype=torch.bool, device=spans.device)
        
        for i in range(sorted_spans.size(0)):
            if not keep_mask[i]:
                continue
            
            current_span = sorted_spans[i]
            current_start, current_end = current_span[0], current_span[1]
            
            for j in range(i + 1, sorted_spans.size(0)):
                if not keep_mask[j]:
                    continue
                
                other_span = sorted_spans[j]
                other_start, other_end = other_span[0], other_span[1]
                
                # Calculate overlap (intersection over union for 1D spans)
                intersection_start = max(current_start.item(), other_start.item())
                intersection_end = min(current_end.item(), other_end.item())
                intersection = max(0, intersection_end - intersection_start)
                
                union = (
                    (current_end - current_start).item() +
                    (other_end - other_start).item() -
                    intersection
                )
                
                iou = intersection / max(union, 1)
                
                if iou >= overlap_threshold:
                    keep_mask[j] = False
        
        return sorted_spans[keep_mask], sorted_scores[keep_mask]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get filtering statistics."""
        filter_rate = 1 - (
            self._total_spans_output / max(self._total_spans_input, 1)
        )
        return {
            "enabled": self.config.enabled,
            "filter_calls": self._filter_calls,
            "total_spans_input": self._total_spans_input,
            "total_spans_output": self._total_spans_output,
            "filter_rate": filter_rate,
            "threshold": self.config.confidence_threshold,
        }


def create_span_filter_from_config(config: Dict[str, Any]) -> GPUSpanFilter:
    """
    Create GPUSpanFilter from pipeline configuration.
    
    Args:
        config: Pipeline config dict.
        
    Returns:
        Configured GPUSpanFilter.
    """
    gliner_cfg = config.get("gliner", {})
    filter_cfg = gliner_cfg.get("gpu_span_filter", {})
    
    return GPUSpanFilter(
        GPUSpanFilterConfig(
            enabled=bool(filter_cfg.get("enabled", True)),
            confidence_threshold=float(gliner_cfg.get("confidence_threshold", 0.85)),
            max_spans_per_sequence=int(filter_cfg.get("max_spans_per_sequence", 100)),
            use_topk=bool(filter_cfg.get("use_topk", False)),
            topk_k=int(filter_cfg.get("topk_k", 50)),
        )
    )


def create_sequence_packer_from_config(
    config: Dict[str, Any],
    tokenizer: Optional[Any] = None,
) -> SequencePacker:
    """
    Create SequencePacker from pipeline configuration.
    
    Args:
        config: Pipeline config dict.
        tokenizer: Optional tokenizer for text packing.
        
    Returns:
        Configured SequencePacker.
    """
    execution_cfg = config.get("execution", {})
    packing_cfg = execution_cfg.get("sequence_packing", {})
    
    return SequencePacker(
        SequencePackerConfig(
            enabled=bool(packing_cfg.get("enabled", False)),
            max_packed_length=int(packing_cfg.get("max_packed_length", 2048)),
            min_sequences_to_pack=int(packing_cfg.get("min_sequences_to_pack", 2)),
            use_block_diagonal_mask=bool(packing_cfg.get("use_block_diagonal_mask", True)),
            pack_by_similarity=bool(packing_cfg.get("pack_by_similarity", True)),
        ),
        tokenizer=tokenizer,
    )
