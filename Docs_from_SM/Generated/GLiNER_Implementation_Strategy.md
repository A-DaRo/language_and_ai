# GLiNER Implementation Strategy Report
## Phase A: Pollution Detection and Mitigation

**Document Type:** Technical Implementation Report  
**Target Audience:** Senior ML Engineer  
**Generated:** January 2, 2026  

---

## Executive Summary

This report provides a deep-dive code analysis of the `urchade/GLiNER` library to guide the implementation of the `GLiNERDetector` wrapper class for Phase A of the Neuro-Symbolic Stylometry Pipeline. The analysis confirms that GLiNER is well-suited for autonomous demographic span detection with the following key findings:

1. **Zero-shot capability confirmed**: The architecture directly supports runtime label swapping without retraining
2. **Raw probability access available**: Sigmoid-activated logits can be extracted for custom thresholding (`> 0.85`)
3. **Flat NER enforcement built-in**: Greedy span selection already implements non-overlapping span filtering
4. **`torch.compile` supported**: The codebase includes native compilation support via `model.compile()`
5. **Inference packing available**: Built-in sequence packing for HPC throughput optimization

---

## A. Codebase Entry Points

### A.1 Model Loading

The primary entry point for loading `urchade/gliner_large-v2.1` is the unified `GLiNER` meta-class:

```python
from gliner import GLiNER

# Load pretrained model (auto-detects UniEncoderSpanGLiNER for gliner_large-v2.1)
model = GLiNER.from_pretrained(
    "urchade/gliner_large-v2.1",
    map_location="cuda",           # Device placement
    compile_torch_model=True,      # Enable torch.compile for HPC
    max_length=512,                # Override max sequence length
)
```

**Internal class resolution flow:**
```
GLiNER.from_pretrained()
    └── _get_gliner_class(config)  # Inspects config.span_mode, labels_encoder, etc.
        └── Returns: UniEncoderSpanGLiNER  # For standard span-based models
            └── UniEncoderSpanGLiNER.from_pretrained()
                └── Loads weights into UniEncoderSpanModel
```

**Key classes involved:**
- [model.py](model.py#L2593): `GLiNER` — Meta-class with `from_pretrained()` factory
- [model.py](model.py#L1098): `BaseEncoderGLiNER` — Provides `inference()` method
- [modeling/base.py](modeling/base.py#L302): `UniEncoderSpanModel` — Core forward pass

### A.2 Batch Inference Signature

The inference method accepts `List[str]` for batch processing:

```python
# File: model.py, Line 1202
@torch.no_grad()
def inference(
    self,
    texts: Union[str, List[str]],          # ✓ Accepts List[str]
    labels: List[str],                      # Entity types to detect
    flat_ner: bool = True,                  # Enforce non-overlapping spans
    threshold: float = 0.5,                 # Confidence threshold
    multi_label: bool = False,              # Single label per span
    batch_size: int = 8,                    # Batch size for DataLoader
    packing_config: Optional[InferencePackingConfig] = None,  # HPC packing
    **external_inputs,
) -> List[List[Dict[str, Any]]]:
```

**Input format:** Raw text strings (not `input_ids`). Tokenization is handled internally via `prepare_inputs()`.

### A.3 Entity Prompt Tokenization

Entity prompts are tokenized and concatenated with input text in [processor.py](data_processing/processor.py#L127):

```python
# File: data_processing/processor.py, Lines 127-151
def prepare_inputs(
    self,
    texts: Sequence[Sequence[str]],
    entities: Union[Sequence[Sequence[str]], Dict[int, Sequence[str]], Sequence[str]],
    ...
) -> Tuple[List[List[str]], List[int]]:
    """
    Prepends entity type special tokens that aggregate entity label information.
    
    Format: [ENT] label_0 [ENT] label_1 ... [ENT] label_N [SEP] token_0 token_1 ... token_M
    """
    input_texts: List[List[str]] = []
    prompt_lengths: List[int] = []

    for i, text in enumerate(texts):
        ents = self._select_entities(i, entities, blank)
        prompt: List[str] = []
        for ent in ents:
            prompt.append(self.ent_token)   # "<<ENT>>" by default
            if add_entities:
                prompt.append(str(ent))      # e.g., "age_statement"
        prompt.append(self.sep_token)        # "<<SEP>>"
        prompt_lengths.append(len(prompt))
        input_texts.append(prompt + list(text))
    return input_texts, prompt_lengths
```

**Critical insight:** The `ent_token` (`<<ENT>>`) and `sep_token` (`<<SEP>>`) are added to the tokenizer vocabulary during model initialization. See [model.py#L215](model.py#L215).

---

## B. API Usage & Customizability

### B.1 Dynamic Label Swapping (Open-Set NER)

Labels can be changed at every inference call without retraining:

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_large-v2.1", map_location="cuda")

# Phase A demographic taxonomy
POLLUTION_TAXONOMY = [
    "age_statement",       # "I am 25", "25M"
    "gender_indicator",    # "As a woman", "my husband"
    "nationality_claim",   # "In my country (Germany)"
    "political_self_id",   # "As a liberal"
]

# Inference with custom labels (zero-shot)
entities = model.inference(
    texts=["As a 25M from Germany, I believe..."],
    labels=POLLUTION_TAXONOMY,
    flat_ner=True,
    threshold=0.85,
)
# Output: [[{'start': 5, 'end': 8, 'text': '25M', 'label': 'age_statement', 'score': 0.92}, ...]]
```

### B.2 Extracting Raw Probability Scores

The inference pipeline applies sigmoid internally. Raw scores are accessible via the `score` field:

```python
# File: model.py, Lines 1269-1285
for start_token_idx, end_token_idx, ent_type, ent_score in output:
    ent_details = {
        "start": start_token_idx_to_text_idx[start_token_idx],
        "end": end_token_idx_to_text_idx[end_token_idx],
        "text": texts[i][start_text_idx:end_text_idx],
        "label": ent_type,
        "score": ent_score,  # ← Sigmoid-activated probability
    }
```

**Custom thresholding at `> 0.85`:**

```python
def detect_pollution_spans(
    model: GLiNER,
    texts: List[str],
    threshold: float = 0.85,
) -> List[List[Dict]]:
    """Detect demographic spans with high-confidence threshold."""
    raw_entities = model.inference(
        texts=texts,
        labels=POLLUTION_TAXONOMY,
        flat_ner=True,
        threshold=threshold,  # Applied in decoder.decode()
    )
    return raw_entities
```

The threshold is applied in [decoder.py](decoding/decoder.py#L103):

```python
# File: decoding/decoder.py, Lines 103-106
def _find_candidate_spans(
    self, probs: torch.Tensor, threshold: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.where(probs > threshold)  # ← Threshold applied here
```

### B.3 Flat NER Enforcement (Greedy Span Selection)

The greedy span selection for flat NER is implemented in [decoder.py](decoding/decoder.py#L53):

```python
# File: decoding/decoder.py, Lines 53-84
def greedy_search(self, spans: List[tuple], flat_ner: bool = True, multi_label: bool = False) -> List[tuple]:
    """
    Perform greedy search to remove overlapping spans.
    Sorts spans by confidence score (descending) and keeps only non-overlapping spans.
    """
    if flat_ner:
        has_ov = partial(has_overlapping, multi_label=multi_label)
    else:
        has_ov = partial(has_overlapping_nested, multi_label=multi_label)

    new_list = []
    span_prob = sorted(spans, key=lambda x: -x[-1])  # Sort by score descending

    for i in range(len(spans)):
        b = span_prob[i]
        flag = False
        for new in new_list:
            if has_ov(b[:-1], new):
                flag = True
                break
        if not flag:
            new_list.append(b)

    new_list = sorted(new_list, key=lambda x: x[0])  # Sort by start position
    return new_list
```

**To enforce flat masking:** Always set `flat_ner=True` in inference calls.

---

## C. Pollution Detection Approach (SOBR Dataset)

### C.1 `detect_spans` Method Implementation

```python
from typing import List, Dict, Tuple
from dataclasses import dataclass
import re

from gliner import GLiNER, InferencePackingConfig


@dataclass
class PollutionSpan:
    """Detected demographic span with metadata."""
    start: int
    end: int
    text: str
    label: str
    score: float


class GLiNERDetector:
    """Wrapper for GLiNER-based demographic span detection."""
    
    POLLUTION_TAXONOMY = [
        "age_statement",
        "gender_indicator", 
        "nationality_claim",
        "political_self_id",
    ]
    
    MASK_MAPPING = {
        "age_statement": "[MASK:AGE]",
        "gender_indicator": "[MASK:GENDER]",
        "nationality_claim": "[MASK:NATIONALITY]",
        "political_self_id": "[MASK:POLITICAL]",
    }
    
    def __init__(
        self,
        model_id: str = "urchade/gliner_large-v2.1",
        device: str = "cuda",
        compile_model: bool = True,
        threshold: float = 0.85,
    ):
        self.threshold = threshold
        self.model = GLiNER.from_pretrained(
            model_id,
            map_location=device,
            compile_torch_model=compile_model,
        )
        self.model.eval()
        
        # Configure inference packing for HPC
        if device == "cuda":
            self._packing_config = InferencePackingConfig(
                max_length=512,
                streams_per_batch=4,
            )
        else:
            self._packing_config = None
    
    def detect_spans(
        self,
        texts: List[str],
        batch_size: int = 64,
    ) -> List[List[PollutionSpan]]:
        """
        Detect demographic pollution spans in a batch of texts.
        
        Args:
            texts: List of raw text strings
            batch_size: Batch size for inference
            
        Returns:
            List of lists containing PollutionSpan objects
        """
        raw_entities = self.model.inference(
            texts=texts,
            labels=self.POLLUTION_TAXONOMY,
            flat_ner=True,
            threshold=self.threshold,
            batch_size=batch_size,
            packing_config=self._packing_config,
        )
        
        results = []
        for entities in raw_entities:
            spans = [
                PollutionSpan(
                    start=e["start"],
                    end=e["end"],
                    text=e["text"],
                    label=e["label"],
                    score=e["score"],
                )
                for e in entities
            ]
            results.append(spans)
        return results
    
    def mask_text(
        self,
        text: str,
        spans: List[PollutionSpan],
    ) -> Tuple[str, List[Dict]]:
        """
        Replace detected spans with typed masks.
        
        Args:
            text: Original text
            spans: Detected pollution spans
            
        Returns:
            Tuple of (masked_text, pollution_log)
        """
        # Sort spans in reverse order to preserve offsets during replacement
        sorted_spans = sorted(spans, key=lambda s: s.start, reverse=True)
        
        masked_text = text
        pollution_log = []
        
        for span in sorted_spans:
            mask_token = self.MASK_MAPPING.get(span.label, "[MASK:UNKNOWN]")
            masked_text = (
                masked_text[:span.start] + 
                mask_token + 
                masked_text[span.end:]
            )
            pollution_log.append({
                "original_text": span.text,
                "start": span.start,
                "end": span.end,
                "label": span.label,
                "confidence": span.score,
                "mask_token": mask_token,
            })
        
        return masked_text, pollution_log
```

### C.2 Handling Long SOBR Posts (>512 Tokens)

**GLiNER does NOT implement internal sliding windows.** The `max_len` config parameter (default: 384) truncates sequences.

**Solution: Implement chunking in the `GLiNERDetector` wrapper:**

```python
def _chunk_text(
    self,
    text: str,
    max_chars: int = 1500,  # ~384 tokens at 4 chars/token average
    overlap: int = 200,
) -> List[Tuple[str, int]]:
    """
    Split long text into overlapping chunks.
    
    Returns:
        List of (chunk_text, char_offset) tuples
    """
    if len(text) <= max_chars:
        return [(text, 0)]
    
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        
        # Find sentence boundary for clean breaks
        if end < len(text):
            # Look for sentence-ending punctuation
            for i in range(end, max(start + max_chars // 2, start), -1):
                if text[i] in '.!?\n':
                    end = i + 1
                    break
        
        chunks.append((text[start:end], start))
        start = end - overlap
    
    return chunks


def detect_spans_long(
    self,
    texts: List[str],
    max_chars: int = 1500,
    overlap: int = 200,
    batch_size: int = 64,
) -> List[List[PollutionSpan]]:
    """
    Detect spans in texts that may exceed model's max_length.
    
    Chunks long texts, runs inference, then merges with offset adjustment.
    """
    # Flatten all chunks with source tracking
    all_chunks = []
    chunk_metadata = []  # (text_idx, char_offset)
    
    for text_idx, text in enumerate(texts):
        for chunk_text, char_offset in self._chunk_text(text, max_chars, overlap):
            all_chunks.append(chunk_text)
            chunk_metadata.append((text_idx, char_offset))
    
    # Batch inference on all chunks
    chunk_results = self.detect_spans(all_chunks, batch_size)
    
    # Reconstruct per-document results with offset adjustment
    results = [[] for _ in texts]
    seen_spans = [set() for _ in texts]  # Deduplicate overlapping chunks
    
    for chunk_idx, spans in enumerate(chunk_results):
        text_idx, char_offset = chunk_metadata[chunk_idx]
        
        for span in spans:
            # Adjust offsets to original document coordinates
            adjusted_span = PollutionSpan(
                start=span.start + char_offset,
                end=span.end + char_offset,
                text=span.text,
                label=span.label,
                score=span.score,
            )
            
            # Deduplicate spans detected in overlap regions
            span_key = (adjusted_span.start, adjusted_span.end, adjusted_span.label)
            if span_key not in seen_spans[text_idx]:
                seen_spans[text_idx].add(span_key)
                results[text_idx].append(adjusted_span)
    
    # Sort spans by start position
    for doc_spans in results:
        doc_spans.sort(key=lambda s: s.start)
    
    return results
```

### C.3 Complete Masking Pipeline

```python
def process_corpus(
    self,
    texts: List[str],
    batch_size: int = 64,
) -> Tuple[List[str], List[List[Dict]]]:
    """
    Full pipeline: Input Text → GLiNER Inference → Span Extraction → Mask Replacement
    
    Returns:
        Tuple of (masked_texts, pollution_logs)
    """
    # Step 1: Detect spans (handles long texts via chunking)
    all_spans = self.detect_spans_long(texts, batch_size=batch_size)
    
    # Step 2: Apply masks and generate logs
    masked_texts = []
    pollution_logs = []
    
    for text, spans in zip(texts, all_spans):
        masked_text, log = self.mask_text(text, spans)
        masked_texts.append(masked_text)
        pollution_logs.append(log)
    
    return masked_texts, pollution_logs
```

---

## D. Requirements of the Approach

### D.1 Memory Estimation

**Model architecture analysis for `gliner_large-v2.1`:**
- Backbone: DeBERTa-v3-large (304M parameters)
- Hidden size: 1024
- Max width (K): 12
- Span representation layer: ~50M parameters
- Total: ~350M parameters

**VRAM calculation for batch_size=256:**

| Component | Calculation | Memory |
|-----------|-------------|--------|
| Model weights (FP16) | 350M × 2 bytes | 700 MB |
| Activations per sample | 512 tokens × 1024 dim × 2 bytes | 1 MB |
| Activations (batch=256) | 256 × 1 MB | 256 MB |
| Span logits (B, L, K, C) | 256 × 512 × 12 × 4 × 4 bytes | 100 MB |
| Gradient-free overhead | ~200 MB | 200 MB |
| **Total (eval mode)** | | **~1.3 GB** |

**Recommendation:** `gliner_large-v2.1` at batch_size=256 fits comfortably within A100's 80GB HBM2e. Consider batch_size=512-1024 for maximum throughput.

### D.2 Latency & Throughput Estimation

**Architecture advantage (BiLM vs LLM):**
- GLiNER uses **bidirectional encoding** (single forward pass)
- LLMs require **autoregressive generation** (sequential decoding)
- Span matching is O(L × K × C) after encoding, not O(L²)

**Throughput estimate on A100:**

| Component | Time per batch (256 samples) |
|-----------|------------------------------|
| Tokenization | ~5 ms |
| Encoder forward | ~20 ms (with torch.compile) |
| Span scoring | ~5 ms |
| Decoding | ~2 ms |
| **Total** | ~32 ms |

**Samples/second:** 256 / 0.032 = **8,000 samples/sec**

This **exceeds the 1,000 samples/sec target** by 8×.

**Optimizations for maximum throughput:**

```python
from gliner import GLiNER, InferencePackingConfig

# HPC-optimized initialization
model = GLiNER.from_pretrained(
    "urchade/gliner_large-v2.1",
    map_location="cuda",
    compile_torch_model=True,            # Enable torch.compile
    _attn_implementation="flash_attention_2",  # Flash Attention 2
)

# Configure inference packing to reduce padding waste
packing_config = InferencePackingConfig(
    max_length=512,
    streams_per_batch=8,  # Pack multiple sequences into dense streams
)

# Bucketed batching for length-homogeneous batches
def bucket_by_length(texts: List[str], num_buckets: int = 4) -> List[List[str]]:
    """Group texts by approximate length to minimize padding."""
    sorted_texts = sorted(enumerate(texts), key=lambda x: len(x[1]))
    bucket_size = len(texts) // num_buckets
    buckets = []
    for i in range(0, len(sorted_texts), bucket_size):
        bucket = [t for _, t in sorted_texts[i:i + bucket_size]]
        buckets.append(bucket)
    return buckets
```

### D.3 `torch.compile` and CUDA Graph Compatibility

**`torch.compile` is natively supported:**

```python
# File: model.py, Lines 209-212
def compile(self):
    """Compile the model using torch.compile for optimization."""
    self.model = torch.compile(self.model)
```

**Graph stability analysis:**

The forward pass in `UniEncoderSpanModel` is mostly static:
1. ✅ Encoder forward (fixed architecture)
2. ✅ Span representation (fixed max_width K=12)
3. ✅ Prompt projection (linear layer)
4. ✅ Dot-product scoring (einsum)

**Potential graph breaks:**
- Dynamic sequence lengths (mitigated by bucketed batching)
- The `_fit_length` padding method uses conditional branching

**CUDA Graphs:** Not directly supported due to variable batch composition. Use `torch.compile(mode="max-autotune", fullgraph=True)` instead.

### D.4 Pre-tokenized Input Support

**GLiNER does NOT accept pre-tokenized `input_ids` directly.** The `inference()` method expects raw `List[str]`.

**However, the tokenizer is a Rust-based HuggingFace tokenizer:**

```python
# File: data_processing/processor.py, Lines 27-33
from transformers import AutoTokenizer

# The tokenizer is a HuggingFace Fast Tokenizer (Rust backend)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
```

**Workaround for custom tokenization:**

If the Rust tokenizer requirement from `phaseA-D_implementation_plan.md > Section 6.2` mandates external tokenization, you can access the internal tokenizer:

```python
# Access the underlying tokenizer
tokenizer = model.data_processor.transformer_tokenizer

# Pre-tokenize externally
encoded = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)

# Note: Direct input_ids injection requires modifying the inference pipeline
```

### D.5 Dependency Requirements

**Core dependencies (inference mode, no gradients):**

```txt
# requirements_gliner.txt
torch>=2.0.0           # For torch.compile support
transformers>=4.36.0   # For AutoTokenizer/AutoModel
huggingface-hub>=0.20.0
safetensors>=0.4.0     # For fast weight loading
numpy>=1.24.0
tqdm>=4.60.0

# Optional for HPC optimization
flash-attn>=2.0.0      # Flash Attention 2
triton>=2.0.0          # For torch.compile kernel fusion
```

**CUDA requirements:**
- CUDA 11.8+ for torch.compile
- cuDNN 8.6+ for optimized attention

**Verify installation:**

```python
import torch
from gliner import GLiNER

# Check torch.compile availability
assert hasattr(torch, 'compile'), "torch.compile requires PyTorch 2.0+"

# Load and verify
model = GLiNER.from_pretrained("urchade/gliner_large-v2.1", map_location="cuda")
model.eval()

# Test inference
entities = model.predict_entities(
    "I am a 25M from Germany.",
    labels=["age_statement", "nationality_claim"],
    threshold=0.5,
)
print(entities)
```

---

## Appendix: Complete `GLiNERDetector` Class

```python
"""
GLiNER-based demographic pollution detector for Phase A.
Implements FR-06, FR-07, FR-08 from phaseA-D_implementation_plan.md.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from gliner import GLiNER, InferencePackingConfig


logger = logging.getLogger(__name__)


@dataclass
class PollutionSpan:
    """Detected demographic span with metadata."""
    start: int
    end: int
    text: str
    label: str
    score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GLiNERDetector:
    """
    GLiNER-based detector for demographic self-identification spans.
    
    Implements the symbolic detection layer of Phase A, detecting spans
    matching the pollution taxonomy with confidence > threshold.
    
    Attributes:
        model: Loaded GLiNER model instance
        threshold: Confidence threshold for span detection (default: 0.85)
        taxonomy: List of entity types to detect
    """
    
    POLLUTION_TAXONOMY = [
        "age_statement",
        "gender_indicator", 
        "nationality_claim",
        "political_self_id",
    ]
    
    MASK_MAPPING = {
        "age_statement": "[MASK:AGE]",
        "gender_indicator": "[MASK:GENDER]",
        "nationality_claim": "[MASK:NATIONALITY]",
        "political_self_id": "[MASK:POLITICAL]",
    }
    
    def __init__(
        self,
        model_id: str = "urchade/gliner_large-v2.1",
        device: str = "cuda",
        compile_model: bool = True,
        threshold: float = 0.85,
        max_length: int = 512,
        packing_streams: int = 4,
    ):
        """
        Initialize the GLiNER detector.
        
        Args:
            model_id: HuggingFace model identifier
            device: Target device ("cuda" or "cpu")
            compile_model: Whether to apply torch.compile
            threshold: Confidence threshold (0.85 per spec)
            max_length: Maximum sequence length
            packing_streams: Number of packed streams for HPC
        """
        self.threshold = threshold
        self.device = device
        self.max_length = max_length
        
        logger.info(f"Loading GLiNER model: {model_id}")
        self.model = GLiNER.from_pretrained(
            model_id,
            map_location=device,
            compile_torch_model=compile_model and device == "cuda",
            max_length=max_length,
        )
        self.model.eval()
        
        # Configure inference packing for HPC throughput
        if device == "cuda":
            self._packing_config = InferencePackingConfig(
                max_length=max_length,
                streams_per_batch=packing_streams,
            )
        else:
            self._packing_config = None
        
        logger.info(f"GLiNERDetector initialized (threshold={threshold}, device={device})")
    
    def detect_spans(
        self,
        texts: List[str],
        batch_size: int = 64,
        taxonomy: Optional[List[str]] = None,
    ) -> List[List[PollutionSpan]]:
        """
        Detect demographic pollution spans in a batch of texts.
        
        Args:
            texts: List of raw text strings
            batch_size: Batch size for inference
            taxonomy: Custom taxonomy (uses default if None)
            
        Returns:
            List of lists containing PollutionSpan objects
        """
        labels = taxonomy or self.POLLUTION_TAXONOMY
        
        with torch.inference_mode():
            raw_entities = self.model.inference(
                texts=texts,
                labels=labels,
                flat_ner=True,
                threshold=self.threshold,
                batch_size=batch_size,
                packing_config=self._packing_config,
            )
        
        results = []
        for entities in raw_entities:
            spans = [
                PollutionSpan(
                    start=e["start"],
                    end=e["end"],
                    text=e["text"],
                    label=e["label"],
                    score=e["score"],
                )
                for e in entities
            ]
            results.append(spans)
        return results
    
    def _chunk_text(
        self,
        text: str,
        max_chars: int = 1500,
        overlap: int = 200,
    ) -> List[Tuple[str, int]]:
        """Split long text into overlapping chunks with char offsets."""
        if len(text) <= max_chars:
            return [(text, 0)]
        
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            
            if end < len(text):
                for i in range(end, max(start + max_chars // 2, start), -1):
                    if text[i] in '.!?\n':
                        end = i + 1
                        break
            
            chunks.append((text[start:end], start))
            start = end - overlap
        
        return chunks
    
    def detect_spans_long(
        self,
        texts: List[str],
        max_chars: int = 1500,
        overlap: int = 200,
        batch_size: int = 64,
    ) -> List[List[PollutionSpan]]:
        """Detect spans in texts that may exceed model's max_length."""
        all_chunks = []
        chunk_metadata = []
        
        for text_idx, text in enumerate(texts):
            for chunk_text, char_offset in self._chunk_text(text, max_chars, overlap):
                all_chunks.append(chunk_text)
                chunk_metadata.append((text_idx, char_offset))
        
        chunk_results = self.detect_spans(all_chunks, batch_size)
        
        results: List[List[PollutionSpan]] = [[] for _ in texts]
        seen_spans: List[set] = [set() for _ in texts]
        
        for chunk_idx, spans in enumerate(chunk_results):
            text_idx, char_offset = chunk_metadata[chunk_idx]
            
            for span in spans:
                adjusted_span = PollutionSpan(
                    start=span.start + char_offset,
                    end=span.end + char_offset,
                    text=span.text,
                    label=span.label,
                    score=span.score,
                )
                
                span_key = (adjusted_span.start, adjusted_span.end, adjusted_span.label)
                if span_key not in seen_spans[text_idx]:
                    seen_spans[text_idx].add(span_key)
                    results[text_idx].append(adjusted_span)
        
        for doc_spans in results:
            doc_spans.sort(key=lambda s: s.start)
        
        return results
    
    def mask_text(
        self,
        text: str,
        spans: List[PollutionSpan],
    ) -> Tuple[str, List[Dict]]:
        """Replace detected spans with typed masks."""
        sorted_spans = sorted(spans, key=lambda s: s.start, reverse=True)
        
        masked_text = text
        pollution_log = []
        
        for span in sorted_spans:
            mask_token = self.MASK_MAPPING.get(span.label, "[MASK:UNKNOWN]")
            masked_text = (
                masked_text[:span.start] + 
                mask_token + 
                masked_text[span.end:]
            )
            pollution_log.append({
                "original_text": span.text,
                "start": span.start,
                "end": span.end,
                "label": span.label,
                "confidence": span.score,
                "mask_token": mask_token,
            })
        
        return masked_text, pollution_log
    
    def process_corpus(
        self,
        texts: List[str],
        batch_size: int = 64,
        output_log_path: Optional[Path] = None,
    ) -> Tuple[List[str], List[List[Dict]]]:
        """
        Full Phase A pipeline: detect and mask demographic spans.
        
        Args:
            texts: Raw SOBR posts
            batch_size: Batch size for inference
            output_log_path: Optional path to save pollution logs
            
        Returns:
            Tuple of (masked_texts, pollution_logs)
        """
        all_spans = self.detect_spans_long(texts, batch_size=batch_size)
        
        masked_texts = []
        pollution_logs = []
        
        for text, spans in zip(texts, all_spans):
            masked_text, log = self.mask_text(text, spans)
            masked_texts.append(masked_text)
            pollution_logs.append(log)
        
        if output_log_path:
            with open(output_log_path, 'w') as f:
                json.dump(pollution_logs, f, indent=2)
            logger.info(f"Pollution logs saved to {output_log_path}")
        
        return masked_texts, pollution_logs
```

---

## References

1. GLiNER source code: `urchade/GLiNER` (analyzed version: 0.2.24)
2. Phase A specification: [phaseA_pollution_filtering.md](phaseA_pollution_filtering.md)
3. Implementation plan: [phaseA-D_implementation_plan.md](phaseA-D_implementation_plan.md)
