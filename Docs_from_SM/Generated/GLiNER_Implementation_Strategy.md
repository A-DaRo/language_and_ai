# GLiNER Implementation Strategy Report
## Phase A: Pollution Detection and Mitigation

**Document Type:** Technical Implementation Report  
**Target Audience:** Senior ML Engineer  
**Generated:** January 2, 2026  

---

## Executive Summary

This report is a **domain-specific implementation specification** for Phase A (Pollution Mitigation) on the SOBR corpus.

The previous draft correctly analyzed the `urchade/GLiNER` codebase, but it did not bridge the gap between generic NER capabilities and SOBR’s constraints. In particular, it implied that SOBR schema columns (e.g., `feeling`, `judging`) map 1:1 to GLiNER labels. That assumption fails because GLiNER labels are **semantic prompts** that are concatenated into the input sequence; ambiguous prompts will over-trigger (e.g., “judging” as a verb), erasing stylometric signal.

This revision introduces the missing SOBR-specific safeguards:

1. **Schema → Prompt decoupling** via a `SOBRTaxonomy` mapping from SOBR concepts to semantically anchored prompts.
2. **Tokenizer-safe typed masks** by registering tokens like `[MASK:AGE]` as special tokens (single-token, non-fragmenting) and propagating this to Phase D.
3. **Distractor prompts** to absorb probability mass for non-self demographic mentions (e.g., “my mother is 55”).
4. **Precision filters** including entity-specific width constraints and strict score thresholding ($> 0.85$).
5. **Center-window retention** for long posts to mitigate bidirectional “edge effects” in chunked inference.

---

## 0. SOBR-Specific Requirements

### 0.1 Schema Contracts

Phase A must be compatible with the unified Arrow schemas in `src/neuro_stylometry/data_engine/schemas.py`:

- `SOBR_SCHEMA`: reads `post`, writes `post_masked`
- `POLLUTION_LOG_SCHEMA`: emits per-span logs (`post_id`, `span_start`, `span_end`, `span_text`, `entity_type`, `confidence`, `mask_token`)

Important: SOBR demographic columns (e.g., `extrovert`, `feeling`, `judging`) are **labels for projection/evaluation**, not literal entity names to detect.

### 0.2 Definition of “Pollution”

For SOBR, “pollution” is restricted to **explicit self-identification** spans that directly encode demographics (e.g., “I’m 25”, “as a German”, “I’m an INTP”, “I’m a liberal”).

We intentionally avoid masking generic trait language (“I feel…”, “I judge…”, “I believe in liberty”) because it can carry stylometric signal.

---

## 1. Critical Architectural Deficiencies (What Is Wrong)

### 1.1 Naive Taxonomy Mapping

- **Failure mode**: Passing schema-like strings (e.g., `"judging"`) as prompts causes over-triggering on verbs/adjectives.
- **Root cause**: In `processor.py`, prompts are concatenated into the token sequence; prompts are embedded semantically.
- **Fix**: Use an explicit mapping from SOBR concepts to unambiguous semantic anchors (Section 2.1).

### 1.2 Tokenizer Fragmentation (The “Mask Bug”)

- **Failure mode**: Mask strings like `[MASK:AGE]` are typically split into multiple subtokens by HuggingFace tokenizers.
- **Impact**: Phase A/Phase D see a “mask” as several tokens, diluting the stable redaction marker needed for LEACE and downstream training.
- **Fix**: Register each typed mask token as a single **special token** in the tokenizer *before* any masking/inference is relied upon.

### 1.3 Lack of “Distractor” Prompts

- **Failure mode**: If only target labels exist, GLiNER is pressured to choose among them even for non-self mentions.
- **Example**: “My mother is 55” should not be treated as the author’s age.
- **Fix**: Add distractor labels (negative sampling) and filter them out after inference.

### 1.4 Context Window “Edge Effects”

- **Failure mode**: Naive overlap chunking keeps predictions at chunk boundaries where bidirectional context is incomplete.
- **Fix**: Center-window retention: discard predictions in the first/last $N$ tokens of a chunk unless at document boundaries.

---

## 2. Actionable Technical Extensions (The Fixes)

### 2.1 `SOBRTaxonomy` Configuration Class

Replace a single flat `POLLUTION_TAXONOMY` list with a schema-aware mapping that produces semantically anchored prompt labels.

**Directives (minimum set):**

- `nationality` → `["country_of_origin", "nationality_statement", "demonym"]`
- `extrovert/sensing/feeling/judging` → `["personality_type_identifier", "mbti_type"]`
- `political_leaning` → `["political_affiliation", "ideology_self_id"]`

### 2.2 Entity-Specific Width Constraints

Apply post-processing constraints after inference:

```python
MAX_WIDTHS = {"age_statement": 5, "gender_indicator": 4, "default": 12}
```

Width is measured in *token count of the detected span* using the same HuggingFace tokenizer as GLiNER.

### 2.3 `SafeMasker` (Token-Aware Replacement)

Masking must be token-aware and validated:

1. Verify typed mask tokens are single-token in the tokenizer (no fragmentation).
2. Replace spans without corrupting surrounding tokenization.
3. Emit `POLLUTION_LOG_SCHEMA` fields (not ad-hoc keys).

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

**SOBR requirement (mask safety):** Typed mask tokens (e.g., `[MASK:AGE]`, `[MASK:NATIONALITY]`, `[MASK:MBTI]`) must be registered as **single special tokens** in `model.data_processor.transformer_tokenizer` before Phase A masking proceeds.

---

## B. API Usage & Customizability

### B.1 Dynamic Label Swapping (Open-Set NER)

Labels can be changed at every inference call without retraining:

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_large-v2.1", map_location="cuda")

# SOBR-safe inference uses (Targets + Distractors), not schema column names.
TARGET_LABELS = [
    "age_statement",
    "gender_indicator",
    "nationality_statement",
    "political_affiliation",
    "mbti_type",
]

DISTRACTORS = [
    "third_person_reference",
    "fictional_character",
    "historical_figure",
]

labels = list(dict.fromkeys(TARGET_LABELS + DISTRACTORS))

raw_entities = model.inference(
    texts=["My mother is 55. I am an INTP from Germany."],
    labels=labels,
    flat_ner=True,
    threshold=0.85,
)

# Post-process: drop distractors and enforce precision constraints.
entities = [[e for e in doc if e["label"] in TARGET_LABELS] for doc in raw_entities]
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
    labels: List[str],
    threshold: float = 0.85,
) -> List[List[Dict]]:
    """Detect demographic spans with high-confidence threshold."""
    raw_entities = model.inference(
        texts=texts,
        labels=labels,
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

from gliner import GLiNER, InferencePackingConfig


@dataclass
class PollutionSpan:
    """Detected demographic span with metadata."""
    start: int
    end: int
    text: str
    label: str
    score: float


class SOBRTaxonomy:
    """SOBR-safe mapping from abstract schema concepts to GLiNER prompt labels."""

    # Prompts are semantic anchors, not SOBR column names.
    # Minimum required directives from the revision plan:
    TARGET_PROMPTS = {
        "nationality": ["country_of_origin", "nationality_statement", "demonym"],
        "mbti": ["personality_type_identifier", "mbti_type"],
        "political": ["political_affiliation", "ideology_self_id"],
        # Additional high-value explicit self-id anchors
        "age": ["age_statement"],
        "gender": ["gender_indicator"],
    }

    MASK_TOKENS = {
        "age_statement": "[MASK:AGE]",
        "gender_indicator": "[MASK:GENDER]",
        "country_of_origin": "[MASK:NATIONALITY]",
        "nationality_statement": "[MASK:NATIONALITY]",
        "demonym": "[MASK:NATIONALITY]",
        "political_affiliation": "[MASK:POLITICAL]",
        "ideology_self_id": "[MASK:POLITICAL]",
        "personality_type_identifier": "[MASK:MBTI]",
        "mbti_type": "[MASK:MBTI]",
    }

    @classmethod
    def get_labels(cls) -> List[str]:
        labels: List[str] = []
        for prompts in cls.TARGET_PROMPTS.values():
            labels.extend(prompts)
        # stable unique
        return list(dict.fromkeys(labels))


class GLiNERDetector:
    """SOBR-safe GLiNER wrapper for demographic self-identification masking."""
    
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

        # NEW: register typed masks as single special tokens to prevent fragmentation
        self._register_special_tokens(list(set(SOBRTaxonomy.MASK_TOKENS.values())))

        # NEW: distractors absorb probability mass for non-self mentions
        self.distractors = [
            "third_person_reference",
            "fictional_character",
            "historical_figure",
        ]
        
        # Configure inference packing for HPC
        if device == "cuda":
            self._packing_config = InferencePackingConfig(
                max_length=512,
                streams_per_batch=4,
            )
        else:
            self._packing_config = None

    def _register_special_tokens(self, mask_tokens: List[str]) -> None:
        tok = self.model.data_processor.transformer_tokenizer
        tok.add_special_tokens({"additional_special_tokens": mask_tokens})
        # If the GLiNER variant requires embedding resize after adding tokens,
        # it must be performed here.

    def _get_inference_labels(self) -> List[str]:
        return list(dict.fromkeys(SOBRTaxonomy.get_labels() + self.distractors))

    def _span_token_len(self, span_text: str) -> int:
        tok = self.model.data_processor.transformer_tokenizer
        return len(tok.tokenize(span_text))
    
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
            labels=self._get_inference_labels(),
            flat_ner=True,
            threshold=self.threshold,
            batch_size=batch_size,
            packing_config=self._packing_config,
        )

        # Post-process filters:
        # 1) remove distractors
        # 2) enforce entity-specific width constraints
        MAX_WIDTHS = {"age_statement": 5, "gender_indicator": 4, "default": 12}

        results: List[List[PollutionSpan]] = []
        for entities in raw_entities:
            spans: List[PollutionSpan] = []
            for e in entities:
                if e["label"] not in SOBRTaxonomy.get_labels():
                    continue

                max_width = MAX_WIDTHS.get(e["label"], MAX_WIDTHS["default"])
                if self._span_token_len(e["text"]) > max_width:
                    continue

                spans.append(
                    PollutionSpan(
                        start=e["start"],
                        end=e["end"],
                        text=e["text"],
                        label=e["label"],
                        score=e["score"],
                    )
                )
            results.append(spans)

        return results
    
    def mask_text(
        self,
        text: str,
        spans: List[PollutionSpan],
    ) -> Tuple[str, List[Dict]]:
        """Token-aware masking with schema-compatible logs."""

        # Precondition: masks must be single tokens (no fragmentation)
        tok = self.model.data_processor.transformer_tokenizer
        for mt in set(SOBRTaxonomy.MASK_TOKENS.values()):
            if len(tok.tokenize(mt)) != 1:
                raise RuntimeError(f"Mask token fragmentation detected for '{mt}'.")

        # Sort spans in reverse order to preserve offsets during replacement
        sorted_spans = sorted(spans, key=lambda s: s.start, reverse=True)
        
        masked_text = text
        pollution_log: List[Dict] = []
        
        for span in sorted_spans:
            mask_token = SOBRTaxonomy.MASK_TOKENS.get(span.label, "[MASK:UNKNOWN]")

            # Basic alignment check (guards against stale offsets)
            if text[span.start:span.end] != span.text:
                # If this fails, masking must be performed using tokenizer offsets mapping.
                raise RuntimeError("Span/text mismatch: masking would corrupt offsets.")

            masked_text = (
                masked_text[:span.start] + 
                mask_token + 
                masked_text[span.end:]
            )
            pollution_log.append({
                # Align to POLLUTION_LOG_SCHEMA
                "span_start": span.start,
                "span_end": span.end,
                "span_text": span.text,
                "entity_type": span.label,
                "confidence": float(span.score),
                "mask_token": mask_token,
            })
        
        return masked_text, pollution_log
```

### C.2 Handling Long SOBR Posts (>512 Tokens)

**GLiNER does NOT implement internal sliding windows.** The `max_len` config parameter (default: 384) truncates sequences.

**Solution: Implement chunking with center-window retention in the `GLiNERDetector` wrapper.**

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
    edge_tokens: int = 20,
    batch_size: int = 64,
) -> List[List[PollutionSpan]]:
    """
        Detect spans in texts that may exceed model's max_length.

        Center-window retention:
        - Discard spans predicted within the first/last `edge_tokens` tokens of a chunk,
            unless that chunk edge coincides with the document boundary.
        - This mitigates bidirectional context loss at chunk edges.
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
    
    # Reconstruct per-document results with offset adjustment and edge filtering
    results = [[] for _ in texts]
    seen_spans = [set() for _ in texts]  # Deduplicate overlapping chunks
    
    for chunk_idx, spans in enumerate(chunk_results):
        text_idx, char_offset = chunk_metadata[chunk_idx]

        # Compute edge character boundaries from token offsets
        tok = self.model.data_processor.transformer_tokenizer
        enc = tok(all_chunks[chunk_idx], return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc["offset_mapping"]
        if len(offsets) > 0:
            left_edge_char = offsets[min(edge_tokens, len(offsets) - 1)][0]
            right_edge_char = offsets[max(len(offsets) - edge_tokens - 1, 0)][1]
        else:
            left_edge_char, right_edge_char = 0, len(all_chunks[chunk_idx])

        chunk_is_doc_start = (char_offset == 0)
        chunk_is_doc_end = (char_offset + len(all_chunks[chunk_idx]) >= len(texts[text_idx]))

        for span in spans:
            # Adjust offsets to original document coordinates
            adjusted_span = PollutionSpan(
                start=span.start + char_offset,
                end=span.end + char_offset,
                text=span.text,
                label=span.label,
                score=span.score,
            )

            # Discard edge predictions unless at document boundary
            in_left_edge = span.start < left_edge_char
            in_right_edge = span.end > right_edge_char
            if (in_left_edge and not chunk_is_doc_start) or (in_right_edge and not chunk_is_doc_end):
                continue
            
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
    post_ids: List[str],
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
    
    for post_id, text, spans in zip(post_ids, texts, all_spans):
        masked_text, span_logs = self.mask_text(text, spans)
        masked_texts.append(masked_text)
        # Add `post_id` for POLLUTION_LOG_SCHEMA compatibility
        pollution_logs.append([
            {"post_id": post_id, **entry}
            for entry in span_logs
        ])
    
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
    labels=["age_statement", "nationality_statement"],
    threshold=0.5,
)
print(entities)
```

---

## E. Tokenizer Synchronization Across Phases (Phase A → Phase D)

Tokenizer-safe masking only holds if Phase D uses the same vocabulary additions.

**Directive:** Phase A must persist the final set of typed mask tokens (e.g., `[MASK:AGE]`, `[MASK:NATIONALITY]`, `[MASK:MBTI]`, `[MASK:POLITICAL]`) as a pipeline artifact. Phase D must then:

1. Load its tokenizer and call:

    ```python
    tokenizer.add_special_tokens({"additional_special_tokens": mask_tokens})
    ```

2. Resize embeddings:

    ```python
    model.resize_token_embeddings(len(tokenizer))
    ```

3. Assert non-fragmentation:

    ```python
    assert len(tokenizer.tokenize("[MASK:AGE]")) == 1
    ```

Without this synchronization, Phase D will see masks as multi-token sequences and the “redaction marker” will not be a stable target for projection or downstream learning.

---

## Appendix: Revised `GLiNERDetector` Class (SOBR-Safe)

```python
"""SOBR-safe GLiNER detector for Phase A."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from gliner import GLiNER, InferencePackingConfig


@dataclass
class PollutionSpan:
    start: int
    end: int
    text: str
    label: str
    score: float


class SOBRTaxonomy:
    TARGET_PROMPTS = {
        "nationality": ["country_of_origin", "nationality_statement", "demonym"],
        "mbti": ["personality_type_identifier", "mbti_type"],
        "political": ["political_affiliation", "ideology_self_id"],
        "age": ["age_statement"],
        "gender": ["gender_indicator"],
    }

    MASK_TOKENS = {
        "age_statement": "[MASK:AGE]",
        "gender_indicator": "[MASK:GENDER]",
        "country_of_origin": "[MASK:NATIONALITY]",
        "nationality_statement": "[MASK:NATIONALITY]",
        "demonym": "[MASK:NATIONALITY]",
        "political_affiliation": "[MASK:POLITICAL]",
        "ideology_self_id": "[MASK:POLITICAL]",
        "personality_type_identifier": "[MASK:MBTI]",
        "mbti_type": "[MASK:MBTI]",
    }

    @classmethod
    def get_labels(cls) -> List[str]:
        labels: List[str] = []
        for v in cls.TARGET_PROMPTS.values():
            labels.extend(v)
        return list(dict.fromkeys(labels))


class GLiNERDetector:
    def __init__(self, model_id: str = "urchade/gliner_large-v2.1", device: str = "cuda", threshold: float = 0.85):
        self.threshold = threshold
        self.model = GLiNER.from_pretrained(model_id, map_location=device)
        self.model.eval()

        # NEW: register typed masks to prevent tokenizer fragmentation
        self._register_special_tokens(list(set(SOBRTaxonomy.MASK_TOKENS.values())))

        # NEW: distractors absorb false positives for non-self mentions
        self.distractors = ["third_person_reference", "fictional_character", "historical_figure"]

        self._packing_config = InferencePackingConfig(max_length=512, streams_per_batch=4) if device == "cuda" else None

    def _register_special_tokens(self, mask_tokens: List[str]) -> None:
        tok = self.model.data_processor.transformer_tokenizer
        tok.add_special_tokens({"additional_special_tokens": mask_tokens})

    def _get_inference_labels(self) -> List[str]:
        return list(dict.fromkeys(SOBRTaxonomy.get_labels() + self.distractors))

    def detect_spans(self, texts: List[str], batch_size: int = 64) -> List[List[PollutionSpan]]:
        raw = self.model.inference(
            texts=texts,
            labels=self._get_inference_labels(),
            flat_ner=True,
            threshold=self.threshold,
            batch_size=batch_size,
            packing_config=self._packing_config,
        )

        tok = self.model.data_processor.transformer_tokenizer
        max_widths = {"age_statement": 5, "gender_indicator": 4, "default": 12}

        out: List[List[PollutionSpan]] = []
        for doc in raw:
            spans: List[PollutionSpan] = []
            for e in doc:
                if e["label"] not in SOBRTaxonomy.get_labels():
                    continue
                max_w = max_widths.get(e["label"], max_widths["default"])
                if len(tok.tokenize(e["text"])) > max_w:
                    continue
                spans.append(PollutionSpan(e["start"], e["end"], e["text"], e["label"], e["score"]))
            out.append(spans)
        return out

    def detect_spans_long(
        self,
        texts: List[str],
        max_chars: int = 1500,
        overlap: int = 200,
        edge_tokens: int = 20,
        batch_size: int = 64,
    ) -> List[List[PollutionSpan]]:
        # Chunk, infer, then drop edge predictions unless at doc boundaries
        chunks: List[str] = []
        meta: List[Tuple[int, int]] = []
        for i, t in enumerate(texts):
            start = 0
            while start < len(t):
                end = min(start + max_chars, len(t))
                chunks.append(t[start:end])
                meta.append((i, start))
                if end == len(t):
                    break
                start = max(0, end - overlap)

        chunk_spans = self.detect_spans(chunks, batch_size=batch_size)

        tok = self.model.data_processor.transformer_tokenizer
        out: List[List[PollutionSpan]] = [[] for _ in texts]
        seen: List[set] = [set() for _ in texts]

        for chunk_text, (doc_i, offset), spans in zip(chunks, meta, chunk_spans):
            enc = tok(chunk_text, return_offsets_mapping=True, add_special_tokens=False)
            offsets = enc["offset_mapping"]
            if len(offsets) > 0:
                left_edge = offsets[min(edge_tokens, len(offsets) - 1)][0]
                right_edge = offsets[max(len(offsets) - edge_tokens - 1, 0)][1]
            else:
                left_edge, right_edge = 0, len(chunk_text)

            chunk_is_start = (offset == 0)
            chunk_is_end = (offset + len(chunk_text) >= len(texts[doc_i]))

            for s in spans:
                if (s.start < left_edge and not chunk_is_start) or (s.end > right_edge and not chunk_is_end):
                    continue
                adj = PollutionSpan(s.start + offset, s.end + offset, s.text, s.label, s.score)
                k = (adj.start, adj.end, adj.label)
                if k in seen[doc_i]:
                    continue
                seen[doc_i].add(k)
                out[doc_i].append(adj)

        for doc_spans in out:
            doc_spans.sort(key=lambda s: s.start)
        return out

    def mask_text(self, text: str, spans: List[PollutionSpan]) -> Tuple[str, List[Dict]]:
        tok = self.model.data_processor.transformer_tokenizer
        for mt in set(SOBRTaxonomy.MASK_TOKENS.values()):
            if len(tok.tokenize(mt)) != 1:
                raise RuntimeError(f"Mask token fragmentation detected for '{mt}'.")

        spans_sorted = sorted(spans, key=lambda s: s.start, reverse=True)
        masked = text
        logs: List[Dict] = []
        for s in spans_sorted:
            if text[s.start:s.end] != s.text:
                raise RuntimeError("Span/text mismatch: offsets are not safe for string splicing.")
            mt = SOBRTaxonomy.MASK_TOKENS.get(s.label, "[MASK:UNKNOWN]")
            masked = masked[:s.start] + mt + masked[s.end:]
            logs.append({
                "span_start": s.start,
                "span_end": s.end,
                "span_text": s.text,
                "entity_type": s.label,
                "confidence": float(s.score),
                "mask_token": mt,
            })
        return masked, logs
```

---

## References

1. GLiNER source code: `urchade/GLiNER` (analyzed version: 0.2.24)
2. Phase A specification: [phaseA_pollution_filtering.md](phaseA_pollution_filtering.md)
3. Implementation plan: [phaseA-D_implementation_plan.md](phaseA-D_implementation_plan.md)
