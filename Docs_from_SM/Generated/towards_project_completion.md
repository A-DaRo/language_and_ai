# What Has To Be Done: Lang-AI

**Neuro-Symbolic Stylometry Pipeline — Project Completion Roadmap**

---

## Abstract

This document specifies the remaining work required to complete the neuro-symbolic stylometry pipeline. Phase A (Pollution Detection & Mitigation) has been implemented; this document catalogs outstanding Phase A improvements, defines the Phase A → Phase D handover contract, and provides a comprehensive specification for Phase D (Constrained Transformer with Causal Head Gating verification).

The pipeline's goal is to train a demographic profiler that learns from **stylometric features** (syntax, function words, discourse markers) rather than **demographic shortcuts** (explicit self-reports like "25M" or "as a woman"). Phase A removes explicit signals via GLiNER masking and implicit linear leakage via LEACE projection. Phase D trains under these constraints and verifies feature attribution via Causal Head Gating.

---

## Introduction: Phase A Implementation Summary

Phase A ("Pollution Detection & Mitigation") has been implemented and provides the foundation for demographic-aware text cleaning. This phase addresses the "Clever Hans" problem where models exploit explicit demographic markers rather than learning true stylometric features.

### Phase A Components

| Component | File | Description |
|-----------|------|-------------|
| **PhaseAPipeline** | [phase_a_pipeline.py](../../src/neuro_stylometry/phase_a_pipeline.py) | Orchestrator coordinating GLiNER detection → masking → LEACE projection |
| **GLiNERDetector** | [pollution_guard/gliner_detector.py](../../src/neuro_stylometry/pollution_guard/gliner_detector.py) | Zero-shot NER using GLiNER-bi-base for demographic span detection |
| **SemanticChunker** | [pollution_guard/semantic_chunker.py](../../src/neuro_stylometry/pollution_guard/semantic_chunker.py) | Sentence-boundary-aware chunking with dynamic budget allocation |
| **SpanMasker** | [pollution_guard/masker.py](../../src/neuro_stylometry/pollution_guard/masker.py) | Typed span replacement (e.g., `[MASK:AGE]`, `[MASK:GENDER]`) |
| **FrozenEmbedder** | [pollution_guard/embedder.py](../../src/neuro_stylometry/pollution_guard/embedder.py) | RoBERTa-base CLS extraction with special token registration |
| **LEACEComputer** | [pollution_guard/leace.py](../../src/neuro_stylometry/pollution_guard/leace.py) | Closed-form projection matrix computation (Strategy C: joint concept matrix) |
| **SOBRDataset** | [data_engine/dataset.py](../../src/neuro_stylometry/data_engine/dataset.py) | Memory-mapped Arrow dataset wrapper |
| **DemographicEncoder** | [pollution_guard/concept_encoding.py](../../src/neuro_stylometry/pollution_guard/concept_encoding.py) | Multi-label concept encoding with missingness indicators |

### Usage

```bash
# Convert raw CSVs to unified Arrow format
python scripts/convert_pandas_to_arrow.py --raw-data-dir datasets --output artifacts/data/sobr.arrow

# Run Phase A pipeline (laptop mode)
python scripts/run_phase_a.py --input artifacts/data/sobr.arrow --output artifacts/phase_a

# Validate handover artifacts
python scripts/validate_phase_a_handover.py artifacts/phase_a
```

### Phase A Outputs

| Artifact | Path | Description |
|----------|------|-------------|
| `clean_dataset.arrow` | `artifacts/phase_a/` | Dataset with populated `post_masked` column |
| `projection_matrix.pt` | `artifacts/phase_a/` | LEACE projection $P$ (768×768, FP32) |
| `pollution_logs.arrow` | `artifacts/phase_a/` | Per-span detection logs (GLiNER output) |

---

## 1. From Phase A (Outstanding TODOs)

### 1.1 Parallelized Chunking (`semantic_chunker.py`)

**Status**: Stub exists, not fully implemented.

The `SemanticChunker` supports a `parallel_chunking_workers` config option but lacks a proper master-worker process routine for HPC environments.

**Required Implementation**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    MASTER PROCESS                               │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  1. Partition documents into N worker chunks               │ │
│  │  2. Spawn workers via multiprocessing.Pool                 │ │
│  │  3. Collect ChunkInfo results + merge offsets              │ │
│  └───────────────────────────────────────────────────────────┘ │
│              │                           │                      │
│              ▼                           ▼                      │
│  ┌─────────────────┐         ┌─────────────────┐               │
│  │  WORKER 0       │   ...   │  WORKER N-1     │               │
│  │  - PySBD init   │         │  - PySBD init   │               │
│  │  - Tokenizer    │         │  - Tokenizer    │               │
│  │  - Chunk docs   │         │  - Chunk docs   │               │
│  └─────────────────┘         └─────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

**File**: `src/neuro_stylometry/pollution_guard/semantic_chunker.py`

**Key considerations**:
- PySBD and HuggingFace tokenizers are not pickle-safe; workers must initialize their own instances
- Use `multiprocessing.get_context("spawn")` for cross-platform safety
- Config keys: `parallel_chunking_workers`, `parallel_chunking_min_texts` (in `conf/base/pipeline.yaml`)

---

### 1.2 Construct `phase_a_metrics.json`

**Status**: Metrics are computed but not persisted as a unified JSON artifact.

The `PhaseAPipeline.run()` method returns metadata in `PhaseArtifacts.metadata`, but does not write a standalone `phase_a_metrics.json` file. This file is required for:
1. Visualization generation (notebooks + runtime)
2. Experiment tracking (W&B, MLflow)
3. Handover validation (explicit recall, amnesic drop thresholds)

**Required Schema** (`phase_a_metrics.json`):

```json
{
  "timestamp": "2026-01-08T12:00:00Z",
  "config_hash": "sha256:...",
  "dataset": {
    "input_path": "artifacts/data/sobr.arrow",
    "num_samples": 50000,
    "num_masked": 48500
  },
  "gliner": {
    "model": "knowledgator/gliner-bi-base-v1.0",
    "confidence_threshold": 0.60,
    "explicit_recall": {
      "overall": 0.92,
      "per_column": {
        "birth_year": 0.95,
        "female": 0.88,
        "nationality": 0.91
      }
    },
    "spans_detected": 12500,
    "spans_per_sample_mean": 0.25
  },
  "leace": {
    "embedding_dim": 768,
    "regularization": 1e-5,
    "idempotence_error": 1.2e-7,
    "condition_number": 1.0
  },
  "probe": {
    "amnesic_drop": {
      "overall": 0.35,
      "per_column": {
        "female": 0.42,
        "birth_year": 0.31
      }
    },
    "acc_before": 0.72,
    "acc_after": 0.47
  },
  "timing": {
    "gliner_seconds": 1200.5,
    "embedding_seconds": 450.2,
    "leace_seconds": 30.1,
    "total_seconds": 1680.8
  }
}
```

**Wiring**: Modify `PhaseAPipeline.run()` to write this JSON alongside other artifacts.

---

### 1.3 Rewrite Notebook Visualizations → `visualizations_phase_a.py`

**Status**: Visualizations exist in notebooks; no reusable Python module.

The notebooks `02_gliner_taxonomy_tuning.ipynb` and `03_leace_visualization.ipynb` contain visualization code that should be refactored into a reusable module for:
1. Runtime report generation (CLI: `--generate-report`)
2. CI/CD artifact generation
3. Consistent styling across experiments

**Target File**: `src/neuro_stylometry/evaluation/visualizations_phase_a.py`

**Required Functions**:

| Function | Source Notebook | Description |
|----------|-----------------|-------------|
| `plot_pca_before_after()` | 03_leace | PCA scatter comparing embeddings colored by demographic |
| `plot_amnesic_drop_bars()` | 03_leace | Bar chart of probe accuracy before/after LEACE |
| `plot_singular_values()` | 03_leace | Projection matrix singular value spectrum |
| `plot_demographic_score_matrix()` | 03_leace | Lower-triangle scatter matrix of demographic scores |
| `plot_cross_demographic_heatmap()` | 03_leace | Heatmap of cross-demographic probe accuracies |
| `plot_embedding_norm_distribution()` | 03_leace | Histogram comparing embedding norms |
| `plot_gliner_confidence_histogram()` | 02_gliner | Confidence score distribution per entity type |
| `plot_span_length_distribution()` | 02_gliner | Distribution of detected span lengths |
| `generate_phase_a_report()` | — | Orchestrator calling all above + saving to PDF/HTML |

**Wiring**: 
- Input: `phase_a_metrics.json` + `projection_matrix.pt` + `pollution_logs.arrow`
- Output: `artifacts/reports/phase_a_report.html` (or PDF)

---

## 2. Phase A → Phase D Transition

### 2.1 Required Handover Artifacts

Phase D training **requires** the following artifacts from Phase A:

| Artifact | Format | Contract |
|----------|--------|----------|
| **`post_masked`** column | Arrow string column | Every row must have non-null masked text; typed masks like `[MASK:AGE]` replace explicit demographic spans |
| **`projection_matrix.pt`** | PyTorch tensor (768×768, FP32) | Idempotence: $\|P^2 - P\|_F / \|P\|_F < 10^{-5}$; stored as `torch.save(P, path)` |

**Validation**: Run `scripts/validate_phase_a_handover.py artifacts/phase_a` before proceeding to Phase D.

---

### 2.2 Possible Improvements (Phase A Tuning)

Before running Phase D at scale, consider tuning these Phase A parameters:

| Parameter | Current | Location | Notes |
|-----------|---------|----------|-------|
| `confidence_threshold` | 0.60 | `conf/base/pipeline.yaml` | Lower → more spans detected (higher recall, lower precision). Raise to 0.70–0.85 if false positives contaminate stylistic text. |
| `width` (per entity) | varies | `conf/base/gliner_taxonomy.yaml` | Maximum token width for detected spans. Current defaults: AGE=6, GENDER=4, NATIONALITY=8. **Consider removing hard caps** or raising significantly—GLiNER naturally constrains span length. |
| `explicit_recall_threshold` | 0.80 | `conf/base/pipeline.yaml` | Quality gate; raise to 0.90+ for stricter enforcement. |
| `amnesic_drop_threshold` | 0.30 | `conf/base/pipeline.yaml` | Target drop in probe accuracy; raise if linear leakage persists. |

---

### 2.3 ⚠️ CRITICAL: Tokenizer Alignment for Phase D

**Problem**: The Phase D tokenizer must treat typed mask tokens (e.g., `[MASK:AGE]`, `[MASK:GENDER]`) as **single tokens**, not split them into subwords.

**Why this matters**:
- If `[MASK:AGE]` is tokenized as `["[", "MASK", ":", "AGE", "]"]` (5 tokens), the Affine Guard layer projects each subword embedding independently, destroying the semantic coherence of the mask.
- The Phase D model could attend to partial mask fragments, re-introducing demographic signal.

**Solution**: Register mask tokens as `additional_special_tokens` in the Phase D tokenizer.

```python
from transformers import AutoTokenizer

# Load base tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Register Phase A mask tokens
mask_tokens = ["[MASK:AGE]", "[MASK:GENDER]", "[MASK:NATIONALITY]", 
               "[MASK:POLITICAL]", "[MASK:MBTI]"]
tokenizer.add_special_tokens({"additional_special_tokens": mask_tokens})

# Resize model embeddings
model.resize_token_embeddings(len(tokenizer))

# Verify single-token encoding
for token in mask_tokens:
    ids = tokenizer.encode(token, add_special_tokens=False)
    assert len(ids) == 1, f"{token} split into {len(ids)} tokens!"
```

**Phase A already does this** in `FrozenEmbedder._register_special_tokens()`. Phase D must replicate this step.

---
## 3. Phase D: Constrained Transformer Architecture

Phase D trains a multi-task demographic classifier under the **Affine Guard** constraint. The key insight is that by projecting embeddings through the LEACE matrix $P$ (computed in Phase A), we **force gradient descent to explore non-linear stylometric features** rather than linear demographic shortcuts.

### 3.0 High-Level Data Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         PHASE D: END-TO-END FLOW                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   post_masked                                                                │
│   "I think [MASK:AGE] is fine, but the policy seems..."                     │
│        │                                                                     │
│        ▼                                                                     │
│   ┌─────────────────┐                                                        │
│   │   TOKENIZER     │  (with registered mask tokens)                        │
│   │   ───────────── │                                                        │
│   │   → input_ids   │                                                        │
│   │   → attention   │                                                        │
│   └─────────────────┘                                                        │
│        │                                                                     │
│        ▼                                                                     │
│   ┌─────────────────┐                                                        │
│   │  EMBEDDING      │  h₀ = Embed(tokens) + PositionalEmbed                 │
│   │  (trainable)    │  Shape: (batch, seq_len, 768)                         │
│   └─────────────────┘                                                        │
│        │                                                                     │
│        ▼                                                                     │
│   ┌─────────────────┐                                                        │
│   │  AFFINE GUARD   │  h_proj = P @ h₀  (P frozen, non-trainable)           │
│   │  ★ FROZEN ★     │  Removes linear demographic signal                    │
│   └─────────────────┘                                                        │
│        │                                                                     │
│        ▼                                                                     │
│   ┌─────────────────┐                                                        │
│   │  TRANSFORMER    │  12 encoder blocks                                    │
│   │  ENCODER        │  Multi-head attention + FFN                           │
│   └─────────────────┘                                                        │
│        │                                                                     │
│        ▼                                                                     │
│   ┌─────────────────┐                                                        │
│   │  CLS POOLING    │  Extract [CLS] representation                         │
│   └─────────────────┘                                                        │
│        │                                                                     │
│        ▼                                                                     │
│   ┌─────────────────┐                                                        │
│   │  MULTI-TASK     │  Per-attribute classification heads                   │
│   │  HEADS          │  loss = Σ (weight_a × CE(logits_a, y_a))              │
│   └─────────────────┘                                                        │
│        │                                                                     │
│        ▼                                                                     │
│   Predictions: {gender, age_bin, nationality, political, mbti_*}            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

### 3.1 Tokenizer (Input Processing)

**Input**: `post_masked` column from `clean_dataset.arrow`

**Implementation File**: (to create) `src/neuro_stylometry/stylometry_net/tokenizer.py`

**Requirements**:
1. Use RoBERTa tokenizer (`roberta-base` or `roberta-large`)
2. Register mask tokens from Phase A taxonomy as additional special tokens
3. Apply padding/truncation to `max_length` (default: 512)
4. Return `input_ids`, `attention_mask`

```python
@dataclass
class TokenizerConfig:
    model_name: str = "roberta-base"
    max_length: int = 512
    mask_tokens: List[str] = field(default_factory=lambda: [
        "[MASK:AGE]", "[MASK:GENDER]", "[MASK:NATIONALITY]",
        "[MASK:POLITICAL]", "[MASK:MBTI]"
    ])
```

---

### 3.2 Embedding Layer

**Operation**: $h_0 = \text{Embed}(\text{tokens}) + \text{PositionalEmbed}$

**Details**:
- Token embeddings: Lookup table of shape `(vocab_size, 768)`
- Positional embeddings: Learned embeddings of shape `(max_length, 768)` (RoBERTa style) or sinusoidal (BERT style)
- Output shape: `(batch_size, seq_len, 768)`

**Note**: Embedding weights are **trainable** in Phase D (unlike Phase A's frozen embedder). The Affine Guard projects embeddings but does not freeze them.

---

### 3.3 Affine Guard Layer

**Operation**: $h_{\text{proj}} = P \cdot h_0$

Where $P$ is the LEACE projection matrix from Phase A (`projection_matrix.pt`).

**Implementation File**: `src/neuro_stylometry/stylometry_net/affine_guard.py` (stub exists)

**Required Implementation**:

```python
class AffineGuard(nn.Module):
    """
    Affine Guard layer applying LEACE projection to embeddings.
    
    The projection matrix P is FROZEN (requires_grad=False).
    Gradients flow THROUGH P to update upstream embeddings,
    but P itself is never updated.
    
    This forces the model to learn features orthogonal to the
    demographic subspace erased by LEACE.
    """
    
    def __init__(self, projection_matrix: torch.Tensor):
        super().__init__()
        # Register as buffer (not parameter) → no grad updates
        self.register_buffer("P", projection_matrix)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Embeddings of shape (batch, seq_len, hidden_dim)
        Returns:
            Projected embeddings of same shape
        """
        # P is (hidden_dim, hidden_dim)
        # h is (batch, seq_len, hidden_dim)
        # Result: (batch, seq_len, hidden_dim)
        return torch.einsum("bsd,dd->bsd", h, self.P)
    
    @classmethod
    def from_checkpoint(cls, path: Path) -> "AffineGuard":
        P = torch.load(path, map_location="cpu")
        return cls(P)
```

**Key Properties**:
- `P` is registered as a **buffer**, not a parameter
- `requires_grad=False` on `P` (automatic for buffers)
- Gradients flow through `P` via chain rule but do not update `P`
- Projection is applied per-token (not just CLS)

---

### 3.4 Transformer Encoder

**Operation**: For $l = 1$ to $L$ (default $L=12$):

$$h_l = \text{EncoderBlock}_l(h_{l-1})$$

Where each encoder block is:

$$h' = \text{LayerNorm}(h + \text{MultiHeadAttn}(h))$$
$$h_{\text{out}} = \text{LayerNorm}(h' + \text{FFN}(h'))$$

**Implementation File**: `src/neuro_stylometry/stylometry_net/transformer.py` (stub exists)

**Options**:
1. **Initialize from pretrained RoBERTa** (recommended): Load `RobertaModel.from_pretrained()`, insert AffineGuard after embeddings
2. **Train from scratch**: Use `RobertaConfig` with random initialization (not recommended—requires more data)

**Architecture Integration**:

```python
class AffineGuardTransformer(nn.Module):
    def __init__(
        self,
        model_name: str = "roberta-base",
        projection_matrix_path: Path = None,
        mask_tokens: List[str] = None,
    ):
        super().__init__()
        
        # Load pretrained RoBERTa
        self.roberta = RobertaModel.from_pretrained(model_name)
        
        # Register mask tokens and resize embeddings
        if mask_tokens:
            self._register_mask_tokens(mask_tokens)
        
        # Insert Affine Guard
        if projection_matrix_path:
            self.affine_guard = AffineGuard.from_checkpoint(projection_matrix_path)
        else:
            self.affine_guard = None  # No projection (baseline)
    
    def forward(self, input_ids, attention_mask):
        # Get embeddings from RoBERTa embedding layer
        embeddings = self.roberta.embeddings(input_ids)
        
        # Apply Affine Guard projection
        if self.affine_guard is not None:
            embeddings = self.affine_guard(embeddings)
        
        # Pass through encoder (skip embedding layer)
        encoder_output = self.roberta.encoder(
            embeddings,
            attention_mask=self._extend_attention_mask(attention_mask),
        )
        
        return encoder_output.last_hidden_state
```

---

### 3.5 CLS Pooling + Multi-Task Classification Heads

**CLS Extraction**: Take the hidden state at position 0 (the `[CLS]` token):

$$h_{\text{CLS}} = h_L[:, 0, :]$$

**Multi-Task Heads**: For each demographic attribute $a \in \{\text{gender}, \text{age\_bin}, \text{nationality}, ...\}$:

$$\text{logits}_a = W_a \cdot h_{\text{CLS}} + b_a$$
$$\mathcal{L}_a = \text{CrossEntropy}(\text{logits}_a, y_a)$$

**Total Loss**:

$$\mathcal{L}_{\text{total}} = \sum_a w_a \cdot \mathcal{L}_a$$

Where $w_a$ are task weights (e.g., inverse class frequency or uniform).

**Implementation File**: `src/neuro_stylometry/stylometry_net/classification_head.py` (stub exists)

**Required Implementation**:

```python
class MultiTaskHead(nn.Module):
    """
    Multi-task classification heads for demographic prediction.
    
    Each attribute gets its own linear head.
    Handles missing labels via masked loss computation.
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        task_configs: Dict[str, int] = None,  # {task_name: num_classes}
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create per-task classification heads
        self.heads = nn.ModuleDict({
            task: nn.Linear(hidden_dim, num_classes)
            for task, num_classes in task_configs.items()
        })
    
    def forward(self, cls_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        cls_embedding = self.dropout(cls_embedding)
        return {
            task: head(cls_embedding)
            for task, head in self.heads.items()
        }
    
    def compute_loss(
        self,
        logits: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        weights: Dict[str, float] = None,
    ) -> torch.Tensor:
        """Compute weighted multi-task loss, ignoring missing labels."""
        weights = weights or {t: 1.0 for t in logits}
        total_loss = 0.0
        
        for task, task_logits in logits.items():
            task_labels = labels[task]
            
            # Mask for valid (non-missing) labels
            valid_mask = task_labels >= 0
            if valid_mask.sum() == 0:
                continue
            
            task_loss = F.cross_entropy(
                task_logits[valid_mask],
                task_labels[valid_mask],
            )
            total_loss += weights[task] * task_loss
        
        return total_loss
```

**Task Configuration** (from SOBR schema):

| Task | Classes | Notes |
|------|---------|-------|
| `female` | 2 | Binary: male/female |
| `birth_year` | 5–10 | Binned into age groups |
| `nationality` | ~30 | Multi-class |
| `political_leaning` | 2–4 | Left/Right (or 4-way compass) |
| `extrovert` | 2 | MBTI E/I |
| `sensing` | 2 | MBTI S/N |
| `feeling` | 2 | MBTI F/T |
| `judging` | 2 | MBTI J/P |

---

### 3.6 Causal Head Gating (Verification)

**Purpose**: After training, verify that the model learned stylometric features (syntax, function words) rather than residual demographic signals.

**Method** (from Nam et al., 2025):
1. Freeze the trained AffineGuardTransformer
2. Learn soft gates $g_{\ell,h} \in [0, 1]$ for each attention head
3. Categorize heads as **Facilitating**, **Interfering**, or **Irrelevant**
4. Compute **Stylometric Validity Score (SVS)**

**Implementation File**: `src/neuro_stylometry/stylometry_net/chg_verifier.py` (stub exists)

**Required Implementation**:

```python
class CausalHeadGating(nn.Module):
    """
    Causal Head Gating for attention head importance analysis.
    
    Learns continuous gates g_{l,h} ∈ [0,1] for each head.
    Gates are applied as soft ablations during forward pass.
    """
    
    def __init__(
        self,
        num_layers: int = 12,
        num_heads: int = 12,
        init_value: float = 1.0,
        regularization: float = 0.01,
    ):
        super().__init__()
        # Learnable gate parameters (one per head)
        self.gate_logits = nn.Parameter(
            torch.full((num_layers, num_heads), init_value)
        )
        self.regularization = regularization
    
    @property
    def gates(self) -> torch.Tensor:
        """Gate values in [0, 1] via sigmoid."""
        return torch.sigmoid(self.gate_logits)
    
    def forward(
        self,
        attention_outputs: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Apply gates to attention head outputs."""
        gated = []
        for l, attn_out in enumerate(attention_outputs):
            # attn_out: (batch, num_heads, seq, head_dim)
            g = self.gates[l].view(1, -1, 1, 1)
            gated.append(attn_out * g)
        return gated
    
    def regularization_loss(self) -> torch.Tensor:
        """L1 sparsity regularization on gate values."""
        return self.regularization * self.gates.abs().mean()
    
    def classify_heads(self, threshold: float = 0.5) -> Dict[str, List[Tuple[int, int]]]:
        """Classify heads into Facilitating/Interfering/Irrelevant."""
        gates = self.gates.detach().cpu().numpy()
        
        facilitating = []
        interfering = []
        irrelevant = []
        
        for l in range(gates.shape[0]):
            for h in range(gates.shape[1]):
                g = gates[l, h]
                if g > threshold:
                    facilitating.append((l, h))
                elif g < 1 - threshold:
                    interfering.append((l, h))
                else:
                    irrelevant.append((l, h))
        
        return {
            "facilitating": facilitating,
            "interfering": interfering,
            "irrelevant": irrelevant,
        }
```

**Stylometric Validity Score (SVS)**:

$$\text{SVS} = \frac{\text{Stylometric Heads}}{\text{Facilitating Heads}}$$

Where "Stylometric Heads" attend primarily to:
- Function words (determiners, prepositions, conjunctions)
- Punctuation patterns
- Syntactic markers (auxiliaries, modals)

**NOT** to:
- Content words (nouns, proper nouns)
- Pronouns (he/she)
- Demographic markers

**SVS Calculation File**: `src/neuro_stylometry/stylometry_net/svs_calculator.py` (stub exists)

---

## 4. Expected Outputs & Deliverables

This section exhaustively lists all artifacts that must be produced for project completion.

### 4.1 Experiment Design: Baseline vs. Constrained

**Critical**: Results are meaningless without a **baseline comparison**. The pipeline must train and evaluate **two models**:

| Model | Affine Guard | Description |
|-------|--------------|-------------|
| **Model A (Dirty Baseline)** | ❌ Disabled | Train on raw `post` column (no masking, no projection) |
| **Model B (Clean Constrained)** | ✅ Enabled | Train on `post_masked` with LEACE projection $P$ |

Both models use identical hyperparameters (learning rate, batch size, epochs). The only difference is input preprocessing and the Affine Guard layer.

---

### 4.2 Training Artifacts

| Artifact | Path | Format | Description |
|----------|------|--------|-------------|
| `model_baseline.pt` | `artifacts/phase_d/baseline/` | PyTorch checkpoint | Dirty baseline model weights |
| `model_constrained.pt` | `artifacts/phase_d/constrained/` | PyTorch checkpoint | Affine Guard constrained model weights |
| `training_log_baseline.jsonl` | `artifacts/phase_d/baseline/` | JSON Lines | Per-step loss, accuracy, learning rate |
| `training_log_constrained.jsonl` | `artifacts/phase_d/constrained/` | JSON Lines | Per-step loss, accuracy, learning rate |
| `config_baseline.yaml` | `artifacts/phase_d/baseline/` | YAML | Full training configuration snapshot |
| `config_constrained.yaml` | `artifacts/phase_d/constrained/` | YAML | Full training configuration snapshot |

---

### 4.3 Evaluation Metrics

| Metric | Scope | Target | File |
|--------|-------|--------|------|
| **Per-task Accuracy** | Both models, per demographic | Report | `phase_d_metrics.json` |
| **Per-task F1 (macro)** | Both models, per demographic | Report | `phase_d_metrics.json` |
| **Confusion Matrices** | Both models, per demographic | Visualization | `confusion_matrices.png` |
| **Calibration Curves** | Both models, per demographic | Visualization | `calibration_curves.png` |
| **Δ Accuracy** | Constrained − Baseline | Expect negative (lower shortcut reliance) | `phase_d_metrics.json` |

---

### 4.4 Causal Head Gating Outputs

| Artifact | Path | Format | Description |
|----------|------|--------|-------------|
| `chg_gates_baseline.pt` | `artifacts/phase_d/chg/` | PyTorch | Learned gates for baseline model |
| `chg_gates_constrained.pt` | `artifacts/phase_d/chg/` | PyTorch | Learned gates for constrained model |
| `head_classification_baseline.json` | `artifacts/phase_d/chg/` | JSON | Facilitating/Interfering/Irrelevant per head |
| `head_classification_constrained.json` | `artifacts/phase_d/chg/` | JSON | Facilitating/Interfering/Irrelevant per head |
| `svs_baseline.json` | `artifacts/phase_d/chg/` | JSON | Stylometric Validity Score for baseline |
| `svs_constrained.json` | `artifacts/phase_d/chg/` | JSON | Stylometric Validity Score for constrained |

---

### 4.5 Attention Analysis Outputs

| Artifact | Description |
|----------|-------------|
| `attention_heatmaps_baseline.html` | Interactive attention visualization (BertViz or custom) |
| `attention_heatmaps_constrained.html` | Interactive attention visualization |
| `pos_attention_distribution.png` | POS tag distribution of tokens attended to by facilitating heads |
| `token_type_attention_comparison.png` | Bar chart: function words vs. content words vs. pronouns |

**Hypothesis to verify**:
- Baseline facilitating heads attend to **pronouns** and **proper nouns** (demographic shortcuts)
- Constrained facilitating heads attend to **determiners**, **prepositions**, **punctuation** (stylometry)

---

### 4.6 Visualizations (Phase D Report)

| Visualization | Description |
|---------------|-------------|
| `learning_curves.png` | Train/val loss over epochs (both models) |
| `accuracy_comparison_bar.png` | Per-task accuracy comparison (baseline vs. constrained) |
| `gate_heatmap_baseline.png` | 12×12 heatmap of gate values (layers × heads) |
| `gate_heatmap_constrained.png` | 12×12 heatmap of gate values |
| `gate_distribution.png` | Histogram of gate values (both models overlaid) |
| `svs_comparison.png` | Bar chart comparing SVS scores |
| `tsne_cls_embeddings.png` | t-SNE of CLS embeddings colored by demographic (both models) |

---

### 4.7 Final Report Artifacts

| Artifact | Path | Format |
|----------|------|--------|
| `phase_a_report.html` | `artifacts/reports/` | HTML report with all Phase A visualizations |
| `phase_d_report.html` | `artifacts/reports/` | HTML report with all Phase D visualizations |
| `experiment_summary.json` | `artifacts/reports/` | Machine-readable summary of all metrics |
| `experiment_summary.md` | `artifacts/reports/` | Human-readable summary with key findings |

---

### 4.8 Summary Metrics JSON Schema

**File**: `artifacts/reports/experiment_summary.json`

```json
{
  "experiment_id": "exp_20260108_120000",
  "timestamp": "2026-01-08T12:00:00Z",
  "phase_a": {
    "explicit_recall": 0.92,
    "amnesic_drop": 0.35,
    "idempotence_error": 1.2e-7
  },
  "phase_d": {
    "baseline": {
      "accuracy": {
        "female": 0.78,
        "birth_year": 0.65,
        "nationality": 0.42
      },
      "svs": 0.25,
      "facilitating_heads": 45,
      "pronoun_attention_ratio": 0.38
    },
    "constrained": {
      "accuracy": {
        "female": 0.62,
        "birth_year": 0.58,
        "nationality": 0.38
      },
      "svs": 0.72,
      "facilitating_heads": 28,
      "pronoun_attention_ratio": 0.08
    },
    "delta": {
      "accuracy_female": -0.16,
      "svs_improvement": 0.47,
      "pronoun_attention_reduction": -0.30
    }
  },
  "hypothesis_confirmed": true,
  "notes": "Constrained model shows 47% improvement in SVS, confirming stylometric feature learning"
}
```

---

## 5. Implementation Checklist

### Phase A Completion
- [ ] Implement parallelized chunking in `semantic_chunker.py`
- [ ] Generate `phase_a_metrics.json` in pipeline run
- [ ] Create `visualizations_phase_a.py` module
- [ ] Wire report generation to CLI

### Phase D Implementation
- [ ] Implement `AffineGuard` module
- [ ] Implement `AffineGuardTransformer` integration
- [ ] Implement `MultiTaskHead` with masked loss
- [ ] Implement `CausalHeadGating` verifier
- [ ] Implement `SVSCalculator`
- [ ] Create training loop with proper logging
- [ ] Create evaluation pipeline

### Experiment Execution
- [ ] Run baseline training (no Affine Guard)
- [ ] Run constrained training (with Affine Guard)
- [ ] Run CHG analysis on both models
- [ ] Generate all visualizations
- [ ] Compile final reports

### Validation
- [ ] Verify tokenizer mask token registration
- [ ] Verify Affine Guard projection correctness
- [ ] Verify CHG gate learning convergence
- [ ] Validate SVS calculation methodology

---

## 6. Simple Optimizations (No External Dependencies)

The following optimizations can be applied without additional C++ compilation or complex dependencies:

| Optimization | Location | Expected Gain |
|--------------|----------|---------------|
| **Mixed Precision (FP16)** | Training loop | 2× throughput on GPU |
| **Gradient Checkpointing** | Transformer encoder | 50% memory reduction |
| **torch.compile** | Model forward pass | 10-30% speedup (PyTorch 2.0+) |
| **DataLoader pinned memory** | `dataloader.py` | Faster CPU→GPU transfer |
| **Prefetch factor=2** | DataLoader | Overlap data loading with compute |

**Already configured** in `conf/base/pipeline.yaml`:
- `pin_memory: true`
- `prefetch_factor: 2`
- `enable_torch_compile: false` (enable manually for PyTorch 2.0+)

---

## Appendix A: File Structure (To Create)

```
src/neuro_stylometry/
├── stylometry_net/
│   ├── __init__.py                 # Exists
│   ├── affine_guard.py             # TODO: Implement AffineGuard
│   ├── transformer.py              # TODO: Implement AffineGuardTransformer
│   ├── classification_head.py      # TODO: Implement MultiTaskHead
│   ├── chg_verifier.py             # TODO: Implement CausalHeadGating
│   ├── svs_calculator.py           # TODO: Implement SVS calculation
│   └── tokenizer.py                # TODO: Create tokenizer wrapper
├── training/
│   ├── __init__.py                 # Exists
│   ├── trainer.py                  # TODO: Training loop
│   └── evaluator.py                # TODO: Evaluation pipeline
├── evaluation/
│   ├── __init__.py                 # Exists
│   ├── visualizations_phase_a.py   # TODO: Phase A viz functions
│   └── visualizations_phase_d.py   # TODO: Phase D viz functions
scripts/
├── run_phase_d.py                  # TODO: Phase D entry point
└── run_verification.py             # TODO: CHG verification entry point
```

---

## Appendix B: Configuration Files (Extend)

**`conf/base/training.yaml`** — extend with:

```yaml
phase_d:
  model:
    base_model: "roberta-base"
    use_affine_guard: true
    projection_matrix_path: "${oc.env:PHASE_A_OUTPUT}/projection_matrix.pt"
    mask_tokens:
      - "[MASK:AGE]"
      - "[MASK:GENDER]"
      - "[MASK:NATIONALITY]"
      - "[MASK:POLITICAL]"
      - "[MASK:MBTI]"
  
  tasks:
    female:
      num_classes: 2
      weight: 1.0
    birth_year:
      num_classes: 6  # Binned: <20, 20-29, 30-39, 40-49, 50-59, 60+
      weight: 1.0
    nationality:
      num_classes: 30
      weight: 0.5  # Lower weight due to class imbalance
    political_leaning:
      num_classes: 2
      weight: 1.0
    extrovert:
      num_classes: 2
      weight: 0.5
    sensing:
      num_classes: 2
      weight: 0.5
    feeling:
      num_classes: 2
      weight: 0.5
    judging:
      num_classes: 2
      weight: 0.5

chg:
  num_epochs: 10
  learning_rate: 1e-3
  regularization: 0.01
  gate_init: 1.0
  classification_threshold: 0.5
```

---

*Document generated: 2026-01-08*
*Pipeline version: Phase A complete, Phase D specification*