# Neuro-Symbolic Stylometry Pipeline: OOP Implementation Plan & Software Specification

**Document Type:** Technical Specification  
**Target Audience:** Senior Python/ML Engineer  
**Execution Environment:** Dual-Mode (Laptop Debug / HPC Production)  
**Data Substrate:** Apache Arrow via Hugging Face Datasets  

---

## 1. Introduction and Context

### 1.1 Architectural Goal: The "Filter-Project-Verify" Pipeline

This document specifies a rigorous Object-Oriented implementation of a **Neuro-Symbolic Stylometry Pipeline** that integrates two distinct computational phases:

1. **Phase A (Pollution Mitigation):** A hybrid neuro-symbolic cleaning system that removes demographic shortcuts from text data, ensuring that downstream models cannot rely on explicit self-identification tokens or implicit linear leakage in embedding space.

2. **Phase D (Constrained Neural Stylometry):** A Transformer-based authorship profiling system that is architecturally constrained to learn stylometric features (syntax, function words, punctuation patterns) rather than demographic shortcuts.

The pipeline architecture follows a **"Filter-then-Project-then-Verify"** paradigm:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              FILTER-PROJECT-VERIFY PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│   ┌─────────────┐    ┌───────────────────────────────────────────────────┐                   │
│   │ Raw SOBR    │    │                PHASE A: POLLUTION GUARD            │                  │
│   │ Corpus      │───►│                                                    │                  │
│   │ (~390k)     │    │  ┌─────────────┐         ┌──────────────────┐     │                  │
│   └─────────────┘    │  │   GLiNER    │────────►│     LEACE        │     │                  │
│                      │  │  (Symbolic  │  Masked │   (Geometric     │     │                  │
│                      │  │   Masking)  │   Data  │   Projection)    │     │                  │
│                      │  └─────────────┘         └────────┬─────────┘     │                  │
│                      │                                   │               │                  │
│                      │         ┌─────────────────────────┼───────────┐   │                  │
│                      │         │                         ▼           │   │                  │
│                      │         │  ┌─────────────────────────────┐   │   │                  │
│                      │         │  │Artifacts:                    │   │   │                  │
│                      │         │  │ • clean_dataset.arrow        │   │   │                  │
│                      │         │  │ • projection_matrix.pt (P)   │   │   │                  │
│                      │         │  │ • pollution_logs.json        │   │   │                  │
│                      │         │  └─────────────────────────────┘   │   │                  │
│                      │         └─────────────────────────────────────┘   │                  │
│                      └───────────────────────────────────────────────────┘                  │
│                                              │                                               │
│                                              ▼                                               │
│   ┌──────────────────────────────────────────────────────────────────────────────────────┐  │
│   │                          PHASE D: CONSTRAINED TRANSFORMER                             │  │
│   │                                                                                        │  │
│   │   ┌──────────────┐    ┌───────────────────┐    ┌────────────────────────────────┐    │  │
│   │   │  Embedding   │───►│   AFFINE GUARD    │───►│  Transformer Encoder Blocks    │    │  │
│   │   │    Layer     │    │   h_proj = P·h_0  │    │  (RoBERTa/BERT Architecture)   │    │  │
│   │   └──────────────┘    │   (Fixed, Frozen) │    └────────────────────────────────┘    │  │
│   │                       └───────────────────┘                    │                      │  │
│   │                                                                ▼                      │  │
│   │                                                     ┌────────────────────┐           │  │
│   │                                                     │ Classification     │           │  │
│   │                                                     │ Head (Author       │           │  │
│   │                                                     │ Profiling)         │           │  │
│   │                                                     └────────────────────┘           │  │
│   │                                                                │                      │  │
│   └────────────────────────────────────────────────────────────────┼──────────────────────┘  │
│                                                                    ▼                         │
│   ┌──────────────────────────────────────────────────────────────────────────────────────┐  │
│   │                         VERIFICATION: CAUSAL HEAD GATING                              │  │
│   │                                                                                        │  │
│   │   • Learn gate parameters g_{l,h} ∈ [0,1] for each attention head                     │  │
│   │   • Identify "Facilitating" heads (high g) vs "Irrelevant" heads (low g)              │  │
│   │   • Compute Stylometric Validity Score (SVS)                                          │  │
│   │   • Validate: Are facilitating heads attending to function words or content words?    │  │
│   │                                                                                        │  │
│   └──────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Scope Definition: Raw SOBR to Verified Stylometric Transformer

The implementation scope encompasses the complete data lifecycle:

| Stage | Input | Processing | Output |
|-------|-------|------------|--------|
| **Data Ingestion** | Raw SOBR Pandas DataFrames (8 demographic files) | Schema consolidation, Arrow conversion, memory-mapping | Unified `sobr.arrow` file |
| **Symbolic Cleaning** | Raw text spans | GLiNER zero-shot NER with demographic taxonomy | Masked text with `[MASK:TYPE]` tokens |
| **Geometric Projection** | Masked embeddings + demographic labels | LEACE covariance computation and nullspace projection | Projection matrix $\mathbf{P}$ (d×d tensor) |
| **Constrained Training** | Cleaned data + frozen $\mathbf{P}$ | Affine Guard injection before encoder blocks | Trained stylometric classifier |
| **Causal Verification** | Trained model | Causal Head Gating (CHG) analysis | Stylometric Validity Score (SVS) |

**Explicit Scope Boundaries:**

- **In Scope:**
  - Complete Phase A pollution detection and mitigation pipeline
  - Complete Phase D constrained Transformer training
  - Causal Head Gating verification framework
  - Dual-mode execution (Laptop/HPC) with seamless switching
  - Apache Arrow data substrate with zero-copy memory mapping
  - NUMA-aware parallelism for AMD EPYC systems
  - BF16/TF32 mixed-precision training for NVIDIA A100/H100

- **Out of Scope:**
  - Multi-node distributed training (DeepSpeed/Megatron-LM)
  - Flash Attention integration for sequences >2048 tokens
  - INT8 quantization for inference deployment
  - Online/incremental LEACE updates

### 1.3 The Dual-Mode Philosophy: Laptop Debug vs. HPC Production

A fundamental design constraint of this implementation is **hardware-agnosticism in design** combined with **hardware-awareness in execution**. The same codebase must execute correctly on:

| Environment | Hardware Profile | Execution Characteristics |
|-------------|------------------|---------------------------|
| **Laptop Debug Mode** | Intel i7 CPU, 16GB RAM, NVIDIA A1000 Mobile (4-8GB VRAM) | Eager execution, mini-batch covariance accumulation, FP16 mixed precision with gradient scaling, 2 DataLoader workers, 10k sample subsets |
| **HPC Production Mode** | AMD EPYC Rome/Genoa (64-128 cores), 512GB DDR5, NVIDIA A100/H100 (80GB HBM2e) | Graph execution via CUDA Graphs, full-batch GPU covariance, BF16 Tensor Cores at 312-989 TFLOPS, 8+ persistent DataLoader workers with prefetching, full dataset processing |

**The Dual-Mode Principle:**

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           DUAL-MODE EXECUTION MODEL                             │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Configuration Layer (Runtime Detection)                                       │
│   ┌─────────────────────────────────────────────────────────────────────────┐  │
│   │  HardwareDetector                                                        │  │
│   │   • Query GPU name, VRAM capacity                                        │  │
│   │   • Query CPU core count, NUMA topology                                  │  │
│   │   • Select execution profile: HPC | LAPTOP | GENERIC                     │  │
│   └─────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                            │
│                     ┌──────────────┴──────────────┐                            │
│                     ▼                              ▼                            │
│   ┌─────────────────────────────┐   ┌─────────────────────────────┐            │
│   │    HPC Execution Path       │   │   Laptop Execution Path     │            │
│   ├─────────────────────────────┤   ├─────────────────────────────┤            │
│   │ • CUDA Graph capture        │   │ • Eager execution           │            │
│   │ • Smart bucketing           │   │ • Simple batching           │            │
│   │ • Full-batch LEACE on GPU   │   │ • Mini-batch accumulation   │            │
│   │ • torch.compile fusion      │   │ • No compilation            │            │
│   │ • BF16 Tensor Cores         │   │ • FP16 + GradScaler         │            │
│   │ • NUMA-pinned workers       │   │ • Default worker affinity   │            │
│   │ • 256+ batch size           │   │ • 16-32 batch size          │            │
│   │ • 8+ persistent workers     │   │ • 2 workers                 │            │
│   └─────────────────────────────┘   └─────────────────────────────┘            │
│                     │                              │                            │
│                     └──────────────┬───────────────┘                            │
│                                    ▼                                            │
│   ┌─────────────────────────────────────────────────────────────────────────┐  │
│   │  Unified API Layer                                                       │  │
│   │   • Same method signatures regardless of execution mode                  │  │
│   │   • Strategy pattern abstracts hardware-specific logic                   │  │
│   │   • Factory pattern instantiates correct implementations                 │  │
│   └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

**Key Invariants Across Modes:**

1. **Numerical Equivalence:** Given identical random seeds and input data, Laptop and HPC modes must produce projection matrices $\mathbf{P}$ with Frobenius norm relative error < 5% and correlation coefficient > 0.95.

2. **API Stability:** Client code interacts exclusively with abstract base classes (`PollutionFilterStrategy`, `TrainerFactory`). Hardware-specific implementations are selected at runtime via factory methods.

3. **Test Portability:** Unit tests written on a laptop must pass without modification on the HPC cluster, modulo performance expectations.

### 1.4 Theoretical Foundation Summary

The pipeline is grounded in three theoretical pillars established in the Phase D literature review:

**Pillar 1: The Shortcut Learning Problem (Hermann et al., 2024; Ye et al., 2025)**

Neural networks exhibit a **bias toward "Available" features**—those that can be extracted via simple linear readout—even when less available features are equally predictive. In the SOBR corpus, explicit demographic tokens (e.g., "25M", "As a mother") are maximally available and will dominate gradient descent dynamics, producing "Clever Hans" models that recognize demographics rather than style.

**Pillar 2: Geometric Concept Erasure (Belrose et al., 2024)**

LEACE (Least-squares Concept Erasure) provides a closed-form solution for computing a projection matrix $\mathbf{P}$ that removes all linear information about a concept (e.g., gender) from an embedding space while minimizing distortion:

$$\mathbf{P} = \mathbf{I} - \mathbf{\Sigma}_{XZ}\mathbf{\Sigma}_{ZZ}^{-1}\mathbf{\Sigma}_{ZX}\mathbf{\Sigma}_{XX}^{-1}$$

By injecting $\mathbf{P}$ as a fixed layer before the Transformer encoder, we mathematically guarantee that Query and Key projections lie in the nullspace of demographic variables.

**Pillar 3: Causal Verification via Head Gating (Nam et al., 2025)**

Causal Head Gating learns continuous gate parameters $g_{\ell,h} \in [0,1]$ for each attention head, identifying which heads causally facilitate the classification task. By comparing gate distributions between a "Dirty" model (trained on raw data) and a "Clean" model (trained with the Affine Guard), we can empirically verify that the intervention successfully redirected attention from demographic shortcuts to stylometric features.

### 1.5 Document Organization

This specification is organized into 13 sequential sections:

| Section | Title | Purpose |
|---------|-------|---------|
| 1 | Introduction and Context | *(Current section)* Establish goals, scope, dual-mode philosophy |
| 2 | E2E Requirement List | Functional and non-functional requirements for the complete pipeline |
| 3 | E2E Design Patterns | Core architectural patterns: Strategy, Factory, Data Object |
| 4 | Project Scaffolding | Directory structure and module organization |
| 5 | Phase A Specification | GLiNER and LEACE integration details |
| 6 | Phase A Requirements | Memory, I/O, and computational constraints for Phase A |
| 7 | Phase A Design Patterns | Strategy pattern application to Phase A |
| 8 | Phase A Implementation Plan | Step-by-step coding tasks for Phase A |
| 9 | Phase A→D Handover | Interface definition for inter-phase artifacts |
| 10 | Phase D Specification | Affine Guard and CHG verification details |
| 11 | Phase D Requirements | HPC-specific training requirements |
| 12 | Phase D Design Patterns | Factory and Observer patterns for Phase D |
| 13 | Phase D Implementation Plan | Step-by-step coding tasks for Phase D |


## 2. End-to-End Requirement List

This section enumerates the complete set of functional and non-functional requirements for the Neuro-Symbolic Stylometry Pipeline. Requirements are identified using the convention `FR-XX` (Functional Requirements) and `NFR-XX` (Non-Functional Requirements).

### 2.1 Functional Requirements

#### 2.1.1 Data Ingestion Requirements

| ID | Requirement | Description | Acceptance Criteria |
|----|-------------|-------------|---------------------|
| **FR-01** | **Pandas-to-Arrow Conversion** | The system shall convert raw SOBR Pandas DataFrames (8 separate demographic files) into a single unified Apache Arrow Table. | Arrow Table schema matches specification in §1.2; conversion completes without data loss. |
| **FR-02** | **Schema Consolidation** | The system shall merge all demographic attributes into a single schema with nullable columns for each attribute (`birth_year`, `female`, `extrovert`, `feeling`, `judging`, `sensing`, `nationality`, `political_leaning`). | All 390,000 samples accessible via single Arrow file; null handling preserves original missing data patterns. |
| **FR-03** | **Memory-Mapped I/O** | The system shall persist the Arrow Table using the IPC File format (Feather v2) and support memory-mapped loading for zero-copy DataLoader access. | `mmap` flag enabled on Arrow dataset load; no dataset serialization during DataLoader worker forking. |
| **FR-04** | **Dictionary Encoding** | Categorical columns (`nationality`, `political_leaning`) shall use Arrow dictionary encoding to minimize memory footprint. | Memory reduction of 30-40% compared to raw string storage verified via pyarrow metadata inspection. |
| **FR-05** | **Author ID Indexing** | The system shall maintain author ID as a dictionary-encoded column to support per-author stratified splitting. | Author-stratified train/validation/test splits achievable via Arrow filtering operations. |

#### 2.1.2 Pollution Detection and Cleaning Requirements

| ID | Requirement | Description | Acceptance Criteria |
|----|-------------|-------------|---------------------|
| **FR-06** | **GLiNER Zero-Shot Detection** | The system shall deploy GLiNER-Large-v2.1 to detect demographic spans using a configurable taxonomy: `["age_statement", "gender_indicator", "nationality_claim", "political_self_id"]`. | Zero-shot inference without task-specific fine-tuning; detection latency < 50ms per sample on A100. |
| **FR-07** | **Confidence-Thresholded Masking** | Detected spans with confidence ≥ 0.85 shall be replaced with typed mask tokens: `[MASK:AGE]`, `[MASK:GENDER]`, `[MASK:NATIONALITY]`, `[MASK:POLITICAL]`. | Masking is deterministic given fixed threshold; original spans logged before replacement. |
| **FR-08** | **Pollution Logging** | The system shall log all detected spans (text, start/end positions, confidence, type) to a structured JSON file (`pollution_logs.json`) for post-hoc analysis. | JSON schema validated; logs recoverable for debugging and taxonomy refinement. |
| **FR-09** | **Embedding Extraction** | The system shall extract CLS token embeddings from a frozen encoder (RoBERTa-base) for all masked samples. | Embeddings stored as float16 tensors; shape validation: `[N, 768]`. |
| **FR-10** | **LEACE Projection Matrix Computation** | The system shall compute the LEACE projection matrix $\mathbf{P}$ using the closed-form formula: $\mathbf{P} = \mathbf{I} - \mathbf{\Sigma}_{XZ}\mathbf{\Sigma}_{ZZ}^{-1}\mathbf{\Sigma}_{ZX}\mathbf{\Sigma}_{XX}^{-1}$. | Matrix is idempotent ($\mathbf{P}^2 = \mathbf{P}$); verification via numerical tolerance check. |
| **FR-11** | **Cholesky-Based Inversion** | The system shall compute $\mathbf{\Sigma}_{ZZ}^{-1}$ via Cholesky decomposition to ensure numerical stability for near-balanced demographic classes. | No `NaN` or `Inf` values in projection matrix; condition number logged. |
| **FR-12** | **Covariance Accumulation (Laptop Mode)** | In laptop mode, the system shall accumulate covariance matrices incrementally across mini-batches using Welford's online algorithm variant. | Final covariance matrices within 5% Frobenius norm error of full-batch computation on sample subsets. |

#### 2.1.3 Constrained Training Requirements

| ID | Requirement | Description | Acceptance Criteria |
|----|-------------|-------------|---------------------|
| **FR-13** | **Affine Guard Injection** | The system shall inject the projection matrix $\mathbf{P}$ as a fixed, non-trainable `nn.Linear` layer immediately after the embedding layer and before the first Transformer encoder block. | `requires_grad=False` for Affine Guard parameters; gradient flow verified to bypass this layer. |
| **FR-14** | **Affine Guard Computation** | The Affine Guard shall compute: $\mathbf{h}_{proj} = \mathbf{P} \cdot \mathbf{h}_0$ where $\mathbf{h}_0 = \text{Embed}(tokens) + \text{PosEmbed}$. | Output tensor shape preserved: `[batch, seq_len, hidden_dim]`. |
| **FR-15** | **Multi-Task Classification Head** | The system shall support multi-head classification for 8 demographic attributes simultaneously, with per-attribute loss weighting. | Independent classification heads; loss aggregation configurable. |
| **FR-16** | **Mixed-Precision Training** | The system shall support BF16 mixed-precision training on HPC (no gradient scaling) and FP16 with GradScaler on laptop. | Numerical stability verified via loss curve comparison with FP32 baseline (< 1% relative error). |
| **FR-17** | **torch.compile Fusion** | In HPC mode, the system shall compile the model with `torch.compile(mode="max-autotune", fullgraph=True)` to enable kernel fusion. | Compilation succeeds without graph breaks; Triton kernels generated for fused operations. |
| **FR-18** | **Checkpoint Serialization** | The system shall save model checkpoints including: model state dict, optimizer state, projection matrix $\mathbf{P}$, training configuration, and random states. | Checkpoint recovery produces bit-identical training continuation. |

#### 2.1.4 Verification Requirements

| ID | Requirement | Description | Acceptance Criteria |
|----|-------------|-------------|---------------------|
| **FR-19** | **Causal Head Gating Training** | The system shall freeze model weights and train gate parameters $g_{\ell,h} \in [0,1]$ for each attention head $(l, h)$ using the validation set. | Gate parameters converge; loss decreases monotonically after warmup. |
| **FR-20** | **Head Classification** | The system shall classify each head as "Facilitating" ($g > 0.7$), "Irrelevant" ($g < 0.3$), or "Neutral" ($0.3 \leq g \leq 0.7$). | Classification thresholds configurable; head distribution logged per model. |
| **FR-21** | **Attention Distribution Extraction** | For facilitating heads, the system shall extract attention weight distributions and compute per-POS-tag attention mass. | Attention matrices extracted without gradient computation; POS tagging via spaCy. |
| **FR-22** | **Stylometric Validity Score (SVS)** | The system shall compute: $SVS = \frac{\sum_{h \in H_{fac}} \text{AttnMass}(h, \text{FunctionWords})}{\sum_{h \in H_{fac}} \text{AttnMass}(h, \text{ContentWords})}$. | SVS computed for both Dirty and Clean models; comparison report generated. |
| **FR-23** | **Comparative Analysis Report** | The system shall generate a structured report comparing Model A (Dirty) vs Model B (Clean) on: gate distributions, attention POS distributions, and SVS scores. | Report in JSON and markdown format; visualizations for key metrics. |

#### 2.1.5 Self-Evaluation Requirements

| ID | Requirement | Description | Acceptance Criteria |
|----|-------------|-------------|---------------------|
| **FR-24** | **Linear Probe Training** | After LEACE projection, the system shall train a simple Logistic Regression probe to predict demographic attributes from cleaned embeddings. | Probe accuracy trackable; scikit-learn LogisticRegression or PyTorch equivalent. |
| **FR-25** | **Amnesic Drop Calculation** | The system shall compute Amnesic Drop = (Probe accuracy before cleaning) - (Probe accuracy after cleaning). | Target: Amnesic Drop > 30% for each protected attribute. |
| **FR-26** | **Explicit Recall Validation** | The system shall compute: Explicit Recall = (Spans detected by GLiNER) / (Spans matched by reference regex patterns). | Target: Explicit Recall > 95%. |

---

### 2.2 Non-Functional Requirements

#### 2.2.1 Memory Constraints

| ID | Requirement | Specification | Mode |
|----|-------------|---------------|------|
| **NFR-01** | **Peak GPU VRAM (HPC)** | ≤ 70 GB on A100/H100 (80 GB capacity) to leave headroom for cuDNN workspace allocation. | HPC |
| **NFR-02** | **Peak GPU VRAM (Laptop)** | ≤ 6 GB on A1000 Mobile (8 GB capacity) including model, embeddings, and batch data. | Laptop |
| **NFR-03** | **Peak System RAM (HPC)** | ≤ 400 GB on 512 GB system after Arrow mmap and DataLoader worker accounting. | HPC |
| **NFR-04** | **Peak System RAM (Laptop)** | ≤ 12 GB on 16 GB system with 10k sample subset. | Laptop |
| **NFR-05** | **Projection Matrix Storage** | $\mathbf{P}$ matrix: 768 × 768 × 4 bytes (FP32) = 2.36 MB for computation; 768 × 768 × 2 bytes (BF16) = 1.18 MB for training. | Both |
| **NFR-06** | **Embedding Cache (HPC)** | Full SOBR embeddings: 390,000 × 768 × 2 bytes (FP16) = 598 MB resident in GPU memory. | HPC |
| **NFR-07** | **Arrow Dataset Footprint** | Unified Arrow file ≤ 1.5 GB after dictionary encoding and LZ4 compression. | Both |

#### 2.2.2 Throughput Targets

| ID | Requirement | Specification | Mode |
|----|-------------|---------------|------|
| **NFR-08** | **GLiNER Inference Throughput (HPC)** | ≥ 1,000 samples/second with bucketed batching and CUDA Graph replay. | HPC |
| **NFR-09** | **GLiNER Inference Throughput (Laptop)** | ≥ 50 samples/second with eager execution and batch size 16. | Laptop |
| **NFR-10** | **LEACE Computation Time (HPC)** | ≤ 10 seconds for full dataset (390k samples) on A100 GPU. | HPC |
| **NFR-11** | **LEACE Computation Time (Laptop)** | ≤ 60 seconds for 10k sample subset on CPU/A1000. | Laptop |
| **NFR-12** | **Training Throughput (HPC)** | ≥ 4,000 samples/second at batch size 256 on H100. | HPC |
| **NFR-13** | **Training Throughput (Laptop)** | ≥ 100 samples/second at batch size 16 on A1000 Mobile. | Laptop |
| **NFR-14** | **Epoch Time (HPC)** | ≤ 100 seconds per epoch for full 390k dataset. | HPC |
| **NFR-15** | **DataLoader Saturation** | GPU utilization ≥ 95% during training (no dataloader starvation). | HPC |

#### 2.2.3 Hardware Utilization Targets

| ID | Requirement | Specification | Mode |
|----|-------------|---------------|------|
| **NFR-16** | **Tensor Core Utilization (HPC)** | BF16 GEMM operations routed to Tensor Cores; verified via `sm__sass_thread_inst_executed_op_hmma` counters in Nsight. | HPC |
| **NFR-17** | **TF32 Fallback Activation** | `torch.backends.cuda.matmul.allow_tf32 = True` enabled by default on Ampere/Hopper. | HPC |
| **NFR-18** | **NUMA Pinning Compliance** | All Python processes and DataLoader workers pinned to the NUMA node hosting the target GPU's PCIe root complex. | HPC |
| **NFR-19** | **Memory Bandwidth Utilization** | STREAM benchmark achieving ≥ 85% of theoretical DDR5 peak (≥ 390 GB/s on Genoa). | HPC |
| **NFR-20** | **CUDA Graph Cache Hit Rate** | ≥ 95% graph cache hits during GLiNER inference after warmup phase (first epoch). | HPC |

#### 2.2.4 Numerical Precision Requirements

| ID | Requirement | Specification | Mode |
|----|-------------|---------------|------|
| **NFR-21** | **Projection Matrix Precision** | $\mathbf{P}$ computed in FP32; cast to BF16/FP16 only during forward pass. | Both |
| **NFR-22** | **Idempotence Tolerance** | $\|\mathbf{P}^2 - \mathbf{P}\|_F / \|\mathbf{P}\|_F < 10^{-5}$. | Both |
| **NFR-23** | **Cross-Mode Correlation** | Laptop-computed $\mathbf{P}$ (10k samples) vs HPC-computed $\mathbf{P}$ (10k samples): Pearson correlation > 0.98. | Both |
| **NFR-24** | **Loss Curve Alignment** | FP32 vs BF16 training: relative validation loss difference < 1% at convergence. | HPC |

#### 2.2.5 Reliability and Reproducibility Requirements

| ID | Requirement | Specification | Mode |
|----|-------------|---------------|------|
| **NFR-25** | **Deterministic Execution** | Given identical random seeds, model outputs are bit-identical across runs (within floating-point non-determinism bounds). | Both |
| **NFR-26** | **Checkpoint Recovery** | Training can resume from any checkpoint without degradation in validation metrics. | Both |
| **NFR-27** | **Graceful VRAM Exhaustion** | On OOM, system logs memory state and provides recovery guidance rather than crashing silently. | Both |
| **NFR-28** | **Configuration Validation** | All hyperparameters validated at startup; invalid configurations raise descriptive exceptions. | Both |

#### 2.2.6 Extensibility Requirements

| ID | Requirement | Specification | Mode |
|----|-------------|---------------|------|
| **NFR-29** | **Taxonomy Extensibility** | New pollution categories can be added to GLiNER taxonomy without code changes (configuration-driven). | Both |
| **NFR-30** | **Encoder Swappability** | Base encoder (RoBERTa) can be replaced with alternative Transformers (BERT, DeBERTa) via factory methods. | Both |
| **NFR-31** | **Metric Plugin Architecture** | New evaluation metrics can be registered via callback system without modifying core training loop. | Both |

---

## 3. End-to-End Design Patterns

This section specifies the architectural design patterns that govern the implementation. These patterns ensure hardware-agnostic design with hardware-aware execution, strict separation of concerns, and extensibility for future requirements.

### 3.1 Pattern Overview

The pipeline employs three core Gang-of-Four design patterns, adapted for ML workloads:

| Pattern | Purpose | Primary Application |
|---------|---------|---------------------|
| **Strategy** | Encapsulate hardware-specific algorithms behind a common interface | Switching between `HPCExecutor` and `LaptopExecutor` for Phase A/D operations |
| **Factory Method** | Delegate object instantiation to subclasses/factory functions | `TrainerFactory`, `ModelFactory`, `DataLoaderFactory` |
| **Data Object** | Define canonical data representations with strong typing | Unified Arrow schema, typed configuration dataclasses |

### 3.2 Strategy Pattern: Execution Mode Abstraction

The Strategy pattern is the **central architectural mechanism** for achieving dual-mode execution. It encapsulates hardware-dependent algorithms (CUDA Graph capture, NUMA pinning, precision selection) behind abstract interfaces, allowing client code to remain hardware-agnostic.

#### 3.2.1 Abstract Strategy Interfaces

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          STRATEGY PATTERN HIERARCHY                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ┌───────────────────────────────────────────────────────────────────────┐  │
│   │                    «abstract» ExecutionStrategy                        │  │
│   │   ─────────────────────────────────────────────────────────────────    │  │
│   │   + get_device() → torch.device                                        │  │
│   │   + get_dtype() → torch.dtype                                          │  │
│   │   + get_batch_size() → int                                             │  │
│   │   + get_num_workers() → int                                            │  │
│   │   + supports_cuda_graphs() → bool                                      │  │
│   │   + supports_torch_compile() → bool                                    │  │
│   │   + configure_numa() → None                                            │  │
│   └───────────────────────────────────────────────────────────────────────┘  │
│                       ▲                              ▲                        │
│                       │                              │                        │
│         ┌─────────────┴──────────────┐  ┌───────────┴─────────────┐         │
│         │      HPCStrategy           │  │    LaptopStrategy       │         │
│         │  ─────────────────────────  │  │ ───────────────────────  │         │
│         │  device: cuda:0            │  │  device: cuda:0 / cpu   │         │
│         │  dtype: bfloat16           │  │  dtype: float16         │         │
│         │  batch_size: 256           │  │  batch_size: 16         │         │
│         │  num_workers: 8            │  │  num_workers: 2         │         │
│         │  cuda_graphs: True         │  │  cuda_graphs: False     │         │
│         │  torch_compile: True       │  │  torch_compile: False   │         │
│         │  numa_node: auto-detected  │  │  numa_node: None        │         │
│         └────────────────────────────┘  └─────────────────────────┘         │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Python Type Specification:**

```python
from abc import ABC, abstractmethod
from typing import Optional
import torch

class ExecutionStrategy(ABC):
    """Abstract base class for hardware-specific execution strategies."""
    
    @abstractmethod
    def get_device(self) -> torch.device:
        """Returns the primary compute device (cuda:N or cpu)."""
        ...
    
    @abstractmethod
    def get_dtype(self) -> torch.dtype:
        """Returns the compute dtype (bfloat16, float16, or float32)."""
        ...
    
    @abstractmethod
    def get_batch_size(self) -> int:
        """Returns the optimal batch size for this hardware profile."""
        ...
    
    @abstractmethod
    def get_num_workers(self) -> int:
        """Returns the number of DataLoader workers."""
        ...
    
    @abstractmethod
    def supports_cuda_graphs(self) -> bool:
        """Returns True if CUDA Graph capture is supported and enabled."""
        ...
    
    @abstractmethod
    def supports_torch_compile(self) -> bool:
        """Returns True if torch.compile is supported and enabled."""
        ...
    
    @abstractmethod
    def configure_numa(self) -> None:
        """Configures NUMA affinity for the current process. No-op on laptops."""
        ...
    
    @abstractmethod
    def get_gradient_scaler(self) -> Optional[torch.cuda.amp.GradScaler]:
        """Returns a GradScaler for FP16 training, or None for BF16."""
        ...
```

#### 3.2.2 Specialized Strategy: PollutionFilterStrategy

Phase A operations require additional strategy methods specific to pollution detection:

```python
from abc import ABC, abstractmethod
from typing import Tuple
import torch
from datasets import Dataset

class PollutionFilterStrategy(ABC):
    """Abstract strategy for Phase A pollution filtering operations."""
    
    @abstractmethod
    def apply_gliner_masking(
        self, 
        dataset: Dataset, 
        taxonomy: list[str],
        confidence_threshold: float
    ) -> Tuple[Dataset, dict]:
        """
        Apply GLiNER-based symbolic masking to the dataset.
        
        Returns:
            Tuple of (masked_dataset, pollution_log_dict)
        """
        ...
    
    @abstractmethod
    def compute_embeddings(
        self,
        dataset: Dataset,
        encoder_name: str
    ) -> torch.Tensor:
        """
        Extract CLS embeddings from the masked dataset.
        
        Returns:
            Tensor of shape [N, hidden_dim] on the appropriate device.
        """
        ...
    
    @abstractmethod
    def compute_leace_projection(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the LEACE projection matrix P.
        
        Returns:
            Projection matrix of shape [hidden_dim, hidden_dim].
        """
        ...
```

**Concrete Implementations:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    POLLUTION FILTER STRATEGY IMPLEMENTATIONS                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                  HPCFilterStrategy                                   │   │
│   │  ────────────────────────────────────────────────────────────────    │   │
│   │  • GLiNER inference with CUDA Graph caching per (batch, seq_len)    │   │
│   │  • Smart bucketing to minimize padding waste                         │   │
│   │  • Full-dataset embedding extraction in single GPU pass             │   │
│   │  • LEACE covariance computed entirely on GPU via batched GEMM       │   │
│   │  • Cholesky decomposition for numerically stable inversion          │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                  LaptopFilterStrategy                                │   │
│   │  ────────────────────────────────────────────────────────────────    │   │
│   │  • GLiNER inference in eager mode, batch size 16                    │   │
│   │  • Simple sequential batching (no bucketing)                        │   │
│   │  • 10k sample subset for development iteration                      │   │
│   │  • Mini-batch covariance accumulation for memory efficiency         │   │
│   │  • CPU fallback available if GPU VRAM insufficient                  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 3.2.3 Specialized Strategy: TrainingStrategy

Phase D training operations require strategy methods for model compilation and optimization:

```python
from abc import ABC, abstractmethod
from typing import Optional, Callable
import torch
import torch.nn as nn

class TrainingStrategy(ABC):
    """Abstract strategy for Phase D training operations."""
    
    @abstractmethod
    def prepare_model(
        self,
        model: nn.Module,
        projection_matrix: torch.Tensor
    ) -> nn.Module:
        """
        Prepare model for training: inject Affine Guard, apply torch.compile.
        
        Returns:
            Prepared model ready for training.
        """
        ...
    
    @abstractmethod
    def create_optimizer(
        self,
        model: nn.Module,
        learning_rate: float
    ) -> torch.optim.Optimizer:
        """Create and configure the optimizer."""
        ...
    
    @abstractmethod
    def create_dataloader(
        self,
        dataset: Dataset,
        shuffle: bool
    ) -> torch.utils.data.DataLoader:
        """Create a DataLoader with hardware-appropriate configuration."""
        ...
    
    @abstractmethod
    def training_step(
        self,
        model: nn.Module,
        batch: dict,
        loss_fn: Callable
    ) -> torch.Tensor:
        """Execute a single training step with appropriate precision context."""
        ...
```

### 3.3 Factory Pattern: Component Instantiation

The Factory pattern centralizes object creation, isolating hardware-detection logic from client code. Three primary factories govern component instantiation:

#### 3.3.1 StrategyFactory

```python
from typing import Literal
from dataclasses import dataclass

@dataclass
class HardwareProfile:
    """Detected hardware characteristics."""
    gpu_name: str
    gpu_vram_gb: float
    cpu_cores: int
    numa_nodes: int
    cuda_version: tuple[int, int]
    
    @property
    def profile_type(self) -> Literal["HPC", "LAPTOP", "GENERIC"]:
        if "A100" in self.gpu_name or "H100" in self.gpu_name:
            return "HPC"
        elif self.gpu_vram_gb < 16:
            return "LAPTOP"
        else:
            return "GENERIC"

class StrategyFactory:
    """Factory for creating hardware-appropriate strategy instances."""
    
    @staticmethod
    def detect_hardware() -> HardwareProfile:
        """Query system hardware and return a HardwareProfile."""
        import torch
        
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            gpu_vram_gb = props.total_memory / (1024**3)
            cuda_version = torch.version.cuda.split(".")[:2]
        else:
            gpu_name = "CPU_ONLY"
            gpu_vram_gb = 0.0
            cuda_version = (0, 0)
        
        import os
        cpu_cores = os.cpu_count() or 1
        
        # NUMA detection (Linux-specific, fallback to 1)
        try:
            numa_nodes = len(os.sched_getaffinity(0)) // cpu_cores or 1
        except AttributeError:
            numa_nodes = 1
        
        return HardwareProfile(
            gpu_name=gpu_name,
            gpu_vram_gb=gpu_vram_gb,
            cpu_cores=cpu_cores,
            numa_nodes=numa_nodes,
            cuda_version=tuple(map(int, cuda_version))
        )
    
    @classmethod
    def create_execution_strategy(cls) -> ExecutionStrategy:
        """Create an ExecutionStrategy appropriate for detected hardware."""
        profile = cls.detect_hardware()
        
        if profile.profile_type == "HPC":
            return HPCStrategy(profile)
        elif profile.profile_type == "LAPTOP":
            return LaptopStrategy(profile)
        else:
            return GenericStrategy(profile)
    
    @classmethod
    def create_filter_strategy(cls) -> PollutionFilterStrategy:
        """Create a PollutionFilterStrategy appropriate for detected hardware."""
        profile = cls.detect_hardware()
        
        if profile.profile_type == "HPC":
            return HPCFilterStrategy(profile)
        else:
            return LaptopFilterStrategy(profile)
    
    @classmethod
    def create_training_strategy(cls) -> TrainingStrategy:
        """Create a TrainingStrategy appropriate for detected hardware."""
        profile = cls.detect_hardware()
        
        if profile.profile_type == "HPC":
            return HPCTrainingStrategy(profile)
        else:
            return LaptopTrainingStrategy(profile)
```

#### 3.3.2 ModelFactory

```python
from typing import Optional
import torch
import torch.nn as nn

class ModelFactory:
    """Factory for creating and configuring model instances."""
    
    @staticmethod
    def create_encoder(
        encoder_name: str = "roberta-base",
        pretrained: bool = True
    ) -> nn.Module:
        """
        Create a frozen encoder for embedding extraction.
        
        Args:
            encoder_name: HuggingFace model identifier
            pretrained: Whether to load pretrained weights
        
        Returns:
            Frozen encoder module
        """
        from transformers import AutoModel
        
        encoder = AutoModel.from_pretrained(encoder_name)
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
        
        return encoder
    
    @staticmethod
    def create_affine_transformer(
        base_model_name: str,
        projection_matrix: torch.Tensor,
        num_labels: int,
        freeze_projection: bool = True
    ) -> nn.Module:
        """
        Create a Transformer with injected Affine Guard.
        
        Args:
            base_model_name: HuggingFace model identifier
            projection_matrix: LEACE projection matrix P [d, d]
            num_labels: Number of classification labels
            freeze_projection: Whether to freeze the Affine Guard layer
        
        Returns:
            AffineGuardTransformer instance
        """
        from .stylometry_net import AffineGuardTransformer
        
        return AffineGuardTransformer(
            base_model_name=base_model_name,
            projection_matrix=projection_matrix,
            num_labels=num_labels,
            freeze_projection=freeze_projection
        )
    
    @staticmethod
    def create_gliner(
        model_name: str = "urchade/gliner_large-v2.1"
    ) -> nn.Module:
        """
        Create a GLiNER model for zero-shot NER.
        
        Returns:
            GLiNER model instance
        """
        from gliner import GLiNER
        
        return GLiNER.from_pretrained(model_name)
```

#### 3.3.3 TrainerFactory

```python
from typing import Optional, Callable
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

@dataclass
class TrainerConfig:
    """Configuration for training loop."""
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_epochs: int = 10
    gradient_accumulation_steps: int = 1
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 50

class TrainerFactory:
    """Factory for creating configured Trainer instances."""
    
    @staticmethod
    def create_trainer(
        model: nn.Module,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        config: TrainerConfig,
        projection_matrix: torch.Tensor,
        callbacks: Optional[list] = None
    ) -> "Trainer":
        """
        Create a Trainer instance with hardware-appropriate configuration.
        
        The factory automatically:
        - Detects hardware and selects execution strategy
        - Configures mixed-precision training
        - Creates optimized DataLoaders
        - Injects the Affine Guard layer
        - Applies torch.compile if supported
        
        Returns:
            Configured Trainer instance
        """
        strategy = StrategyFactory.create_training_strategy()
        
        # Prepare model (inject Affine Guard, compile if HPC)
        prepared_model = strategy.prepare_model(model, projection_matrix)
        
        # Create DataLoaders
        train_loader = strategy.create_dataloader(train_dataset, shuffle=True)
        eval_loader = strategy.create_dataloader(eval_dataset, shuffle=False)
        
        # Create optimizer
        optimizer = strategy.create_optimizer(prepared_model, config.learning_rate)
        
        # Get precision configuration
        dtype = strategy.get_dtype()
        scaler = strategy.get_gradient_scaler()
        
        return Trainer(
            model=prepared_model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            optimizer=optimizer,
            config=config,
            dtype=dtype,
            scaler=scaler,
            callbacks=callbacks or []
        )
```

### 3.4 Data Object Pattern: Unified Schema Definition

The Data Object pattern enforces type safety and schema consistency across the pipeline via Python dataclasses and Apache Arrow type specifications.

#### 3.4.1 Arrow Schema Definition

```python
import pyarrow as pa

# Canonical schema for the unified SOBR dataset
SOBR_SCHEMA = pa.schema([
    pa.field("author_id", pa.dictionary(pa.int32(), pa.string()), nullable=False),
    pa.field("post", pa.large_string(), nullable=False),
    pa.field("post_masked", pa.large_string(), nullable=True),  # After GLiNER masking
    pa.field("birth_year", pa.int16(), nullable=True),
    pa.field("extrovert", pa.bool_(), nullable=True),
    pa.field("feeling", pa.bool_(), nullable=True),
    pa.field("female", pa.bool_(), nullable=True),
    pa.field("judging", pa.bool_(), nullable=True),
    pa.field("sensing", pa.bool_(), nullable=True),
    pa.field("nationality", pa.dictionary(pa.int8(), pa.string()), nullable=True),
    pa.field("political_leaning", pa.dictionary(pa.int8(), pa.string()), nullable=True),
])

# Schema for pollution logs
POLLUTION_LOG_SCHEMA = pa.schema([
    pa.field("sample_id", pa.int64(), nullable=False),
    pa.field("span_text", pa.string(), nullable=False),
    pa.field("span_start", pa.int32(), nullable=False),
    pa.field("span_end", pa.int32(), nullable=False),
    pa.field("entity_type", pa.string(), nullable=False),
    pa.field("confidence", pa.float32(), nullable=False),
])
```

#### 3.4.2 Configuration Dataclasses

```python
from dataclasses import dataclass, field
from typing import List, Optional, Literal
from pathlib import Path

@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration for the entire pipeline."""
    
    # Paths
    raw_data_dir: Path
    output_dir: Path
    checkpoint_dir: Path
    
    # Phase A Configuration
    gliner_model: str = "urchade/gliner_large-v2.1"
    gliner_taxonomy: List[str] = field(default_factory=lambda: [
        "age_statement",
        "gender_indicator", 
        "nationality_claim",
        "political_self_id"
    ])
    gliner_confidence_threshold: float = 0.85
    encoder_model: str = "roberta-base"
    
    # Phase D Configuration
    base_transformer: str = "roberta-base"
    hidden_dim: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    
    # Training Configuration
    learning_rate: float = 2e-4
    batch_size_hpc: int = 256
    batch_size_laptop: int = 16
    max_epochs: int = 10
    warmup_ratio: float = 0.1
    
    # Hardware Configuration
    force_mode: Optional[Literal["HPC", "LAPTOP"]] = None
    seed: int = 42

@dataclass
class PhaseAArtifacts:
    """Typed container for Phase A output artifacts."""
    
    clean_dataset_path: Path
    projection_matrix_path: Path
    pollution_log_path: Path
    
    # Metrics
    explicit_recall: float
    amnesic_drop: float
    bertscore_recall: float
    perplexity_delta: float
    
    def validate(self) -> bool:
        """Verify all artifact files exist and metrics meet thresholds."""
        if not self.clean_dataset_path.exists():
            raise FileNotFoundError(f"Clean dataset not found: {self.clean_dataset_path}")
        if not self.projection_matrix_path.exists():
            raise FileNotFoundError(f"Projection matrix not found: {self.projection_matrix_path}")
        if self.explicit_recall < 0.95:
            raise ValueError(f"Explicit recall {self.explicit_recall} below threshold 0.95")
        if self.amnesic_drop < 0.30:
            raise ValueError(f"Amnesic drop {self.amnesic_drop} below threshold 0.30")
        return True

@dataclass
class PhaseDModelOutput:
    """Typed container for Phase D model outputs."""
    
    logits: torch.Tensor  # [batch, num_labels]
    attention_weights: Optional[List[torch.Tensor]] = None  # List of [batch, heads, seq, seq]
    hidden_states: Optional[List[torch.Tensor]] = None  # List of [batch, seq, hidden]
    
    def get_predictions(self) -> torch.Tensor:
        """Return predicted class indices."""
        return self.logits.argmax(dim=-1)
    
    def get_probabilities(self) -> torch.Tensor:
        """Return softmax probabilities."""
        return torch.softmax(self.logits, dim=-1)
```

### 3.5 Pattern Interactions

The three patterns interact as follows:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        PATTERN INTERACTION DIAGRAM                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ┌───────────────────────┐                                                  │
│   │   PipelineConfig      │◄─────────────────────────────────────────────┐   │
│   │   (Data Object)       │                                              │   │
│   └───────────┬───────────┘                                              │   │
│               │ passed to                                                │   │
│               ▼                                                          │   │
│   ┌───────────────────────┐                                              │   │
│   │   StrategyFactory     │──────────────────┐                          │   │
│   │   (Factory)           │                  │                          │   │
│   └───────────┬───────────┘                  │                          │   │
│               │ creates                       │ creates                  │   │
│               ▼                               ▼                          │   │
│   ┌───────────────────────┐     ┌───────────────────────┐               │   │
│   │   HPCStrategy or      │     │   ModelFactory        │               │   │
│   │   LaptopStrategy      │     │   (Factory)           │               │   │
│   │   (Strategy)          │     └───────────┬───────────┘               │   │
│   └───────────┬───────────┘                 │ creates                   │   │
│               │                              ▼                           │   │
│               │                 ┌───────────────────────┐               │   │
│               │                 │   AffineGuard-        │               │   │
│               │                 │   Transformer         │               │   │
│               │                 │   (Domain Object)     │               │   │
│               │                 └───────────┬───────────┘               │   │
│               │                             │                            │   │
│               └─────────────┬───────────────┘                            │   │
│                             │ both used by                               │   │
│                             ▼                                            │   │
│               ┌───────────────────────┐                                  │   │
│               │   TrainerFactory      │                                  │   │
│               │   (Factory)           │                                  │   │
│               └───────────┬───────────┘                                  │   │
│                           │ creates                                      │   │
│                           ▼                                              │   │
│               ┌───────────────────────┐     ┌───────────────────────┐   │   │
│               │   Trainer             │────►│   PhaseDArtifacts     │───┘   │
│               │   (Orchestrator)      │     │   (Data Object)       │       │
│               └───────────────────────┘     └───────────────────────┘       │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Interaction Flow:**

1. `PipelineConfig` (Data Object) is loaded from YAML/JSON and validated
2. `StrategyFactory` queries hardware and instantiates `HPCStrategy` or `LaptopStrategy`
3. `ModelFactory` creates the base encoder and `AffineGuardTransformer`
4. `TrainerFactory` combines strategy, model, and config to create a `Trainer`
5. Training produces `PhaseDModelOutput` objects (Data Object) per batch
6. Final artifacts are packaged into `PhaseDArtifacts` (Data Object) for downstream use

---

## 4. Project Scaffolding

This section defines the directory structure and module organization for the Neuro-Symbolic Stylometry Pipeline. The scaffolding follows Python packaging best practices with explicit separation between source code, configuration, tests, and scripts.

### 4.1 Directory Structure

```
neuro_stylometry_pipeline/
├── pyproject.toml                      # PEP 517/518 build configuration
├── setup.cfg                           # Package metadata and dependencies
├── README.md                           # Project documentation
├── LICENSE                             # MIT License
├── .pre-commit-config.yaml             # Pre-commit hooks (ruff, mypy)
│
├── conf/                               # Configuration files (YAML/JSON)
│   ├── base/                           # Base configuration (shared)
│   │   ├── pipeline.yaml               # PipelineConfig defaults
│   │   ├── gliner_taxonomy.yaml        # Pollution detection taxonomy
│   │   └── training.yaml               # TrainerConfig defaults
│   ├── hpc/                            # HPC-specific overrides
│   │   ├── pipeline.yaml               # A100/H100 settings
│   │   └── training.yaml               # Large batch, BF16 settings
│   ├── laptop/                         # Laptop-specific overrides
│   │   ├── pipeline.yaml               # Subset size, debug flags
│   │   └── training.yaml               # Small batch, FP16 settings
│   └── experiments/                    # Experiment-specific configs
│       └── example_experiment.yaml     # Example experiment config
│
├── src/                                # Source code root
│   └── neuro_stylometry/               # Main package
│       ├── __init__.py                 # Package initialization
│       ├── __main__.py                 # CLI entry point
│       ├── config.py                   # Configuration loading and validation
│       │
│       ├── data_engine/                # Data loading and Arrow operations
│       │   ├── __init__.py
│       │   ├── schemas.py              # Arrow schema definitions
│       │   ├── converter.py            # Pandas → Arrow conversion
│       │   ├── dataset.py              # HuggingFace Dataset wrappers
│       │   ├── dataloader.py           # PyTorch DataLoader factories
│       │   └── bucketing.py            # Smart bucketing for variable-length sequences
│       │
│       ├── pollution_guard/            # Phase A: Pollution Detection & Mitigation
│       │   ├── __init__.py
│       │   ├── gliner_detector.py      # GLiNER wrapper for span detection
│       │   ├── masker.py               # Span masking logic
│       │   ├── leace.py                # LEACE projection matrix computation
│       │   ├── embedder.py             # Frozen encoder embedding extraction
│       │   ├── probe.py                # Linear probe for self-evaluation
│       │   └── strategies/             # Strategy pattern implementations
│       │       ├── __init__.py
│       │       ├── base.py             # PollutionFilterStrategy ABC
│       │       ├── hpc.py              # HPCFilterStrategy
│       │       └── laptop.py           # LaptopFilterStrategy
│       │
│       ├── stylometry_net/             # Phase D: Constrained Neural Stylometry
│       │   ├── __init__.py
│       │   ├── affine_guard.py         # AffineGuard nn.Module
│       │   ├── transformer.py          # AffineGuardTransformer
│       │   ├── classification_head.py  # Multi-task classification heads
│       │   ├── chg_verifier.py         # Causal Head Gating verification
│       │   ├── svs_calculator.py       # Stylometric Validity Score
│       │   └── strategies/             # Strategy pattern implementations
│       │       ├── __init__.py
│       │       ├── base.py             # TrainingStrategy ABC
│       │       ├── hpc.py              # HPCTrainingStrategy
│       │       └── laptop.py           # LaptopTrainingStrategy
│       │
│       ├── hardware_ops/               # Hardware-aware operations
│       │   ├── __init__.py
│       │   ├── detection.py            # Hardware profile detection
│       │   ├── numa.py                 # NUMA pinning utilities
│       │   ├── cuda_graphs.py          # CUDA Graph capture and replay
│       │   ├── precision.py            # Mixed-precision configuration
│       │   └── memory.py               # Memory management utilities
│       │
│       ├── training/                   # Training loop and utilities
│       │   ├── __init__.py
│       │   ├── trainer.py              # Main Trainer class
│       │   ├── callbacks.py            # Callback base class and implementations
│       │   ├── checkpointing.py        # Checkpoint save/load logic
│       │   ├── metrics.py              # Metric computation and logging
│       │   └── optimizer.py            # Optimizer factory and scheduling
│       │
│       ├── evaluation/                 # Evaluation and analysis
│       │   ├── __init__.py
│       │   ├── attention_analysis.py   # Attention weight extraction and POS analysis
│       │   ├── comparative_report.py   # Model A vs Model B comparison
│       │   └── visualizations.py       # Plotting utilities
│       │
│       ├── factories/                  # Factory pattern implementations
│       │   ├── __init__.py
│       │   ├── strategy_factory.py     # StrategyFactory
│       │   ├── model_factory.py        # ModelFactory
│       │   └── trainer_factory.py      # TrainerFactory
│       │
│       └── utils/                      # Shared utilities
│           ├── __init__.py
│           ├── logging.py              # Logging configuration
│           ├── seeding.py              # Reproducibility (random seeds)
│           ├── timing.py               # Performance timing decorators
│           └── validation.py           # Input validation utilities
│
├── tests/                              # Test suite
│   ├── conftest.py                     # Pytest fixtures
│   ├── unit/                           # Unit tests
│   │   ├── test_schemas.py
│   │   ├── test_leace.py
│   │   ├── test_affine_guard.py
│   │   ├── test_strategies.py
│   │   └── test_factories.py
│   ├── integration/                    # Integration tests
│   │   ├── test_phase_a_pipeline.py
│   │   ├── test_phase_d_training.py
│   │   └── test_dual_mode.py
│   └── golden/                         # Golden reference tests
│       ├── test_projection_consistency.py
│       └── fixtures/                   # Test fixtures (small data samples)
│           ├── sample_posts.json
│           └── expected_masks.json
│
├── scripts/                            # Executable scripts
│   ├── convert_pandas_to_arrow.py      # One-time data conversion
│   ├── run_phase_a.py                  # Phase A execution
│   ├── run_phase_d.py                  # Phase D execution
│   ├── run_verification.py             # CHG verification
│   ├── generate_report.py              # Generate comparison report
│   └── slurm/                          # HPC job scripts
│       ├── phase_a.sbatch              # SLURM script for Phase A
│       ├── phase_d.sbatch              # SLURM script for Phase D
│       └── verify.sbatch               # SLURM script for verification
│
├── notebooks/                          # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_gliner_taxonomy_tuning.ipynb
│   ├── 03_leace_visualization.ipynb
│   └── 04_attention_analysis.ipynb
│
└── artifacts/                          # Generated artifacts (gitignored)
    ├── data/                           # Processed datasets
    │   ├── sobr_raw/                   # Raw SOBR Pandas files
    │   ├── sobr_unified.arrow          # Unified Arrow dataset
    │   └── sobr_clean.arrow            # After Phase A cleaning
    ├── models/                         # Trained model checkpoints
    │   ├── phase_a/                    # Phase A artifacts
    │   │   ├── projection_matrix.pt    # LEACE P matrix
    │   │   └── pollution_logs.json     # Detected spans
    │   └── phase_d/                    # Phase D checkpoints
    │       ├── model_dirty/            # Model A (baseline)
    │       └── model_clean/            # Model B (with Affine Guard)
    └── reports/                        # Generated reports
        ├── phase_a_metrics.json
        ├── phase_d_metrics.json
        └── comparison_report.md
```

### 4.2 Key Module Specifications

#### 4.2.1 `data_engine` Module

**Purpose:** Manage all data operations from raw ingestion to PyTorch DataLoader creation.

| File | Responsibility | Key Classes/Functions |
|------|---------------|----------------------|
| `schemas.py` | Define Arrow schemas for dataset and pollution logs | `SOBR_SCHEMA`, `POLLUTION_LOG_SCHEMA`, `validate_schema()` |
| `converter.py` | Convert 8 Pandas DataFrames to unified Arrow Table | `PandasToArrowConverter`, `merge_demographic_frames()` |
| `dataset.py` | Wrap Arrow tables as HuggingFace Datasets with memory mapping | `SOBRDataset`, `load_arrow_dataset()` |
| `dataloader.py` | Create PyTorch DataLoaders with hardware-appropriate configuration | `DataLoaderFactory`, `create_tokenizing_dataloader()` |
| `bucketing.py` | Implement sequence-length bucketing for efficient batching | `BucketSampler`, `compute_bucket_boundaries()` |

**Dependencies:**
- `pyarrow >= 14.0.0`
- `datasets >= 2.16.0`
- `torch >= 2.2.0`

#### 4.2.2 `pollution_guard` Module

**Purpose:** Implement Phase A pollution detection, masking, and geometric projection.

| File | Responsibility | Key Classes/Functions |
|------|---------------|----------------------|
| `gliner_detector.py` | Wrap GLiNER for zero-shot span detection | `GLiNERDetector`, `detect_spans()`, `load_taxonomy()` |
| `masker.py` | Apply typed masks to detected spans | `SpanMasker`, `apply_masks()`, `format_mask_token()` |
| `leace.py` | Compute LEACE projection matrix | `LEACEComputer`, `compute_covariance()`, `cholesky_inverse()` |
| `embedder.py` | Extract CLS embeddings from frozen encoder | `FrozenEmbedder`, `batch_embed()` |
| `probe.py` | Train linear probes for self-evaluation | `LinearProbe`, `compute_amnesic_drop()` |
| `strategies/base.py` | Abstract strategy interface | `PollutionFilterStrategy` (ABC) |
| `strategies/hpc.py` | HPC implementation with CUDA Graphs and full-batch LEACE | `HPCFilterStrategy` |
| `strategies/laptop.py` | Laptop implementation with mini-batch accumulation | `LaptopFilterStrategy` |

**Dependencies:**
- `gliner >= 0.2.0`
- `transformers >= 4.37.0`
- `scikit-learn >= 1.4.0` (for linear probe)

#### 4.2.3 `stylometry_net` Module

**Purpose:** Implement Phase D constrained Transformer architecture and verification.

| File | Responsibility | Key Classes/Functions |
|------|---------------|----------------------|
| `affine_guard.py` | Define the Affine Guard layer | `AffineGuard(nn.Module)`, `forward()` |
| `transformer.py` | Integrate Affine Guard with base Transformer | `AffineGuardTransformer(nn.Module)`, `inject_guard()` |
| `classification_head.py` | Multi-task classification heads | `MultiTaskHead`, `compute_losses()` |
| `chg_verifier.py` | Implement Causal Head Gating | `CHGVerifier`, `learn_gates()`, `classify_heads()` |
| `svs_calculator.py` | Compute Stylometric Validity Score | `SVSCalculator`, `extract_attention()`, `compute_pos_mass()` |
| `strategies/base.py` | Abstract training strategy interface | `TrainingStrategy` (ABC) |
| `strategies/hpc.py` | HPC training with torch.compile and BF16 | `HPCTrainingStrategy` |
| `strategies/laptop.py` | Laptop training with FP16 and GradScaler | `LaptopTrainingStrategy` |

**Dependencies:**
- `transformers >= 4.37.0`
- `spacy >= 3.7.0` (for POS tagging)

#### 4.2.4 `hardware_ops` Module

**Purpose:** Encapsulate all hardware-specific operations for dual-mode execution.

| File | Responsibility | Key Classes/Functions |
|------|---------------|----------------------|
| `detection.py` | Detect and profile hardware capabilities | `HardwareDetector`, `HardwareProfile`, `detect_profile()` |
| `numa.py` | Configure NUMA affinity | `NUMAConfigurator`, `pin_to_node()`, `configure_workers()` |
| `cuda_graphs.py` | Manage CUDA Graph capture and replay | `GraphCache`, `capture_graph()`, `replay_graph()` |
| `precision.py` | Configure mixed-precision training | `PrecisionManager`, `get_autocast_context()`, `get_scaler()` |
| `memory.py` | Monitor and manage GPU/CPU memory | `MemoryMonitor`, `check_vram()`, `defragment_allocator()` |

**Dependencies:**
- `torch >= 2.2.0`
- `nvidia-ml-py >= 12.0.0` (optional, for detailed GPU monitoring)

### 4.3 Configuration Management

#### 4.3.1 Hydra-Style Configuration Hierarchy

The configuration system uses a composition pattern inspired by Hydra:

```yaml
# conf/base/pipeline.yaml (defaults)
gliner:
  model: "urchade/gliner_large-v2.1"
  confidence_threshold: 0.85

encoder:
  model: "roberta-base"
  hidden_dim: 768

paths:
  raw_data: "${oc.env:SOBR_DATA_PATH,./artifacts/data/sobr_raw}"
  output: "./artifacts"

# conf/hpc/pipeline.yaml (HPC overrides)
defaults:
  - /base/pipeline

execution:
  enable_cuda_graphs: true
  enable_torch_compile: true
  numa_pinning: true

dataloader:
  batch_size: 256
  num_workers: 8
  persistent_workers: true
  prefetch_factor: 4
  pin_memory: true

precision:
  dtype: "bfloat16"
  use_grad_scaler: false

# conf/laptop/pipeline.yaml (Laptop overrides)
defaults:
  - /base/pipeline

execution:
  enable_cuda_graphs: false
  enable_torch_compile: false
  numa_pinning: false

dataloader:
  batch_size: 16
  num_workers: 2
  persistent_workers: false
  prefetch_factor: 2
  pin_memory: true

precision:
  dtype: "float16"
  use_grad_scaler: true

subset:
  enabled: true
  size: 10000
```

#### 4.3.2 Configuration Loading

```python
# src/neuro_stylometry/config.py

from dataclasses import dataclass
from typing import Optional, Literal
from pathlib import Path
from omegaconf import OmegaConf

def load_config(
    config_path: Path,
    overrides: Optional[dict] = None,
    mode: Optional[Literal["hpc", "laptop"]] = None
) -> PipelineConfig:
    """
    Load and merge configuration from YAML files.
    
    Args:
        config_path: Path to experiment config or base config
        overrides: Dictionary of runtime overrides
        mode: Explicit mode selection (auto-detected if None)
    
    Returns:
        Validated PipelineConfig dataclass
    """
    # Load base config
    base_cfg = OmegaConf.load(Path(__file__).parent.parent.parent / "conf/base/pipeline.yaml")
    
    # Detect or use specified mode
    if mode is None:
        from .hardware_ops.detection import HardwareDetector
        profile = HardwareDetector.detect()
        mode = "hpc" if profile.profile_type == "HPC" else "laptop"
    
    # Load mode-specific overrides
    mode_cfg = OmegaConf.load(Path(__file__).parent.parent.parent / f"conf/{mode}/pipeline.yaml")
    
    # Load experiment config if provided
    experiment_cfg = OmegaConf.load(config_path) if config_path.exists() else OmegaConf.create()
    
    # Merge configurations (later overrides earlier)
    merged = OmegaConf.merge(base_cfg, mode_cfg, experiment_cfg)
    
    # Apply runtime overrides
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.create(overrides))
    
    # Resolve interpolations and convert to dataclass
    OmegaConf.resolve(merged)
    return OmegaConf.to_object(merged, PipelineConfig)
```

### 4.4 Entry Points

#### 4.4.1 CLI Entry Point

```python
# src/neuro_stylometry/__main__.py

import click
from pathlib import Path
from .config import load_config

@click.group()
@click.option("--config", "-c", type=click.Path(exists=True), help="Experiment config path")
@click.option("--mode", type=click.Choice(["hpc", "laptop", "auto"]), default="auto")
@click.pass_context
def cli(ctx, config, mode):
    """Neuro-Symbolic Stylometry Pipeline CLI."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(
        Path(config) if config else Path("conf/base/pipeline.yaml"),
        mode=None if mode == "auto" else mode
    )

@cli.command()
@click.option("--input-dir", type=click.Path(exists=True), required=True)
@click.option("--output-path", type=click.Path(), required=True)
@click.pass_context
def convert_data(ctx, input_dir, output_path):
    """Convert Pandas DataFrames to unified Arrow format."""
    from .data_engine.converter import PandasToArrowConverter
    
    converter = PandasToArrowConverter(ctx.obj["config"])
    converter.convert(Path(input_dir), Path(output_path))

@cli.command()
@click.option("--dataset", type=click.Path(exists=True), required=True)
@click.option("--output-dir", type=click.Path(), required=True)
@click.pass_context
def run_phase_a(ctx, dataset, output_dir):
    """Execute Phase A pollution detection and mitigation."""
    from .pollution_guard import PhaseAPipeline
    from .factories.strategy_factory import StrategyFactory
    
    strategy = StrategyFactory.create_filter_strategy()
    pipeline = PhaseAPipeline(ctx.obj["config"], strategy)
    artifacts = pipeline.run(Path(dataset), Path(output_dir))
    
    click.echo(f"Phase A complete. Artifacts saved to {output_dir}")
    click.echo(f"  Explicit Recall: {artifacts.explicit_recall:.2%}")
    click.echo(f"  Amnesic Drop: {artifacts.amnesic_drop:.2%}")

@cli.command()
@click.option("--dataset", type=click.Path(exists=True), required=True)
@click.option("--projection", type=click.Path(exists=True), required=True)
@click.option("--output-dir", type=click.Path(), required=True)
@click.pass_context
def run_phase_d(ctx, dataset, projection, output_dir):
    """Execute Phase D constrained training."""
    from .training.trainer import Trainer
    from .factories.trainer_factory import TrainerFactory
    import torch
    
    projection_matrix = torch.load(projection)
    trainer = TrainerFactory.create_trainer(
        config=ctx.obj["config"],
        projection_matrix=projection_matrix
    )
    trainer.train(Path(dataset), Path(output_dir))

@cli.command()
@click.option("--model-a", type=click.Path(exists=True), required=True, help="Dirty model path")
@click.option("--model-b", type=click.Path(exists=True), required=True, help="Clean model path")
@click.option("--output", type=click.Path(), required=True)
@click.pass_context
def verify(ctx, model_a, model_b, output):
    """Run Causal Head Gating verification and generate comparison report."""
    from .stylometry_net.chg_verifier import CHGVerifier
    from .evaluation.comparative_report import ReportGenerator
    
    verifier = CHGVerifier(ctx.obj["config"])
    
    gates_a = verifier.learn_gates(Path(model_a))
    gates_b = verifier.learn_gates(Path(model_b))
    
    report = ReportGenerator.generate(gates_a, gates_b, Path(output))
    click.echo(f"Verification report saved to {output}")

if __name__ == "__main__":
    cli()
```

#### 4.4.2 SLURM Job Script

```bash
#!/bin/bash
# scripts/slurm/phase_a.sbatch

#SBATCH --job-name=neuro_stylometry_phase_a
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/phase_a_%j.out
#SBATCH --error=logs/phase_a_%j.err

# Load modules
module load python/3.11
module load cuda/12.1
module load cudnn/8.9.7

# Activate virtual environment
source /home/$USER/venvs/neuro_stylometry/bin/activate

# NUMA pinning: bind to the NUMA node with the GPU
NUMA_NODE=$(nvidia-smi topo -m | grep GPU0 | awk '{print $NF}' | head -1)
export CUDA_VISIBLE_DEVICES=0

# Run Phase A with NUMA binding
numactl --cpunodebind=${NUMA_NODE} --membind=${NUMA_NODE} \
    python -m neuro_stylometry run-phase-a \
        --dataset /scratch/$USER/data/sobr_unified.arrow \
        --output-dir /scratch/$USER/artifacts/phase_a \
        --config conf/experiments/my_experiment.yaml \
        --mode hpc

echo "Phase A completed at $(date)"
```

### 4.5 Dependency Management

#### 4.5.1 pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "neuro-stylometry"
version = "0.1.0"
description = "Neuro-Symbolic Stylometry Pipeline with Pollution Mitigation"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
authors = [
    {name = "Your Name", email = "your.email@university.edu"}
]

dependencies = [
    # Core ML
    "torch>=2.2.0",
    "transformers>=4.37.0",
    "datasets>=2.16.0",
    
    # Arrow/Data
    "pyarrow>=14.0.0",
    "pandas>=2.2.0",
    
    # GLiNER
    "gliner>=0.2.0",
    
    # NLP
    "spacy>=3.7.0",
    "tokenizers>=0.15.0",
    
    # Configuration
    "omegaconf>=2.3.0",
    "hydra-core>=1.3.0",
    
    # CLI
    "click>=8.1.0",
    
    # Utilities
    "tqdm>=4.66.0",
    "numpy>=1.26.0",
    "scikit-learn>=1.4.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.2.0",
    "mypy>=1.8.0",
    "pre-commit>=3.6.0",
]
notebooks = [
    "jupyter>=1.0.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
]
hpc = [
    "nvidia-ml-py>=12.0.0",  # For detailed GPU monitoring
]

[project.scripts]
neuro-stylometry = "neuro_stylometry.__main__:cli"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.mypy]
python_version = "3.10"
strict = true
ignore_missing_imports = true
```

---

## 5. Phase A Specification: The Pollution Guard

Phase A constitutes the **Neuro-Symbolic Cleaner** responsible for detecting and removing demographic shortcuts from the SOBR corpus. This section provides the detailed technical specification for the hybrid pipeline combining **GLiNER** (symbolic span detection) and **LEACE** (geometric concept erasure).

### 5.1 Neuro-Symbolic Architecture Overview

The Pollution Guard operates as a two-stage cascade:

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                              PHASE A: POLLUTION GUARD ARCHITECTURE                        │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                           │
│   Stage 1: SYMBOLIC MASKING (GLiNER)                                                     │
│   ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│   │                                                                                     │ │
│   │   Raw Text: "As a 25M, I think politics are broken in Germany."                   │ │
│   │                │                                                                   │ │
│   │                ▼                                                                   │ │
│   │   ┌─────────────────────────────────────────────────────────────────────────────┐ │ │
│   │   │  GLiNER Zero-Shot NER                                                        │ │ │
│   │   │  Taxonomy: [age_statement, gender_indicator, nationality_claim, ...]        │ │ │
│   │   │  Confidence Threshold: 0.85                                                  │ │ │
│   │   └─────────────────────────────────────────────────────────────────────────────┘ │ │
│   │                │                                                                   │ │
│   │                ▼                                                                   │ │
│   │   Detected Spans:                                                                  │ │
│   │     • "25M" → age_statement (0.92)                                                │ │
│   │     • "Germany" → nationality_claim (0.88)                                        │ │
│   │                │                                                                   │ │
│   │                ▼                                                                   │ │
│   │   Masked Text: "As a [MASK:AGE], I think politics are broken in [MASK:NATION]."  │ │
│   │                                                                                     │ │
│   └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                              │                                            │
│                                              ▼                                            │
│   Stage 2: GEOMETRIC PROJECTION (LEACE)                                                  │
│   ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│   │                                                                                     │ │
│   │   Masked Text → Frozen Encoder (RoBERTa) → CLS Embeddings [N × 768]               │ │
│   │                                              │                                      │ │
│   │                                              ▼                                      │ │
│   │   ┌─────────────────────────────────────────────────────────────────────────────┐ │ │
│   │   │  LEACE Computation                                                           │ │ │
│   │   │  Concept: Z (Gender/Age Labels)                                              │ │ │
│   │   │  Covariance: Σ_XZ, Σ_ZZ, Σ_XX                                               │ │ │
│   │   │  Projection: P = I - Σ_XZ·Σ_ZZ⁻¹·Σ_ZX·Σ_XX⁻¹                                │ │ │
│   │   └─────────────────────────────────────────────────────────────────────────────┘ │ │
│   │                │                                                                   │ │
│   │                ▼                                                                   │ │
│   │   Output: Projection Matrix P [768 × 768]                                         │ │
│   │           Properties: Idempotent (P² = P), Nullspace of Z                         │ │
│   │                                                                                     │ │
│   └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                           │
│   Artifacts:                                                                              │
│   ├── clean_dataset.arrow  (Masked text corpus)                                          │
│   ├── projection_matrix.pt (P matrix for Phase D)                                        │
│   └── pollution_logs.json  (All detected spans for analysis)                             │
│                                                                                           │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 GLiNER Integration: Symbolic Span Detection

#### 5.2.1 Model Specification

**GLiNER (Generalist Linear Named Entity Recognition)** is a zero-shot NER model that detects arbitrary entity types without task-specific fine-tuning. We deploy the `urchade/gliner_large-v2.1` variant for its superior generalization on soft semantic categories.

| Property | Specification |
|----------|---------------|
| **Model** | `urchade/gliner_large-v2.1` |
| **Architecture** | BERT-based bi-encoder with linear span scoring |
| **Parameters** | ~350M |
| **Context Window** | 512 tokens (subword) |
| **Zero-Shot Capability** | Detects any entity type described in natural language |
| **Confidence Output** | Continuous score ∈ [0, 1] per detected span |

#### 5.2.2 Taxonomy Definition

The taxonomy defines the entity types that constitute "pollution" in the stylometry context:

```python
# conf/base/gliner_taxonomy.yaml

taxonomy:
  # Age-related self-identification
  age_statement:
    description: "Explicit mention of age or age-related status"
    examples:
      - "I am 25"
      - "25M"
      - "As a boomer"
      - "I'm in my 30s"
      - "(26F)"
    negative_examples:
      - "25 years of experience"  # Not self-identification
      - "The 25th day"
    
  # Gender-related self-identification
  gender_indicator:
    description: "Explicit indication of gender identity or gendered relationships"
    examples:
      - "As a woman"
      - "my husband"
      - "As a mother"
      - "I'm a guy"
      - "(25F)"
    negative_examples:
      - "The woman in the story"  # Third person
      - "Women's rights"  # General topic
    
  # Nationality or location self-identification
  nationality_claim:
    description: "Self-identification with a country, region, or nationality"
    examples:
      - "In my country (Germany)"
      - "As an American"
      - "Here in the UK"
      - "I'm from Brazil"
    negative_examples:
      - "German cars"  # Not self-identification
      - "The American economy"
    
  # Political self-identification
  political_self_id:
    description: "Explicit statement of political affiliation or ideology"
    examples:
      - "As a liberal"
      - "I'm a conservative"
      - "As a Democrat"
      - "Being on the left"
    negative_examples:
      - "Liberal policies"  # Topic discussion
      - "The conservative party"

# Runtime configuration
inference:
  confidence_threshold: 0.85
  max_spans_per_sample: 10
  overlap_strategy: "merge"  # Merge overlapping spans
```

#### 5.2.3 GLiNER Detector Implementation

```python
# src/neuro_stylometry/pollution_guard/gliner_detector.py

from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
from gliner import GLiNER

@dataclass
class DetectedSpan:
    """A detected demographic span in the text."""
    text: str
    start: int
    end: int
    entity_type: str
    confidence: float
    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "entity_type": self.entity_type,
            "confidence": self.confidence
        }

class GLiNERDetector:
    """Wrapper for GLiNER zero-shot NER detection."""
    
    def __init__(
        self,
        model_name: str = "urchade/gliner_large-v2.1",
        taxonomy: List[str] = None,
        confidence_threshold: float = 0.85,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the GLiNER detector.
        
        Args:
            model_name: HuggingFace model identifier
            taxonomy: List of entity type labels to detect
            confidence_threshold: Minimum confidence for span acceptance
            device: Compute device (auto-detected if None)
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = GLiNER.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.taxonomy = taxonomy or [
            "age_statement",
            "gender_indicator",
            "nationality_claim",
            "political_self_id"
        ]
        self.confidence_threshold = confidence_threshold
    
    @torch.no_grad()
    def detect_spans(
        self,
        text: str,
        return_all: bool = False
    ) -> List[DetectedSpan]:
        """
        Detect demographic spans in a single text.
        
        Args:
            text: Input text to analyze
            return_all: If True, return all spans regardless of threshold
        
        Returns:
            List of DetectedSpan objects
        """
        # GLiNER expects entity labels as a list
        raw_entities = self.model.predict_entities(
            text,
            self.taxonomy,
            threshold=0.0 if return_all else self.confidence_threshold
        )
        
        spans = []
        for entity in raw_entities:
            span = DetectedSpan(
                text=entity["text"],
                start=entity["start"],
                end=entity["end"],
                entity_type=entity["label"],
                confidence=entity["score"]
            )
            if return_all or span.confidence >= self.confidence_threshold:
                spans.append(span)
        
        return self._merge_overlapping_spans(spans)
    
    def detect_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[List[DetectedSpan]]:
        """
        Detect spans in a batch of texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for inference
        
        Returns:
            List of span lists, one per input text
        """
        all_spans = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_entities = self.model.batch_predict_entities(
                batch,
                self.taxonomy,
                threshold=self.confidence_threshold
            )
            
            for entities in batch_entities:
                spans = [
                    DetectedSpan(
                        text=e["text"],
                        start=e["start"],
                        end=e["end"],
                        entity_type=e["label"],
                        confidence=e["score"]
                    )
                    for e in entities
                ]
                all_spans.append(self._merge_overlapping_spans(spans))
        
        return all_spans
    
    def _merge_overlapping_spans(
        self,
        spans: List[DetectedSpan]
    ) -> List[DetectedSpan]:
        """Merge overlapping spans, keeping highest confidence."""
        if not spans:
            return spans
        
        # Sort by start position
        sorted_spans = sorted(spans, key=lambda s: (s.start, -s.confidence))
        merged = [sorted_spans[0]]
        
        for span in sorted_spans[1:]:
            last = merged[-1]
            # Check for overlap
            if span.start < last.end:
                # Keep the higher confidence span
                if span.confidence > last.confidence:
                    merged[-1] = span
            else:
                merged.append(span)
        
        return merged
```

### 5.3 LEACE Integration: Geometric Concept Erasure

#### 5.3.1 Mathematical Foundation

LEACE (Least-squares Concept Erasure) computes a projection matrix $\mathbf{P}$ that removes all linear information about a concept $\mathbf{Z}$ from embeddings $\mathbf{X}$ while minimizing distortion.

**Optimization Objective:**

$$\mathbf{P} = \arg\min_{\mathbf{P}} \|\mathbf{P}\mathbf{X} - \mathbf{X}\|_F^2 \quad \text{s.t.} \quad \mathbf{P}\mathbf{X} \perp \mathbf{Z}$$

**Closed-Form Solution:**

$$\mathbf{P} = \mathbf{I} - \mathbf{\Sigma}_{XZ}\mathbf{\Sigma}_{ZZ}^{-1}\mathbf{\Sigma}_{ZX}\mathbf{\Sigma}_{XX}^{-1}$$

Where:
- $\mathbf{\Sigma}_{XZ} = \mathbb{E}[(\mathbf{X} - \mu_X)(\mathbf{Z} - \mu_Z)^T]$ (Cross-covariance)
- $\mathbf{\Sigma}_{ZZ} = \mathbb{E}[(\mathbf{Z} - \mu_Z)(\mathbf{Z} - \mu_Z)^T]$ (Concept covariance)
- $\mathbf{\Sigma}_{XX} = \mathbb{E}[(\mathbf{X} - \mu_X)(\mathbf{X} - \mu_X)^T]$ (Embedding covariance)

**Key Properties of $\mathbf{P}$:**

| Property | Description | Verification |
|----------|-------------|--------------|
| **Idempotence** | $\mathbf{P}^2 = \mathbf{P}$ | $\|\mathbf{P}^2 - \mathbf{P}\|_F < 10^{-5}$ |
| **Nullspace** | $\text{Cov}(\mathbf{P}\mathbf{X}, \mathbf{Z}) = \mathbf{0}$ | Linear probe accuracy ≈ majority baseline |
| **Minimal Distortion** | Minimizes $\|\mathbf{P}\mathbf{x} - \mathbf{x}\|$ | BERTScore(raw, projected) > 0.85 |

#### 5.3.2 LEACE Computer Implementation

```python
# src/neuro_stylometry/pollution_guard/leace.py

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

@dataclass
class LEACEResult:
    """Container for LEACE computation results."""
    projection_matrix: torch.Tensor  # [d, d]
    mean_embedding: torch.Tensor      # [d]
    mean_concept: torch.Tensor        # [k]
    condition_number: float
    idempotence_error: float
    
    def validate(self) -> bool:
        """Verify projection matrix properties."""
        P = self.projection_matrix
        
        # Check idempotence
        P_squared = P @ P
        idemp_error = torch.norm(P_squared - P, p='fro') / torch.norm(P, p='fro')
        
        # Check for NaN/Inf
        has_nan = torch.isnan(P).any()
        has_inf = torch.isinf(P).any()
        
        return idemp_error < 1e-4 and not has_nan and not has_inf

class LEACEComputer:
    """
    Computes LEACE projection matrices for concept erasure.
    
    Supports both full-batch (HPC) and mini-batch (laptop) modes.
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_concepts: int = 1,
        regularization: float = 1e-4,
        device: Optional[torch.device] = None
    ):
        """
        Initialize LEACE computer.
        
        Args:
            hidden_dim: Embedding dimensionality
            num_concepts: Number of concept dimensions (k)
            regularization: Regularization for matrix inversion
            device: Compute device
        """
        self.hidden_dim = hidden_dim
        self.num_concepts = num_concepts
        self.eps = regularization
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Accumulators for mini-batch mode
        self._reset_accumulators()
    
    def _reset_accumulators(self):
        """Reset covariance accumulators."""
        d, k = self.hidden_dim, self.num_concepts
        self._n_samples = 0
        self._sum_X = torch.zeros(d, device=self.device, dtype=torch.float32)
        self._sum_Z = torch.zeros(k, device=self.device, dtype=torch.float32)
        self._sum_XX = torch.zeros(d, d, device=self.device, dtype=torch.float32)
        self._sum_XZ = torch.zeros(d, k, device=self.device, dtype=torch.float32)
        self._sum_ZZ = torch.zeros(k, k, device=self.device, dtype=torch.float32)
    
    def accumulate_batch(
        self,
        embeddings: torch.Tensor,
        concepts: torch.Tensor
    ) -> None:
        """
        Accumulate statistics from a mini-batch.
        
        Args:
            embeddings: [batch, hidden_dim] float tensor
            concepts: [batch, num_concepts] float tensor (one-hot or soft labels)
        """
        X = embeddings.to(device=self.device, dtype=torch.float32)
        Z = concepts.to(device=self.device, dtype=torch.float32)
        
        batch_size = X.shape[0]
        self._n_samples += batch_size
        
        # Accumulate sums for mean computation
        self._sum_X += X.sum(dim=0)
        self._sum_Z += Z.sum(dim=0)
        
        # Accumulate outer products for covariance
        self._sum_XX += X.T @ X
        self._sum_XZ += X.T @ Z
        self._sum_ZZ += Z.T @ Z
    
    def compute_from_accumulators(self) -> LEACEResult:
        """
        Compute projection matrix from accumulated statistics.
        
        Returns:
            LEACEResult containing projection matrix and metadata
        """
        n = self._n_samples
        
        # Compute means
        mu_X = self._sum_X / n
        mu_Z = self._sum_Z / n
        
        # Compute centered covariances
        Sigma_XX = self._sum_XX / n - torch.outer(mu_X, mu_X)
        Sigma_XZ = self._sum_XZ / n - torch.outer(mu_X, mu_Z)
        Sigma_ZZ = self._sum_ZZ / n - torch.outer(mu_Z, mu_Z)
        
        return self._compute_projection(Sigma_XX, Sigma_XZ, Sigma_ZZ, mu_X, mu_Z)
    
    def compute_full_batch(
        self,
        embeddings: torch.Tensor,
        concepts: torch.Tensor
    ) -> LEACEResult:
        """
        Compute projection matrix from full dataset (HPC mode).
        
        Args:
            embeddings: [N, hidden_dim] float tensor (full dataset on GPU)
            concepts: [N, num_concepts] float tensor
        
        Returns:
            LEACEResult containing projection matrix and metadata
        """
        X = embeddings.to(device=self.device, dtype=torch.float32)
        Z = concepts.to(device=self.device, dtype=torch.float32)
        
        n = X.shape[0]
        
        # Compute means
        mu_X = X.mean(dim=0)
        mu_Z = Z.mean(dim=0)
        
        # Center the data
        X_centered = X - mu_X.unsqueeze(0)
        Z_centered = Z - mu_Z.unsqueeze(0)
        
        # Compute covariances via batched GEMM (Tensor Core accelerated)
        Sigma_XX = (X_centered.T @ X_centered) / n
        Sigma_XZ = (X_centered.T @ Z_centered) / n
        Sigma_ZZ = (Z_centered.T @ Z_centered) / n
        
        return self._compute_projection(Sigma_XX, Sigma_XZ, Sigma_ZZ, mu_X, mu_Z)
    
    def _compute_projection(
        self,
        Sigma_XX: torch.Tensor,
        Sigma_XZ: torch.Tensor,
        Sigma_ZZ: torch.Tensor,
        mu_X: torch.Tensor,
        mu_Z: torch.Tensor
    ) -> LEACEResult:
        """
        Compute the LEACE projection matrix from covariance matrices.
        
        Uses Cholesky decomposition for numerically stable inversion.
        """
        d = Sigma_XX.shape[0]
        
        # Add regularization for numerical stability
        Sigma_XX_reg = Sigma_XX + self.eps * torch.eye(d, device=self.device)
        Sigma_ZZ_reg = Sigma_ZZ + self.eps * torch.eye(
            Sigma_ZZ.shape[0], device=self.device
        )
        
        # Cholesky-based inversion of Sigma_ZZ
        try:
            L_ZZ = torch.linalg.cholesky(Sigma_ZZ_reg)
            Sigma_ZZ_inv = torch.cholesky_inverse(L_ZZ)
        except RuntimeError:
            # Fallback to pseudo-inverse if Cholesky fails
            Sigma_ZZ_inv = torch.linalg.pinv(Sigma_ZZ_reg)
        
        # Cholesky-based inversion of Sigma_XX
        try:
            L_XX = torch.linalg.cholesky(Sigma_XX_reg)
            Sigma_XX_inv = torch.cholesky_inverse(L_XX)
        except RuntimeError:
            Sigma_XX_inv = torch.linalg.pinv(Sigma_XX_reg)
        
        # Compute projection matrix: P = I - Sigma_XZ @ Sigma_ZZ^-1 @ Sigma_ZX @ Sigma_XX^-1
        # Note: Sigma_ZX = Sigma_XZ.T
        intermediate = Sigma_XZ @ Sigma_ZZ_inv @ Sigma_XZ.T @ Sigma_XX_inv
        P = torch.eye(d, device=self.device) - intermediate
        
        # Compute metrics
        condition_number = torch.linalg.cond(Sigma_ZZ_reg).item()
        P_squared = P @ P
        idempotence_error = (
            torch.norm(P_squared - P, p='fro') / torch.norm(P, p='fro')
        ).item()
        
        return LEACEResult(
            projection_matrix=P,
            mean_embedding=mu_X,
            mean_concept=mu_Z,
            condition_number=condition_number,
            idempotence_error=idempotence_error
        )
```

### 5.4 Span Masking Logic

```python
# src/neuro_stylometry/pollution_guard/masker.py

from typing import List, Tuple
import re

class SpanMasker:
    """Applies typed masks to detected demographic spans."""
    
    # Mapping from entity types to mask tokens
    MASK_TOKENS = {
        "age_statement": "[MASK:AGE]",
        "gender_indicator": "[MASK:GENDER]",
        "nationality_claim": "[MASK:NATION]",
        "political_self_id": "[MASK:POLITICAL]",
        "default": "[MASK:DEMO]"
    }
    
    def __init__(self, preserve_spacing: bool = True):
        """
        Initialize the span masker.
        
        Args:
            preserve_spacing: If True, preserve original whitespace around masks
        """
        self.preserve_spacing = preserve_spacing
    
    def apply_masks(
        self,
        text: str,
        spans: List["DetectedSpan"]
    ) -> Tuple[str, List[dict]]:
        """
        Replace detected spans with typed mask tokens.
        
        Args:
            text: Original text
            spans: List of DetectedSpan objects
        
        Returns:
            Tuple of (masked_text, mask_log)
        """
        if not spans:
            return text, []
        
        # Sort spans by start position (descending) to avoid index shifting
        sorted_spans = sorted(spans, key=lambda s: s.start, reverse=True)
        
        masked_text = text
        mask_log = []
        
        for span in sorted_spans:
            mask_token = self.MASK_TOKENS.get(
                span.entity_type,
                self.MASK_TOKENS["default"]
            )
            
            # Replace the span
            masked_text = (
                masked_text[:span.start] +
                mask_token +
                masked_text[span.end:]
            )
            
            mask_log.append({
                "original_text": span.text,
                "start": span.start,
                "end": span.end,
                "entity_type": span.entity_type,
                "confidence": span.confidence,
                "mask_token": mask_token
            })
        
        # Reverse log to match original text order
        mask_log.reverse()
        
        return masked_text, mask_log
    
    def extract_mask_positions(self, masked_text: str) -> List[Tuple[int, int, str]]:
        """
        Extract positions of mask tokens in masked text.
        
        Returns:
            List of (start, end, mask_type) tuples
        """
        positions = []
        
        for mask_type, mask_token in self.MASK_TOKENS.items():
            for match in re.finditer(re.escape(mask_token), masked_text):
                positions.append((match.start(), match.end(), mask_type))
        
        return sorted(positions, key=lambda x: x[0])
```

### 5.5 Embedding Extraction

```python
# src/neuro_stylometry/pollution_guard/embedder.py

from typing import Optional, List
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class FrozenEmbedder:
    """Extracts CLS embeddings from a frozen transformer encoder."""
    
    def __init__(
        self,
        model_name: str = "roberta-base",
        device: Optional[torch.device] = None,
        max_length: int = 512
    ):
        """
        Initialize the frozen embedder.
        
        Args:
            model_name: HuggingFace model identifier
            device: Compute device
            max_length: Maximum sequence length for tokenization
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.max_length = max_length
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.hidden_dim = self.model.config.hidden_size
    
    @torch.no_grad()
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        return_dtype: torch.dtype = torch.float16
    ) -> torch.Tensor:
        """
        Extract CLS embeddings for a list of texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for inference
            return_dtype: Output tensor dtype
        
        Returns:
            Tensor of shape [len(texts), hidden_dim]
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Move to device
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Extract CLS embeddings (first token)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            
            all_embeddings.append(cls_embeddings.to(dtype=return_dtype))
        
        return torch.cat(all_embeddings, dim=0)
    
    def embed_single(self, text: str) -> torch.Tensor:
        """Embed a single text, returning [1, hidden_dim] tensor."""
        return self.embed_texts([text], batch_size=1)
```

---

## 6. Phase A Requirements List

This section enumerates the detailed technical requirements specific to Phase A implementation, covering I/O constraints, performance targets, and algorithmic specifications.

### 6.1 Memory-Mapped I/O Requirements

| ID | Requirement | Specification | Rationale |
|----|-------------|---------------|-----------|
| **PA-IO-01** | **Arrow Memory Mapping** | All Arrow dataset reads must use `memory_map=True` via `pyarrow.ipc.open_file()` or HuggingFace `load_from_disk(keep_in_memory=False)`. | Enables lazy page loading, prevents full dataset materialization in RAM. |
| **PA-IO-02** | **NVMe Local Storage** | HPC execution must read datasets from node-local NVMe (`/scratch/`, `/tmp/`) rather than networked filesystems (Lustre, GPFS). | Random access patterns in shuffled batching cause excessive NFS round-trips; local NVMe provides 3.5 GB/s vs. 200 MB/s networked reads. |
| **PA-IO-03** | **Page Cache Warmup** | First epoch may pre-fault Arrow pages via sequential scan before shuffled access begins. | Pre-faulting amortizes page fault latency over initial scan rather than random training iterations. |
| **PA-IO-04** | **Zero-Copy DataLoader** | DataLoader workers must receive Arrow table pointers via shared memory IPC, not pickle serialization. | Eliminates O(dataset_size × num_workers) memory duplication on process fork. |
| **PA-IO-05** | **Dict Encoding Preservation** | Dictionary-encoded columns (`nationality`, `political_leaning`) must remain encoded during all operations; decoding occurs only at final output. | Maintains 30-40% memory reduction; decoding is O(1) per access via index lookup. |

### 6.2 Tokenization Requirements

| ID | Requirement | Specification | Rationale |
|----|-------------|---------------|-----------|
| **PA-TOK-01** | **Rust Tokenizers (HPC)** | HPC mode must use `tokenizers` library (Rust backend) for all tokenization operations. | 10-20× speedup over Python `re` module; parallelizable across CPU cores. |
| **PA-TOK-02** | **Python Regex Fallback (Laptop)** | Laptop mode may use standard `re` module for taxonomy pattern matching when `tokenizers` compilation fails. | Avoids Rust toolchain dependency on consumer machines. |
| **PA-TOK-03** | **Pre-tokenization Cache** | For repeated inference passes (e.g., hyperparameter sweeps), tokenized sequences should be cached to Arrow binary columns. | Eliminates redundant tokenization; cache hit rate > 90% after first pass. |
| **PA-TOK-04** | **Maximum Sequence Length** | All tokenizers must truncate to `max_length=512` tokens; longer sequences are truncated with `truncation=True`. | GLiNER context window is 512 tokens; RoBERTa positional embeddings max at 512. |
| **PA-TOK-05** | **Subword Alignment** | Span detection must operate on character offsets, then map to subword token boundaries for embedding extraction. | GLiNER returns character offsets; transformer models consume subword tokens. |

### 6.3 Smart Bucketing for GLiNER Inference

| ID | Requirement | Specification | Rationale |
|----|-------------|---------------|-----------|
| **PA-BUCK-01** | **Bucket Boundaries** | Compute 8-12 buckets via dynamic programming to minimize total padding waste + bucket count trade-off. | Fewer buckets = more padding waste; too many buckets = CUDA Graph cache explosion. |
| **PA-BUCK-02** | **Padding Ratio Target** | Mean padding ratio across all buckets must be ≤ 15% (vs. ~60% for random batching). | Padding tokens consume Tensor Core FLOPs with zero information content. |
| **PA-BUCK-03** | **Bucket Shuffle** | Within each epoch, shuffle sample order within buckets, then round-robin across buckets. | Maintains bucketing efficiency while providing stochastic gradient estimates. |
| **PA-BUCK-04** | **Dynamic Re-bucketing** | If sequence length distribution shifts (e.g., new data splits), recompute bucket boundaries. | Static boundaries become suboptimal as data evolves. |

**Bucket Boundary Algorithm:**

```python
def compute_bucket_boundaries(
    sequence_lengths: List[int],
    max_buckets: int = 10,
    target_padding_ratio: float = 0.15
) -> List[int]:
    """
    Compute optimal bucket boundaries using dynamic programming.
    
    Objective: Minimize total padding waste subject to bucket count constraint.
    
    Args:
        sequence_lengths: List of sequence lengths for all samples
        max_buckets: Maximum number of buckets
        target_padding_ratio: Target padding ratio per bucket
    
    Returns:
        List of boundary sequence lengths (e.g., [64, 128, 256, 512])
    """
    sorted_lengths = sorted(sequence_lengths)
    n = len(sorted_lengths)
    
    # precompute cumulative statistics
    cum_sum = [0] * (n + 1)
    cum_max = [0] * (n + 1)
    for i, length in enumerate(sorted_lengths):
        cum_sum[i + 1] = cum_sum[i] + length
        cum_max[i + 1] = length  # Last element is max since sorted
    
    # dp[i][k] = minimum waste to bucket samples [0, i) using k buckets
    INF = float('inf')
    dp = [[INF] * (max_buckets + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    
    # backtrack[i][k] = optimal split point for dp[i][k]
    backtrack = [[0] * (max_buckets + 1) for _ in range(n + 1)]
    
    def waste(start: int, end: int) -> float:
        """Compute padding waste for bucket [start, end)."""
        if start >= end:
            return 0
        max_len = sorted_lengths[end - 1]  # Max in range
        total_len = cum_sum[end] - cum_sum[start]
        padded_len = max_len * (end - start)
        return padded_len - total_len
    
    for i in range(1, n + 1):
        for k in range(1, min(i, max_buckets) + 1):
            for j in range(k - 1, i):
                candidate_waste = dp[j][k - 1] + waste(j, i)
                if candidate_waste < dp[i][k]:
                    dp[i][k] = candidate_waste
                    backtrack[i][k] = j
    
    # Find optimal number of buckets
    best_k = 1
    for k in range(1, max_buckets + 1):
        total_waste = dp[n][k]
        total_len = cum_sum[n]
        if total_len > 0 and total_waste / total_len <= target_padding_ratio:
            best_k = k
            break
    
    # Backtrack to get boundaries
    boundaries = []
    i = n
    k = best_k
    while k > 0:
        j = backtrack[i][k]
        if j < i:
            boundaries.append(sorted_lengths[i - 1])  # Max length in bucket
        i = j
        k -= 1
    
    return sorted(set(boundaries))
```

### 6.4 CUDA Graph Constraints for GLiNER

| ID | Requirement | Specification | Rationale |
|----|-------------|---------------|-----------|
| **PA-GRAPH-01** | **Graph Cache Key** | Cache CUDA graphs per `(batch_size, sequence_length)` tuple; maximum 100 cached graphs. | Different input shapes require different graphs; LRU eviction for memory. |
| **PA-GRAPH-02** | **Warmup Phase** | First 10 batches per bucket execute eagerly before graph capture. | CUDA Graph capture requires stable memory layout; warmup prevents reallocation. |
| **PA-GRAPH-03** | **Static Input Buffers** | Graph capture requires pre-allocated input tensors of fixed shape; copy data into static buffers before replay. | Graph replay cannot handle dynamically shaped inputs. |
| **PA-GRAPH-04** | **Disable during Debugging** | CUDA graphs must be disabled when `CUDA_LAUNCH_BLOCKING=1` is set (debugging mode). | Graph capture is incompatible with synchronous kernel execution. |

### 6.5 LEACE Computation Constraints

| ID | Requirement | Specification | Rationale |
|----|-------------|---------------|-----------|
| **PA-LEACE-01** | **FP32 Covariance Accumulation** | All covariance matrix computations must use FP32 precision, regardless of embedding dtype. | FP16 covariance accumulation suffers from catastrophic cancellation; 6-8 bit precision loss observed. |
| **PA-LEACE-02** | **Cholesky Regularization** | Add $\epsilon = 10^{-4}$ to covariance matrix diagonals before Cholesky decomposition. | Near-balanced classes yield near-singular $\Sigma_{ZZ}$; regularization ensures positive definiteness. |
| **PA-LEACE-03** | **Condition Number Logging** | Log condition number of $\Sigma_{ZZ}$ and $\Sigma_{XX}$ for each LEACE computation. | Condition numbers > 10^6 indicate numerical instability; alerts for investigation. |
| **PA-LEACE-04** | **Pseudo-Inverse Fallback** | If Cholesky decomposition fails (non-positive-definite matrix), fall back to Moore-Penrose pseudo-inverse. | Graceful degradation for pathological data distributions. |
| **PA-LEACE-05** | **Idempotence Validation** | Compute $\|\mathbf{P}^2 - \mathbf{P}\|_F / \|\mathbf{P}\|_F$ and assert < $10^{-5}$. | Non-idempotent projection indicates computation error; fail fast. |

### 6.6 Self-Evaluation Probe Constraints

| ID | Requirement | Specification | Rationale |
|----|-------------|---------------|-----------|
| **PA-PROBE-01** | **Probe Architecture** | Logistic Regression with L2 regularization (C=1.0); no hidden layers. | Linear probe isolates linear information content; non-linear probes can memorize samples. |
| **PA-PROBE-02** | **Train/Test Split** | Probe trained on 80% of dataset, evaluated on 20% holdout (author-stratified). | Prevents author leakage between train and test. |
| **PA-PROBE-03** | **Amnesic Drop Threshold** | If Amnesic Drop < 30% for any protected attribute, re-run Phase A with expanded taxonomy or lower threshold. | Insufficient drop indicates residual leakage; iterative refinement required. |
| **PA-PROBE-04** | **Majority Baseline** | Report majority class baseline accuracy alongside probe accuracy. | Amnesic = Majority ± 2% indicates successful erasure. |

### 6.7 Memory Budget Allocation (HPC)

| Component | Allocation | Notes |
|-----------|------------|-------|
| **Arrow mmap overhead** | 0 GB (lazy) | Pages faulted on demand from OS page cache |
| **GLiNER model** | ~1.5 GB | FP16 weights on GPU |
| **GLiNER batch activations** | ~2 GB | Batch size 256, seq len 256 |
| **CUDA Graph cache** | ~5 GB | 100 cached graphs max |
| **RoBERTa encoder** | ~0.5 GB | Frozen, FP16 |
| **Full embedding cache** | ~0.6 GB | 390k × 768 × FP16 |
| **Covariance matrices** | ~9 MB | 768 × 768 × FP32 × 3 |
| **Projection matrix P** | ~2.4 MB | 768 × 768 × FP32 |
| **Headroom for cuDNN** | ~10 GB | Workspace allocation |
| **Total GPU VRAM** | ~20 GB | Well within 80 GB A100 envelope |

### 6.8 Memory Budget Allocation (Laptop)

| Component | Allocation | Notes |
|-----------|------------|-------|
| **Arrow subset (10k)** | ~30 MB | Loaded to system RAM |
| **GLiNER model** | ~1.5 GB | FP16 on GPU |
| **GLiNER batch activations** | ~50 MB | Batch size 16, seq len 256 |
| **RoBERTa encoder** | ~0.5 GB | Frozen, FP16 |
| **Embedding batch** | ~2 MB | 256 × 768 × FP16 per batch |
| **Covariance accumulators** | ~9 MB | Same as HPC |
| **Total GPU VRAM** | ~2.5 GB | Within 4-8 GB A1000 envelope |
| **System RAM headroom** | ~4 GB | For Python interpreter, data structures |

---

## 7. Phase A Design Patterns

This section provides the detailed class specifications for Phase A strategy pattern implementations, including full method signatures and HPC/Laptop behavioral differences.

### 7.1 Strategy Class Hierarchy

```
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                           PHASE A STRATEGY CLASS HIERARCHY                             │
├───────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   ┌──────────────────────────────────────────────────────────────────────────────┐    │
│   │              «abstract» PollutionFilterStrategy                               │    │
│   ├──────────────────────────────────────────────────────────────────────────────┤    │
│   │  - config: PipelineConfig                                                    │    │
│   │  - hardware_profile: HardwareProfile                                         │    │
│   ├──────────────────────────────────────────────────────────────────────────────┤    │
│   │  + __init__(config, hardware_profile)                                        │    │
│   │  + detect_spans(dataset) → Tuple[Dataset, PollutionLog]        «abstract»    │    │
│   │  + compute_embeddings(dataset) → Tensor                        «abstract»    │    │
│   │  + compute_leace(embeddings, labels) → LEACEResult             «abstract»    │    │
│   │  + run_self_evaluation(embeddings, labels, P) → ProbeMetrics   «abstract»    │    │
│   │  + execute_pipeline(dataset) → PhaseAArtifacts                 «template»    │    │
│   └──────────────────────────────────────────────────────────────────────────────┘    │
│                          ▲                                ▲                            │
│                          │                                │                            │
│         ┌────────────────┴────────────────┐  ┌───────────┴───────────────┐            │
│         │        HPCFilterStrategy        │  │    LaptopFilterStrategy   │            │
│         ├─────────────────────────────────┤  ├───────────────────────────┤            │
│         │  - graph_cache: GraphCache      │  │  - sample_size: int       │            │
│         │  - bucket_sampler: BucketSampler│  │  - accumulator: LEACEAcc  │            │
│         ├─────────────────────────────────┤  ├───────────────────────────┤            │
│         │  + detect_spans()               │  │  + detect_spans()         │            │
│         │    → CUDA Graph + Bucketing     │  │    → Eager + Sequential   │            │
│         │  + compute_embeddings()         │  │  + compute_embeddings()   │            │
│         │    → Full dataset GPU pass      │  │    → Mini-batch CPU/GPU   │            │
│         │  + compute_leace()              │  │  + compute_leace()        │            │
│         │    → Batched GEMM on Tensor Core│  │    → Accumulator pattern  │            │
│         └─────────────────────────────────┘  └───────────────────────────┘            │
│                                                                                        │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Abstract Base Class: PollutionFilterStrategy

```python
# src/neuro_stylometry/pollution_guard/strategies/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List
from pathlib import Path
import torch
from datasets import Dataset

from ..gliner_detector import DetectedSpan
from ..leace import LEACEResult
from ...config import PipelineConfig
from ...hardware_ops.detection import HardwareProfile

@dataclass
class PollutionLog:
    """Container for pollution detection results."""
    total_samples: int
    samples_with_spans: int
    total_spans_detected: int
    spans_by_type: dict[str, int]
    sample_logs: List[dict]  # Per-sample span details
    
    def to_json(self, path: Path) -> None:
        import json
        with open(path, 'w') as f:
            json.dump({
                "summary": {
                    "total_samples": self.total_samples,
                    "samples_with_spans": self.samples_with_spans,
                    "total_spans_detected": self.total_spans_detected,
                    "spans_by_type": self.spans_by_type,
                    "pollution_rate": self.samples_with_spans / self.total_samples
                },
                "sample_logs": self.sample_logs
            }, f, indent=2)

@dataclass
class ProbeMetrics:
    """Container for linear probe evaluation metrics."""
    accuracy_before: float
    accuracy_after: float
    amnesic_drop: float
    majority_baseline: float
    
    @property
    def is_erasure_successful(self) -> bool:
        return self.amnesic_drop >= 0.30

class PollutionFilterStrategy(ABC):
    """
    Abstract base class for Phase A pollution filtering strategies.
    
    Defines the template method pattern for the Phase A pipeline,
    with abstract methods for hardware-specific implementations.
    """
    
    def __init__(
        self,
        config: PipelineConfig,
        hardware_profile: HardwareProfile
    ):
        self.config = config
        self.hardware_profile = hardware_profile
        self.device = self._get_device()
    
    @abstractmethod
    def _get_device(self) -> torch.device:
        """Return the primary compute device for this strategy."""
        ...
    
    @abstractmethod
    def detect_spans(
        self,
        dataset: Dataset
    ) -> Tuple[Dataset, PollutionLog]:
        """
        Detect and mask demographic spans in the dataset.
        
        Args:
            dataset: Input dataset with 'post' column
        
        Returns:
            Tuple of (masked_dataset with 'post_masked' column, pollution_log)
        """
        ...
    
    @abstractmethod
    def compute_embeddings(
        self,
        dataset: Dataset,
        text_column: str = "post_masked"
    ) -> torch.Tensor:
        """
        Extract CLS embeddings from the masked texts.
        
        Args:
            dataset: Dataset with masked text column
            text_column: Name of the text column to embed
        
        Returns:
            Tensor of shape [len(dataset), hidden_dim] on appropriate device
        """
        ...
    
    @abstractmethod
    def compute_leace(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> LEACEResult:
        """
        Compute the LEACE projection matrix.
        
        Args:
            embeddings: [N, hidden_dim] tensor of CLS embeddings
            labels: [N] tensor of concept labels (e.g., gender)
        
        Returns:
            LEACEResult containing projection matrix and metadata
        """
        ...
    
    @abstractmethod
    def run_self_evaluation(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        projection_matrix: torch.Tensor
    ) -> ProbeMetrics:
        """
        Evaluate erasure effectiveness via linear probes.
        
        Args:
            embeddings: Original CLS embeddings [N, hidden_dim]
            labels: Concept labels [N]
            projection_matrix: LEACE projection matrix [hidden_dim, hidden_dim]
        
        Returns:
            ProbeMetrics with before/after accuracy and amnesic drop
        """
        ...
    
    def execute_pipeline(
        self,
        dataset: Dataset,
        output_dir: Path
    ) -> "PhaseAArtifacts":
        """
        Template method executing the full Phase A pipeline.
        
        This method orchestrates the strategy-specific implementations.
        """
        from ...config import PhaseAArtifacts
        
        # Step 1: Span detection and masking
        masked_dataset, pollution_log = self.detect_spans(dataset)
        
        # Save pollution log
        pollution_log_path = output_dir / "pollution_logs.json"
        pollution_log.to_json(pollution_log_path)
        
        # Step 2: Embedding extraction
        embeddings = self.compute_embeddings(masked_dataset)
        
        # Step 3: Extract labels for LEACE (e.g., gender)
        labels = self._extract_concept_labels(masked_dataset)
        
        # Step 4: Compute LEACE projection
        leace_result = self.compute_leace(embeddings, labels)
        
        # Validate projection matrix
        if not leace_result.validate():
            raise ValueError(
                f"LEACE projection matrix validation failed: "
                f"idempotence_error={leace_result.idempotence_error}"
            )
        
        # Save projection matrix
        projection_path = output_dir / "projection_matrix.pt"
        torch.save({
            "P": leace_result.projection_matrix,
            "mean_embedding": leace_result.mean_embedding,
            "mean_concept": leace_result.mean_concept,
            "condition_number": leace_result.condition_number
        }, projection_path)
        
        # Step 5: Self-evaluation
        probe_metrics = self.run_self_evaluation(
            embeddings, labels, leace_result.projection_matrix
        )
        
        # Step 6: Save clean dataset
        clean_dataset_path = output_dir / "clean_dataset.arrow"
        masked_dataset.save_to_disk(clean_dataset_path)
        
        # Compute additional metrics
        explicit_recall = self._compute_explicit_recall(masked_dataset, pollution_log)
        bertscore_recall = self._compute_bertscore_sample(dataset, masked_dataset)
        perplexity_delta = self._compute_perplexity_delta(dataset, masked_dataset)
        
        return PhaseAArtifacts(
            clean_dataset_path=clean_dataset_path,
            projection_matrix_path=projection_path,
            pollution_log_path=pollution_log_path,
            explicit_recall=explicit_recall,
            amnesic_drop=probe_metrics.amnesic_drop,
            bertscore_recall=bertscore_recall,
            perplexity_delta=perplexity_delta
        )
    
    def _extract_concept_labels(self, dataset: Dataset) -> torch.Tensor:
        """Extract concept labels (e.g., gender) for LEACE computation."""
        # Default: use 'female' column as binary concept
        if "female" in dataset.column_names:
            labels = torch.tensor(dataset["female"], dtype=torch.float32)
            return labels.unsqueeze(1).to(self.device)
        else:
            raise ValueError("No concept column found for LEACE")
    
    def _compute_explicit_recall(
        self,
        dataset: Dataset,
        pollution_log: PollutionLog
    ) -> float:
        """Compute recall against regex reference patterns."""
        # Simplified: compare GLiNER detections to simple regex patterns
        import re
        
        REFERENCE_PATTERNS = [
            r"\b\d{1,2}[MF]\b",  # Age+Gender shorthand
            r"\bI am \d+\b",     # Age statements
            r"\bAs a (man|woman|guy|girl)\b",  # Gender
        ]
        
        regex_detections = 0
        gliner_detections = pollution_log.total_spans_detected
        
        for sample in dataset["post"]:
            for pattern in REFERENCE_PATTERNS:
                regex_detections += len(re.findall(pattern, sample, re.IGNORECASE))
        
        if regex_detections == 0:
            return 1.0  # No reference patterns to match
        
        return min(gliner_detections / regex_detections, 1.0)
    
    def _compute_bertscore_sample(
        self,
        original: Dataset,
        masked: Dataset,
        sample_size: int = 1000
    ) -> float:
        """Compute mean BERTScore recall on a sample."""
        # Placeholder: actual implementation would use bert_score library
        return 0.90  # Expected value based on mask token preservation
    
    def _compute_perplexity_delta(
        self,
        original: Dataset,
        masked: Dataset,
        sample_size: int = 1000
    ) -> float:
        """Compute perplexity increase from masking."""
        # Placeholder: actual implementation would use GPT-2 perplexity
        return 0.15  # Expected 10-20% increase from masking
```

### 7.3 HPC Strategy Implementation

```python
# src/neuro_stylometry/pollution_guard/strategies/hpc.py

from typing import Tuple
import torch
from datasets import Dataset

from .base import PollutionFilterStrategy, PollutionLog, ProbeMetrics
from ..gliner_detector import GLiNERDetector
from ..masker import SpanMasker
from ..embedder import FrozenEmbedder
from ..leace import LEACEComputer, LEACEResult
from ..probe import LinearProbe
from ...hardware_ops.cuda_graphs import GraphCache
from ...data_engine.bucketing import BucketSampler, compute_bucket_boundaries

class HPCFilterStrategy(PollutionFilterStrategy):
    """
    High-Performance Computing strategy for Phase A.
    
    Optimizations:
    - CUDA Graph caching for GLiNER inference
    - Smart bucketing to minimize padding
    - Full-batch LEACE computation on GPU
    - Tensor Core acceleration via BF16 GEMM
    """
    
    def _get_device(self) -> torch.device:
        return torch.device("cuda:0")
    
    def _initialize_components(self):
        """Lazy initialization of heavy components."""
        if not hasattr(self, "_detector"):
            self._detector = GLiNERDetector(
                model_name=self.config.gliner_model,
                taxonomy=self.config.gliner_taxonomy,
                confidence_threshold=self.config.gliner_confidence_threshold,
                device=self.device
            )
            self._masker = SpanMasker()
            self._embedder = FrozenEmbedder(
                model_name=self.config.encoder_model,
                device=self.device
            )
            self._leace = LEACEComputer(
                hidden_dim=self.config.hidden_dim,
                device=self.device
            )
            self._graph_cache = GraphCache(max_size=100)
    
    def detect_spans(
        self,
        dataset: Dataset
    ) -> Tuple[Dataset, PollutionLog]:
        """
        Detect spans using bucketed batching and CUDA Graph caching.
        """
        self._initialize_components()
        
        texts = dataset["post"]
        
        # Compute bucket boundaries based on tokenized lengths
        token_lengths = self._compute_token_lengths(texts)
        bucket_boundaries = compute_bucket_boundaries(
            token_lengths,
            max_buckets=10,
            target_padding_ratio=0.15
        )
        
        # Create bucket sampler
        bucket_sampler = BucketSampler(
            lengths=token_lengths,
            boundaries=bucket_boundaries,
            batch_size=256
        )
        
        # Process batches with CUDA Graph caching
        all_spans = [[] for _ in range(len(texts))]
        sample_logs = []
        spans_by_type = {}
        
        for batch_indices in bucket_sampler:
            batch_texts = [texts[i] for i in batch_indices]
            batch_len = len(batch_texts[0])  # Approximate for this bucket
            
            # Check graph cache
            cache_key = (len(batch_texts), batch_len)
            
            if cache_key in self._graph_cache:
                # Replay cached graph
                batch_spans = self._graph_cache.replay(
                    cache_key,
                    batch_texts,
                    self._detector
                )
            else:
                # Capture new graph after warmup
                batch_spans = self._detector.detect_batch(
                    batch_texts,
                    batch_size=len(batch_texts)
                )
                
                if self._graph_cache.should_capture(cache_key):
                    self._graph_cache.capture(
                        cache_key,
                        batch_texts,
                        self._detector
                    )
            
            # Store spans
            for i, (idx, spans) in enumerate(zip(batch_indices, batch_spans)):
                all_spans[idx] = spans
                
                for span in spans:
                    span_type = span.entity_type
                    spans_by_type[span_type] = spans_by_type.get(span_type, 0) + 1
                
                if spans:
                    sample_logs.append({
                        "sample_id": idx,
                        "spans": [s.to_dict() for s in spans]
                    })
        
        # Apply masks
        masked_texts = []
        for text, spans in zip(texts, all_spans):
            masked_text, _ = self._masker.apply_masks(text, spans)
            masked_texts.append(masked_text)
        
        # Create new dataset with masked column
        masked_dataset = dataset.add_column("post_masked", masked_texts)
        
        pollution_log = PollutionLog(
            total_samples=len(texts),
            samples_with_spans=sum(1 for spans in all_spans if spans),
            total_spans_detected=sum(len(spans) for spans in all_spans),
            spans_by_type=spans_by_type,
            sample_logs=sample_logs
        )
        
        return masked_dataset, pollution_log
    
    def compute_embeddings(
        self,
        dataset: Dataset,
        text_column: str = "post_masked"
    ) -> torch.Tensor:
        """
        Extract embeddings in a single GPU pass for full dataset.
        """
        self._initialize_components()
        
        texts = dataset[text_column]
        
        # Large batch size for HPC
        embeddings = self._embedder.embed_texts(
            texts,
            batch_size=256,
            return_dtype=torch.float16
        )
        
        return embeddings  # [N, 768] on GPU
    
    def compute_leace(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> LEACEResult:
        """
        Compute LEACE using full-batch GPU computation.
        
        Leverages Tensor Core acceleration for covariance GEMM.
        """
        self._initialize_components()
        
        # One-hot encode labels if necessary
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        
        # Full-batch computation on GPU
        return self._leace.compute_full_batch(embeddings, labels)
    
    def run_self_evaluation(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        projection_matrix: torch.Tensor
    ) -> ProbeMetrics:
        """
        Run linear probe evaluation on GPU.
        """
        # Project embeddings
        projected = embeddings @ projection_matrix.T
        
        # Move to CPU for sklearn
        embeddings_cpu = embeddings.float().cpu().numpy()
        projected_cpu = projected.float().cpu().numpy()
        labels_cpu = labels.squeeze().cpu().numpy()
        
        # Train probes
        probe_before = LinearProbe()
        accuracy_before = probe_before.fit_evaluate(embeddings_cpu, labels_cpu)
        
        probe_after = LinearProbe()
        accuracy_after = probe_after.fit_evaluate(projected_cpu, labels_cpu)
        
        # Majority baseline
        majority_baseline = max(labels_cpu.mean(), 1 - labels_cpu.mean())
        
        return ProbeMetrics(
            accuracy_before=accuracy_before,
            accuracy_after=accuracy_after,
            amnesic_drop=accuracy_before - accuracy_after,
            majority_baseline=majority_baseline
        )
    
    def _compute_token_lengths(self, texts: list[str]) -> list[int]:
        """Compute token lengths for bucketing."""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.encoder_model)
        lengths = []
        
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            lengths.append(min(len(tokens), 512))
        
        return lengths
```

### 7.4 Laptop Strategy Implementation

```python
# src/neuro_stylometry/pollution_guard/strategies/laptop.py

from typing import Tuple
import torch
from datasets import Dataset

from .base import PollutionFilterStrategy, PollutionLog, ProbeMetrics
from ..gliner_detector import GLiNERDetector
from ..masker import SpanMasker
from ..embedder import FrozenEmbedder
from ..leace import LEACEComputer, LEACEResult
from ..probe import LinearProbe

class LaptopFilterStrategy(PollutionFilterStrategy):
    """
    Laptop/Debug strategy for Phase A.
    
    Optimizations for limited resources:
    - Eager execution (no CUDA Graphs)
    - Small batch sizes
    - Mini-batch covariance accumulation
    - Optional CPU fallback
    - Dataset subsetting for development
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_size = getattr(
            self.config, 'subset_size', 10000
        )
    
    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")
    
    def _initialize_components(self):
        """Lazy initialization of components."""
        if not hasattr(self, "_detector"):
            self._detector = GLiNERDetector(
                model_name=self.config.gliner_model,
                taxonomy=self.config.gliner_taxonomy,
                confidence_threshold=self.config.gliner_confidence_threshold,
                device=self.device
            )
            self._masker = SpanMasker()
            self._embedder = FrozenEmbedder(
                model_name=self.config.encoder_model,
                device=self.device
            )
            self._leace = LEACEComputer(
                hidden_dim=self.config.hidden_dim,
                device=self.device
            )
    
    def _maybe_subset(self, dataset: Dataset) -> Dataset:
        """Optionally subset dataset for laptop development."""
        if len(dataset) > self.sample_size:
            indices = torch.randperm(len(dataset))[:self.sample_size].tolist()
            return dataset.select(indices)
        return dataset
    
    def detect_spans(
        self,
        dataset: Dataset
    ) -> Tuple[Dataset, PollutionLog]:
        """
        Detect spans using simple sequential batching.
        """
        self._initialize_components()
        
        # Subset for laptop mode
        working_dataset = self._maybe_subset(dataset)
        texts = working_dataset["post"]
        
        # Simple batch processing (no bucketing)
        batch_size = 16
        all_spans = []
        sample_logs = []
        spans_by_type = {}
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Eager execution
            batch_spans = self._detector.detect_batch(
                batch_texts,
                batch_size=batch_size
            )
            
            for j, spans in enumerate(batch_spans):
                all_spans.append(spans)
                
                for span in spans:
                    span_type = span.entity_type
                    spans_by_type[span_type] = spans_by_type.get(span_type, 0) + 1
                
                if spans:
                    sample_logs.append({
                        "sample_id": i + j,
                        "spans": [s.to_dict() for s in spans]
                    })
        
        # Apply masks
        masked_texts = []
        for text, spans in zip(texts, all_spans):
            masked_text, _ = self._masker.apply_masks(text, spans)
            masked_texts.append(masked_text)
        
        # Create new dataset with masked column
        masked_dataset = working_dataset.add_column("post_masked", masked_texts)
        
        pollution_log = PollutionLog(
            total_samples=len(texts),
            samples_with_spans=sum(1 for spans in all_spans if spans),
            total_spans_detected=sum(len(spans) for spans in all_spans),
            spans_by_type=spans_by_type,
            sample_logs=sample_logs
        )
        
        return masked_dataset, pollution_log
    
    def compute_embeddings(
        self,
        dataset: Dataset,
        text_column: str = "post_masked"
    ) -> torch.Tensor:
        """
        Extract embeddings in mini-batches.
        """
        self._initialize_components()
        
        texts = dataset[text_column]
        
        # Smaller batch size for laptop
        embeddings = self._embedder.embed_texts(
            texts,
            batch_size=32,
            return_dtype=torch.float16
        )
        
        return embeddings
    
    def compute_leace(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> LEACEResult:
        """
        Compute LEACE using mini-batch accumulation.
        
        Memory-efficient for limited GPU VRAM.
        """
        self._initialize_components()
        
        # One-hot encode labels if necessary
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        
        # Reset accumulators
        self._leace._reset_accumulators()
        
        # Process in mini-batches
        batch_size = 512
        n_samples = embeddings.shape[0]
        
        for i in range(0, n_samples, batch_size):
            batch_emb = embeddings[i:i + batch_size]
            batch_lab = labels[i:i + batch_size]
            
            self._leace.accumulate_batch(batch_emb, batch_lab)
        
        # Compute final projection
        return self._leace.compute_from_accumulators()
    
    def run_self_evaluation(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        projection_matrix: torch.Tensor
    ) -> ProbeMetrics:
        """
        Run linear probe evaluation.
        
        Same as HPC but with smaller samples if needed.
        """
        # Project embeddings
        projected = embeddings @ projection_matrix.T
        
        # Move to CPU for sklearn
        embeddings_cpu = embeddings.float().cpu().numpy()
        projected_cpu = projected.float().cpu().numpy()
        labels_cpu = labels.squeeze().cpu().numpy()
        
        # Train probes
        probe_before = LinearProbe()
        accuracy_before = probe_before.fit_evaluate(embeddings_cpu, labels_cpu)
        
        probe_after = LinearProbe()
        accuracy_after = probe_after.fit_evaluate(projected_cpu, labels_cpu)
        
        # Majority baseline
        majority_baseline = max(labels_cpu.mean(), 1 - labels_cpu.mean())
        
        return ProbeMetrics(
            accuracy_before=accuracy_before,
            accuracy_after=accuracy_after,
            amnesic_drop=accuracy_before - accuracy_after,
            majority_baseline=majority_baseline
        )
```

### 7.5 Linear Probe Implementation

```python
# src/neuro_stylometry/pollution_guard/probe.py

from typing import Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class LinearProbe:
    """
    Simple logistic regression probe for evaluating linear information content.
    """
    
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        self.C = C
        self.max_iter = max_iter
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
    
    def fit_evaluate(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Fit probe on train split and evaluate on test split.
        
        Args:
            embeddings: [N, dim] numpy array
            labels: [N] numpy array of binary labels
        
        Returns:
            Test set accuracy
        """
        # Train/test split (author-stratified if author IDs available)
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels
        )
        
        # Fit logistic regression
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver='lbfgs',
            random_state=self.random_state
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        return self.model.score(X_test, y_test)
```

---

## 8. Phase A Implementation Plan

This section provides the step-by-step development tasks for implementing Phase A (The Pollution Guard).

### 8.1 Development Phases

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          PHASE A IMPLEMENTATION ROADMAP                                 │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   Week 1: Data Infrastructure                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│   │  Day 1-2: Arrow schema definition and Pandas-to-Arrow converter                │  │
│   │  Day 3-4: Memory-mapped dataset loading and HuggingFace integration            │  │
│   │  Day 5:   Author-stratified splitting and DataLoader factory                   │  │
│   └─────────────────────────────────────────────────────────────────────────────────┘  │
│                                              │                                          │
│                                              ▼                                          │
│   Week 2: GLiNER Integration                                                           │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│   │  Day 1-2: GLiNER wrapper and taxonomy configuration                            │  │
│   │  Day 3:   Span masking logic with typed tokens                                 │  │
│   │  Day 4:   Bucketing algorithm and BucketSampler                                │  │
│   │  Day 5:   CUDA Graph caching infrastructure                                    │  │
│   └─────────────────────────────────────────────────────────────────────────────────┘  │
│                                              │                                          │
│                                              ▼                                          │
│   Week 3: LEACE and Embedding Pipeline                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│   │  Day 1-2: Frozen embedder with RoBERTa                                         │  │
│   │  Day 3:   LEACE computer (full-batch and mini-batch modes)                     │  │
│   │  Day 4:   Linear probe for self-evaluation                                     │  │
│   │  Day 5:   Strategy pattern wiring and factory                                  │  │
│   └─────────────────────────────────────────────────────────────────────────────────┘  │
│                                              │                                          │
│                                              ▼                                          │
│   Week 4: Integration and Testing                                                      │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│   │  Day 1-2: PhaseAPipeline orchestration and CLI integration                     │  │
│   │  Day 3:   Unit tests for all components                                        │  │
│   │  Day 4:   Integration tests (laptop mode on sample data)                       │  │
│   │  Day 5:   HPC deployment and full dataset validation                           │  │
│   └─────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Task Breakdown

#### 8.2.1 Data Infrastructure Tasks

| Task ID | Description | Files | Verification |
|---------|-------------|-------|--------------|
| **PA-T01** | Define Arrow schemas for SOBR and pollution logs | `data_engine/schemas.py` | Schema validates against sample data |
| **PA-T02** | Implement Pandas-to-Arrow converter with dictionary encoding | `data_engine/converter.py` | Memory reduction ≥30% vs raw |
| **PA-T03** | Create memory-mapped dataset wrapper | `data_engine/dataset.py` | mmap flag verified in PyArrow |
| **PA-T04** | Implement author-stratified train/val/test splitting | `data_engine/dataset.py` | No author overlap between splits |
| **PA-T05** | Create DataLoader factory with hardware-aware configuration | `data_engine/dataloader.py` | Zero-copy verified via memory profiling |

#### 8.2.2 GLiNER Integration Tasks

| Task ID | Description | Files | Verification |
|---------|-------------|-------|--------------|
| **PA-T06** | Create GLiNER wrapper with taxonomy loading | `pollution_guard/gliner_detector.py` | Detects sample spans correctly |
| **PA-T07** | Implement typed span masking | `pollution_guard/masker.py` | Mask tokens match entity types |
| **PA-T08** | Develop bucketing algorithm (DP-based) | `data_engine/bucketing.py` | Padding ratio ≤15% on SOBR |
| **PA-T09** | Create BucketSampler for bucketed batching | `data_engine/bucketing.py` | Batches grouped by length |
| **PA-T10** | Implement CUDA Graph cache | `hardware_ops/cuda_graphs.py` | Graph replay verified, cache hit rate >90% |

#### 8.2.3 LEACE Pipeline Tasks

| Task ID | Description | Files | Verification |
|---------|-------------|-------|--------------|
| **PA-T11** | Create frozen embedder with CLS extraction | `pollution_guard/embedder.py` | Output shape [N, 768], grad disabled |
| **PA-T12** | Implement full-batch LEACE computation | `pollution_guard/leace.py` | Idempotence error <1e-5 |
| **PA-T13** | Implement mini-batch LEACE accumulation | `pollution_guard/leace.py` | Matches full-batch within 5% error |
| **PA-T14** | Create linear probe evaluator | `pollution_guard/probe.py` | Accuracy matches sklearn reference |
| **PA-T15** | Wire strategy pattern infrastructure | `pollution_guard/strategies/*.py` | Factory creates correct strategy |

#### 8.2.4 Integration Tasks

| Task ID | Description | Files | Verification |
|---------|-------------|-------|--------------|
| **PA-T16** | Implement PhaseAPipeline orchestrator | `pollution_guard/__init__.py` | All artifacts generated correctly |
| **PA-T17** | Create CLI commands for Phase A | `__main__.py` | Commands execute end-to-end |
| **PA-T18** | Write unit tests | `tests/unit/test_*.py` | 90%+ coverage |
| **PA-T19** | Write integration tests | `tests/integration/test_phase_a_pipeline.py` | Full pipeline passes |
| **PA-T20** | Deploy and validate on HPC | `scripts/slurm/phase_a.sbatch` | Metrics meet targets |

### 8.3 Verification Checklist

Before proceeding to Phase D, verify:

- [ ] **Artifact Generation**: `clean_dataset.arrow`, `projection_matrix.pt`, `pollution_logs.json` created
- [ ] **Explicit Recall**: ≥95% of reference regex patterns detected by GLiNER
- [ ] **Amnesic Drop**: ≥30% drop in linear probe accuracy after projection
- [ ] **Idempotence**: $\|\mathbf{P}^2 - \mathbf{P}\|_F / \|\mathbf{P}\|_F < 10^{-5}$
- [ ] **Memory Budget**: HPC: <20GB VRAM, Laptop: <3GB VRAM
- [ ] **Throughput**: HPC: ≥1000 samples/sec GLiNER, Laptop: ≥50 samples/sec

---

## 9. Phase A to D Handover Requirements

This section specifies the artifacts and contracts governing the handover from Phase A to Phase D.

### 9.1 Required Artifacts

| Artifact | Path | Format | Description |
|----------|------|--------|-------------|
| **Projection Matrix** | `projection_matrix.pt` | PyTorch checkpoint | Contains `P` [768, 768], `mean_embedding`, `mean_concept`, `condition_number` |
| **Clean Dataset** | `clean_dataset.arrow` | Arrow IPC | Unified dataset with `post_masked` column containing GLiNER-masked text |
| **Pollution Logs** | `pollution_logs.json` | JSON | Per-sample span detection logs for analysis |
| **Phase A Metrics** | `phase_a_metrics.json` | JSON | Explicit recall, amnesic drop, BERTScore, perplexity delta |

### 9.2 Contract Specifications

#### 9.2.1 Projection Matrix Contract

```python
@dataclass
class ProjectionMatrixContract:
    """Contract for Phase A → Phase D projection matrix handover."""
    
    # Tensor requirements
    shape: Tuple[int, int] = (768, 768)
    dtype: torch.dtype = torch.float32
    device_agnostic: bool = True  # Can be loaded to any device
    
    # Validation requirements
    max_idempotence_error: float = 1e-5
    no_nan: bool = True
    no_inf: bool = True
    
    # Companion data
    includes_mean_embedding: bool = True
    includes_condition_number: bool = True
```

#### 9.2.2 Clean Dataset Contract

```python
@dataclass
class CleanDatasetContract:
    """Contract for Phase A → Phase D clean dataset handover."""
    
    # Schema requirements
    required_columns: List[str] = field(default_factory=lambda: [
        "author_id",
        "post",
        "post_masked",
        "female"  # At least one demographic label
    ])
    
    # Format requirements
    format: str = "arrow"
    memory_mappable: bool = True
    
    # Data integrity
    no_empty_masked_posts: bool = True
    mask_tokens_valid: bool = True  # Only allowed mask token types
```

### 9.3 Handover Validation Script

```python
# scripts/validate_phase_a_handover.py

from pathlib import Path
import torch
from datasets import load_from_disk

def validate_handover(artifact_dir: Path) -> bool:
    """Validate all Phase A artifacts before Phase D training."""
    
    errors = []
    
    # 1. Validate projection matrix
    p_path = artifact_dir / "projection_matrix.pt"
    if not p_path.exists():
        errors.append("Missing projection_matrix.pt")
    else:
        checkpoint = torch.load(p_path, map_location="cpu")
        P = checkpoint["P"]
        
        # Shape check
        if P.shape != (768, 768):
            errors.append(f"Invalid P shape: {P.shape}")
        
        # Idempotence check
        P2 = P @ P
        idemp_error = torch.norm(P2 - P, p='fro') / torch.norm(P, p='fro')
        if idemp_error > 1e-5:
            errors.append(f"Idempotence error too high: {idemp_error}")
        
        # NaN/Inf check
        if torch.isnan(P).any():
            errors.append("Projection matrix contains NaN")
        if torch.isinf(P).any():
            errors.append("Projection matrix contains Inf")
    
    # 2. Validate clean dataset
    ds_path = artifact_dir / "clean_dataset.arrow"
    if not ds_path.exists():
        errors.append("Missing clean_dataset.arrow")
    else:
        dataset = load_from_disk(ds_path)
        
        # Column check
        for col in ["post_masked", "female"]:
            if col not in dataset.column_names:
                errors.append(f"Missing column: {col}")
        
        # Empty check
        empty_count = sum(1 for x in dataset["post_masked"] if not x.strip())
        if empty_count > 0:
            errors.append(f"{empty_count} empty masked posts")
    
    # 3. Validate metrics
    metrics_path = artifact_dir / "phase_a_metrics.json"
    if metrics_path.exists():
        import json
        with open(metrics_path) as f:
            metrics = json.load(f)
        
        if metrics.get("amnesic_drop", 0) < 0.30:
            errors.append(f"Amnesic drop below threshold: {metrics['amnesic_drop']}")
    
    if errors:
        print("Phase A validation FAILED:")
        for e in errors:
            print(f"  ❌ {e}")
        return False
    else:
        print("Phase A validation PASSED ✓")
        return True
```

---

## 10. Phase D Specification: The Constrained Transformer

This section specifies Phase D, which trains a demographic profiler under the Affine Guard constraint and verifies learned features via Causal Head Gating.

### 10.1 Constrained Training Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        PHASE D: CONSTRAINED TRANSFORMER ARCHITECTURE                    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   Input Tokens                                                                          │
│        │                                                                               │
│        ▼                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│   │  Embedding Layer                                                                 │  │
│   │  h₀ = Embed(tokens) + PositionalEmbed                                           │  │
│   └─────────────────────────────────────────────────────────────────────────────────┘  │
│        │                                                                               │
│        ▼                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│   │  ★ AFFINE GUARD LAYER ★                                                          │  │
│   │  ─────────────────────────────────────────────────────────────────────────────  │  │
│   │  h_proj = P @ h₀                                                                 │  │
│   │                                                                                  │  │
│   │  Where P is the LEACE projection matrix from Phase A                            │  │
│   │  Properties:                                                                     │  │
│   │    • P is FROZEN (requires_grad=False)                                          │  │
│   │    • P removes linear demographic information                                   │  │
│   │    • Gradients flow THROUGH P but do not UPDATE P                               │  │
│   └─────────────────────────────────────────────────────────────────────────────────┘  │
│        │                                                                               │
│        ▼                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│   │  Transformer Encoder Blocks (L=12)                                              │  │
│   │  ─────────────────────────────────────────────────────────────────────────────  │  │
│   │  For l = 1 to L:                                                                │  │
│   │    h_l = EncoderBlock_l(h_{l-1})                                                │  │
│   │      = LayerNorm(h + MultiHeadAttn(h))                                          │  │
│   │      = LayerNorm(h + FFN(h))                                                    │  │
│   └─────────────────────────────────────────────────────────────────────────────────┘  │
│        │                                                                               │
│        ▼                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│   │  Multi-Task Classification Heads                                                │  │
│   │  ─────────────────────────────────────────────────────────────────────────────  │  │
│   │  For each attribute a ∈ {gender, age, nationality, ...}:                       │  │
│   │    logits_a = Linear_a(CLS_embedding)                                           │  │
│   │    loss_a = CrossEntropy(logits_a, labels_a)                                    │  │
│   │                                                                                  │  │
│   │  total_loss = Σ (weight_a × loss_a)                                             │  │
│   └─────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Affine Guard Layer Implementation

```python
# src/neuro_stylometry/stylometry_net/affine_guard.py

import torch
import torch.nn as nn

class AffineGuard(nn.Module):
    """
    Frozen linear projection layer that removes demographic information.
    
    Implements: h_proj = P @ h_0
    Where P is the LEACE projection matrix computed in Phase A.
    """
    
    def __init__(
        self,
        projection_matrix: torch.Tensor,
        freeze: bool = True
    ):
        """
        Initialize the Affine Guard layer.
        
        Args:
            projection_matrix: LEACE projection matrix [hidden_dim, hidden_dim]
            freeze: If True, projection is non-trainable
        """
        super().__init__()
        
        hidden_dim = projection_matrix.shape[0]
        
        # Register as buffer (not parameter) if frozen
        if freeze:
            self.register_buffer("P", projection_matrix)
        else:
            self.P = nn.Parameter(projection_matrix)
        
        self.frozen = freeze
        self.hidden_dim = hidden_dim
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project hidden states through the demographic nullspace.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim] tensor
        
        Returns:
            Projected hidden states [batch, seq_len, hidden_dim]
        """
        # P @ h for each position
        # hidden_states: [B, S, D], P: [D, D]
        # Result: [B, S, D]
        return torch.einsum("bsd,dd->bsd", hidden_states, self.P)
    
    def extra_repr(self) -> str:
        return f"hidden_dim={self.hidden_dim}, frozen={self.frozen}"
```

### 10.3 AffineGuardTransformer Integration

```python
# src/neuro_stylometry/stylometry_net/transformer.py

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig

from .affine_guard import AffineGuard
from .classification_head import MultiTaskHead

class AffineGuardTransformer(nn.Module):
    """
    Transformer with injected Affine Guard constraint.
    
    The Affine Guard is inserted between the embedding layer
    and the first encoder block, forcing all subsequent
    representations to exist in the demographic nullspace.
    """
    
    def __init__(
        self,
        base_model_name: str,
        projection_matrix: torch.Tensor,
        num_labels_per_task: dict[str, int],
        freeze_projection: bool = True
    ):
        """
        Initialize the constrained transformer.
        
        Args:
            base_model_name: HuggingFace model identifier
            projection_matrix: LEACE projection matrix [hidden_dim, hidden_dim]
            num_labels_per_task: Dict mapping task names to label counts
            freeze_projection: If True, Affine Guard is frozen
        """
        super().__init__()
        
        # Load base model
        self.config = RobertaConfig.from_pretrained(base_model_name)
        self.roberta = RobertaModel.from_pretrained(base_model_name)
        
        # Extract embedding layers
        self.embeddings = self.roberta.embeddings
        
        # Insert Affine Guard
        self.affine_guard = AffineGuard(
            projection_matrix=projection_matrix,
            freeze=freeze_projection
        )
        
        # Keep encoder blocks
        self.encoder = self.roberta.encoder
        
        # Multi-task classification heads
        self.classifier = MultiTaskHead(
            hidden_dim=self.config.hidden_size,
            num_labels_per_task=num_labels_per_task
        )
        
        # Store task names for loss computation
        self.task_names = list(num_labels_per_task.keys())
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: dict[str, torch.Tensor] = None,
        output_attentions: bool = False
    ):
        """
        Forward pass with Affine Guard projection.
        
        Args:
            input_ids: [batch, seq_len] token IDs
            attention_mask: [batch, seq_len] attention mask
            labels: Optional dict of task_name -> label tensor
            output_attentions: If True, return attention weights
        
        Returns:
            Dict with logits, optional loss, optional attentions
        """
        # Step 1: Embeddings
        embedding_output = self.embeddings(input_ids)
        
        # Step 2: Affine Guard projection
        projected = self.affine_guard(embedding_output)
        
        # Step 3: Extended attention mask
        extended_attention_mask = self._get_extended_attention_mask(
            attention_mask, input_ids.shape
        )
        
        # Step 4: Encoder blocks
        encoder_outputs = self.encoder(
            projected,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions
        )
        
        sequence_output = encoder_outputs.last_hidden_state
        
        # Step 5: CLS token for classification
        cls_output = sequence_output[:, 0, :]
        
        # Step 6: Multi-task classification
        logits = self.classifier(cls_output)
        
        # Step 7: Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = self.classifier.compute_loss(logits, labels)
        
        output = {
            "logits": logits,
            "loss": loss,
            "cls_embedding": cls_output
        }
        
        if output_attentions:
            output["attentions"] = encoder_outputs.attentions
        
        return output
    
    def _get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: tuple
    ) -> torch.Tensor:
        """Convert attention mask to extended format."""
        extended = attention_mask[:, None, None, :]
        extended = (1.0 - extended) * torch.finfo(self.roberta.dtype).min
        return extended
```

### 10.4 Causal Head Gating Verification

```python
# src/neuro_stylometry/stylometry_net/chg_verifier.py

from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from tqdm import tqdm

@dataclass
class HeadGates:
    """Container for learned attention head gates."""
    gates: torch.Tensor  # [num_layers, num_heads]
    layer_gate_means: torch.Tensor  # [num_layers]
    
    def classify_heads(
        self,
        facilitating_threshold: float = 0.7,
        irrelevant_threshold: float = 0.3
    ) -> Dict[str, List[Tuple[int, int]]]:
        """Classify heads by gate value."""
        facilitating = []
        irrelevant = []
        neutral = []
        
        num_layers, num_heads = self.gates.shape
        
        for layer in range(num_layers):
            for head in range(num_heads):
                g = self.gates[layer, head].item()
                
                if g > facilitating_threshold:
                    facilitating.append((layer, head))
                elif g < irrelevant_threshold:
                    irrelevant.append((layer, head))
                else:
                    neutral.append((layer, head))
        
        return {
            "facilitating": facilitating,
            "irrelevant": irrelevant,
            "neutral": neutral
        }

class CHGVerifier:
    """
    Causal Head Gating verification for trained models.
    
    Learns gate parameters g_{l,h} ∈ [0,1] for each attention head
    that indicate the head's importance for the classification task.
    """
    
    def __init__(
        self,
        num_layers: int = 12,
        num_heads: int = 12,
        learning_rate: float = 0.01,
        num_epochs: int = 50
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.lr = learning_rate
        self.num_epochs = num_epochs
    
    def learn_gates(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> HeadGates:
        """
        Learn gate parameters for attention heads.
        
        Freezes model weights and trains only gate parameters.
        """
        # Initialize gate parameters (all start at 0.5)
        gates = nn.Parameter(
            torch.full((self.num_layers, self.num_heads), 0.5, device=device)
        )
        
        # Freeze model
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        # Optimizer for gates only
        optimizer = torch.optim.Adam([gates], lr=self.lr)
        
        # Training loop
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = {k: v.to(device) for k, v in batch["labels"].items()}
                
                # Forward with attention outputs
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_attentions=True
                )
                
                # Apply sigmoid to get gate values in [0, 1]
                gate_values = torch.sigmoid(gates)
                
                # Modulate attention outputs
                modulated_loss = self._compute_modulated_loss(
                    outputs, gate_values, labels, model
                )
                
                # Backward on gates
                optimizer.zero_grad()
                modulated_loss.backward()
                optimizer.step()
                
                epoch_loss += modulated_loss.item()
        
        # Final gate values
        final_gates = torch.sigmoid(gates).detach()
        layer_means = final_gates.mean(dim=1)
        
        return HeadGates(
            gates=final_gates,
            layer_gate_means=layer_means
        )
    
    def _compute_modulated_loss(
        self,
        outputs: dict,
        gate_values: torch.Tensor,
        labels: dict,
        model: nn.Module
    ) -> torch.Tensor:
        """Compute loss with attention modulated by gates."""
        # Simplified: use original loss weighted by mean gate value
        # Full implementation would re-run forward with modulated attention
        base_loss = outputs["loss"]
        gate_mean = gate_values.mean()
        
        # Encourage sparse gates (L1 on gates)
        sparsity_loss = 0.1 * gate_values.abs().mean()
        
        return base_loss * gate_mean + sparsity_loss
```

---

## 11. Phase D Requirements List

### 11.1 Model Architecture Requirements

| ID | Requirement | Specification |
|----|-------------|---------------|
| **PD-ARCH-01** | Affine Guard Placement | Immediately after embedding layer, before first encoder block |
| **PD-ARCH-02** | Guard Freezing | `requires_grad=False` for projection matrix P |
| **PD-ARCH-03** | Gradient Flow | Gradients flow through P but do not update P |
| **PD-ARCH-04** | Multi-Task Heads | Independent classification heads per demographic attribute |
| **PD-ARCH-05** | Base Model | RoBERTa-base (12 layers, 12 heads, 768 hidden) |

### 11.2 Training Requirements

| ID | Requirement | Specification |
|----|-------------|---------------|
| **PD-TRAIN-01** | Mixed Precision (HPC) | BF16 with `torch.autocast`, no GradScaler |
| **PD-TRAIN-02** | Mixed Precision (Laptop) | FP16 with GradScaler |
| **PD-TRAIN-03** | torch.compile (HPC) | `mode="max-autotune"`, `fullgraph=True` |
| **PD-TRAIN-04** | Optimizer | AdamW with weight decay 0.01 |
| **PD-TRAIN-05** | Learning Rate | 2e-5 with linear warmup (10%) and cosine decay |
| **PD-TRAIN-06** | Batch Size (HPC) | 256 per GPU |
| **PD-TRAIN-07** | Batch Size (Laptop) | 16 per GPU |

### 11.3 Verification Requirements

| ID | Requirement | Specification |
|----|-------------|---------------|
| **PD-VER-01** | CHG Training | Freeze model, train gate parameters only |
| **PD-VER-02** | Gate Initialization | All gates start at 0.5 (sigmoid pre-activation) |
| **PD-VER-03** | Head Classification | Facilitating: g>0.7, Irrelevant: g<0.3, Neutral: otherwise |
| **PD-VER-04** | SVS Computation | Ratio of function word attention to content word attention |
| **PD-VER-05** | Comparative Report | Side-by-side Model A (dirty) vs Model B (clean) |

---

## 12. Phase D Design Patterns

### 12.1 Training Strategy Pattern

```python
# src/neuro_stylometry/stylometry_net/strategies/base.py

from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class TrainingStrategy(ABC):
    """Abstract strategy for Phase D training."""
    
    @abstractmethod
    def prepare_model(self, model: nn.Module, P: torch.Tensor) -> nn.Module:
        """Inject Affine Guard and apply optimizations."""
        ...
    
    @abstractmethod
    def get_autocast_context(self):
        """Return the appropriate autocast context."""
        ...
    
    @abstractmethod
    def training_step(self, model, batch, loss_fn) -> torch.Tensor:
        """Execute one training step with appropriate precision."""
        ...
```

### 12.2 HPC Training Strategy

```python
# src/neuro_stylometry/stylometry_net/strategies/hpc.py

class HPCTrainingStrategy(TrainingStrategy):
    """HPC training with BF16, torch.compile, and large batches."""
    
    def prepare_model(self, model: nn.Module, P: torch.Tensor) -> nn.Module:
        # Inject Affine Guard
        model = self._inject_affine_guard(model, P)
        
        # Apply torch.compile
        model = torch.compile(
            model,
            mode="max-autotune",
            fullgraph=True
        )
        
        # Move to BF16
        model = model.to(dtype=torch.bfloat16)
        
        return model
    
    def get_autocast_context(self):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    
    def training_step(self, model, batch, loss_fn) -> torch.Tensor:
        with self.get_autocast_context():
            outputs = model(**batch)
            return outputs["loss"]
```

### 12.3 Laptop Training Strategy

```python
# src/neuro_stylometry/stylometry_net/strategies/laptop.py

class LaptopTrainingStrategy(TrainingStrategy):
    """Laptop training with FP16, GradScaler, and small batches."""
    
    def __init__(self):
        self.scaler = torch.cuda.amp.GradScaler()
    
    def prepare_model(self, model: nn.Module, P: torch.Tensor) -> nn.Module:
        # Inject Affine Guard (no torch.compile)
        model = self._inject_affine_guard(model, P)
        return model
    
    def get_autocast_context(self):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    
    def training_step(self, model, batch, loss_fn) -> torch.Tensor:
        with self.get_autocast_context():
            outputs = model(**batch)
            loss = outputs["loss"]
        
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
        
        return loss
```

---

## 13. Phase D Implementation Plan

### 13.1 Development Phases

| Week | Focus | Key Deliverables |
|------|-------|------------------|
| 1 | Model Architecture | `AffineGuard`, `AffineGuardTransformer`, `MultiTaskHead` |
| 2 | Training Loop | `Trainer`, precision management, checkpointing |
| 3 | Verification | `CHGVerifier`, `SVSCalculator`, attention analysis |
| 4 | Integration | CLI commands, comparative reports, full pipeline |

### 13.2 Task Breakdown

| Task ID | Description | Files |
|---------|-------------|-------|
| **PD-T01** | Implement AffineGuard layer | `stylometry_net/affine_guard.py` |
| **PD-T02** | Create AffineGuardTransformer | `stylometry_net/transformer.py` |
| **PD-T03** | Implement MultiTaskHead | `stylometry_net/classification_head.py` |
| **PD-T04** | Create Trainer class | `training/trainer.py` |
| **PD-T05** | Implement precision management | `hardware_ops/precision.py` |
| **PD-T06** | Add checkpointing logic | `training/checkpointing.py` |
| **PD-T07** | Implement CHGVerifier | `stylometry_net/chg_verifier.py` |
| **PD-T08** | Create SVSCalculator | `stylometry_net/svs_calculator.py` |
| **PD-T09** | Implement attention analysis | `evaluation/attention_analysis.py` |
| **PD-T10** | Create comparative report generator | `evaluation/comparative_report.py` |
| **PD-T11** | Wire CLI commands | `__main__.py` |
| **PD-T12** | Write unit tests | `tests/unit/test_phase_d_*.py` |
| **PD-T13** | Write integration tests | `tests/integration/test_phase_d_training.py` |
| **PD-T14** | Deploy and validate on HPC | `scripts/slurm/phase_d.sbatch` |

### 13.3 Final Verification Checklist

Before declaring the pipeline complete:

- [ ] **Model A (Dirty)**: Trained on raw data, checkpoints saved
- [ ] **Model B (Clean)**: Trained with Affine Guard, checkpoints saved
- [ ] **CHG Analysis**: Gate distributions computed for both models
- [ ] **SVS Scores**: Clean model SVS > Dirty model SVS
- [ ] **Comparative Report**: Generated with visualizations
- [ ] **All Tests Passing**: Unit + integration tests at 90%+ coverage
- [ ] **Documentation Complete**: README, docstrings, type hints

---

## 14. Conclusion

This implementation plan provides a comprehensive, rigorous specification for the Neuro-Symbolic Stylometry Pipeline, covering:

1. **End-to-End Requirements**: 26 functional and 31 non-functional requirements
2. **Design Patterns**: Strategy, Factory, and Data Object patterns for dual-mode execution
3. **Phase A (Pollution Guard)**: GLiNER + LEACE hybrid cleaner with full class specifications
4. **Phase D (Constrained Transformer)**: Affine Guard injection and CHG verification
5. **Handover Contracts**: Explicit artifact specifications between phases

The dual-mode architecture (HPC/Laptop) ensures the system can be developed iteratively on consumer hardware and deployed efficiently on high-performance clusters, while maintaining identical algorithmic behavior across environments.

---

*Document Version: 1.0*  
*Last Updated: 2024-12-27*

