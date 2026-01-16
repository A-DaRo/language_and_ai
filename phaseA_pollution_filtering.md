This report delineates the final, validated architecture for **Phase A: Pollution Detection and Mitigation**.

### **Executive Summary: The Hybrid Neuro-Symbolic Pipeline**

We reject the pure latent-space approach (SAE/RepE) in favor of a **Hybrid Neuro-Symbolic Architecture** combining **GLiNER** (symbolic/content layer) and **LEACE** (geometric/style layer).

**Core Argument:** Autonomous pollution detection requires solving two distinct failure modes:
1.  **Explicit Leakage (The "Elazar" Condition):** Demographic traits explicitly stated in text (e.g., "As a 25M...") act as shortcuts that bypass linear subspace removal. *Elazar & Goldberg (2018)* proved that adversarial removal fails if these input tokens remain.
2.  **Implicit Leakage (The "Belrose" Condition):** Even with explicit tokens removed, stylistic choices (syntax, adjective frequency) encode demographic priors linearly in the embedding space. *Belrose et al. (2024)* proved that **LEACE** provides the optimal closed-form removal for this residual signal.

The proposed pipeline is **autonomous** because it uses zero-shot generalization (GLiNER) to detect new pollution types without training data, and **preserves stylometry** by targeting only the specific linear subspaces of the demographic traits, leaving the orthogonal "style subspace" intact.

***

### **1. Extended Literature Review & Rationale**

#### **1.1 The Necessity of Neuro-Symbolic Cleaning**
*   **Failure of Pure Latent Methods:** *Elazar & Goldberg (2018)*  and *Ravfogel et al. (2020)*  established that removing bias from embeddings is mathematically impossible if the input text contains strong explicit signals. Non-linear classifiers can simply "re-derive" the bias from the tokens. Thus, **Phase A must perform text-level redaction.**[1][2]
*   **Superiority of GLiNER for Autonomous Detection:** Traditional NER fails on "soft" concepts (e.g., "political affiliation"). *Urchade et al. (2023, 2024)*  demonstrate that **GLiNER** achieves State-of-the-Art (SOTA) zero-shot performance, generalizing to unseen entity types like `[gender_indicator]` or `[age_statement]` better than LLMs (GPT-4) or traditional BERT-NER. This enables the "autonomous" requirement: you define a taxonomy, and the system detects it without labeled training data.[3][4]

#### **1.2 The Role of LEACE in Stylometry**
*   **Residual Linear Leakage:** Once text is redacted, the remaining signal is "implicit style." *Belrose et al. (2024)*  introduced **LEACE**, proving it removes *all* linear information about a concept with the *minimum possible damage* to the rest of the embedding.[5]
*   **Why LEACE fits Phase D:** Phase D (Neural Stylometry) relies on attention mechanisms. *Belrose et al.* show that LEACE acts as an "affine guard," ensuring that the attention heads cannot trivially attend to a "gender dimension." This forces the Phase D model to learn deeper, structural stylometric features (like parse tree depth or function word usage) rather than demographic shortcuts.

#### **1.3 Rejection of SAE + RepE**
*   **Instability Risks:** *Kissane et al. (2025)*  ("Are Sparse Autoencoders Useful?") highlight that SAEs suffer from "dead latents" and training instability, making them unreliable for a deterministic cleaning pipeline.[6]
*   **Concept Erosion:** *Lyu et al. (2024)*  warn that generative erasure (like RepE steering) can cause "concept erosion," degrading the coherence of the text—a critical failure for stylometric analysis which relies on subtle coherence markers.[7]

***

### **2. End-to-End Pipeline Specification (Phase A)**

**Input:** Raw SOBR posts $D = \{x_1, x_2, ...\}$.
**Artifacts Output:** `SOBR-Clean` (Text), `SOBR-Embed-Projected` (Vectors), `Pollution_Log` (JSON).

#### **Step 1: Autonomous Symbolic Detection (GLiNER)**
*   **Objective:** Identify spans of text that constitute explicit demographic signaling.
*   **Mechanism:** Deploy `GLiNER-Large-v2.1` in zero-shot mode.
*   **Prompt Taxonomy (The "Self-Evaluation" Criteria):**
    ```python
    labels = [
        "age_statement",       # "I am 25", "25M"
        "gender_indicator",    # "As a woman", "my husband"
        "nationality_claim",   # "In my country (Germany)"
        "political_self_id"    # "As a liberal"
    ]
    ```
*   **Action:** For every detected span $s$ with confidence $> 0.85$:
    1.  Replace $s$ with a typed mask: `[MASK:GENDER]`, `[MASK:AGE]`.
    2.  Log the span text into `Pollution_Log` for post-hoc analysis (but never pass to Phase D).
*   **Output:** `SOBR-Redacted` (Text with masks).

#### **Step 2: Geometric Concept Erasure (LEACE)**
*   **Objective:** Neutralize the "gender/age direction" in the embedding space of the *remaining* text.
*   **Mechanism:**
    1.  **Encoder:** Use a frozen transformer (e.g., `RoBERTa-base`) to embed `SOBR-Redacted`.
    2.  **Concept Definition:** Use the labels from the `Pollution_Log` as weak supervision.
        *   $X$: Embeddings of posts containing `[MASK:GENDER]`.
        *   $Z$: The redacted gender label (Male/Female).
    3.  **Covariance Calculation:** Compute the covariance matrices $\Sigma_{XZ}$ and $\Sigma_{ZZ}$.
    4.  **LEACE Projection:** Compute the projection matrix $P = I - \Sigma_{XZ}\Sigma_{ZZ}^{-1}\Sigma_{ZX}$.
    5.  **Global Scrubbing:** Apply $P$ to *all* embeddings in the dataset: $x'_{clean} = P \cdot x_{redacted}$.
*   **Output:** $P$ (The Projection Matrix) and $X_{clean}$ (The scrubbed embeddings).

#### **Step 3: Autonomous Self-Evaluation (The "Loop")**
*   **Probe:** Train a lightweight linear classifier (Logistic Regression) on $X_{clean}$ to predict $Z$ (Gender/Age).
*   **Success Criterion:** If Probe Accuracy $\approx$ Majority Class Baseline (50% for balanced binary), the pipeline is successful. If Accuracy $> 55\%$, increase LEACE aggressiveness (or expand GLiNER taxonomy) and repeat.

***

### **3. Evaluation Metrics**

To rigorously quantify success, Phase A must report:

| Metric Category | Metric Name | Source / Definition | Target |
| :--- | :--- | :--- | :--- |
| **Sanitization** | **Amnesic Drop** | Difference in probe accuracy before/after cleaning. (*Elazar & Goldberg, 2018*) | $> 30\%$ drop |
| **Sanitization** | **Explicit Recall** | Percentage of regex-identifiable patterns caught by GLiNER. | $> 95\%$ |
| **Preservation** | **BERTScore (Recall)** | Semantic similarity between $x_{raw}$ and $x_{clean}$. (*Zhang et al., 2020*) | $> 0.85$ |
| **Stylometry** | **Perplexity Delta** | Change in language model perplexity ($PPL(x_{clean}) - PPL(x_{raw})$). | $< 5.0$ |
| **Utility** | **POS-Ngram Stability** | Correlation of POS-tag n-gram frequencies between raw/clean. | $R^2 > 0.90$ |

*Note: The "POS-Ngram Stability" is the critical metric for Phase D. It proves that while we removed "gender," we kept the "syntax."*

***

### **4. Enriching Phase D (The Handover)**

Phase A does not just "clean data"; it provides **structural constraints** for Phase D.

1.  **The Projection Matrix ($P$):** Instead of just giving clean text, Phase A gives Phase D the matrix $P$. Phase D can insert this matrix as a **fixed layer** immediately after its embedding layer.
    *   *Benefit:* This guarantees that the Phase D model *cannot* represent gender linearly, forcing it to learn non-linear stylometric features deep in the network.
2.  **Typed Masks:** The `[MASK:GENDER]` tokens serve as "negative attention anchors." In Phase D, you can penalize the model if it attends heavily to these mask tokens, further enforcing the separation of content/demographics from style.

### **References**

1.  **Elazar, Y., & Goldberg, Y. (2018).** Adversarial Removal of Demographic Attributes from Text Data. *EMNLP*.
2.  **Ravfogel, S., et al. (2020).** Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection. *ACL*.
3.  **Urchade, H., et al. (2023).** GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer. *NeurIPS*.
4.  **Zaratiana, U., et al. (2024).** GLiNER multi-task: Generalist Lightweight Model for Various Information Extraction Tasks. *arXiv*.
5.  **Belrose, N., et al. (2024).** LEACE: Perfect Linear Concept Erasure in Closed Form. *NeurIPS*.
6.  **Kissane, E., et al. (2025).** Are Sparse Autoencoders Useful? A Case Study in Sparse Probing. *ICML/arXiv*.
7.  **Lyu, M., et al. (2024).** One-dimensional Adapter to Rule Them All: Concepts Diffusion Models and Erasing Applications. *CVPR*.
8.  **Pilán, I., et al. (2022).** The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework. *COLING*.