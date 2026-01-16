# Technical Report: Theoretical Foundations for Pollution-Aware Neural Stylometry

**Project Phase:** A $\rightarrow$ D Integration
**Subject:** Literature Review and Architectural Synthesis
**Date:** October 26, 2025

## 1. Executive Summary

This report establishes the theoretical justification for the **Phase A (Pollution Mitigation)** to **Phase D (Neural Stylometry)** pipeline. The central objective is to distinguish between *author profiling based on shortcuts* (e.g., explicit self-identification tokens) and *profiling based on stylometry* (e.g., syntactic structure).

We synthesize ten recent papers to argue that standard training on the SOBR corpus will inevitably result in "Clever Hans" models due to the high **availability** of pollution features. To counteract this, we propose a pipeline that utilizes **Geometric Concept Erasure (LEACE)** to sanitize inputs, ensuring that the **Gradient Descent** dynamics of the Phase D Transformer converge on stylometric causal structures rather than demographic shortcuts. Verification is proposed via **Causal Head Gating (CHG)**.

---

## 2. Extended Literature Review & Theoretical Analysis

The following section analyzes the provided literature, categorizing it into **The Problem** (Shortcut Learning), **The Solution** (Geometric Erasure), and **The Verification** (Causal Interpretability).

### 2.1 The Problem: Shortcut Learning and Data Contamination

**1. *The Clever Hans Mirage* (Ye et al., 2025)**
*   **Core Concept:** Models often rely on "spurious correlations"—non-essential features (backgrounds, specific tokens) that correlate with labels but do not cause them.
*   **Relevance:** In the SOBR corpus, a user writing "I am a mother" is a spurious feature for stylometry. It predicts the label "Female" perfectly but contains zero information about the user's *writing style*.
*   **Key Finding:** Standard Empirical Risk Minimization (ERM) exacerbates this by latching onto the easiest features to minimize loss. Without intervention, a model will always prefer the explicit token over the subtle stylistic pattern.

**2. *On the Foundations of Shortcut Learning* (Hermann et al., 2024)**
*   **Core Concept:** Feature selection in neural networks is a trade-off between **Predictivity** (how reliably a feature indicates a label) and **Availability** (how easily the feature can be extracted/decoded).
*   **Technical Insight:** Deep linear networks are unbiased regarding availability, but **non-linear networks (like Transformers with ReLU/GELU) exhibit a strong bias toward "Available" features**, even if they are less predictive than "Core" features.
*   **Application:** Explicit pollution (e.g., "25M") is highly *Available* (linear readout). Stylometry (e.g., parse tree depth) is low *Availability* (requires complex non-linear extraction). Therefore, the Transformer will naturally ignore style in favor of pollution.

**3. *SOBR: A Corpus for Stylometry...* (Emmery et al., 2024)**
*   **Core Concept:** This paper introduces the dataset but explicitly warns of "contamination."
*   **Key Finding:** Authors note that "in-text self-reports" (e.g., "(25M)") act as dominant features.
*   **Implication:** This validates the need for Phase A. If we do not remove these reports, we are not performing stylometry; we are performing Information Extraction (IE) on demographic labels.

---

### 2.2 The Solution: Geometric Projection and Causal structure

**4. *LEACE: Perfect Linear Concept Erasure...* (Belrose et al., 2024)**
*   **Core Concept:** This paper provides a closed-form algebraic method to remove information from representations.
*   **Technical Specification:** LEACE finds a projection matrix $\mathbf{P}$ that minimizes the movement of embeddings ($\|\mathbf{P}\mathbf{x} - \mathbf{x}\|$) subject to the constraint that the correlation with the concept $\mathbf{z}$ (gender) is zero.
    $$ \mathbf{P} = \mathbf{I} - \mathbf{\Sigma}_{XZ}\mathbf{\Sigma}_{ZZ}^{-1}\mathbf{\Sigma}_{ZX} $$
*   **Application (Phase A):** By applying $\mathbf{P}$ to the input embeddings of the Phase D model, we mathematically guarantee that the input vectors reside in the nullspace of the demographic attribute. This forces the model to look for non-linear, structural features (style).

**5. *How Transformers Learn Causal Structure...* (Nichani et al., 2024)**
*   **Core Concept:** This paper provides a theoretical proof of what Transformers actually learn during Gradient Descent.
*   **Technical Insight:** The gradient of the attention matrix converges to the **Mutual Information** between tokens. The first attention layer effectively learns the adjacency matrix of the latent causal graph generating the data.
*   **Application:** If our data contains the causal path $Pollution \rightarrow Label$, the first attention layer will learn to attend to pollution tokens. By using LEACE/masking to break this correlation, we alter the mutual information landscape, forcing Gradient Descent to discover the path $Style \rightarrow Label$.

**6. *Towards Causal Foundation Model (CInA)* (Zhang et al., 2024)**
*   **Core Concept:** Establish a duality between optimal covariate balancing and self-attention.
*   **Technical Insight:** Self-attention mechanisms can be viewed as solving a dual SVM problem to re-weight inputs for causal inference.
*   **Application:** This supports the idea that the attention mechanism is the correct place to intervene. If we manipulate the input embeddings (via LEACE), the attention mechanism (acting as a causal weight estimator) will be forced to re-distribute weights away from demographic confounders.

---

### 2.3 The Verification: Interpreting the Black Box

**7. *Causal Head Gating (CHG)* (Nam et al., 2025)**
*   **Core Concept:** A method to identify which specific attention heads are responsible for a task.
*   **Technical Specification:** CHG learns a continuous gate parameter $g_{\ell, h} \in [0, 1]$ for each head. It categorizes heads as **Facilitating**, **Interfering**, or **Irrelevant**.
*   **Application (Phase D):** This is our primary verification metric.
    *   *Step 1:* Identify heads that facilitate gender prediction in a "Dirty" model (Pollution Heads).
    *   *Step 2:* Apply Phase A cleaning.
    *   *Step 3:* Re-run CHG. If Phase A worked, the "Pollution Heads" should become "Irrelevant" (gates $\to 0$).

**8. *Investigating Debiasing Methods using Causal Mediation Analysis* (Jeoung & Diesner)**
*   **Core Concept:** Decomposing model decisions into **Direct Effects** (input $\to$ output) and **Indirect Effects** (input $\to$ component $\to$ output).
*   **Application:** We can calculate the *Total Effect* of specific tokens (e.g., "he/she") on the final prediction. In a successfully stylometric model, the Total Effect of pronouns should be negligible compared to the Total Effect of syntactic markers.

**9. *Disentangling Linguistic Features...* (Karwa & Singh)**
*   **Core Concept:** Introducing **Embedding Dimension Importance (EDI)** scores to map specific vector dimensions to linguistic properties (e.g., polarity, tense).
*   **Application:** We can use this to audit the LEACE projection. After projecting out "gender," we can use EDI to verify that dimensions corresponding to "formality" or "complexity" (style) remain intact.

---

## 3. Synthesis: The Phase A $\rightarrow$ Phase D Architecture

Based on the theoretical foundation above, we define the integration of Phase A and Phase D. This is not merely a data-cleaning step, but a **Structural Intervention** on the Transformer's learning dynamics.

### 3.1 The "Filter-then-Project" Pipeline

We assume the input text $X$ contains two distinct signals: $X = X_{style} \oplus X_{poll}$.
*   $X_{poll}$: High Availability, High Predictivity (Shortcuts).
*   $X_{style}$: Low Availability, High Predictivity (Target).

**Step 1: Symbolic Hard-Masking (GLiNER)**
Using the taxonomy from *SOBR*, we deploy GLiNER to detect explicit spans.
*   **Input:** "As a 25M, I think..."
*   **Output:** "As a [MASK], I think..."
*   **Theory:** This removes the highest-availability features described by *Hermann et al.*, preventing the "Clever Hans" effect (*Ye et al.*).

**Step 2: Affine Guarding (LEACE)**
Using the method from *Belrose et al.*, we compute the projection matrix $P$ on the *masked* dataset.
*   **Input:** $Embedding(S_{masked})$
*   **Concept:** $Z$ (Gender/Age)
*   **Output:** $P_{orth} = \text{LEACE}(X, Z)$
*   **Theory:** This removes residual linear leakage. Even if "makeup" is not masked, its vector direction pointing toward "Female" is flattened.

**Step 3: Constrained Attention Training (Phase D)**
We initialize the Phase D Transformer (e.g., RoBERTa). We inject $P_{orth}$ as a permanent, non-trainable layer **before** the first attention block.
$$ \mathbf{h}_0 = P_{orth} \cdot \mathbf{E}(tokens) + \mathbf{Pos} $$
*   **Mechanism:** As per *Nichani et al.*, the attention heads will attempt to learn the causal graph. By projecting the inputs through $P_{orth}$, we ensure that the covariance between any Query/Key pair and the demographic variable $Z$ is zero.
$$ \text{Cov}(Q, K) \not\propto Z $$
*   **Result:** The gradient descent has no choice but to descend the loss landscape using features orthogonal to $Z$ (i.e., $X_{style}$).

### 3.2 Evaluation Strategy

To confirm the pipeline's success in a university project context, we propose a comparative study using **Causal Head Gating**.

1.  **Train Model A (Dirty):** Standard training on raw data.
2.  **Train Model B (Clean):** Training with GLiNER masking + LEACE projection layer.
3.  **Analysis:**
    *   Apply CHG to both models.
    *   Extract the top-10 "Facilitating" heads for Gender Prediction.
    *   **Metric:** Compute the **Part-of-Speech (POS) distribution** of the tokens these heads attend to.
    *   **Hypothesis:**
        *   Model A heads will attend to **Proper Nouns** and **Pronouns** (Shortcuts).
        *   Model B heads will attend to **Determiners, Prepositions, and Punctuation** (Stylometric markers).

## 4. Conclusion

The integration of Phase A and Phase D is grounded in the theory that Neural Networks are lazy learners (*Simplicity Bias/Shortcut Learning*). To force a model to learn the complex, latent distribution of "Authorship Style," we must surgically remove the "Available" shortcuts. By combining **GLiNER** (symbolic removal) and **LEACE** (geometric removal), we create a dataset and architectural constraint that satisfies the conditions required for Gradient Descent to learn the true causal graph of stylometry.

---

# Technical Report: Causal Disentanglement Pipeline Architecture
**Project Phase:** A $\rightarrow$ D Integration & Phase D Finalization
**Date:** October 26, 2025
**Subject:** Theoretical Specification of the Filter-Project-Verify Pipeline

## 1. Executive Summary

This report defines the technical architecture for integrating **Phase A (Pollution Detection)** with **Phase D (Neural Stylometry)**. We propose a "Filter-then-Project" architecture that operationalizes the theoretical insights from recent literature (2024-2025) regarding shortcut learning and concept erasure.

The core innovation is the treatment of pollution not as noise, but as a **dominant causal confounder**. By combining symbolic hard-masking (GLiNER) with geometric subspace projection (LEACE), we mathematically constrain the Phase D Transformer. This constraint forces the model's gradient descent dynamics to bypass demographic shortcuts and converge on stylometric features (syntax, function words), thereby validating the presence of authorship style.

---

## 2. Theoretical Basis for the Pipeline Design

The pipeline is designed to solve the **Availability-Predictivity Trade-off** described by *Hermann et al. (2024)*.

*   **Premise:** Neural networks optimize for the "most available" feature that minimizes loss.
*   **The Problem:** In author profiling, demographic markers (e.g., "I am 25") are highly available (linear readout) and highly predictive. Stylometric features (e.g., recursive syntax depth) are low availability (complex non-linear extraction) but highly predictive.
*   **The Consequence:** Without intervention, models act as "Clever Hans" (*Ye et al., 2025*), learning the demographic markers and ignoring style.
*   **The Solution:** We must artificially reduce the availability of demographic features to zero.

We achieve this via two distinct mechanisms:
1.  **Symbolic Intervention:** Removing explicit tokens (breaking the causal edge $Token \rightarrow Label$).
2.  **Geometric Intervention:** Collapsing the vector subspace associated with the demographic concept (breaking the causal edge $Embedding \rightarrow Label$).

---

## 3. Detailed Pipeline Specification

### 3.1 Phase A $\rightarrow$ Phase D Transition: The "Affine Guard"

Phase A is not just a data cleaning step; it is a **parameter generation step**. It produces two distinct artifacts that define the architecture of Phase D.

#### Artifact 1: The Masked Corpus ($D_{masked}$)
*   **Source Theory:** *Elazar & Goldberg (2018)* prove that if explicit tokens remain, non-linear classifiers can reconstruct attributes even from "debiased" embeddings.
*   **Mechanism:** We employ **GLiNER** (Generalist Linear NER) to detect spans associated with `gender_indicator` and `age_statement`.
*   **Operation:**
    $$ x_{clean} = \text{Replace}(x_{raw}, \text{span} \in \text{GLiNER}(x), \text{"[MASK]"}) $$
    This creates a "hard constraint" on the input space.

#### Artifact 2: The Nullspace Projection Matrix ($\mathbf{P}$)
*   **Source Theory:** *Belrose et al. (2024)* introduce LEACE (Least-Squares Concept Erasure). They demonstrate that even after removing explicit tokens, demographic information remains encoded linearly in the embeddings of neutral words (implicit leakage).
*   **Mechanism:**
    1.  We collect embeddings $\mathbf{X}$ from the masked corpus.
    2.  We collect the protected labels $\mathbf{Z}$ (e.g., Gender).
    3.  We compute the covariance matrices $\mathbf{\Sigma}_{XZ}$, $\mathbf{\Sigma}_{ZZ}$, and $\mathbf{\Sigma}_{XX}$.
    4.  We compute the optimal projection matrix $\mathbf{P}$:
        $$ \mathbf{P} = \mathbf{I} - \mathbf{\Sigma}_{XZ}\mathbf{\Sigma}_{ZZ}^{-1}\mathbf{\Sigma}_{ZX}\mathbf{\Sigma}_{XX}^{-1} $$
*   **Properties of $\mathbf{P}$:**
    *   **Idempotence:** $\mathbf{P}^2 = \mathbf{P}$. It is a true projection.
    *   **Orthogonality:** For any input vector $\mathbf{h}$, the projected vector $\mathbf{h}' = \mathbf{P}\mathbf{h}$ has zero linear correlation with the concept $\mathbf{Z}$.

### 3.2 Phase D: Constrained Transformer Architecture

We do not simply train a Transformer on cleaned text. We modify the Transformer's internal calculation to enforce the constraints generated in Phase A.

**Standard Transformer Block (Simplified):**
$$ \mathbf{h}_{i+1} = \text{Attention}(\mathbf{h}_i) + \text{MLP}(\mathbf{h}_i) $$

**Proposed "Affine Guard" Architecture:**
We insert the matrix $\mathbf{P}$ (calculated in Phase A) as a fixed, non-trainable linear layer **immediately after** the embedding lookup table and **before** the first Transformer encoder block.

$$ \mathbf{h}_0 = \mathbf{Embed}(tokens) + \mathbf{PosEmbed} $$
$$ \mathbf{h}_{projected} = \mathbf{P} \cdot \mathbf{h}_0 $$
$$ \mathbf{h}_1 = \text{EncoderBlock}_1(\mathbf{h}_{projected}) $$

**Theoretical Implication (The "Nichani" Dynamics):**
*Nichani et al. (2024)* prove that during gradient descent, the attention matrix $A$ attempts to learn the adjacency matrix of the causal graph underlying the data.
*   By inserting $\mathbf{P}$, we ensure that the input features $\mathbf{h}_{projected}$ lie in the nullspace of the demographic variable $Z$.
*   Therefore, the **Query ($Q$)** and **Key ($K$)** projections ($Q = W_Q \mathbf{h}_{projected}$) also lie in a subspace orthogonal to $Z$ (assuming linear $W_Q$ initialization).
*   Consequently, the dot product $QK^T$ (which drives attention) *cannot* be maximized by demographic similarity. The gradient $\nabla A$ becomes zero with respect to demographic features.
*   **Result:** The optimization process is forced to find the *next best* predictor. According to our hypothesis, this is **Stylometry** (syntax, function words).

---

## 4. Phase D Finalization: Verification via Causal Head Gating

Training the model is insufficient; we must prove it learned style, not noise. We employ the **Causal Head Gating (CHG)** framework proposed by *Nam et al. (2025)* to audit the trained model.

### 4.1 The Verification Logic
Standard attention analysis (looking at "high attention weights") is flawed because attention is not explanation. CHG provides a causal metric: it learns a gate parameter to determine if a head actually *causes* a drop in loss.

### 4.2 The Audit Protocol

**Step 1: Train Two Models**
1.  **Model A (Reference):** Trained on Raw SOBR data (Contains Shortcuts).
2.  **Model B (Intervention):** Trained with the Affine Guard ($\mathbf{P}$) and Masked Data.

**Step 2: Learn Causal Gates ($g$)**
For both models, we freeze the weights and learn a gate parameter $g_{l,h} \in [0, 1]$ for every head $(l,h)$ using the validation set.
*   Objective: Minimize Loss w.r.t. $g$.
*   Interpretation: High $g$ = Head is **Facilitating** (Crucial). Low $g$ = Head is **Irrelevant**.

**Step 3: Comparative Topology Analysis**
We extract the set of "Facilitating Heads" for Model A ($H_A$) and Model B ($H_B$). We analyze what tokens these heads attend to.

*   **Hypothesis 1 (Shortcuts):** Heads in $H_A$ will show high attention mass on **Content Words** (Nouns, Proper Nouns) and **Explicit Tokens** (if any remain). This confirms Model A is a "Clever Hans."
*   **Hypothesis 2 (Stylometry):** Heads in $H_B$ will show high attention mass on **Function Words** (Determiners, Prepositions, Auxiliaries) and **Punctuation**.

### 4.3 The Stylometric Validity Metric (SVS)
We define a quantitative metric for the university project report:
$$ SVS = \frac{\sum_{h \in H_{fac}} \text{AttnMass}(h, \text{FunctionWords})}{\sum_{h \in H_{fac}} \text{AttnMass}(h, \text{ContentWords})} $$

*   **Target:** We expect $SVS(\text{Model B}) \gg SVS(\text{Model A})$.
*   If $SVS(\text{Model B})$ is high, we have empirically proven that by removing demographic availability (Phase A), we forced the model to rely on stylometric features (Phase D).

---

## 5. Conclusion

This pipeline represents a rigorous application of current AI safety and interpretability theory.
1.  We acknowledge the **Shortcut Learning** problem (*Hermann et al.*).
2.  We solve it via **Geometric Erasure** (*Belrose et al.*) and **Symbolic Masking** (*SOBR*).
3.  We enforce this solution architecturally via the **Affine Guard** injection.
4.  We verify the result using **Causal Head Gating** (*Nam et al.*), moving beyond correlation to causal attribution of the model's behavior.

This satisfies the project requirements: it is technically grounded, architecturally precise, and offers a clear metric for success.

---

### **References**

[1] N. Belrose, D. Schneider-Joseph, S. Ravfogel, R. Cotterell, E. Raff, and S. Biderman, "LEACE: Perfect Linear Concept Erasure in Closed Form," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2024.

[2] Y. Elazar and Y. Goldberg, "Adversarial Removal of Demographic Attributes from Text Data," in *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, Brussels, Belgium, 2018, pp. 11–21.

[3] C. Emmery, M. Miotto, S. Kramp, and B. Kleinberg, "SOBR: A Corpus for Stylometry, Obfuscation, and Bias on Reddit," in *Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING)*, Torino, Italy, 2024, pp. 14967–14983.

[4] K. L. Hermann, H. Mobahi, T. Fel, and M. C. Mozer, "On the Foundations of Shortcut Learning," in *International Conference on Learning Representations (ICLR)*, 2024.

[5] S. Jeoung and J. Diesner, "What Changed? Investigating Debiasing Methods using Causal Mediation Analysis," *University of Illinois-Urbana Champaign*.

[6] S. Karwa and N. Singh, "Disentangling Linguistic Features with Dimension-Wise Analysis of Vector Embeddings," in *Proceedings of the 5th Workshop on Trustworthy NLP (TrustNLP 2025)*, 2025, pp. 461–488.

[7] A. J. Nam, H. C. Conklin, Y. Yang, T. L. Griffiths, J. D. Cohen, and S.-J. Leslie, "Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2025.

[8] E. Nichani, A. Damian, and J. D. Lee, "How Transformers Learn Causal Structure with Gradient Descent," *arXiv preprint arXiv:2402.14735*, 2024.

[9] J. Vig, S. Gehrmann, Y. Belinkov, S. Qian, D. Nevo, Y. Singer, and S. Shieber, "Investigating Gender Bias in Language Models Using Causal Mediation Analysis," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, 2020, pp. 12388–12401.

[10] W. Ye, L. Jiang, E. Xie, et al., "The Clever Hans Mirage: A Comprehensive Survey on Spurious Correlations in Machine Learning," *arXiv preprint arXiv:2402.12715*, 2025.

[11] J. Zhang, J. Jennings, A. Hilmkil, N. Pawlowski, C. Zhang, and C. Ma, "Towards Causal Foundation Model: on Duality between Causal Inference and Attention," *arXiv preprint arXiv:2310.00809*, 2024.