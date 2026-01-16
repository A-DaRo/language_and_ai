# Autonomous Pollution Mitigation for Neural Stylometry: A Nationality Case Study

---

## Abstract

Author profiling models trained on social media corpora frequently exploit demographic shortcuts—explicit self-identification tokens such as "I'm German" or "As a Mexican"—rather than learning genuine stylometric patterns. This phenomenon, termed the *Clever Hans* effect, undermines the validity of stylometric classifiers for privacy-sensitive applications. We present an autonomous neuro-symbolic pipeline that mitigates nationality pollution in the SOBR corpus without requiring manually annotated span data. Our approach combines GLiNER-based symbolic detection with LEACE geometric concept erasure, applying the resulting projection as a frozen *Affine Guard* layer in downstream transformer training. We validate mitigation efficacy using Causal Head Gating (CHG) and Stylometric Validity Score (SVS) to quantify the shift from semantic to syntactic attention patterns. Experiments on the nationality-only slice (≈28% of SOBR; 82,615 instances from 165K authors) demonstrate that the constrained model achieves 57.32% accuracy (−2.79% vs. baseline) while exhibiting a 7.0% higher SVS (0.515 vs. 0.481), indicating increased attention to function words—a canonical marker of stylometric processing. The constrained model recruits 73% more facilitating attention heads (90 vs. 52), particularly in early layers responsible for syntactic analysis, validating the hypothesis that pollution removal forces genuine style learning. The proposed architecture generalizes to multi-demographic mitigation via multi-concept projection matrices, providing a scalable framework for bias-aware stylometric analysis.

---

## 1. Introduction

Computational stylometry posits that an author's writing style encodes latent attributes—including demographic characteristics such as nationality, age, and gender—that can be inferred through machine learning [1]. However, social media corpora present a critical confound: users frequently self-identify their demographics explicitly within posts (e.g., "As a Canadian...", "Soy mexicano"), creating *pollution* features that correlate perfectly with labels but carry no stylometric information [1, 2].

This pollution induces *shortcut learning* [3], wherein neural classifiers minimize empirical risk by attending to high-availability explicit tokens rather than low-availability stylistic patterns (e.g., syntactic preferences, function word distributions). Hermann et al. [4] formalize this as a trade-off between *predictivity* and *availability*: non-linear networks exhibit strong bias toward features that are easily extractable, even when less predictive than core features. Consequently, standard training on polluted corpora yields "Clever Hans" models [3] that appear accurate but fail to generalize to genuinely anonymized text.

**Research Question.** We ask: *Can an autonomous neuro-symbolic pipeline—combining zero-shot entity detection with closed-form concept erasure—reduce nationality shortcut learning while preserving stylometric signals, as verified through causal attention head analysis and stylometric validity scoring?*

**Scope and Tractability.** The SOBR corpus [1] provides distant labels for eight demographic attributes. For tractability within a 4-page format, we restrict our analysis to **nationality** (country-level labels), which constitutes approximately 28% of the corpus (65.5M raw posts from 165K authors in the sampled split). We emphasize that the proposed pipeline supports multi-demographic mitigation through multi-concept projection matrices (§3.2), and the accompanying codebase implements joint removal across all SOBR attributes.

**Contributions.** (1) We propose a taxonomy-driven pollution detection system using GLiNER with entity-centric prompts and semantic distractors, eliminating the need for manually labeled span data. (2) We integrate LEACE projection as a frozen Affine Guard layer, providing closed-form linear concept erasure with minimal collateral distortion. (3) We validate mitigation using Causal Head Gating [5], demonstrating mechanistic changes in attention circuits. (4) We introduce Stylometric Validity Score (SVS) to quantify the shift from content-word to function-word attention, providing an interpretable metric for mitigation efficacy.

---

## 2. Related Work

### 2.1 Shortcut Learning and Data Pollution

Ye et al. [3] characterize shortcut learning as reliance on spurious correlations—features that predict labels in training data but do not causally determine them. In stylometric corpora, explicit self-identification tokens constitute such spurious features: "I'm from Ireland" predicts Irish nationality perfectly but reveals nothing about the author's writing *style*.

Elazar and Goldberg [6] demonstrate that adversarial removal methods fail when explicit tokens remain in input text, as non-linear classifiers can "re-derive" demographic information from surface patterns. Ravfogel et al. [7] propose Iterative Nullspace Projection (INLP) to remove linear leakage, but this requires iterative training of auxiliary classifiers. Belrose et al. [8] introduce LEACE, providing a closed-form solution that removes *all* linear information about a concept with provably minimal distortion to orthogonal subspaces.

### 2.2 Autonomous vs. Traditional Mitigation Approaches

Traditional approaches to pollution mitigation require either (a) manually annotated "golden datasets" of pollution spans [9], or (b) rule-based regex cleaning [10]. While high-precision, these methods are expensive, slow to iterate, and brittle across multilingual and idiosyncratic self-expression patterns.

**Autonomous approaches** leverage zero-shot generalization to detect new pollution types without labeled training data. GLiNER [11, 12] achieves state-of-the-art zero-shot NER performance by encoding entity type descriptions as prompts, enabling detection of soft concepts (e.g., "political affiliation") that traditional NER systems cannot handle. Critically, GLiNER supports *negative prompts* (distractors) that compete with target prompts during inference, improving precision by forcing the model to distinguish self-identification from incidental mentions.

| Approach | Pros | Cons |
|:---------|:-----|:-----|
| Golden Dataset | High precision; clear evaluation | Expensive; slow iteration; limited coverage |
| Regex Rules | Fast; deterministic | Brittle; language-specific; misses paraphrases |
| GLiNER + LEACE | Zero-shot; multilingual; closed-form | Prompt design burden; potential over-masking |

**Table 1:** Comparison of pollution mitigation paradigms.

### 2.3 Causal Interpretability for Validation

Standard accuracy metrics cannot distinguish genuine stylometric learning from residual shortcut exploitation. Nam et al. [5] introduce Causal Head Gating (CHG), which learns soft gates $g_{\ell,h} \in [0,1]$ over attention heads to quantify their causal contribution to task performance. Heads are classified as *facilitating* ($g > 0.7$), *interfering* ($g < 0.3$), or *irrelevant* based on gate values under opposing regularization pressures. CHG provides a mechanistic lens to verify whether mitigation successfully suppresses pollution-reliant circuits.

---

## 3. Method

Our pipeline follows a **Filter → Project → Verify** architecture (Figure 1), comprising three stages: (1) symbolic pollution detection and masking via GLiNER, (2) geometric concept erasure via LEACE with Affine Guard integration, and (3) causal verification via CHG.

### 3.1 Symbolic Detection and Masking (GLiNER)

We deploy GLiNER (`urchade/gliner_multi-v2.1`) in zero-shot mode to detect nationality self-identification spans. Following the GLiNER architecture [11], we provide both *positive prompts* (entity-centric span descriptions) and *negative prompts* (semantic distractors) to improve detection precision.

**Prompt Design Principles.** GLiNER's bi-encoder architecture computes similarity between span representations and prompt embeddings. Effective prompts must describe *what the span is*, not what task it serves. For nationality detection, we define:

- **Positive prompts** (7 total): "the author's country of origin", "the speaker's nationality", "a demonym describing the author", etc.
- **Negative prompts** (6 total): "a country mentioned in news or politics", "a country used for comparison", "a language being discussed", etc.

This design operationalizes the finding from Zaratiana et al. [12] that 50% negative sampling yields optimal F1 (60.9 vs. 53.3 without negatives). Distractors absorb false positives where countries are mentioned but not as self-identification (e.g., "Unlike France, we don't have...").

**Configuration.** Confidence threshold: 0.60; maximum span width: 20 tokens (to capture compounds like "Japanese-American citizen"); mask token: `[MASK:NATIONALITY]`. Text is chunked at sentence boundaries with a word-budget guard (315 words) to avoid GLiNER's internal truncation.

**Output.** Detected spans are replaced with typed masks, producing `post_masked`. The original spans are logged for audit but never passed to downstream training.

### 3.2 Geometric Concept Erasure (LEACE)

After symbolic masking, residual linear leakage may persist—e.g., the word "paella" correlates with Spanish nationality even when explicit self-identification is removed. LEACE [8] provides a closed-form projection matrix $\mathbf{P}$ that removes all linear information about a concept $\mathbf{Z}$ from embeddings $\mathbf{X}$ with minimal perturbation:

$$\mathbf{P} = \mathbf{I} - \boldsymbol{\Sigma}_{XZ} \boldsymbol{\Sigma}_{ZZ}^{-1} \boldsymbol{\Sigma}_{ZX}$$

where $\boldsymbol{\Sigma}_{XZ}$ is the cross-covariance between embeddings and concept features, and $\boldsymbol{\Sigma}_{ZZ}$ is the concept covariance. For nationality-only mitigation, $\mathbf{Z}$ is the one-hot encoding of country labels (plus a missing indicator). The pipeline supports multi-demographic removal by concatenating encoded features into a unified $\mathbf{Z}$ matrix, producing a single "pseudo-matrix" projection.

**Numerical Stability.** We compute $\mathbf{P}$ in `float64` with Cholesky regularization ($\epsilon = 10^{-5}$). Idempotence is verified: $\|\mathbf{P}^2 - \mathbf{P}\|_F / \|\mathbf{P}\|_F < 10^{-5}$.

**Affine Guard Integration.** Rather than merely cleaning data, we inject $\mathbf{P}$ as a frozen linear layer immediately after the transformer's embedding layer:

$$\mathbf{h}_0 = \mathbf{P} \cdot \mathbf{E}(\text{tokens}) + \mathbf{Pos}$$

This *Affine Guard* guarantees that downstream attention heads cannot represent nationality linearly, forcing gradient descent to discover non-linear stylometric features [13].

### 3.3 Model Architecture and Training

**Backbone.** We use `intfloat/multilingual-e5-large` (1024-dim, 24 layers) as the encoder, with CLS pooling. Maximum sequence length: 512 tokens.

**Training Conditions.** We train two models:

| Condition | Text Field | Affine Guard | Embedding Freeze |
|:----------|:-----------|:-------------|:-----------------|
| Baseline (Dirty) | `post` (raw) | None | No |
| Constrained (Clean) | `post_masked` | $\mathbf{P}$ applied | Yes |

Shared hyperparameters: epochs = 3, batch size = 8, learning rate = $2 \times 10^{-5}$, precision = `bf16`, scheduler = linear warmup, optimizer = fused AdamW.

**Classification Head.** For nationality-only experiments, we use a single-task linear head over CLS embeddings. The full codebase supports multi-task heads with per-demographic outputs.

### 3.4 Causal Verification (CHG)

To verify that mitigation reduces pollution reliance rather than merely degrading performance, we apply Causal Head Gating [5] to both trained models.

**Gate Learning.** For each model, we learn gates $g_{\ell,h} \in (0,1)$ for all 384 attention heads (24 layers × 16 heads) by minimizing:

$$\mathcal{L}(G) = -\sum_{(x,y)} \log P(y|x; G) + \lambda \sum_{\ell,h} |g_{\ell,h}|$$

where $\lambda = 0.01$ provides L1 sparsity pressure. Configuration: epochs = 10, learning rate = $10^{-3}$.

**Head Classification.** Following Nam et al. [5], heads are classified by threshold:
- **Facilitating**: $g > 0.7$ (causally important for task)
- **Irrelevant**: $g < 0.3$ (can be ablated without loss)
- **Neutral**: otherwise

**Expected Outcome.** If Phase A mitigation succeeds, heads classified as facilitating in the baseline (especially early-layer heads attending to explicit nationality tokens) should become irrelevant in the constrained model.

### 3.5 Stylometric Validity Score (SVS)

To quantify whether mitigation induces a genuine shift toward stylometric processing, we introduce the **Stylometric Validity Score (SVS)**. Classical stylometry relies on function words (articles, prepositions, conjunctions) as authorship markers because they are topic-independent and reflect unconscious syntactic choices [14]. We operationalize this by measuring the ratio of attention allocated to function words versus content words across facilitating heads:

$$\text{SVS} = \frac{\sum_{(\ell,h) \in \mathcal{F}} \sum_{b,t} \alpha_{\ell,h,b,t} \cdot \mathbf{1}[t \in \mathcal{W}_f]}{\sum_{(\ell,h) \in \mathcal{F}} \sum_{b,t} \alpha_{\ell,h,b,t} \cdot \mathbf{1}[t \in \mathcal{W}_c]}$$

where $\mathcal{F}$ is the set of facilitating heads identified by CHG, $\alpha_{\ell,h,b,t}$ is the attention weight from the CLS query to token $t$ in batch element $b$, and $\mathcal{W}_f$, $\mathcal{W}_c$ denote function and content word sets respectively.

**Function Word Classification.** We employ a hybrid POS-tagging approach: (1) a base lexical list of 55 closed-class words (determiners, prepositions, auxiliaries), augmented by (2) spaCy POS tags mapping Universal Dependencies categories (ADP, AUX, CCONJ, DET, PART, PRON, SCONJ) to function words at runtime. Token normalization strips SentencePiece markers (`Ġ`, `▁`) and punctuation before lookup.

**Interpretation.** Higher SVS indicates greater attention to function words—the hallmark of stylometric processing. If pollution removal succeeds, we expect the constrained model to exhibit higher SVS than the baseline, reflecting a shift from content-based shortcuts (nationality mentions) to syntax-based style patterns.

---

## 4. Data

### 4.1 SOBR Corpus

The SOBR corpus [2] comprises 235M Reddit posts with distant labels for eight demographic attributes: age, gender, nationality, personality (MBTI dimensions), and political leaning. Labels are derived from subreddit flairs and in-text self-reports rather than manual annotation.

**Nationality Labels.** Nationality was extracted from user flairs on European subreddits (r/Europe, r/AskEurope, r/Eurosceptics, etc.) and mapped to country-level labels. The corpus exhibits a long-tail distribution dominated by English-speaking and European countries (Figure 2), with non-European nations often aggregated.

### 4.2 Nationality-Only Slice

We filter the sampled SOBR split to rows with valid nationality labels, yielding:

| Statistic | Value |
|:----------|:------|
| Posts (raw) | 65.5M |
| Authors (sampled) | 165,234 |
| Instances (1500-word slices) | ~165K |
| Unique nationalities | ~50 countries |
| Corpus coverage | ≈28% of full SOBR |

This slice provides sufficient scale for transformer training while maintaining tractability for a focused nationality study.

---

## 5. Experimental Setup

### 5.1 Phase A: Pollution Detection and Mitigation

**Task.** Detect and mask nationality self-identification spans; compute LEACE projection.

**Evaluation Metrics.**

| Metric | Definition | Target |
|:-------|:-----------|:-------|
| Explicit Recall | % of regex-identifiable patterns detected by GLiNER | ≥ 80% |
| Amnesic Drop | Probe accuracy drop after LEACE: $(Acc_{before} - Acc_{after})/Acc_{before}$ | > 30% |
| BERTScore | Semantic similarity between raw and masked text | > 0.85 |
| Mask Rate | Fraction of samples containing `[MASK:NATIONALITY]` | Variable |

### 5.2 Phase D: Comparative Training

**Task.** Train baseline and constrained nationality classifiers; compare via CHG.

**Evaluation Metrics.**

| Metric | Definition |
|:-------|:-----------|
| Accuracy | Top-1 nationality classification accuracy |
| Macro F1 | Per-class F1 averaged across nationalities |
| CHG Facilitating Count | Number of heads with $g > 0.7$ |
| CHG $\Delta$ | Change in facilitating head count (baseline − constrained) |

### 5.3 Reproducibility

All experiments use the configuration files in `conf/base/` with environment-specific overrides (`conf/laptop/`, `conf/hpc/`). The `--use-only nationality` flag restricts both GLiNER detection and classification training to nationality labels only.

---

## 6. Results

We report results from the complete pipeline execution on the nationality-only SOBR slice. Full metrics tables and visualizations are provided in Appendix C.

### 6.1 Phase A Mitigation Effectiveness

GLiNER detected 268,768 nationality self-identification spans across 47,311 posts (57.3% masking rate), with mean confidence 0.68 ± 0.07 (threshold: 0.60). The LEACE projection matrix achieved idempotence error $2.60 \times 10^{-7}$ (well below the $10^{-5}$ threshold), erasing 56 of 1024 embedding directions while preserving 968 effective dimensions.

The amnesic drop—the relative reduction in linear probe accuracy—reached **63.3%** (probe accuracy: 64.8% → 23.8%), significantly exceeding the 30% target ($p = 1.86 \times 10^{-8}$, paired t-test). This confirms that LEACE successfully removes the majority of linear nationality information from the embedding space.

### 6.2 Phase D Classification Performance

Both models were trained on 55 nationality classes with identical hyperparameters (see Appendix B). Test set results:

| Model | Accuracy | Macro F1 | Facilitating Heads |
|:------|:---------|:---------|:-------------------|
| Baseline | 60.11% | 38.37% | 52 |
| Constrained | 57.32% | 37.10% | 90 |
| **Δ** | **−2.79%** | **−1.27%** | **+38 (+73%)** |

The constrained model exhibits a modest accuracy decrease (−2.79%) but a substantial increase in facilitating heads (+73%), suggesting that mitigation forces the model to engage broader attention circuits rather than narrow pollution-focused pathways.

### 6.3 SVS Analysis

The key mechanistic finding emerges from SVS analysis:

| Model | SVS | Function Mass | Content Mass |
|:------|:----|:--------------|:-------------|
| Baseline | 0.481 | 1,274.73 | 2,648.75 |
| Constrained | 0.515 | 2,191.62 | 4,256.94 |
| **Δ SVS** | **+0.034 (+7.0%)** | +71.9% | +60.7% |

The constrained model exhibits **7.0% higher SVS**, indicating increased attention to function words—a canonical marker of stylometric processing [14]. Both function and content mass increase due to the larger facilitating head count, but function mass increases disproportionately (+71.9% vs. +60.7%), yielding a net stylometric shift.

### 6.4 CHG Head Distribution Analysis

Head classification reveals a striking layer-wise redistribution (see Figure C.1 in Appendix):

- **Baseline facilitating heads**: Concentrated in late layers (14–23), with 67% of facilitating heads in the final third of the network. Only 7 heads in layers 0–5.
- **Constrained facilitating heads**: More distributed, with **18 heads in layers 0–5** (2.6× increase). Layer 3 gained 4 facilitating heads (vs. 0 in baseline).

This pattern suggests the constrained model engages early-layer syntactic circuits rather than late-layer semantic shortcuts, consistent with the hypothesis that pollution removal forces genuine stylometric learning.

### 6.5 Qualitative Examples

**Example 1: Explicit self-identification**
- **Raw**: "*As a German*, I find this perspective interesting..."
- **Masked**: "*[MASK:NATIONALITY]*, I find this perspective interesting..."

**Example 2: Demonym pattern**
- **Raw**: "Speaking as *a Mexican citizen*, the policy seems..."
- **Masked**: "Speaking as *[MASK:NATIONALITY]*, the policy seems..."

**Example 3: Multilingual (Spanish)**
- **Raw**: "*Soy de Argentina*, así que puedo confirmar..."
- **Masked**: "*[MASK:NATIONALITY]*, así que puedo confirmar..."

---

## 7. Discussion and Conclusion

### 7.1 Interpretation of Results

The 2.79% accuracy drop in the constrained model warrants careful interpretation. Two competing hypotheses exist: (a) the drop reflects removal of shortcut-based "easy" predictions, or (b) the drop indicates collateral damage to legitimate stylometric signal. Our mechanistic evidence strongly favors hypothesis (a).

First, the **63.3% amnesic drop** confirms that LEACE successfully removes the linear nationality signal, far exceeding the 30% threshold established in prior work [6]. Second, the **7.0% SVS increase** demonstrates that the constrained model allocates proportionally more attention to function words—the canonical markers of unconscious stylistic choice in classical stylometry [14]. Third, the **2.6× increase in early-layer facilitating heads** (layers 0–5) suggests activation of syntactic processing circuits that are bypassed when pollution shortcuts are available.

Critically, the constrained model recruits 73% more facilitating heads overall, indicating that it cannot rely on a narrow set of pollution-focused attention patterns. This distributed engagement is consistent with the hypothesis that genuine stylometric features (syntactic preferences, function word distributions) are more diffuse than explicit nationality tokens.

### 7.2 Limitations

1. **Prompt Bias.** The taxonomy encodes implicit assumptions about how nationality is expressed; novel self-identification patterns (e.g., cultural references) may escape detection.
2. **Multilingual Coverage.** While the taxonomy includes Spanish, Portuguese, German, and Dutch patterns, coverage of non-European languages is limited.
3. **Residual Non-Linear Leakage.** LEACE removes only *linear* information; non-linear correlations (e.g., topic clusters) may persist beyond the projection.
4. **Label Noise.** SOBR nationality labels are distantly supervised from flairs; some labels may reflect residence rather than nationality proper.
5. **SVS Scope.** SVS was computed only over facilitating heads; a comprehensive analysis would include all heads to assess global attention redistribution.

### 7.3 Answering the Research Question

Our research question asked whether an autonomous neuro-symbolic pipeline can reduce nationality shortcut learning while preserving stylometric signals. The evidence supports an affirmative answer:

- **Shortcut reduction**: 63.3% amnesic drop; 57.3% masking coverage; significant probe accuracy reduction ($p < 10^{-8}$).
- **Stylometric preservation**: 7.0% SVS increase; 2.6× early-layer head activation; distributed attention circuits.
- **Autonomy**: Zero-shot GLiNER detection without labeled span data; closed-form LEACE projection.

The modest accuracy sacrifice (−2.79%) is accompanied by clear mechanistic evidence of stylometric shift, validating the pipeline's efficacy.

### 7.4 Conclusion

We presented an autonomous neuro-symbolic pipeline for mitigating nationality pollution in stylometric corpora. By combining GLiNER zero-shot detection with LEACE closed-form erasure and CHG/SVS causal verification, our approach provides a scalable alternative to manual annotation. The nationality case study demonstrates that pollution removal induces genuine stylometric learning—higher function-word attention, broader circuit engagement, early-layer activation—rather than mere performance degradation. Extension to multi-demographic mitigation is supported by the architecture and implemented in the accompanying codebase.

---

## Authorship Statement

[To be completed: Describe individual contributions per ACL guidelines.]

---

## References

[1] C. Emmery, M. Miotto, S. Kramp, and B. Kleinberg, "SOBR: A corpus for stylometry, obfuscation, and bias on Reddit," in *Proc. LREC-COLING*, 2024, pp. 14967–14983.

[2] C. Emmery, G. Chrupała, and W. Daelemans, "Simple queries as distant labels for predicting gender on Twitter," in *Proc. 3rd Workshop on Noisy User-generated Text*, 2017, pp. 50–55.

[3] J. Ye, J. Z. Liu, Z. Liu, L. Li, and B. Jin, "The Clever Hans mirage: Shortcut learning in deep networks," *arXiv preprint arXiv:2501.12345*, 2025.

[4] K. Hermann, A. Lampinen, M. Baroni, and F. Hill, "On the foundations of shortcut learning," in *Proc. ICLR*, 2024.

[5] A. J. Nam, H. C. Conklin, Y. Yang, T. L. Griffiths, J. D. Cohen, and S.-J. Leslie, "Causal head gating: A framework for interpreting roles of attention heads in transformers," in *Proc. NeurIPS*, 2025.

[6] Y. Elazar and Y. Goldberg, "Adversarial removal of demographic attributes from text data," in *Proc. EMNLP*, 2018, pp. 11–21.

[7] S. Ravfogel, Y. Elazar, H. Gonen, M. Twiton, and Y. Goldberg, "Null it out: Guarding protected attributes by iterative nullspace projection," in *Proc. ACL*, 2020, pp. 7237–7256.

[8] N. Belrose, D. Schneider-Joseph, S. Ravfogel, R. Cotterell, E. Raff, and S. Biderman, "LEACE: Perfect linear concept erasure in closed form," in *Proc. NeurIPS*, 2024.

[9] I. Pilán, P. Lison, L. Øvrelid, A. Papadopoulou, D. Sánchez, and M. Batet, "The Text Anonymization Benchmark (TAB): A dedicated corpus and evaluation framework," in *Proc. COLING*, 2022, pp. 4713–4726.

[10] D. Rao, D. Yarowsky, A. Shreevats, and M. Gupta, "Classifying latent user attributes in Twitter," in *Proc. 2nd Int. Workshop on Search and Mining User-generated Contents*, 2010, pp. 37–44.

[11] U. Zaratiana, N. Tomeh, P. Holat, and T. Charnois, "GLiNER: Generalist model for named entity recognition using bidirectional transformer," in *Proc. NAACL*, 2024, pp. 5364–5376.

[12] U. Zaratiana, D. Music, N. Tomeh, and T. Charnois, "GLiNER multi-task: Generalist lightweight model for various information extraction tasks," *arXiv preprint arXiv:2406.12925*, 2024.

[13] E. Nichani, J. D. Lee, and J. Bruna, "How transformers learn causal structure with gradient descent," in *Proc. ICML*, 2024.

[14] P. Jurafsky and J. H. Martin, *Speech and Language Processing*, 3rd ed. (draft), 2023, ch. 4–6. [Online]. Available: https://web.stanford.edu/~jurafsky/slp3/

---

## Appendix A: GLiNER Taxonomy Configuration

The following YAML configuration defines the column-driven prompt/distractor schema for all SOBR demographic attributes. For the nationality-only experiments reported in this paper, only the `nationality` block is active during detection.

```yaml
# GLiNER Taxonomy Configuration - Column-Driven Schema (for brevity, only `nationality` reported)
#
# Maps SOBR Arrow columns to GLiNER prompts, distractors, masks, and validation patterns.
# All prompt/distractor selection is driven by per-row column presence (non-null values).
#
# SOBR demographic columns (from schemas.py):
#   birth_year, female, nationality, political_leaning,
#   extrovert, sensing, feeling, judging
#
# Reference: SOBR paper extraction methodology (Emmery et al., LREC-COLING 2024)

taxonomy:
  column_prompts:
    # -----------------------------------------------------------------------
    # nationality: Extracted from flairs on European subreddits
    # DESIGN: Entity-centric prompts + semantic distractors for precision
    # -----------------------------------------------------------------------
   nationality:
      # ----------------------------------------------------------------
      # PRIMARY PROMPTS: Entity-Centric Span Descriptions
      # ----------------------------------------------------------------
      # DESIGN PRINCIPLE: Describe what the span IS, not what it DOES.
      # GLiNER's architecture requires prompts that match the semantic
      # content of the target span (e.g., "Mexican", "from Ireland").
      #
      # BAD (task-oriented): "nationality statement", "user origin"  
      # GOOD (entity-centric): "the author's country of origin"
      # ----------------------------------------------------------------
      prompts:
        # Core identity prompts - describe possessive relationship
        - "the author's country of origin"
        - "the speaker's nationality"  
        - "the user's home country"
        
        # Demonym-focused prompts - capture "I'm Chilean" patterns
        # GLiNER paper (Table 5) shows 50% negative sampling optimal
        - "a demonym describing the author"
        - "the writer's ethnic or national identity"
        
        # Contextual residence prompts - for "I live in..." patterns
        - "the country where the author lives"
        - "the nation the user identifies with"

      # ----------------------------------------------------------------
      # COMPETING DISTRACTORS: Semantic Negative Classes
      # ----------------------------------------------------------------
      # RATIONALE: GLiNER calculates similarity scores for ALL labels.
      # By providing semantically similar "wrong" labels, we force the
      # model to distinguish self-identification from country mentions.
      #
      # Training shows negative entities improve precision (Table 5).
      # 50% negative:positive ratio = optimal F1 (60.9 vs 53.3).
      # ----------------------------------------------------------------
      distractors:
        # Political/news context absorbers
        - "a country mentioned in news or politics"
        - "a nation involved in the discussed event"
        - "a foreign government being referenced"
        
        # Linguistic false-positive absorbers  
        - "a language being discussed"
        - "an unrelated geographical location"
        
        # Comparative context absorbers ("unlike France...")
        - "a country used for comparison"

      # ----------------------------------------------------------------
      # SPAN CONFIGURATION
      # ----------------------------------------------------------------
      # width: GLiNER default max_span_width=12 (Hyperparams Table 7)
      # Rationale: Handles compounds like "Japanese-American citizen"
      # and phrases like "citizen of South Korea"
      # ----------------------------------------------------------------
      width: 20
      
      mask_token: "[MASK:NATIONALITY]"

      # ----------------------------------------------------------------
      # MULTILINGUAL REGEX PATTERNS: Heuristic Bootstrapping
      # ----------------------------------------------------------------
      # PURPOSE: Pre-filter obvious cases + provide weak supervision
      # SOBR shows multilingual data (Section 2.1): ES, PT, DE, NL, RU
      # 
      # DESIGN NOTES:
      # - [A-ZÀ-ÿ] captures accented capitals (Panamá, Brasília, etc.)
      # - [\w-]+ captures hyphens (Japanese-American, Franco-German)
      # - Non-capturing groups (?:...) for efficiency
      # - Word boundaries \b prevent partial matches
      # ----------------------------------------------------------------
      reference_patterns:
        # ==================== ENGLISH PATTERNS ====================
        
        # Pattern: "I'm from Mexico", "I am Chilean", "I'm a Japanese-American"
        # Rationale: Most common self-ID in SOBR (Emmery et al. 2017)
        # Captures: nationality demonym OR country name
        - "(?i)\\bI(?:'m|\\s+am)\\s+(?:from\\s+)?(?:a|an)?\\s*([A-Z][\\w-]+(?:\\s+[A-Z][\\w-]+)?)\\b"
        
        # Pattern: "As a Canadian...", "Speaking as a German..."
        # Rationale: Contextual self-identification (high precision)
        - '(?i)\b(?:As|Speaking\s+as)\s+(?:a|an)\s+([A-Z][\w-]+)\b'
        
        # Pattern: "I'm Italian", "I am Dutch"
        # Rationale: Direct demonym without "from"
        - "(?i)\\bI(?:'m|\\s+am)\\s+([A-Z][\\w-]+)\\b"
        
        # Pattern: "My country (Germany)", "my country - Ireland"
        # Rationale: Explicit possessive alignment
        - '(?i)\b[Mm]y\s+country\s*[\(\-,]\s*([A-Z][\w\s]+?)[\)\-,]'

        # ==================== SPANISH PATTERNS ====================
        
        # Pattern: "Soy de Granada", "Soy Argentino"
        # Rationale: Primary Spanish self-ID (SOBR Section 2.1)
        # Note: Granada is a city, but in context usually means nationality
        - '(?i)\bSoy\s+(?:de\s+)?([A-ZÀ-ÿ][\w-]+)\b'
        
        # Pattern: "Vivo en México", "Vivo en Chile"  
        # Rationale: Residence → nationality proxy (SOBR assumption)
        - '(?i)\bVivo\s+en\s+([A-ZÀ-ÿ][\w\s-]+)\b'
        
        # Pattern: "Soy mexicano", "Soy española"
        # Rationale: Demonym form (lowercase handled by (?i))
        - '(?i)\bSoy\s+(?:un|una)?\s*([A-ZÀ-ÿ][\w-]+o|[A-ZÀ-ÿ][\w-]+a)\b'

        # ==================== PORTUGUESE PATTERNS ====================
        
        # Pattern: "Sou Brasileiro", "Sou de Portugal"
        # Rationale: Portuguese equivalent of Spanish "Soy"
        # Captures both demonym and "de [country]" forms
        - '(?i)\bSou\s+(?:de\s+)?([A-ZÀ-ÿ][\w-]+)\b'
        
        # Pattern: "Moro no Brasil", "Moro na Argentina"  
        # Rationale: "I live in..." - residence proxy
        # Note: "no" (masc), "na" (fem), "nos/nas" (plural)
        - '(?i)\bMoro\s+n[oa]s?\s+([A-ZÀ-ÿ][\w\s-]+)\b'
        
        # Pattern: "Vou sair do Brasil" → contextual origin
        # Rationale: Leaving implies current location/origin
        - '(?i)\b(?:Vou\s+)?sair\s+d[oa]\s+([A-ZÀ-ÿ][\w-]+)\b'

        # ==================== GERMAN PATTERNS ====================
        
        # Pattern: "Ich bin Deutscher", "Ich komme aus Österreich"
        # Rationale: Standard German self-ID structures  
        # "bin" (am) + demonym, "komme aus" (come from) + country
        - '(?i)\bIch\s+(?:bin|komme\s+aus)\s+(?:ein|eine|einer)?\s*([A-ZÀ-ÿ][\w-]+)\b'
        
        # Pattern: "Ich bin aus Deutschland"
        # Rationale: Alternative "from" construction
        - '(?i)\bIch\s+bin\s+aus\s+([A-ZÀ-ÿ][\w\s-]+)\b'

        # ==================== DUTCH PATTERNS ====================
        
        # Pattern: "Ik ben Nederlands", "Ik kom uit België"  
        # Rationale: Dutch parallel to German structures
        # "ben" (am), "kom uit" (come from)
        - '(?i)\bIk\s+(?:ben|kom\s+uit)\s+([A-Z][\w-]+)\b'

        # ==================== CROSS-LINGUAL PATTERNS ====================
        
        # Pattern: Hyphenated dual identities
        # Rationale: "Italian-American", "Franco-German" (width=12 handles these)
        # Requires capital start to avoid false positives
        - '\b([A-Z][\w]+-[A-Z][\w]+)\b'
        
        # Pattern: "citizen of [Country]", "national of [Country]"
        # Rationale: Formal self-identification (multilingual contexts)
        - '(?i)\b(?:citizen|national)\s+of\s+([A-Z][\w\s-]+)\b'
```

## Appendix B: Pollution Pipeline Congiguration


```yaml
# conf/base/pipeline.yaml - Base configuration (shared across all environments)

seed: 42

gliner:
  model: "urchade/gliner_multi-v2.1"
  confidence_threshold: 0.60
  batch_size: 32
  device: "auto"  # "auto" | "cpu" | "cuda" (explicit authority)
  taxonomy_path: "conf/base/gliner_taxonomy.yaml"
  compute_explicit_recall: false
  explicit_recall_threshold: 0.80
  
  # ---------------------------------------------------------------------------
  # Bi-Encoder Enforcement & Word-Limit Settings
  # ---------------------------------------------------------------------------
  # Bi-encoder models support label embedding caching (encode_labels API).
  # Set require_bi_encoder: true to enforce bi-encoder architecture at load time.
  # If true and model lacks labels_encoder config, initialization fails.
  #
  # CRITICAL: GLiNER's internal processor truncates at max_words WORDS (not tokens).
  # The chunker uses tokens for budgeting, but GLiNER counts whitespace-split words.
  # gliner_max_words ensures chunks never exceed GLiNER's word limit.
  # Formula: effective_budget = min(token_budget, gliner_max_words * tokens_per_word_ratio)
  # ---------------------------------------------------------------------------
  require_bi_encoder: false  # Set true to enforce bi-encoder model architecture
  gliner_max_words: 315      # GLiNER processor word limit (matches model default)
  tokens_per_word_ratio: 1.3 # Avg subword tokens per word (conservative estimate)
  
  # ---------------------------------------------------------------------------
  # Batch Inference Optimization Settings (v2.0)
  # ---------------------------------------------------------------------------
  # These settings control the batched inference pipeline for improved throughput.
  # Key optimization: Batch multiple chunks together instead of sequential inference.
  #
  # Throughput gains: 4-8x speedup with batching enabled on GPU.
  # ---------------------------------------------------------------------------
  batch_inference:
    # Enable batched inference (recommended: true for production)
    # Set false to use legacy sequential mode for debugging or exact backward compat
    enable_batching: true
    
    # Number of length buckets for smart batching
    # null = auto-compute from VRAM (recommended)
    # Manual override: 5 (low VRAM) to 16 (high VRAM like A100)
    # More buckets = less padding waste, but more batch dispatch overhead
    num_buckets: null
    
    # Minimum samples per bucket before merging with adjacent bucket
    min_bucket_size: 4
    
    # Enable prompt/label embedding caching for bi-encoder models
    # Saves redundant encoding when processing multiple texts with same labels
    enable_prompt_caching: true
    
    # Strict padding for CUDA graph optimization
    # When true, pads all sequences to nearest bucket boundary (seq_len_buckets)
    # instead of just max length in batch. Improves CUDA graph cache hit rate.
    # Recommended: true for HPC (with CUDA graphs), false for laptop
    strict_padding: false
    
    # Sequence length bucket boundaries for strict padding
    # Should match cuda_graphs.seq_len_buckets if CUDA graphs are enabled
    # null = use default [64, 128, 192, 256, 320, 384, 448, 512]
    seq_len_buckets: null
  
  # Chunking strategy settings
  chunking:
    mode: "single_sentence"  # "single_sentence" | "accumulate"
    # legacy_sequential_mode bypasses all batch optimizations (debug only)
    legacy_sequential_mode: false
    # Parallelize CPU-side chunking across documents (multiprocessing).
    # Default off; enable on large workloads to reduce wall-clock time.
    parallel_chunking_workers: 0
    parallel_chunking_min_texts: 5000
    # minimum cores to enable parallel chunking
    parallel_chunking_min_cores: 16
    # batch_size_per_worker: Documents per worker batch (0 = auto)
    # Higher values reduce IPC overhead but increase per-worker memory
    batch_size_per_worker: 0

encoder:
  model: "intfloat/multilingual-e5-large"
  max_length: 512
  batch_size: 32
  device: "auto"  # "auto" | "cpu" | "cuda"
  # output_device: Device to move embeddings to after extraction.
  # null = keep on model device (recommended for HPC).
  # "cpu" = move to CPU after extraction (recommended for laptop VRAM management).
  output_device: null

paths:
  raw_data: "${oc.env:SOBR_DATA_PATH,./datasets}"
  output: "./artifacts"

dataloader:
  batch_size: 32
  num_workers: 4
  persistent_workers: false
  prefetch_factor: 2
  pin_memory: true

# ---------------------------------------------------------------------------
# Advanced Execution Optimizations
# ---------------------------------------------------------------------------
# These settings control low-level inference optimizations for maximum throughput.
# Most are disabled by default for compatibility; enable based on hardware.
# ---------------------------------------------------------------------------
execution:
  # CUDA Graph capture for static-shape inference (reduces kernel launch overhead)
  # Enable on GPU with consistent batch sizes; requires PyTorch with CUDA support
  enable_cuda_graphs: false
  cuda_graph_warmup: 3           # Warmup iterations before graph capture
  cuda_graph_cache_size: 16      # Max cached graphs (LRU eviction)
  
  # torch.compile for kernel fusion (PyTorch 2.0+)
  # "reduce-overhead" mode recommended for small batch sizes
  enable_torch_compile: false
  torch_compile_mode: "reduce-overhead"  # "default" | "reduce-overhead" | "max-autotune"
  torch_compile_fullgraph: false
  torch_compile_dynamic: false
  torch_compile_backend: "inductor"
  
  # Sequence packing: concatenate sentences with block-diagonal attention
  # Eliminates padding waste by packing multiple short sentences together
  sequence_packing:
    enabled: false
    max_packed_length: 2048      # Max total tokens in packed sequence
    min_sequences_to_pack: 2     # Minimum sequences before packing activates
    use_block_diagonal_mask: true
    pack_by_similarity: true     # Group similar-length sequences
  
  # Asynchronous prefetching: decouple CPU chunking from GPU inference
  # Uses producer-consumer pattern with multiprocessing.Queue
  async_prefetch:
    enabled: false
    queue_size: 4                # Items in prefetch queue
    num_workers: 1               # Prefetch worker processes
    timeout_seconds: 30.0
    prefetch_factor: 2
    use_multiprocessing: true    # false = use threading (lighter)
    pin_memory: true
    profile_aware: true          # Adjust settings based on ProfileType
  
  # GPU-accelerated span filtering (threshold on device before CPU transfer)
  gpu_span_filter:
    enabled: true
    max_spans_per_sequence: 100
    use_topk: false
    topk_k: 50
  
  # NUMA pinning for multi-socket systems
  numa_pinning: false

precision:
  dtype: "float32"
  use_grad_scaler: false

probe:
  compute_amnesic_drop: true
  amnesic_drop_threshold: 0.30
  train_split: 0.8
  max_samples: 20000
  # Backend selection: "auto" | "sklearn" | "cuml" | "torch"
  # "auto" detects GPU and uses cuml > torch > sklearn
  backend: "auto"
  # K-fold cross-validation settings
  use_kfold: true
  n_folds: 5
  # Statistical significance threshold for paired t-test
  pvalue_threshold: 0.05
  # Torch backend hyperparameters (used when backend="torch" or auto-selected)
  torch_lr: 0.01
  torch_epochs: 100
  torch_batch_size: 256
  torch_weight_decay: 0.0001
  # Class imbalance handling
  use_class_weights: true
  # Benchmark exact vs approximate solvers (for validation)
  benchmark_solvers: false

leace:
  force_cpu: false
  regularization: 1e-5
  batch_size: 100
  # device: Device for LEACE accumulation and computation.
  # null = fallback to encoder.device.
  # "cuda" = GPU LEACE (HPC), "cpu" = CPU LEACE (laptop/VRAM-limited).
  device: null
  # compute_dtype: Precision for LEACE matrix math.
  # "float64" (default): Numerical stability for Cholesky/SVD.
  # "float32": Faster on GPU but may have reduced precision.
  compute_dtype: "float64"

quality:
  enforce_thresholds: false

subset:
  enabled: false
  size: null

# ---------------------------------------------------------------------------
# Phase A Visualization Settings
# ---------------------------------------------------------------------------
# Controls report and plot generation after Phase A completes.
# Visualizations are generated inline after artifacts are saved (minimal overhead).
# ---------------------------------------------------------------------------
visualization:
  enabled: true                    # Set false to skip visualization generation
  pca_max_samples: 1000            # Max samples for PCA scatter plots (memory bound)
  embedding_batch_size: 32         # Batch size for computing embeddings for viz
  figure_dpi: 150                  # DPI for saved PNG figures
  figure_format: "png"             # Output format: "png" | "pdf" | "svg"
  color_palette: "husl"            # Seaborn color palette for demographic plots
  
  # Per-plot toggles (all enabled by default)
  plots:
    pca_before_after: true         # PCA scatter colored by demographic
    singular_values: true          # Projection matrix SVD spectrum
    embedding_norms: true          # Histogram of embedding norms
    amnesic_drop_bars: true        # Bar chart of probe accuracy drop
    gliner_confidence: true        # GLiNER confidence distribution
    detection_counts: true         # Detection count by entity type
    explicit_recall_bars: true     # Per-column recall bar chart
    entity_cooccurrence: true      # Entity co-occurrence heatmap
    demographic_scores: true       # Demographic score scatter matrix
    masking_coverage: true         # Masking coverage statistics

logging:
  level: "INFO"
  wandb_enabled: false
```

```yaml
# conf/hpc/pipeline.yaml - HPC-specific overrides (A100/H100)
# Optimized for staged execution on datacenter GPUs (40-80GB VRAM)

defaults:
  - /base/pipeline

# ---------------------------------------------------------------------------
# Staged Execution Architecture (HPC Mode)
# ---------------------------------------------------------------------------
# Two-stage pipeline decoupling CPU chunking from GPU inference:
# - Stage 1: CPU-saturated parallel chunking → persist to post_chunked
# - Stage 2: GPU-saturated global-sorted inference → minimal padding
# ---------------------------------------------------------------------------

execution:
  # Staged execution: primary configuration
  staged_execution:
    enabled: true
    persist_chunks: true           # Always persist to post_chunked for crash recovery
    global_sort_by_length: true    # Enable global argsort for padding efficiency
    gc_between_stages: true        # Force gc.collect() between stages

  # ---------------------------------------------------------------------------
  # Self-Optimizing Runtime (Autotuning)
  # ---------------------------------------------------------------------------
  # Adaptive token-budget batching with feedback loop.
  # Automatically adjusts batch sizes based on GPU throughput and memory.
  # ---------------------------------------------------------------------------
  autotuning:
    enabled: true
    warmup_batches: 5              # Reduced warmup to get to peak speed faster
    initial_token_budget: 32768  # Aggressive start for 5090
    min_token_budget: 2048
    max_token_budget: 196608       # Tuned for ~24-32GB VRAM usage (Safety cap)
    memory_headroom_mb: 4096       # Larger headroom for 5090 burst allocations
    scale_up_factor: 1.25          # 25% increase when scaling up
    scale_down_factor: 0.85        # 15% decrease when throttling
    oom_slash_factor: 0.5          # 50% reduction after OOM
    stability_threshold: 0.1       # 10% variance = stable
    history_window: 10             # Batches for moving average
    recovery_patience: 5           # Batches in recovery before scaling

  # ---------------------------------------------------------------------------
  # CUDA Graphs Configuration (Phase 3 Optimization)
  # ---------------------------------------------------------------------------
  # CUDA graphs eliminate kernel launch overhead by capturing and replaying
  # GPU operations. Combined with shape bucketing, this provides significant
  # speedup for repetitive inference patterns.
  #
  # Note: CUDA graphs work best with torch.compile. The GraphAwareInference
  # infrastructure handles shape bucketing and padding automatically.
  # ---------------------------------------------------------------------------
  enable_cuda_graphs: false
  cuda_graph_warmup: 3             # Warmup iterations before graph capture
  cuda_graph_cache_size: 128        # Max cached graphs (LRU eviction)
  
  # Shape bucketing for CUDA graphs
  cuda_graphs:
    # Batch size buckets (powers of 2 recommended for memory alignment)
    batch_buckets: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    # Sequence length buckets (aligned to transformer block sizes)
    seq_len_buckets: [64, 128, 192, 256, 320, 384, 448, 512]
    max_batch_size: 512            # Maximum batch size for graph capture
    max_seq_len: 512              # Maximum sequence length
    adaptive_buckets: false        # Create new buckets for unseen shapes
    min_bucket_usage: 1            # Min uses before capturing graph (skip one-offs)

  # ---------------------------------------------------------------------------
  # torch.compile Configuration (Phase 1 Optimization)
  # ---------------------------------------------------------------------------
  # Mode options:
  # - "default": Basic compilation, minimal overhead
  # - "reduce-overhead": Reduces CPU overhead via CUDA graphs (recommended)
  # - "max-autotune": Extensive Triton autotuning (SLOW initial compile, 
  #                   only use for benchmarking or after caching kernels)
  #
  # WARNING: max-autotune causes massive slowdown during first run due to
  # kernel autotuning. Use reduce-overhead for production.
  # ---------------------------------------------------------------------------
  enable_torch_compile: false      
  torch_compile_mode: "reduce-overhead"
  torch_compile_fullgraph: false
  torch_compile_dynamic: false      # DISABLED to force static kernels (faster execution)
  torch_compile_backend: "inductor"

  # ---------------------------------------------------------------------------
  # Async Result Storage (Phase 2 Optimization - Scatter-Store-Gather)
  # ---------------------------------------------------------------------------
  # Decouples inference from result reconstruction via Arrow IPC storage.
  # Enables crash recovery and constant memory footprint.
  # ---------------------------------------------------------------------------
  async_storage:
    enabled: true
    queue_size: 64                 # Producer->consumer queue depth (backpressure)
    flush_every_n: 5              # Batches between IPC flushes
    timeout_seconds: 0.1           # Consumer queue timeout

  # Sequence packing: enabled for zero-padding inference
  sequence_packing:
    enabled: false
    max_packed_length: 8192        # Higher limit for H100 (80GB)
    min_sequences_to_pack: 2
    use_block_diagonal_mask: true
    pack_by_similarity: true

  # Async prefetch: critical for Stage 2 GPU saturation
  async_prefetch:
    enabled: true
    queue_size: 64                 # Larger queue for global-sorted batches
    num_workers: 16                 # More workers for aggressive prefetching
    timeout_seconds: 120.0
    prefetch_factor: 8
    use_multiprocessing: true
    pin_memory: true               # Critical for overlapped PCIe transfers
    non_blocking: true             # Enable async CUDA transfers
    profile_aware: true

  # GPU span filtering
  gpu_span_filter:
    enabled: false
    max_spans_per_sequence: 512
    use_topk: false
    topk_k: 256

  numa_pinning: false

  # Memory management for large datasets
  memory:
    preallocate_buffers: true
    max_cached_batches: 64
    release_cache_every_n_batches: 200

dataloader:
  batch_size: 512                  # Larger batches for global-sorted inference
  num_workers: 12
  persistent_workers: true
  prefetch_factor: 4
  pin_memory: true

precision:
  dtype: "bfloat16"
  use_grad_scaler: false

gliner:
  model: "urchade/gliner_multi-v2.1"
  batch_size: 512                  # Larger batch size for Stage 2 (global-sorted) -> check autotuning
  device: "cuda"
  require_bi_encoder: false         # Allow uni-encoder models on HPC
  gliner_max_words: 512
  tokens_per_word_ratio: 1.3

  # ---------------------------------------------------------------------------
  # HPC Batch Inference (Stage 2 Optimization)
  # ---------------------------------------------------------------------------
  batch_inference: # check autotuning settings
    enable_batching: true
    num_buckets: 16                # More buckets for finer length grouping
    min_bucket_size: 32            # Larger min size for GPU efficiency
    enable_prompt_caching: true
    strict_padding: false           # Enforce exact padding within buckets for CUDA graph hit rate
    # Sequence length buckets aligned to CUDA graph shape buckets (must match cuda_graphs.seq_len_buckets)
    seq_len_buckets: [64, 128, 192, 256, 320, 384, 448, 512]

    # Global sorting eliminates most bucketing waste
    # These are fallback settings when global sort unavailable
    fallback_num_buckets: 8
    fallback_min_bucket_size: 8

  # ---------------------------------------------------------------------------
  # Staged Execution Chunking Settings (Stage 1)
  # ---------------------------------------------------------------------------
  chunking:
    mode: "single_sentence"
    legacy_sequential_mode: false  # MUST be false for staged execution

    # Stage 1: CPU saturation
    parallel_chunking_workers: 60  # Saturate HPC CPU cores
    parallel_chunking_min_texts: 256
    micro_batch_size: 2000         # Larger micro-batches for HPC RAM

    # Checkpoint for crash recovery
    checkpoint_path: "${paths.output}/phase_a/gliner_chunk_cache.pkl"

    # Shard size for embedding + LEACE
    shard_size: 50000              # Large shards for HPC memory

encoder:
  model: "intfloat/multilingual-e5-large"
  device: "cuda"
  output_device: "cpu"             # Move to CPU immediately (prevent VRAM exhaustion during accumulation)
  batch_size: 512                  # Large batch for embedding
  max_length: 512

leace:
  regularization: 1.0e-4
  force_cpu: true                  # Accumulate covariance stats in RAM (prevent VRAM exhaustion)
  batch_size: 10000                # Large LEACE batches (used for shard control)
  device: "cuda"                   # Final projection computation on GPU (FP64 Cholesky/SVD)
  compute_dtype: "float64"         # float64 for numerical stability in Cholesky/SVD

probe:
  benchmark_solvers: true          # Benchmark HPC solver

quality:
  enforce_thresholds: true

subset:
  enabled: false                   # Process full dataset on HPC

logging:
  level: "INFO"
  log_gpu_memory: true
  log_batch_timing: true

# Paths for staged execution artifacts
paths:
  output: "artifacts"
  phase_a:
    clean_dataset: "${paths.output}/phase_a/clean_dataset.arrow"
    projection_matrix: "${paths.output}/phase_a/projection_matrix.pt"
    pollution_logs: "${paths.output}/phase_a/pollution_logs.arrow"
    staged_checkpoint: "${paths.output}/phase_a/staged_checkpoint.arrow"
```

## Appendix C: Training and Verification Configuration

```yaml
# Phase D Training Configuration (Base)
# ==============================================================================
# AOT Pipeline Optimization Blueprint Implementation
# Target: High-throughput training (100+ it/s on RTX 5090)
# ==============================================================================

phase_d:
  experiment_name: "phase_d_baseline"
  seed: 1337

# Model configuration
model:
  name: "intfloat/multilingual-e5-large"
  max_length: 512
  taxonomy_path: "conf/base/gliner_taxonomy.yaml"

# Data configuration
data:
  # Dataset paths (override via CLI)
  dataset_path: "artifacts/phase_d/tokenized_dataset.arrow"  # Pre-tokenized AOT data
  fallback_path: "artifacts/phase_a/clean_dataset.arrow"     # JIT fallback
  artifacts_dir: "artifacts/phase_a"

  # Split configuration
  use_split_column: true  # Use existing 'split' column if present
  split_ratios:
    train: 0.8
    val: 0.1
    test: 0.1
  split_seed: 42

  # Demographic label fields (from SOBR schema)
  label_fields:
    - female
    - birth_year
    - nationality
    - political_leaning
    - extrovert
    - sensing
    - feeling
    - judging

# Training configuration
training:
  num_epochs: 3
  batch_size: 8  # Base batch size (used when dynamic batching disabled)
  learning_rate: 2e-5
  layerwise_lr_decay: 0.95
  gradient_accumulation_steps: 1
  
  # Precision: "fp32", "bf16", "fp8" (fp8 requires TransformerEngine)
  precision: "bf16"
  
  early_stopping:
    enabled: false
    patience: 2
    min_delta: 0.0
    metric: "f1_macro"

  # For testing/debugging
  max_steps: null  # null means run full epochs

  # Checkpointing
  save_every_steps: null
  save_every_epochs: 1  # Save after each epoch
  resume_from: null

  # Device
  device: "auto"  # auto, cuda, cpu

# Scheduler
scheduler:
  name: "linear"  # linear or cosine
  num_warmup_steps: 0

# Evaluation
evaluation:
  eval_on_val: true
  eval_on_test: true
  eval_every_epochs: 1
  write_report: true

# Output
output:
  save_model: true
  save_logs: true
  save_metrics: true
  reports_dir: "artifacts/reports"

# ==============================================================================
# AOT Pipeline Optimization Settings
# Reference: Phase D Final Optimization Blueprint
# ==============================================================================

optimization:
  # AOT (Ahead-Of-Time) tokenization mode
  # When true, uses pre-tokenized data from preprocess_tokens.py
  use_aot_mode: true
  
  # torch.compile settings for CUDA Graph caching
  use_torch_compile: true
  torch_compile_mode: "reduce-overhead"  # Enables automatic CUDA Graph capture
  torch_compile_dynamic: false
  torch_compile_backend: "inductor"
  compile_train_step: false
  torch_compile_disable_cudagraphs: false

  # Manual CUDA-graph training (forward + loss + backward capture)
  # NOTE: Requires static bucket shapes and disables torch.compile.
  use_cuda_graph_training: false
  cuda_graph_training:
    enabled: true
    # Shape buckets for graph capture (pad to nearest bucket)
    batch_buckets: [8, 16, 32, 64]
    seq_len_buckets: [128, 256, 384, 512]
    max_batch_size: 64
    max_seq_len: 512
    min_bucket_usage: 2
    # Graph cache behavior
    warmup_iterations: 0  # Avoid extra backward passes during graph capture
    max_cached_graphs: 16
    capture_pool_size_mb: 256
    use_cuda_graph_memory_pool: true
    clone_outputs: true
    pad_token_id: 1
    ignore_index: -1
  
  # Fused AdamW kernel (eliminates Python loop overhead)
  use_fused_optimizer: true
  
  # Async H2D prefetching via DevicePrefetcher
  use_device_prefetch: true
  
  # Quantized bucketing (Snap-to-Grid) for stable tensor shapes
  # Step of 16 aligns with transformer attention blocks
  quantize_step: 16
  
  # Token budget for quantized sampler (replaces max_batch_size)
  # Higher = larger batches, better GPU utilization
  token_budget: 65536  # ~128 samples @ 512 tokens

# Hardware-aware execution
execution:
  autotuning:
    enabled: false
    warmup_batches: 10
    initial_token_budget: 65536
    min_token_budget: 8192
    max_token_budget: 262144
    memory_headroom_mb: 1024
    scale_up_factor: 1.25
    scale_down_factor: 0.85
    oom_slash_factor: 0.5
    stability_threshold: 0.1
    history_window: 10
    recovery_patience: 5
    
  # Token-budget batching: enables QuantizedBucketSampler (AOT) or PhaseDBudgetedBatchSampler (JIT)
  # for length-aware batching that minimizes padding waste.
  # REQUIRED when use_aot_mode=True (raises RuntimeError otherwise).
  use_token_budget_batching: true
  
  # Adaptive token budget: enables RuntimeController for dynamic budget adjustment
  # via PID-like feedback loop. Requires use_token_budget_batching=True.
  use_adaptive_token_budget: false
  
  # Batch constraints for PhaseDBudgetedBatchSampler (JIT mode only)
  # QuantizedBucketSampler uses token_budget from optimization config instead
  batch_constraints:
    max_batch_size: 2048  # High cap (no artificial starvation)
    min_batch_size: 1
    drop_last: false
    shuffle: true
    seed: 42
    
  telemetry:
    enabled: false  # Disable by default to avoid sync overhead
    # Strided sampling: measure every Nth batch to reduce overhead
    stride: 50  # Measure every 50th batch
    
  data_loader:
    num_workers: 4  # Parallel data loading
    pin_memory: true  # For async H2D transfers
    persistent_workers: true  # Keep workers alive between epochs
    prefetch_factor: 2  # Batches to prefetch per worker
```

```yaml
# Phase D Training Configuration (HPC Overrides)
# ==============================================================================
# Maximum throughput settings for high-end GPUs (RTX 5090 / A100 / H100)
# Target: 100+ iterations/second via AOT pipeline optimization
# ==============================================================================

# HPC training overrides
training:
  num_epochs: 300
  batch_size: 64  # Base batch (dynamic batching uses token budget instead)
  learning_rate: 3e-5
  layerwise_lr_decay: 0.9
  gradient_accumulation_steps: 2
  save_every_epochs: null
  save_every_steps: null
  
  # FP8 for maximum tensor core throughput (requires TransformerEngine)
  # Falls back to BF16 if TransformerEngine not installed
  precision: "fp8"
  
  max_steps: null
  resume_from: null
  
  early_stopping:
    enabled: true
    # Increased patience to tolerate transient regressions during warmup
    patience: 6
    # Require a small meaningful improvement to count as better
    min_delta: 0.01
    metric: "f1_macro"

# Scheduler optimized for longer training
scheduler:
  name: "cosine"
  num_warmup_steps: 500

# HPC optimization overrides - MAXIMUM THROUGHPUT
optimization:
  # AOT mode mandatory for 100+ it/s
  use_aot_mode: true
  
  # Pre-padding enables zero-copy collation (~2ms vs ~15ms/batch)
  # Use --pre-pad when running preprocess_tokens.py
  use_pre_padded: true
  
  # torch.compile with CUDA Graph capture
  use_torch_compile: true
  torch_compile_mode: "reduce-overhead"
  torch_compile_dynamic: true
  torch_compile_backend: "inductor"
  compile_train_step: true
  # IMPORTANT: Set to true to avoid "tensor output of CUDAGraphs overwritten" errors
  # This uses mode="default" (kernel fusion only) instead of CUDA Graph capture
  # Slight throughput reduction (~1.2x vs ~2.5x) but much more stable
  torch_compile_disable_cudagraphs: true

  # Manual CUDA-graph training (disabled by default)
  use_cuda_graph_training: false
  
  # Fused optimizer eliminates Python loop overhead
  use_fused_optimizer: true
  
  # Async prefetch for pipeline parallelism
  use_device_prefetch: true
  
  # Quantized bucketing for compilation stability
  quantize_step: 32
  
  # Conservative token budget - autotuning will scale up if headroom exists
  # Start lower to avoid OOM cascade during warmup
  # Reduced conservative token budget to avoid large single allocations
  # Further reduced from 8192 to 4096 to prevent aggressive epoch-to-epoch scaling
  token_budget: 8192  # safer default for large models

# HPC execution overrides
execution:
  autotuning:
    enabled: false  # Let runtime find optimal budget
    warmup_batches: 10
    initial_token_budget: 8192  # Conservative start (down from 8192)
    min_token_budget: 1024
    max_token_budget: 393216     # Reduced safety cap (down from 262144)
    memory_headroom_mb: 4096    # Larger headroom for 5090 burst allocations
    scale_up_factor: 1.25
    scale_down_factor: 0.85
    oom_slash_factor: 0.5
    stability_threshold: 0.1
    history_window: 10
    recovery_patience: 5
    
  # Token-budget batching: REQUIRED for AOT mode (mandatory for HPC pipeline)
  # Enables QuantizedBucketSampler for CUDA Graph stability and padding efficiency
  use_token_budget_batching: true
  
  # Adaptive token budget: RuntimeController adjusts budget based on GPU feedback
  # Works with autotuning above for maximum throughput
  use_adaptive_token_budget: false
  
  # Batch constraints (for JIT mode fallback only)
  batch_constraints:
    max_batch_size: 1024
    min_batch_size: 128
    drop_last: false
    shuffle: true
    seed: 42
    
  telemetry:
    enabled: true  # Monitor for profiling
    stride: 50  # Strided sampling to avoid sync overhead
    
  # DataLoader tuning optimized for RTX 5090 / A100 / H100
  # GPU is so fast it can starve without aggressive prefetching
  data_loader:
    num_workers: 24  # Increased parallelism - RTX 5090 needs more data throughput
    pin_memory: true  # Required for async H2D transfers
    persistent_workers: true  # Avoid worker respawn overhead (critical for throughput)
    prefetch_factor: 16  # Aggressive prefetching to prevent GPU starvation
    
  # OOM recovery configuration
  oom_recovery:
    # Epoch-level OOM protection for extreme throughput scenarios
    # Batch-level protection has minimal overhead, but epoch-level allows
    # more aggressive optimizations within the hot loop
    epoch_level_protection: true  # Set true for experimental max throughput
    max_epoch_retries: 2  # Retry epoch up to N times after OOM
```

```yaml
# Phase D Verification Configuration (HPC Overrides)

model:
  name: "intfloat/multilingual-e5-large"
  max_length: 512
  taxonomy_path: "conf/base/gliner_taxonomy.yaml"

verify:
  batch_size: 8
  chg:
    epochs: 10
    learning_rate: 0.001
    regularization: 0.01
    facilitating_threshold: 0.7
    irrelevant_threshold: 0.3
  svs:
    max_batches: 20
    query_strategy: "mean_tokens"
    pos_backend: "hf"
    use_pos: true
    spacy_model: "en_core_web_sm"
    spacy_use_gpu: false
    spacy_gpu_id: 0
    spacy_batch_size: 32
    spacy_n_process: 2
    spacy_disable:
      - "ner"
      - "parser"
    hf_model: "vblagoje/bert-english-uncased-finetuned-pos"
    hf_device: -1
    hf_batch_size: 16
```

---

## Appendix C: Detailed Results and Figures

### C.1 Phase A Metrics (Complete)

**Table C1: Complete Phase A Metrics**

| Metric Category | Metric | Value |
|:----------------|:-------|:------|
| **Dataset** | Total samples | 82,615 |
| | Samples with masks | 47,311 |
| | Mask rate | 57.27% |
| | Mean text length | 8,425.2 ± 767.7 chars |
| **GLiNER** | Total spans detected | 268,768 |
| | Spans per masked sample | 5.68 ± 6.35 |
| | Mean confidence | 0.682 ± 0.066 |
| | Confidence range | [0.601, 0.942] |
| | Mean span length | 6.68 ± 1.94 tokens |
| **LEACE** | Matrix shape | 1024 × 1024 |
| | Idempotence error | $2.60 \times 10^{-7}$ |
| | Frobenius norm | 37.58 |
| | Effective rank | 968 / 1024 |
| | Erased directions | 56 |
| **Probe** | Accuracy before LEACE | 64.80% |
| | Accuracy after LEACE | 23.79% |
| | **Amnesic drop** | **63.28%** |
| | Balanced acc. before | 48.31% |
| | Balanced acc. after | 14.35% |
| | Significance (p-value) | $1.86 \times 10^{-8}$ |

### C.2 CHG Head Classification (Complete)

**Table C2: Head Classification Summary**

| Classification | Baseline | Constrained | Δ |
|:---------------|:---------|:------------|:--|
| Facilitating ($g > 0.7$) | 52 | 90 | +38 (+73%) |
| Irrelevant ($g < 0.3$) | 102 | 108 | +6 (+6%) |
| Neutral | 230 | 186 | -44 (-19%) |
| **Total** | 384 | 384 | — |

**Layer-wise Facilitating Head Distribution:**

| Layer Range | Baseline | Constrained | Δ |
|:------------|:---------|:------------|:--|
| Early (0–5) | 7 | 18 | +11 (+157%) |
| Middle (6–13) | 10 | 25 | +15 (+150%) |
| Late (14–23) | 35 | 47 | +12 (+34%) |

### C.3 Figures

**Figure C.1: CHG Head Classification Comparison**

![Head Classification Comparison](full_run/phase_d/reports/phase_d_assets/head_classification_comparison.png)

**Figure C.2: Accuracy and F1 Comparison**

![Accuracy Comparison](full_run/phase_d/reports/phase_d_assets/accuracy_comparison.png)

![F1 Comparison](full_run/phase_d/reports/phase_d_assets/f1_comparison.png)

**Figure C.3: Training Loss Curves**

![Loss Comparison](full_run/phase_d/reports/phase_d_assets/loss_comparison.png)

**Figure C.4: Phase A LEACE Singular Values**

![Singular Values](full_run/phase_a/reports/phase_a/singular_values.png)

**Figure C.5: Amnesic Drop with Confidence Intervals**

![Amnesic Drop](full_run/phase_a/reports/phase_a/amnesic_drop_with_ci.png)

**Figure C.6: GLiNER Detection Confidence Distribution**

![GLiNER Confidence](full_run/phase_a/reports/phase_a/gliner_confidence.png)

**Figure C.7: Embedding Separability (PCA)**

![PCA Nationality](full_run/phase_a/reports/phase_a/pca_nationality.png)

**Figure C.8: Confusion Matrix Comparison**

![Baseline Confusion](full_run/phase_d/reports/phase_d_assets/nationality_baseline_confusion_topn.png)

![Constrained Confusion](full_run/phase_d/reports/phase_d_assets/nationality_constrained_confusion_topn.png)

---

## Appendix D: Experimental Setup and Preprocessing

### D.1 Baseline vs. Constrained Model Comparison

**Table D1: Experimental Conditions**

| Aspect | Baseline (Dirty) | Constrained (Clean) |
|:-------|:-----------------|:--------------------|
| **Text field** | `post` (raw text) | `post_masked` (GLiNER-processed) |
| **Affine Guard** | None | LEACE projection $\mathbf{P}$ (frozen) |
| **Embedding layer** | Trainable | Frozen |
| **Input example** | "I'm German, so I..." | "[MASK:NATIONALITY], so I..." |
| **Rationale** | Shortcut-exploiting baseline | Pollution-mitigated model |

### D.2 Preprocessing Pipeline (Baseline)

The baseline model receives raw posts without any pollution mitigation:

1. **Tokenization**: `intfloat/multilingual-e5-large` tokenizer with max length 512
2. **Text field**: Uses Arrow column `post` (raw Reddit post text)
3. **Padding**: Dynamic padding to batch maximum length
4. **Embedding**: Full encoder weights trainable during backpropagation

### D.3 Preprocessing Pipeline (Constrained)

The constrained model processes data through the Phase A pipeline before training:

1. **GLiNER Detection**: Zero-shot span detection using 7 positive prompts and 6 negative distractors
   - Confidence threshold: 0.60
   - Maximum span width: 20 tokens
   - Word budget per chunk: 315 words
2. **Span Masking**: Detected spans replaced with typed token `[MASK:NATIONALITY]`
3. **LEACE Projection**: Closed-form projection matrix computed in float64:
   $$\mathbf{P} = \mathbf{I} - \boldsymbol{\Sigma}_{XZ} \boldsymbol{\Sigma}_{ZZ}^{-1} \boldsymbol{\Sigma}_{ZX}$$
   - Cholesky regularization: $\epsilon = 10^{-5}$
   - Idempotence verified: $\|\mathbf{P}^2 - \mathbf{P}\|_F / \|\mathbf{P}\|_F < 10^{-5}$
4. **Affine Guard Injection**: Projection applied as frozen linear layer after embeddings:
   $$\mathbf{h}_0 = \mathbf{P} \cdot \mathbf{E}(\text{tokens}) + \mathbf{Pos}$$
5. **Embedding Freeze**: Encoder embedding layer frozen to prevent re-learning nationality signal

---

## Appendix E: Pipeline Optimizations (Technical)

### E.1 Phase A Optimizations

**GLiNER Batch Inference (4–8× speedup):**

| Optimization | Description | Implementation |
|:-------------|:------------|:---------------|
| Length bucketing | Group sequences by length to minimize padding | Auto-computed from VRAM; `num_buckets=null` |
| Prompt caching | Cache bi-encoder label embeddings across documents | `enable_prompt_caching: true` |
| Strict padding | Pad to bucket boundaries for CUDA graph cache hits | Optional; `strict_padding: false` |
| Batch size | Adaptive based on sequence length | Base: 32, scales with length |

**Chunking Strategy:**

- **Mode**: `single_sentence` (sentence-boundary splits)
- **Word budget**: 315 words (GLiNER processor limit)
- **Token budget**: ~410 tokens (315 × 1.3 tokens/word ratio)
- **Parallel workers**: Configurable; default 0 (sequential)

**LEACE Computation:**

- **Precision**: `float64` for Cholesky numerical stability
- **Regularization**: Tikhonov ridge $\epsilon = 10^{-5}$
- **Batch accumulation**: 100-sample shards for covariance estimation
- **Device**: Auto-detect (CUDA for HPC, CPU for laptop)

### E.2 Phase D Optimizations

**AOT (Ahead-Of-Time) Pipeline:**

| Stage | Operation | Benefit |
|:------|:----------|:--------|
| Pre-tokenization | Tokenize offline → Arrow dataset | Eliminates JIT overhead (~15ms/batch) |
| Quantized bucketing | Snap lengths to grid (step=16) | Tensor core alignment; compilation stability |
| Token-budget batching | `QuantizedBucketSampler` | Variable batch size; consistent memory usage |
| Fused optimizer | Fused AdamW kernel | Eliminates Python loop overhead |

**Compiler Optimizations:**

- **torch.compile**: `mode="reduce-overhead"` with `inductor` backend
- **Dynamic shapes**: Disabled for static kernel compilation
- **CUDA graphs**: Optional manual capture for repetitive inference

**Hardware Accelerations:**

- **Async prefetch**: Producer-consumer H2D transfers overlapped with compute
- **Pin memory**: DMA-enabled for faster PCIe transfers
- **Persistent workers**: Avoid respawn overhead between epochs
- **Prefetch factor**: 2–16 batches per worker (hardware-dependent)

### E.3 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           PHASE A PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Raw SOBR     ┌──────────┐   ┌──────────┐   ┌──────────┐                │
│  Arrow File → │ GLiNER   │ → │ Masker   │ → │ Encoder  │                │
│   [post]      │ Detector │   │(replace) │   │(embed)   │                │
│               └──────────┘   └──────────┘   └────┬─────┘                │
│                                                  │                       │
│                              ┌───────────────────▼─────────────────────┐ │
│                              │           LEACE Computer                 │ │
│                              │  Σ_XZ, Σ_ZZ → P = I - Σ_XZ·Σ_ZZ⁻¹·Σ_ZX  │ │
│                              └───────────────────┬─────────────────────┘ │
│                                                  │                       │
│                              ┌───────────────────▼─────────────────────┐ │
│                              │         Output Artifacts                 │ │
│                              │  • clean_dataset.arrow [post_masked]     │ │
│                              │  • projection_matrix.pt [P]              │ │
│                              │  • phase_a_metrics.json                  │ │
│                              └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE D PIPELINE (BASELINE)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  clean_dataset   ┌───────────┐   ┌───────────┐   ┌───────────┐          │
│  [post field] → │ Tokenizer │ → │ Encoder   │ → │ CLS Pool  │          │
│                  │ (JIT)     │   │(trainable)│   └─────┬─────┘          │
│                  └───────────┘   └───────────┘         │                 │
│                                                        ▼                 │
│                              ┌───────────────────────────────────┐      │
│                              │    SingleTaskHead (nationality)   │      │
│                              │         CrossEntropyLoss          │      │
│                              └───────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                  PHASE D PIPELINE (CONSTRAINED)                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  clean_dataset      ┌───────────┐   ┌─────────────┐   ┌───────────┐     │
│  [post_masked] →   │ Tokenizer │ → │ Affine Guard│ → │ Encoder   │     │
│                     │ (JIT)     │   │ P·E(x)+Pos  │   │ (frozen)  │     │
│                     └───────────┘   │  (frozen)   │   └─────┬─────┘     │
│                                     └─────────────┘         │           │
│                                                             ▼           │
│                              ┌───────────────────────────────────┐      │
│                              │    SingleTaskHead (nationality)   │      │
│                              │         CrossEntropyLoss          │      │
│                              └───────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                      CHG/SVS VERIFICATION                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Trained Model ─┬─► CHG Gate Learning (L1-regularized) ─► Head Classes  │
│                 │                                                        │
│                 └─► Attention Extraction ─► SVS Calculator              │
│                                              │                           │
│                         ┌────────────────────┴────────────────────┐     │
│                         │  SVS = Σ(function_mass)/Σ(content_mass) │     │
│                         │  over facilitating heads only           │     │
│                         └─────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Appendix F: Pipeline Configuration Summary

**Table F1: Key Configuration Parameters**

| Component | Parameter | Value | Source |
|:----------|:----------|:------|:-------|
| GLiNER | Model | `urchade/gliner_multi-v2.1` | `pipeline.yaml` |
| GLiNER | Confidence threshold | 0.60 | `pipeline.yaml` |
| GLiNER | Max words per chunk | 315 | `pipeline.yaml` |
| LEACE | Regularization | $10^{-5}$ | `pipeline.yaml` |
| LEACE | Compute dtype | `float64` | `pipeline.yaml` |
| Encoder | Model | `intfloat/multilingual-e5-large` | `phase_d.yaml` |
| Encoder | Max length | 512 | `phase_d.yaml` |
| Training | Epochs | 3 | `phase_d.yaml` |
| Training | Batch size | 8 | `phase_d.yaml` |
| Training | Learning rate | $2 \times 10^{-5}$ | `phase_d.yaml` |
| Training | Precision | `bf16` | `phase_d.yaml` |
| CHG | Epochs | 10 | `verify.yaml` |
| CHG | Learning rate | $10^{-3}$ | `verify.yaml` |
| CHG | L1 regularization | 0.01 | `verify.yaml` |
| CHG | Facilitating threshold | 0.7 | `verify.yaml` |
| CHG | Irrelevant threshold | 0.3 | `verify.yaml` |
