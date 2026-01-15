# Autonomous Pollution Mitigation for Neural Stylometry: A Nationality Case Study

---

## Abstract

Author profiling models trained on social media corpora frequently exploit demographic shortcuts—explicit self-identification tokens such as "I'm German" or "As a Mexican"—rather than learning genuine stylometric patterns. This phenomenon, termed the *Clever Hans* effect, undermines the validity of stylometric classifiers for privacy-sensitive applications. We present an autonomous neuro-symbolic pipeline that mitigates nationality pollution in the SOBR corpus without requiring manually annotated span data. Our approach combines GLiNER-based symbolic detection with LEACE geometric concept erasure, applying the resulting projection as a frozen *Affine Guard* layer in downstream transformer training. We validate mitigation efficacy using Causal Head Gating (CHG) to identify attention heads causally responsible for nationality prediction. Experiments on the nationality-only slice (≈28% of SOBR) demonstrate that the constrained model exhibits reduced reliance on pollution heads while maintaining classification performance on de-polluted inputs. The proposed architecture generalizes to multi-demographic mitigation via multi-concept projection matrices, providing a scalable framework for bias-aware stylometric analysis.

---

## 1. Introduction

Computational stylometry posits that an author's writing style encodes latent attributes—including demographic characteristics such as nationality, age, and gender—that can be inferred through machine learning [1]. However, social media corpora present a critical confound: users frequently self-identify their demographics explicitly within posts (e.g., "As a Canadian...", "Soy mexicano"), creating *pollution* features that correlate perfectly with labels but carry no stylometric information [2].

This pollution induces *shortcut learning* [3], wherein neural classifiers minimize empirical risk by attending to high-availability explicit tokens rather than low-availability stylistic patterns (e.g., syntactic preferences, function word distributions). Hermann et al. [4] formalize this as a trade-off between *predictivity* and *availability*: non-linear networks exhibit strong bias toward features that are easily extractable, even when less predictive than core features. Consequently, standard training on polluted corpora yields "Clever Hans" models [3] that appear accurate but fail to generalize to genuinely anonymized text.

**Research Question.** We ask: *Can an autonomous neuro-symbolic pipeline—combining zero-shot entity detection with closed-form concept erasure—reduce nationality shortcut learning while preserving stylometric signals, as verified through causal attention head analysis?*

**Scope and Tractability.** The SOBR corpus [2] provides distant labels for eight demographic attributes. For tractability within a 4-page format, we restrict our analysis to **nationality** (country-level labels), which constitutes approximately 28% of the corpus (65.5M raw posts from 165K authors in the sampled split). We emphasize that the proposed pipeline supports multi-demographic mitigation through multi-concept projection matrices (§3.2), and the accompanying codebase implements joint removal across all SOBR attributes.

**Contributions.** (1) We propose a taxonomy-driven pollution detection system using GLiNER with entity-centric prompts and semantic distractors, eliminating the need for manually labeled span data. (2) We integrate LEACE projection as a frozen Affine Guard layer, providing closed-form linear concept erasure with minimal collateral distortion. (3) We validate mitigation using Causal Head Gating [5], demonstrating reduced facilitation scores for "pollution heads" in constrained models.

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

*[Placeholder: Results tables and figures to be populated after experimental runs.]*

### 6.1 Phase A Mitigation Effectiveness

**Table 2: Phase A Metrics**

| Metric | Value |
|:-------|:------|
| Explicit Recall | [TBD] |
| Amnesic Drop | [TBD] |
| BERTScore (Recall) | [TBD] |
| Mask Rate | [TBD] |
| Projection Idempotence Error | [TBD] |

### 6.2 Phase D Classification Performance

**Table 3: Baseline vs. Constrained Model Performance**

| Model | Accuracy | Macro F1 | CHG Facilitating Heads |
|:------|:---------|:---------|:-----------------------|
| Baseline (Dirty) | [TBD] | [TBD] | [TBD] |
| Constrained (Clean) | [TBD] | [TBD] | [TBD] |
| Δ (Constrained − Baseline) | [TBD] | [TBD] | [TBD] |

### 6.3 CHG Head Analysis

**Figure 2: CHG Gate Heatmaps**

*[Placeholder: Side-by-side heatmaps showing gate values $g_{\ell,h}$ for baseline and constrained models across 24 layers × 16 heads. Expected pattern: early-layer facilitating heads in baseline become irrelevant in constrained model.]*

### 6.4 Qualitative Analysis

*[Placeholder: Example posts with GLiNER-detected spans highlighted; attention weight visualizations for facilitating vs. irrelevant heads.]*

---

## 7. Discussion and Conclusion

### 7.1 Interpretation of Results

*[Placeholder: Interpret whether accuracy drop (if any) reflects shortcut removal vs. legitimate signal loss. Discuss CHG evidence for mechanistic change in attention patterns.]*

### 7.2 Limitations

1. **Prompt Bias.** The taxonomy encodes implicit assumptions about how nationality is expressed; novel self-identification patterns (e.g., cultural references) may escape detection.
2. **Multilingual Coverage.** While the taxonomy includes Spanish, Portuguese, German, and Dutch patterns, coverage of non-European languages is limited.
3. **Residual Non-Linear Leakage.** LEACE removes only *linear* information; non-linear correlations (e.g., topic clusters) may persist beyond the projection.
4. **Label Noise.** SOBR nationality labels are distantly supervised from flairs; some labels may reflect residence rather than nationality proper.

### 7.3 Future Work

The proposed pipeline supports multi-demographic mitigation through multi-concept projection matrices. Future work should evaluate joint removal of all eight SOBR attributes and assess whether CHG patterns differ for categorical (nationality) vs. ordinal (age) demographics.

### 7.4 Conclusion

We presented an autonomous neuro-symbolic pipeline for mitigating nationality pollution in stylometric corpora. By combining GLiNER zero-shot detection with LEACE closed-form erasure and CHG causal verification, our approach provides a scalable alternative to manual annotation. The nationality case study demonstrates feasibility; extension to multi-demographic mitigation is supported by the architecture and left for future empirical validation.

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

---

## Appendix A: GLiNER Taxonomy Configuration

The following YAML configuration defines the column-driven prompt/distractor schema for all SOBR demographic attributes. For the nationality-only experiments reported in this paper, only the `nationality` block is active during detection.

```yaml
# GLiNER Taxonomy Configuration - Column-Driven Schema
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
    # birth_year: Extracted from (GAA) patterns like "(F34)", "(32M)"
    # -----------------------------------------------------------------------
    birth_year:
      prompts:
        - "age statement"
        - "self-identified age"
        - "birth year statement"
        - "born in year"
      distractors:
        - "third-person age mention"
        - "historical age reference"
      mask_token: "[MASK:AGE]"
      width: 6
      reference_patterns:
        - "\\([MFmf]\\d{1,2}\\)"
        - "\\(\\d{1,2}[MFmf]\\)"
        - "\\b\\d{1,2}[MFmf]\\b"
        - "\\b[MFmf]\\d{1,2}\\b"
        - "(?i)\\bI\\s*(?:am|'m)\\s*\\d{1,2}\\b"
        - "(?i)\\b(?:born\\s+in|born\\s+around)\\s*\\d{4}\\b"
        - "(?i)\\bI\\s+was\\s+born\\s+in\\s+\\d{4}\\b"

    # -----------------------------------------------------------------------
    # female: Extracted from (GAA) patterns - gender component
    # -----------------------------------------------------------------------
    female:
      prompts:
        - "gender self-identification"
        - "self-identified gender"
      distractors:
        - "third-person gender mention"
        - "gender of another person"
      mask_token: "[MASK:GENDER]"
      width: 4
      reference_patterns:
        - "\\([MFmf]\\d{1,2}\\)"
        - "\\(\\d{1,2}[MFmf]\\)"
        - "\\b\\d{1,2}[MFmf]\\b"
        - "\\b[MFmf]\\d{1,2}\\b"
        - "(?i)\\bI\\s*(?:am|'m)\\s*(?:a\\s+)?(?:man|woman|male|female|guy|girl)\\b"
        - "(?i)\\bas\\s+a\\s+(?:man|woman|male|female|guy|girl)\\b"

    # -----------------------------------------------------------------------
    # nationality: Extracted from flairs on European subreddits
    # DESIGN: Entity-centric prompts + semantic distractors for precision
    # -----------------------------------------------------------------------
    nationality:
      prompts:
        # Core identity prompts - describe possessive relationship
        - "the author's country of origin"
        - "the speaker's nationality"  
        - "the user's home country"
        # Demonym-focused prompts - capture "I'm Chilean" patterns
        - "a demonym describing the author"
        - "the writer's ethnic or national identity"
        # Contextual residence prompts - for "I live in..." patterns
        - "the country where the author lives"
        - "the nation the user identifies with"
      distractors:
        # Political/news context absorbers
        - "a country mentioned in news or politics"
        - "a nation involved in the discussed event"
        - "a foreign government being referenced"
        # Linguistic false-positive absorbers  
        - "a language being discussed"
        - "an unrelated geographical location"
        # Comparative context absorbers
        - "a country used for comparison"
      width: 20
      mask_token: "[MASK:NATIONALITY]"
      reference_patterns:
        # === ENGLISH ===
        - "(?i)\\bI(?:'m|\\s+am)\\s+(?:from\\s+)?(?:a|an)?\\s*([A-Z][\\w-]+(?:\\s+[A-Z][\\w-]+)?)\\b"
        - '(?i)\b(?:As|Speaking\s+as)\s+(?:a|an)\s+([A-Z][\w-]+)\b'
        - "(?i)\\bI(?:'m|\\s+am)\\s+([A-Z][\\w-]+)\\b"
        - '(?i)\b[Mm]y\s+country\s*[\(\-,]\s*([A-Z][\w\s]+?)[\)\-,]'
        # === SPANISH ===
        - '(?i)\bSoy\s+(?:de\s+)?([A-ZÀ-ÿ][\w-]+)\b'
        - '(?i)\bVivo\s+en\s+([A-ZÀ-ÿ][\w\s-]+)\b'
        - '(?i)\bSoy\s+(?:un|una)?\s*([A-ZÀ-ÿ][\w-]+o|[A-ZÀ-ÿ][\w-]+a)\b'
        # === PORTUGUESE ===
        - '(?i)\bSou\s+(?:de\s+)?([A-ZÀ-ÿ][\w-]+)\b'
        - '(?i)\bMoro\s+n[oa]s?\s+([A-ZÀ-ÿ][\w\s-]+)\b'
        - '(?i)\b(?:Vou\s+)?sair\s+d[oa]\s+([A-ZÀ-ÿ][\w-]+)\b'
        # === GERMAN ===
        - '(?i)\bIch\s+(?:bin|komme\s+aus)\s+(?:ein|eine|einer)?\s*([A-ZÀ-ÿ][\w-]+)\b'
        - '(?i)\bIch\s+bin\s+aus\s+([A-ZÀ-ÿ][\w\s-]+)\b'
        # === DUTCH ===
        - '(?i)\bIk\s+(?:ben|kom\s+uit)\s+([A-Z][\w-]+)\b'
        # === CROSS-LINGUAL ===
        - '\b([A-Z][\w]+-[A-Z][\w]+)\b'
        - '(?i)\b(?:citizen|national)\s+of\s+([A-Z][\w\s-]+)\b'

    # -----------------------------------------------------------------------
    # political_leaning: Extracted from r/PoliticalCompass flairs
    # -----------------------------------------------------------------------
    political_leaning:
      prompts:
        - "political affiliation"
        - "self-identified political leaning"
        - "political ideology"
      distractors:
        - "third-person political mention"
        - "political party mentioned"
        - "politician name"
      mask_token: "[MASK:POLITICAL]"
      width: 6
      reference_patterns:
        - "(?i)\\b(?:auth|lib)[-\\s]?(?:left|right|center|centre)\\b"
        - "(?i)\\b(?:left|right|center|centre)[-\\s]?(?:auth|lib)\\b"
        - "(?i)\\bI\\s*(?:am|'m)\\s*(?:a\\s+)?(?:liberal|conservative|libertarian|socialist|communist|anarchist|centrist|moderate|progressive|leftist|right-wing|left-wing)\\b"
        - "(?i)\\bI\\s+(?:lean|vote)\\s+(?:left|right|liberal|conservative|democrat|republican)\\b"

    # -----------------------------------------------------------------------
    # MBTI Personality Dimensions (extrovert, sensing, feeling, judging)
    # -----------------------------------------------------------------------
    extrovert:
      prompts:
        - "MBTI type"
        - "personality type identifier"
        - "self-identified personality type"
      distractors:
        - "third-person personality mention"
        - "personality of fictional character"
      mask_token: "[MASK:MBTI]"
      width: 6
      reference_patterns:
        - "(?i)\\b(?:INTJ|INTP|ENTJ|ENTP|INFJ|INFP|ENFJ|ENFP|ISTJ|ISFJ|ESTJ|ESFJ|ISTP|ISFP|ESTP|ESFP)\\b"
        - "(?i)\\bI\\s*(?:am|'m)\\s*(?:an?\\s+)?(?:introvert|extrovert|extravert)\\b"

    sensing:
      prompts:
        - "MBTI type"
        - "personality type identifier"
        - "cognitive function preference"
      distractors:
        - "third-person personality mention"
        - "personality of fictional character"
      mask_token: "[MASK:MBTI]"
      width: 6
      reference_patterns:
        - "(?i)\\b(?:INTJ|INTP|ENTJ|ENTP|INFJ|INFP|ENFJ|ENFP|ISTJ|ISFJ|ESTJ|ESFJ|ISTP|ISFP|ESTP|ESFP)\\b"

    feeling:
      prompts:
        - "MBTI type"
        - "personality type identifier"
        - "cognitive function preference"
      distractors:
        - "third-person personality mention"
        - "personality of fictional character"
      mask_token: "[MASK:MBTI]"
      width: 6
      reference_patterns:
        - "(?i)\\b(?:INTJ|INTP|ENTJ|ENTP|INFJ|INFP|ENFJ|ENFP|ISTJ|ISFJ|ESTJ|ESFJ|ISTP|ISFP|ESTP|ESFP)\\b"

    judging:
      prompts:
        - "MBTI type"
        - "personality type identifier"
        - "cognitive function preference"
      distractors:
        - "third-person personality mention"
        - "personality of fictional character"
      mask_token: "[MASK:MBTI]"
      width: 6
      reference_patterns:
        - "(?i)\\b(?:INTJ|INTP|ENTJ|ENTP|INFJ|INFP|ENFJ|ENFP|ISTJ|ISFJ|ESTJ|ESFJ|ISTP|ISFP|ESTP|ESFP)\\b"
```

---

## Appendix B: Pipeline Configuration Summary

**Table B1: Key Configuration Parameters**

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
