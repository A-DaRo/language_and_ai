# Advanced Project Planning: Neural Stylometry and Author Profiling

| Metadata | Value |
| :--- | :--- |
| **Document Type** | Advanced Project Planning |
| **Version** | 1.0 |
| **Last Updated** | 2025-12-05 |
| **Status** | Draft |
| **Related Documents** | [interim_project_analysis.md](./interim_project_analysis.md), [interim_assignment.md](../interim_assignment.md), [SOBR.md](../SOBR.md) |

---

## Index

- Index (this section)
- Executive Summary
- Part I: Foundation Phase — Data Pollution Detection and Mitigation
  - 1.1 Phase Overview
  - 1.2 Refined Methodology
  - 1.3 Deliverables
- Part II: Advanced Neural Approaches to Stylometric Classification
  - 2.1 Project D: Attention-Based Stylometric Feature Extraction
  - 2.2 Project E: Contrastive Learning for Domain-Invariant Author Representations
  - 2.3 Project F: Transformer Fine-Tuning with Pollution-Aware Training Objectives
  - 2.4 Project G: Multi-Task Learning for Joint Attribute Prediction
- Part III: Beyond Course Material — Advanced Topics and Extensions
  - 3.1 Adversarial Stylometry and Privacy-Preserving NLP
  - 3.2 Explainability in Author Profiling
  - 3.3 Cross-Lingual and Multilingual Stylometry
  - 3.4 Temporal Dynamics in Writing Style
- Part IV: Literature Research and Reading Suggestions
  - 4.1 Foundational Works in Computational Stylometry
  - 4.2 Neural Author Profiling
  - 4.3 Bias and Fairness in NLP
  - 4.4 Privacy-Preserving NLP and Adversarial Stylometry
  - 4.5 Interpretability and Explainability
  - 4.6 Advanced Transformer Architectures
- Part V: Recommended Research Trajectories
  - 5.1 Trajectory A: From Pollution Detection to Robust Classification
  - 5.2 Trajectory B: From Feature Analysis to Interpretable Neural Profiling
  - 5.3 Trajectory C: From Profiling to Privacy Protection
- References

---

## Executive Summary

This document extends the preliminary project planning sketches from the interim project analysis, developing a comprehensive research trajectory that progresses from foundational data quality work to advanced neural approaches for stylometric author profiling. The planning is structured as a multi-phase research programme, beginning with Project A (Data Pollution Detection and Mitigation) as the essential foundational step, and progressively introducing sophisticated neural architectures, training paradigms, and evaluation methodologies.

The document proposes four advanced projects beyond the initial sketches: (D) Attention-based stylometric feature extraction using Transformer architectures; (E) Contrastive learning approaches for learning domain-invariant author representations; (F) Pollution-aware fine-tuning objectives for pre-trained language models; and (G) Multi-task learning frameworks for joint prediction of correlated author attributes. Each project builds upon the data quality foundations established in Phase I and leverages progressively more sophisticated techniques from deep learning and modern NLP.

Beyond the scope of course materials, this document explores cutting-edge research directions including adversarial stylometry, explainability in author profiling, cross-lingual stylometry, and temporal dynamics in writing style. A comprehensive literature review provides reading suggestions organized by topic, ranging from foundational works in computational stylometry to recent advances in privacy-preserving NLP and interpretable machine learning.

---

# Part I: Foundation Phase — Data Pollution Detection and Mitigation

---

## 1.1 Phase Overview

Project A from the initial analysis serves as the indispensable foundation for all subsequent work. The rationale is straightforward: any classifier trained on contaminated data—regardless of architectural sophistication—will exploit pollution signals rather than genuine stylometric features. Neural networks, with their capacity to memorize training patterns, are particularly susceptible to this failure mode. Therefore, rigorous data quality assessment must precede model development.

**Core Principle**: A Transformer achieving 95% accuracy on polluted data provides less scientific value than a logistic regression achieving 65% on clean data. The former measures shortcut exploitation; the latter measures stylometric signal strength.

**Phase Objectives**:
1. Develop comprehensive pollution detection pipelines for all five SOBR attributes
2. Quantify the proportion of classification signal attributable to pollution versus style
3. Produce cleaned dataset variants for subsequent neural experiments
4. Establish baseline performance bounds on genuinely stylistic classification

## 1.2 Refined Methodology

### 1.2.1 Extended Pollution Pattern Taxonomy

Beyond the basic regex patterns outlined in Project A, a comprehensive detection system should address:

**Level 1: Explicit Self-Reports**
- Age-gender format patterns: `\(([MF])\s*(\d{2})\)`, `\[(\d{2})([MF])\]`
- Natural language declarations: `I('m| am) (a\s+)?(\d{1,2})(-|\s)?(year|yr|y\.?o\.?|yo)`
- Relative age statements: `As a (teenager|twentysomething|millennial|boomer)`
- Gender declarations: `I('m| am) (a\s+)?(man|woman|male|female|guy|girl|dude|lady)`

**Level 2: Implicit Attribute Markers**
- Demographic vocabulary: generational slang (zoomer/boomer terms), gendered interests
- Temporal references: "When I was in high school in 1998" → birth year inference
- Cultural references: media consumption patterns indicative of age cohorts

**Level 3: Domain-Specific Contamination**
- Subreddit-specific vocabulary (r/politics terminology, MBTI jargon)
- Community norms and memes (recognizable phrases, formatting conventions)
- Meta-discussion of attributes ("As an INTJ, I think...")

### 1.2.2 Detection Pipeline Architecture

```
Input Corpus → Preprocessing → Pattern Detection → Annotation → Filtering
                    ↓                  ↓               ↓           ↓
              Tokenization      Regex Engine      Pollution    Clean/Redact
              Case Norm         NER Tagging       Scores       Variants
                                Dependency Parse
```

**Key Components**:
1. **Multi-level regex engine**: Hierarchical patterns with confidence weighting
2. **Named Entity Recognition**: Detect demographic entities (nationalities, political affiliations)
3. **Dependency parsing**: Identify self-referential constructions ("I am a [ATTR]")
4. **Contextual validation**: Distinguish genuine self-reports from quotations or hypotheticals

### 1.2.3 Quantification Metrics

**Pollution Prevalence Rate (PPR)**:
$$\text{PPR}_a = \frac{|\{d \in D : \text{has\_pollution}(d, a)\}|}{|D|}$$

Where $a$ denotes an attribute and $D$ the document collection.

**Signal Decomposition**:
$$\text{Accuracy}_{\text{total}} = \text{Accuracy}_{\text{pollution}} + \text{Accuracy}_{\text{style}} + \text{Accuracy}_{\text{interaction}}$$

Estimate components through:
- $\text{Accuracy}_{\text{pollution}}$: Train on pollution features only
- $\text{Accuracy}_{\text{style}}$: Train on cleaned data
- $\text{Accuracy}_{\text{interaction}}$: Residual after controlling for both

## 1.3 Deliverables

| Deliverable | Description | Format |
|-------------|-------------|--------|
| Pollution Detection Toolkit | Modular Python package for pattern detection | GitHub repository |
| Annotated Corpus | SOBR with pollution annotations per post | JSON/Parquet |
| Dataset Variants | SOBR-Clean, SOBR-Redacted, SOBR-NoDomain | Distributed files |
| Baseline Benchmarks | Performance comparison across contamination levels | Reproducible experiments |
| Technical Report | Documentation of patterns, precision/recall, coverage | Markdown/PDF |

---

# Part II: Advanced Neural Approaches to Stylometric Classification

---

## 2.1 Project D: Attention-Based Stylometric Feature Extraction

| Metadata | Value |
| :--- | :--- |
| **Project Title** | Learning Where to Look: Attention Mechanisms for Stylometric Feature Discovery |
| **Primary Focus** | Interpretable neural stylometry using attention visualization |
| **Core Hypothesis** | Self-attention weights in fine-tuned Transformers will concentrate on function words, syntactic markers, and punctuation patterns rather than content words, indicating learned stylometric focus |
| **Required Foundation** | Completed Phase I (clean data), Week 6–7 materials |

### Executive Summary

This project investigates whether Transformer attention mechanisms, when trained for author profiling on cleaned data, learn to attend to genuinely stylometric features. By analyzing attention patterns across layers and heads, we can both improve interpretability and validate that models exploit style rather than content. The hypothesis predicts that successful stylometric classification will produce attention distributions concentrated on function words (pronouns, prepositions, conjunctions) and structural markers rather than topical vocabulary.

### Research Questions and Hypotheses

**RQ1**: Do attention patterns in author profiling models differ from those in content-focused tasks (e.g., topic classification)?

**H1**: Attention entropy (measured across token positions) will be higher for author profiling than topic classification, reflecting distributed focus on stylistic elements rather than concentrated focus on keywords.

**RQ2**: Which attention heads specialize in stylometric features?

**H2**: Specific attention heads (identifiable through probing) will show strong correlation with linguistic features known to encode style (function word ratios, punctuation patterns, sentence length distributions).

**RQ3**: Can attention-based feature extraction improve robustness to domain shift?

**H3**: Classifiers using attention-extracted features will show smaller performance degradation on held-out subreddits compared to standard fine-tuned models.

### Methodology

#### 3.1 Model Architecture

**Base Model**: RoBERTa-base (125M parameters) or BERT-base, selected for interpretability

**Fine-Tuning Setup**:
- Classification head on [CLS] token representation
- Binary/multi-class cross-entropy loss per attribute
- Training on SOBR-Clean (Phase I output)
- Early stopping on validation macro-F1

**Attention Extraction**:
```python
# Pseudocode for attention extraction
outputs = model(input_ids, attention_mask, output_attentions=True)
attentions = outputs.attentions  # Tuple of (batch, heads, seq, seq)
cls_attentions = [layer[:, :, 0, :] for layer in attentions]  # [CLS] attending to all tokens
```

#### 3.2 Attention Analysis Pipeline

**Step 1: Aggregate Attention by Token Type**

For each head $h$ in layer $l$, compute:
$$\bar{a}_{h,l}^{\text{func}} = \frac{1}{|T_{\text{func}}|} \sum_{t \in T_{\text{func}}} a_{h,l}([\text{CLS}], t)$$

Where $T_{\text{func}}$ denotes function word tokens. Similarly for content words, punctuation, etc.

**Step 2: Head Specialization Score**

Define specialization as the ratio of attention to stylistic versus content features:
$$S_{h,l} = \frac{\bar{a}_{h,l}^{\text{func}} + \bar{a}_{h,l}^{\text{punct}}}{\bar{a}_{h,l}^{\text{content}}}$$

Heads with $S_{h,l} > 1$ are classified as "stylometric heads."

**Step 3: Probing Classifiers**

Train linear probes on attention-weighted representations to predict:
- Function word ratios
- Average sentence length
- Punctuation density
- Type-token ratio (TTR)

High probing accuracy indicates the representation encodes these features.

#### 3.3 Attention-Guided Feature Extraction

**Attention-Weighted Vocabulary Features**:
Rather than uniform TF-IDF, weight features by learned attention:
$$\text{TF-IDF}_{\text{att}}(w, d) = \text{TF-IDF}(w, d) \cdot \bar{a}(w)$$

Where $\bar{a}(w)$ is the average attention the model assigns to word $w$ across the corpus.

**Stylometric Head Ensemble**:
Extract representations from only the top-$k$ stylometric heads (by $S_{h,l}$), concatenate, and train a lightweight classifier.

### Evaluation

| Metric | Purpose |
|--------|---------|
| Macro F1 | Primary classification performance |
| Attention entropy | Measure of attention dispersion |
| Head specialization distribution | Identification of stylometric heads |
| Probing accuracy | Validation that heads encode linguistic features |
| Cross-domain transfer | Performance on held-out subreddits |

### Expected Outcomes

1. **Attention pattern catalog**: Visualization of typical attention patterns for successful vs. unsuccessful classifications
2. **Head taxonomy**: Classification of attention heads into stylometric, content-focused, and mixed types
3. **Improved interpretability**: Explanations of model predictions grounded in linguistic features
4. **Transferable features**: Attention-guided representations for domain-robust classification

### Literature Connections

- Clark et al. (2019): What does BERT look at? An analysis of BERT's attention
- Vig & Belinkov (2019): Analyzing the structure of attention in a Transformer language model
- Rogers et al. (2020): A primer in BERTology: What we know about how BERT works

---

## 2.2 Project E: Contrastive Learning for Domain-Invariant Author Representations

| Metadata | Value |
| :--- | :--- |
| **Project Title** | Style, Not Substance: Contrastive Learning for Domain-Invariant Author Embeddings |
| **Primary Focus** | Representation learning that disentangles style from content |
| **Core Hypothesis** | Contrastive objectives that push together same-author instances across domains while separating different-author instances will learn representations robust to topical variation |
| **Required Foundation** | Completed Phase I, Project D insights, familiarity with contrastive learning |

### Executive Summary

A fundamental challenge in stylometric classification is domain shift: models trained on one set of topics (subreddits) may fail when evaluated on different topics, not because writing style has changed but because the model has learned content-based shortcuts. This project proposes a contrastive learning approach that explicitly encourages the model to learn author representations invariant to domain/topic while preserving stylistic distinctiveness. By constructing positive pairs (same author, different subreddits) and negative pairs (different authors, same subreddit), the training signal encourages domain-invariant, author-discriminative representations.

### Research Questions and Hypotheses

**RQ1**: Can contrastive pre-training improve cross-domain generalization in author profiling?

**H1**: Models pre-trained with cross-domain contrastive objectives will exhibit <10% performance degradation when transferred to held-out subreddits, compared to >25% for standard fine-tuned models.

**RQ2**: Do contrastive representations encode domain information?

**H2**: Linear probes trained to predict subreddit from contrastive representations will achieve near-chance accuracy, while probes for author attributes will significantly exceed chance.

**RQ3**: What is the optimal contrastive objective for stylometry?

**H3**: Hard negative mining (same subreddit, similar writing style, different author) will produce more discriminative representations than random negative sampling.

### Methodology

#### 3.1 Contrastive Objective Formulation

**Standard InfoNCE Loss**:
$$\mathcal{L}_{\text{NCE}} = -\log \frac{\exp(\text{sim}(z_i, z_i^+) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(z_i, z_j^-) / \tau)}$$

Where $z_i$ is the anchor representation, $z_i^+$ a positive (same author), and $z_j^-$ negatives (different authors).

**Domain-Adversarial Contrastive Loss**:
Incorporate a domain discriminator with gradient reversal:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NCE}} - \lambda \mathcal{L}_{\text{domain}}$$

Where $\mathcal{L}_{\text{domain}}$ is the domain classification loss (reversed gradients push representations toward domain invariance).

#### 3.2 Pair Construction Strategy

| Pair Type | Construction | Purpose |
|-----------|--------------|---------|
| Positive | Same author, different subreddits | Force domain invariance |
| Easy Negative | Random different author | Establish baseline separation |
| Hard Negative | Different author, same subreddit | Prevent content-based shortcuts |
| Semi-Hard Negative | Different author, similar style metrics | Push beyond surface features |

**Implementation Note**: Hard negatives require style similarity estimation. Use function word distributions, punctuation patterns, and sentence length distributions as cheap proxy metrics.

#### 3.3 Multi-Stage Training

**Stage 1: Contrastive Pre-Training**
- Encoder: RoBERTa/BERT backbone
- Projection head: 2-layer MLP → 256-dim embedding
- Training: 100K contrastive pairs per attribute
- Output: Pre-trained encoder weights

**Stage 2: Downstream Classification**
- Freeze or fine-tune encoder
- Add classification head
- Train on labeled attribute data
- Compare frozen vs. fine-tuned performance

**Stage 3: Zero-Shot Domain Transfer**
- Evaluate on entirely held-out subreddits
- No adaptation, pure transfer

### Evaluation Protocol

| Condition | Description | Metric |
|-----------|-------------|--------|
| In-domain | Standard train/test split | Macro F1 |
| Cross-domain | Train on subset of subreddits, test on others | Transfer F1 |
| Domain probe | Linear probe for subreddit prediction | Accuracy (lower = better) |
| Attribute probe | Linear probe for author attributes | Accuracy (higher = better) |

### Expected Outcomes

1. **Domain-invariant representations**: Embeddings that cluster by author rather than subreddit
2. **Transfer learning improvements**: Reduced performance gap between in-domain and cross-domain evaluation
3. **Disentanglement metrics**: Quantified separation of style and content in learned representations
4. **Pre-trained models**: Publicly releasable checkpoints for stylometric research

### Literature Connections

- Khosla et al. (2020): Supervised contrastive learning
- Chen et al. (2020): A simple framework for contrastive learning of visual representations
- Ganin et al. (2016): Domain-adversarial training of neural networks
- Wegmann et al. (2022): Author embeddings for authorship attribution

---

## 2.3 Project F: Transformer Fine-Tuning with Pollution-Aware Training Objectives

| Metadata | Value |
| :--- | :--- |
| **Project Title** | Teaching Models to Ignore Shortcuts: Pollution-Aware Training for Author Profiling |
| **Primary Focus** | Training objectives that explicitly discourage pollution exploitation |
| **Core Hypothesis** | Models trained with auxiliary losses penalizing reliance on pollution patterns will generalize better to clean test data than standard fine-tuned models |
| **Required Foundation** | Completed Phase I (with pollution annotations), Projects D–E insights |

### Executive Summary

Even with cleaned training data, pre-trained language models may retain knowledge of pollution patterns from their pre-training corpora (Reddit is extensively present in Common Crawl, etc.). This project develops training objectives that explicitly discourage the model from relying on pollution patterns, even when they appear in inputs. The approach combines standard classification loss with auxiliary losses that penalize high confidence when pollution is present or reward correct predictions on explicitly polluted inputs with masked pollution regions.

### Research Questions and Hypotheses

**RQ1**: Do standard fine-tuned models exploit residual pollution knowledge from pre-training?

**H1**: Fine-tuned models will show significantly higher confidence on polluted test examples compared to clean examples, indicating exploitation of pre-trained pollution associations.

**RQ2**: Can pollution-aware training objectives improve robustness?

**H2**: Models trained with pollution-aware losses will show <5% confidence difference between polluted and clean test examples.

**RQ3**: What is the optimal strategy for pollution-aware training?

**H3**: Multi-task training with both classification and pollution detection objectives will outperform single-objective approaches.

### Methodology

#### 3.1 Training Objective Design

**Baseline**: Standard cross-entropy loss on cleaned data:
$$\mathcal{L}_{\text{CE}} = -\sum_{i} y_i \log \hat{y}_i$$

**Pollution Penalty Loss**:
When pollution is present in input $x$ (even if from augmentation), penalize confident predictions:
$$\mathcal{L}_{\text{poll}} = \mathbb{1}[\text{polluted}(x)] \cdot \max(0, \max_c(\hat{y}_c) - \tau_{\text{poll}})$$

Where $\tau_{\text{poll}}$ is a confidence threshold (e.g., 0.6). This discourages overconfidence on polluted inputs.

**Masked Pollution Consistency Loss**:
For polluted inputs, ensure predictions are consistent when pollution is masked:
$$\mathcal{L}_{\text{consist}} = D_{\text{KL}}(\hat{y}(x) \| \hat{y}(x_{\text{masked}}))$$

Where $x_{\text{masked}}$ replaces pollution patterns with [MASK] tokens.

**Combined Objective**:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_1 \mathcal{L}_{\text{poll}} + \lambda_2 \mathcal{L}_{\text{consist}}$$

#### 3.2 Data Augmentation for Robustness

**Pollution Injection**: Randomly inject synthetic pollution patterns into clean training examples:
- Age/gender formats: "(M35)", "[25F]"
- Self-reports: "I'm a [ATTR]" with ground-truth or counterfactual labels
- Domain markers: Subreddit-specific vocabulary

**Counterfactual Pollution**: Inject pollution patterns that contradict the ground-truth label. Models must learn to ignore these.

**Curriculum Strategy**:
1. Epoch 1–3: Clean data only
2. Epoch 4–6: 10% polluted examples
3. Epoch 7–10: 30% polluted examples (including counterfactual)

#### 3.3 Multi-Task Framework

Train jointly on:
1. **Primary task**: Author attribute prediction
2. **Auxiliary task 1**: Pollution detection (binary: does input contain pollution?)
3. **Auxiliary task 2**: Pollution type classification (which attribute is polluted?)

**Gradient manipulation**: Reverse gradients from auxiliary tasks to prevent pollution feature exploitation:
$$\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}_{\text{CE}} + \beta \nabla_\theta \mathcal{L}_{\text{aux}}$$

### Evaluation

| Metric | Description |
|--------|-------------|
| Clean accuracy | Performance on pollution-free test data |
| Polluted accuracy | Performance on test data with pollution (should match clean) |
| Counterfactual accuracy | Performance when pollution contradicts label |
| Confidence gap | Difference in max confidence between polluted/clean |
| Pollution detection AUC | Auxiliary task performance (diagnostic) |

### Expected Outcomes

1. **Robust classifiers**: Models that maintain performance regardless of pollution presence
2. **Counterfactual resilience**: Correct predictions even when pollution signals contradict true labels
3. **Training recipes**: Best practices for pollution-aware training hyperparameters
4. **Ablation analysis**: Contribution of each loss component to robustness

### Literature Connections

- Utama et al. (2020): Mind the trade-off: Debiasing NLU models without degrading the in-distribution performance
- Clark et al. (2019): Don't take the easy way out: Ensemble based methods for avoiding known dataset biases
- Sagawa et al. (2020): Distributionally robust neural networks for group shifts

---

## 2.4 Project G: Multi-Task Learning for Joint Attribute Prediction

| Metadata | Value |
| :--- | :--- |
| **Project Title** | The Whole Author: Multi-Task Learning for Holistic Author Profiling |
| **Primary Focus** | Leveraging attribute correlations for improved prediction |
| **Core Hypothesis** | Joint prediction of correlated attributes (e.g., age and political leaning) will improve performance on individual tasks compared to single-task models, particularly for attributes with limited training data |
| **Required Foundation** | Completed Phase I, understanding of multi-task learning |

### Executive Summary

Author attributes are not independent: age correlates with language use patterns, political views often cluster with demographic factors, and personality dimensions may interact with communication style. This project develops a multi-task learning framework that predicts all five SOBR attributes jointly, sharing representations in lower layers while using attribute-specific heads. The hypothesis is that shared representations will capture general stylometric features useful across tasks, while task-specific heads learn attribute-discriminative patterns.

### Research Questions and Hypotheses

**RQ1**: Do shared representations improve low-resource attribute prediction?

**H1**: Multi-task models will show >10% improvement on attributes with limited positive examples (e.g., minority political leanings, rare nationalities) compared to single-task models.

**RQ2**: Which layers should be shared vs. task-specific?

**H2**: Optimal performance will be achieved with shared lower layers (1–9) and task-specific upper layers (10–12), reflecting the progression from generic to task-specific representations in Transformers.

**RQ3**: How do attribute correlations affect joint prediction?

**H3**: Task pairs with positive correlation (same authors tend to have certain attribute combinations) will show larger improvements from joint training than uncorrelated pairs.

### Methodology

#### 3.1 Architecture

**Shared Encoder**: BERT/RoBERTa layers 1–$k$ (shared across tasks)

**Task-Specific Towers**: Layers $k+1$ to $L$ with separate parameters per task

**Task Heads**:
- Age: 5-way softmax (age brackets)
- Gender: Binary sigmoid
- Nationality: Multi-class softmax (top-$N$ countries)
- Personality: 4 binary sigmoids (MBTI dimensions)
- Political: 3-way softmax

#### 3.2 Training Strategy

**Loss Combination**:
$$\mathcal{L}_{\text{MTL}} = \sum_{t \in \text{tasks}} w_t \mathcal{L}_t$$

**Dynamic Weight Adjustment**:
Use uncertainty weighting (Kendall et al., 2018):
$$w_t = \frac{1}{2\sigma_t^2}$$

Where $\sigma_t$ is a learned task-specific uncertainty.

**Alternating vs. Joint Updates**:
- **Joint**: All tasks updated every batch
- **Alternating**: Cycle through tasks per batch
- **Proportional**: Sample batches proportional to task difficulty

#### 3.3 Handling Missing Labels

SOBR has very low multi-label co-occurrence (Table 2 in SOBR paper: only 0.02% have all five attributes). Strategies:

1. **Masked losses**: Only compute loss for available labels
2. **Pseudo-labeling**: Use confident single-task predictions as soft labels
3. **Attribute clustering**: Group authors with similar partial attribute profiles

### Evaluation

| Metric | Description |
|--------|-------------|
| Per-task F1 | Individual attribute performance |
| MTL gain | Improvement over single-task baseline |
| Transfer ratio | Performance on rare classes with MTL vs. single-task |
| Correlation exploitation | Performance on attribute pairs by correlation strength |

### Expected Outcomes

1. **Multi-task models**: Trained models predicting all five attributes jointly
2. **Layer sharing analysis**: Optimal sharing configuration for stylometric MTL
3. **Attribute correlation insights**: Quantified relationships between SOBR attributes
4. **Low-resource improvements**: Demonstrated gains for minority classes

### Literature Connections

- Caruana (1997): Multitask learning
- Liu et al. (2019): Multi-task deep neural networks for natural language understanding
- Kendall et al. (2018): Multi-task learning using uncertainty to weigh losses

---

# Part III: Beyond Course Material — Advanced Topics and Extensions

This section explores research directions that extend significantly beyond the course syllabus, providing pathways for advanced investigation and connection to frontier research in NLP.

---

## 3.1 Adversarial Stylometry and Privacy-Preserving NLP

### Overview

Adversarial stylometry represents the defensive counterpart to author profiling: developing techniques that allow individuals to protect their privacy by obfuscating identifying stylistic features. This is a rapidly evolving research area with direct implications for online safety, whistleblower protection, and resistance to surveillance.

### Key Research Directions

**Style Transfer for Anonymization**:
Modify text to adopt the stylistic characteristics of a different author demographic while preserving semantic content. Challenges include maintaining fluency, factual accuracy, and avoiding style artifacts that themselves become identifiable.

**Lexical Substitution Attacks**:
Emmery et al. (2021) demonstrated that targeted substitution of words with synonyms or paraphrases can significantly reduce profiling accuracy. The challenge lies in finding substitutions that are semantically appropriate, stylistically neutral, and imperceptible to human readers.

**Perturbation Detection**:
Concurrent research aims to detect when text has been adversarially modified. This creates an arms race between obfuscation and detection methods.

### Potential Project Extensions

- Develop obfuscation methods specifically targeting SOBR attributes
- Evaluate transferability: does obfuscating gender also affect age detection?
- Measure trade-off between obfuscation effectiveness and text quality
- Design human evaluation protocols for assessing naturalness of obfuscated text

### Key Papers

- Emmery et al. (2021): Adversarial stylometry in the wild: Transferable lexical substitution attacks on author profiling
- Shetty et al. (2018): A4NT: Author attribute anonymity by adversarial training of neural machine translation
- Brennan et al. (2012): Adversarial stylometry: Circumventing authorship recognition

---

## 3.2 Explainability in Author Profiling

### Overview

As author profiling systems are deployed in consequential settings (content moderation, fraud detection, security screening), the need for explainable predictions becomes critical. Users and auditors must understand why a system classified an author in a particular way, both to validate correctness and to identify potential biases.

### Key Research Directions

**Feature Attribution Methods**:
Techniques like LIME, SHAP, and Integrated Gradients can identify which input features most influenced a prediction. For stylometry, this means identifying specific words, phrases, or syntactic patterns the model relies upon.

**Concept-Based Explanations**:
Rather than token-level attribution, explain predictions in terms of interpretable linguistic concepts: "This author is classified as older because of high formality, low emoji usage, and preference for complex sentence structures."

**Counterfactual Explanations**:
"What minimal change would flip the prediction?" For author profiling: "If the author used fewer exclamation points, the model would classify them as female instead of male."

### Potential Project Extensions

- Develop linguistically-grounded explanations for SOBR predictions
- Compare explainability of different model architectures (logistic regression vs. Transformers)
- Evaluate whether explanations reveal problematic biases
- Design user studies to assess explanation utility

### Key Papers

- Ribeiro et al. (2016): "Why should I trust you?": Explaining the predictions of any classifier
- Lundberg & Lee (2017): A unified approach to interpreting model predictions
- Kim et al. (2018): Interpretability beyond feature attribution

---

## 3.3 Cross-Lingual and Multilingual Stylometry

### Overview

Most computational stylometry research focuses on English, yet writing style is fundamentally language-dependent. Cross-lingual stylometry investigates whether stylistic features transfer across languages and how multilingual models can be trained for author profiling.

### Key Research Directions

**Language-Invariant Style Features**:
Some stylistic features (punctuation patterns, sentence length distributions, paragraph structure) may transfer across languages. Identifying and leveraging these could enable zero-shot cross-lingual profiling.

**Multilingual Transformers for Stylometry**:
Models like mBERT and XLM-R provide multilingual representations. Can these be fine-tuned for author profiling in one language and transferred to another?

**Non-Native Speaker Detection**:
Identifying non-native speakers involves detecting L1 interference patterns—this is closely related to nationality detection in SOBR but focuses on linguistic transfer rather than cultural content.

### Potential Project Extensions

- Evaluate mBERT/XLM-R on SOBR nationality task with non-European languages
- Develop language-agnostic stylometric features
- Study how nationality detection conflates language proficiency with cultural identity

### Key Papers

- Rabinovich et al. (2018): Native language cognate effects on second language lexical choice
- Kramp et al. (2023): Native language identification with Big Bird embeddings
- Pires et al. (2019): How multilingual is multilingual BERT?

---

## 3.4 Temporal Dynamics in Writing Style

### Overview

Writing style is not static. Individuals' linguistic patterns evolve over time due to aging, cultural shifts, platform adaptation, and deliberate stylistic changes. Understanding temporal dynamics is crucial for longitudinal author profiling and for distinguishing genuine style evolution from data contamination.

### Key Research Directions

**Style Drift Detection**:
Identify when an author's style has changed significantly, potentially indicating account compromise, ghostwriting, or identity change.

**Age-Cohort Linguistic Evolution**:
Track how the linguistic markers of a given birth cohort change as they age, versus how new cohorts differ from older ones at the same age.

**Platform-Induced Adaptation**:
Reddit has evolved significantly since its founding. Users may adapt their style to changing platform norms, affecting the validity of models trained on historical data.

### Potential Project Extensions

- Analyze SOBR temporal metadata to study style evolution
- Develop age-cohort correction factors for temporal bias
- Model style drift as a latent variable in author profiling

### Key Papers

- Nguyen et al. (2013): How old do you think I am? A study of language and age in Twitter
- Danescu-Niculescu-Mizil et al. (2013): No country for old members: User lifecycle and linguistic change in online communities
- Rosenthal & McKeown (2011): Age prediction in blogs: A study of style, content, and online behavior

---

# Part IV: Literature Research and Reading Suggestions

This section provides a curated bibliography organized by topic, ranging from foundational works to recent advances. Each entry includes a brief annotation explaining its relevance to the project.

---

## 4.1 Foundational Works in Computational Stylometry

| Reference | Annotation |
|-----------|------------|
| Schler et al. (2006). Effects of age and gender on blogging. | Seminal work establishing the feasibility of age and gender prediction from writing style. Introduces key feature categories still used today. |
| Argamon et al. (2009). Automatically profiling the author of an anonymous text. | Comprehensive overview of stylometric methods pre-neural era. Establishes function word and syntactic feature importance. |
| Koppel et al. (2009). Computational methods in authorship attribution. | Authoritative survey of classical authorship attribution techniques. Essential background for understanding feature engineering approaches. |
| Stamatatos (2009). A survey of modern authorship attribution methods. | Detailed survey covering character n-grams, vocabulary richness, and syntactic features. Complements Koppel et al. with different emphasis. |
| Juola (2006). Authorship attribution. | Introductory survey accessible to newcomers. Good starting point before diving into technical literature. |

---

## 4.2 Neural Author Profiling

| Reference | Annotation |
|-----------|------------|
| Bagnall (2015). Author identification using multi-headed recurrent neural networks. | Early application of RNNs to authorship attribution. Demonstrates character-level modeling effectiveness. |
| Ruder et al. (2016). Character-level and multi-channel convolutional neural networks for large-scale authorship attribution. | CNN approach achieving strong results. Useful comparison point for Transformer-based methods. |
| Shrestha et al. (2017). Convolutional neural networks for authorship attribution of short texts. | Addresses the challenge of limited text length. Relevant for SOBR slice-based evaluation. |
| Wegmann et al. (2022). Same author or just same topic? Towards content-independent style representations. | Directly addresses the content-style confound central to SOBR. Proposes contrastive learning approach. |
| Hay et al. (2020). Representation and classification of author attributes via attention-based neural networks. | Applies attention mechanisms to author profiling. Provides methodological foundation for Project D. |

---

## 4.3 Bias and Fairness in NLP

| Reference | Annotation |
|-----------|------------|
| Blodgett et al. (2020). Language (technology) is power: A critical survey of "bias" in NLP. | Essential critical perspective on bias definitions and measurement. Should inform ethical framing of SOBR research. |
| Bolukbasi et al. (2016). Man is to computer programmer as woman is to homemaker? Debiasing word embeddings. | Foundational work on embedding bias. Relevant for understanding bias propagation in learned representations. |
| Zhao et al. (2019). Gender bias in contextualized word embeddings. | Extends bias analysis to Transformer embeddings. Directly relevant for BERT/RoBERTa-based profiling. |
| Sun et al. (2019). Mitigating gender bias in natural language processing: Literature review. | Comprehensive survey of debiasing techniques. Useful reference for mitigation strategies. |
| Dev et al. (2022). On measures of biases and harms in NLP. | Recent framework for categorizing and measuring bias. Provides structured approach to bias evaluation. |

---

## 4.4 Privacy-Preserving NLP and Adversarial Stylometry

| Reference | Annotation |
|-----------|------------|
| Emmery et al. (2018). Style obfuscation by invariance. | Introduces invariance-based obfuscation approach. Theoretically grounded adversarial stylometry. |
| Emmery et al. (2021). Adversarial stylometry in the wild: Transferable lexical substitution attacks on author profiling. | State-of-the-art adversarial stylometry. Demonstrates practical attacks on real systems. |
| Shetty et al. (2018). A4NT: Author attribute anonymity by adversarial training of neural machine translation. | Neural style transfer for anonymization. Sophisticated approach with quality-preservation focus. |
| Potthast et al. (2018). Overview of the author obfuscation task at PAN 2018. | Shared task overview with standardized evaluation. Useful benchmark reference. |
| Brennan et al. (2012). Adversarial stylometry: Circumventing authorship recognition to preserve privacy and anonymity. | Foundational work establishing the adversarial stylometry paradigm. Essential background reading. |

---

## 4.5 Interpretability and Explainability

| Reference | Annotation |
|-----------|------------|
| Clark et al. (2019). What does BERT look at? An analysis of BERT's attention. | Empirical analysis of BERT attention patterns. Methodological foundation for Project D. |
| Rogers et al. (2020). A primer in BERTology: What we know about how BERT works. | Comprehensive survey of BERT analysis research. Essential reading for interpretable Transformer work. |
| Vig & Belinkov (2019). Analyzing the structure of attention in a Transformer language model. | Detailed attention analysis methodology. Provides tools for attention visualization and interpretation. |
| Ribeiro et al. (2016). "Why should I trust you?": Explaining the predictions of any classifier. | Introduces LIME for model-agnostic explanations. Applicable to any classifier architecture. |
| Lundberg & Lee (2017). A unified approach to interpreting model predictions. | SHAP values for unified feature attribution. Principled approach to explanation. |

---

## 4.6 Advanced Transformer Architectures

| Reference | Annotation |
|-----------|------------|
| Zaheer et al. (2020). Big Bird: Transformers for longer sequences. | Long-context Transformer used in SOBR BB-LR baseline. Essential for understanding long-document processing. |
| Beltagy et al. (2020). Longformer: The long-document Transformer. | Alternative long-context approach. Useful comparison to Big Bird. |
| Devlin et al. (2019). BERT: Pre-training of deep bidirectional Transformers for language understanding. | Foundational pre-trained Transformer paper. Essential background. |
| Liu et al. (2019). RoBERTa: A robustly optimized BERT pretraining approach. | Improved BERT pre-training. Often preferred for fine-tuning experiments. |
| Khosla et al. (2020). Supervised contrastive learning. | Contrastive learning with labels. Methodological foundation for Project E. |

---

# Part V: Recommended Research Trajectories

This section synthesizes the preceding material into coherent research trajectories, each representing a complete path from foundational work to advanced investigation.

---

## 5.1 Trajectory A: From Pollution Detection to Robust Classification

**Path**: Phase I → Project F → Project D

**Narrative**: Begin by thoroughly understanding and mitigating data pollution (Phase I). With clean data established, develop training objectives that actively discourage models from exploiting any residual pollution signals (Project F). Finally, use attention analysis (Project D) to verify that trained models attend to genuinely stylometric features rather than shortcuts.

**Key Milestones**:
1. Pollution detection toolkit with >90% recall
2. Cleaned SOBR variants with documented preprocessing
3. Pollution-aware training achieving <5% confidence gap between polluted/clean test data
4. Attention visualization showing concentration on function words and syntactic markers

**Timeline**: 8–10 weeks for complete trajectory

**Suitable For**: Students interested in data quality, training robustness, and model interpretability.

---

## 5.2 Trajectory B: From Feature Analysis to Interpretable Neural Profiling

**Path**: Project B (Feature Ablation) → Project D → Explainability Extension

**Narrative**: Start by systematically analyzing which traditional feature types (content vs. function words, character n-grams, etc.) contribute to classification (Project B). Use these insights to design attention-based neural models that explicitly focus on identified stylometric features (Project D). Extend to full explainability with concept-based explanations grounded in linguistic theory.

**Key Milestones**:
1. Feature contribution matrix across all SOBR attributes
2. Identified stylometric vs. content-based feature sets
3. Attention heads specialized for validated stylometric features
4. Linguistically-grounded explanations for model predictions

**Timeline**: 10–12 weeks for complete trajectory

**Suitable For**: Students interested in linguistics, interpretability, and the science of writing style.

---

## 5.3 Trajectory C: From Profiling to Privacy Protection

**Path**: Phase I → Project E → Adversarial Stylometry Extension

**Narrative**: Begin with data quality work (Phase I), then develop domain-invariant representations that capture true stylistic signatures (Project E). These representations serve dual purposes: enabling robust profiling for legitimate applications (fraud detection, content moderation) and informing privacy-protective obfuscation methods. The trajectory culminates in developing and evaluating obfuscation techniques that reduce profiling accuracy while maintaining text quality.

**Key Milestones**:
1. Clean data baselines for all attributes
2. Contrastive representations achieving high attribute accuracy with low domain accuracy
3. Identification of most predictive stylistic features (obfuscation targets)
4. Obfuscation methods reducing profiling accuracy by >20% with imperceptible modifications

**Timeline**: 12–14 weeks for complete trajectory

**Suitable For**: Students interested in privacy, security, and the societal implications of NLP technology.

---

# References

Argamon, S., Koppel, M., Pennebaker, J.W., & Schler, J. (2009). Automatically profiling the author of an anonymous text. *Communications of the ACM*, 52(2), 119–123.

Bagnall, D. (2015). Author identification using multi-headed recurrent neural networks. *arXiv preprint arXiv:1506.04891*.

Beltagy, I., Peters, M.E., & Cohan, A. (2020). Longformer: The long-document Transformer. *arXiv preprint arXiv:2004.05150*.

Blodgett, S.L., Barocas, S., Daumé III, H., & Wallach, H. (2020). Language (technology) is power: A critical survey of "bias" in NLP. *Proceedings of ACL 2020*, 5454–5476.

Bolukbasi, T., Chang, K.W., Zou, J.Y., Saligrama, V., & Kalai, A.T. (2016). Man is to computer programmer as woman is to homemaker? Debiasing word embeddings. *Advances in NeurIPS 2016*, 4349–4357.

Brennan, M., Afroz, S., & Greenstadt, R. (2012). Adversarial stylometry: Circumventing authorship recognition to preserve privacy and anonymity. *ACM TISSEC*, 15(3), 1–22.

Caruana, R. (1997). Multitask learning. *Machine Learning*, 28(1), 41–75.

Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. *ICML 2020*, 1597–1607.

Clark, K., Khandelwal, U., Levy, O., & Manning, C.D. (2019). What does BERT look at? An analysis of BERT's attention. *BlackboxNLP@ACL 2019*, 276–286.

Clark, C., Yatskar, M., & Zettlemoyer, L. (2019). Don't take the easy way out: Ensemble based methods for avoiding known dataset biases. *EMNLP 2019*, 4069–4082.

Danescu-Niculescu-Mizil, C., West, R., Jurafsky, D., Leskovec, J., & Potts, C. (2013). No country for old members: User lifecycle and linguistic change in online communities. *WWW 2013*, 307–318.

Dev, S., et al. (2022). On measures of biases and harms in NLP. *Findings of AACL-IJCNLP 2022*, 246–267.

Devlin, J., Chang, M.W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional Transformers for language understanding. *NAACL-HLT 2019*, 4171–4186.

Emmery, C., Kádár, Á., & Chrupała, G. (2021). Adversarial stylometry in the wild: Transferable lexical substitution attacks on author profiling. *EACL 2021*, 2388–2402.

Emmery, C., Manjavacas, E., & Chrupała, G. (2018). Style obfuscation by invariance. *COLING 2018*, 984–996.

Emmery, C., Miotto, M., Kramp, S., & Kleinberg, B. (2024). SOBR: A corpus for stylometry, obfuscation, and bias on Reddit. *LREC-COLING 2024*, 14967–14983.

Ganin, Y., et al. (2016). Domain-adversarial training of neural networks. *JMLR*, 17(59), 1–35.

Hay, T., Kramp, S., & Emmery, C. (2020). Representation and classification of author attributes via attention-based neural networks. *Working Notes of CLEF 2020*.

Juola, P. (2006). Authorship attribution. *Foundations and Trends in Information Retrieval*, 1(3), 233–334.

Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. *CVPR 2018*, 7482–7491.

Khosla, P., et al. (2020). Supervised contrastive learning. *NeurIPS 2020*.

Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., & Viegas, F. (2018). Interpretability beyond feature attribution: Quantitative testing with concept activation vectors. *ICML 2018*, 2668–2677.

Koppel, M., Schler, J., & Argamon, S. (2009). Computational methods in authorship attribution. *JASIST*, 60(1), 9–26.

Kramp, S., Cassani, G., & Emmery, C. (2023). Native language identification with Big Bird embeddings. *arXiv preprint arXiv:2309.06923*.

Liu, X., He, P., Chen, W., & Gao, J. (2019). Multi-task deep neural networks for natural language understanding. *ACL 2019*, 4487–4496.

Liu, Y., et al. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv preprint arXiv:1907.11692*.

Lundberg, S.M., & Lee, S.I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*, 4765–4774.

Nguyen, D., Smith, N.A., & Rosé, C.P. (2013). How old do you think I am? A study of language and age in Twitter. *ICWSM 2013*.

Pires, T., Schlinger, E., & Garrette, D. (2019). How multilingual is multilingual BERT? *ACL 2019*, 4996–5001.

Potthast, M., et al. (2018). Overview of the author obfuscation task at PAN 2018. *CLEF 2018 Working Notes*.

Rabinovich, E., Tsvetkov, Y., & Wintner, S. (2018). Native language cognate effects on second language lexical choice. *TACL*, 6, 329–342.

Ribeiro, M.T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *KDD 2016*, 1135–1144.

Rogers, A., Kovaleva, O., & Rumshisky, A. (2020). A primer in BERTology: What we know about how BERT works. *TACL*, 8, 842–866.

Rosenthal, S., & McKeown, K. (2011). Age prediction in blogs: A study of style, content, and online behavior in pre-and post-social media generations. *ACL 2011*, 763–772.

Ruder, S., Ghaffari, P., & Breslin, J.G. (2016). Character-level and multi-channel convolutional neural networks for large-scale authorship attribution. *arXiv preprint arXiv:1609.06686*.

Sagawa, S., et al. (2020). Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. *ICLR 2020*.

Schler, J., Koppel, M., Argamon, S., & Pennebaker, J.W. (2006). Effects of age and gender on blogging. *AAAI Spring Symposium on Computational Approaches to Analyzing Weblogs*, 199–205.

Shetty, R., Schiele, B., & Fritz, M. (2018). A4NT: Author attribute anonymity by adversarial training of neural machine translation. *USENIX Security 2018*, 1633–1650.

Shrestha, P., Sierra, S., Gonzalez, F., Montes, M., Rosso, P., & Solorio, T. (2017). Convolutional neural networks for authorship attribution of short texts. *EACL 2017*, 669–674.

Stamatatos, E. (2009). A survey of modern authorship attribution methods. *JASIST*, 60(3), 538–556.

Sun, T., et al. (2019). Mitigating gender bias in natural language processing: Literature review. *ACL 2019*, 1630–1640.

Utama, P.A., Mober, N., & Gurevych, I. (2020). Mind the trade-off: Debiasing NLU models without degrading the in-distribution performance. *ACL 2020*, 8717–8729.

Vig, J., & Belinkov, Y. (2019). Analyzing the structure of attention in a Transformer language model. *BlackboxNLP@ACL 2019*, 63–76.

Wegmann, A., Nguyen, D., & Groves, D. (2022). Same author or just same topic? Towards content-independent style representations. *RepL4NLP@ACL 2022*, 249–268.

Zaheer, M., et al. (2020). Big Bird: Transformers for longer sequences. *NeurIPS 2020*.

Zhao, J., et al. (2019). Gender bias in contextualized word embeddings. *NAACL-HLT 2019*, 629–634.
