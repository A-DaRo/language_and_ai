# Interim Assignment: Document Analysis and Project Planning

| Metadata | Value |
| :--- | :--- |
| **Document Type** | Analysis & Planning |
| **Version** | 1.0 |
| **Last Updated** | 2025-12-05 |
| **Status** | Draft |
| **Related Documents** | [interim_assignment.md](../interim_assignment.md), [SOBR.md](../SOBR.md), [course_introduction.md](../../Course_Material/Generated/course_introduction.md) |

---

## Index

- Index (this section)
- Executive Summary
- Part I: Document Analysis Report
  - 1. Interim Assignment Brief Analysis
    - 1.1 Learning Objectives
    - 1.2 Scope and Constraints
    - 1.3 Research Proposal Requirements
    - 1.4 Research Paper Requirements
  - 2. SOBR Corpus Analysis
    - 2.1 Corpus Overview and Purpose
    - 2.2 Data Collection Methodology
    - 2.3 Author Attributes and Labels
    - 2.4 Dataset Variants and Sampling Strategies
    - 2.5 Data Contamination Issues
    - 2.6 Baseline Models and Performance
    - 2.7 Research Avenues Identified by Authors
    - 2.8 Ethical Considerations
  - 3. Course Material Relevance Analysis
    - 3.1 Most Relevant Weeks
    - 3.2 Week 2: Text Preprocessing and Normalization
    - 3.3 Week 3: Classification and Language Modeling
    - 3.4 Week 4: Representation Learning
    - 3.5 Supplementary Weeks
  - 4. Synthesis: Alignment of Course Material with Assignment
- Part II: Project Planning Sketches
  - Project A: Data Pollution Detection and Mitigation via Text Normalization
  - Project B: Feature Ablation Study on Author Profiling Bias
  - Project C: Comparative Analysis of Representation Methods for Stylometric Classification
- References

---

## Executive Summary

This document presents a comprehensive analysis of the Language and AI interim assignment, the SOBR (Stylometry, Obfuscation, and Bias on Reddit) corpus, and relevant course materials. The analysis identifies Week 3 (Classification and Language Modeling) and Week 4 (Representation Learning) as the most directly applicable course content, with Week 2 (Text Preprocessing) providing essential foundational techniques for addressing the assignment's core challenge: investigating and mitigating data pollution issues in stylometric author profiling tasks.

The SOBR corpus comprises 235 million Reddit posts with distant labels for five author attributes (age, gender, nationality, personality, and political leaning), deliberately containing data contamination through in-text self-reports and in-domain posting patterns. The assignment explicitly requires students to investigate and mitigate these pollution issues rather than merely predicting labels.

Three distinct project planning sketches are proposed: (A) a text normalization approach using regular expressions and preprocessing pipelines to detect and remove polluting patterns; (B) a feature ablation study systematically analyzing which feature types encode genuine stylometric signals versus content-based shortcuts; and (C) a comparative analysis of sparse versus dense representations to evaluate which representation paradigm is more robust to data contamination. Each project addresses the assignment's dual criteria while leveraging distinct subsets of course material.

---

# Part I: Document Analysis Report

---

## 1. Interim Assignment Brief Analysis

### 1.1 Learning Objectives

The interim assignment articulates four formal learning objectives:

1. **Evaluate the suitability of natural language data sources for a data science problem** — This objective requires critical assessment of the SOBR corpus, including its biases, labeling methodology, and representativeness.

2. **Implement NLP methods to analyze and transform natural language data** — This necessitates practical application of preprocessing, feature extraction, and vectorization techniques from Weeks 1–4.

3. **Apply machine learning to language data** — Students must implement classification pipelines using algorithms covered in Week 3, potentially extending to neural approaches from Weeks 4–7.

4. **Explain the limitations of NLP techniques** — This objective mandates critical reflection on model failures, bias propagation, and the gap between benchmark performance and real-world applicability.

These objectives collectively emphasize methodological rigor, critical evaluation, and awareness of limitations—not merely achieving high classification accuracy.

### 1.2 Scope and Constraints

The assignment imposes two explicit constraints:

1. **Data Constraint**: The project must utilize the provided SOBR corpus, which focuses on stylometry and author profiling. Deviation is permitted only if the second criterion is satisfied.

2. **Differentiation Constraint**: "Solely predicting the labels is insufficient." The assignment explicitly requires investigation and mitigation of data pollution issues—pieces of text that directly encode label information (e.g., in-text mentions of gender and age).

This framing distinguishes the assignment from a standard classification task: success is measured not by maximizing accuracy but by demonstrating understanding of data quality issues and proposing principled solutions.

### 1.3 Research Proposal Requirements

The formative research proposal (1–2 pages, preference for 1 page) requires:

| Section | Content | Notes |
|---------|---------|-------|
| Research Questions / Hypotheses | Precise, testable claims | Framing as hypotheses strengthens the work |
| Literature Review | Minimum 3 closely related papers | Emphasize compare-and-contrast, not summaries |
| Data | Label and split description | Must align with research questions |
| Evaluation (Metrics) | Quantitative metrics | Justify departures from standard metrics |
| Models | Baselines and investigation models | Clarify how models test hypotheses |
| General Reasoning | Synthesis of above sections | Foundation for abstract |
| Progress Summary | Current status and obstacles | Enables targeted feedback |

### 1.4 Research Paper Requirements

The summative research paper (4 pages, ACL format) follows standard NLP paper structure:

- **Abstract** (~0.25 column): Context, proposal, findings, significance
- **Introduction** (0.5–1 column): Research question, motivation, contributions
- **Related Work** (0.75–1 column): Organized thematic groups, connection to own work
- **Data**: Corpus description, statistics, examples
- **Method and Experimental Setup**: Model descriptions, preprocessing, evaluation protocol
- **Results**: Quantitative outcomes, tables/figures
- **Discussion and Conclusion**: Interpretation, error analysis, broader implications

Critical evaluation criteria (per rubric):
- Quality of research question and its grounding
- Appropriateness of methodology
- Rigor of evaluation
- Quality of writing and presentation
- **Not** the magnitude of "positive" results

---

## 2. SOBR Corpus Analysis

### 2.1 Corpus Overview and Purpose

The SOBR (Stylometry, Obfuscation, and Bias on Reddit) corpus represents a large-scale distantly annotated resource for computational stylometry research. Published at LREC-COLING 2024, the corpus was developed by researchers at Tilburg University and University College London.

**Scale**: 235,630,014 labeled Reddit posts spanning two years of Reddit snapshots from Pushshift.io.

**Primary Research Goals**:
1. Enable study of stylometric profiling techniques
2. Support development of author obfuscation methods for privacy protection
3. Facilitate investigation of bias in classifiers trained on web corpora

**Dual-Use Framing**: The authors explicitly acknowledge computational stylometry as dual-use research—profiling techniques enable both beneficial applications (fraud detection, content moderation) and harmful applications (surveillance, demographic targeting). The corpus is intended to support defensive research: understanding profiling capabilities to develop countermeasures.

### 2.2 Data Collection Methodology

**Source Platform**: Reddit, a content discussion website with over 52 million users, organized into topical subreddits. The platform's structure provides three labeling mechanisms:

1. **Flairs**: User-settable status messages displayed alongside usernames, often themed to specific subreddits
2. **Post-level self-reports**: Textual patterns like "(F34)" indicating female, age 34
3. **Subreddit membership**: Association with attribute-specific communities

**Distant Labeling Strategy**: Following precedent from Beller et al. (2014) and Emmery et al. (2017), the corpus employs distant supervision—labels are inferred from heuristic patterns rather than gold-standard annotation. This approach trades precision for scale: the pipeline runs on consumer hardware within a day, enabling collection at a scale infeasible for manual annotation.

**Collection Pipeline**:
1. Extract author IDs from attribute-relevant subreddits and self-reports
2. Retrieve all posts from labeled authors across the full two-year snapshot
3. Assign author-level labels to all their posts
4. Annotate posts with metadata indicating in-domain origin

### 2.3 Author Attributes and Labels

The corpus includes five author attributes:

| Attribute | Labeling Method | Label Space | Notes |
|-----------|-----------------|-------------|-------|
| **Age** | Regex on self-reports "(GAA)" | Birth year (continuous → binned) | Stored as inferred birth year |
| **Gender** | Regex on self-reports "(GAA)" | Binary: Male/Female | M/m → male, F/f → female |
| **Nationality** | Flairs from Europe-focused subreddits | Country-level (51+ categories) | Mapped from flair text manually |
| **Personality** | Flairs from MBTI subreddits | 4 binary dimensions (16 types) | E/I, S/N, T/F, J/P |
| **Political Leaning** | Flairs from r/PoliticalCompass | 3-way: Left/Center/Right | Economic axis only |

**Distribution Characteristics** (Figures 1–2 in SOBR paper):
- Gender: Approximately 57.3% male, 42.7% female (less skewed than general Reddit demographics)
- Age: Mean birth year ~1988, SD ~12.4 years
- Personality: Heavily skewed toward INT(J/P) types
- Political Leaning: Bell-shaped distribution centered on centrist

**Attribute Co-occurrence**: Table 2 in the SOBR paper reveals that multi-label authors are rare (~0.02% have all five attributes). This limits cross-correlation analysis and multi-label prediction experiments.

### 2.4 Dataset Variants and Sampling Strategies

The raw corpus (235M posts) is processed into structured datasets for model training:

**Slicing Strategy**: Author post histories are segmented into slices of 1,500 words each. Authors with fewer than 1,500 total words are excluded, as are excess words not fitting a complete slice. This approach:
- Standardizes instance length
- Reduces impact of scattered self-reports across an author's history
- Enables evaluation on authors with limited post history (majority have single slice)

**Sampling Variants**:

| Variant | Strategy | Purpose |
|---------|----------|---------|
| Random | 10,000 random authors | General-purpose evaluation |
| Stratified | Preserves label distribution | Maintains corpus statistics |
| Balanced | Undersampled to minority class | Addresses class imbalance |

**Instance Counts** (Table 1):
- Age: 83,748 (Random), not applicable (Balanced due to sparsity)
- Gender: 89,272 (Random), 103,047 (Stratified), 85,293 (Balanced)
- Nationality: 165,234 (Random), not applicable (Balanced)
- Personality: 326,520 (Random), 289,860 (Stratified), 180,800 (Balanced)
- Political: 114,463 (Random), 120,363 (Stratified), 97,012 (Balanced)

### 2.5 Data Contamination Issues

**This section addresses the assignment's core challenge.**

The SOBR corpus deliberately contains data contamination to enable pollution mitigation research. Three contamination sources are identified:

#### 2.5.1 In-Text Self-Reports

Posts containing explicit mentions of author attributes (e.g., "I'm a 34-year-old woman") provide trivial classification signals unrelated to stylometric properties. While age/gender regex patterns have been filtered from the current corpus version, the authors acknowledge that broader self-report patterns ("I'm a ...") remain challenging to remove completely.

**Implication**: Classifiers trained on contaminated data may learn to recognize self-report patterns rather than genuine stylistic features. Such models will fail on out-of-domain data lacking explicit self-reports.

#### 2.5.2 In-Domain Posting

Authors labeled via a specific subreddit (e.g., r/infp for personality, r/PoliticalCompass for political leaning) will have posts from those subreddits in their training data. Posts from in-domain subreddits may contain:
- Topical vocabulary correlated with labels (e.g., political terminology)
- Subreddit-specific norms and memes
- Discussion of the attribute itself

Kramp et al. (2023) demonstrated that classifying authors from the same subreddit is significantly easier than cross-domain classification, as classifiers exploit content words and idiosyncratic cues.

**Corpus Annotation**: The SOBR corpus annotates each post with `*_in_domain` flags indicating whether the post originates from a label-source subreddit. This enables filtering during preprocessing.

#### 2.5.3 Bot and Moderator Artifacts

Despite filtering for known bot accounts, repeating patterns from moderators, automated responses, and unverified bots remain in the corpus. These patterns introduce noise unrelated to author stylometry.

**Mitigation Strategies** (Section 2.3 of SOBR paper):
1. **Slicing**: Spreading posts across slices dilutes impact of isolated self-reports
2. **Author-level splitting**: Preventing author overlap between train/test sets
3. **Domain filtering**: Excluding in-domain subreddits using corpus annotations
4. **Test set curation**: Selecting authors with few slices to assess generalization

### 2.6 Baseline Models and Performance

The SOBR paper establishes three baseline models:

| Model | Architecture | Input Representation |
|-------|--------------|---------------------|
| **LR** | Logistic Regression (sklearn) | TF-IDF over n-grams (token uni/bigrams, character trigrams) |
| **fastText** | Single embedding layer + hierarchical softmax | Token uni/bigrams, embedding size 50 |
| **BB-LR** | Big Bird embeddings → Logistic Regression | Transformer [CLS] embeddings (bigbird-roberta-base) |

**Performance Summary** (Table 3, Macro $F_1$):

| Task | Majority | LR | fastText | BB-LR |
|------|----------|-----|----------|-------|
| Age | .633 | .548 | .577 | .522 |
| Gender | .706 | .797 | **.825** | .781 |
| E/I | .848 | .725 | .756 | .683 |
| S/N | .940 | .888 | .922 | .900 |
| T/F | .784 | .720 | .683 | .648 |
| J/P | .740 | .635 | .666 | .568 |
| Nationality | .167 | .559 | **.620** | .517 |
| Political | .568 | .440 | .505 | .430 |

**Key Observations**:
- Most tasks underperform majority baseline, indicating task difficulty with unoptimized models
- Gender and nationality show above-baseline performance
- fastText generally outperforms more complex BB-LR (without fine-tuning)
- MBTI dimensions heavily skewed (S/N baseline at .940), making evaluation challenging

**Stratified vs. Balanced Impact**: Stratification improves personality dimension performance; balancing degrades overall performance due to severe undersampling.

### 2.7 Research Avenues Identified by Authors

The SOBR paper identifies three primary research directions:

#### 2.7.1 Adversarial Stylometry and Author Obfuscation

Developing perturbation techniques (character-level, translation, paraphrasing, word substitution) that reduce classifier accuracy toward chance level without compromising text quality. The corpus enables evaluation across multiple attributes simultaneously.

#### 2.7.2 Bias Investigation

The corpus can support:
- Data-sided bias analysis (representation skew in Reddit data)
- Model-sided bias analysis (bias inherited by classifiers)
- Downstream bias measurement (harm propagation)
- Bias mitigation research (counterfactual augmentation, debiasing)

#### 2.7.3 Profiling Model Development

Despite ethical concerns, understanding profiling capabilities is necessary for developing countermeasures. The corpus enables controlled study of stylometric inference.

### 2.8 Ethical Considerations

The SOBR paper dedicates significant attention to ethical dimensions:

**Dual-Use Acknowledgment**: Profiling enables both beneficial (fraud detection, content moderation) and harmful (surveillance, discrimination) applications.

**Dissemination Controls**: Corpus access requires fair-use agreement; data is anonymized; personal identifiers are stripped using TextWash.

**Demographic Attribute Limitations**: The paper acknowledges psychometric limitations of MBTI, binary gender representation, and US-centric political framing.

**Platform Bias**: Reddit users skew WEIRD (Western, Educated, Industrialized, Rich, Democratic), affecting generalizability.

---

## 3. Course Material Relevance Analysis

### 3.1 Most Relevant Weeks

Based on analysis of the assignment requirements and SOBR corpus characteristics, the following relevance ranking emerges:

| Week | Topic | Relevance | Justification |
|------|-------|-----------|---------------|
| **Week 3** | Classification and Language Modeling | ★★★★★ | Naive Bayes, logistic regression, evaluation metrics—directly applicable to baseline replication and extension |
| **Week 4** | Representation Learning | ★★★★★ | TF-IDF, PPMI, Word2Vec—core feature representations for stylometric classification |
| **Week 2** | Text Preprocessing | ★★★★☆ | Regular expressions, tokenization, normalization—essential for data pollution mitigation |
| **Week 6** | Deep Learning for NLP | ★★★☆☆ | RNNs, Transformers—advanced models for comparison (BB-LR baseline) |
| **Week 7** | Large Language Models | ★★★☆☆ | Contextual embeddings, bias discussion—relevant for advanced extensions |
| **Week 1** | Text Representation and Similarity | ★★★☆☆ | Bag-of-Words, cosine similarity—foundational but subsumed by later weeks |
| **Week 5** | Sequence Labeling | ★★☆☆☆ | HMMs, Viterbi—less directly applicable unless extending to token-level analysis |

### 3.2 Week 2: Text Preprocessing and Normalization

**Relevance to Assignment**: Week 2 material is essential for the data pollution mitigation component. Key applicable techniques:

**Regular Expressions** (Section 3):
- Pattern matching for detecting self-reports: e.g., `/\([MF]\d{2}\)/` for "(M25)", "(F34)"
- Extended patterns for informal self-reports: `/[Ii]'?m a \d{2}[\s-]?y(ear)?[\s-]?o(ld)?/`
- Disjunction and character classes for flexible matching
- Substitution operations for redaction: `re.sub(pattern, "[REDACTED]", text)`

**Tokenization** (Section 4):
- Subword tokenization (BPE) for handling vocabulary variance
- Heaps' Law ($|V| = kN^\beta$) for understanding vocabulary growth with corpus size
- Token normalization for reducing spurious vocabulary inflation

**Text Normalization** (Section 5):
- Case folding to reduce lexical variance
- Lemmatization for morphological normalization
- Stemming (Porter algorithm) for aggressive vocabulary reduction

**Evaluation Metrics** (Section 6):
- Confusion matrix formulation
- Precision, recall, F-score derivation
- Interpretation in context of profiling (false positives = privacy violations, false negatives = security gaps)

### 3.3 Week 3: Classification and Language Modeling

**Relevance to Assignment**: Week 3 provides the core classification machinery for author profiling.

**N-gram Language Models** (Section 4):
- Smoothing techniques (Laplace, add-$k$, interpolation) directly applicable to Naive Bayes with sparse vocabulary
- Perplexity as evaluation metric for language-based features

**Naive Bayes Classification** (Section 5):
- Generative model directly applicable to text classification
- Bag-of-words assumption aligns with traditional stylometry
- Add-one smoothing (Equation 5.8): $\hat{P}(w_i \mid c) = \frac{\text{count}(w_i, c) + 1}{\sum_{w \in V} \text{count}(w, c) + |V|}$
- Log-space computation for numerical stability
- Baseline model for comparison

**Logistic Regression** (Section 6):
- Discriminative alternative to Naive Bayes
- Sigmoid function: $\sigma(z) = \frac{1}{1 + e^{-z}}$
- Cross-entropy loss: $L_{CE} = -\sum_{i} y_i \log \hat{y}_i$
- Gradient descent optimization
- L1/L2 regularization for feature selection and overfitting prevention
- SOBR baseline uses logistic regression with TF-IDF features

**Evaluation Methodology** (Section 11):
- Precision: $\text{TP} / (\text{TP} + \text{FP})$
- Recall: $\text{TP} / (\text{TP} + \text{FN})$
- Macro F1 score (used in SOBR baselines)
- Cross-validation for robust estimation
- Overfitting/underfitting diagnosis

### 3.4 Week 4: Representation Learning

**Relevance to Assignment**: Week 4 addresses the fundamental question of how to represent text for stylometric classification.

**TF-IDF Weighting** (Section 3.3):
- $w_{t,d} = \text{tf}_{t,d} \times \text{idf}_t$
- SOBR LR baseline uses TF-IDF with sublinear TF scaling and IDF smoothing
- Character n-grams for stylistic features (capitalization, punctuation patterns)

**Pointwise Mutual Information** (Section 5):
- $\text{PMI}(w, c) = \log_2 \frac{P(w, c)}{P(w)P(c)}$
- PPMI for non-negative association scores
- Alternative to TF-IDF for capturing word associations

**Word2Vec and Skip-Gram** (Section 6):
- Dense embeddings as alternative to sparse TF-IDF
- Skip-Gram with Negative Sampling (SGNS)
- fastText (SOBR baseline) uses similar embedding approach with subword information
- Properties: semantic clustering, analogical reasoning

**Bias in Embeddings** (Section 6.6):
- Embeddings inherit corpus biases
- Gender, racial, occupational biases documented
- Directly relevant to SOBR's bias investigation goals
- Debiasing techniques as potential extension

**Neural Network Fundamentals** (Section 7–10):
- Activation functions (sigmoid, tanh, ReLU)
- Feedforward architectures
- Cross-entropy loss and backpropagation
- Foundation for understanding fastText and BB-LR baselines

### 3.5 Supplementary Weeks

**Week 6: Deep Learning for NLP**
- Big Bird embeddings (BB-LR baseline) require understanding of Transformer attention
- Self-attention: $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
- Contextual embeddings for capturing position-dependent meaning

**Week 7: Large Language Models**
- BERT-style masked language modeling
- Fine-tuning pretrained models
- Bias and fairness in LLMs
- Prompting paradigms (potential extension for zero-shot profiling)

---

## 4. Synthesis: Alignment of Course Material with Assignment

The assignment's dual requirements—(1) stylometric classification on SOBR data and (2) investigation/mitigation of data pollution—map cleanly onto course content:

| Assignment Component | Primary Week | Secondary Weeks | Key Techniques |
|---------------------|--------------|-----------------|----------------|
| Data preprocessing | Week 2 | — | Regex pattern matching, tokenization, normalization |
| Pollution detection | Week 2 | Week 3 | Pattern-based filtering, frequency analysis |
| Feature extraction | Week 4 | Week 1 | TF-IDF, n-grams, embeddings |
| Classification | Week 3 | Week 4, 6 | Naive Bayes, logistic regression, neural models |
| Evaluation | Week 3 | Week 2 | Precision/recall/F1, cross-validation |
| Bias analysis | Week 4 | Week 7 | Embedding bias, representation analysis |

**Critical Insight**: The assignment's emphasis on pollution mitigation transforms a standard classification task into a data quality investigation. Students must demonstrate not only technical proficiency in building classifiers but also critical thinking about what signals classifiers exploit and whether those signals reflect genuine stylometric properties.

---

# Part II: Project Planning Sketches

This section presents three distinct project approaches, each addressing the assignment requirements while emphasizing different aspects of course material.

---

## Project A: Data Pollution Detection and Mitigation via Text Normalization

| Metadata | Value |
| :--- | :--- |
| **Project Title** | Detecting and Removing Data Pollution in Stylometric Corpora: A Regex-Based Approach |
| **Primary Focus** | Preprocessing and data quality |
| **Core Hypothesis** | Systematic removal of self-report patterns and in-domain content significantly reduces classifier performance, exposing reliance on pollution rather than stylistic features |
| **Primary Course Material** | Week 2 (Preprocessing), Week 3 (Classification) |

### Executive Summary

This project investigates the extent to which author profiling classifiers trained on the SOBR corpus exploit data pollution—explicit self-reports and in-domain topical signals—rather than genuine stylometric features. Using regular expression-based detection and removal, we quantify the performance degradation when pollution is systematically eliminated. The hypothesis predicts that classifiers relying on contaminated features will exhibit substantial accuracy drops when these features are removed, while classifiers trained on cleaned data will demonstrate more robust cross-domain generalization.

### 1. Objectives and Success Criteria

**Primary Objective**: Develop a preprocessing pipeline that detects and removes data pollution from the SOBR corpus.

**Secondary Objective**: Quantify the impact of pollution removal on classifier performance across all five attributes.

**Success Criteria**:
1. Identification of pollution patterns achieving >90% recall on manually annotated sample
2. Quantifiable performance difference between contaminated and cleaned classifiers
3. Demonstration of improved cross-domain generalization on held-out subreddits

### 2. Research Questions and Hypotheses

**RQ1**: To what extent do standard classification models exploit explicit self-reports rather than stylometric features?

**H1**: Classifiers trained on raw SOBR data will exhibit >15% performance degradation when evaluated on data with self-reports removed.

**RQ2**: Does in-domain content provide unfair classification advantage?

**H2**: Excluding posts from label-source subreddits will reduce classification accuracy by >10% for personality and political leaning tasks.

### 3. Methodology

#### 3.1 Pollution Detection Pipeline

**Phase 1: Self-Report Pattern Development**

Develop hierarchical regex patterns of increasing specificity:

| Pattern Category | Example Regex | Target |
|-----------------|---------------|--------|
| Age-Gender format | `/\(([MF])\s*(\d{2})\)/i` | "(M25)", "(F34)" |
| Age self-report | `/[Ii]('m|'m| am) (\d{2})\s*(y(ear)?s?\s*old)?/` | "I'm 25 years old" |
| Gender self-report | `/[Ii]('m|'m| am) (a\s+)?(man|woman|male|female|guy|girl)/i` | "I'm a woman" |
| Nationality statement | `/[Ii]('m|'m| am) (from|in) ([A-Z][a-z]+)/` | "I'm from Germany" |
| MBTI mention | `/[Ii]('m|'m| am) (an?\s+)?(IN|EN|IS|ES)[TF][JP]/` | "I'm an INTJ" |
| Political statement | `/[Ii]('m|'m| am) (a\s+)?(liberal|conservative|leftist|centrist)/i` | "I'm a liberal" |

**Phase 2: Pattern Validation**
- Sample 500 posts per attribute
- Manual annotation for pollution presence
- Calculate precision/recall of each pattern
- Iteratively refine patterns

**Phase 3: Corpus Filtering**
- Apply validated patterns to full corpus
- Generate filtered dataset variants:
  - **SOBR-Clean**: All detected patterns removed
  - **SOBR-Redacted**: Patterns replaced with `[REDACTED]` tokens
  - **SOBR-NoDomain**: In-domain posts excluded (using `*_in_domain` flags)

#### 3.2 Classification Experiments

**Models**:
- Multinomial Naive Bayes (baseline)
- Logistic Regression with TF-IDF (replicating SOBR LR baseline)

**Experimental Conditions**:
1. Train on raw data, test on raw data (original baseline)
2. Train on raw data, test on cleaned data (pollution exploitation test)
3. Train on cleaned data, test on cleaned data (cleaned baseline)
4. Train on cleaned data, test on raw data (robustness test)

**Evaluation**:
- Macro F1 score (per SOBR protocol)
- Per-class precision/recall breakdown
- Performance delta across conditions

### 4. Data Split and Evaluation Protocol

- Use SOBR Random sample (10,000 authors)
- 80/10/10 train/validation/test split at author level
- Cross-validation (5-fold) on training set for hyperparameter tuning
- Final evaluation on held-out test set

### 5. Expected Outcomes

1. **Catalog of pollution patterns**: Documented regex patterns with precision/recall statistics
2. **Quantified pollution impact**: Performance comparison across contamination levels
3. **Cleaned dataset**: Preprocessed SOBR variant for future research
4. **Recommendations**: Best practices for data quality in distant-supervision pipelines

### 6. Timeline

| Week | Milestone |
|------|-----------|
| 1 | Pattern development and validation on sample |
| 2 | Full corpus processing; baseline classifier training |
| 3 | Experimental comparisons; results analysis |
| 4 | Paper writing and refinement |

### 7. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Patterns miss subtle self-reports | High | Medium | Accept recall limitations; document coverage |
| Minimal performance difference | Medium | High | This would be a valid negative result; analyze feature weights |
| Computational resources insufficient | Low | Medium | Use smaller sample; prioritize key attributes |

---

## Project B: Feature Ablation Study on Author Profiling Bias

| Metadata | Value |
| :--- | :--- |
| **Project Title** | Content vs. Style: A Feature Ablation Study of Bias Sources in Author Profiling |
| **Primary Focus** | Feature analysis and bias investigation |
| **Core Hypothesis** | Content words (nouns, entities) contribute disproportionately to classification, while function words (stylistic markers) provide more robust but weaker signals |
| **Primary Course Material** | Week 3 (Classification), Week 4 (Representation) |

### Executive Summary

This project investigates the relative contributions of content-based versus style-based features to author profiling performance. By systematically ablating feature categories—content words, function words, character n-grams, and syntactic patterns—we identify which feature types encode genuine stylometric signals versus content-based shortcuts that may not generalize. The hypothesis predicts that content words will dominate classification performance but exhibit poor cross-domain transfer, while function word features will show more stable performance across domains.

### 1. Objectives and Success Criteria

**Primary Objective**: Decompose classifier performance into contributions from content, style, and hybrid features.

**Secondary Objective**: Identify feature types that generalize across domains versus those that exploit domain-specific signals.

**Success Criteria**:
1. Quantified performance contribution of each feature category
2. Identified feature types with highest cross-domain transfer
3. Recommendations for robust stylometric feature engineering

### 2. Research Questions and Hypotheses

**RQ1**: What proportion of classification performance derives from content versus stylistic features?

**H1**: Removing content words (nouns, proper nouns, domain-specific vocabulary) will reduce classification accuracy by >30%.

**RQ2**: Which feature types exhibit cross-domain robustness?

**H2**: Function word features and character n-gram features will show <10% performance degradation when evaluated on out-of-domain subreddits.

**RQ3**: Do stylistic features capture genuine author characteristics or subreddit norms?

**H3**: Classifiers trained on single-subreddit data will underperform classifiers trained on multi-subreddit data when evaluated on held-out subreddits.

### 3. Methodology

#### 3.1 Feature Extraction Pipeline

**Feature Categories**:

| Category | Features | Rationale |
|----------|----------|-----------|
| **Content Words** | Nouns, proper nouns, domain vocabulary | Topical content |
| **Function Words** | Pronouns, prepositions, conjunctions, determiners | Stylistic markers |
| **Character N-grams** | Char 3-grams and 4-grams | Morphological and orthographic patterns |
| **Syntactic Patterns** | POS tag n-grams | Grammatical structure |
| **Punctuation** | Punctuation frequencies, emoji usage | Paralinguistic style |

**Implementation**:
- SpaCy for tokenization and POS tagging
- Custom extractors for each feature category
- TF-IDF weighting within each category
- Concatenated or isolated feature vectors

#### 3.2 Ablation Experiments

**Ablation Conditions**:
1. **Full model**: All features
2. **Content-only**: Content words only
3. **Style-only**: Function words + character n-grams + punctuation
4. **No-content**: All features except content words
5. **No-function**: All features except function words
6. **Char-only**: Character n-grams only

**Cross-Domain Protocol**:
- Split training data by subreddit source
- Train on subreddits A, B, C; test on subreddit D
- Measure performance transfer across domain boundaries

### 4. Evaluation

- Macro F1 score per condition
- Feature importance analysis (logistic regression coefficients)
- Cross-domain transfer ratio: $\text{Transfer} = \frac{F1_{\text{cross-domain}}}{F1_{\text{in-domain}}}$

### 5. Expected Outcomes

1. **Feature contribution matrix**: Performance by feature category and attribute
2. **Transfer analysis**: Identification of domain-robust features
3. **Bias diagnosis**: Documentation of content leakage in stylometric classification
4. **Feature engineering guidelines**: Recommendations for robust author profiling

### 6. Timeline

| Week | Milestone |
|------|-----------|
| 1 | Feature extraction pipeline development |
| 2 | Ablation experiments (single-domain) |
| 3 | Cross-domain transfer experiments |
| 4 | Analysis and paper writing |

### 7. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| POS tagging errors on Reddit text | High | Medium | Validate on sample; consider robust taggers |
| Insufficient subreddit diversity | Medium | Medium | Focus on attributes with diverse sources |
| Minimal content/style distinction | Low | High | Valid finding; reframe as style-content entanglement |

---

## Project C: Comparative Analysis of Representation Methods for Stylometric Classification

| Metadata | Value |
| :--- | :--- |
| **Project Title** | Sparse vs. Dense: Comparing Text Representations for Bias-Robust Author Profiling |
| **Primary Focus** | Representation learning and model comparison |
| **Core Hypothesis** | Dense embeddings (Word2Vec, fastText) are more susceptible to content-based bias than sparse representations (TF-IDF, PPMI) due to their tendency to capture topical rather than stylistic similarity |
| **Primary Course Material** | Week 4 (Representation), Week 3 (Classification) |

### Executive Summary

This project systematically compares sparse (TF-IDF, PPMI) and dense (Word2Vec, fastText) text representations for author profiling, with specific attention to their robustness against data pollution. The hypothesis posits that dense embeddings, by capturing semantic similarity, will more readily encode content-based signals that correlate with labels but do not generalize. Sparse representations, by contrast, may capture more superficial lexical patterns that reflect genuine stylistic choices. The study evaluates both in-domain and cross-domain performance to assess representation robustness.

### 1. Objectives and Success Criteria

**Primary Objective**: Compare sparse and dense representations for author profiling across multiple evaluation dimensions.

**Secondary Objective**: Identify which representation paradigm is more robust to data pollution.

**Success Criteria**:
1. Comprehensive performance comparison across representations
2. Quantified bias sensitivity per representation type
3. Recommendations for representation selection in stylometric tasks

### 2. Research Questions and Hypotheses

**RQ1**: Do sparse and dense representations exhibit different performance profiles across author attributes?

**H1**: Dense embeddings will outperform sparse representations on content-correlated attributes (nationality, political leaning) but underperform on stylistically-grounded attributes (personality dimensions).

**RQ2**: Which representation is more robust to data pollution?

**H2**: Sparse representations will show smaller performance degradation when in-domain content is removed, indicating less reliance on topical signals.

**RQ3**: How do representations encode author characteristics differently?

**H3**: Visualization of embedding spaces will reveal that dense representations cluster by topic/subreddit, while sparse representations cluster by writing style characteristics.

### 3. Methodology

#### 3.1 Representation Methods

| Method | Type | Implementation | Dimension |
|--------|------|----------------|-----------|
| **TF-IDF** | Sparse | sklearn TfidfVectorizer | $|V|$ (~50k) |
| **PPMI** | Sparse | Custom co-occurrence + PPMI transform | $|V|$ (~50k) |
| **Word2Vec** | Dense | gensim, pre-trained or corpus-trained | 300 |
| **fastText** | Dense | fasttext library (replicating SOBR) | 300 |
| **Document Embedding** | Dense | Averaged word embeddings | 300 |

#### 3.2 Classification Framework

**Classifier**: Logistic Regression (consistent across representations for fair comparison)

**Experimental Conditions**:
1. In-domain evaluation (standard split)
2. Cross-domain evaluation (held-out subreddits)
3. Cleaned data evaluation (pollution removed)

#### 3.3 Analysis Methods

**Quantitative**:
- Macro F1, precision, recall per condition
- Performance delta across conditions
- Statistical significance testing (paired bootstrap)

**Qualitative**:
- t-SNE visualization of representation spaces
- Nearest neighbor analysis for key terms
- Feature/dimension importance analysis

### 4. Evaluation Protocol

- SOBR Random sample with author-level splits
- 5-fold cross-validation for in-domain
- Subreddit-stratified splits for cross-domain
- Paired comparisons with bootstrap confidence intervals

### 5. Expected Outcomes

1. **Performance comparison table**: Representations × Attributes × Conditions
2. **Robustness ranking**: Which representations transfer best
3. **Visualization**: Embedding space structure by author attribute
4. **Practical guidelines**: Representation selection for stylometry

### 6. Timeline

| Week | Milestone |
|------|-----------|
| 1 | Representation extraction for all methods |
| 2 | In-domain classification experiments |
| 3 | Cross-domain and cleaned data experiments |
| 4 | Analysis, visualization, paper writing |

### 7. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Embedding training computationally expensive | Medium | Medium | Use pre-trained embeddings; subset corpus |
| Representation dimensions incomparable | Medium | Low | Normalize by effective dimensionality |
| Minimal performance differences | Medium | Medium | Focus on robustness metrics, not absolute performance |

---

## References

Beller, C., Knowles, R., Harman, C., Bergsma, S., Mitchell, M., & Van Durme, B. (2014). I'm a belieber: Social roles via self-identification and conceptual attributes. *Proceedings of ACL 2014*, 181–186.

Emmery, C., Chrupała, G., & Daelemans, W. (2017). Simple queries as distant labels for predicting gender on Twitter. *Proceedings of W-NUT@EMNLP 2017*, 50–55.

Emmery, C., Miotto, M., Kramp, S., & Kleinberg, B. (2024). SOBR: A corpus for stylometry, obfuscation, and bias on Reddit. *LREC-COLING 2024*, 14967–14983.

Jurafsky, D., & Martin, J.H. (2024). *Speech and Language Processing* (3rd ed., draft). Stanford University.

Kramp, S., Cassani, G., & Emmery, C. (2023). Native language identification with Big Bird embeddings. *arXiv preprint* arXiv:2309.06923.

Mikolov, T., Sutskever, I., Chen, K., Corrado, G.S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. *Advances in NeurIPS 2013*, 3111–3119.

Schler, J., Koppel, M., Argamon, S., & Pennebaker, J.W. (2006). Effects of age and gender on blogging. *Computational Approaches to Analyzing Weblogs*, 199–205.
