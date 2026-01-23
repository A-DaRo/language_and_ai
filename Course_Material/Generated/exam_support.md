# Exam Companion: Language and AI (Weeks 1–7)

| Document Property | Value |
| :--- | :--- |
| **Purpose** | High-yield exam revision guide with worked examples |
| **Scope** | Weeks 1–7: Text representation, normalization, classification, language modeling, embeddings, neural networks, sequence labeling, RNNs, LSTMs, attention, Transformers, and large language models |
| **Design Philosophy** | Formula → Semantic explanation → Step-by-step computation |
| **Related Material** | Week folders, lab sessions, generated chapters |

---

## Index

1. **Week 1: Text Representation & Similarity**
   - Core Formulas with Semantic Explanations
   - Worked Example 1: TF-IDF Calculation (Full Matrix)
   - Worked Example 2: Cosine Similarity (Step-by-Step)
   - Worked Example 3: Euclidean Distance
   - Worked Example 4: Jaccard Coefficient (Binary Vectors)

2. **Week 2: Text Normalization, Regex & Edit Distance**
   - Core Formulas with Semantic Explanations
   - Regex Quick Reference Card
   - Worked Example 1: Minimum Edit Distance (Full DP Matrix with Backtrace)
   - Worked Example 2: BPE Tokenization (Iterative Merging)
   - Worked Example 3: Precision, Recall, F1 Computation

3. **Week 3: Classification & Language Modeling**
   - Core Formulas with Semantic Explanations
   - Worked Example 1: Bigram Probability & Perplexity
   - Worked Example 2: Laplace Smoothing
   - Worked Example 3: Naive Bayes Classification (Log-Space)
   - Worked Example 4: Information Gain for ID3
   - Worked Example 5: Gradient Descent Step (Logistic Regression)

4. **Week 4: Representation Learning & Neural Networks**
   - Core Formulas with Semantic Explanations (PMI, PPMI, Skip-Gram, Negative Sampling)
   - Activation Functions Reference
   - Key Definitions (Distributional Hypothesis, Sparse vs Dense Embeddings)
   - Worked Example 1: PPMI Calculation
   - Worked Example 2: Neural Network Forward Pass (Single Neuron)
   - Worked Example 3: Two-Layer Network Forward Pass
   - Worked Example 4: Backpropagation Gradient Calculation

5. **Week 5: Sequence Labeling & Information Extraction**
   - Core Formulas with Semantic Explanations (HMM, Viterbi, CRF)
   - BIO Tagging Scheme
   - HMM vs CRF Comparison
   - Worked Example 1: Viterbi Algorithm — "the cat sat"
   - Worked Example 2: HMM Parameter Estimation
   - Worked Example 3: BIO Tagging
   - Worked Example 4: Viterbi with Larger Tagset

6. **Week 6: Deep Learning for Sequences (RNNs, LSTMs, Attention, Transformers)**
   - Core Formulas with Semantic Explanations (RNN, LSTM gates, Attention, Self-Attention)
   - Key Definitions (BPTT, Vanishing Gradients, Teacher Forcing, Encoder-Decoder)
   - Worked Example 1: RNN Forward Pass — "So long"
   - Worked Example 2: LSTM Gate Computation
   - Worked Example 3: Self-Attention Computation

7. **Week 7: Large Language Models (BERT, GPT, Decoding, Prompting)**
   - Core Formulas with Semantic Explanations (Causal LM, MLM, Temperature, Top-p, LoRA, RLHF)
   - Decoding Strategies Reference
   - Architecture Comparison: BERT vs GPT vs T5
   - Key Definitions (Contextual Embeddings, ICL, CoT, Finetuning, Hallucination)
   - Worked Example 1: Naive Bayes Sentiment Classification
   - Worked Example 2: Beam Search Decoding
   - Worked Example 3: Top-p (Nucleus) Sampling
   - Worked Example 4: Temperature Effect on Sampling

8. **Summary and Exam Strategy**
   - Weekly Checklists (Weeks 1–7)
   - Five-Minute Pre-Exam Review

---

## WEEK 1: Text Representation & Similarity

### Core Formulas with Semantic Explanations

#### Term Frequency (TF)

| **Formula** | $\text{tf}(t,d) = \lvert\{i: \tau(d)_i = t\}\rvert$ |
|-------------|-----------------------------------------------------|
| **Variables** | $t$ = term, $d$ = document, $\tau(d)$ = tokenized sequence of $d$ |
| **What it computes** | The raw count of how many times term $t$ appears in document $d$. |
| **Semantic meaning** | Measures term importance within a single document—more occurrences suggest greater relevance to that document's content. |

#### Log-Dampened Term Frequency

| **Formula** | $\text{tf}_{\log}(t,d) = \ln(\text{tf}(t,d) + 1)$ |
|-------------|--------------------------------------------------|
| **What it computes** | Applies logarithmic scaling to raw term counts. |
| **Semantic meaning** | Diminishes the impact of very high frequencies. A word appearing 100× is not 100× more important than one appearing once; the log compresses this gap. The "+1" prevents $\ln(0)$ for absent terms. |
| **Information-theoretic justification** | Each additional occurrence of a term contributes diminishing marginal information. |

#### Inverse Document Frequency (IDF)

| **Formula** | $\text{idf}_t = \log_{10}\left(\frac{N}{\text{df}_t}\right)$ |
|-------------|-------------------------------------------------------------|
| **Variables** | $N$ = total number of documents in corpus; $\text{df}_t$ = number of documents containing term $t$ |
| **What it computes** | The rarity of a term across the entire corpus. |
| **Semantic meaning** | **Penalizes common words** (like "the," "is") that appear in many documents and thus provide little discriminative power. **Rewards rare terms** that distinguish specific documents. If a term appears in all documents, $\text{idf} = \log(N/N) = 0$. |

#### TF-IDF Weight

| **Formula** | $w_{t,d} = \ln(\text{tf}(t,d) + 1) \cdot \log_{10}\left(\frac{N}{\text{df}_t}\right)$ |
|-------------|--------------------------------------------------------------------------------------|
| **What it computes** | A composite weight balancing within-document importance (TF) against corpus-wide rarity (IDF). |
| **Semantic meaning** | High TF-IDF indicates a term that is **frequent in this document** but **rare across the corpus**—a strong signal of document-specific content. Low TF-IDF means the term is either rare in this document or common everywhere. |

#### Euclidean Distance ($\ell_2$ Distance)

| **Formula** | $d_E(\vec{p},\vec{q}) = \sqrt{\sum_{i=1}^n (p_i - q_i)^2}$ |
|-------------|-----------------------------------------------------------|
| **What it computes** | The straight-line distance between two points in $n$-dimensional space. |
| **Semantic meaning** | Measures absolute positional difference between vectors. **Sensitive to magnitude**: two documents with identical word proportions but different lengths will appear distant. Best used when vector magnitudes are meaningful or after normalization. |

#### Jaccard Coefficient

| **Formula** | $J(A,B) = \frac{\lvert A \cap B\rvert}{\lvert A \cup B\rvert}$ |
|-------------|----------------------------------------------------------------|
| **Variables** | $A, B$ = sets of terms (or binary feature vectors treated as sets) |
| **What it computes** | The ratio of shared elements to total unique elements. |
| **Semantic meaning** | Measures **set overlap** independent of frequency. Useful for binary presence/absence features. A Jaccard of 1.0 means identical sets; 0.0 means no overlap. Ignores how often terms appear—only whether they appear at all. |

#### Cosine Similarity

| **Formula** | $\cos(\vec{p},\vec{q}) = \frac{\vec{p} \cdot \vec{q}}{\lVert\vec{p}\rVert_2 \lVert\vec{q}\rVert_2}$ |
|-------------|-----------------------------------------------------------------------------------------------------|
| **Variables** | $\vec{p} \cdot \vec{q}$ = dot product; $\lVert\vec{p}\rVert_2$ = Euclidean norm (magnitude) |
| **What it computes** | The cosine of the angle between two vectors. |
| **Semantic meaning** | Measures **directional alignment** regardless of magnitude. Two documents with the same word proportions have cosine similarity 1.0 even if one is twice as long. Ideal for comparing documents of varying lengths. Range: $[-1, 1]$, but $[0, 1]$ for non-negative TF vectors. |

#### Dot Product

| **Formula** | $\vec{p} \cdot \vec{q} = \sum_{i=1}^n p_i q_i$ |
|-------------|------------------------------------------------|
| **Semantic meaning** | Sums the products of corresponding components. Geometrically: $\vec{p} \cdot \vec{q} = \lVert\vec{p}\rVert \lVert\vec{q}\rVert \cos\theta$. Large when vectors point in similar directions with large magnitudes. |

#### $\ell_2$ Norm (Euclidean Magnitude)

| **Formula** | $\lVert\vec{p}\rVert_2 = \sqrt{\sum_{i=1}^n p_i^2}$ |
|-------------|-----------------------------------------------------|
| **Semantic meaning** | The length of the vector in Euclidean space. Used to normalize vectors to unit length, enabling fair comparison of direction regardless of magnitude. |

---

### Worked Example 1: TF-IDF Calculation (Full Matrix)

**Problem Statement**: Given three documents, compute the complete TF-IDF weight matrix.

**Documents**:
1. $d_1$: `"the cat sat on the mat"`
2. $d_2$: `"my cat sat on my cat"`
3. $d_3$: `"my cat sat on the mat on my cat"`

---

**Step 1: Tokenize and Build Vocabulary**

Extract unique terms and sort alphabetically:

$$V = \{\texttt{cat}, \texttt{mat}, \texttt{my}, \texttt{on}, \texttt{sat}, \texttt{the}\}$$

Vocabulary size: $|V| = 6$

---

**Step 2: Compute Raw Term Frequency Matrix**

Count occurrences of each term in each document:

| Document | `cat` | `mat` | `my` | `on` | `sat` | `the` |
|----------|-------|-------|------|------|-------|-------|
| $d_1$: "the cat sat on the mat" | 1 | 1 | 0 | 1 | 1 | 2 |
| $d_2$: "my cat sat on my cat" | 2 | 0 | 2 | 1 | 1 | 0 |
| $d_3$: "my cat sat on the mat on my cat" | 2 | 1 | 2 | 2 | 1 | 1 |

---

**Step 3: Compute Document Frequency ($\text{df}_t$)**

For each term, count how many documents contain it (at least once):

| Term | `cat` | `mat` | `my` | `on` | `sat` | `the` |
|------|-------|-------|------|------|-------|-------|
| Documents containing term | $d_1, d_2, d_3$ | $d_1, d_3$ | $d_2, d_3$ | $d_1, d_2, d_3$ | $d_1, d_2, d_3$ | $d_1, d_3$ |
| $\text{df}_t$ | 3 | 2 | 2 | 3 | 3 | 2 |

---

**Step 4: Compute Inverse Document Frequency ($\text{idf}_t$)**

Formula: $\text{idf}_t = \log_{10}\left(\frac{N}{\text{df}_t}\right)$ where $N = 3$ documents.

| Term | `cat` | `mat` | `my` | `on` | `sat` | `the` |
|------|-------|-------|------|------|-------|-------|
| Calculation | $\log_{10}(3/3)$ | $\log_{10}(3/2)$ | $\log_{10}(3/2)$ | $\log_{10}(3/3)$ | $\log_{10}(3/3)$ | $\log_{10}(3/2)$ |
| $\text{idf}_t$ | 0.000 | 0.176 | 0.176 | 0.000 | 0.000 | 0.176 |

**Semantic Insight**: Terms appearing in all 3 documents (`cat`, `on`, `sat`) have $\text{idf} = 0$, meaning they provide no discriminative value—every document contains them, so they cannot distinguish documents from one another.

---

**Step 5: Compute TF-IDF Weights**

Formula: $w_{t,d} = \ln(\text{tf}(t,d) + 1) \cdot \text{idf}_t$

**Sample Calculations**:

For $d_1$, term `the` ($\text{tf} = 2$, $\text{idf} = 0.176$):
$$w_{\texttt{the}, d_1} = \ln(2 + 1) \cdot 0.176 = \ln(3) \cdot 0.176 = 1.099 \cdot 0.176 = \mathbf{0.193}$$

For $d_1$, term `mat` ($\text{tf} = 1$, $\text{idf} = 0.176$):
$$w_{\texttt{mat}, d_1} = \ln(1 + 1) \cdot 0.176 = \ln(2) \cdot 0.176 = 0.693 \cdot 0.176 = \mathbf{0.122}$$

For $d_1$, term `cat` ($\text{tf} = 1$, $\text{idf} = 0.000$):
$$w_{\texttt{cat}, d_1} = \ln(1 + 1) \cdot 0.000 = 0.693 \cdot 0 = \mathbf{0.000}$$

**Complete TF-IDF Matrix**:

| Document | `cat` | `mat` | `my` | `on` | `sat` | `the` |
|----------|-------|-------|------|------|-------|-------|
| $d_1$ | 0.000 | 0.122 | 0.000 | 0.000 | 0.000 | 0.193 |
| $d_2$ | 0.000 | 0.000 | 0.193 | 0.000 | 0.000 | 0.000 |
| $d_3$ | 0.000 | 0.122 | 0.193 | 0.000 | 0.000 | 0.122 |

---

**Key Observations**:

1. **Zero-IDF terms vanish**: `cat`, `on`, `sat` contribute nothing to any document's representation because they appear everywhere.

2. **Discriminative terms remain**: Only `mat`, `my`, `the` have non-zero weights—these terms actually distinguish documents.

3. **TF scaling matters**: In $d_1$, `the` appears twice while `mat` appears once. The TF-IDF for `the` (0.193) exceeds `mat` (0.122) because $\ln(3) > \ln(2)$.

4. **Sparse representations**: Most of the matrix is zero, which is typical for TF-IDF in real corpora where vocabulary is large and most terms are absent from most documents.

For **document 1** (`cat` appears 1 time, $\text{idf}\_{\text{cat}} = 0.000$):
$$w\_{\text{cat},d\_1} = \ln(1+1) \cdot 0.000 = \ln(2) \cdot 0 = 0.000$$

For **document 1** (`the` appears 2 times, $\text{idf}\_{\text{the}} = 0.176$):
$$w\_{\text{the},d\_1} = \ln(2+1) \cdot 0.176 = \ln(3) \cdot 0.176 = 1.099 \cdot 0.176 = 0.193$$

For **document 1** (`mat` appears 1 time, $\text{idf}\_{\text{mat}} = 0.176$):
$$w\_{\text{mat},d\_1} = \ln(1+1) \cdot 0.176 = \ln(2) \cdot 0.176 = 0.693 \cdot 0.176 = 0.122$$

**Full TF-IDF Matrix**:

| Doc | `cat` | `mat` | `my` | `on` | `sat` | `the` |
|-----|-------|-------|------|------|-------|-------|
| $d\_1$ | 0.000 | 0.122 | 0.000 | 0.000 | 0.000 | 0.193 |
| $d\_2$ | 0.000 | 0.000 | 0.193 | 0.000 | 0.000 | 0.000 |
| $d\_3$ | 0.000 | 0.122 | 0.193 | 0.000 | 0.000 | 0.122 |

**Key Observations**:
- Terms appearing in all documents (`cat`, `on`, `sat`) have $\text{idf} = 0$, thus TF-IDF weight = 0
- Terms appearing in fewer documents (`the`, `mat`, `my`) receive higher IDF weights
- TF-IDF downweights common terms and emphasizes discriminative vocabulary

---

### Worked Example 2: Cosine Similarity (Step-by-Step)

**Problem Statement**: Compute the cosine similarity between two document vectors.

**Setup**: Given vectors (from lab session examples):
$$\vec{x}_2 = (9.7, 9.9), \quad \vec{x}_3 = (1.3, 2.7)$$

**Formula**:
$$\cos(\vec{p},\vec{q}) = \frac{\vec{p} \cdot \vec{q}}{\lVert\vec{p}\rVert_2 \cdot \lVert\vec{q}\rVert_2}$$

---

**Step 1: Compute the Dot Product**

The dot product sums the products of corresponding components:

$$\vec{x}_2 \cdot \vec{x}_3 = (9.7 \times 1.3) + (9.9 \times 2.7)$$
$$= 12.61 + 26.73 = \mathbf{39.34}$$

**Semantic insight**: The dot product is large when both vectors have large values in the same dimensions. Here, both vectors have their largest component in dimension 2.

---

**Step 2: Compute the $\ell_2$ Norms (Magnitudes)**

The norm is the Euclidean length of the vector:

For $\vec{x}_2$:
$$\lVert\vec{x}_2\rVert_2 = \sqrt{(9.7)^2 + (9.9)^2} = \sqrt{94.09 + 98.01} = \sqrt{192.10} = \mathbf{13.86}$$

For $\vec{x}_3$:
$$\lVert\vec{x}_3\rVert_2 = \sqrt{(1.3)^2 + (2.7)^2} = \sqrt{1.69 + 7.29} = \sqrt{8.98} = \mathbf{2.997}$$

**Semantic insight**: $\vec{x}_2$ has magnitude 13.86 while $\vec{x}_3$ has magnitude 2.997—$\vec{x}_2$ is about 4.6× longer. Without normalization, this magnitude difference would dominate distance-based comparisons.

---

**Step 3: Compute Cosine Similarity**

$$\cos(\vec{x}_2, \vec{x}_3) = \frac{39.34}{13.86 \times 2.997} = \frac{39.34}{41.53} = \mathbf{0.947}$$

---

**Interpretation**:

- Cosine similarity of **0.947** indicates the vectors are nearly **aligned in direction** (angle ≈ 18.7°).
- Despite $\vec{x}_2$ being nearly 5× longer than $\vec{x}_3$, they point in almost the same direction.
- This is why cosine similarity is preferred for document comparison: a long document and a short document covering the same topics will have high cosine similarity even if their TF values differ greatly.

**Geometric relationship**: $\cos\theta = 0.947 \Rightarrow \theta = \arccos(0.947) \approx 18.7°$

---

### Worked Example 3: Euclidean Distance

**Problem Statement**: Compute the Euclidean ($\ell_2$) distance between two document vectors.

**Setup**: Given vectors (from lab session examples):
$$\vec{x}_1 = (6.6, 6.2), \quad \vec{x}_4 = (1.3, 1.3)$$

**Formula**:
$$d_E(\vec{p},\vec{q}) = \sqrt{\sum_{i=1}^n (p_i - q_i)^2}$$

---

**Step 1: Compute Component-Wise Differences**

| Dimension | $\vec{x}_1$ | $\vec{x}_4$ | Difference | Squared |
|-----------|-------------|-------------|------------|---------|
| 1 | 6.6 | 1.3 | 5.3 | 28.09 |
| 2 | 6.2 | 1.3 | 4.9 | 24.01 |

---

**Step 2: Sum the Squared Differences**

$$\sum_{i} (p_i - q_i)^2 = 28.09 + 24.01 = \mathbf{52.10}$$

---

**Step 3: Take the Square Root**

$$d_E = \sqrt{52.10} = \mathbf{7.218}$$

---

**Interpretation**:

- The Euclidean distance of **7.218** represents the straight-line distance between the two points in 2D space.
- This metric is **sensitive to vector magnitude**: if one document is much longer (higher TF counts), it will appear distant from shorter documents even if they cover similar topics.
- **Contrast with cosine**: The same vectors would have high cosine similarity if they point in similar directions, regardless of magnitude.

**When to use Euclidean distance**: When absolute magnitudes are meaningful (e.g., comparing documents of similar length) or after $\ell_2$ normalization.

---

### Worked Example 4: Jaccard Coefficient (Binary Vectors)

**Problem Statement**: Compute the Jaccard similarity coefficient between two binary feature vectors.

**Setup**: Given binary vectors (from lab session examples):
$$\vec{x}_1 = [1, 0, 1, 1, 1, 0, 1, 1, 0, 0]$$
$$\vec{x}_2 = [0, 1, 0, 1, 1, 0, 1, 1, 0, 0]$$

Each position represents a feature (e.g., presence of a term); 1 = present, 0 = absent.

**Formula**:
$$J(A,B) = \frac{\lvert A \cap B\rvert}{\lvert A \cup B\rvert}$$

---

**Step 1: Identify Intersection (positions where both = 1)**

Compare element-wise to find positions where both vectors have value 1:

| Position | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|----------|---|---|---|---|---|---|---|---|---|---|
| $\vec{x}_1$ | 1 | 0 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 0 |
| $\vec{x}_2$ | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 1 | 0 | 0 |
| Both = 1? | ✗ | ✗ | ✗ | **✓** | **✓** | ✗ | **✓** | **✓** | ✗ | ✗ |

Intersection positions: {3, 4, 6, 7}

$$\lvert A \cap B\rvert = 4$$

---

**Step 2: Identify Union (positions where at least one = 1)**

Positions with at least one 1: {0, 1, 2, 3, 4, 6, 7}

$$\lvert A \cup B\rvert = 7$$

---

**Step 3: Compute Jaccard Coefficient**

$$J(\vec{x}_1, \vec{x}_2) = \frac{4}{7} = \mathbf{0.571}$$

---

**Interpretation**:

- Jaccard coefficient of **0.571** indicates moderate overlap between the two sets.
- 4 features are shared out of 7 total features present in either vector.
- Features that are 0 in both vectors (positions 5, 8, 9) do **not** contribute to the calculation—Jaccard only considers presences, not joint absences.

**When to use Jaccard**:
- Binary presence/absence features (e.g., does document contain word $w$?)
- Set comparison where frequency is irrelevant
- Situations where shared absences should not inflate similarity (contrast with simple matching coefficient)

---

## WEEK 2: Text Normalization, Regex & Edit Distance

### Core Formulas with Semantic Explanations

#### Heaps' Law (Vocabulary Growth)

| **Formula** | $\lvert V\rvert = kN^\beta$ where $0.67 < \beta < 0.75$ |
|-------------|--------------------------------------------------------|
| **Variables** | $\lvert V\rvert$ = vocabulary size (unique types), $N$ = corpus size (total tokens), $k$ = empirical constant |
| **What it computes** | Predicts how vocabulary grows as corpus size increases. |
| **Semantic meaning** | Vocabulary grows **sublinearly** with corpus size. Doubling the corpus does not double the vocabulary—most new tokens are repetitions of existing types. This explains why NLP systems can achieve reasonable coverage with finite vocabularies. |

#### Precision

| **Formula** | $P = \frac{TP}{TP + FP}$ |
|-------------|--------------------------|
| **Variables** | $TP$ = true positives (correctly predicted positive), $FP$ = false positives (incorrectly predicted positive) |
| **What it computes** | Of all instances predicted as positive, what fraction are actually positive? |
| **Semantic meaning** | Measures **exactness**. High precision means few false alarms—when the system says "positive," it is usually right. Low precision means many false positives. |
| **When to prioritize** | When false positives are costly (e.g., spam filter incorrectly blocking important emails). |

#### Recall (Sensitivity)

| **Formula** | $R = \frac{TP}{TP + FN}$ |
|-------------|--------------------------|
| **Variables** | $FN$ = false negatives (actual positives incorrectly predicted as negative) |
| **What it computes** | Of all actual positive instances, what fraction did we correctly identify? |
| **Semantic meaning** | Measures **completeness**. High recall means we find most positives—few slip through. Low recall means we miss many actual positives. |
| **When to prioritize** | When false negatives are costly (e.g., medical diagnosis missing a disease). |

#### F1-Score (Harmonic Mean)

| **Formula** | $F_1 = \frac{2PR}{P + R} = \frac{2TP}{2TP + FP + FN}$ |
|-------------|------------------------------------------------------|
| **What it computes** | A single metric balancing precision and recall. |
| **Semantic meaning** | The **harmonic mean** penalizes extreme imbalances. If $P = 0.9$ and $R = 0.1$, the arithmetic mean is 0.5, but $F_1 = 0.18$—reflecting that the system is effectively useless at finding positives. F1 requires both precision and recall to be reasonably high. |

#### Edit Distance (Levenshtein) Recurrence

| **Formula** | $D[i,j] = \min\begin{cases} D[i-1,j] + \text{del\_cost} \\ D[i,j-1] + \text{ins\_cost} \\ D[i-1,j-1] + \text{sub\_cost} \end{cases}$ |
|-------------|-----------------------------------------------------------------------------------------------------------------------------------|
| **Base cases** | $D[i,0] = i \cdot \text{del\_cost}$; $D[0,j] = j \cdot \text{ins\_cost}$ |
| **What it computes** | The minimum cost to transform source string (rows) into target string (columns). |
| **Semantic meaning of each operation**: |
| — $D[i-1,j] + \text{del}$ | Delete character $s_i$ from source (move up in matrix). |
| — $D[i,j-1] + \text{ins}$ | Insert character $t_j$ into source (move left in matrix). |
| — $D[i-1,j-1] + \text{sub}$ | Substitute $s_i$ with $t_j$; cost = 0 if they match, else substitution cost. |
| **Standard Levenshtein costs** | Insert = 1, Delete = 1, Substitute = 2 (or 1 in some variants). |

---

### Regex Quick Reference Card

| **Operator** | **Meaning** | **Example** |
|--------------|-------------|-------------|
| `.` | Match any character (except newline) | `n.p` matches `nop`, `nlp`, `n8p` |
| `[abc]` | Character class: match $a$, $b$, or $c$ | `[Yy]ou` matches `You`, `you` |
| `[^abc]` | Negated class: match anything except $a$, $b$, $c$ | `[^0-9]+` matches sequences without digits |
| `*` | Zero or more occurrences | `colou*r` matches `color`, `colour`, `colouur` |
| `+` | One or more occurrences | `\d+` matches sequences of digits |
| `?` | Zero or one occurrence | `colou?r` matches `color`, `colour` |
| `^` | Start of string | `^The` matches `The` only at beginning |
| `$` | End of string | `end$` matches `end` only at conclusion |
| `\b` | Word boundary | `\bthe\b` matches `the` but not `there` |
| `\d` | Digit shorthand (`[0-9]`) | `\d{3}` matches three digits |
| `\w` | Word character (`[a-zA-Z0-9_]`) | `\w+` matches alphanumeric sequences |
| `\s` | Whitespace (`[ \t\n\r]`) | `\s+` matches whitespace sequences |
| `\W`, `\D`, `\S` | Negated shortcuts | `\W` = non-word character |
| `\|` | Alternation (OR) | `cat\|dog` matches `cat` or `dog` |
| `(...)` | Capture group | `(ab)+` matches `ab`, `abab`, `ababab` |

---

### Worked Example 1: Minimum Edit Distance (Full DP Matrix with Backtrace)

**Problem Statement**: Compute the minimum edit distance to transform `intention` into `execution`.

**Costs**: Insertion = 1, Deletion = 1, Substitution = 2

**Source**: `intention` (length 9)  
**Target**: `execution` (length 9)

---

**Step 1: Initialize the DP Matrix**

Create a $(10 \times 10)$ matrix (source length + 1 rows, target length + 1 columns).

**Base case intuition**:
- $D[i, 0] = i$: Transforming the first $i$ characters of source into empty string requires $i$ deletions.
- $D[0, j] = j$: Transforming empty string into first $j$ characters of target requires $j$ insertions.

---

**Step 2: Fill the Matrix Using the Recurrence**

For each cell $D[i,j]$, compute:

$$D[i,j] = \min\begin{cases}
D[i-1,j] + 1 & \text{(delete } s_i \text{)} \\
D[i,j-1] + 1 & \text{(insert } t_j \text{)} \\
D[i-1,j-1] + \text{cost} & \text{(substitute: cost = 0 if } s_i = t_j \text{, else 2)}
\end{cases}$$

**Complete DP Matrix**:

|     | ε | e | x | e | c | u | t | i | o | n |
|-----|---|---|---|---|---|---|---|---|---|---|
| **ε** | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| **i** | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 6 | 7 | 8 |
| **n** | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 7 | 8 | 7 |
| **t** | 3 | 4 | 5 | 6 | 7 | 8 | 7 | 8 | 9 | 8 |
| **e** | 4 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 9 |
| **n** | 5 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11| 10|
| **t** | 6 | 5 | 6 | 7 | 8 | 9 | 8 | 9 | 10| 11|
| **i** | 7 | 6 | 7 | 8 | 9 | 10| 9 | 8 | 9 | 10|
| **o** | 8 | 7 | 8 | 9 | 10| 11| 10| 9 | 8 | 9 |
| **n** | 9 | 8 | 9 | 10| 11| 12| 11| 10| 9 | **8** |

---

**Step 3: Read Final Distance**

$$\boxed{D[9,9] = 8}$$

This means 8 units of edit cost are required to transform `intention` → `execution`.

---

**Step 4: Detailed Cell Calculation Example**

Let us compute $D[4,1]$ (source prefix `inte`, target prefix `e`):

- Source character: $s_4 = $ `e`
- Target character: $t_1 = $ `e`
- Characters match, so substitution cost = 0.

$$D[4,1] = \min\begin{cases}
D[3,1] + 1 = 4 + 1 = 5 & \text{(delete 'e')} \\
D[4,0] + 1 = 4 + 1 = 5 & \text{(insert 'e')} \\
D[3,0] + 0 = 3 + 0 = 3 & \text{(match 'e' = 'e')}
\end{cases} = 3$$

The minimum is 3, achieved by the diagonal move (match).

---

**Step 5: Backtrace to Recover Alignment**

The backtrace procedure reconstructs the optimal sequence of edit operations by tracing backwards from $D[n,m]$ to $D[0,0]$.

**General Backtrace Protocol**:

At each cell $D[i,j]$, determine which predecessor cell contributed the current value by checking the recurrence relation in reverse:

| **Current Cell Came From** | **Direction** | **Operation** | **Condition** |
|----------------------------|---------------|---------------|---------------|
| $D[i-1,j-1]$ | Diagonal (↖) | **Match** if $s_i = t_j$; **Substitute** otherwise | $D[i,j] = D[i-1,j-1] + \text{cost}$ |
| $D[i-1,j]$ | Up (↑) | **Delete** $s_i$ from source | $D[i,j] = D[i-1,j] + 1$ |
| $D[i,j-1]$ | Left (←) | **Insert** $t_j$ into source | $D[i,j] = D[i,j-1] + 1$ |

When multiple predecessors yield the same minimum, multiple optimal alignments exist; choose any one.

---

**Backtrace for `intention` → `execution`**:

Starting at $D[9,9] = 8$, we trace back by identifying which predecessor contributed each cell's value:

| **Cell** | **Value** | **Source Char** | **Target Char** | **Predecessor Options** | **Chosen Path** | **Operation** |
|----------|-----------|-----------------|-----------------|-------------------------|-----------------|---------------|
| $(9,9)$ | 8 | `n` | `n` | ↖ $D[8,8]=8+0=8$ ✓ | Diagonal | **Match** |
| $(8,8)$ | 8 | `o` | `o` | ↖ $D[7,7]=8+0=8$ ✓ | Diagonal | **Match** |
| $(7,7)$ | 8 | `i` | `i` | ↖ $D[6,6]=8+0=8$ ✓ | Diagonal | **Match** |
| $(6,6)$ | 8 | `t` | `t` | ↖ $D[5,5]=8+0=8$ ✓ | Diagonal | **Match** |
| $(5,5)$ | 8 | `n` | `u` | ↖ $D[4,4]=6+2=8$ ✓ | Diagonal | **Substitute** `n`→`u` |
| $(4,4)$ | 6 | `e` | `c` | ← $D[4,3]=5+1=6$ ✓ | Left | **Insert** `c` |
| $(4,3)$ | 5 | `e` | `e` | ↖ $D[3,2]=5+0=5$ ✓ | Diagonal | **Match** |
| $(3,2)$ | 5 | `t` | `x` | ↖ $D[2,1]=3+2=5$ ✓ | Diagonal | **Substitute** `t`→`x` |
| $(2,1)$ | 3 | `n` | `e` | ↖ $D[1,0]=1+2=3$ ✓ | Diagonal | **Substitute** `n`→`e` |
| $(1,0)$ | 1 | `i` | ε | ↑ $D[0,0]=0+1=1$ ✓ | Up | **Delete** `i` |
| $(0,0)$ | 0 | — | — | — | **Done** | — |

---

**Reconstructed Operation Sequence** (reading the trace in reverse, from start to end):

| **Step** | **Operation** | **Cost** | **Cumulative** |
|----------|---------------|----------|----------------|
| 1 | Delete `i` | 1 | 1 |
| 2 | Substitute `n` → `e` | 2 | 3 |
| 3 | Substitute `t` → `x` | 2 | 5 |
| 4 | Match `e` = `e` | 0 | 5 |
| 5 | Insert `c` | 1 | 6 |
| 6 | Substitute `n` → `u` | 2 | 8 |
| 7 | Match `t` = `t` | 0 | 8 |
| 8 | Match `i` = `i` | 0 | 8 |
| 9 | Match `o` = `o` | 0 | 8 |
| 10 | Match `n` = `n` | 0 | 8 |

---

**Final Alignment Visualization**:

```
Source: i  n  t  e  -  n  t  i  o  n
        D  S  S  M  I  S  M  M  M  M
Target: -  e  x  e  c  u  t  i  o  n
```

**Legend**: D = Delete, S = Substitute, M = Match, I = Insert

**Cost Verification**:
$$\text{Total} = 1_{\text{del}} + 3 \times 2_{\text{sub}} + 1_{\text{ins}} + 5 \times 0_{\text{match}} = 1 + 6 + 1 + 0 = 8 \; \checkmark$$

---

**Algorithm Pseudocode** (for reference):

```
MIN-EDIT-DISTANCE(source, target):
    n ← length(source)
    m ← length(target)
    D ← (n+1) × (m+1) matrix
    
    // Base cases
    for i ← 0 to n: D[i,0] ← i
    for j ← 0 to m: D[0,j] ← j
    
    // Fill matrix
    for i ← 1 to n:
        for j ← 1 to m:
            if source[i] = target[j]:
                cost ← 0
            else:
                cost ← 2  // substitution cost
            D[i,j] ← min(D[i-1,j] + 1,      // delete
                         D[i,j-1] + 1,      // insert
                         D[i-1,j-1] + cost) // substitute/match
    
    return D[n,m]
```

---

### Worked Example 2: BPE Tokenization (Iterative Merging)

**Setup**: Corpus with token frequencies:
```
low_      × 5
lowest_   × 2
newer_    × 6
wider_    × 3
new_      × 2
```

**Initial Vocabulary**: All individual characters: `{l, o, w, _, e, s, t, n, r, i, d}`

**Step 1: Count All Adjacent Pairs**

Count bigrams across all tokens (weighted by frequency):
- `(l, o)`: 5 (from `low_`) + 2 (from `lowest_`) = 7
- `(o, w)`: 5 + 2 = 7
- `(w, _)`: 5 = 5
- `(w, e)`: 2 (from `lowest_`) = 2
- `(e, s)`: 2 = 2
- `(s, t)`: 2 = 2
- `(t, _)`: 2 = 2
- `(n, e)`: 6 (from `newer_`) + 2 (from `new_`) = 8
- `(e, w)`: 6 + 2 = 8
- `(w, e)` (in `newer`): 6 = 6
- `(e, r)`: 6 (from `newer_`) + 3 (from `wider_`) = 9 **← Most frequent**
- `(r, _)`: 6 + 3 = 9
- `(w, i)`: 3 = 3
- `(i, d)`: 3 = 3
- `(d, e)`: 3 = 3

**Most frequent pair**: `(e, r)` with count 9

**Iteration 1: Merge `e + r → er`**

Update vocabulary: `{l, o, w, _, e, s, t, n, r, i, d, er}`

Update corpus:
```
low_      × 5
lowest_   × 2
newer_    → new er_ × 6
wider_    → wid er_ × 3
new_      × 2
```

**Step 2: Count All Pairs Again**

New most frequent pair after merge: `(er, _)` with count 9

**Iteration 2: Merge `er + _ → er_`**

Update vocabulary: `{l, o, w, _, e, s, t, n, r, i, d, er, er_}`

Update corpus:
```
low_      × 5
lowest_   × 2
newer_    → new er_  × 6
wider_    → wid er_  × 3
new_      × 2
```

**Step 3: Count All Pairs Again**

New most frequent pair: `(n, e)` with count 8

**Iteration 3: Merge `n + e → ne`**

Update vocabulary: `{l, o, w, _, e, s, t, n, r, i, d, er, er_, ne}`

Update corpus:
```
low_      × 5
lowest_   × 2
newer_    → ne w er_  × 6
wider_    → wid er_   × 3
new_      → ne w_     × 2
```

**Final Vocabulary After 3 Merges**: `{l, o, w, _, e, s, t, n, r, i, d, er, er_, ne}`

**Key Insight**: BPE iteratively merges most frequent adjacent pairs, balancing vocabulary size with tokenization efficiency.

---

### Worked Example 3: Precision, Recall, F1 Computation

**Setup**: Binary classification task with confusion matrix:

|              | **Predicted Positive** | **Predicted Negative** |
|--------------|------------------------|------------------------|
| **Actual Positive** | TP = 40 | FN = 5 |
| **Actual Negative** | FP = 10 | TN = 45 |

**Step 1: Compute Precision**

$$P = \frac{TP}{TP + FP} = \frac{40}{40 + 10} = \frac{40}{50} = 0.80$$

**Interpretation**: 80% of predicted positives are correct.

**Step 2: Compute Recall**

$$R = \frac{TP}{TP + FN} = \frac{40}{40 + 5} = \frac{40}{45} = 0.889$$

**Interpretation**: 88.9% of actual positives are correctly identified.

**Step 3: Compute F1-Score**

$$F_1 = \frac{2PR}{P + R} = \frac{2 \cdot 0.80 \cdot 0.889}{0.80 + 0.889} = \frac{1.422}{1.689} = 0.842$$

**Alternative Formula**:
$$F_1 = \frac{2TP}{2TP + FP + FN} = \frac{2 \cdot 40}{2 \cdot 40 + 10 + 5} = \frac{80}{95} = 0.842$$

**Interpretation**: F1-score balances precision and recall; 0.842 indicates strong performance.

---

## WEEK 3: Classification & Language Modeling

### Core Formulas with Semantic Explanations

#### Chain Rule of Probability

| **Formula** | $P(w_{1:n}) = \prod_{k=1}^n P(w_k \mid w_{1:k-1})$ |
|-------------|---------------------------------------------------|
| **What it computes** | Decomposes joint probability of a sequence into a product of conditional probabilities. |
| **Semantic meaning** | The probability of a sentence equals the probability of the first word, times the probability of the second word given the first, times the probability of the third given the first two, etc. This is mathematically exact but intractable for long sequences. |

#### Bigram Approximation (Markov Assumption)

| **Formula** | $P(w_{1:n}) \approx \prod_{k=1}^n P(w_k \mid w_{k-1})$ |
|-------------|-------------------------------------------------------|
| **What it computes** | Approximates each conditional by considering only the immediately preceding word. |
| **Semantic meaning** | Assumes limited memory: the probability of the next word depends **only on the current word**, not the entire history. This makes estimation tractable at the cost of ignoring long-range dependencies. |

#### Bigram MLE (Maximum Likelihood Estimation)

| **Formula** | $P(w_n \mid w_{n-1}) = \frac{C(w_{n-1}, w_n)}{C(w_{n-1})}$ |
|-------------|-----------------------------------------------------------|
| **Variables** | $C(w_{n-1}, w_n)$ = count of bigram; $C(w_{n-1})$ = count of preceding word |
| **What it computes** | Relative frequency of the bigram divided by frequency of the context word. |
| **Semantic meaning** | If we see `want to` 608 times out of 927 occurrences of `want`, we estimate $P(\text{to} \mid \text{want}) = 608/927 \approx 0.66$. This is the most likely probability given the observed data. |

#### Perplexity

| **Formula** | $\text{PP}(W) = P(w_1, \ldots, w_N)^{-1/N} = \sqrt[N]{\prod_{i=1}^N \frac{1}{P(w_i \mid w_{1:i-1})}}$ |
|-------------|------------------------------------------------------------------------------------------------------|
| **What it computes** | The inverse geometric mean of the probabilities assigned by the model. |
| **Semantic meaning** | Measures how "surprised" the model is by the test data. **Lower perplexity = better model**. Intuitively, perplexity represents the **average branching factor**—the effective number of equally likely choices the model faces at each position. A perfect model (always assigns probability 1) has perplexity 1. |

#### Laplace (Add-1) Smoothing

| **Formula** | $P_{\text{Lap}}(w_n \mid w_{n-1}) = \frac{C(w_{n-1}, w_n) + 1}{C(w_{n-1}) + \lvert V\rvert}$ |
|-------------|----------------------------------------------------------------------------------------------|
| **Variables** | $\lvert V\rvert$ = vocabulary size |
| **What it computes** | Adds 1 to every bigram count, ensuring no zero probabilities. |
| **Semantic meaning** | Prevents zero probability for unseen bigrams (which would make entire sequence probability zero). The denominator adds $\lvert V\rvert$ to maintain proper normalization. **Caveat**: Redistributes too much probability mass to rare events for large vocabularies—more sophisticated smoothing (Kneser-Ney) is often preferred. |

#### Naive Bayes Classifier

| **Formula** | $\hat{c} = \arg\max_c P(c) \prod_{i=1}^n P(w_i \mid c)$ |
|-------------|--------------------------------------------------------|
| **What it computes** | Chooses the class maximizing prior probability times product of word likelihoods. |
| **Semantic meaning** | A **generative classifier**: models how documents are generated by each class, then uses Bayes' rule to invert. The "naive" assumption is that words are **conditionally independent** given the class—each word contributes independently to the classification decision. Despite this unrealistic assumption, Naive Bayes often performs surprisingly well. |

#### NB Likelihood (Add-1 Smoothed)

| **Formula** | $\hat{P}(w \mid c) = \frac{\text{count}(w, c) + 1}{\sum_{w' \in V} \text{count}(w', c) + \lvert V\rvert}$ |
|-------------|----------------------------------------------------------------------------------------------------------|
| **What it computes** | Smoothed probability of word $w$ appearing in documents of class $c$. |
| **Semantic meaning** | Without smoothing, a word never seen in class $c$ would have probability 0, making any document containing it impossible under class $c$. Add-1 smoothing ensures every word has at least some small probability in every class. |

#### Entropy

| **Formula** | $H(Y) = -\sum_{k=1}^K p_k \log_2 p_k$ |
|-------------|---------------------------------------|
| **What it computes** | The average number of bits needed to encode outcomes from distribution $Y$. |
| **Semantic meaning** | Measures **uncertainty** in a distribution. Maximum entropy = uniform distribution (most uncertain). Minimum entropy = deterministic distribution (no uncertainty). A fair coin has entropy 1 bit; a biased coin (90% heads) has lower entropy. |

#### Information Gain

| **Formula** | $\text{IG}(Y, X) = H(Y) - H(Y \mid X)$ |
|-------------|----------------------------------------|
| **What it computes** | Reduction in entropy of $Y$ after observing feature $X$. |
| **Semantic meaning** | Measures how much knowing feature $X$ helps predict label $Y$. High IG means the feature is informative for classification. ID3 decision trees select the feature with highest information gain at each split. |

#### Sigmoid Function

| **Formula** | $\sigma(z) = \frac{1}{1 + e^{-z}}$ |
|-------------|-----------------------------------|
| **What it computes** | Maps any real number to the interval $(0, 1)$. |
| **Semantic meaning** | Converts a linear combination of features into a probability. Large positive $z$ → probability near 1; large negative $z$ → probability near 0; $z = 0$ → probability 0.5. |

#### Binary Cross-Entropy Loss

| **Formula** | $\mathcal{L} = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]$ |
|-------------|----------------------------------------------------------|
| **Variables** | $y$ = true label (0 or 1); $\hat{y}$ = predicted probability |
| **What it computes** | Negative log-likelihood of the true label under the predicted distribution. |
| **Semantic meaning** | Penalizes confident wrong predictions heavily. If $y = 1$ and $\hat{y} = 0.01$, loss = $-\log(0.01) = 4.6$ (high). If $y = 1$ and $\hat{y} = 0.99$, loss = $-\log(0.99) = 0.01$ (low). Forces the model to assign high probability to the correct class. |

#### Gradient Update Rule

| **Formula** | $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$ |
|-------------|-------------------------------------------------------------|
| **Variables** | $\theta$ = parameters (weights, bias); $\eta$ = learning rate; $\nabla$ = gradient |
| **What it computes** | Updates parameters in the direction that decreases loss. |
| **Semantic meaning** | The gradient points in the direction of steepest increase in loss. Subtracting it moves parameters toward lower loss. Learning rate $\eta$ controls step size—too large causes oscillation, too small causes slow convergence. |

#### Logistic Regression Gradient

| **Formula** | $\frac{\partial \mathcal{L}}{\partial w_j} = (\hat{y} - y) x_j$ |
|-------------|----------------------------------------------------------------|
| **What it computes** | How much weight $w_j$ should change to reduce loss. |
| **Semantic meaning** | If $\hat{y} > y$ (predicted too high), the gradient is positive, so we decrease $w_j$ for features where $x_j > 0$. If $\hat{y} < y$ (predicted too low), gradient is negative, so we increase $w_j$. The update is proportional to the error $(\hat{y} - y)$ and the feature value $x_j$. |

---

### Worked Example 1: Bigram Probability & Perplexity

**Setup**: Training corpus with three sentences:
1. `<s> I am Sam </s>`
2. `<s> Sam I am </s>`
3. `<s> I do not like green eggs and ham </s>`

**Step 1: Count Unigrams**

| Token | Count | Token | Count |
|-------|-------|-------|-------|
| `<s>` | 3 | `I` | 3 |
| `am` | 2 | `Sam` | 2 |
| `do` | 1 | `not` | 1 |
| `like` | 1 | `green` | 1 |
| `eggs` | 1 | `and` | 1 |
| `ham` | 1 | `</s>` | 3 |

**Step 2: Count Bigrams**

| Bigram | Count | Bigram | Count |
|--------|-------|--------|-------|
| `<s> I` | 2 | `<s> Sam` | 1 |
| `I am` | 2 | `I do` | 1 |
| `am Sam` | 1 | `am </s>` | 1 |
| `Sam I` | 1 | `Sam </s>` | 1 |
| `do not` | 1 | `not like` | 1 |
| `like green` | 1 | `green eggs` | 1 |
| `eggs and` | 1 | `and ham` | 1 |
| `ham </s>` | 1 | | |

**Step 3: Compute Bigram Probabilities (MLE)**

$$P(w_n \mid w_{n-1}) = \frac{C(w_{n-1}, w_n)}{C(w_{n-1})}$$

Examples:
$$P(\text{I} \mid \langle s \rangle) = \frac{C(\langle s \rangle, \text{I})}{C(\langle s \rangle)} = \frac{2}{3}$$

$$P(\text{am} \mid \text{I}) = \frac{C(\text{I}, \text{am})}{C(\text{I})} = \frac{2}{3}$$

$$P(\text{Sam} \mid \text{am}) = \frac{C(\text{am}, \text{Sam})}{C(\text{am})} = \frac{1}{2}$$

$$P(\langle /s \rangle \mid \text{Sam}) = \frac{C(\text{Sam}, \langle /s \rangle)}{C(\text{Sam})} = \frac{1}{2}$$

**Step 4: Compute Sentence Probability**

Test sentence: `<s> I am Sam </s>`

$$P(w_{1:4}) = P(\text{I} \mid \langle s \rangle) \cdot P(\text{am} \mid \text{I}) \cdot P(\text{Sam} \mid \text{am}) \cdot P(\langle /s \rangle \mid \text{Sam})$$

$$= \frac{2}{3} \cdot \frac{2}{3} \cdot \frac{1}{2} \cdot \frac{1}{2} = \frac{4}{36} = \frac{1}{9} \approx 0.111$$

**Step 5: Compute Perplexity**

Perplexity measures average branching factor; lower is better.

$$\text{PP}(W) = P(w_{1:N})^{-1/N}$$

For sentence with $N = 4$ tokens (excluding `<s>`):

$$\text{PP} = (0.111)^{-1/4} = (9)^{1/4} = 1.732$$

**Interpretation**: On average, the model is choosing between ~1.73 equally likely next words.

---

### Worked Example 2: Laplace Smoothing

**Setup**: From a large corpus, we have:
- Bigram count: $C(\text{want}, \text{to}) = 608$
- Unigram count: $C(\text{want}) = 927$
- Vocabulary size: $\lvert V\rvert = 1446$

**MLE Estimate (No Smoothing)**:

$$P(\text{to} \mid \text{want}) = \frac{608}{927} = 0.656$$

**Laplace (Add-1) Smoothing**:

$$P_{\text{Lap}}(\text{to} \mid \text{want}) = \frac{C(\text{want}, \text{to}) + 1}{C(\text{want}) + \lvert V\rvert} = \frac{608 + 1}{927 + 1446} = \frac{609}{2373} = 0.257$$

**Step-by-Step Computation**:

Numerator: $608 + 1 = 609$

Denominator: $927 + 1446 = 2373$

Division: $609 \div 2373 = 0.2567 \approx 0.257$

**Observation**: Laplace smoothing redistributes probability mass to unseen events, significantly reducing probability of observed bigrams. For high-frequency pairs, this can be too aggressive; alternative smoothing (Kneser-Ney, interpolation) is often preferred.

---

### Worked Example 3: Naive Bayes Classification (Log-Space)

**Setup**: Classify text as **spam** or **ham**. Training data:

**Class Priors**:
- 3 spam documents
- 2 ham documents
- Total: 5 documents

**Word Counts**:

| Word | Count in Spam | Count in Ham |
|------|---------------|--------------|
| `great` | 2 | 0 |
| `movie` | 3 | 1 |
| `bad` | 0 | 2 |

**Vocabulary**: $V = \\{\text{great}, \text{movie}, \text{bad}\\}$, $\lvert V\rvert = 3$

**Total Word Counts per Class**:
- Spam: $2 + 3 = 5$ words
- Ham: $0 + 1 + 2 = 3$ words

**Test Document**: `"bad movie"`

---

**Step 1: Compute Class Priors**

$$P(\text{spam}) = \frac{3}{5} = 0.6$$

$$P(\text{ham}) = \frac{2}{5} = 0.4$$

---

**Step 2: Compute Likelihoods with Add-1 Smoothing**

Formula:
$$\hat{P}(w \mid c) = \frac{\text{count}(w, c) + 1}{\sum_{w' \in V} \text{count}(w', c) + \lvert V\rvert}$$

**For Spam**:

Denominator: $5 + 3 = 8$

$$P(\text{bad} \mid \text{spam}) = \frac{0 + 1}{8} = \frac{1}{8} = 0.125$$

$$P(\text{movie} \mid \text{spam}) = \frac{3 + 1}{8} = \frac{4}{8} = 0.5$$

**For Ham**:

Denominator: $3 + 3 = 6$

$$P(\text{bad} \mid \text{ham}) = \frac{2 + 1}{6} = \frac{3}{6} = 0.5$$

$$P(\text{movie} \mid \text{ham}) = \frac{1 + 1}{6} = \frac{2}{6} = 0.333$$

---

**Step 3: Compute Posterior (Probability Space)**

$$P(\text{spam} \mid \text{doc}) \propto P(\text{spam}) \cdot P(\text{bad} \mid \text{spam}) \cdot P(\text{movie} \mid \text{spam})$$

$$= 0.6 \cdot 0.125 \cdot 0.5 = 0.0375$$

$$P(\text{ham} \mid \text{doc}) \propto P(\text{ham}) \cdot P(\text{bad} \mid \text{ham}) \cdot P(\text{movie} \mid \text{ham})$$

$$= 0.4 \cdot 0.5 \cdot 0.333 = 0.0666$$

**Classification**: $\arg\max(0.0375, 0.0666) = \text{ham}$

---

**Step 4: Compute in Log-Space (Preferred for Numerical Stability)**

$$\log P(\text{spam} \mid \text{doc}) = \log P(\text{spam}) + \log P(\text{bad} \mid \text{spam}) + \log P(\text{movie} \mid \text{spam})$$

Using natural logarithms:
$$= \ln(0.6) + \ln(0.125) + \ln(0.5)$$
$$= -0.511 + (-2.079) + (-0.693) = -3.283$$

$$\log P(\text{ham} \mid \text{doc}) = \log P(\text{ham}) + \log P(\text{bad} \mid \text{ham}) + \log P(\text{movie} \mid \text{ham})$$

$$= \ln(0.4) + \ln(0.5) + \ln(0.333)$$
$$= -0.916 + (-0.693) + (-1.099) = -2.708$$

**Classification**: $\arg\max(-3.283, -2.708) = \text{ham}$

**Key Advantage**: Log-space avoids numerical underflow when multiplying many small probabilities.

---

### Worked Example 4: Information Gain for ID3

**Setup**: Binary classification task with 9 instances:
- 5 spam, 4 ham

**Feature**: `free` (binary: 0 or 1)

**Data Split**:
- `free = 1`: 4 instances (3 spam, 1 ham)
- `free = 0`: 5 instances (2 spam, 3 ham)

---

**Step 1: Compute Entropy of Label Distribution**

$$H(Y) = -\sum_{k} p_k \log_2 p_k$$

$$p_{\text{spam}} = \frac{5}{9}, \quad p_{\text{ham}} = \frac{4}{9}$$

$$H(Y) = -\left(\frac{5}{9} \log_2 \frac{5}{9} + \frac{4}{9} \log_2 \frac{4}{9}\right)$$

Compute:
$$\frac{5}{9} \log_2 \frac{5}{9} = 0.556 \cdot (-0.848) = -0.471$$

$$\frac{4}{9} \log_2 \frac{4}{9} = 0.444 \cdot (-1.170) = -0.520$$

$$H(Y) = -(-0.471 - 0.520) = 0.991 \text{ bits}$$

---

**Step 2: Compute Conditional Entropy $H(Y \mid X)$**

$$H(Y \mid X) = \sum_{v \in X} P(v) H(Y \mid X=v)$$

**For `free = 1`** (4 instances: 3 spam, 1 ham):

$$H(Y \mid \text{free}=1) = -\left(\frac{3}{4} \log_2 \frac{3}{4} + \frac{1}{4} \log_2 \frac{1}{4}\right)$$

$$= -\left(0.75 \cdot (-0.415) + 0.25 \cdot (-2.000)\right) = -(-0.311 - 0.500) = 0.811$$

**For `free = 0`** (5 instances: 2 spam, 3 ham):

$$H(Y \mid \text{free}=0) = -\left(\frac{2}{5} \log_2 \frac{2}{5} + \frac{3}{5} \log_2 \frac{3}{5}\right)$$

$$= -\left(0.4 \cdot (-1.322) + 0.6 \cdot (-0.737)\right) = -(-0.529 - 0.442) = 0.971$$

**Weighted Conditional Entropy**:

$$H(Y \mid X) = P(\text{free}=1) \cdot H(Y \mid \text{free}=1) + P(\text{free}=0) \cdot H(Y \mid \text{free}=0)$$

$$= \frac{4}{9} \cdot 0.811 + \frac{5}{9} \cdot 0.971 = 0.360 + 0.539 = 0.899$$

---

**Step 3: Compute Information Gain**

$$\text{IG}(Y, X) = H(Y) - H(Y \mid X) = 0.991 - 0.899 = 0.092 \text{ bits}$$

**Interpretation**: Feature `free` reduces entropy by 0.092 bits, providing modest information for classification.

---

### Worked Example 5: Gradient Descent Step (Logistic Regression)

**Setup**: Single training instance for binary classification.

**Parameters**:
- Weights: $\mathbf{w} = [0.5, -0.3]$
- Bias: $b = 0.1$
- Learning rate: $\eta = 0.1$

**Training Instance**:
- Features: $\mathbf{x} = [2, 1]$
- Label: $y = 1$

---

**Step 1: Compute Linear Combination**

$$z = \mathbf{w}^\top \mathbf{x} + b = w_1 x_1 + w_2 x_2 + b$$

$$z = (0.5)(2) + (-0.3)(1) + 0.1 = 1.0 - 0.3 + 0.1 = 0.8$$

---

**Step 2: Compute Prediction via Sigmoid**

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}} = \frac{1}{1 + e^{-0.8}}$$

$$e^{-0.8} \approx 0.449, \quad \hat{y} = \frac{1}{1.449} \approx 0.690$$

---

**Step 3: Compute Gradients**

**Gradient w.r.t. $w_1$**:
$$\frac{\partial \mathcal{L}}{\partial w_1} = (\hat{y} - y) x_1 = (0.690 - 1) \cdot 2 = -0.310 \cdot 2 = -0.620$$

**Gradient w.r.t. $w_2$**:
$$\frac{\partial \mathcal{L}}{\partial w_2} = (\hat{y} - y) x_2 = (0.690 - 1) \cdot 1 = -0.310$$

**Gradient w.r.t. $b$**:
$$\frac{\partial \mathcal{L}}{\partial b} = (\hat{y} - y) = 0.690 - 1 = -0.310$$

---

**Step 4: Update Parameters**

$$w_1 \leftarrow w_1 - \eta \frac{\partial \mathcal{L}}{\partial w_1} = 0.5 - 0.1 \cdot (-0.620) = 0.5 + 0.062 = 0.562$$

$$w_2 \leftarrow w_2 - \eta \frac{\partial \mathcal{L}}{\partial w_2} = -0.3 - 0.1 \cdot (-0.310) = -0.3 + 0.031 = -0.269$$

$$b \leftarrow b - \eta \frac{\partial \mathcal{L}}{\partial b} = 0.1 - 0.1 \cdot (-0.310) = 0.1 + 0.031 = 0.131$$

---

**Updated Parameters**:
- $\mathbf{w} = [0.562, -0.269]$
- $b = 0.131$

**Interpretation**: Weights move toward increasing $z$ (and thus $\hat{y}$) for this positive example, reducing loss.

---

## WEEK 4: Representation Learning & Neural Networks

### Core Formulas with Semantic Explanations

#### Pointwise Mutual Information (PMI)

| **Formula** | $\text{PMI}(w,c) = \log_2 \frac{P(w,c)}{P(w)P(c)}$ |
|-------------|---------------------------------------------------|
| **Variables** | $w$ = target word; $c$ = context word; $P(w,c)$ = joint probability; $P(w)$, $P(c)$ = marginals |
| **What it computes** | The log-ratio of observed co-occurrence to expected co-occurrence under independence. |
| **Semantic meaning** | Measures **association strength** beyond chance. If two words appear together more than their individual frequencies predict, PMI is positive (attraction). If less, PMI is negative (avoidance). PMI = 0 means exactly the expected co-occurrence under independence. |
| **Why the log?** | Converts multiplicative independence ($P(w,c) = P(w)P(c)$) into additive comparison. Also compresses the range. |

#### Positive PMI (PPMI)

| **Formula** | $\text{PPMI}(w,c) = \max(\text{PMI}(w,c), 0)$ |
|-------------|----------------------------------------------|
| **What it computes** | PMI clamped to non-negative values. |
| **Semantic meaning** | **Eliminates unreliable negative associations.** Negative PMI indicates avoidance, but reliably estimating non-co-occurrence requires enormous corpora. Most word pairs never co-occur, yielding $\text{PMI} = -\infty$. PPMI treats all non-associations as zero. |

#### Context Smoothing ($\alpha$-weighting)

| **Formula** | $P_\alpha(c) = \frac{\text{count}(c)^\alpha}{\sum_{c'} \text{count}(c')^\alpha}$ |
|-------------|---------------------------------------------------------------------------------|
| **Typical value** | $\alpha = 0.75$ |
| **What it computes** | A smoothed probability distribution over contexts, raising rare contexts while dampening frequent ones. |
| **Semantic meaning** | Without smoothing, very frequent context words (like "the") dominate PMI calculations. The $\alpha$ exponent **raises the relative probability of rare contexts**, giving them more influence in the denominator of PMI. |

#### Cosine Similarity for Embeddings

| **Formula** | $\cos(\vec{v},\vec{w}) = \frac{\vec{v} \cdot \vec{w}}{\lVert\vec{v}\rVert \lVert\vec{w}\rVert}$ |
|-------------|-----------------------------------------------------------------------------------------------|
| **What it computes** | The angle between two word vectors in embedding space. |
| **Semantic meaning** | In embedding space, semantically similar words cluster in similar directions. Cosine similarity measures this directional alignment, ignoring magnitude. Two synonyms will have cosine ≈ 1; unrelated words will have cosine ≈ 0. |

#### Skip-Gram Probability

| **Formula** | $P(+ \mid w,c) = \sigma(\vec{c} \cdot \vec{w}) = \frac{1}{1 + e^{-\vec{c} \cdot \vec{w}}}$ |
|-------------|-------------------------------------------------------------------------------------------|
| **Variables** | $\vec{w}$ = target word embedding; $\vec{c}$ = context word embedding |
| **What it computes** | The probability that word $c$ is a true context of word $w$. |
| **Semantic meaning** | **Dot product as unnormalized similarity**: large dot product → high sigmoid → high probability. The model learns embeddings such that true word-context pairs have high dot products while random pairs have low dot products. |

#### Negative Sampling Loss

| **Formula** | $L = -\log\sigma(\vec{c}_{\text{pos}} \cdot \vec{w}) - \sum_{i=1}^{k} \log\sigma(-\vec{c}_{\text{neg}_i} \cdot \vec{w})$ |
|-------------|------------------------------------------------------------------------------------------------------------------------|
| **Variables** | $\vec{c}_{\text{pos}}$ = true context embedding; $\vec{c}_{\text{neg}_i}$ = sampled noise embeddings; $k$ = number of negative samples |
| **What it computes** | Binary cross-entropy loss for distinguishing true contexts from noise. |
| **Semantic meaning** | **First term**: Push the true context closer to the target (increase dot product). **Second term**: Push noise contexts away from the target (decrease dot products). Minimizing this loss shapes the embedding space so that real co-occurrences are close while random pairs are far. |

#### Single Neuron Computation

| **Formula** | $z = \mathbf{w} \cdot \mathbf{x} + b$; $\quad y = f(z)$ |
|-------------|--------------------------------------------------------|
| **Variables** | $\mathbf{w}$ = weights; $\mathbf{x}$ = inputs; $b$ = bias; $f$ = activation function |
| **What it computes** | Weighted sum of inputs, shifted by bias, then passed through nonlinearity. |
| **Semantic meaning** | Each weight $w_i$ determines how much input $x_i$ contributes to the decision. The bias shifts the activation threshold. The nonlinearity $f$ allows modeling non-linear decision boundaries. |

#### Feedforward Layer

| **Formula** | $\mathbf{h} = g(W\mathbf{x} + \mathbf{b})$ |
|-------------|-------------------------------------------|
| **Variables** | $W$ = weight matrix ($n_{\text{out}} \times n_{\text{in}}$); $g$ = activation function (element-wise) |
| **What it computes** | Transforms input vector through learned linear transformation plus nonlinearity. |
| **Semantic meaning** | The hidden layer learns an **internal representation** useful for the task. Each row of $W$ defines one hidden unit; each unit detects a different feature combination of the inputs. |

#### Softmax Function

| **Formula** | $\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$ |
|-------------|---------------------------------------------------------------|
| **What it computes** | Converts a vector of raw scores (logits) into a probability distribution. |
| **Semantic meaning** | **Exponential amplifies differences**: the largest logit gets disproportionately more probability. All outputs are positive and sum to 1, making this a valid probability distribution over $K$ classes. |

#### Cross-Entropy Loss

| **Formula** | $L = -\sum_{k=1}^{K} y_k \log \hat{y}_k$ (or simply $L = -\log \hat{y}_c$ for one-hot labels) |
|-------------|-----------------------------------------------------------------------------------------------|
| **Variables** | $y_k$ = true label (one-hot); $\hat{y}_k$ = predicted probability; $c$ = correct class index |
| **What it computes** | The negative log probability of the correct class. |
| **Semantic meaning** | **Heavily penalizes confident wrong predictions.** If $\hat{y}_c = 0.01$, then $L = -\log(0.01) = 4.6$. If $\hat{y}_c = 0.99$, then $L = 0.01$. Encourages the model to be confident about the correct class. |

#### Gradient Descent Update

| **Formula** | $\theta \leftarrow \theta - \eta \nabla_\theta L$ |
|-------------|--------------------------------------------------|
| **Variables** | $\theta$ = parameters; $\eta$ = learning rate; $\nabla_\theta L$ = gradient of loss |
| **What it computes** | Updates parameters in the direction that decreases loss. |
| **Semantic meaning** | The gradient points toward steepest loss increase; we move in the **opposite direction** to reduce loss. Learning rate controls step size: too large → overshoot; too small → slow convergence. |

---

### Activation Functions Reference

| Function | Formula | Derivative | Output Range | Key Property |
|----------|---------|------------|--------------|--------------|
| **Sigmoid** | $\sigma(z) = \frac{1}{1+e^{-z}}$ | $\sigma(z)(1-\sigma(z))$ | $(0,1)$ | Squashes to probabilities; vanishing gradient for large $\lvert z\rvert$ |
| **Tanh** | $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - \tanh^2(z)$ | $(-1,1)$ | Zero-centered; still has vanishing gradient |
| **ReLU** | $\max(0, z)$ | $0$ if $z<0$; $1$ if $z \geq 0$ | $[0,\infty)$ | Sparse, efficient; no vanishing gradient for $z>0$; "dead neurons" possible |

---

### Key Definitions

| Term | Definition |
|------|------------|
| **Distributional Hypothesis** | Words appearing in similar contexts have similar meanings—"know a word by the company it keeps." |
| **Sparse Embeddings** | High-dimensional vectors (size $\lvert V\rvert$), mostly zeros (TF-IDF, PPMI matrices). |
| **Dense Embeddings** | Low-dimensional learned vectors (50–300 dims), real-valued (Word2Vec, GloVe). |
| **Skip-Gram** | Word2Vec variant: predict context words given target word. |
| **XOR Problem** | Demonstrates single-layer perceptrons cannot solve non-linearly separable problems; motivates hidden layers. |
| **Computation Graph** | DAG representing forward computation; enables automatic differentiation via backpropagation. |
| **Backpropagation** | Algorithm applying chain rule backwards through computation graph to compute gradients. |

---

### Worked Example 1: PPMI Calculation

**Given** co-occurrence matrix (context window ±1):

|             | computer | data | result | pie | sugar | **Row Sum** |
|-------------|----------|------|--------|-----|-------|-------------|
| cherry      | 0        | 0    | 0      | 5   | 10    | 15          |
| strawberry  | 0        | 0    | 0      | 3   | 8     | 11          |
| digital     | 10       | 20   | 5      | 0   | 0     | 35          |
| information | 15       | 25   | 10     | 0   | 0     | 50          |
| **Col Sum** | 25       | 45   | 15     | 8   | 18    | **111**     |

---

**Step 1: Compute Marginal Probabilities**

$$P(\text{cherry}) = \frac{15}{111} = 0.135, \quad P(\text{digital}) = \frac{35}{111} = 0.315, \quad P(\text{information}) = \frac{50}{111} = 0.450$$

$$P(\text{sugar}) = \frac{18}{111} = 0.162, \quad P(\text{data}) = \frac{45}{111} = 0.405$$

---

**Step 2: Compute PPMI for Selected Pairs**

**PPMI(cherry, sugar)**:
$$P(\text{cherry}, \text{sugar}) = \frac{10}{111} = 0.090$$
$$\text{PMI} = \log_2 \frac{0.090}{0.135 \times 0.162} = \log_2 \frac{0.090}{0.0219} = \log_2 4.11 = 2.04$$
$$\text{PPMI(cherry, sugar)} = \max(2.04, 0) = \mathbf{2.04}$$

**PPMI(digital, sugar)**:
$$P(\text{digital}, \text{sugar}) = \frac{0}{111} = 0$$
$$\text{PMI} = \log_2 0 = -\infty$$
$$\text{PPMI(digital, sugar)} = \max(-\infty, 0) = \mathbf{0}$$

**PPMI(information, data)**:
$$P(\text{information}, \text{data}) = \frac{25}{111} = 0.225$$
$$\text{PMI} = \log_2 \frac{0.225}{0.450 \times 0.405} = \log_2 \frac{0.225}{0.182} = \log_2 1.24 = 0.31$$
$$\text{PPMI(information, data)} = \mathbf{0.31}$$

---

**Interpretation**: 
- Cherry–sugar has high PPMI (2.04) → strong co-occurrence beyond chance (food context)
- Digital–sugar has PPMI = 0 → never co-occur (different semantic domains)
- Information–data has moderate PPMI (0.31) → co-occur slightly more than chance

---

### Worked Example 2: Neural Network Forward Pass (Single Neuron)

**Given**:
- Weights: $\mathbf{w} = [0.2, 0.3, 0.9]$
- Bias: $b = 0.5$
- Input: $\mathbf{x} = [0.5, 0.6, 0.1]$
- Activation: Sigmoid

---

**Step 1: Compute Weighted Sum (Pre-activation)**

$$z = \mathbf{w} \cdot \mathbf{x} + b = (0.2)(0.5) + (0.3)(0.6) + (0.9)(0.1) + 0.5$$

$$z = 0.10 + 0.18 + 0.09 + 0.5 = 0.87$$

---

**Step 2: Apply Activation Function**

$$y = \sigma(0.87) = \frac{1}{1 + e^{-0.87}} = \frac{1}{1 + 0.419} = \frac{1}{1.419} = \mathbf{0.705}$$

---

**Interpretation**: The neuron produces output 0.705 ≈ 70.5%, meaning it "fires" fairly strongly for this input pattern.

---

### Worked Example 3: Two-Layer Network Forward Pass

**Given**:
- Input: $\mathbf{x} = [1.0, 2.0]^T$ (2-dimensional)
- Hidden layer: 2 units with ReLU activation
- Output layer: 3 classes with softmax

**Weights**:
$$W^{[1]} = \begin{bmatrix} 0.5 & -0.5 \\ 0.3 & 0.8 \end{bmatrix}, \quad \mathbf{b}^{[1]} = \begin{bmatrix} 0.1 \\ -0.2 \end{bmatrix}$$

$$W^{[2]} = \begin{bmatrix} 0.4 & 0.2 \\ -0.3 & 0.6 \\ 0.1 & -0.1 \end{bmatrix}, \quad \mathbf{b}^{[2]} = \begin{bmatrix} 0.0 \\ 0.1 \\ -0.1 \end{bmatrix}$$

---

**Step 1: Hidden Layer Pre-activation**

$$\mathbf{z}^{[1]} = W^{[1]}\mathbf{x} + \mathbf{b}^{[1]} = \begin{bmatrix} 0.5 & -0.5 \\ 0.3 & 0.8 \end{bmatrix} \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix} + \begin{bmatrix} 0.1 \\ -0.2 \end{bmatrix}$$

$$= \begin{bmatrix} 0.5(1) + (-0.5)(2) \\ 0.3(1) + 0.8(2) \end{bmatrix} + \begin{bmatrix} 0.1 \\ -0.2 \end{bmatrix} = \begin{bmatrix} -0.5 \\ 1.9 \end{bmatrix} + \begin{bmatrix} 0.1 \\ -0.2 \end{bmatrix} = \begin{bmatrix} -0.4 \\ 1.7 \end{bmatrix}$$

---

**Step 2: Apply ReLU Activation**

$$\mathbf{h} = \text{ReLU}(\mathbf{z}^{[1]}) = \begin{bmatrix} \max(0, -0.4) \\ \max(0, 1.7) \end{bmatrix} = \begin{bmatrix} 0 \\ 1.7 \end{bmatrix}$$

---

**Step 3: Output Layer Pre-activation (Logits)**

$$\mathbf{z}^{[2]} = W^{[2]}\mathbf{h} + \mathbf{b}^{[2]} = \begin{bmatrix} 0.4 & 0.2 \\ -0.3 & 0.6 \\ 0.1 & -0.1 \end{bmatrix} \begin{bmatrix} 0 \\ 1.7 \end{bmatrix} + \begin{bmatrix} 0.0 \\ 0.1 \\ -0.1 \end{bmatrix}$$

$$= \begin{bmatrix} 0.4(0) + 0.2(1.7) \\ -0.3(0) + 0.6(1.7) \\ 0.1(0) + (-0.1)(1.7) \end{bmatrix} + \begin{bmatrix} 0.0 \\ 0.1 \\ -0.1 \end{bmatrix} = \begin{bmatrix} 0.34 \\ 1.02 \\ -0.17 \end{bmatrix} + \begin{bmatrix} 0.0 \\ 0.1 \\ -0.1 \end{bmatrix} = \begin{bmatrix} 0.34 \\ 1.12 \\ -0.27 \end{bmatrix}$$

---

**Step 4: Apply Softmax**

$$\sum_j e^{z_j} = e^{0.34} + e^{1.12} + e^{-0.27} = 1.40 + 3.06 + 0.76 = 5.22$$

$$\hat{\mathbf{y}} = \begin{bmatrix} \frac{1.40}{5.22} \\ \frac{3.06}{5.22} \\ \frac{0.76}{5.22} \end{bmatrix} = \begin{bmatrix} 0.268 \\ 0.586 \\ 0.146 \end{bmatrix}$$

---

**Step 5: Compute Cross-Entropy Loss** (if true class is 1, i.e., $y = [0, 1, 0]^T$)

$$L = -\log \hat{y}_1 = -\log(0.586) = 0.534$$

---

**Interpretation**: The network predicts class 1 with 58.6% probability. Since that's the correct class, the loss is moderate (0.534). Perfect confidence would give $L = 0$.

---

### Worked Example 4: Backpropagation Gradient Calculation

**Given** a single sigmoid unit for binary classification:
- Forward: $z = w_1 x_1 + w_2 x_2 + b$, then $\hat{y} = \sigma(z)$
- Loss: $L = -(y \log \hat{y} + (1-y)\log(1-\hat{y}))$
- For sigmoid + cross-entropy: $\frac{\partial L}{\partial z} = \hat{y} - y$

**Values**: $x_1 = 0.8$, $x_2 = 0.4$, $w_1 = 0.5$, $w_2 = 0.3$, $b = -0.1$, $y = 1$ (positive class)

---

**Forward Pass**:
$$z = 0.5(0.8) + 0.3(0.4) + (-0.1) = 0.40 + 0.12 - 0.1 = 0.42$$
$$\hat{y} = \sigma(0.42) = \frac{1}{1 + e^{-0.42}} = \frac{1}{1.657} = 0.603$$

---

**Backward Pass (Gradients)**:

The key gradient (for cross-entropy + sigmoid):
$$\frac{\partial L}{\partial z} = \hat{y} - y = 0.603 - 1 = -0.397$$

Apply chain rule for each weight:
$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w_1} = (-0.397)(x_1) = (-0.397)(0.8) = -0.318$$

$$\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w_2} = (-0.397)(x_2) = (-0.397)(0.4) = -0.159$$

$$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial b} = (-0.397)(1) = -0.397$$

---

**Gradient Descent Update** (learning rate $\eta = 0.1$):

$$w_1 \leftarrow 0.5 - 0.1(-0.318) = 0.5 + 0.032 = 0.532$$
$$w_2 \leftarrow 0.3 - 0.1(-0.159) = 0.3 + 0.016 = 0.316$$
$$b \leftarrow -0.1 - 0.1(-0.397) = -0.1 + 0.040 = -0.060$$

---

**Interpretation**: The gradients are negative because we want to **increase** $\hat{y}$ toward $y = 1$. The update adds to each weight, pushing the pre-activation $z$ higher, which increases $\sigma(z)$.

---

## WEEK 5: Sequence Labeling & Information Extraction

### Core Formulas with Semantic Explanations

#### HMM Definition (5-tuple)

| **Definition** | $\lambda = (Q, A, O, B, \pi)$ |
|----------------|------------------------------|
| **Components** | $Q$ = hidden states (e.g., POS tags); $A$ = transition matrix; $O$ = observations (words); $B$ = emission matrix; $\pi$ = initial state distribution |
| **What it represents** | A generative probabilistic model for sequences where hidden states generate observations. |
| **Semantic meaning** | The model assumes we observe words but not their underlying grammatical categories (tags). It learns statistical patterns of which tags follow which, and which words each tag produces. |

#### Markov Assumption

| **Formula** | $P(q_i \mid q_1, \ldots, q_{i-1}) = P(q_i \mid q_{i-1})$ |
|-------------|----------------------------------------------------------|
| **What it computes** | The probability of the current state depends only on the previous state. |
| **Semantic meaning** | **History compression**: instead of conditioning on the entire sequence history (exponentially many possibilities), we assume the immediate past is sufficient. This makes computation tractable. |

#### Output Independence Assumption

| **Formula** | $P(o_i \mid q_1, \ldots, q_T, o_1, \ldots, o_T) = P(o_i \mid q_i)$ |
|-------------|-------------------------------------------------------------------|
| **What it computes** | The observation at position $i$ depends only on the hidden state at position $i$. |
| **Semantic meaning** | Each word is generated solely based on its tag—not influenced by neighboring words or tags. This is a strong (often unrealistic) assumption that simplifies the model. |

#### Transition Probability (MLE)

| **Formula** | $P(t_i \mid t_{i-1}) = \frac{C(t_{i-1}, t_i)}{C(t_{i-1})}$ |
|-------------|-----------------------------------------------------------|
| **Variables** | $C(t_{i-1}, t_i)$ = count of tag bigram; $C(t_{i-1})$ = count of first tag |
| **What it computes** | The conditional probability of transitioning from one tag to another. |
| **Semantic meaning** | Captures **grammatical constraints**: after a determiner, a noun is likely; after a noun, a verb is likely. These probabilities encode sequential structure learned from tagged corpora. |

#### Emission Probability (MLE)

| **Formula** | $P(w_i \mid t_i) = \frac{C(t_i, w_i)}{C(t_i)}$ |
|-------------|------------------------------------------------|
| **Variables** | $C(t_i, w_i)$ = count of word $w$ with tag $t$; $C(t_i)$ = total count of tag $t$ |
| **What it computes** | The probability of generating a specific word from a tag. |
| **Semantic meaning** | Captures **lexical information**: "the" is almost always a determiner, "cat" is usually a noun. Words have preferred tags based on their grammatical function. |

#### Viterbi Initialization

| **Formula** | $v_1(j) = \pi_j \cdot b_j(o_1)$ |
|-------------|--------------------------------|
| **Variables** | $\pi_j$ = probability of starting in state $j$; $b_j(o_1)$ = emission probability of first observation from state $j$ |
| **What it computes** | The probability of the best path to state $j$ at the first time step. |
| **Semantic meaning** | At the start, there's only one way to reach any state: begin there and emit the first word. This initializes the dynamic programming table. |

#### Viterbi Recurrence

| **Formula** | $v_t(j) = \max_i \left[ v_{t-1}(i) \cdot a_{ij} \cdot b_j(o_t) \right]$ |
|-------------|-----------------------------------------------------------------------|
| **Variables** | $v_{t-1}(i)$ = best path probability to state $i$ at $t-1$; $a_{ij}$ = transition probability; $b_j(o_t)$ = emission probability |
| **What it computes** | The probability of the most probable path ending in state $j$ at time $t$. |
| **Semantic meaning** | To reach state $j$ at time $t$, we could have come from any state $i$ at $t-1$. We take the **maximum** over all such paths—this is dynamic programming exploiting optimal substructure. |

#### Viterbi Backpointer

| **Formula** | $\text{bt}_t(j) = \arg\max_i \left[ v_{t-1}(i) \cdot a_{ij} \right]$ |
|-------------|---------------------------------------------------------------------|
| **What it computes** | The predecessor state that gives the best path to state $j$. |
| **Semantic meaning** | Records **which state we came from** to achieve the maximum. After completing the forward pass, we trace these pointers backward to recover the full optimal tag sequence. |

#### CRF Probability

| **Formula** | $P(Y \mid X) = \frac{1}{Z(X)} \exp\left(\sum_k w_k F_k(X, Y)\right)$ |
|-------------|---------------------------------------------------------------------|
| **Variables** | $Y$ = label sequence; $X$ = observation sequence; $F_k$ = feature functions; $w_k$ = learned weights; $Z(X)$ = partition function |
| **What it computes** | The conditional probability of a label sequence given observations. |
| **Semantic meaning** | Unlike HMMs (which model $P(X,Y)$), CRFs directly model **what we want**: $P(Y \mid X)$. This allows **arbitrary overlapping features** (word shapes, prefixes, entire context) without modeling their distribution. |

---

### BIO Tagging Scheme

| Tag | Meaning | Usage |
|-----|---------|-------|
| **B-X** | **B**eginning of entity type X | First token of a multi-token entity |
| **I-X** | **I**nside entity type X | Continuation tokens of an entity |
| **O** | **O**utside any entity | Non-entity tokens |

**Example**: "Jane Smith works at United Nations"

| Token | Tag | Explanation |
|-------|-----|-------------|
| Jane | B-PER | Begins a person entity |
| Smith | I-PER | Continues the person entity |
| works | O | Not an entity |
| at | O | Not an entity |
| United | B-ORG | Begins an organization entity |
| Nations | I-ORG | Continues the organization entity |

---

### HMM vs CRF Comparison

| Aspect | HMM | CRF |
|--------|-----|-----|
| **Model Type** | Generative: models $P(X,Y)$ | Discriminative: models $P(Y \mid X)$ directly |
| **Features** | Limited to transition + emission | Arbitrary overlapping features |
| **Feature Access** | Current observation only | Entire input sequence $X$ |
| **Viterbi Operation** | Multiplication | Addition (log-space) |
| **Training** | Maximum Likelihood (count-based) | Conditional log-likelihood (gradient-based) |
| **Unknown Words** | Problematic (zero emission) | Handles via word features (shape, suffix) |

---

### Worked Example 1: Viterbi Algorithm — "the cat sat"

**Given**:
- States: $Q = \{\text{DET}, \text{NN}, \text{VB}\}$
- Observations: $O = [\text{the}, \text{cat}, \text{sat}]$

**Transition Probabilities $A$** (from rows to columns):

| From \ To | DET | NN | VB |
|-----------|-----|----|----|
| ⟨s⟩ (start) | 0.8 | 0.1 | 0.1 |
| DET | 0.1 | 0.8 | 0.1 |
| NN | 0.1 | 0.2 | 0.7 |
| VB | 0.4 | 0.3 | 0.3 |

**Emission Probabilities $B$**:

| State | the | cat | sat |
|-------|-----|-----|-----|
| DET | 0.9 | 0.01 | 0.01 |
| NN | 0.01 | 0.7 | 0.1 |
| VB | 0.01 | 0.1 | 0.8 |

---

**Step 1: Initialization ($t = 1$, word = "the")**

$$v_1(\text{DET}) = P(\text{DET} \mid \langle s\rangle) \times P(\text{the} \mid \text{DET}) = 0.8 \times 0.9 = \mathbf{0.720}$$
$$v_1(\text{NN}) = 0.1 \times 0.01 = 0.001$$
$$v_1(\text{VB}) = 0.1 \times 0.01 = 0.001$$

| State | $v_1$ | Backpointer |
|-------|-------|-------------|
| DET | **0.720** | ⟨s⟩ |
| NN | 0.001 | ⟨s⟩ |
| VB | 0.001 | ⟨s⟩ |

---

**Step 2: Recursion ($t = 2$, word = "cat")**

For each state $j$, compute $v_2(j) = \max_i [v_1(i) \cdot a_{ij} \cdot b_j(\text{cat})]$:

**$v_2(\text{DET})$**:
- From DET: $0.720 \times 0.1 \times 0.01 = 0.00072$
- From NN: $0.001 \times 0.1 \times 0.01 = 0.000001$
- From VB: $0.001 \times 0.4 \times 0.01 = 0.000004$
- Max: **0.00072** (from DET)

**$v_2(\text{NN})$**:
- From DET: $0.720 \times 0.8 \times 0.7 = \mathbf{0.4032}$
- From NN: $0.001 \times 0.2 \times 0.7 = 0.00014$
- From VB: $0.001 \times 0.3 \times 0.7 = 0.00021$
- Max: **0.4032** (from DET)

**$v_2(\text{VB})$**:
- From DET: $0.720 \times 0.1 \times 0.1 = 0.0072$
- From NN: $0.001 \times 0.7 \times 0.1 = 0.00007$
- From VB: $0.001 \times 0.3 \times 0.1 = 0.00003$
- Max: **0.0072** (from DET)

| State | $v_2$ | Backpointer |
|-------|-------|-------------|
| DET | 0.00072 | DET |
| NN | **0.4032** | DET |
| VB | 0.0072 | DET |

---

**Step 3: Recursion ($t = 3$, word = "sat")**

**$v_3(\text{DET})$**:
- From DET: $0.00072 \times 0.1 \times 0.01 = 0.00000072$
- From NN: $0.4032 \times 0.1 \times 0.01 = 0.000403$
- From VB: $0.0072 \times 0.4 \times 0.01 = 0.0000288$
- Max: **0.000403** (from NN)

**$v_3(\text{NN})$**:
- From DET: $0.00072 \times 0.8 \times 0.1 = 0.0000576$
- From NN: $0.4032 \times 0.2 \times 0.1 = 0.00806$
- From VB: $0.0072 \times 0.3 \times 0.1 = 0.000216$
- Max: **0.00806** (from NN)

**$v_3(\text{VB})$**:
- From DET: $0.00072 \times 0.1 \times 0.8 = 0.0000576$
- From NN: $0.4032 \times 0.7 \times 0.8 = \mathbf{0.2258}$
- From VB: $0.0072 \times 0.3 \times 0.8 = 0.00173$
- Max: **0.2258** (from NN)

| State | $v_3$ | Backpointer |
|-------|-------|-------------|
| DET | 0.000403 | NN |
| NN | 0.00806 | NN |
| VB | **0.2258** | NN |

---

**Step 4: Termination & Backtrace**

Best final state: $\arg\max_j v_3(j) = \text{VB}$ with probability 0.2258

Backtrace:
- $t=3$: VB, backpointer → NN
- $t=2$: NN, backpointer → DET
- $t=1$: DET

**Final tagging**: the/DET cat/NN sat/VB ✓

---

### Worked Example 2: HMM Parameter Estimation from Corpus

**Given** tagged training corpus:
```
the/DET cat/NN sat/VB
the/DET dog/NN ran/VB
a/DET cat/NN slept/VB
```

---

**Count Tag Unigrams**:
- $C(\text{DET}) = 3$ (the, the, a)
- $C(\text{NN}) = 3$ (cat, dog, cat)
- $C(\text{VB}) = 3$ (sat, ran, slept)

**Count Tag Bigrams**:
- $C(\langle s\rangle, \text{DET}) = 3$ (all sentences start with DET)
- $C(\text{DET}, \text{NN}) = 3$ (DET always followed by NN)
- $C(\text{NN}, \text{VB}) = 3$ (NN always followed by VB)
- $C(\text{VB}, \langle/s\rangle) = 3$ (VB always ends sentence)

**Count Word-Tag Pairs**:
- $C(\text{DET}, \text{the}) = 2$
- $C(\text{DET}, \text{a}) = 1$
- $C(\text{NN}, \text{cat}) = 2$
- $C(\text{NN}, \text{dog}) = 1$
- $C(\text{VB}, \text{sat}) = 1$
- $C(\text{VB}, \text{ran}) = 1$
- $C(\text{VB}, \text{slept}) = 1$

---

**Transition Probabilities**:

$$P(\text{DET} \mid \langle s\rangle) = \frac{C(\langle s\rangle, \text{DET})}{C(\langle s\rangle)} = \frac{3}{3} = 1.0$$

$$P(\text{NN} \mid \text{DET}) = \frac{C(\text{DET}, \text{NN})}{C(\text{DET})} = \frac{3}{3} = 1.0$$

$$P(\text{VB} \mid \text{NN}) = \frac{C(\text{NN}, \text{VB})}{C(\text{NN})} = \frac{3}{3} = 1.0$$

---

**Emission Probabilities**:

$$P(\text{the} \mid \text{DET}) = \frac{C(\text{DET}, \text{the})}{C(\text{DET})} = \frac{2}{3} = 0.667$$

$$P(\text{a} \mid \text{DET}) = \frac{C(\text{DET}, \text{a})}{C(\text{DET})} = \frac{1}{3} = 0.333$$

$$P(\text{cat} \mid \text{NN}) = \frac{C(\text{NN}, \text{cat})}{C(\text{NN})} = \frac{2}{3} = 0.667$$

$$P(\text{dog} \mid \text{NN}) = \frac{C(\text{NN}, \text{dog})}{C(\text{NN})} = \frac{1}{3} = 0.333$$

---

**Interpretation**: This tiny corpus yields deterministic transitions (DET→NN→VB) but varied emissions. In reality, larger corpora produce smoother probability distributions.

---

### Worked Example 3: BIO Tagging

**Sentence**: "Marie Curie was born in Warsaw in 1867"

| Token | Entity Type | BIO Tag | Explanation |
|-------|-------------|---------|-------------|
| Marie | Person | B-PER | Begins person entity |
| Curie | Person | I-PER | Continues person entity |
| was | — | O | Outside any entity |
| born | — | O | Outside any entity |
| in | — | O | Outside any entity |
| Warsaw | Location | B-LOC | Begins location entity (single token) |
| in | — | O | Outside any entity |
| 1867 | Date | B-DATE | Begins date entity (single token) |

**Key insight**: The B- prefix marks entity boundaries. Without it, we couldn't distinguish:
- "[Marie Curie] [Pierre Curie]" (two PER entities)
- "[Marie Curie Pierre Curie]" (one PER entity)

The BIO encoding makes this unambiguous: B-PER marks each entity's start.

---

### Worked Example 4: Viterbi with Larger Tagset — "Janet will back the bill"

**States**: {NNP (proper noun), MD (modal), VB (verb), JJ (adjective), NN (noun), RB (adverb), DT (determiner)}

**Key Ambiguities**:
- "Janet": NNP (likely proper noun)
- "will": MD (modal) or NN (testament)
- "back": VB (verb), JJ (adjective), NN (noun), or RB (adverb)
- "the": DT (determiner)
- "bill": NN (noun) or VB (verb)

**Transition/Emission Information** (from WSJ corpus patterns):
- $P(\text{MD} \mid \text{NNP})$ is moderate (names can be followed by modals)
- $P(\text{VB} \mid \text{MD})$ is high (modals are typically followed by verbs)
- $P(\text{DT} \mid \text{VB})$ is moderate (verbs can be followed by NPs starting with DT)
- $P(\text{NN} \mid \text{DT})$ is high (determiners are typically followed by nouns)

**Best Path Summary** (full trellis omitted for brevity):

| Position | Word | Best Tag | Reasoning |
|----------|------|----------|-----------|
| 1 | Janet | NNP | High emission prob for capitalized word |
| 2 | will | MD | High $P(\text{MD} \mid \text{NNP})$, high $P(\text{will} \mid \text{MD})$ |
| 3 | back | VB | High $P(\text{VB} \mid \text{MD})$—modals followed by verbs |
| 4 | the | DT | High $P(\text{DT} \mid \text{VB})$, $P(\text{the} \mid \text{DT}) \approx 1$ |
| 5 | bill | NN | High $P(\text{NN} \mid \text{DT})$, "bill" is common noun |

**Final tagging**: Janet/NNP will/MD back/VB the/DT bill/NN

**Interpretation**: The Viterbi algorithm resolves the ambiguity of "back" (which could be verb, adjective, adverb, or noun) by considering the context: "will back" strongly suggests verb usage because modals are almost always followed by verbs.

---

### Worked Example 5: Complete HMM Construction and Viterbi Decoding — "She can fish in rivers"

This comprehensive example demonstrates the full pipeline: (1) estimating HMM parameters from a tagged training corpus, (2) constructing transition and emission probability matrices, and (3) applying the Viterbi algorithm to find the most probable POS tag sequence for a 5-word sentence with lexical ambiguity.

---

#### Part A: Training Corpus and Parameter Estimation

**Given**: A small tagged training corpus (6 sentences):

```
She/PRP runs/VBZ fast/RB
They/PRP can/MD swim/VB
I/PRP fish/VB in/IN lakes/NNS
The/DT can/NN is/VBZ empty/JJ
Fish/NN swim/VB in/IN rivers/NNS
She/PRP can/MD run/VB in/IN parks/NNS
```

**Target sentence to tag**: `"She can fish in rivers"`

**Key Ambiguities**:
- **"can"**: MD (modal verb, "She can swim") or NN (noun, "a can of beans")
- **"fish"**: VB (verb, "I fish daily") or NN (noun, "the fish swims")

**States (POS Tags)**: $Q = \{\text{PRP}, \text{VBZ}, \text{RB}, \text{MD}, \text{VB}, \text{IN}, \text{NNS}, \text{DT}, \text{NN}, \text{JJ}\}$

---

**Step A.1: Count Tag Unigrams**

Count the total occurrences of each tag in the corpus:

| Tag | Occurrences | Count |
|-----|-------------|-------|
| PRP | She, They, I, She | 4 |
| VBZ | runs, is | 2 |
| RB | fast | 1 |
| MD | can, can | 2 |
| VB | swim, fish, swim, run | 4 |
| IN | in, in, in | 3 |
| NNS | lakes, rivers, parks | 3 |
| DT | The | 1 |
| NN | can, Fish | 2 |
| JJ | empty | 1 |

**Total tags**: $N = 23$

---

**Step A.2: Count Tag Bigrams**

Count transitions between consecutive tags (including start symbol $\langle s \rangle$):

| Bigram | Occurrences | Count |
|--------|-------------|-------|
| $\langle s \rangle \to \text{PRP}$ | 4 sentences start with PRP | 4 |
| $\langle s \rangle \to \text{DT}$ | 1 sentence starts with DT | 1 |
| $\langle s \rangle \to \text{NN}$ | 1 sentence starts with NN | 1 |
| $\text{PRP} \to \text{VBZ}$ | She→runs | 1 |
| $\text{PRP} \to \text{MD}$ | They→can, She→can | 2 |
| $\text{PRP} \to \text{VB}$ | I→fish | 1 |
| $\text{VBZ} \to \text{RB}$ | runs→fast | 1 |
| $\text{VBZ} \to \text{JJ}$ | is→empty | 1 |
| $\text{MD} \to \text{VB}$ | can→swim, can→run | 2 |
| $\text{VB} \to \text{IN}$ | fish→in, swim→in, run→in | 3 |
| $\text{VB} \to \langle /s \rangle$ | swim (end of sentence 2) | 1 |
| $\text{IN} \to \text{NNS}$ | in→lakes, in→rivers, in→parks | 3 |
| $\text{DT} \to \text{NN}$ | The→can | 1 |
| $\text{NN} \to \text{VBZ}$ | can→is | 1 |
| $\text{NN} \to \text{VB}$ | Fish→swim | 1 |

---

**Step A.3: Compute Transition Probabilities (MLE)**

$$P(t_i \mid t_{i-1}) = \frac{C(t_{i-1}, t_i)}{C(t_{i-1})}$$

**Initial State Distribution $\pi$ (transitions from $\langle s \rangle$)**:

$$P(\text{PRP} \mid \langle s \rangle) = \frac{C(\langle s \rangle, \text{PRP})}{C(\langle s \rangle)} = \frac{4}{6} = 0.667$$

$$P(\text{DT} \mid \langle s \rangle) = \frac{1}{6} = 0.167$$

$$P(\text{NN} \mid \langle s \rangle) = \frac{1}{6} = 0.167$$

All other tags: $P(t \mid \langle s \rangle) = 0$

**Transition Matrix $A$ (selected relevant entries)**:

| From $\downarrow$ \ To $\rightarrow$ | PRP | VBZ | RB | MD | VB | IN | NNS | DT | NN | JJ |
|--------------------------------------|-----|-----|----|----|----|----|-----|----|----|-----|
| $\langle s \rangle$ | 0.667 | 0 | 0 | 0 | 0 | 0 | 0 | 0.167 | 0.167 | 0 |
| PRP | 0 | 0.25 | 0 | 0.50 | 0.25 | 0 | 0 | 0 | 0 | 0 |
| MD | 0 | 0 | 0 | 0 | 1.0 | 0 | 0 | 0 | 0 | 0 |
| VB | 0 | 0 | 0 | 0 | 0 | 0.75 | 0 | 0 | 0 | 0 |
| IN | 0 | 0 | 0 | 0 | 0 | 0 | 1.0 | 0 | 0 | 0 |
| NN | 0 | 0.50 | 0 | 0 | 0.50 | 0 | 0 | 0 | 0 | 0 |

**Sample calculation for $P(\text{MD} \mid \text{PRP})$**:
$$P(\text{MD} \mid \text{PRP}) = \frac{C(\text{PRP}, \text{MD})}{C(\text{PRP})} = \frac{2}{4} = 0.50$$

**Sample calculation for $P(\text{VB} \mid \text{MD})$**:
$$P(\text{VB} \mid \text{MD}) = \frac{C(\text{MD}, \text{VB})}{C(\text{MD})} = \frac{2}{2} = 1.0$$

**Semantic insight**: The transition $P(\text{VB} \mid \text{MD}) = 1.0$ reflects the grammatical constraint that modal verbs (can, will, should) are always followed by base-form verbs in this corpus.

---

**Step A.4: Count Word-Tag Pairs**

| Word | Tag | Count |
|------|-----|-------|
| She | PRP | 2 |
| They | PRP | 1 |
| I | PRP | 1 |
| runs | VBZ | 1 |
| is | VBZ | 1 |
| fast | RB | 1 |
| can | MD | 2 |
| can | NN | 1 |
| swim | VB | 2 |
| fish | VB | 1 |
| run | VB | 1 |
| in | IN | 3 |
| lakes | NNS | 1 |
| rivers | NNS | 1 |
| parks | NNS | 1 |
| The | DT | 1 |
| Fish | NN | 1 |
| empty | JJ | 1 |

---

**Step A.5: Compute Emission Probabilities (MLE)**

$$P(w_i \mid t_i) = \frac{C(t_i, w_i)}{C(t_i)}$$

**Emission Matrix $B$ (relevant entries for target sentence)**:

| Tag | She | can | fish | in | rivers |
|-----|-----|-----|------|----|---------| 
| PRP | 0.50 | 0 | 0 | 0 | 0 |
| MD | 0 | 1.0 | 0 | 0 | 0 |
| VB | 0 | 0 | 0.25 | 0 | 0 |
| IN | 0 | 0 | 0 | 1.0 | 0 |
| NNS | 0 | 0 | 0 | 0 | 0.333 |
| NN | 0 | 0.333 | 0.333* | 0 | 0 |

*Note: We assume "fish" (lowercase) can be emitted by NN with probability 0.333, treating "Fish" and "fish" as the same lemma after normalization.

**Sample calculation for $P(\text{She} \mid \text{PRP})$**:
$$P(\text{She} \mid \text{PRP}) = \frac{C(\text{PRP}, \text{She})}{C(\text{PRP})} = \frac{2}{4} = 0.50$$

**Sample calculation for $P(\text{can} \mid \text{MD})$**:
$$P(\text{can} \mid \text{MD}) = \frac{C(\text{MD}, \text{can})}{C(\text{MD})} = \frac{2}{2} = 1.0$$

**Sample calculation for $P(\text{can} \mid \text{NN})$**:
$$P(\text{can} \mid \text{NN}) = \frac{C(\text{NN}, \text{can})}{C(\text{NN})} = \frac{1}{2} = 0.50$$

---

#### Part B: Viterbi Algorithm Application

**Observation sequence**: $O = [\text{She}, \text{can}, \text{fish}, \text{in}, \text{rivers}]$

**Relevant states**: For computational tractability, we focus on states that can emit the observed words: $\{\text{PRP}, \text{MD}, \text{NN}, \text{VB}, \text{IN}, \text{NNS}\}$

---

**Step B.1: Initialization ($t = 1$, word = "She")**

$$v_1(j) = P(j \mid \langle s \rangle) \times P(\text{She} \mid j)$$

| State | $\pi_j$ | $b_j(\text{She})$ | Calculation | $v_1(j)$ |
|-------|---------|-------------------|-------------|----------|
| PRP | 0.667 | 0.50 | $0.667 \times 0.50$ | **0.333** |
| MD | 0 | 0 | $0 \times 0$ | 0 |
| NN | 0.167 | 0 | $0.167 \times 0$ | 0 |
| VB | 0 | 0 | $0 \times 0$ | 0 |
| IN | 0 | 0 | $0 \times 0$ | 0 |
| NNS | 0 | 0 | $0 \times 0$ | 0 |

**Summary Table ($t = 1$)**:

| State | $v_1$ | Backpointer |
|-------|-------|-------------|
| PRP | **0.333** | $\langle s \rangle$ |
| MD | 0 | — |
| NN | 0 | — |
| VB | 0 | — |
| IN | 0 | — |
| NNS | 0 | — |

**Interpretation (sentence start)**: Only PRP has non-zero probability because "She" is exclusively a personal pronoun in our corpus, and sentences commonly begin with pronouns ($P(\text{PRP} \mid \langle s \rangle) = 0.667$).

---

**Step B.2: Recursion ($t = 2$, word = "can")**

$$v_2(j) = \max_i \left[ v_1(i) \cdot a_{ij} \cdot b_j(\text{can}) \right]$$

**$v_2(\text{MD})$** (can as modal verb):
- From PRP: $v_1(\text{PRP}) \times P(\text{MD} \mid \text{PRP}) \times P(\text{can} \mid \text{MD})$
  $$= 0.333 \times 0.50 \times 1.0 = \mathbf{0.167}$$
- All other predecessors: 0 (either $v_1(i) = 0$ or $a_{i,\text{MD}} = 0$)
- **Max: 0.167** (from PRP)

**$v_2(\text{NN})$** (can as noun):
- From PRP: $v_1(\text{PRP}) \times P(\text{NN} \mid \text{PRP}) \times P(\text{can} \mid \text{NN})$
  $$= 0.333 \times 0 \times 0.50 = 0$$
- All other predecessors: 0
- **Max: 0** (no valid path)

**Summary Table ($t = 2$)**:

| State | $v_2$ | Backpointer |
|-------|-------|-------------|
| PRP | 0 | — |
| MD | **0.167** | PRP |
| NN | 0 | — |
| VB | 0 | — |
| IN | 0 | — |
| NNS | 0 | — |

**Interpretation (first ambiguity resolved)**: Although "can" could be a modal (MD) or noun (NN), only the MD path survives because:
1. The transition $P(\text{NN} \mid \text{PRP}) = 0$ in our corpus (pronouns are never followed by nouns directly)
2. The transition $P(\text{MD} \mid \text{PRP}) = 0.50$ is valid (pronouns can be followed by modals)

The Viterbi algorithm exploits the grammatical constraint that personal pronouns are typically followed by verbs or modals, not nouns.

---

**Step B.3: Recursion ($t = 3$, word = "fish")**

$$v_3(j) = \max_i \left[ v_2(i) \cdot a_{ij} \cdot b_j(\text{fish}) \right]$$

**$v_3(\text{VB})$** (fish as verb):
- From MD: $v_2(\text{MD}) \times P(\text{VB} \mid \text{MD}) \times P(\text{fish} \mid \text{VB})$
  $$= 0.167 \times 1.0 \times 0.25 = \mathbf{0.0417}$$
- All other predecessors: 0
- **Max: 0.0417** (from MD)

**$v_3(\text{NN})$** (fish as noun):
- From MD: $v_2(\text{MD}) \times P(\text{NN} \mid \text{MD}) \times P(\text{fish} \mid \text{NN})$
  $$= 0.167 \times 0 \times 0.333 = 0$$
- All other predecessors: 0
- **Max: 0** (no valid path)

**Summary Table ($t = 3$)**:

| State | $v_3$ | Backpointer |
|-------|-------|-------------|
| PRP | 0 | — |
| MD | 0 | — |
| NN | 0 | — |
| VB | **0.0417** | MD |
| IN | 0 | — |
| NNS | 0 | — |

**Interpretation (second ambiguity resolved)**: "Fish" could be a verb (VB) or noun (NN), but only the VB path survives. The critical constraint is $P(\text{VB} \mid \text{MD}) = 1.0$: modals are always followed by verbs. The noun interpretation fails because $P(\text{NN} \mid \text{MD}) = 0$—you cannot say *"She can table"* in standard grammar.

---

**Step B.4: Recursion ($t = 4$, word = "in")**

$$v_4(j) = \max_i \left[ v_3(i) \cdot a_{ij} \cdot b_j(\text{in}) \right]$$

**$v_4(\text{IN})$** (in as preposition):
- From VB: $v_3(\text{VB}) \times P(\text{IN} \mid \text{VB}) \times P(\text{in} \mid \text{IN})$
  $$= 0.0417 \times 0.75 \times 1.0 = \mathbf{0.0313}$$
- All other predecessors: 0
- **Max: 0.0313** (from VB)

**Summary Table ($t = 4$)**:

| State | $v_4$ | Backpointer |
|-------|-------|-------------|
| PRP | 0 | — |
| MD | 0 | — |
| NN | 0 | — |
| VB | 0 | — |
| IN | **0.0313** | VB |
| NNS | 0 | — |

**Interpretation**: "In" is unambiguously a preposition (IN) in this context. The transition probability $P(\text{IN} \mid \text{VB}) = 0.75$ reflects that verbs are often followed by prepositional phrases.

---

**Step B.5: Recursion ($t = 5$, word = "rivers")**

$$v_5(j) = \max_i \left[ v_4(i) \cdot a_{ij} \cdot b_j(\text{rivers}) \right]$$

**$v_5(\text{NNS})$** (rivers as plural noun):
- From IN: $v_4(\text{IN}) \times P(\text{NNS} \mid \text{IN}) \times P(\text{rivers} \mid \text{NNS})$
  $$= 0.0313 \times 1.0 \times 0.333 = \mathbf{0.0104}$$
- All other predecessors: 0
- **Max: 0.0104** (from IN)

**Summary Table ($t = 5$)**:

| State | $v_5$ | Backpointer |
|-------|-------|-------------|
| PRP | 0 | — |
| MD | 0 | — |
| NN | 0 | — |
| VB | 0 | — |
| IN | 0 | — |
| NNS | **0.0104** | IN |

**Interpretation (sentence end)**: "Rivers" is unambiguously a plural noun (NNS). The transition $P(\text{NNS} \mid \text{IN}) = 1.0$ captures the pattern that prepositions are followed by noun phrases (their objects).

---

#### Part C: Termination and Backtrace

**Step C.1: Identify Best Final State**

$$\hat{q}_T = \arg\max_j v_5(j) = \text{NNS} \quad \text{with probability } 0.0104$$

**Step C.2: Backtrace**

Trace backpointers from $t = 5$ to $t = 1$:

| $t$ | Best State | Backpointer → Previous State |
|-----|------------|------------------------------|
| 5 | NNS | → IN |
| 4 | IN | → VB |
| 3 | VB | → MD |
| 2 | MD | → PRP |
| 1 | PRP | → $\langle s \rangle$ |

**Recovered path**: $\langle s \rangle$ → PRP → MD → VB → IN → NNS

---

**Step C.3: Final Tagged Sentence**

$$\boxed{\text{She/PRP can/MD fish/VB in/IN rivers/NNS}}$$

**Joint probability of the best path**:
$$P^* = 0.0104$$

---

#### Part D: Interpretation and Analysis

**Summary of Probability Flow**:

| Position | Word | Tag | $v_t$ | Key Probability Factor |
|----------|------|-----|-------|------------------------|
| 1 (start) | She | PRP | 0.333 | $P(\text{PRP} \mid \langle s \rangle) = 0.667$ |
| 2 | can | MD | 0.167 | $P(\text{MD} \mid \text{PRP}) = 0.50$ |
| 3 | fish | VB | 0.0417 | $P(\text{VB} \mid \text{MD}) = 1.0$ |
| 4 | in | IN | 0.0313 | $P(\text{IN} \mid \text{VB}) = 0.75$ |
| 5 (end) | rivers | NNS | 0.0104 | $P(\text{NNS} \mid \text{IN}) = 1.0$ |

**How Viterbi Resolved Ambiguities**:

1. **Sentence-initial position ("She")**: The high prior $P(\text{PRP} \mid \langle s \rangle) = 0.667$ combined with the emission $P(\text{She} \mid \text{PRP}) = 0.50$ made PRP the only viable starting state. Sentences frequently begin with pronouns.

2. **Ambiguous "can" (MD vs NN)**: Although "can" could be a modal verb ("I can swim") or a noun ("a tin can"), the transition constraint $P(\text{NN} \mid \text{PRP}) = 0$ eliminated the noun interpretation. Grammatically, personal pronouns are not directly followed by common nouns in English.

3. **Ambiguous "fish" (VB vs NN)**: The critical constraint $P(\text{VB} \mid \text{MD}) = 1.0$ resolved this ambiguity. Modal verbs obligatorily select for following verbs, making the noun reading impossible (*"She can table" is ungrammatical).

4. **Sentence-internal "in"**: Prepositions are unambiguous in isolation; the transition $P(\text{IN} \mid \text{VB}) = 0.75$ reflects verbs' tendency to take prepositional phrase complements.

5. **Sentence-final "rivers"**: The object of a preposition is typically a noun phrase. The deterministic transition $P(\text{NNS} \mid \text{IN}) = 1.0$ reflects this grammatical requirement.

**Key Insight**: The Viterbi algorithm successfully disambiguates lexically ambiguous words by exploiting **sequential constraints encoded in transition probabilities**. Even though "can" and "fish" each have multiple possible tags, the grammatical context (preceding and following tags) narrows down the possibilities. This demonstrates how HMMs capture syntactic structure through local dependencies.

---

## WEEK 6: Deep Learning for Sequences (RNNs, LSTMs, Attention, Transformers)

### Core Formulas with Semantic Explanations

#### RNN Hidden State Update

| **Formula** | $h_t = g(U h_{t-1} + W x_t)$ |
|-------------|------------------------------|
| **Variables** | $h_t$ = hidden state at time $t$; $h_{t-1}$ = previous hidden state; $x_t$ = input at time $t$; $U$ = hidden-to-hidden weights; $W$ = input-to-hidden weights; $g$ = activation (tanh or ReLU) |
| **What it computes** | A new internal representation by combining the current input with memory of all previous inputs. |
| **Semantic meaning** | The hidden state acts as a **memory** that carries forward information from the past. The matrix $U$ determines how much of the previous state is retained; $W$ determines how the current input is integrated. The recurrence $h_{t-1} \to h_t$ enables the network to model **sequential dependencies** without a fixed context window. |

#### RNN Output Layer

| **Formula** | $\hat{y}_t = \text{softmax}(V h_t)$ |
|-------------|-------------------------------------|
| **Variables** | $\hat{y}_t$ = probability distribution over vocabulary (or classes); $V$ = hidden-to-output weights |
| **What it computes** | Predicts the next word (or label) given the current hidden state. |
| **Semantic meaning** | The hidden state $h_t$ encodes all information from $x_1, \ldots, x_t$. The output layer projects this compressed representation to vocabulary size and normalizes via softmax to produce a valid probability distribution. |

#### Cross-Entropy Loss (Language Modeling)

| **Formula** | $L = -\log \hat{y}_t[w_{t+1}]$ |
|-------------|-------------------------------|
| **What it computes** | Negative log probability of the true next word $w_{t+1}$. |
| **Semantic meaning** | **Heavily penalizes confident wrong predictions.** If the model assigns probability 0.01 to the correct word, $L = -\log(0.01) = 4.6$. If probability 0.99, $L = 0.01$. Minimizing this loss trains the model to assign high probability to actual next words. |

#### Perplexity

| **Formula** | $\text{PP}(W) = \exp\left(\frac{1}{N}\sum_{t=1}^{N} L_t\right) = P(W)^{-1/N}$ |
|-------------|-----------------------------------------------------------------------------|
| **What it computes** | Geometric mean of inverse probabilities—the "effective branching factor" of the model. |
| **Semantic meaning** | Perplexity = 100 means the model is "as confused as if choosing uniformly among 100 words." **Lower perplexity = better model.** A perplexity of 1 would mean perfect prediction. |

#### Weight Tying

| **Formula** | $V = E^T$ (output weights = transposed input embeddings) |
|-------------|----------------------------------------------------------|
| **What it computes** | Shares parameters between input embeddings and output projection. |
| **Semantic meaning** | Words that are semantically similar should be both **easy to predict** when they appear next and **represented similarly** as inputs. Weight tying enforces this symmetry and reduces model size by $|V| \times d$ parameters. |

---

#### LSTM Forget Gate

| **Formula** | $f_t = \sigma(U_f h_{t-1} + W_f x_t + b_f)$ |
|-------------|---------------------------------------------|
| **What it computes** | A vector of values in $(0,1)$ determining how much of the previous cell state to **retain**. |
| **Semantic meaning** | Acts as a **learned erasure mechanism**. Values near 0 "forget" the corresponding cell dimension; values near 1 "remember" it. Enables the network to clear irrelevant information when context changes (e.g., crossing a sentence boundary). |

#### LSTM Input Gate

| **Formula** | $i_t = \sigma(U_i h_{t-1} + W_i x_t + b_i)$ |
|-------------|---------------------------------------------|
| **What it computes** | A vector of values in $(0,1)$ determining how much **new information** to add to the cell state. |
| **Semantic meaning** | Acts as a **learned write gate**. Controls which dimensions of the candidate content $g_t$ actually get written to memory. Enables selective updating—some information may be ignored even if computed. |

#### LSTM Candidate Content

| **Formula** | $g_t = \tanh(U_g h_{t-1} + W_g x_t + b_g)$ |
|-------------|-------------------------------------------|
| **What it computes** | Potential new content to add to cell state, scaled to $(-1, 1)$. |
| **Semantic meaning** | Represents "what we could remember" about the current input and context. The tanh squashes values to prevent unbounded growth. This candidate is then filtered by the input gate before being added to memory. |

#### LSTM Cell State Update

| **Formula** | $c_t = (c_{t-1} \odot f_t) + (g_t \odot i_t)$ |
|-------------|----------------------------------------------|
| **Variables** | $\odot$ = element-wise (Hadamard) multiplication |
| **What it computes** | New cell state by selectively forgetting old content and adding new gated content. |
| **Semantic meaning** | This is the **core memory update**: the forget gate $f_t$ erases irrelevant old information; the input gate $i_t$ controls what new information is written. The **additive** update (not multiplicative) creates a "gradient highway" that mitigates vanishing gradients—gradients can flow backwards through the addition unchanged. |

#### LSTM Output Gate

| **Formula** | $o_t = \sigma(U_o h_{t-1} + W_o x_t + b_o)$ |
|-------------|---------------------------------------------|
| **What it computes** | A vector controlling which parts of the cell state are **exposed** as output. |
| **Semantic meaning** | The cell state may contain more information than is currently relevant. The output gate selects which dimensions to reveal in the hidden state for downstream processing or prediction. |

#### LSTM Hidden State

| **Formula** | $h_t = o_t \odot \tanh(c_t)$ |
|-------------|------------------------------|
| **What it computes** | Gated, squashed cell content that serves as the layer's output. |
| **Semantic meaning** | Combines the long-term memory (cell state) with the output gate to produce a **short-term output representation**. The tanh ensures values are bounded; the output gate determines what aspects of memory are relevant for the current timestep. |

---

#### Encoder-Decoder Context Vector

| **Formula** | $c = h_n^e$ |
|-------------|-------------|
| **What it computes** | The final encoder hidden state, passed to the decoder as a summary of the source sequence. |
| **Semantic meaning** | In basic seq2seq, this single vector must encode **everything** about the source—a severe **bottleneck**. The decoder uses this fixed representation for all output positions, which limits performance on long sequences. |

#### Attention Score (Dot-Product)

| **Formula** | $\text{score}(h_i^d, h_j^e) = h_i^d \cdot h_j^e$ |
|-------------|--------------------------------------------------|
| **What it computes** | Similarity between decoder state $h_i^d$ and encoder state $h_j^e$. |
| **Semantic meaning** | Measures **relevance**: how much should the decoder at step $i$ "attend to" source position $j$? High dot product = high similarity = high attention. This enables the decoder to focus on different source words at different timesteps. |

#### Attention Weights

| **Formula** | $\alpha_{ij} = \frac{\exp(\text{score}(h_i^d, h_j^e))}{\sum_{k=1}^{n} \exp(\text{score}(h_i^d, h_k^e))}$ |
|-------------|------------------------------------------------------------------------------------------------------|
| **What it computes** | Normalized attention distribution over source positions. |
| **Semantic meaning** | Softmax converts raw scores to a probability distribution summing to 1. The weight $\alpha_{ij}$ represents how much "attention" decoder step $i$ pays to source position $j$. Visualizing these weights reveals **soft alignments** between source and target. |

#### Attention Context Vector

| **Formula** | $c_i = \sum_{j=1}^{n} \alpha_{ij} h_j^e$ |
|-------------|------------------------------------------|
| **What it computes** | Weighted average of encoder hidden states—a **dynamic** context for decoder step $i$. |
| **Semantic meaning** | Unlike the fixed bottleneck, the attention context is **customized** for each output position. When generating "verde" (green in Spanish), the attention may weight "green" heavily; when generating "bruja" (witch), it focuses on "witch." This resolves the bottleneck problem. |

---

#### Transformer Self-Attention (Scaled Dot-Product)

| **Formula** | $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$ |
|-------------|-----------------------------------------------------------------------------------|
| **Variables** | $Q$ = query matrix; $K$ = key matrix; $V$ = value matrix; $d_k$ = key dimension |
| **What it computes** | Each position attends to all positions, weighted by query-key similarity, then aggregates values. |
| **Semantic meaning** | Each word asks "what should I attend to?" (query), offers "what I can be matched against" (key), and provides "what I contribute if matched" (value). The $\sqrt{d_k}$ scaling prevents dot products from growing large and pushing softmax into saturated regions with vanishing gradients. This is the **foundation of Transformers**. |

#### Query-Key-Value Projections

| **Formulas** | $Q = XW^Q$, $\quad K = XW^K$, $\quad V = XW^V$ |
|--------------|-----------------------------------------------|
| **Variables** | $X \in \mathbb{R}^{n \times d}$ = input sequence; $W^Q, W^K, W^V$ = learned projection matrices |
| **What it computes** | Projects each position into three distinct roles for attention. |
| **Semantic meaning** | A word may offer different information as a **query** (what it's looking for), **key** (how it can be found), and **value** (what it contributes). Separate projections allow the model to learn these different roles independently. |

#### Multi-Head Attention

| **Formula** | $\text{MultiHead}(X) = [\text{head}_1 \oplus \ldots \oplus \text{head}_H] W^O$ |
|-------------|------------------------------------------------------------------------------|
| **Variables** | $\text{head}_h = \text{Attention}(XW_h^Q, XW_h^K, XW_h^V)$; $\oplus$ = concatenation |
| **What it computes** | Runs $H$ attention operations in parallel with different projections, then combines results. |
| **Semantic meaning** | Different heads can capture **different types of relationships**: one head may focus on syntactic dependencies (subject-verb), another on semantic similarity, another on positional patterns. The output projection $W^O$ integrates these diverse perspectives into a unified representation. |

#### Causal Masking

| **Formula** | $\text{score}(x_i, x_j) = \begin{cases} \frac{q_i \cdot k_j}{\sqrt{d_k}} & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$ |
|-------------|----------------------------------------------------------------------------------------------------------------------------------------|
| **What it computes** | Prevents position $i$ from attending to future positions $j > i$. |
| **Semantic meaning** | Essential for **autoregressive generation**: when predicting word $t+1$, the model can only see words $1, \ldots, t$. Setting scores to $-\infty$ before softmax ensures future positions receive zero attention weight. |

#### Positional Encoding (Sinusoidal)

| **Formulas** | $p_{i,2k} = \sin\left(\frac{i}{10000^{2k/d}}\right)$, $\quad p_{i,2k+1} = \cos\left(\frac{i}{10000^{2k/d}}\right)$ |
|--------------|------------------------------------------------------------------------------------------------------------------|
| **What it computes** | Unique position vectors encoding sequence order. |
| **Semantic meaning** | Self-attention is **permutation-invariant** (treats inputs as a set). Positional encodings inject order information by adding position-specific vectors to input embeddings. Different dimensions encode position at different frequencies, allowing the model to learn relative positions (e.g., "3 positions apart"). |

#### Transformer Block

| **Formulas** | $z = \text{LayerNorm}(x + \text{MultiHeadAttn}(x))$; $\quad y = \text{LayerNorm}(z + \text{FFN}(z))$ |
|--------------|-----------------------------------------------------------------------------------------------------|
| **What it computes** | One layer of a Transformer: self-attention followed by feedforward, with residual connections and normalization. |
| **Semantic meaning** | **Residual connections** ($x + f(x)$) enable gradient flow and allow layers to learn incremental refinements. **Layer normalization** stabilizes training by normalizing activations. The **FFN** adds nonlinearity and parameter capacity. Stacking many blocks creates deep representations. |

#### Layer Normalization

| **Formula** | $\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta$ |
|-------------|---------------------------------------------------------------------|
| **Variables** | $\mu, \sigma$ = mean and standard deviation of $x$; $\gamma, \beta$ = learned scale and shift |
| **What it computes** | Normalizes activations to zero mean and unit variance, then applies learned affine transform. |
| **Semantic meaning** | Prevents activations from growing unboundedly or vanishing. Unlike batch normalization, layer norm operates **per-example**, making it suitable for variable-length sequences and small batch sizes. |

---

### Key Definitions

| Term | Definition |
|------|------------|
| **Backpropagation Through Time (BPTT)** | Applying backpropagation to an RNN by "unrolling" it across timesteps. Gradients flow backwards through the recurrence, but may vanish or explode over long sequences. |
| **Vanishing Gradient Problem** | Gradients shrink exponentially when backpropagated through many timesteps, preventing learning of long-range dependencies. LSTMs mitigate this via additive cell updates. |
| **Teacher Forcing** | Training technique where the decoder receives the **gold (correct)** previous token as input, rather than its own prediction. Speeds training but may cause exposure bias. |
| **Encoder-Decoder (Seq2Seq)** | Architecture separating input encoding from output generation; enables mapping between sequences of different lengths (e.g., translation). |
| **Bottleneck Problem** | In basic seq2seq, the single context vector $c = h_n^e$ must encode the entire source—limiting capacity for long sequences. Attention solves this. |
| **Residual Connection** | Adding input to output: $y = x + f(x)$. Enables gradient flow and allows layers to learn incremental changes rather than complete transformations. |
| **Autoregressive Generation** | Generating sequences one token at a time, where each token is conditioned on all previously generated tokens. |
| **Weight Tying** | Sharing the embedding matrix and output projection: $V = E^T$. Reduces parameters and improves performance. |

---

### Worked Example 1: RNN Forward Pass — "So long"

**Task**: Compute the forward pass and cross-entropy loss for an RNN predicting "long" from "So" and "and" from "long".

**Setup**:
- **Vocabulary** (size 5): and(0), for(1), long(2), so(3), thanks(4)
- **Embedding dimension**: 2
- **Hidden dimension**: 2
- **Activation**: ReLU

**Parameters**:

Embeddings $E$ (each row is a word's embedding):
| Word | $e_0$ | $e_1$ |
|------|-------|-------|
| and | 0.5 | 0.1 |
| for | 0.2 | 0.4 |
| long | 0.7 | 0.3 |
| so | 0.1 | 0.8 |
| thanks | 0.6 | 0.2 |

Weight matrices:
$$W = \begin{bmatrix} 0.3 & 0.5 \\ 0.2 & 0.4 \end{bmatrix}, \quad U = \begin{bmatrix} 0.1 & 0.2 \\ 0.3 & 0.1 \end{bmatrix}, \quad V = \begin{bmatrix} 0.4 & 0.2 \\ 0.1 & 0.5 \\ 0.3 & 0.3 \\ 0.2 & 0.1 \\ 0.5 & 0.4 \end{bmatrix}$$

Initial hidden state: $h_0 = [0, 0]^T$

---

**Step 1: Process "so" (index 3) → Predict "long" (index 2)**

**Embedding lookup**:
$$x_1 = E[3] = [0.1, 0.8]^T$$

**Hidden state computation**:
$$z_1 = U \cdot h_0 + W \cdot x_1 = \begin{bmatrix}0\\0\end{bmatrix} + \begin{bmatrix} 0.3(0.1) + 0.5(0.8) \\ 0.2(0.1) + 0.4(0.8) \end{bmatrix} = \begin{bmatrix} 0.03 + 0.40 \\ 0.02 + 0.32 \end{bmatrix} = \begin{bmatrix} 0.43 \\ 0.34 \end{bmatrix}$$

$$h_1 = \text{ReLU}(z_1) = [0.43, 0.34]^T \quad \text{(both positive, unchanged)}$$

**Output computation**:
$$\text{logits}_1 = V \cdot h_1 = \begin{bmatrix} 0.4(0.43) + 0.2(0.34) \\ 0.1(0.43) + 0.5(0.34) \\ 0.3(0.43) + 0.3(0.34) \\ 0.2(0.43) + 0.1(0.34) \\ 0.5(0.43) + 0.4(0.34) \end{bmatrix} = \begin{bmatrix} 0.172 + 0.068 \\ 0.043 + 0.170 \\ 0.129 + 0.102 \\ 0.086 + 0.034 \\ 0.215 + 0.136 \end{bmatrix} = \begin{bmatrix} 0.240 \\ 0.213 \\ 0.231 \\ 0.120 \\ 0.351 \end{bmatrix}$$

**Softmax**:
$$\exp(\text{logits}_1) = [1.271, 1.237, 1.260, 1.127, 1.421]^T, \quad \sum = 6.316$$

$$\hat{y}_1 = \frac{1}{6.316}[1.271, 1.237, 1.260, 1.127, 1.421]^T = [0.201, 0.196, 0.200, 0.178, 0.225]^T$$

**Loss for predicting "long" (index 2)**:
$$L_1 = -\log(\hat{y}_1[2]) = -\log(0.200) = 1.61$$

---

**Step 2: Process "long" (index 2) → Predict "and" (index 0)**

**Embedding lookup**:
$$x_2 = E[2] = [0.7, 0.3]^T$$

**Hidden state computation**:
$$z_2 = U \cdot h_1 + W \cdot x_2 = \begin{bmatrix} 0.1(0.43) + 0.2(0.34) \\ 0.3(0.43) + 0.1(0.34) \end{bmatrix} + \begin{bmatrix} 0.3(0.7) + 0.5(0.3) \\ 0.2(0.7) + 0.4(0.3) \end{bmatrix}$$
$$= \begin{bmatrix} 0.043 + 0.068 \\ 0.129 + 0.034 \end{bmatrix} + \begin{bmatrix} 0.21 + 0.15 \\ 0.14 + 0.12 \end{bmatrix} = \begin{bmatrix} 0.111 \\ 0.163 \end{bmatrix} + \begin{bmatrix} 0.36 \\ 0.26 \end{bmatrix} = \begin{bmatrix} 0.471 \\ 0.423 \end{bmatrix}$$

$$h_2 = \text{ReLU}(z_2) = [0.471, 0.423]^T$$

**Output computation**:
$$\text{logits}_2 = V \cdot h_2 = \begin{bmatrix} 0.4(0.471) + 0.2(0.423) \\ 0.1(0.471) + 0.5(0.423) \\ 0.3(0.471) + 0.3(0.423) \\ 0.2(0.471) + 0.1(0.423) \\ 0.5(0.471) + 0.4(0.423) \end{bmatrix} = \begin{bmatrix} 0.273 \\ 0.259 \\ 0.268 \\ 0.137 \\ 0.405 \end{bmatrix}$$

**Softmax**:
$$\exp(\text{logits}_2) = [1.314, 1.296, 1.307, 1.147, 1.499]^T, \quad \sum = 6.563$$

$$\hat{y}_2 = [0.200, 0.197, 0.199, 0.175, 0.228]^T$$

**Loss for predicting "and" (index 0)**:
$$L_2 = -\log(\hat{y}_2[0]) = -\log(0.200) = 1.61$$

---

**Final Results**:
- Average loss: $L = \frac{1}{2}(1.61 + 1.61) = 1.61$
- Perplexity: $\text{PP} = e^{1.61} = 5.0$

**Interpretation**: A perplexity of 5.0 means the model is "as uncertain as choosing uniformly among 5 words" (our vocabulary size), indicating this untrained model has not yet learned meaningful patterns.

---

### Worked Example 2: LSTM Gate Computation

**Task**: Compute one LSTM timestep, showing all gate values and the cell/hidden state updates.

**Setup**:
- **Input**: $x_t = [0.5, 0.3]^T$
- **Previous states**: $h_{t-1} = [0.2, 0.4]^T$, $c_{t-1} = [0.6, 0.8]^T$
- **Dimension**: input = 2, hidden = 2

**Simplified weights** (for illustration; real LSTMs have separate $U$ and $W$):

For each gate, assume $W_{gate} = \begin{bmatrix} 0.5 & 0.3 \\ 0.2 & 0.4 \end{bmatrix}$ and $U_{gate} = \begin{bmatrix} 0.1 & 0.2 \\ 0.3 & 0.1 \end{bmatrix}$, with $b = [0, 0]^T$.

The pre-activation is computed as: $z_{gate} = U_{gate} \cdot h_{t-1} + W_{gate} \cdot x_t$

---

**Step 1: Compute pre-activations** (same for all gates in this simplified example):

$$z = U \cdot h_{t-1} + W \cdot x_t = \begin{bmatrix} 0.1(0.2) + 0.2(0.4) \\ 0.3(0.2) + 0.1(0.4) \end{bmatrix} + \begin{bmatrix} 0.5(0.5) + 0.3(0.3) \\ 0.2(0.5) + 0.4(0.3) \end{bmatrix}$$
$$= \begin{bmatrix} 0.02 + 0.08 \\ 0.06 + 0.04 \end{bmatrix} + \begin{bmatrix} 0.25 + 0.09 \\ 0.10 + 0.12 \end{bmatrix} = \begin{bmatrix} 0.10 \\ 0.10 \end{bmatrix} + \begin{bmatrix} 0.34 \\ 0.22 \end{bmatrix} = \begin{bmatrix} 0.44 \\ 0.32 \end{bmatrix}$$

---

**Step 2: Compute gates**

**Forget gate** (what to keep from old cell state):
$$f_t = \sigma([0.44, 0.32]^T) = [\sigma(0.44), \sigma(0.32)]^T = [0.608, 0.579]^T$$

*Interpretation*: About 60% of the previous cell state will be retained in each dimension.

**Input gate** (what to add):
$$i_t = \sigma([0.44, 0.32]^T) = [0.608, 0.579]^T$$

**Candidate content**:
$$g_t = \tanh([0.44, 0.32]^T) = [\tanh(0.44), \tanh(0.32)]^T = [0.414, 0.309]^T$$

*Interpretation*: This is the potential new information to write to memory.

**Output gate** (what to expose):
$$o_t = \sigma([0.44, 0.32]^T) = [0.608, 0.579]^T$$

---

**Step 3: Update cell state**

$$c_t = (c_{t-1} \odot f_t) + (g_t \odot i_t)$$
$$= ([0.6, 0.8] \odot [0.608, 0.579]) + ([0.414, 0.309] \odot [0.608, 0.579])$$
$$= [0.365, 0.463] + [0.252, 0.179]$$
$$= [0.617, 0.642]$$

*Interpretation*: The cell state combines ~60% of the old memory ($[0.365, 0.463]$) with gated new content ($[0.252, 0.179]$).

---

**Step 4: Compute hidden state**

$$h_t = o_t \odot \tanh(c_t)$$
$$= [0.608, 0.579] \odot [\tanh(0.617), \tanh(0.642)]$$
$$= [0.608, 0.579] \odot [0.548, 0.566]$$
$$= [0.333, 0.328]$$

**Summary of LSTM state transition**:
| Quantity | Value | Interpretation |
|----------|-------|----------------|
| $f_t$ | $[0.608, 0.579]$ | Keep ~60% of old cell state |
| $i_t$ | $[0.608, 0.579]$ | Write ~60% of candidate to cell |
| $g_t$ | $[0.414, 0.309]$ | Candidate new content |
| $c_t$ | $[0.617, 0.642]$ | Updated long-term memory |
| $h_t$ | $[0.333, 0.328]$ | Output representation |

---

### Worked Example 3: Self-Attention Computation

**Task**: Compute self-attention for a 3-word sequence "the cat sat".

**Setup**:
- **Sequence length**: $n = 3$
- **Embedding dimension**: $d = 4$
- **Key/value dimension**: $d_k = d_v = 2$

**Input embeddings** $X$ (rows are words):
$$X = \begin{bmatrix} 0.1 & 0.2 & 0.3 & 0.4 \\ 0.5 & 0.6 & 0.7 & 0.8 \\ 0.2 & 0.3 & 0.4 & 0.5 \end{bmatrix} \quad \text{(the, cat, sat)}$$

**Projection matrices** (simplified):
$$W^Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \\ 0 & 0 \end{bmatrix}, \quad W^K = \begin{bmatrix} 0 & 0 \\ 0 & 0 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad W^V = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \\ 0 & 0 \end{bmatrix}$$

---

**Step 1: Compute Q, K, V**

$$Q = XW^Q = \begin{bmatrix} 0.1 & 0.2 \\ 0.5 & 0.6 \\ 0.2 & 0.3 \end{bmatrix}, \quad K = XW^K = \begin{bmatrix} 0.3 & 0.4 \\ 0.7 & 0.8 \\ 0.4 & 0.5 \end{bmatrix}, \quad V = XW^V = \begin{bmatrix} 0.1 & 0.2 \\ 0.5 & 0.6 \\ 0.2 & 0.3 \end{bmatrix}$$

*Interpretation*: Each word now has a query (what it looks for), key (how it can be matched), and value (what it contributes).

---

**Step 2: Compute attention scores** $QK^T$

$$QK^T = \begin{bmatrix} 0.1 & 0.2 \\ 0.5 & 0.6 \\ 0.2 & 0.3 \end{bmatrix} \begin{bmatrix} 0.3 & 0.7 & 0.4 \\ 0.4 & 0.8 & 0.5 \end{bmatrix}$$

$$= \begin{bmatrix} 0.1(0.3)+0.2(0.4) & 0.1(0.7)+0.2(0.8) & 0.1(0.4)+0.2(0.5) \\ 0.5(0.3)+0.6(0.4) & 0.5(0.7)+0.6(0.8) & 0.5(0.4)+0.6(0.5) \\ 0.2(0.3)+0.3(0.4) & 0.2(0.7)+0.3(0.8) & 0.2(0.4)+0.3(0.5) \end{bmatrix}$$

$$= \begin{bmatrix} 0.11 & 0.23 & 0.14 \\ 0.39 & 0.83 & 0.50 \\ 0.18 & 0.38 & 0.23 \end{bmatrix}$$

---

**Step 3: Scale by** $\sqrt{d_k} = \sqrt{2} \approx 1.414$

$$\frac{QK^T}{\sqrt{d_k}} = \begin{bmatrix} 0.078 & 0.163 & 0.099 \\ 0.276 & 0.587 & 0.354 \\ 0.127 & 0.269 & 0.163 \end{bmatrix}$$

---

**Step 4: Apply softmax** (row-wise)

For row 1: $[0.078, 0.163, 0.099]$
- $\exp$: $[1.081, 1.177, 1.104]$, sum = $3.362$
- Softmax: $[0.321, 0.350, 0.328]$

For row 2: $[0.276, 0.587, 0.354]$
- $\exp$: $[1.318, 1.799, 1.425]$, sum = $4.542$
- Softmax: $[0.290, 0.396, 0.314]$

For row 3: $[0.127, 0.269, 0.163]$
- $\exp$: $[1.135, 1.309, 1.177]$, sum = $3.621$
- Softmax: $[0.314, 0.361, 0.325]$

**Attention weights**:
$$A = \begin{bmatrix} 0.321 & 0.350 & 0.328 \\ 0.290 & 0.396 & 0.314 \\ 0.314 & 0.361 & 0.325 \end{bmatrix}$$

*Interpretation*: "cat" (row 2) attends most to itself (0.396), while "the" and "sat" distribute attention more evenly.

---

**Step 5: Compute output** $Y = AV$

$$Y = \begin{bmatrix} 0.321 & 0.350 & 0.328 \\ 0.290 & 0.396 & 0.314 \\ 0.314 & 0.361 & 0.325 \end{bmatrix} \begin{bmatrix} 0.1 & 0.2 \\ 0.5 & 0.6 \\ 0.2 & 0.3 \end{bmatrix}$$

For row 1:
$$y_1 = 0.321[0.1, 0.2] + 0.350[0.5, 0.6] + 0.328[0.2, 0.3]$$
$$= [0.032, 0.064] + [0.175, 0.210] + [0.066, 0.098] = [0.273, 0.372]$$

Final output (contextualized representations):
$$Y = \begin{bmatrix} 0.273 & 0.372 \\ 0.292 & 0.396 \\ 0.281 & 0.382 \end{bmatrix}$$

*Interpretation*: Each word's output is now a weighted combination of all words' values. "cat" (middle row) has the highest values because it attended most to itself, which has the largest value vector.

---

## WEEK 7: Large Language Models (BERT, GPT, Decoding, Prompting)

### Core Formulas with Semantic Explanations

#### Causal Language Model Objective

| **Formula** | $L = -\frac{1}{T}\sum_{t=1}^{T} \log P(w_t \mid w_{<t})$ |
|-------------|----------------------------------------------------------|
| **What it computes** | Average negative log probability of predicting each word given all preceding words. |
| **Semantic meaning** | The model learns to predict the next token by attending only to the **left context**. This is the foundation of GPT-style models. Lower loss = better next-word prediction. The causal mask ensures no "cheating" by looking ahead. |

#### Masked Language Model (MLM) Objective

| **Formula** | $L_{MLM} = -\frac{1}{|M|}\sum_{i \in M} \log P(x_i \mid \mathbf{h}^i)$ |
|-------------|-----------------------------------------------------------------------|
| **Variables** | $M$ = set of masked positions; $\mathbf{h}^i$ = contextual representation at position $i$ |
| **What it computes** | Average negative log probability of recovering masked tokens from their bidirectional context. |
| **Semantic meaning** | Unlike causal LM, MLM can attend to **both left and right context**. This enables BERT-style models to learn richer representations because each prediction uses the entire sentence. The 15% masking rate balances learning signal with preserving context. |

#### Temperature Scaling

| **Formula** | $\hat{y}_i = \text{softmax}(z_i / \tau) = \frac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}}$ |
|-------------|-----------------------------------------------------------------------------------|
| **Variables** | $z_i$ = logit (raw score) for word $i$; $\tau$ = temperature |
| **What it computes** | Reshapes the probability distribution before sampling. |
| **Semantic meaning** | **$\tau < 1$ (low temperature)**: Sharpens distribution → high-probability words dominate → more deterministic, focused output. **$\tau > 1$ (high temperature)**: Flattens distribution → rare words get more chance → more diverse, creative output. **$\tau \to 0$**: Approaches greedy decoding (always pick max). |

#### Top-$p$ (Nucleus) Sampling

| **Formula** | Sample from smallest set $V^{(p)}$ where $\sum_{w \in V^{(p)}} P(w \mid w_{<t}) \geq p$ |
|-------------|----------------------------------------------------------------------------------------|
| **What it computes** | The "nucleus" of high-probability words whose cumulative probability exceeds threshold $p$. |
| **Semantic meaning** | **Adapts to distribution shape**: When probability is concentrated (confident prediction), the nucleus is small; when diffuse (uncertain), more candidates are included. Unlike top-$k$ (fixed count), top-$p$ dynamically adjusts vocabulary size. Typical $p = 0.9$ or $0.95$. |

#### Top-$k$ Sampling

| **Formula** | Sample from the $k$ highest-probability words after renormalizing |
|-------------|------------------------------------------------------------------|
| **What it computes** | Truncates the long tail of low-probability words. |
| **Semantic meaning** | Prevents sampling rare, potentially incoherent words. When $k=1$, this is greedy decoding. The fixed $k$ doesn't adapt to distribution shape, which is why top-$p$ is often preferred. |

#### Beam Search Score

| **Formula** | $\text{score}(w_1, \ldots, w_t) = \sum_{i=1}^{t} \log P(w_i \mid w_{<i})$ |
|-------------|-------------------------------------------------------------------------|
| **What it computes** | Cumulative log probability of a partial sequence. |
| **Semantic meaning** | Beam search maintains $k$ candidate sequences, extending each and keeping the top $k$ by score. This explores multiple hypotheses simultaneously, often finding higher-probability sequences than greedy. Used for tasks like translation where output quality matters more than diversity. |

#### Sequence Classification with BERT

| **Formula** | $\hat{y} = \text{softmax}(\mathbf{h}_{[CLS]} \cdot W_C)$ |
|-------------|----------------------------------------------------------|
| **Variables** | $\mathbf{h}_{[CLS]}$ = final-layer representation of `[CLS]` token; $W_C$ = classifier weights |
| **What it computes** | Class probabilities for the entire input sequence. |
| **Semantic meaning** | The `[CLS]` token aggregates information from the entire sequence via bidirectional attention. A simple linear classifier on top converts this pooled representation to task predictions. Only $W_C$ may be trained (frozen encoder) or all parameters finetuned. |

#### Token Classification (NER)

| **Formula** | $\hat{y}_i = \text{softmax}(\mathbf{h}_i^L \cdot W_K)$ |
|-------------|-------------------------------------------------------|
| **Variables** | $\mathbf{h}_i^L$ = final-layer representation at position $i$; $W_K \in \mathbb{R}^{d \times (2n+1)}$ for $n$ entity types |
| **What it computes** | Per-token label distribution (B-X, I-X, O tags). |
| **Semantic meaning** | Each token's contextual embedding is classified independently. The $2n+1$ output dimensions cover B and I tags for each of $n$ entity types plus O. Subword tokens require alignment back to words. |

#### LoRA (Low-Rank Adaptation)

| **Formula** | $\mathbf{h} = \mathbf{x}W + \mathbf{x}AB$ where $A \in \mathbb{R}^{d \times r}$, $B \in \mathbb{R}^{r \times d}$, $r \ll d$ |
|-------------|---------------------------------------------------------------------------------------------------------------------------|
| **What it computes** | Adds a low-rank update to frozen weights. |
| **Semantic meaning** | Instead of finetuning all $d \times d$ parameters in $W$, LoRA trains only $r \times (2d)$ parameters in low-rank matrices $A$ and $B$. The original $W$ is frozen. This dramatically reduces memory/compute while achieving similar performance. Common $r$ values: 4–64. |

#### RLHF Reward Model Loss

| **Formula** | $L_{RM} = -\log \sigma(R(x, y_1) - R(x, y_2))$ |
|-------------|-----------------------------------------------|
| **Variables** | $y_1$ = preferred response; $y_2$ = dispreferred response; $R$ = reward model |
| **What it computes** | Trains the reward model to assign higher scores to human-preferred outputs. |
| **Semantic meaning** | Given pairs of responses where humans indicate a preference, the reward model learns to predict that preference. The sigmoid ensures the loss is based on the **difference** in scores, not absolute values. This learned reward signal then guides policy optimization. |

#### Contextual Sense Embedding

| **Formula** | $\mathbf{v}_s = \frac{1}{n}\sum_{i=1}^{n} \mathbf{v}_i$ for all instances of sense $s$ |
|-------------|--------------------------------------------------------------------------------------|
| **What it computes** | Average contextual embedding across all occurrences of a word sense. |
| **Semantic meaning** | Different senses of a polysemous word (e.g., "bank" as financial institution vs. river bank) cluster in different regions of embedding space. Averaging instances of each sense creates a sense prototype. New instances can be disambiguated by nearest-neighbor to these prototypes. |

---

### Decoding Strategies Reference

| Strategy | Method | Use Case | Pros | Cons |
|----------|--------|----------|------|------|
| **Greedy** | Always pick $\arg\max P(w \mid w_{<t})$ | Deterministic tasks | Fast, simple | Generic, repetitive |
| **Beam Search** | Keep top-$k$ sequences by cumulative log-prob | Translation, summarization | Higher quality | Still deterministic, slow |
| **Temperature** | Divide logits by $\tau$ before softmax | Controlling creativity | Simple parameter | Doesn't truncate tail |
| **Top-$k$** | Sample from $k$ most probable words | Open-ended generation | Reduces incoherence | Fixed $k$ is suboptimal |
| **Top-$p$** | Sample from nucleus with cumulative prob $\geq p$ | Open-ended generation | Adapts to distribution | Slightly more complex |

**Typical combinations**: Temperature ($\tau = 0.7$) + Top-$p$ ($p = 0.9$) for creative tasks; Beam search ($k = 5$) for translation.

---

### Architecture Comparison: BERT vs GPT vs T5

| Aspect | BERT | GPT | T5 |
|--------|------|-----|-----|
| **Architecture** | Encoder-only | Decoder-only | Encoder-Decoder |
| **Attention** | Bidirectional | Causal (left-to-right) | Bidirectional (encoder) + Causal (decoder) |
| **Pretraining Objective** | Masked LM (+ NSP for original) | Causal LM | Span corruption (denoising) |
| **Primary Use** | Understanding tasks (classification, NER, QA extraction) | Generation tasks (text completion, dialogue) | Both understanding and generation |
| **Finetuning Pattern** | Add task head, finetune all or freeze | Prompt-based, few-shot, or finetune | Text-to-text format for all tasks |
| **Input Format** | `[CLS] text [SEP]` | `<prompt>` | `task prefix: input` |
| **Typical Size** | 110M–340M | 117M–175B+ | 220M–11B |

---

### Key Definitions

| Term | Definition |
|------|------------|
| **Contextual Embedding** | A word representation that depends on surrounding context. The same word gets different vectors in different sentences. Contrast with static embeddings (Word2Vec) where each word has one fixed vector. |
| **Polysemy** | A word having multiple distinct meanings (e.g., "bank" = financial institution or river edge). Contextual embeddings naturally capture this—different senses cluster separately. |
| **Finetuning** | Continuing to train a pretrained model on task-specific data. Can update all parameters or just a subset (e.g., classifier head). |
| **In-Context Learning (ICL)** | Learning from examples in the prompt without gradient updates. The model "learns" the task pattern from demonstrations during inference. |
| **Few-Shot Prompting** | Providing labeled examples in the prompt to demonstrate the task. Often improves performance by clarifying expected format and task. |
| **Zero-Shot Prompting** | Describing the task in natural language without examples. Relies on pretrained knowledge to generalize. |
| **Chain-of-Thought (CoT)** | Prompting technique that includes reasoning steps in demonstrations. Improves performance on multi-step reasoning tasks by encouraging the model to "show its work." |
| **Instruction Tuning (SFT)** | Finetuning on (instruction, response) pairs to improve instruction-following. The model learns to interpret and execute natural language commands. |
| **RLHF** | Reinforcement Learning from Human Feedback. Trains a reward model on human preferences, then optimizes the LLM to maximize reward while staying close to the base model. |
| **Hallucination** | Generating plausible-sounding but factually incorrect content. A major challenge because LLMs are trained to be fluent, not factual. |
| **KV Cache** | Stores previously computed key and value vectors during generation. Avoids recomputation at each step, dramatically accelerating inference. |
| **Anisotropy** | Embedding vectors tending to point in similar directions, causing high cosine similarity even for unrelated words. Mitigated by standardization (z-scoring). |

---

### Worked Example 1: Naive Bayes Sentiment Classification

**Task**: Classify "predictable with no fun" as positive or negative using Naive Bayes with Laplace smoothing.

**Training data**:
- **Positive (+)**: "fun", "couple", "fun", "love", "love"
- **Negative (−)**: "just", "plain", "boring", "predictable", "with", "no"

**Counts**:
| | + | − |
|---|---|---|
| Total words | 5 | 6 |
| Vocabulary | 11 unique words across both classes |

**Class priors** (with Laplace smoothing):
$$P(+) = \frac{5 + 1}{11 + 2} = \frac{6}{13} = 0.462$$
$$P(-) = \frac{6 + 1}{11 + 2} = \frac{7}{13} = 0.538$$

---

**Step 1: Compute word likelihoods with Laplace smoothing**

Formula: $P(w \mid c) = \frac{\text{count}(w, c) + 1}{\text{total}_c + |V|}$

For the test document "predictable with no fun":

| Word | Count in + | $P(w \mid +)$ | Count in − | $P(w \mid -)$ |
|------|------------|---------------|------------|---------------|
| predictable | 0 | $\frac{0+1}{5+11} = \frac{1}{16} = 0.0625$ | 1 | $\frac{1+1}{6+11} = \frac{2}{17} = 0.118$ |
| with | 0 | $\frac{1}{16} = 0.0625$ | 1 | $\frac{2}{17} = 0.118$ |
| no | 0 | $\frac{1}{16} = 0.0625$ | 1 | $\frac{2}{17} = 0.118$ |
| fun | 2 | $\frac{2+1}{16} = \frac{3}{16} = 0.1875$ | 0 | $\frac{1}{17} = 0.059$ |

---

**Step 2: Compute log posteriors**

**Log-posterior for positive**:
$$\log P(+ \mid d) \propto \log P(+) + \sum_w \log P(w \mid +)$$
$$= \log(0.462) + \log(0.0625) + \log(0.0625) + \log(0.0625) + \log(0.1875)$$
$$= -0.77 + (-2.77) + (-2.77) + (-2.77) + (-1.67) = -10.75$$

**Log-posterior for negative**:
$$\log P(- \mid d) \propto \log P(-) + \sum_w \log P(w \mid -)$$
$$= \log(0.538) + \log(0.118) + \log(0.118) + \log(0.118) + \log(0.059)$$
$$= -0.62 + (-2.14) + (-2.14) + (-2.14) + (-2.83) = -9.87$$

---

**Step 3: Classification**

Since $-9.87 > -10.75$, we classify as **negative**.

**Interpretation**: Despite "fun" being a strong positive indicator, the three negative words ("predictable", "with", "no") collectively outweigh it. The phrase "with no fun" is correctly interpreted as negative sentiment.

---

### Worked Example 2: Beam Search Decoding

**Task**: Generate text using beam search with beam width $k = 2$.

**Setup**:
- **Vocabulary**: {the, cat, sat, dog, ran}
- **Start**: `<s>`
- **Model probabilities** (simplified):

| Context | the | cat | sat | dog | ran |
|---------|-----|-----|-----|-----|-----|
| `<s>` | 0.5 | 0.1 | 0.05 | 0.3 | 0.05 |
| `<s> the` | 0.05 | 0.4 | 0.05 | 0.45 | 0.05 |
| `<s> dog` | 0.2 | 0.1 | 0.1 | 0.1 | 0.5 |
| `<s> the cat` | 0.1 | 0.05 | 0.6 | 0.05 | 0.2 |
| `<s> the dog` | 0.1 | 0.1 | 0.2 | 0.1 | 0.5 |

---

**Step 1: Expand from `<s>`** (keep top-2)

| Sequence | Log-prob |
|----------|----------|
| `<s> the` | $\log(0.5) = -0.69$ ✓ |
| `<s> dog` | $\log(0.3) = -1.20$ ✓ |
| `<s> cat` | $\log(0.1) = -2.30$ |
| `<s> sat` | $\log(0.05) = -3.00$ |
| `<s> ran` | $\log(0.05) = -3.00$ |

**Beams after step 1**: `<s> the` (−0.69), `<s> dog` (−1.20)

---

**Step 2: Expand each beam** (keep top-2 overall)

From `<s> the`:
| Sequence | Cumulative log-prob |
|----------|---------------------|
| `<s> the dog` | $-0.69 + \log(0.45) = -0.69 + (-0.80) = -1.49$ |
| `<s> the cat` | $-0.69 + \log(0.4) = -0.69 + (-0.92) = -1.61$ |

From `<s> dog`:
| Sequence | Cumulative log-prob |
|----------|---------------------|
| `<s> dog ran` | $-1.20 + \log(0.5) = -1.20 + (-0.69) = -1.89$ |
| `<s> dog the` | $-1.20 + \log(0.2) = -1.20 + (-1.61) = -2.81$ |

**Top-2 overall**: `<s> the dog` (−1.49), `<s> the cat` (−1.61)

---

**Step 3: Expand again**

From `<s> the dog`:
| Sequence | Cumulative log-prob |
|----------|---------------------|
| `<s> the dog ran` | $-1.49 + \log(0.5) = -2.18$ |
| `<s> the dog sat` | $-1.49 + \log(0.2) = -3.10$ |

From `<s> the cat`:
| Sequence | Cumulative log-prob |
|----------|---------------------|
| `<s> the cat sat` | $-1.61 + \log(0.6) = -2.12$ ← **Best!** |
| `<s> the cat ran` | $-1.61 + \log(0.2) = -3.22$ |

**Final beam**: `<s> the cat sat` (−2.12) wins over `<s> the dog ran` (−2.18)

**Interpretation**: Beam search found "the cat sat" with probability $e^{-2.12} = 0.12$, which is higher than greedy's path "the dog" ($0.5 \times 0.45 = 0.225$ for first two words, but "ran" gives $0.225 \times 0.5 = 0.1125 < 0.12$).

---

### Worked Example 3: Top-$p$ (Nucleus) Sampling

**Task**: Apply top-$p$ sampling with $p = 0.9$ to select the next word.

**Given probability distribution**:
| Word | Probability |
|------|-------------|
| the | 0.35 |
| a | 0.25 |
| one | 0.15 |
| some | 0.10 |
| this | 0.08 |
| that | 0.04 |
| many | 0.03 |

---

**Step 1: Sort by probability** (already sorted)

**Step 2: Compute cumulative probabilities**

| Word | Prob | Cumulative |
|------|------|------------|
| the | 0.35 | 0.35 |
| a | 0.25 | 0.60 |
| one | 0.15 | 0.75 |
| some | 0.10 | 0.85 |
| this | 0.08 | 0.93 ← exceeds 0.9 |
| that | 0.04 | 0.97 |
| many | 0.03 | 1.00 |

**Step 3: Find nucleus** (smallest set with cumulative ≥ 0.9)

Nucleus $V^{(p)} = \{\text{the, a, one, some, this}\}$ (cumulative = 0.93)

**Step 4: Renormalize**

| Word | Original | Renormalized |
|------|----------|--------------|
| the | 0.35 | 0.35/0.93 = 0.376 |
| a | 0.25 | 0.25/0.93 = 0.269 |
| one | 0.15 | 0.15/0.93 = 0.161 |
| some | 0.10 | 0.10/0.93 = 0.108 |
| this | 0.08 | 0.08/0.93 = 0.086 |
| **Total** | 0.93 | 1.000 |

**Step 5: Sample** from renormalized distribution

**Interpretation**: Words outside the nucleus ("that", "many") are excluded entirely, even though they have non-zero probability. This prevents sampling very unlikely words while maintaining diversity among reasonable choices.

---

### Worked Example 4: Temperature Effect on Sampling

**Task**: Show how temperature $\tau$ affects the distribution over vocabulary.

**Given logits**: $z = [2.0, 1.0, 0.5, 0.0, -0.5]$ for words A, B, C, D, E.

**Compute softmax at different temperatures**:

| Temperature | Formula | Distribution |
|-------------|---------|--------------|
| $\tau = 0.5$ (sharp) | $\text{softmax}(z/0.5)$ | A: 0.731, B: 0.180, C: 0.066, D: 0.018, E: 0.005 |
| $\tau = 1.0$ (standard) | $\text{softmax}(z)$ | A: 0.506, B: 0.186, C: 0.113, D: 0.069, E: 0.042 |
| $\tau = 2.0$ (flat) | $\text{softmax}(z/2)$ | A: 0.337, B: 0.206, C: 0.158, D: 0.121, E: 0.093 |

**Calculation for $\tau = 0.5$**:
$$z/\tau = [4.0, 2.0, 1.0, 0.0, -1.0]$$
$$\exp(z/\tau) = [54.6, 7.39, 2.72, 1.0, 0.37], \quad \sum = 66.08$$
$$\text{softmax} = [0.826, 0.112, 0.041, 0.015, 0.006]$$

(Approximate values shown in table for clarity.)

**Interpretation**:
- **Low $\tau$ (0.5)**: Word A dominates with 73% probability—output is nearly deterministic.
- **Standard $\tau$ (1.0)**: A still likely (50%), but B, C have meaningful probability.
- **High $\tau$ (2.0)**: Distribution is flattened—even E has 9% chance. More diverse but potentially less coherent.

---

## Summary and Exam Strategy

### Week 1 Checklist: Text Representation & Similarity

| **Can you...** | **Formula to Know** |
|----------------|---------------------|
| Compute TF for a term in a document? | $\text{tf}(t,d) = $ count of $t$ in $d$ |
| Compute log-dampened TF? | $\ln(\text{tf} + 1)$ |
| Compute document frequency? | $\text{df}_t = $ number of docs containing $t$ |
| Compute IDF? | $\text{idf}_t = \log_{10}(N / \text{df}_t)$ |
| Combine TF and IDF? | $w_{t,d} = \ln(\text{tf}+1) \cdot \text{idf}$ |
| Compute Euclidean distance? | $\sqrt{\sum(p_i - q_i)^2}$ |
| Compute Jaccard coefficient? | $\lvert A \cap B\rvert / \lvert A \cup B\rvert$ |
| Compute cosine similarity? | $\frac{\vec{p} \cdot \vec{q}}{\lVert\vec{p}\rVert \lVert\vec{q}\rVert}$ |
| Explain why IDF = 0 for common terms? | They appear in all docs, so $N/\text{df} = 1$, $\log(1) = 0$ |

---

### Week 2 Checklist: Normalization & Edit Distance

| **Can you...** | **Formula to Know** |
|----------------|---------------------|
| Write a regex to match words? | `\w+` or `[a-zA-Z]+` |
| Write a regex to match digits? | `\d+` or `[0-9]+` |
| Initialize edit distance base cases? | $D[i,0] = i$, $D[0,j] = j$ |
| Apply edit distance recurrence? | $\min(D[i-1,j]+1, D[i,j-1]+1, D[i-1,j-1]+\text{cost})$ |
| Compute precision? | $\text{TP} / (\text{TP} + \text{FP})$ |
| Compute recall? | $\text{TP} / (\text{TP} + \text{FN})$ |
| Compute F1 score? | $2PR / (P + R)$ or $2\text{TP} / (2\text{TP} + \text{FP} + \text{FN})$ |
| Explain BPE merge procedure? | Repeatedly merge most frequent adjacent pair |

---

### Week 3 Checklist: Classification & Language Modeling

| **Can you...** | **Formula to Know** |
|----------------|---------------------|
| Compute bigram probability (MLE)? | $C(w_{n-1}, w_n) / C(w_{n-1})$ |
| Apply Laplace smoothing to bigrams? | $(C + 1) / (C_{context} + V)$ |
| Compute perplexity from probabilities? | $\text{PP} = P(W)^{-1/N}$ |
| State the Naive Bayes classification rule? | $\arg\max_c P(c) \prod_i P(w_i \mid c)$ |
| Compute NB likelihood with smoothing? | $(\text{count}(w,c) + 1) / (\text{total}_c + V)$ |
| Convert to log-space? | $\log P(c) + \sum_i \log P(w_i \mid c)$ |
| Compute entropy of a distribution? | $-\sum p_i \log_2 p_i$ |
| Compute information gain? | $H(Y) - H(Y \mid X)$ |
| Apply sigmoid function? | $\sigma(z) = 1 / (1 + e^{-z})$ |
| Compute logistic regression gradient? | $(\hat{y} - y) \cdot x_j$ |
| Perform one gradient descent step? | $w \leftarrow w - \eta \cdot \text{gradient}$ |

---

### Week 4 Checklist: Representation Learning & Neural Networks

| **Can you...** | **Formula to Know** |
|----------------|---------------------|
| Compute PMI for a word-context pair? | $\text{PMI}(w,c) = \log_2 \frac{P(w,c)}{P(w)P(c)}$ |
| Convert PMI to PPMI? | $\text{PPMI} = \max(\text{PMI}, 0)$ |
| Explain the distributional hypothesis? | "Words in similar contexts have similar meanings" |
| Distinguish sparse vs dense embeddings? | Sparse: high-dim, mostly zeros (PPMI); Dense: low-dim, learned (Word2Vec) |
| State the Skip-Gram objective? | Predict context words from target word |
| Compute a single neuron output? | $y = f(\mathbf{w} \cdot \mathbf{x} + b)$ |
| Apply softmax to convert logits to probabilities? | $\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$ |
| Compute cross-entropy loss? | $L = -\log \hat{y}_c$ for correct class $c$ |
| Perform a forward pass through a 2-layer network? | Input → $W^{[1]}\mathbf{x} + b^{[1]}$ → ReLU → $W^{[2]}\mathbf{h} + b^{[2]}$ → softmax |
| Compute backpropagation gradient for sigmoid+CE? | $\frac{\partial L}{\partial z} = \hat{y} - y$ |
| State ReLU and its derivative? | $\max(0,z)$; derivative: 0 if $z<0$, 1 if $z \geq 0$ |

---

### Week 5 Checklist: Sequence Labeling & Information Extraction

| **Can you...** | **Formula to Know** |
|----------------|---------------------|
| Define an HMM (5-tuple)? | $(Q, A, O, B, \pi)$ = states, transitions, observations, emissions, initial |
| State the Markov assumption? | $P(q_i \mid q_1,\ldots,q_{i-1}) = P(q_i \mid q_{i-1})$ |
| Compute transition probability from counts? | $P(t_i \mid t_{i-1}) = C(t_{i-1}, t_i) / C(t_{i-1})$ |
| Compute emission probability from counts? | $P(w_i \mid t_i) = C(t_i, w_i) / C(t_i)$ |
| Initialize Viterbi at $t=1$? | $v_1(j) = \pi_j \cdot b_j(o_1)$ |
| Apply Viterbi recurrence? | $v_t(j) = \max_i [v_{t-1}(i) \cdot a_{ij} \cdot b_j(o_t)]$ |
| Store and use backpointers? | $\text{bt}_t(j) = \arg\max_i [v_{t-1}(i) \cdot a_{ij}]$ |
| Tag a sentence with BIO scheme? | B-X = begin entity X; I-X = inside; O = outside |
| Distinguish HMM from CRF? | HMM: generative $P(X,Y)$; CRF: discriminative $P(Y \mid X)$ |
| Explain why CRF allows richer features? | CRF can access entire input $X$; HMM only sees current observation |

---

### Week 6 Checklist: Deep Learning for Sequences

| **Can you...** | **Formula to Know** |
|----------------|---------------------|
| Write the RNN hidden state update? | $h_t = g(Uh_{t-1} + Wx_t)$ |
| Write the RNN output layer? | $\hat{y}_t = \text{softmax}(Vh_t)$ |
| Compute cross-entropy loss for LM? | $L = -\log \hat{y}_t[w_{t+1}]$ |
| Convert loss to perplexity? | $\text{PP} = e^L$ or $\text{PP} = P(W)^{-1/N}$ |
| Explain weight tying? | $V = E^T$; share embeddings and output weights |
| List all 6 LSTM equations? | $f_t, i_t, g_t, c_t, o_t, h_t$ with gates and cell update |
| Explain what each LSTM gate does? | Forget: erase memory; Input: write new; Output: expose |
| State the LSTM cell update? | $c_t = (c_{t-1} \odot f_t) + (g_t \odot i_t)$ (additive!) |
| Explain why LSTMs mitigate vanishing gradients? | Additive cell update creates gradient highway |
| Compute attention weights? | $\alpha_{ij} = \text{softmax}(h_i^d \cdot h_j^e)$ |
| Compute attention context vector? | $c_i = \sum_j \alpha_{ij} h_j^e$ |
| Write scaled dot-product self-attention? | $\text{softmax}(QK^T/\sqrt{d_k})V$ |
| Explain Q, K, V roles? | Query: what I seek; Key: how I'm found; Value: what I contribute |
| Explain causal masking? | Set future scores to $-\infty$ before softmax |
| State sinusoidal positional encoding? | $\sin(i/10000^{2k/d})$, $\cos(i/10000^{2k/d})$ |

---

### Week 7 Checklist: Large Language Models

| **Can you...** | **Formula to Know** |
|----------------|---------------------|
| State the causal LM objective? | $L = -\sum_t \log P(w_t \mid w_{<t})$ |
| State the masked LM (MLM) objective? | $L = -\sum_{i \in M} \log P(x_i \mid \mathbf{h}^i)$ |
| Distinguish BERT vs GPT attention? | BERT: bidirectional; GPT: causal (left-to-right only) |
| Apply temperature to logits? | $\text{softmax}(z/\tau)$; low $\tau$ = sharp, high $\tau$ = flat |
| Define top-$p$ (nucleus) sampling? | Sample from smallest set with cumulative prob $\geq p$ |
| Define top-$k$ sampling? | Sample from $k$ highest-probability words |
| Explain beam search? | Keep top-$k$ sequences by cumulative log-prob |
| Explain BERT sequence classification? | $\hat{y} = \text{softmax}(\mathbf{h}_{[CLS]} W_C)$ |
| Explain LoRA? | Low-rank matrices $A, B$ added to frozen weights |
| Explain RLHF? | Train reward model on preferences, then optimize policy |
| Define in-context learning? | Learning from prompt examples without gradient updates |
| Define chain-of-thought prompting? | Include reasoning steps in demonstrations |
| Explain finetuning vs prompting? | Finetuning updates weights; prompting uses fixed model |
| Explain the hallucination problem? | LLMs generate plausible but false content |
| Explain KV cache? | Store computed keys/values to speed autoregressive generation |

---

### Five-Minute Pre-Exam Review

**TF-IDF**: High weight = frequent in this doc, rare across corpus.

**Cosine Similarity**: Measures direction, not magnitude. Normalize by $\ell_2$ norms.

**Edit Distance**: DP matrix; base cases are row/col indices; each cell = min of three options.

**Smoothing**: Add 1 (or $k$) to counts; add $V$ (or $kV$) to denominator.

**Naive Bayes**: Multiply prior by all word likelihoods. Use log-space to avoid underflow.

**Information Gain**: Entropy before minus weighted entropy after. Higher = more informative feature.

**Gradient Descent**: Move opposite to gradient direction. Step size = learning rate × gradient.

**Perplexity**: Lower = better model. Equals inverse probability raised to $1/N$. Also $e^{\text{average CE loss}}$.

**PMI/PPMI**: Measures co-occurrence beyond chance. Positive = attraction; zero = no observed association.

**Neural Network Forward Pass**: Input → linear transform → activation → ... → softmax → probabilities.

**Backpropagation**: Apply chain rule backwards. For sigmoid+CE: $\frac{\partial L}{\partial z} = \hat{y} - y$.

**HMM**: Generative model; learns $P(\text{tag}_i \mid \text{tag}_{i-1})$ and $P(\text{word} \mid \text{tag})$.

**Viterbi**: DP over trellis. Each cell = max(prev × transition × emission). Backtrace for path.

**BIO Tagging**: B-X begins entity, I-X continues, O is outside. Enables multi-token entity extraction.

**RNN**: $h_t = g(Uh_{t-1} + Wx_t)$; hidden state carries memory of all previous inputs.

**LSTM**: Six equations with three gates (forget, input, output). Cell state uses additive update → gradient highway.

**Attention**: Weighted average of encoder states; weights = softmax(dot products). Solves bottleneck problem.

**Self-Attention**: $\text{softmax}(QK^T/\sqrt{d_k})V$. Each position attends to all positions. Foundation of Transformers.

**BERT vs GPT**: BERT = bidirectional encoder (MLM); GPT = causal decoder (next-word prediction).

**Temperature**: $\tau < 1$ sharpens (more deterministic); $\tau > 1$ flattens (more diverse).

**Top-$p$**: Nucleus sampling—adaptive cutoff based on cumulative probability, not fixed count.

**Chain-of-Thought**: Include reasoning steps in prompt → improves multi-step reasoning.

---

**You are exam-ready for Weeks 1–7 if you can:**

1. ✓ Build a TF-IDF matrix from scratch and explain why common terms get zero weight
2. ✓ Compute cosine similarity between two vectors step-by-step
3. ✓ Fill in an edit distance matrix and trace back the optimal alignment
4. ✓ Calculate Naive Bayes posteriors in log-space with add-1 smoothing
5. ✓ Compute entropy, conditional entropy, and information gain for a feature
6. ✓ Perform one iteration of gradient descent for logistic regression
7. ✓ Calculate PPMI for a word-context pair from a co-occurrence matrix
8. ✓ Perform a forward pass through a two-layer neural network with matrix arithmetic
9. ✓ Compute backpropagation gradients for a single sigmoid unit
10. ✓ Fill in a Viterbi trellis and trace back the optimal tag sequence
11. ✓ Estimate HMM transition and emission probabilities from a tagged corpus
12. ✓ Apply BIO tagging to a sentence with named entities
13. ✓ Compute an RNN forward pass step-by-step with matrix arithmetic
14. ✓ Trace through LSTM gate computations and explain each gate's role
15. ✓ Compute self-attention scores, weights, and output for a short sequence
16. ✓ Compare BERT and GPT: attention type, training objective, use cases
17. ✓ Apply temperature scaling and explain its effect on the distribution
18. ✓ Perform top-$p$ sampling: sort, cumulate, find nucleus, renormalize
19. ✓ Execute beam search to find optimal sequence
20. ✓ Explain in-context learning, chain-of-thought, and RLHF conceptually