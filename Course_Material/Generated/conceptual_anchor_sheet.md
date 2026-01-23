# Conceptual Anchor Sheet: Language and AI

| Document Property | Value |
| :--- | :--- |
| **Purpose** | Theory-only exam cheatsheet (one double-sided A4) |
| **Design** | Definitions, intuitions, and comparative logic—no equations |
| **Use** | Quick visual scanning under exam pressure |

---

## Index

1. [Part 1: Text Representation & Metrics](#part-1-text-representation--metrics)
2. [Part 2: Classical Models & Classification](#part-2-classical-models--classification)
3. [Part 3: Vector Semantics & Embeddings](#part-3-vector-semantics--embeddings)
4. [Part 4: Sequence Modeling](#part-4-sequence-modeling)
5. [Part 5: Modern Architectures & LLMs](#part-5-modern-architectures--llms)
6. [Comparative Tables](#comparative-tables)
7. [Limitation → Solution Ladder](#limitation--solution-ladder)
8. [Ethics & Social Impact](#ethics--social-impact)

---

## Part 1: Text Representation & Metrics

### Preprocessing Pipeline

**Tokenization**: Segmenting raw text into meaningful units (tokens). Defines the granularity of all downstream representations; errors here propagate through the entire pipeline.

**Case Folding**: Reduces vocabulary size by collapsing "The" → "the", but destroys meaningful distinctions (e.g., "US" vs. "us").

**Stopword Removal**: Discards high-frequency, low-information words (the, is, and). Reduces noise but may lose negation cues ("not good" → "good").

### Normalization: Stemming vs. Lemmatization

**Stemming**: *Heuristic, rule-based* truncation that strips affixes without linguistic knowledge. Fast but crude—"studies" → "studi", "better" → "better" (fails irregular forms). May produce non-words.

**Lemmatization**: *Linguistically informed* reduction to dictionary form using morphological analysis and POS tags. "studies" → "study", "better" → "good". Slower but preserves semantic validity.

> **Trade-off**: Stemming sacrifices accuracy for speed; lemmatization sacrifices speed for linguistic correctness.

### Vocabulary & Sparsity

**Type**: A unique word form in the vocabulary.  
**Token**: An individual occurrence of a type in the corpus.  
**Heap's Law**: Vocabulary grows sublinearly with corpus size—larger corpora always introduce new rare words.  
**Sparsity Problem**: Most terms are absent from most documents, yielding high-dimensional vectors dominated by zeros.

### Similarity vs. Distance

**TF-IDF Intuition**: Balances *local importance* (how frequent is this term in THIS document?) against *global rarity* (how rare is this term across ALL documents?). High TF-IDF = frequent here, rare elsewhere = discriminative.

**IDF Penalization**: Terms appearing in all documents (the, is) receive IDF → 0, contributing nothing to document representation. Rewards terms that distinguish documents.

**Log Dampening**: Raw counts are compressed logarithmically because a word appearing 100× is not 100× more important—diminishing marginal information.

### Distance Metrics: When to Use Which

**Cosine Similarity**: Measures *directional alignment* (angle between vectors). Ignores magnitude—ideal for comparing documents of different lengths. A short and long document on the same topic will have high cosine similarity.

**Euclidean Distance**: Measures *absolute positional difference* (straight-line distance). Sensitive to magnitude—two documents with identical word proportions but different lengths appear distant. Use only when magnitudes are meaningful or after normalization.

**Jaccard Coefficient**: Measures *set overlap* (shared terms / total unique terms). Binary presence/absence only—ignores frequency. Useful for comparing term sets, not weighted vectors.

> **Key Distinction**: Cosine asks "Do these point the same direction?" Euclidean asks "How far apart are these points?" Jaccard asks "How much do these sets overlap?"

### Regex & Pattern Matching

**Regular Expressions**: Declarative pattern language for string matching. Enables systematic text extraction, substitution, and validation.

**Greedy vs. Lazy Matching**: Greedy (`*`, `+`) matches as much as possible; lazy (`*?`, `+?`) matches as little as possible.

**Error Types in Matching**:
- **False Positive**: Matching what shouldn't match (e.g., "there" when searching for "the")
- **False Negative**: Missing what should match (e.g., missing "The" with lowercase pattern)

---

## Part 2: Classical Models & Classification

### Language Models: Predicting the Next Word

**N-gram Language Model**: Estimates probability of a word given preceding context. Uses *Markov assumption*—the next word depends only on the last $n-1$ words, not the entire history. Trades long-range accuracy for tractable estimation.

**Unigram**: No context; probability based only on word frequency. Ignores word order entirely.

**Bigram**: One word of context. "want" followed by "to" is likely; estimates conditional probabilities from corpus counts.

**Perplexity**: Measures model quality as *average branching factor*—the effective number of equally likely words the model considers at each position. **Lower perplexity = better model.** A model that always predicts correctly has perplexity 1.

### The Zero-Probability Problem & Smoothing

**Problem**: Unseen n-grams receive probability 0 under MLE, making entire sequences impossible if they contain any unseen combination.

**Laplace (Add-1) Smoothing**: Adds 1 to every count, ensuring no zero probabilities. Redistributes mass to unseen events. Too aggressive for large vocabularies—steals too much probability from observed events.

**Kneser-Ney Smoothing**: More sophisticated; redistributes probability based on how many different contexts a word appears in, not just frequency.

> **Key Insight**: Smoothing is a bias-variance trade-off—we accept bias (distorted estimates) to reduce variance (unstable estimates from sparse data).

### Naive Bayes Classification

**Core Assumption**: The *conditional independence assumption*—features (words) are independent given the class label. "Naive" because this is almost never true in language (words co-occur meaningfully), yet the classifier often performs well regardless.

**Intuition**: A generative model that asks "How likely is this document to have been generated by class X?" Multiplies prior probability of the class by likelihood of each word appearing in that class.

**Why It Works Despite Being Wrong**: Even if absolute probabilities are miscalibrated, the *ranking* of classes often remains correct. We only need to identify the most probable class, not estimate exact probabilities.

**Add-1 Smoothing in NB**: Prevents words unseen in a class from zeroing out that class's probability. Every word has at least minimal probability in every class.

### Generative vs. Discriminative Models

**Generative (Naive Bayes)**: Models full joint distribution $P(X, Y)$—how features and labels occur together. Makes assumptions about data generation process. Can generate synthetic examples.

**Discriminative (Logistic Regression)**: Models decision boundary $P(Y | X)$ directly—focuses only on distinguishing classes. Makes fewer assumptions, often more accurate when assumptions of generative models are violated.

> **Analogy**: Generative = "What does a cat look like?" vs. Discriminative = "How do I tell cats from dogs?"

### Logistic Regression

**Intuition**: A linear classifier that uses the *sigmoid function* to squash a weighted sum of features into a probability. The weights determine feature importance for classification.

**Sigmoid Function**: Maps any real number to (0, 1). Large positive inputs → probability near 1; large negative → near 0; zero → 0.5.

**Decision Boundary**: The hyperplane where $P(Y=1) = 0.5$. Points on one side classified as positive, other side as negative.

**Cross-Entropy Loss**: Penalizes confident wrong predictions heavily. Forces model to assign high probability to correct class.

**Gradient Descent**: Iteratively adjusts weights in direction that reduces loss. Learning rate controls step size—too large causes oscillation, too small causes slow convergence.

### Decision Trees & Information Theory

**Entropy**: Measures *uncertainty* in a distribution. Maximum entropy = uniform (most uncertain); minimum = deterministic (no uncertainty). Unit: bits.

**Information Gain**: Reduction in entropy after observing a feature. High IG means the feature is highly informative for classification.

**ID3 Algorithm**: Recursively selects feature with highest information gain to split on, building a tree until leaves are pure (single class) or no features remain.

> **Overfitting Risk**: Deep trees memorize training data. Pruning or depth limits prevent this.

### Support Vector Machines (SVMs)

**Intuition**: Finds the *maximum-margin hyperplane*—the decision boundary that maximizes distance to the nearest training points (support vectors).

**Margin**: Distance from boundary to nearest points. Larger margin → better generalization to unseen data.

**Kernel Trick**: Maps data to higher-dimensional space where linear separation becomes possible, without explicitly computing the transformation.

### k-Nearest Neighbors (k-NN)

**Intuition**: "Tell me who your neighbors are, and I'll tell you who you are." Classifies a point based on majority vote among its k closest training examples.

**Lazy Learning**: No explicit training phase; all computation happens at prediction time.

**Distance Metric Choice**: Critical—Euclidean sensitive to scale; cosine focuses on direction.

**Curse of Dimensionality**: In high dimensions, all points become approximately equidistant, making neighbor-based methods ineffective.

---

## Part 3: Vector Semantics & Embeddings

### The Distributional Hypothesis

**Core Principle**: "You shall know a word by the company it keeps" (Firth). Words appearing in similar contexts have similar meanings.

**Implication**: We can learn word meaning from raw text without explicit definitions—semantic similarity emerges from usage patterns.

### From Counts to Vectors

**Sparse Representations (Traditional)**: High-dimensional vectors (vocabulary-sized), mostly zeros. TF-IDF, co-occurrence counts, PPMI matrices.

**Dense Representations (Embeddings)**: Low-dimensional (50–300), real-valued vectors learned to capture semantic relationships. Word2Vec, GloVe.

> **Key Shift**: Sparse vectors encode *presence*; dense vectors encode *meaning*.

### Pointwise Mutual Information (PMI) & PPMI

**PMI Intuition**: Measures association strength between words *beyond chance*. If "ice" and "cream" co-occur far more than their individual frequencies predict, PMI is high (strong attraction).

**PPMI (Positive PMI)**: Clamps negative values to zero. Negative PMI means "co-occurs less than expected"—but reliably detecting non-co-occurrence requires massive data. PPMI treats all non-associations as zero.

> **Why PPMI Works**: Prioritizes *informative* co-occurrences over frequent but uninformative ones (like co-occurrence with "the").

### Word2Vec: Prediction-Based Embeddings

**Core Idea**: Learn embeddings by predicting words from context (or vice versa). Good embeddings make good predictions.

**Skip-Gram**: Given a target word, predict surrounding context words. "cat" → predict "the", "sat", "on", "mat".

**CBOW (Continuous Bag of Words)**: Given context words, predict the target. "the", "sat", "on", "mat" → predict "cat".

**Negative Sampling**: Instead of predicting over entire vocabulary (expensive), contrast true context against randomly sampled "noise" words. Learn embeddings that score true pairs high and noise pairs low.

> **Why It Works**: The prediction task forces embeddings to encode distributional similarity—words used similarly must have similar embeddings to make similar predictions.

### Embedding Properties

**Semantic Arithmetic**: $\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}$. Linear relationships encode analogies.

**Clustering**: Words with similar meanings cluster in embedding space. Synonyms are neighbors.

**Dimensionality**: Dense embeddings (300D) capture nuanced relationships that sparse vectors (vocabulary-sized) cannot efficiently encode.

### Neural Network Foundations

**Neuron**: Computes weighted sum of inputs plus bias, passes through nonlinear activation. The activation function enables learning non-linear decision boundaries.

**Activation Functions**:
- **Sigmoid**: Squashes to (0,1), interprets as probability. Suffers *vanishing gradients* at extremes.
- **Tanh**: Squashes to (-1,1), zero-centered. Still vanishing gradient problem.
- **ReLU**: $\max(0, x)$. No vanishing gradient for positive inputs. Simple, efficient. Risk of "dead neurons" if always negative.

**Softmax**: Converts raw scores (logits) into probability distribution over classes. Exponential amplifies differences—largest logit dominates.

**Cross-Entropy Loss**: Measures divergence between predicted and true distributions. Heavily penalizes confident wrong predictions.

### Forward and Backward Propagation

**Forward Pass**: Input flows through network layers, each applying linear transformation + nonlinearity, until final output.

**Backpropagation**: Chain rule applied backward through computation graph. Computes gradients of loss with respect to all parameters.

**Gradient Descent**: Updates parameters opposite to gradient direction. Learning rate controls step size—too large overshoots, too small converges slowly.

### XOR Problem & Deep Networks

**XOR Problem**: Single-layer perceptrons cannot learn XOR (non-linearly separable). Motivates the need for hidden layers.

**Hidden Layers**: Create intermediate representations that make the problem linearly separable. Deep networks stack many such transformations.

---

## Part 4: Sequence Modeling

### Hidden Markov Models (HMMs)

**Core Idea**: A generative model that assumes observed data (words) are produced by hidden states (tags) following probabilistic transitions. We see words; tags are latent.

**Markov Assumption**: The current state depends only on the immediately preceding state—history is compressed to one step back. Enables tractable computation.

**Output Independence**: Each observation depends only on the state that produced it, not on neighboring observations or states.

**Transition Probabilities**: Capture *grammatical constraints*—after a determiner, a noun is likely; after a modal verb, a base verb is almost certain.

**Emission Probabilities**: Capture *lexical information*—"the" is almost always a determiner; "fish" can be noun or verb.

### Viterbi Algorithm

**Purpose**: Finds the most probable sequence of hidden states (optimal tagging) given an observation sequence.

**Dynamic Programming**: Exploits optimal substructure—the best path to any state at time $t$ depends only on the best paths to all states at time $t-1$. Avoids exponential enumeration.

**Backpointers**: Record *which previous state* yielded the maximum at each step. After forward pass, trace back to recover the full optimal path.

> **Key Insight**: Viterbi resolves lexical ambiguity ("can" = modal vs. noun) by exploiting sequential constraints—grammatical context eliminates impossible paths.

### BIO Tagging Scheme

**B-X**: Beginning of entity type X (first token of a multi-token entity)  
**I-X**: Inside entity type X (continuation tokens)  
**O**: Outside any entity  

> **Why B/I Distinction**: Without it, we cannot distinguish "[Marie Curie] [Pierre Curie]" (two entities) from "[Marie Curie Pierre Curie]" (one entity).

### HMM vs. CRF

**HMM (Generative)**: Models joint probability $P(X, Y)$. Limited features—only current observation informs emission.

**CRF (Discriminative)**: Models conditional probability $P(Y | X)$ directly. Allows *arbitrary overlapping features* (word shapes, prefixes, entire input context). Handles unknown words better via feature engineering.

### Recurrent Neural Networks (RNNs)

**Core Idea**: Hidden state carries information forward through the sequence, updating at each timestep with both current input and previous state.

**Memory Mechanism**: The recurrence $h_{t-1} \to h_t$ allows the network to model sequential dependencies without a fixed context window.

**Limitation**: Gradients must flow backward through many timesteps during training. With repeated multiplication, they either *vanish* (preventing learning of long-range dependencies) or *explode* (causing instability).

### Vanishing Gradient Problem

**Cause**: During backpropagation through time, gradients are multiplied by the same weight matrix at each step. If eigenvalues < 1, gradients shrink exponentially; if > 1, they explode.

**Consequence**: The network cannot learn dependencies spanning many timesteps. Information from early in the sequence fails to influence later predictions.

### LSTM: Long Short-Term Memory

**Solution**: Introduce a *cell state* (long-term memory highway) and *gating mechanisms* that regulate information flow.

**Forget Gate**: Decides what to *erase* from the cell state. Allows the network to clear irrelevant information.

**Input Gate**: Decides what *new information* to write to the cell state. Enables selective updating.

**Output Gate**: Decides what to *expose* from the cell state to the hidden state for downstream use.

**Cell State Update (Additive)**: $c_t = f_t \odot c_{t-1} + i_t \odot g_t$. The *addition* (not multiplication) creates a gradient highway—gradients can flow backward through the addition unchanged, mitigating vanishing gradients.

> **Key Insight**: LSTM gates learn *when* to remember and *when* to forget, preserving important information across long sequences.

### Encoder-Decoder (Seq2Seq)

**Architecture**: Encoder compresses input sequence into a context vector; decoder generates output sequence from this context.

**Bottleneck Problem**: A single fixed vector must encode everything about the source. Works poorly for long sequences—information gets lost.

### Attention Mechanism

**Solution**: Allow the decoder to *dynamically* focus on different parts of the encoder output at each generation step.

**Attention Weights**: Learned relevance scores indicating how much each source position matters for the current output position. Forms a soft alignment between source and target.

**Context Vector**: Weighted average of encoder states, customized for each decoder step. When translating "green" to "verde," attention focuses on "green."

> **Key Insight**: Attention resolves the fixed-context bottleneck—the decoder can access the entire source sequence, not just a compressed summary.

---

## Part 5: Modern Architectures & LLMs

### Transformers: The Foundation

**Self-Attention**: Each position attends to all positions in the sequence, weighted by learned relevance scores. No recurrence—all computations happen in parallel.

**Query-Key-Value Framework**:
- **Query**: "What am I looking for?"
- **Key**: "How can I be matched?"  
- **Value**: "What do I contribute if matched?"

**Scaled Dot-Product**: $QK^T / \sqrt{d_k}$. The scaling prevents dot products from growing too large, which would push softmax into saturation.

**Multi-Head Attention**: Multiple parallel attention operations with different projections. Different heads capture different types of relationships (syntactic, semantic, positional).

**Positional Encoding**: Self-attention is permutation-invariant (ignores order). Sinusoidal encodings inject position information, allowing the model to learn relative positions.

**Residual Connections + Layer Normalization**: Enable training very deep networks. Residuals allow gradient flow; LayerNorm stabilizes activations.

### Causal vs. Bidirectional Attention

**Causal (GPT-style)**: Each position can only attend to previous positions. Enables autoregressive generation—predict next token from left context only.

**Bidirectional (BERT-style)**: Each position attends to all positions. Captures richer context but cannot generate autoregressively.

### BERT: Encoder-Only

**Architecture**: Stacked Transformer encoder blocks with bidirectional attention.

**Masked Language Model (MLM)**: Predict randomly masked tokens from full bidirectional context. 15% of tokens masked during training.

**Use Case**: Understanding tasks—classification, NER, extractive QA. The `[CLS]` token aggregates sequence information for classification.

**Contextual Embeddings**: Same word gets different vectors in different contexts. "Bank" in "river bank" vs. "investment bank" clusters separately.

### GPT: Decoder-Only

**Architecture**: Stacked Transformer decoder blocks with causal masking.

**Causal LM**: Predict next token given all preceding tokens. Each position sees only left context.

**Use Case**: Generation tasks—text completion, dialogue, creative writing. Generates one token at a time, autoregressively.

**Scaling Laws**: Performance improves predictably with model size, data size, and compute. Larger models exhibit emergent abilities not present in smaller ones.

### T5: Encoder-Decoder

**Text-to-Text Framework**: All tasks formatted as "task prefix: input" → "output". Classification, translation, summarization—all converted to text generation.

**Span Corruption**: Pretraining objective that masks spans of tokens (not just individual tokens).

### Decoding Strategies

**Greedy**: Always pick highest-probability next token. Fast but generic, often repetitive.

**Beam Search**: Maintain $k$ candidate sequences, extending each and keeping top $k$ by cumulative probability. Better quality, still deterministic.

**Temperature**: Divide logits by $\tau$ before softmax. Low $\tau$ → sharper distribution (more deterministic); high $\tau$ → flatter (more diverse).

**Top-k Sampling**: Sample from $k$ most probable tokens only. Truncates long tail of unlikely words.

**Top-p (Nucleus) Sampling**: Sample from smallest set of tokens whose cumulative probability exceeds $p$. Adapts dynamically to distribution shape.

### Prompting & In-Context Learning

**Zero-Shot**: Describe task in natural language, no examples. "Translate English to French: Hello."

**Few-Shot**: Provide labeled examples in the prompt. Model learns task pattern from demonstrations during inference—no gradient updates.

**Chain-of-Thought (CoT)**: Include reasoning steps in examples. "Let's think step by step..." Improves multi-step reasoning by encouraging the model to show its work.

**Instruction Tuning (SFT)**: Finetune on (instruction, response) pairs. Improves instruction-following ability.

### RLHF: Reinforcement Learning from Human Feedback

**Problem**: LLMs are trained to predict likely text, not necessarily helpful/safe text.

**Solution**:
1. Collect human comparisons of model outputs (which response is better?)
2. Train a *reward model* to predict human preferences
3. Optimize the LLM policy to maximize reward while staying close to base model (KL penalty)

**Outcome**: Models that are more aligned with human intentions—helpful, harmless, honest.

### Parameter-Efficient Finetuning

**LoRA (Low-Rank Adaptation)**: Instead of updating all weights, add small low-rank matrices $A$ and $B$ to frozen weights. Trains far fewer parameters with similar performance.

**Why It Works**: Task-specific adaptations often lie in a low-dimensional subspace. Full finetuning is overkill.

### Key LLM Challenges

**Hallucination**: Models generate plausible-sounding but factually incorrect content. Trained for fluency, not factuality.

**Context Length**: Self-attention has $O(n^2)$ complexity. Long documents strain memory and computation.

**Bias**: Models reflect biases in training data. Can perpetuate or amplify harmful stereotypes.

---

## Comparative Tables

### Distance Metrics: When to Use Which

| Metric | Measures | Sensitive To | Best For |
|--------|----------|--------------|----------|
| **Euclidean** | Absolute position (straight-line) | Magnitude differences | Equal-length documents, after normalization |
| **Cosine** | Directional alignment (angle) | Direction only, ignores magnitude | Documents of varying lengths, topic similarity |
| **Jaccard** | Set overlap (shared/total) | Presence only, ignores frequency | Binary features, set comparison |

### Normalization: Stemming vs. Lemmatization

| Aspect | Stemming | Lemmatization |
|--------|----------|---------------|
| **Approach** | Heuristic, rule-based truncation | Linguistically informed, uses morphology |
| **Speed** | Fast | Slower |
| **Output** | May produce non-words ("studi", "accur") | Always valid dictionary forms ("study", "accurate") |
| **Irregular Forms** | Often fails ("better" → "better") | Handles correctly ("better" → "good") |
| **Trade-off** | Speed over accuracy | Accuracy over speed |

### Generative vs. Discriminative Models

| Aspect | Generative (Naive Bayes) | Discriminative (Logistic Regression) |
|--------|--------------------------|-------------------------------------|
| **Models** | Joint $P(X, Y)$ | Conditional $P(Y \mid X)$ directly |
| **Assumptions** | Strong (independence) | Fewer |
| **Features** | Limited by model structure | Arbitrary, overlapping |
| **Can Generate Data?** | Yes | No |
| **Typically Better When** | Small training data | Large training data, violated assumptions |

### Sequence Labeling: HMM vs. CRF

| Aspect | HMM | CRF |
|--------|-----|-----|
| **Model Type** | Generative | Discriminative |
| **Feature Access** | Current observation only | Entire input sequence |
| **Feature Design** | Limited (emission only) | Arbitrary overlapping features |
| **Unknown Words** | Problematic (zero emission) | Handles via features (prefixes, shapes) |
| **Training** | Count-based MLE | Gradient-based optimization |

### RNN vs. LSTM vs. Transformer

| Aspect | Vanilla RNN | LSTM | Transformer |
|--------|-------------|------|-------------|
| **Memory** | Hidden state | Cell state + hidden | Attention over all positions |
| **Long-range** | Vanishing gradient | Gating preserves gradients | Direct attention, no decay |
| **Parallelization** | Sequential (slow) | Sequential (slow) | Fully parallel (fast) |
| **Position Encoding** | Implicit (order of processing) | Implicit | Explicit (sinusoidal/learned) |

### BERT vs. GPT

| Aspect | BERT | GPT |
|--------|------|-----|
| **Architecture** | Encoder-only | Decoder-only |
| **Attention** | Bidirectional | Causal (left-to-right) |
| **Pretraining** | Masked LM (predict masked tokens) | Causal LM (predict next token) |
| **Primary Strength** | Understanding (classification, NER, QA) | Generation (completion, dialogue) |
| **Can Generate Text?** | Not natively | Yes, autoregressively |

### Static vs. Contextual Embeddings

| Aspect | Static (Word2Vec, GloVe) | Contextual (BERT, GPT) |
|--------|--------------------------|------------------------|
| **Representation** | One vector per word | Different vector per occurrence |
| **Polysemy** | Single averaged meaning | Context-appropriate sense |
| **Training** | Shallow window-based | Deep Transformer pretraining |
| **Size** | Millions of parameters | Billions of parameters |

---

## Limitation → Solution Ladder

The conceptual evolution of NLP methods, showing how each limitation motivated the next innovation:

### Representation Problems

| Limitation | Solution | Key Insight |
|------------|----------|-------------|
| **Raw counts favor common words** | TF-IDF weighting | Penalize words appearing everywhere (IDF); reward discriminative terms |
| **High-dimensional sparse vectors** | Dense embeddings (Word2Vec) | Learn compact representations capturing semantic relationships |
| **Static embeddings ignore context** | Contextual embeddings (BERT) | Same word gets different vectors based on surrounding context |

### Language Modeling Problems

| Limitation | Solution | Key Insight |
|------------|----------|-------------|
| **Zero probability for unseen n-grams** | Smoothing (Laplace, Kneser-Ney) | Redistribute probability mass to unseen events |
| **Fixed n-gram context window** | RNNs (recurrent connection) | Hidden state carries information forward through arbitrary-length sequences |
| **Vanishing gradients in RNNs** | LSTMs (gating + cell state) | Additive cell update creates gradient highway; gates regulate information flow |
| **Sequential processing is slow** | Transformers (attention) | Parallel computation; every position attends to every position directly |
| **No position information in attention** | Positional encodings | Inject position via sinusoidal/learned vectors |

### Sequence-to-Sequence Problems

| Limitation | Solution | Key Insight |
|------------|----------|-------------|
| **Fixed-size bottleneck vector** | Attention mechanism | Decoder dynamically attends to relevant encoder states at each step |
| **Separate encoder and decoder** | Decoder-only (GPT) | Single model handles both understanding and generation via causal masking |
| **Left-context only limits understanding** | Bidirectional encoder (BERT) | Full context in both directions for deep understanding |

### Classification Problems

| Limitation | Solution | Key Insight |
|------------|----------|-------------|
| **Independence assumption is unrealistic** | Logistic regression (discriminative) | Model decision boundary directly; fewer assumptions |
| **Linear decision boundaries insufficient** | Neural networks (hidden layers) | Transform inputs to make problem linearly separable |
| **Overfitting on small data** | Pretraining + finetuning | Leverage massive unlabeled corpora, then adapt to task |

### Sequence Labeling Problems

| Limitation | Solution | Key Insight |
|------------|----------|-------------|
| **HMM sees only current observation** | CRFs (discriminative) | Access entire input; arbitrary overlapping features |
| **Unknown words have zero emission** | Feature-based CRFs | Use word shapes, prefixes, suffixes—not just word identity |
| **Hand-crafted features are limited** | Neural sequence labelers (BiLSTM-CRF) | Learn features automatically; CRF layer models label dependencies |

### LLM Problems

| Limitation | Solution | Key Insight |
|------------|----------|-------------|
| **Models predict likely, not helpful** | RLHF | Train on human preferences; optimize for helpfulness |
| **Full finetuning is expensive** | LoRA (low-rank adaptation) | Most task-specific information lies in low-rank subspace |
| **Models hallucinate confidently** | Retrieval augmentation, citations | Ground generation in retrieved facts; provide sources |

---

## Ethics & Social Impact

### Language as Proxy for Identity

Language is *situated*—produced in a specific context by an individual with demographic characteristics. Text carries *latent information* about the author (age, gender, location, education). NLP models trained on this data can inadvertently learn and amplify these correlations.

### Key Ethical Concepts

**Exclusion**: Models trained on demographically biased data (e.g., predominantly white, male, Western authors) perform worse on underrepresented groups. The *i.i.d.* assumption implies all language is like the training data—a false assumption that disadvantages minorities.

> *Consequence*: Technology becomes less usable for already marginalized groups, reinforcing existing inequalities.

**Overgeneralization**: Classification models assign labels to individuals based on group patterns. False positives seem harmless for age prediction but become problematic for sensitive attributes (sexual orientation, religion, political views).

> *Question to ask*: "Would a false answer be worse than no answer?"

**Bias Confirmation**: If research repeatedly associates certain characteristics (e.g., violence, negative emotion) with specific groups, it can reinforce existing stereotypes through the *availability heuristic*—people infer that frequently researched associations are more important.

**Topic Overexposure**: Research trends (e.g., "neural" papers) create biases about what matters. Overexposing certain topics or groups can distort perception of their prevalence or importance.

**Underexposure**: Focusing predominantly on English (a global outlier morphologically and syntactically) limits typological diversity in NLP research. Most resources exist for Indo-European languages; small languages are underserved.

### Dual Use

NLP techniques can be used for beneficial or harmful purposes:
- Stylometric analysis: Historical scholarship vs. identifying dissidents
- Text classification: Spam detection vs. censorship
- Language generation: Creative writing vs. fake reviews

> *Responsibility*: Researchers may not directly cause harm, but should acknowledge how their work can be appropriated and lead informed discourse.

### Countermeasures

| Problem | Solution |
|---------|----------|
| **Exclusion** | Downsample over-represented groups; use demographic priors |
| **Overgeneralization** | Add dummy labels; error weighting; confidence thresholds |
| **Bias in Embeddings** | Debias word vectors; evaluate for demographic fairness |
| **Data Imbalance** | Stratified sampling; data augmentation for minorities |

### Core Principle

NLP directly affects individuals' lives. As language technology becomes ubiquitous, ethical considerations must be integrated from data collection through deployment—not treated as an afterthought.
