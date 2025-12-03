# LANGUAGE & AI: REPRESENTATION

[IMAGE: Abstract black and white drawing with flowing, overlapping lines]

Chris Emmery
Department of Cognitive Science & AI
Tilburg University

@cmry • @_cmry • @cmry • cmry.github.io

---

[IMAGE: Recycle symbol icon] RECAP PREVIOUS LECTURES

*   We looked at some interesting language **features**.
*   We discussed how to convert text into **vectors**, using a discrete $n$-gram representation.
*   Using this vector space, we looked at distance and how it might be employed for **similarity**.
*   We also looked at how we might infer **probabilistic** models from data, both to **model**, and **predict** from, the data.

---

[IMAGE: Abacus icon] MEANING AS COUNTS

[IMAGE: Group of young children sitting together, looking attentively at something off-screen.]

---

[IMAGE: Speech bubble icon] HUMAN LANGUAGE & AI

*   Language / text is distinctive: unique driver behind our evolutionary development and dominance.
*   Any instruction to an AI problem can be phrased as a language problem, but explaining what is happening is also a problem of language!
*   Language is social, contextualized, and has deliberate meaning. Intentional and evolving communication.

---

[IMAGE: Brain icon] MORE FEATURES

*   Discrete, symbolic, and categorical ([IMAGE: Christmas tree icon] = tree, [IMAGE: Person running icon] = run).
*   Symbols hold across media: speech, gestures, text.

> Cognitive Science: does our brain contain symbolic representations, or is it just continuous neural activations over **sequences**?

---

[IMAGE: Yarn/thread icon] REPRESENTING LANGUAGE AS A SEQUENCE

We have mostly looked at condensed representations (BoW, one vector per input), but ignored sequential properties:

$$
V = \text{(is it language or sequential)} \begin{bmatrix}
0 & 0 & 1 & 0 & 0 \\
1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 1 & 0 \\
1 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0
\end{bmatrix}
$$

---

[IMAGE: Rocket icon] DISTRIBUTIONAL SIMILARITY:
REPRESENTING CONTEXT AS A VECTOR

"You shall know a word by the company it keeps" (Firth, 1957)

| Target Word | Context (preceding) | Context (following) |
| :--- | :--- | :--- |
| **cherry** | is traditionally followed by | pie, a traditional dessert |
| **strawberry** | often mixed, such as | rhubarb pie. Apple pie |
| **digital** | computer peripherals and personal | assistants. These devices usually |
| **information** | a computer. This includes | available on the internet |

| | aardvark | ... | computer | data | result | pie | sugar | ... |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **cherry** | 0 | ... | 2 | 8 | 9 | 442 | 25 | ... |
| **strawberry** | 0 | ... | 0 | 0 | 1 | 60 | 19 | ... |
| **digital** | 0 | ... | 1670 | 1683 | 85 | 5 | 4 | ... |
| **information** | 0 | ... | 3325 | 3982 | 378 | 5 | 13 | ... |

> Works with $tf \cdot idf$ and Cosine Similarity!

---

[IMAGE: Printer icon] BEYOND TF $\cdot$ IDF: POSITIVE POINTWISE MUTUAL INFORMATION

$$
I(x, y) = \log_2 \frac{P(x, y)}{P(x)P(y)} \quad \text{PMI}(w, c) = \log_2 \frac{P(w, c)}{P(w)P(c)}
$$

$$
\text{PPMI}(w, c) = \max \left(\log_2 \frac{P(w, c)}{P(w)P(c)}, 0 \right)
$$

---

[IMAGE: Eyes icon] PPMI SINGLE EXAMPLE

$$
P(w = \text{information}, c = \text{data}) = \frac{3982}{11716} = .3399
$$

$$
P(w = \text{information}) = \frac{7703}{11716} = .6575
$$

$$
P(c = \text{data}) = \frac{5673}{11716} = .4842
$$

$$
\text{ppmi}(\text{information}, \text{data}) = \log_2 \frac{.3399}{.6575 \times .4842} = .0944
$$

!! Rare words get high PPMI, offset:

$$
\text{PPMI}_{\alpha}(w, c) = \max \left( \log_2 \frac{P(w, c)}{P(w) P_{\alpha}(c)}, 0 \right)
\quad P_{\alpha}(c) = \frac{\text{count}(c)^{\alpha}}{\sum_c \text{count}(c)^{\alpha}}
$$

---

[IMAGE: Vise icon] INTRODUCING REPRESENTATIONAL DENSITY

*   The vectors so far are very **sparse** and large ($|V|$).
*   In turns out that for most language problems, **dense** vectors work much better.
*   Why? Less weights, easier to optimize, dimensionality better represented.

---

SINGULAR VALUE DECOMPOSITION (SVD)
$\rightarrow$
LATENT SEMANTIC ANALYSIS (LSA)

[IMAGE: Diagram illustrating SVD decomposition of an $n \times m$ matrix into $n \times r$, $r \times r$ (Diagonal matrix: concept strengths), and $r \times m$ matrices. Labels indicate $n$ documents, $m$ terms, and $r$ concepts.]

Images from the Polo Club of Data Science.

---

[IMAGE: Bar chart icon] LSA INTUITIONS

[IMAGE: Diagram illustrating the factorization of a $6 \times 5$ document-term matrix into a $6 \times 2$ document-concept similarity matrix, a $2 \times 2$ diagonal concept strength matrix (e.g., 9.64 for CS-concept), and a $2 \times 5$ term-concept similarity matrix. The dimensions show how documents (CS docs, MD docs) and terms (data, info, retrieval, brain, lung) are mapped to two concepts (CS concept, MD concept).]

document-concept
similarity matrix

term-concept
similarity matrix

---

[IMAGE: Scissors icon] LOWER-RANK APPROXIMATIONS

[IMAGE: Three diagrams illustrating SVD approximations: (a) full SVD, (b) reduced SVD, and (c) truncated SVD. Truncated SVD approximates matrix $A$ as the product of $U$ (M x K), $\Sigma$ (K x K), and $V^T$ (K x N), where $K$ is the lower rank.]

Image from Susanne Suter (2013)

---

[IMAGE: Sunglasses icon] PREDICTING WORD MEANING

[IMAGE: Blank image placeholder for a figure/diagram.]

---

[IMAGE: Cat icon] LEXICAL SEMANTICS

*   BoW models ascribe **atomic** meaning: LIFE
*   We want to have a broader sense of meaning:
    *   cat similar to dog (categories)
    *   cold is opposite of hot (antonyms)
    *   fake/copy different connotations
    *   capture related sequences: buy, sell, pay
    *   mouse = rodent and controller (polysemy)
    *   couch = sofa (synonymy)
*   Useful for meaning-related tasks (Q&A).
*   Have a sense of similarity, relatedness, and semantic role.

---

[IMAGE: Archer/bow and arrow icon] VECTOR SEMANTICS

'Embed' words; map from one structure to the other.

[IMAGE: Three plots illustrating vector analogies. Left: Male-Female (man to king, woman to queen). Center: Verb tense (walking to walked, swimming to swam). Right: Country-Capital (lines connecting country names to their respective capitals).]

> **Count! vs. Predict!**

---

[IMAGE: Boxing glove icon] WORD2VEC

*   Algorithm suite: Continuous Bag-of-Words (**CBOW**, predict word given context), and (**SGNS**, Skip-Gram with Negative Sampling) — we focus on the latter.
*   Fast, efficient to train, share embeddings (**gensim**).
*   Static (different from transformers).

> **Task:** “Is word $w$ likely to show up near apricot?”
> **Target:** Self-supervised, actual word is the gold standard.
> $\rightarrow$ only use embedding encoding

Mikolov et al. (*2013a*), Mikolov et al. *2013b*.

---

[IMAGE: Sparkles icon] DIFFERENCE WITH 'FANCY' NLMS

*   Binary classification, simple Logistic Regression.
*   How? Skip-grams:
    *   Treat the target word and a neighboring context word as **positive examples**.
    *   Randomly sample other words in the lexicon to get **negative samples**.
    *   Use logistic regression to train a classifier to distinguish those two cases.
    *   Use the learned weights as the **embeddings**.

---

[IMAGE: Bar chart/graph icon] CLASSIFIER

| | lemon, | a | tablespoon | of | apricot | jam, | a | pinch | |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | $c1$ | $c2$ | $w$ | $c3$ | $c4$ | | | | |

| $w$ | $c_{pos}$ | | $w$ | $c_{neg}^*$ |
| :--- | :--- | :--- | :--- | :--- |
| apricot | tablespoon | | apricot | aardvark |
| apricot | of | | apricot | my |
| apricot | jam | | apricot | where |
| apricot | a | | apricot | coaxial |

$$
P(+ \mid w, c) \quad P(- \mid w, c) = 1 - P(+ \mid w, c) \quad \text{Similarity}(w, c) \approx c \cdot w
$$

$$
\sigma(x) = \frac{1}{1 + \exp(-x)} \quad P(+ \mid w, c_{1:L}) = L \sigma(\mathbf{c} \cdot \mathbf{w}) \prod_{i=1}^L
$$

> Maximize dot product between $c_{pos}$, minimize $c_{neg}$.
> *Negative examples are sampled using weighted $\alpha = 0.75$ unigram count, similar to PPMI.

---

[IMAGE: Diamond icon] EMBEDDING REPRESENTATION

[IMAGE: Diagram showing the relationship between target embedding matrix W and context embedding matrix C, illustrating how the similarity between a target word j and a context word k is calculated via the dot product of the corresponding row vector in W and column vector in C. $W$ is $|V| \times d$, $C$ is $d \times |V|$.]

> Either we: represent word $i$ (above $j$) with the vector $w_i + c_i$. Or: yeet $C$, and represent word $i$ by the vector $w_i$. Windows size $L$ important to tune.

---

[IMAGE: Crown icon] EMBEDDING PROPERTIES

[IMAGE: Diagram illustrating word embedding properties and analogy. Tables show 7D embeddings for various words, which are then visualized in 2D space after dimensionality reduction. The visualizations show semantic relationships preserved, such as the distance between "cat" and "kitten" vs. "dog" and the vector offset for gender/royalty (man/king/woman/queen).]

Source image unknown.

---

[IMAGE: Test tube icon] EVALUATING EMBEDDINGS

Downstream and intrinsic evaluations. Latter:

$$
\hat{b}^* = \arg \min_x \text{distance}(\mathbf{x}, \mathbf{a}^* - \mathbf{a} + \mathbf{b})
$$

> Allocational and representational harm (also see this talk by Kate Crawford). Try it yourself.

---

[IMAGE: Robot icon] NEURAL MODELS OF LANGUAGE

[IMAGE: Black and white close-up of tangled, bare branches.]

---

[IMAGE: Unicorn icon] THE NEURAL IN DEEP LEARNING

*   Early work biologically inspired, not anymore.
*   This lecture: **feedforward network**.
*   Stack of logistic regression models, but:
    *   Non-linearities; one layer can learn any function.
    *   Representation learning; no hand-crafting features.

---

[IMAGE: Pool ball 8 icon] SINGLE NEURON

[IMAGE: Diagram of a single neuron showing inputs ($x_1, x_2, x_3$) multiplied by weights ($w_1, w_2, w_3$), summed with bias ($b$), and passed through an activation function ($\sigma$) to produce output ($y$).]

$$
y = \sigma(\mathbf{w} \cdot \mathbf{x} + b) = \frac{1}{1 + \exp(-(\mathbf{w} \cdot \mathbf{x} + b))}
$$

$$
\text{Often uses tanh } \left( y = \frac{e^z - e^{-z}}{e^z + e^{-z}} \right) \text{ or ReLU}
$$

$$
y = \max (z, 0) \text{ as activations rather than sigmoid.}
$$

---

[IMAGE: Spiderweb icon] NEURAL NETWORK

[IMAGE: Diagram showing a fully connected feedforward neural network structure (Input layer $\rightarrow$ Hidden layer $\rightarrow$ Output layer) and two plots: (a) The original $x$ space showing non-linearly separable data, and (b) The new (linearly separable) $h$ space, illustrating how the hidden layer transforms the data for easier classification.]

---

[IMAGE: Red X icon] FORMALLY

$$
h = \sigma(Wx + b)
$$

$$
z = Uh
$$

$$
y = \text{softmax}(z) \quad \text{softmax}(z_i) = \frac{\exp(z_i)}{\sum_{j=1}^d \exp(z_j)} \quad 1 \le i \le d
$$

for $i$ in $1...n$

$$
\mathbf{z}^{[i]} = \mathbf{W}^{[i]} \mathbf{a}^{[i-1]} + \mathbf{b}^{[i]}
$$

$$
\mathbf{a}^{[i]} = g^{[i]} (\mathbf{z}^{[i]})
$$

$$
\hat{y} = \mathbf{a}^{[n]}
$$

---

[IMAGE: Thumbs up icon] NNS FOR CLASSIFICATION

*   Learn representation for input documents.
*   Output predictions using softmax.
*   **Pre-train**: use existing word embeddings as input!
*   Encode longer histories, generalize over different meanings.

[IMAGE: Diagram of a Neural Network for document classification. Input word embeddings are concatenated, fed through a hidden layer (h) via weights W, and then mapped to an output layer (y) via U to produce probabilities p(+) for positive, negative, and neutral classes using softmax.]

---

[IMAGE: Globe icon] NNS FOR LANGUAGE MODELLING

*   Approximate probability of a word based on $N$ previous word embeddings: $P(w_t \mid w_{t-N+1}, \dots, w_{t-1})$.
*   We get these by multiplying a one-hot vector (shown earlier) by embedding matrix $E$.

$$
e = [E x_{t-3}; E x_{t-2}; E x_{t-1}]
$$

$$
h = \sigma(We + b)
$$

$$
z = Uh
$$

$$
\hat{y} = \text{softmax}(z)
$$

[IMAGE: Diagram of a Neural Network for Language Modeling. The input is a one-hot vector for the current word $w_t$ (preceded by context $w_{t-1}$ to $w_{t-3}$). The one-hot vector $x$ is multiplied by embedding matrix $E$ to get embedding $e$. $e$ is concatenated and fed into the hidden layer $h$ via $W$, and $h$ outputs predictions $y$ via $U$ and softmax (e.g., $p(\text{aardvark}\mid...)$, $p(\text{fish}\mid...)$, $p(\text{zebra}\mid...)$).]

---

[IMAGE: Blank page 30]

---

[IMAGE: Notepad and pencil icon] A NOTE ON BACKPROP

*   Modern NN/DL Libraries (PyTorch, TensorFlow, Keras) have auto-differentiation.
*   If you want to read the chapters on derivatives, the chain rule, gradient descent, backpropagation, etc. in detail: knock yourself out.
*   I will not ask you to calculate gradient updates by hand.

---

[IMAGE: Waving hand icon] QUESTIONS

Post on the Discussion board and join class on Thursdays!