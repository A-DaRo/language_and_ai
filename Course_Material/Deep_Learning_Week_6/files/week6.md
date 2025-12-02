# LANGUAGE & AI: DEEP LEARNING

[IMAGE: Black and white image of a twisted, bare tree canopy]

Chris Emmery
Department of Cognitive Science & AI
Tilburg University

@cmry • @_cmry • @cmry • cmry.github.io

---

[IMAGE: Recycle symbol icon] RECAP PREVIOUS LECTURES

*   We looked at how language might be **noisy** as input.
*   We discussed several way to **represent** and **model** language: count and prediction-based, and sequential.
*   We looked at framing **prediction** tasks around those respective representations.

> Today, we discuss Deep Learning: end-to-end representation learning and predictions in one black box.

---

[IMAGE: Recycling arrows icon] RECURRENT MODELS

[IMAGE: Black and white photo showing multiple illuminated light bulbs against a dark ceiling]

---

[IMAGE: Left arrow icon] RELATION TO MATERIAL

*   Representing word similarity (Weeks 1 and 4).
*   Predicting outputs given input (Weeks 3 and 5).
*   Sequential (or temporal) structure (Week 5).
*   Semantic representations (Week 4)
*   Context windows (Week 4).

---

[IMAGE: Rice sheaves icon] RECURRENT NEURAL NETWORKS (RNNS)

*   Recurrent unit: encodes earlier steps in future decisions via 'lookback' for the hidden layers.

---

[IMAGE: Rooster icon] HOW DO THEY WORK?
Magic? No, just **U**.

[IMAGE: Diagram illustrating the architecture of a simple Recurrent Neural Network cell, showing inputs $h_{t-1}$ and $x_t$ transforming via matrices $U$ and $W$ respectively into the hidden state $h_t$, which then transforms via $V$ into the output $y_t$.]

---

[IMAGE: Egg icon] FORMALLY

$$
h = \sigma(Wx + b) \quad \rightarrow \quad h_t = g (Uh_{t-1} + Wx_t)
$$

$$
z = Uh \quad \quad \quad \quad \quad \quad \hat{y}_t = \text{softmax}(Vh_t)
$$

$$
\hat{y} = \text{softmax}(z)
$$

*   Dimensionality given $d_{in}$, $d_h$, and $d_{out}$:
*   $W \in \mathbb{R}^{d_h \times d_{in}}$,
*   $U \in \mathbb{R}^{d_h \times d_h}$, and
*   $V \in \mathbb{R}^{d_{out} \times d_h}$.

---

RNNS AS LMS

$$
e = [E x_{t-3}; E x_{t-2}; E x_{t-1}] \quad \quad e_t = E x_t
$$

$$
h = \sigma(We + b) \quad \quad \quad \quad \quad h_t = g (Uh_{t-1} + W e_t)
$$

$$
z = Uh \quad \quad \quad \quad \quad \quad \quad \hat{y}_t = \text{softmax}(V h_t)
$$

$$
\hat{y} = \text{softmax}(z)
$$

Actual probabilities as dot product:
$$
P (w_{t+1} = i \mid w_1,..., w_t) = y_t[i]
$$
$$
P (w_{1:n}) = \prod_{i=1}^{n} y_i [w_i]
$$

Cross-entropy loss over predicted probabilities:
$$
L_{CE} = - \sum_{w \in V} y_t[w] \log \hat{y}_t[w]
$$
$$
L_{CE} (\hat{y}_t, y_t) = - \log \hat{y}_t [w_{t+1}]
$$

> LM Cross-entropy loss for = probability the model assigns to the **correct next word** (one-hot vector).

---

[IMAGE: Blank page 9]

---

[IMAGE: Cereal bowl icon] PUTTING IT TOGETHER

[IMAGE: Diagram of an unrolled RNN for language modeling. The sequence "So long and thanks for all" is processed word by word. Each word is converted to an Input Embedding (e). This feeds into an RNN block (h). The output of the hidden state $h$ passes through $V h$ (Softmax over Vocabulary) to predict the Next word, and the loss for each prediction (e.g., $-\log y_{long}$) is calculated. The total loss is calculated as the average $\frac{1}{T} \sum_{t=1}^T L_{CE}$.]

---

[IMAGE: Sandwich icon] GENERALIZING RNN ARCHITECTURE TO OTHER TASKS

[IMAGE: Two diagrams showing RNN generalizations.
Left Diagram (Sequence Tagging): An RNN processes the sequence "Janet will back the bill". The output layer performs Argmax/Softmax over tags (NNP, MD, VB, DT, NN) based on the hidden state $h$ via $Vh$.
Right Diagram (Sequence Classification): An RNN processes inputs $x_1$ through $x_n$. The final hidden state $h_n$ is fed into a Feed-Forward Network (FFN) and then a Softmax layer for a single classification output.]

---

[IMAGE: Sandwich icon] LET'S INCREASE THE COMPLEXITY!

*   Like feed-forward layers: **stack** RNNs, use top as input.
*   **Bidirectional** RNNs (every step / last): $h_t = [h_t^f; h_t^b]$.
*   !! Remaining issues:
    *   Two tasks in one:
        *   Carrying info forward.
        *   Representation for current task.
    *   Short context focus.
    *   Vanishing gradients.

---

[IMAGE: Baguette icon] LONG SHORT-TERM MEMORY (LSTM) NETWORK

*   Manage information required in the long term.
*   Adds context vector and masking gates to the representation.

---

[IMAGE: Cookie icon] REMAINING LIMITATIONS

*   Recurrence leads to information loss and training issues.
*   Sequential nature blocks parallelism (as in e.g. CNNs).

---

[IMAGE: Robot icon] TRANSFORMERS

[IMAGE: Photo of several Sesame Street characters (Elmo, Ernie, Big Bird, Cookie Monster, etc.).]

---

[IMAGE: Eyes icon] ATTENTION

$$
\alpha_{i j} = \text{softmax} (\text{score}(x_i, x_j))\forall j \le i
$$

$$
\text{score}(x_i, x_j) = x_i \cdot x_j
$$

$$
= \frac{\exp(\text{score}(x_i, x_j))}{\sum_{k=1}^i \exp(\text{score}(x_i, x_k))} \forall j \le i
$$

$$
y_i = \sum_{j \le i} \alpha_{i j} x_j
$$

[IMAGE: Diagram of a Self-Attention Layer. Inputs $x_1$ to $x_5$ are connected to the self-attention layer. Arrows show that $x_i$ can attend to all preceding and current inputs ($x_j$ where $j \le i$), resulting in outputs $y_1$ to $y_5$.]

---

[IMAGE: Robot and eyes icons] TRANSFORMER ATTENTION

*   **query** ($q_i$): attention focus wrt preceding inputs -- $W^Q$
*   **key** ($k_i$): preceding input being wrt attention focus -- $W^K$
*   **value** ($v_i$): computes the output for attention focus -- $W^V$

> Project $x_i$:
>
> $q_i = W^Q x_i$; $k_i = W^K x_i$; $v_i = W^V x_i$, where
>
> dims: $1 \times d$. Later: $W^Q \in \mathbb{R}^{d \times d}$, $W^K \in \mathbb{R}^{d \times d}$,
>
> and $W^V \in \mathbb{R}^{d \times d}$.

---

[IMAGE: Blank page 18]

---

[IMAGE: Abacus and eyes icons] MATRIX SELF-ATTENTION

$$
\text{score}(x_i, x_j) = q_i \cdot k_j
$$

$$
\downarrow
$$

$$
\text{score}(x_i, x_j) = \frac{q_i \cdot k_j}{\sqrt{d_k}}
$$

$$
y_i = \sum_{j \le i} \alpha_{i j} V_j
$$

Actually: $X \in \mathbb{R}^{N \times d}$ (embeddings),
so: $Q \in \mathbb{R}^{N \times d}$, $K \in \mathbb{R}^{N \times d}$,
and $V \in \mathbb{R}^{N \times d}$, giving:

$$
Q = X W^Q; K = X W^K; V = X W^V
$$

which gives:

$$
\text{SelfAttention} (Q, K, V) = \text{softmax} \left( \frac{Q K^T}{\sqrt{d_k}} \right) V
$$

---

[IMAGE: Landscape painting icon] VISUALLY

[IMAGE: Two diagrams illustrating masked self-attention.
Left Diagram: Shows the mechanism for calculating $y_3$. The input vectors $x_1, x_2, x_3$ generate key, query, and value vectors. Key/Query comparisons feed into a Softmax layer (outputting $\alpha_{i,j}$). The outputs are weighted and summed with the value vectors to produce the Output Vector $y_3$.
Right Diagram: Shows the attention matrix structure where future tokens are masked. The matrix shows $q_i \cdot k_j$ values. Entries where $j > i$ (representing attention to future tokens) are set to $-\infty$. For $N=5$, the upper triangle contains $-\infty$.

Setting weights to -inf to hide future.

---

[IMAGE: Car icon] TRANSFORMER BLOCKS

$$
z = \text{LayerNorm}(x + \text{Self Attn}(x))
$$

$$
y = \text{LayerNorm}(z + \text{FFNN}(z))
$$

$$
\mu = \frac{1}{d_h} \sum_{i=1}^{d_h} x_i
$$

$$
\hat{x} = \frac{(x - \mu)}{\sigma} \quad \quad \text{LayerNorm} = \gamma \hat{x} + \beta
$$

$$
\sigma = \sqrt{\frac{1}{d_h} \sum_{i=1}^{d_h} (x_i - \mu)^2}
$$

---

MORE TRICKS

$$
\text{MultiHeadAttn} (X) = (\text{head}_1 \oplus \text{head}_2 \oplus ... \oplus \text{head}_h) W^O
$$

$$
Q = X W_i^Q; K = X W_i^K; V = X W_i^V
$$

$$
\text{head}_i = \text{SelfAttention} (Q, K, V)
$$

[IMAGE: Two diagrams showing Multihead Attention and Positional Embeddings.
Left Diagram (Multihead Attention): Input $X$ feeds into four separate attention heads (Head 1 to Head 4), each using its own projection matrices ($W^Q, W^K, W^V$). The outputs of the heads are concatenated and projected down to dimension $d$ using $W^O$ to form the output $Y_n$.
Right Diagram (Composite Embeddings): Word Embeddings are added element-wise to Position Embeddings to create Composite Embeddings. These composite embeddings (for words "Janet", "will", "back", "the", "bill") are fed into multiple stacked Transformer Blocks.]

---

[IMAGE: Hammer icon] APPLIED

[IMAGE: Diagram showing a Transformer Block architecture applied to language modeling. The sequence is "So long and thanks for...". Input Embeddings are fed into a Transformer Block. The output of the Transformer Block is passed through a Linear Layer and then Softmax over Vocabulary to predict the Next word. Loss is calculated for each prediction (e.g., $-\log y_{all}$) and averaged, $\frac{1}{T} \sum_{t=1}^T L_{CE}$.]

---

[IMAGE: Sunset icon] DEEP LEARNING LANDSCAPE

[IMAGE: A large collection of various metal tools (wrenches, screwdrivers, pliers, etc.) organized on a white surface.]

---

[IMAGE: Waving hand icon] QUESTIONS

Post on the Discussion board and join class on Thursdays!