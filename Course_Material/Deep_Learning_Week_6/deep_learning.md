# 🤖 Deep Learning

**Page Properties**

| Property | Value |
| :--- | :--- |
| **Week** | Week 6 |
| **Book Chapters** | Chapter 8 |
| **Slides** | [Attachment: week6.pdf] |
| **Recordings** | Empty |
| **Solutions** | Empty |

---

## **Table of Contents**

*   🎽 Warm-up Questions
*   📺 Lecture Videos
*   🔄 Recurrent Neural Network Language Model (RNNLM)
    *   🚩 Task 1

---

## 🎽 Warm-up Questions

---

> ℹ️ These questions are intended to prime your brain for the materials we will be covering in the videos below. They are optional, and there are no expectations wrt the answers.

*   We have looked at sequential models in the previous lecture. The tricky thing about simple probabilistic models is that they don’t capture longer dependencies well (i.e., what is the exact effect that word 2 in a sequence might have on the interpretation of words 8, 9 and 10). Can think of some ways to track / store this information?
*   To augment the simple feed-forward neural networks we have seen before with sequentiality, two things are commonly employed: recurrence (weights connect to each other per step in the sequence), and attention (a fixed set of separate weights is learned over the entire sequence). Briefly investigate the gist of these two mechanism (they are explained in the lecture). Can you think of ways it might improve both classifiers and language models?
*   All these augmentation steps are massive engineering feats, but also increasingly demand computational power. Can you think of draw-backs of these requirements? For example, what if the base costs to do research increase? What if research becomes inaccessible to the public, or even certain groups of researchers, due to such increasing demands?

## 📺 Lecture Videos

---

> 🖼️ Slides are available at the top of the page!

1️⃣ **Recurrent Models**

[Video: Recurrent Models]

2️⃣ **Transformers**

[Video: Transformers]

3️⃣ **Deep Learning Landscape**

[Video: Deep Learning Landscape]

## 🔄 Recurrent Neural Network Language Model (RNNLM)

---

Consider an RNNLM receiving two inputs, predicting up to the third word (note that we try to predict the next word at each step, in the picture below $\text{So long and}$, where the input is just $\text{So long}$):

[Image: RNNLM Diagram]

Let's go over an example to demonstrate how a few words are 'forwarded' through the network to arrive at a prediction **for each timestep** $t$. Here, we can imagine we have the following vocabulary / index mapping $V$:

| word | and | for | long | so | thanks |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **index** | 1 | 2 | 3 | 4 | 5 |

Similarly, we have a 2-dimensional (typically this is 300-1000-ish) embedding matrix $E$:

| **word** | and | for | long | so | thanks |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **dim** 1 | 0.087 | 0.698 | 0.474 | 0.698 | 0.122 |
| **dim** 2 | 0.940 | 0.711 | 0.897 | 0.978 | 0.175 |

We encode the words in the input ($\text{So long}$) as one-hot vectors:

| **input** | and | for | long | so | thanks |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $x_0$​ | 0 | 0 | 0 | 1 | 0 |
| $x_1$​ | 0 | 0 | 1 | 0 | 0 |

The following steps are given by:

$$
\begin{array}{l}
\mathbf{function\ }\mathtt{ForwardRNN}(x, \text{network}): y\\
\quad\quad h_0 \leftarrow 0 \\
     \quad\quad \mathbf{for}\ i \leftarrow \operatorname{to} \mathtt{length}(x) \mathbf{\ do} \\
\quad\quad\quad\quad h_i \leftarrow g(\mathbf{Uh}_{i-1} + \mathbf{Wx}_i) \\
\quad\quad\quad\quad y_i \leftarrow f (\mathbf{Vh}_i) \\
\quad\quad \mathbf{return\ } y \\
\mathbf{end\ function}
\end{array}
$$

Above, $i = t$ (so notice that an input index matches a timestep, timestep is just RNN lingo). For each $t$, we select from our word embedding matrix the columns where $x_n$ is 1. This gives the following input dimensionality (1*d)*1, where d is the dimensionality of the word embedding In our case, for each $t$, we get 2*1:

$$
\mathbf{E}{\mathbf{x}_{t_0}} = \left[ \begin{array}{l}
0.698 \\
0.978 \\
\end{array} \right] \quad
\mathbf{E}{\mathbf{x}_{t_1}} = \left[ \begin{array}{l}
0.474 \\
0.897 \\
\end{array} \right]
$$

In the first step (before weights are updated via backpropagation), the weight matrices $\mathbf{W}$, $\mathbf{U}$, and $\mathbf{V}$ are <span style="text-decoration:underline">initialized randomly</span>. To keep the example minimal, let's assume we have 2 hidden neurons (denoted as $h$). Remember, visually, we are doing:

[Image: RNN Diagram detail (W, U, V connections)]

Then: $\mathbf{W} \in \mathbb{R}^{d_{h} \times d_\text{in}}$, $\mathbf{U} \in \mathbb{R}^{d_{h} \times d_{h}}$, and $\mathbf{V} \in \mathbb{R}^{d_{\text {out }} \times d_{h}}$. Here, $\text{in}$ is our embedding dimensionality, and $\text{out}$ the vocabulary size. Hence, $\mathbf{W}$ is $2 \times 2$, $\mathbf{U}$ $2 \times 2$, and $\mathbf{V}$ $5 \times 2$. So:

$$
\mathbf{W} =
$$
$$
\mathbf{U} =
$$
$$
\mathbf{V} =
$$

> 🚧 In case the above tables don’t show (they don’t for some reason on my end), here’s a copy:
>
> [Image: W, U, V matrices image copy]

> ‼️ We’ll set the biases to 0 for this example, so we don’t have to worry about them, but note that all learned parameters ($\mathbf{W}, \mathbf{U}, \mathbf{V}$) might have added biases. Remember that these parameters are initialized randomly, and do not change per timestep (they can be updated after a full pass through the data, for example).

Now we have all our parts, given the equation:

$$
\begin{equation}
\begin{aligned}
\mathbf{e}_{t} &=\mathbf{E} \mathbf{x}_{t} \\
\mathbf{h}_{t} &=g\left(\mathbf{U h}_{t-1}+\mathbf{W e}_{t}\right) \\
\hat{\mathbf{y}}_{t} &=\operatorname{softmax}\left(\mathbf{V h}_{t}\right)
\end{aligned},
\end{equation}
$$

we can roll out our forwarding steps. At the first timestep, $h_{t-1}$ is initialized to zero:

$$
\textbf{h}_0 = \mathbf{U} \bullet \mathbf{h}_{t-1} + \mathbf{W} \bullet \mathbf{e}_t = 
\left[ \begin{array}{ll}
0.156 & 0.156 \\
0.058 & 0.866 \\
\end{array} \right] \bullet
\left[ \begin{array}{l}
0.000 \\
0.000 \\
\end{array} \right] +
\left[ \begin{array}{ll}
0.375 & 0.951 \\
0.732 & 0.599 \\
\end{array} \right] \bullet \left[ \begin{array}{l}
0.698 \\
0.978 \\
\end{array} \right] = 
\left[ \begin{array}{l}
1.192 \\
1.097 \\
\end{array} \right] 
$$

> 💡 Note that $\bullet$ denotes matrix multiplication (`a @ b`) or the dot product (multiply, sum if uneven), so one column operation looks like: $0.375 \times 0.698 + 0.951 \times 0.978 \approx 1.192$.

We’ll use ReLU activations for $g$:

$$
g(x) = \max(0, x)
$$

If you notice, for this example, we can just take the values, as they will always be positive (🙂):

$$
g\left(\left[ \begin{array}{l}
1.192 \\
1.097 \\
\end{array} \right]\right) = \left[ \begin{array}{l}
1.192 \\
1.097 \\
\end{array} \right]
$$

For our word prediction, we weight and map $\textbf{h}_0$ to the output dimensionality, and pass it through our softmax:

$$
f(z) = \frac{e^z}{\sum^K_{i=1} e^{z_i}}
$$

This gives:

$$
\mathbf{\hat{y}}_t = \operatorname{softmax}(\textbf{V}\textbf{h}_t) = \quad f\left(
\left[ \begin{array}{ll}
0.601 & 0.683  \\
0.021 & 0.970  \\
0.832 & 0.212  \\
0.182 & 0.183  \\
0.304 & 0.525  \\
\end{array} \right] \bullet  \quad \left[ \begin{array}{ll}
1.192 \\
1.097 \\
\end{array} \right] = \left[ \begin{array}{l}
1.466 \\
1.089 \\
1.224 \\
0.418 \\
0.938 \\
\end{array} \right] \right) = 
\left[ \begin{array}{l}
0.293 \\
0.201 \\
0.230 \\
0.103 \\
0.173 \\
\end{array} \right]
$$

Finally, if we compute $\arg \max_y$ we get the predicted word index: 0, which means after $\text{So}$ we predict “and” ($V_0$), which is incorrect. Note that this is just the first pass in which we initialized all our weight matrices randomly. The network will, based on the error, tune the weights. 

Anyway, that was timestep 0! Now we have an actual representation for $\textbf{h}_0$, which we are going to pass into the next step, rather than our zero matrix before. For $\mathbf{e}_t$ we use the embedding for our second word: $\text{long}$.

$$
\textbf{h}_1 = \mathbf{U} \bullet \mathbf{h}_{t-1} + \mathbf{W}\mathbf{e}_t = 
\left[ \begin{array}{ll}
0.156 & 0.156 \\
0.058 & 0.866 \\
\end{array} \right] \bullet
\left[ \begin{array}{l}
1.192 \\
1.097 \\
\end{array} \right] + \\
\left[ \begin{array}{ll}
0.375 & 0.951 \\
0.732 & 0.599 \\
\end{array} \right] \bullet \left[ \begin{array}{l}
0.474 \\
0.897 \\
\end{array} \right] = 
\left[ \begin{array}{l}
1.388 \\
1.903 \\
\end{array} \right] 
$$

Note that $\mathbf{U}$ and $\mathbf{W}$ haven’t updated at this step. Next, we compute the predicted word:

$$
\mathbf{\hat{y}}_t = \operatorname{softmax}(\textbf{V}\textbf{h}_t) = \quad f\left(
\left[ \begin{array}{ll}
0.601 & 0.683  \\
0.021 & 0.970  \\
0.832 & 0.212  \\
0.182 & 0.183  \\
0.304 & 0.525  \\
\end{array} \right] \bullet  \quad \left[ \begin{array}{ll}
1.388 \\
1.903 \\
\end{array} \right] = \left[ \begin{array}{l}
2.134 \\
1.875 \\
1.558 \\
0.601 \\
1.421 \\
\end{array} \right] \right) = 
\left[ \begin{array}{l}
0.329 \\
0.254 \\
0.185 \\
0.071 \\
0.161 \\
\end{array} \right]
$$

Another prediction for “and” ($V_0$), but this time correct! To compute the loss, we sum over all individual losses. For Language Modelling, the loss is based on knowing the next word, so:

$$
\frac{1}{T} \sum_{t=1}^{T} L_{C E} \quad \text{where} : \left(\hat{\mathbf{y}}_{t}, \mathbf{y}_{t}\right)=-\log \hat{\mathbf{y}}_{t}\left[w_{t+1}\right]
$$

Note that $[w_{t+1}]$ refers to the index of the **ACTUAL** next word (meaning ‘long’ at step 0 and ‘and’ at step 1) indexing into the softmax matrix). This gives:

$$
\frac{1}{2} \times (-\ln(0.185) + -\ln(0.329)) = 
1.312
$$

^You can get the above with dot products, but this looks neater on paper. Note that if this is input into $\exp$, we have the perplexity of the model.

### 🚩 Task 1

---

*   Consider the following sentence encoding and embedding matrix:

    | **word** | a | network | neural |
    | :--- | :--- | :--- | :--- |
    | **dim** 1 | 0.432 | 0.291 | 0.732 |

    | **input** | a | network | neural |
    | :--- | :--- | :--- | :--- |
    | $x_0$​ | 1 | 0 | 0 |
    | $x_1$​ | 0 | 0 | 1 |
    | $x_2$​ | 0 | 1 | 0 |

    Calculate $\textbf{E}\textbf{x}_{t_0}$ and $\textbf{E}\textbf{x}_{t_1}$.

    **Solution**

    $$
    \textbf{E}\textbf{x}_{t_0} = \left[
    \begin{array}{l}
    0.432 \\
    \end{array}\right] \quad
    \textbf{E}\textbf{x}_{t_1} = \left[
    \begin{array}{l}
    0.732 \\
    \end{array}
    \right]
    $$

*   Calculate $\mathbf{\hat{y}}_t$ given $\mathbf{V}$ below.

    $$
    \mathbf{V} = \left[
    \begin{array}{ll}
    0.331 & 0.064 \\ 
    0.311 & 0.325 \\
    0.730 & 0.638 \\
    \end{array}
    \right] 
    $$

    **Solution**

    $$
    \mathbf{\hat{y}}_t =  f\left(
    \left[ \begin{array}{ll}
    0.331 & 0.064 \\ 
    0.311 & 0.325 \\
    0.730 & 0.638 \\
    \end{array} \right] \bullet  \quad \left[\begin{array}{l} 0.333 \\ 0.032 \\ \end{array} \right] = \left[ \begin{array}{l}
    0.112 \\
    0.114 \\
    0.264 \\
    \end{array} \right] \right) = 
    \left[ \begin{array}{l}
    0.316 \\
    0.316 \\
    0.368 \\
    \end{array} \right]
    $$

    Here we predict index 3 ($\text{neural}$). Note: given a tie, you can assume $\arg \max$ returns the first element.