# Representation

[Cover Image Placeholder: Image of Pieter Bruegel the Elder's Tower of Babel]

| Property | Value |
| :--- | :--- |
| **Week** | Week 4 |
| **Book Chapters** | Chapters 6, 7 |
| **Slides** | [PDF: week4.pdf] |
| **Recordings** | Empty |
| **Solutions** | Empty |

**Table of Contents**

- [🎽 Warm-up Questions](#warm-up-questions)
- [📺 Lecture Videos](#lecture-videos)
- [🎲 Positive Pointwise Mutual Information](#positive-pointwise-mutual-information)
  - [🚩 Task 1](#task-1)
- [⚡ Forward Propagation in Neural Nets](#forward-propagation-in-neural-nets)
  - [🚩 Task 2](#task-2)

***

## 🎽 Warm-up Questions
---
> ℹ️ These questions are intended to prime your brain for the materials we will be covering in the videos below. They are optional, and there are no expectations wrt the answers.

* Try to find information about the differences and commonalities between Logistic Regression and Neural Networks. How do you think Representation Learning techniques might aid the language tasks we've looked at so far?
* Learning representations from scratch is quite a data-intensive task. Can you reason why?
* Can you think of cases where one would prefer 'classic' word count features with an algorithm like a Decision Tree over Neural Networks that learn their own abstract representations?
* Can you think of (ethical) issues that Representation Learning might cause? Keep in mind that associations between words are learned completely from scratch, and there is little control over input and output this way.

## 📺 Lecture Videos
---
1. **Meaning as Counts**
[Video Placeholder]

2. **Predicting Word Meaning**
[Video Placeholder]

3. **Neural Models of Language**
[Video Placeholder]

## 🎲 Positive Pointwise Mutual Information
---
Remember that we can calculate a PPMI weighting for a given cell in a word co-occurrence matrix using the following equation:

$$\operatorname{PPMI}(w, c)=\max \left(\log_{2} \frac{P(w, c)}{P(w) P(c)}, 0\right)$$

### 🚩 Task 1
---

| $\downarrow w, c \rightarrow$ | aardvark | baboon | cat | dolphin | kākāpō | $\sum$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| air | 0 | 3 | 1 | 6 | 3 | 13 |
| ball | 2 | 0 | 1 | 1 | 1 | 5 |
| breathes | 1 | 0 | 1 | 27 | 2 | 31 |
| claws | 664 | 12 | 1211 | 5 | 305 | 2197 |
| water | 0 | 4 | 1 | 14 | 0 | 19 |
| $\sum$ | 667 | 19 | 1215 | 53 | 311 | 2265 |

* Using the co-occurrence matrix below, calculate PPMI(claws, cat).

<details>
<summary>**Solution**</summary>
$$\operatorname{PPMI}(\text{claws}, \text{cat})=\max \left(\log_{2} \frac{P(w, c)}{P(w) P(c)}, 0\right) = \log_2 \left(\frac{1211 / 2265}{2197/2265 \cdot 1215/2265} \right) = 0.039$$
</details>

## ⚡ Forward Propagation in Neural Nets
---
> ‼️ I unfortunately was a bit inconsistent with intermediate rounding for this exercise. If you're off by some decimal, you did it correctly.

Consider a simple feedforward neural network with a single hidden layer and three output nodes, as depicted in the book:

[Image Placeholder: Diagram of a single hidden layer feedforward neural network for sentence classification.]

Let's go over an example to demonstrate how a few words are 'forwarded' through the network to arrive at a prediction. Here, we can imagine we have the following vocabulary / index mapping $V$:

| **word** | a | dessert | great | it | was |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **index** | 1 | 2 | 3 | 4 | 5 |

Similarly, we have a 2-dimensional (typically this is 300-1000-ish) embedding matrix $E$:

| **word** | a | dessert | great | it | was |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **dim** 1 | 0.087 | 0.698 | 0.474 | 0.698 | 0.122 |
| **dim** 2 | 0.940 | 0.711 | 0.897 | 0.978 | 0.175 |

We encode the words in the input (dessert was great) as one-hot vectors:

| **input** | a | dessert | great | it | was |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $x_0$ | 0 | 1 | 0 | 0 | 0 |
| $x_1$ | 0 | 0 | 0 | 0 | 1 |
| $x_3$ | 0 | 0 | 1 | 0 | 0 |

We then select from our word embedding matrix the columns where $x_n$ is 1, which we concatenate. This gives the following input array which has dimensionality (3*d)*1, where d is the dimensionality of the word embedding, and 3 the number of inputs. In our case 6*1:

$$X = \left[ \begin{array}{l}
0.698 \\
0.711 \\
0.112 \\
0.157 \\
0.474 \\
0.897 \\
\end{array} \right]$$

In the first step (before weights are updated via backpropagation), the first weight matrix $W$ is [initialized randomly](https://en.wikipedia.org/wiki/Weight_(deep_learning)#Initialization). To keep the example minimal, let's assume we have 3 hidden neurons, making $W$ 3*6, as such:

| | $x_0$ dim 1 | $x_0$ dim 2 | $x_1$ dim 1 | $x_1$ dim 2 | $x_2$ dim 1 | $x_2$ dim 2 | **bias** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| h1 | 0.981 | 0.683 | 0.958 | 0.960 | 0.548 | 0.920 | 1 |
| h2 | 0.475 | 0.685 | 0.203 | 0.927 | 0.431 | 0.177 | 1 |
| h3 | 0.019 | 0.517 | 0.765 | 0.068 | 0.862 | 0.118 | 1 |

The last column is not included in the book's example, but remember that every neuron has a bias (same as Logistic Regression, but here it's a column) that is added to the summation of $W \cdot X^T$. Biases are generally initialized to one, rather than randomly. So it's:

$$\quad \\ W \bullet X^T + B_W = 
\left[ \begin{array}{llllll}
0.981 & 0.683 & 0.958 & 0.960 & 0.548 & 0.920  \\ 
0.475 & 0.685 & 0.203 & 0.927 & 0.431 & 0.177  \\ 
0.019 & 0.517 & 0.756 & 0.068 & 0.862 & 0.118  \\ 

\end{array} \right] \bullet \left[ \begin{array}{l}
0.698 \\
0.711 \\
0.112 \\
0.157 \\
0.474 \\
0.897 \\
\end{array} \right] + \left[ \begin{array}{l}
1 \\ 1 \\ 1 \\
\end{array} \right]  = 
\left[ \begin{array}{l}
3.513  \\ 
2.35  \\ 
1.991  \\ 
\end{array} \right] \\ \quad$$

> 💡 Note that $\bullet$ is the dot product (multiply + sum), so one column operation looks like: $0.981 \times 0.698 + 0.683 \times 0.711 + 0.958 \times 0.112 + 0.960 \times 0.157 + 0.548 \times 0.474 + 0.920 \times 0.897 + 1 \approx 3.513$.

This input layer still needs to go through and activation function. We can play easy and forward it through a ReLU (which will retain the current values), but let's try $\tanh$:

$$\sigma(z) = \frac{e^{z}-e^{-z}}{e^{z}+e^{-z}} \quad \sigma\left(\left[ \begin{array}{l}
3.513  \\ 
2.35  \\ 
1.991  \\ 
\end{array} \right]\right) = \left[ \begin{array}{l}
0.998  \\ 
0.982  \\ 
0.963  \\ 
\end{array} \right]$$

> 💡 One column operation looks like: $\frac{\exp(3.513359)-\exp(-3.513359)}{\exp(3.513359) + \exp(-3.513359)} \approx 0.998
$.

Voilà, this gives us $h$, which is indeed 3*1. This forms the input to our randomly initialized matrix $U$. Typically, this one maps more values of $h$ (i.e., more than 3) into the same dimensionality as the ouput. In our example we have one hidden state with 3 neurons, and 3 output nodes, so nothing exciting happens in this step. We have, again, random $U$, and (again, initialized to 1) biases $B_U$:

| | h1 | h2 | h3 | b |
| :--- | :--- | :--- | :--- | :--- |
| $y_0$ | 0.176 | 0.190 | 0.398 | 1 |
| $y_1$ | 0.199 | 0.588 | 0.512 | 1 |
| $y_2$ | 0.244 | 0.921 | 0.819 | 1 |

Then it's the same stuff as before:

$$U \bullet h + B_U = 
\left[ \begin{array}{lll}
0.176 & 0.190 & 0.398 \\
0.199 & 0.588 & 0.512 \\
0.244 & 0.921 & 0.819 \\
\end{array} \right] \bullet \left[ \begin{array}{l}
0.998 \\
0.982 \\
0.936 \\
\end{array} \right] + \left[ \begin{array}{l}
1 \\ 1 \\ 1 \\
\end{array} \right]  = 
\left[ \begin{array}{l}
1.764  \\ 
2.269  \\ 
2.937  \\ 
\end{array} \right] \\ \quad \\$$

All that's left is to run softmax over this output to give us $y$ (3*1):

$$f(z) = \frac{e^z}{\sum^K_{i=1} e^{z_i}}
 \quad f\left(\left[ \begin{array}{l}
0.941  \\ 
0.979  \\ 
0.994  \\ 
\end{array} \right] \right) =  \left[ \begin{array}{l}
       0.323 \\
       0.336 \\
       0.341 \\
\end{array} \right]$$

Note that $f$ here is the softmax.

Finally, if we compute $\arg \max_y$ we get the predicted label: `2`, which in [this example](/29d979eeca9f81cc9c15cbd0304ea327?pvs=25) means we classify $\hat{y}_{\text{neu}}$ (neutral). If we want to calculate the loss for our softmax probabilities, we can use cross-entropy (this is cross-entropy for a single binary prediction):

$$L_{\log}(y, p) = -(y \log (p) + (1 - y) \log (1 - p))$$

We only provide cross-entropy the value $\max_y$ (i.e., 0.341) and the true label (let's assume that's 0, or $\hat{y}_{\text{pos}}$):

$$-(0 \cdot \ln(0.341) + (1 - 0) \cdot \ln(1 - 0.341)) = 0.417$$

> 💡 Backpropagation propagates the error of this prediction back to all the parameters ($W$, $U$, $B_W$, $B_U$, and if we don't freeze the embeddings, also $X$). These multidimensional error planes form a gradient we can determine the next update steps on (this is [SGD](https://en.wikipedia.org/wiki/Gradient_descent#Momentum_or_heavy_ball_method), similar as in Logistic Regression). These updates tune the weights to decrease the error (no guarantee we will always find an optimal set of weights though, see local minima).

### 🚩 Task 2
---
* Consider the following sentence encoding and embedding matrix:

| **word** | a | network | neural |
| :--- | :--- | :--- | :--- |
| **dim** 1 | 0.375 | 0.951 | 0.732 |

| **input** | a | network | neural |
| :--- | :--- | :--- | :--- |
| $x_0$ | 1 | 0 | 0 |
| $x_1$ | 0 | 0 | 1 |
| $x_3$ | 0 | 1 | 0 |

Calculate $X$.

<details>
<summary>**Solution**</summary>
$$X = \left[
\begin{array}{l}
0.375 \\ 0.732 \\ 0.951 \\
\end{array}
\right]$$
</details>

* Given this $X$, and the $W$ matrix of weights, and biases $B$ below, calculate the output of this layer (no activation yet).

$$W = \left[
\begin{array}{lll}
0.599 & 0.156 & 0.156 \\ 
0.058 & 0.866 & 0.601 \\ 
0.708 & 0.021 & 0.970 \\
\end{array}
\right] \quad
B_W = \left[
\begin{array}{l}
1 \\
1 \\
1 \\
\end{array}
\right]$$

<details>
<summary>**Solution**</summary>
$$\left[
\begin{array}{lll}
0.599 & 0.156 & 0.156 \\ 
0.058 & 0.866 & 0.601 \\ 
0.708 & 0.021 & 0.970 \\
\end{array}
\right] \bullet \left[ \begin{array}{l}
0.375 \\ 0.732 \\ 0.951 \\
\end{array} \right] + \left[
\begin{array}{l}
1 \\
1 \\
1 \\
\end{array}
\right]
=  \left[
\begin{array}{l}
1.487 \\
2.227 \\
1.203 \\
\end{array}
\right]$$
</details>

* Apply the sigmoid function to this output.

<details>
<summary>**Solution**</summary>
$$\sigma(z) = \frac{1}{1 + e^{-z}} \quad \sigma\left(
\left[ \begin{array}{l}
1.487 \\
2.227 \\
1.203 \\
\end{array}  \right]
\right) = \left[ \begin{array}{l}
0.816 \\
0.903 \\
0.901 \\
\end{array}  \right]$$
</details>

* There are two output nodes in $U$ (see below). Calculate the output probabilities using softmax.

$$U = \left[
\begin{array}{lll}
0.832 & 0.212 & 0.182 \\ 
0.183 & 0.304 & 0.525 \\ 
\end{array}
\right] \quad
B_U = \left[
\begin{array}{l}
1 \\
1 \\
\end{array}
\right]$$

<details>
<summary>**Solution**</summary>
$$U \bullet h + B_U = \left[
\begin{array}{lll}
0.832 & 0.212 & 0.182 \\ 
0.183 & 0.304 & 0.525 \\ 
\end{array}
\right] \bullet \left[ \begin{array}{l}
0.816 \\
0.903 \\
0.901 \\
\end{array} \right] + \left[
\begin{array}{l}
1 \\
1 \\
\end{array}
\right] = \left[
\begin{array}{l}
2.034 \\
1.897 \\
\end{array}
\right] \quad\quad f\left(\left[ \begin{array}{l}
2.034  \\ 
1.897  \\ 
\end{array} \right] \right) =  \left[ \begin{array}{l}
       0.534 \\
       0.466 \\
\end{array} \right]$$
</details>