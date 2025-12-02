---
# ⛏️ Information Extraction

**Cover Image Placeholder:** A tool that connects everyday work into one space. It gives you and your teams AI tools—search, writing, note-taking—inside an all-in-one, flexible workspace.

| Property | Value |
| :--- | :--- |
| Week | Week 5 |
| Book Chapters | Chapters 17 and 20 (20.1 mostly) |
| Slides | [week5.pdf](files/week5.pdf) |
| Recordings | Empty |
| Solutions | Empty |

## Table of Contents

*   [🎽 Warm-up Questions](#warm-up-questions)
*   [📺 Lecture Videos](#lecture-videos)
*   [🎲 Hidden Markov Models](#hidden-markov-models)
    *   [🚩 Task 1](#task-1)

---

## 🎽 Warm-up Questions

***

> ℹ️ These questions are intended to prime your brain for the materials we will be covering in the videos below. They are optional, and there are no expectations wrt the answers.

*   You've made it to week 5 **and** are still looking at the warm-up exercises. Quite a feat! So, we have gone over the basic building blocks of NLP: gathering and processing language input, how to model occurrences and co-occurences, fitting linear and probabilistic models, and representing inputs in richer ways. Although, we still haven't looked at language as a sequence! Can you think of concrete examples from tasks where sequentially might be relevant? Meaning: where seeing words in a particular order might be important?
*   Sequentiality can be relevant for many word-level classification tasks, but also document, or sentence-level. Whenever we have some sort of dependence, we can't pretend we can cram everything into a vector. Consider the task of determining if a word is a verb. What types of words can be found around verbs? Why would we need this information to determine if something is a verb? You can take the word 'bank' as an example.
*   Text data can be home to a plethora of interesting information about real-world events. Can you name a few examples of such information, and think of how would you approach automatic extraction of such events? What about the entities that are involved in the event? Mapping relations, maybe?

---

## 📺 Lecture Videos

***

> 🖼️ Slides are available at the top of the page!

### 1️⃣ **NLP for Data Science**

[Video Placeholder: NLP for Data Science]

### 2️⃣ **Sequence Classification**

[Video Placeholder: Sequence Classification]

### 3️⃣ **Information Extraction**

[Video Placeholder: Information Extraction]

---

## 🎲 Hidden Markov Models

***

Remember that a Hidden Markov Model (HMM) jointly models the *observed* (words) and *hidden* (labels) events. Formally: $P(o_i\ |\ q_i)$, i.e. $o_i$ depends only on the state that produced the observation $q_i$. Similar to other probability-based models, we can construct our HMM using MLE. We have our transition probabilities *A* (for hidden events):

$$
P\left(t_{i} \mid t_{i-1}\right)=\frac{C\left(t_{i-1}, t_{i}\right)}{C\left(t_{i-1}\right)},
$$

and emission probabilities *B* (for observed events):

$$
P\left(w_{i} \mid t_{i}\right)=\frac{C\left(t_{i}, w_{i}\right)}{C\left(t_{i}\right)}.
$$

To tag a sequence with our $\operatorname{HMM}\ \lambda = (A, B)$, where the sequence is $O = o_1 , o_2 , \ldots, o_T$ the model tries to find the most probable sequence of states $Q = q_1\ q_2\ q_3\ \ldots\ q_T$. This gives $\hat{t}$:

$$
\hat{t}_{1: n}=\underset{t_{1} \ldots t_{n}}{\operatorname{argmax}}\ P\left(t_{1} \ldots t_{n} \mid w_{1} \ldots w_{n}\right) \approx \\ \underset{t_{1} \ldots t_{n}}{\operatorname{argmax}} \prod_{i=1}^{n} \overbrace{P\left(w_{i} \mid t_{i}\right)}^{\text {emission}}\ \overbrace{P\left(t_{i} \mid t_{i-1}\right)}^{transition}.
$$

Alternatively, we can decode using the Viterbi algorithm (see slides for the vectorized version):

**Input**
*   Observation space $O = \{o_1, o_2, \ldots, o_n\}$.
*   State space $S= \{s_1, s_2, \ldots, s_K\}$.
*   Initial probabilities $\Pi = \left(\pi_1, \pi_2, \ldots, \pi_K\right)$ such that $ \pi_i$ stores the probability that $x_1 = s_i$.
*   Observations $Y = (y_1, y_2, \ldots, y_t)$ such that $y_t = o_i$ if observation at time $t$ is $o_i$.
*   Transitions $A\ (K \times K)$ such that $A_{i,j}$ stores the transition probabilities from state $s_i$ to $s_j$.
*   Emissions $B\ (K \times N)$ such that $B_{i,j}$ stores the probability of observing $o_j$ from state $s_i$.

> 💡 $T_n$ here denotes the [trellis](https://en.wikipedia.org/wiki/Trellis_(graph))es we are filling (one for probabilities, one for backpointers), and $k$ the prior state. Note that because we fill $T_1$ in the first loop with $\pi$ (for NLP, usually this is P(something given `<s`>)) there is always a prior state. In the second loop block it’s implicitly set to the result of $\arg\max$.

$$
\begin{array}{l}
\mathbf{function\ }\mathtt{VITERBI}(O,S,\Pi,Y,A,B): X\\
     \quad\quad \mathbf{for} \operatorname{each\ state\ } i=1,2,\ldots,K \mathbf{\ do} \\
\quad\quad\quad\quad T_1[i,1]\leftarrow\pi_i\cdot B_{iy_1} \\
\quad\quad\quad\quad T_2[i,1]\leftarrow  0 \\
\quad\quad \mathbf{end\ for} \\
\quad\quad \mathbf{for} \operatorname{each\ observation\ } j=2,3,\ldots,T \mathbf{\ do} \\
\quad\quad\quad\quad \mathbf{for} \operatorname{each\ observation\ } i=1,2,\ldots,K \mathbf{\ do} \\
\quad\quad\quad\quad\quad\quad T_1[i,j] \gets \max_{k}{(T_1[k,j-1]\cdot A_{ki} \cdot B_{iy_j})} \\
\quad\quad\quad\quad\quad\quad T_2[i,j] \gets \arg\max_{k}{(T_1[k,j-1]\cdot A_{ki} \cdot B_{iy_j}) } \\
\quad\quad\quad\quad \mathbf{end\ for} \\
\quad\quad \mathbf{end\ for} \\
\quad\quad \mathbf{for}\ j=T,T-1,\ldots,2 \mathbf{\ do} \\
\quad\quad\quad\quad z_{j-1}\leftarrow T_2[z_j,j] \\
\quad\quad\quad\quad  x_{j-1}\leftarrow s_{z_{j-1}} \\
\quad\quad \mathbf{end\ for} \\
\quad\quad \mathbf{return\ } X \\
\mathbf{end\ function}
\end{array}
$$

### 🚩 Task 1

***

*   Given the following sentence / label pairs, create first-order (2-gram) transition probabilities for $A$ and $B$. Don’t forget `<s`> and `</s`> tokens.

| $s_1$ | i | like | apple | pie |
| :--- | :--- | :--- | :--- | :--- |
| $y_1$ | PRON | VERB | NOUN | NOUN |
| $s_2$ | do | you | like | pie |
| $y_2$ | AUX | PRON | VERB | NOUN |
| $s_3$ | apple | like | apple | pie |
| $y_3$ | NOUN | ADP | NOUN | NOUN |

### **Solution (Full)**

$A$ — given $C\left(t_{i-1}, t_{i}\right)$:

| label ➡️ / ⬇️ previous label | `<s`> | ADP | AUX | NOUN | PRON | VERB | `</s`> | $\sum$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `<s`> | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 3 |
| ADP | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| AUX | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| NOUN | 0 | 1 | 0 | 2 | 0 | 0 | 3 | 6 |
| PRON | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 2 |
| VERB | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| `</s`> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

> ℹ️ Note that the last column sums to nr of tags + nr of sentences (because of `<s`>).

$A$ probabilities (by applying $P\left(t_{i} \mid t_{i-1}\right)=\frac{C\left(t_{i-1}, t_{i}\right)}{C\left(t_{i-1}\right)}$):

| label ➡️ / ⬇️ previous label | `<s`> | ADP | AUX | NOUN | PRON | VERB | `</s`> |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `<s`> | 0 | 0 | 0.333 | 0.333 | 0.333 | 0 | 0 |
| ADP | 0 | 0 | 0 | 1.000 | 0 | 0 | 0 |
| AUX | 0 | 0 | 0 | 0 | 1.000 | 0 | 0 |
| NOUN | 0 | 0.167 | 0 | 0.333 | 0 | 0 | 0.500 |
| PRON | 0 | 0 | 0 | 0 | 0 | 1.000 | 0 |
| VERB | 0 | 1.000 | 0 | 0 | 0 | 0 | 0 |
| `</s`> | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

$B$ counts — given $C\left(t_{i}, w_{i}\right)$:

| label ➡️ / ⬇️ word | `<s`> | ADP | AUX | NOUN | PRON | VERB | `</s`> |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `<s`> | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| apple | 0 | 0 | 0 | 3 | 0 | 0 | 0 |
| do | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| i | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| like | 0 | 1 | 0 | 0 | 0 | 2 | 0 |
| pie | 0 | 0 | 0 | 3 | 0 | 0 | 0 |
| you | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| `</s`> | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| $\sum$ | 3 | 1 | 1 | 6 | 2 | 2 | 3 |

> ℹ️ Note that the order is counterintuitive: we want to know, given HIDDEN STATE, what might produce an OBSERVED state, rather than the other way around.

Then for $B$ probabilities we get — given $\frac{C\left(t_{i}, w_{i}\right)}{C\left(t_{i}\right)}$:

| label ➡️ / ⬇️ word | `<s`> | ADP | AUX | NOUN | PRON | VERB | `</s`> |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `<s`> | 1.000 | 0 | 0 | 0 | 0 | 0 | 0 |
| apple | 0 | 0 | 0 | 0.500 | 0 | 0 | 0 |
| do | 0 | 0 | 1.000 | 0 | 0 | 0 | 0 |
| i | 0 | 0 | 0 | 0 | 0.500 | 0 | 0 |
| like | 0 | 1.000 | 0 | 0 | 0 | 1.000 | 0 |
| pie | 0 | 0 | 0 | 0.5000 | 0 | 0 | 0 |
| you | 0 | 0 | 0 | 0 | 0.500 | 0 | 0 |
| `</s`> | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 |

We get:

| input $S$ ➡️ / ⬇️ predictions $X$ | you | do | like | apple |
| :--- | :--- | :--- | :--- | :--- |
| ADP | P(ADP\|<s\>) * P(you\|ADP) = 0.000 * 0.000 = 0.000 | P(ADP\|max t-1 = PRON) * P(do\|ADP) = 0.000 * 0.000 = 0.000 | | |
| AUX | P(AUX\|<s\>) * P(you\|AUX) = 0.333 * 0.000 = 0.000 | P(AUX\|max t-1 = PRON) * P(do\|AUX) = 0.000 * 1.000 = 0.000 | | |
| NOUN | P(NOUN\|<s\>) * P(you\|NOUN) = 0.333 * 0.000 = 0.000 | P(NOUN\|max t-1 = PRON) * P(do\|NOUN) = 0.000 * 0.000 = 0.000 | | |
| PRON | P(PRON\|<s\>) * P(you\|PRON) = 0.333 * 0.500 = 0.167 | P(PRON\|max t-1 = PRON) * P(do\|PRON) = 0.000 * 0.000 = 0.000 | | |
| VERB | P(VERB\|<s\>) * P(you\|VERB) = 0.000 * 0.000 = 0.000 | P(VERB\|max t-1 = PRON) * P(do\|VERB) = 0.000 * 0.000 = 0.000 | | |

> ℹ️ If we had more observations, or applied smoothing (🙂) some of these probabilities in the second column would not have been zero.

---