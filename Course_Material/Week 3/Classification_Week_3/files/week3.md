# LANGUAGE & AI: CLASSIFICATION

[IMAGE: Bookshelves and a ladder in a library setting]

Chris Emmery
Department of Cognitive Science & AI
Tilburg University

@cmry • @_cmry • @cmry • cmry.github.io

---

# PROBABILITY-BASED REPRESENTATIONS OF LANGUAGE

[IMAGE: Aerial view of a port with stacked shipping containers and gantry cranes]

---

# n-GRAM LANGUAGE MODEL

*   How likely is word $w$ given context $c$? How likely is $c$?
*   We call these models Language Models (LMs).
*   Useful for *classification*, *spelling correction*, *machine translation* (NLP in general, really).
*   $n$ can be any value: bigram, trigrams, etc.

---

# SEQUENCE PROBABILITIES

$$
P(\text{bike} | \text{I lost my balance and fell off my}) = \frac{C(\text{I lost my balance and fell off my bike})}{C(\text{I lost my balance and fell off my})}
$$

$$
P(w_{1:n}) = P(w_1) P(w_2 | w_1) P(w_3 | w_{1:2}) \dots P(w_n | w_{1:n-1})
$$

$$
= \prod_{k=1}^n P(w_k | w_{1:k-1})
$$

> Can we estimate this under noisy language constraints? We can **approximate** given limited context.

---

# MLE UNDER THE MARKOV ASSUMPTION

$$
P(w_n|w_{1:n-1}) \approx P(w_n|w_{n-1}), \text{ i.e. } P(\text{bike}|\text{my})
$$

$$
P(w_n|w_{n-1}) = \frac{C(w_{n-1} w_n)}{C(w_{n-1})}, \text{ where } C(xy) \text{ is the bigram count.}
$$

> MLE = Maximum Likelihood Estimation

---

# EXAMPLE

```
<s> I am Sam </s>
<s> Sam I am </s>
<s> I do not like green eggs and ham </s>
```

$$
P(w_n|w_{n-1}) = \frac{C(w_{n-1} w_n)}{C(w_{n-1})}
$$

$$
P(\text{I} | \text{<s>}) = \frac{2}{3} = .67 \quad P(\text{Sam}|\text{<s>}) = \frac{1}{3} = .33 \quad P(\text{am}|\text{I}) = \frac{2}{3}
$$

$$
P(\text{<s>}|\text{Sam}) = \frac{1}{2} = .50 \quad P(\text{Sam}|\text{am}) = \frac{1}{2} = .50 \quad P(\text{do}|\text{I}) = \frac{1}{3}
$$

---

# LARGER (REAL) EXAMPLE

[IMAGE: Abacus icon]

---

# SYSTEM EVALUATION

[IMAGE: Checkmark icon]

We aren't classifying or ranking anything... so how do we evaluate probability models?

$$
PP(W) = \sqrt[N]{\prod_{i=1}^N \frac{1}{P(w_i | w_{i-1})}}
$$

$$
\exp(\log p_1 + \log p_2 + \log p_3 + \log p_4)
$$

---

# !! SEVERAL ISSUES

*   Test set has unseen words (Out of Vocabulary, OOV).
    *   Fix by introducing `<UNK>` in the training set (pre-defined vocabulary or frequency-based).
*   Test set has unseen sequences ($P = 0$).
    *   Smooth our probabilities (many options).

---

# SMOOTHING (LAPLACE)

$$
P_{\text{Laplace}} (w_n|w_{n-1}) = \frac{C(w_{n-1} w_n)+1}{C(w_{n-1})+V}
$$

| | i | want | to | eat | chinese | food | lunch | spend |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Total** | 2533 | 927 | 2417 | 746 | 158 | 1093 | 341 | 278 |

**Figure 3.6 Add-one smoothed bigram counts** for eight of the words (out of V = 1446) in the Berkeley Restaurant Project corpus of 9332 sentences. Previously-zero counts are in gray.

| | i | want | to | eat | chinese | food | lunch | spend |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **i** | 6 | 828 | 1 | 10 | 1 | 1 | 1 | 3 |
| **want** | 3 | 1 | 609 | 2 | 7 | 7 | 6 | 2 |
| **to** | 3 | 1 | 5 | 687 | 3 | 1 | 7 | 212 |
| **eat** | 1 | 1 | 3 | 1 | 17 | 3 | 43 | 1 |
| **chinese** | 2 | 1 | 1 | 1 | 1 | 83 | 2 | 1 |
| **food** | 16 | 1 | 16 | 1 | 2 | 5 | 1 | 1 |
| **lunch** | 3 | 1 | 1 | 1 | 1 | 2 | 1 | 1 |
| **spend** | 2 | 1 | 2 | 1 | 1 | 1 | 1 | 1 |

**Figure 3.7 Add-one smoothed bigram probabilities** for eight of the words (out of V = 1446) corpus of 9332 sentences. Previously-zero probabilities are in gray.

| | i | want | to | eat | chinese | food | lunch | spend |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **i** | 0.0015 | 0.21 | 0.00025 | 0.0025 | 0.00025 | 0.00025 | 0.00025 | 0.00075 |
| **want** | 0.0013 | 0.00042 | 0.26 | 0.00084 | 0.0029 | 0.0029 | 0.0025 | 0.00084 |
| **to** | 0.00078 | 0.00026 | 0.0013 | 0.18 | 0.00078 | 0.00026 | 0.0018 | 0.055 |
| **eat** | 0.00046 | 0.00046 | 0.0014 | 0.00046 | 0.0078 | 0.0014 | 0.02 | 0.00046 |
| **chinese** | 0.0012 | 0.00062 | 0.00062 | 0.00062 | 0.00062 | 0.052 | 0.0012 | 0.00062 |
| **food** | 0.0063 | 0.00039 | 0.0063 | 0.00039 | 0.00079 | 0.002 | 0.00039 | 0.00039 |
| **lunch** | 0.0017 | 0.00056 | 0.00056 | 0.00056 | 0.00056 | 0.0011 | 0.00056 | 0.00056 |
| **spend** | 0.0012 | 0.00058 | 0.0012 | 0.00058 | 0.00058 | 0.00058 | 0.00058 | 0.00058 |

---

# OTHER OPTIONS

*   **Add-$k$**: add fractional counts, infer $k$ from data.
*   **Backoff**: get probabilities from first lower order gram.
*   **Interpolation**: get from all the grams, weigh them using $\lambda$ (higher order typically receives more weight).
*   **Kneser-Ney**: weigh grams by the amount of contexts they have appeared in.

> See book for more detail.

---

# GENERATIVE MODEL

[IMAGE: Figure 3.3. A visualization of the sampling distribution, showing blocks for word frequencies (the, of, a, to, in, however, polyphonic) and a number line illustrating cumulative probabilities from 0 to 1.]

**Figure 3.3** A visualization of the sampling distribution for sampling sentences by repeatedly sampling unigrams. The blue bar represents the frequency of each word. The number line shows the cumulative probabilities. If we choose a random number between 0 and 1, it will fall in an interval corresponding to some word. The expectation for the random number to fall in the larger intervals of one of the frequent words (*the, of, a*) is much higher than in the smaller interval of one of the rare words (*polyphonic*).

---

# NAIVE BAYES

We assume all features contribute independently to the classification, regardless of correlations between the features.

$$
x = \langle\text{great, movie}\rangle
$$

$$
P(\text{POS} | x_1, x_2) = P(x_1 | \text{POS}) P(x_2 | \text{POS}) P(\text{POS})
$$

$$
P(\text{NEG} | x_1, x_2) = P(x_1 | \text{NEG}) P(x_2 | \text{NEG}) P(\text{NEG})
$$

$$
\hat{y} = \arg \max_y P(y) \prod_{j=1}^J P(x_j | y)
$$

---

# ? HOW-TO NAIVE BAYES

| $x_1 = \text{bad}$ | $x_2 = \text{great}$ | $x_3 = \text{movie}$ | $y$ |
| :---: | :---: | :---: | :---: |
| 0 | 1 | 1 | 👍 |
| 0 | 1 | 1 | 👍 |
| 0 | 0 | 1 | 👍 |
| 1 | 0 | 1 | 👎 |
| 1 | 0 | 0 | 👎 |

$$
P(y | x_1, x_2, x_3) = P(x_1 | y) P(x_2 | y) P(x_3 | y) P(y)
$$

| | $y=\text{👍}$ | $y=\text{👎}$ |
| :--- | :--- | :--- |
| $P(\text{bad})$ | $0/3$ | $2/2$ |
| $P(\text{great})$ | $2/3$ | $0/2$ |
| $P(\text{movie})$ | $3/3$ | $1/2$ |
| $P(y)$ | $3/5$ | $2/5$ |

$$
D = \text{"bad movie"}
$$

$$
P(\text{👍} | D) = P(x_1) \ast P(x_3) \ast P(\text{👍}) = 0/3 \ast 3/3 \ast 3/5 = 0.00
$$
$$
P(\text{👎} | D) = P(x_1) \ast P(x_3) \ast P(\text{👎}) = 2/2 \ast 1/2 \ast 2/5 = 0.20
$$

---

# RULE-BASED DATA SPLITS: DECISION TREES (ID3)

[IMAGE: Several crows perched on a street lamp against a teal sky.]

---

# CLASSIFICATION RULES

| data | language | learning | mining | text | vision | $y$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0 | 1 | 0 | 0 | 1 | CV |
| 1 | 1 | 1 | 0 | 1 | 0 | NLP |
| 1 | 0 | 1 | 1 | 1 | 0 | TM |

```python
if 'vision' in d:
    label = 'CV'
else:
    if 'language' in d:
        label = 'NLP'
    else:
        label = 'TM'
```

---

# INFERRING RULES (DECISIONS) BY INFORMATION GAIN

| free | money | account | transfer | $y$ |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 1 | 0 | 0 | spam |
| 0 | 1 | 1 | 1 | ham |
| 1 | 0 | 1 | 0 | ham |
| 1 | 0 | 1 | 1 | spam |
| 0 | 1 | 0 | 1 | spam |
| 1 | 0 | 0 | 1 | spam |
| 0 | 0 | 1 | 0 | ham |
| 0 | 1 | 1 | 0 | ham |
| 0 | 1 | 0 | 0 | spam |

$$
E(C) = - \sum_{i=1}^n P(c_i) \log_2 P(c_i)
$$

$$
E(\text{ham}) = - \left( \frac{4}{9} \cdot \log_2 \frac{4}{9} \right)
$$

$$
E(\text{spam}) = - \left( \frac{5}{9} \cdot \log_2 \frac{5}{9} \right)
$$

$$
E(Y) = E(\text{ham}) + E(\text{spam})
$$

$$
E(Y) = -(-0.52 + -0.47)
$$

$$
= 0.991
$$

$E = \text{Entropy}, P = \text{proportion / probability}, C = \text{class}, IG = \text{Information Gain}$

---

# INFERRING RULES (DECISIONS) BY INFORMATION GAIN

| | spam | ham | total |
| :---: | :---: | :---: | :---: |
| **free** 0 | 2 | 3 | 5 |
| **free** 1 | 3 | 1 | 4 |

$$
E(\text{free, } Y) = \sum_{c \in Y} P(c) E(c)
$$

$$
E(\text{free}=0, Y) = - (P(\text{free}=0, \text{spam}) \cdot \log_2 P(\text{free}=0, \text{spam})) - (P(\text{free}=0, \text{ham}) \cdot \log_2 P(\text{free}=0, \text{ham}))
$$

$$
E(\text{free}=1, Y) = - (P(\text{free}=1, \text{spam}) \cdot \log_2 P(\text{free}=1, \text{spam})) - (P(\text{free}=1, \text{ham}) \cdot \log_2 P(\text{free}=1, \text{ham}))
$$

$$
E(\text{free, } Y) = P(\text{free}=0) \cdot E(\text{free}=0, Y) + P(\text{free}=1) \cdot E(\text{free}=1, Y)
$$

---
*(Page 19 is blank)*
---

# INFERRING RULES (DECISIONS) BY INFORMATION GAIN

| | spam | ham | total |
| :---: | :---: | :---: | :---: |
| **free** 0 | 2 | 3 | 5 |
| **free** 1 | 3 | 1 | 4 |

$$
E(\text{free, } Y) = \sum_{c \in Y} P(c) E(c)
$$

$$
E(\text{free}=0, Y) = - \left( \frac{2}{5} \cdot \log_2 \frac{2}{5} \right) - \left( \frac{3}{5} \cdot \log_2 \frac{3}{5} \right)
$$

$$
E(\text{free}=1, Y) = - \left( \frac{3}{4} \cdot \log_2 \frac{3}{4} \right) - \left( \frac{1}{4} \cdot \log_2 \frac{1}{4} \right)
$$

$$
E(\text{free, } Y) = \frac{5}{9} \cdot 0.971 + \frac{4}{9} \cdot 0.811 = 0.762
$$

*(Self-correction: The OCR for the intermediate calculation on E(free=0, Y) and E(free=1, Y) seems to have swapped the fractions compared to the final result line, but I will transcribe the final line as written on the slide.)*

$$
E(\text{free, } Y) = \frac{4}{9} \cdot 0.811 + \frac{5}{9} \cdot 0.722 = 0.762
$$

$$
IG = E(Y) - E(\text{free, } Y) = 0.991 - 0.762 = 0.229
$$

---

# ID3 ALGORITHM

[IMAGE: Pine tree icon]

*   Feature (word) with highest Information Gain will be split on first.
*   Instances divided over both sides calculations will be repeated on leftovers (**recursion**).
*   If all leftover instances belong to one class, make decision.
*   Create more and more rules until some stopping criterion.

> More info: here and here.

---

# LINEAR DECISIONS WITH REGRESSION

[IMAGE: Ruler icon]

[IMAGE: Image of a stone wall (Hadrian's Wall) running over rolling brown hills.]

---

# DECISIONS AS A FUNCTION

[IMAGE: Line chart icon]

$$
f(X) = a \cdot x + b \text{ or } Y = \beta_0 + \beta_1 \cdot X
$$

---

# EXAMPLE

[IMAGE: Beer mug icon]

| city | students (X) | alcohol (Y) | $(X - \bar{X})^2$ | $(X - \bar{X}) \cdot (Y - \bar{Y})$ |
| :--- | :---: | :---: | :---: | :---: |
| Tilburg | 26 | 41 | $(26 - 18)^2 = 64$ | $(26 - 18) \ast (41 - 29) = 96$ |
| Eindhoven | 21 | 37 | $(21 - 18)^2 = 9$ | $(21 - 18) \ast (37 - 29) = 24$ |
| Wageningen | 6 | 9 | $(06 - 18)^2 = 144$ | $(06 - 18) \ast (09 - 29) = 240$ |
| $\sum$ | 53 | 87 | 217 | 360 |

*   $\bar{X} = 53/3 \approx 18, \bar{Y} = 87/3 = 29$
*   $\beta_1 = \frac{\sum_{i=1}^n (x_i-\bar{x})(y_i-\bar{y})}{\sum_{i=1}^n (x_i-\bar{x})^2} = \frac{360}{217} \approx 1.66$
*   $\beta_0 = \bar{Y} - \beta_1 \cdot \bar{X} = 29 - 1.66 \cdot 18 = -0.88$
*   $\hat{y} = \beta_0 + \beta_1 \cdot X = -0.88 + 1.66 \cdot X$

> Sources: RIVM, infogram (2016, 2020)

---

# RESULT

[IMAGE: Toasting beer mugs icon]

---

# EVALUATION

[IMAGE: Toasting champagne glasses icon]

$$
\text{RMSE} = \sqrt{\frac{\sum_{i=1}^N (\hat{y}_i-y_i)^2}{N}}
$$

$$
= \sqrt{\frac{(42.28-41)^2+(33.98-37)^2+(9.08-9)^2}{3}} = 1.89
$$

$$
R^2 = 1 - \frac{\sum_{i=1}^N (y_i-\hat{y}_i)^2}{\sum_{i=1}^N (y_i-\bar{y})^2} = 1 - \frac{10.7652}{608} = .982
$$

---

# LOGISTIC REGRESSION

[IMAGE: Scale/balance icon]

$$
P(y = 1) = \sigma(z) = \frac{1}{1+e^{-(\beta_0+\beta_1 \cdot x)}}
$$

$$
P(y = 0) = 1 - \sigma(z) = 1 - \frac{1}{1+e^{-(\beta_0+\beta_1 \cdot x)}}
$$

---

# LOGISTIC REGRESSION FIT EXPLAINED

[IMAGE: Monitor icon]

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cross_entropy_loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

def fit(X, y, lr=0.01, num_iter=1000):
    theta = np.zeros(X.shape[1])
    for i in range(num_iter):
        z = np.dot(X, theta) # theta = coefficient
        h = sigmoid(z)        # also ŷ
        gradient = np.dot(X.T, (h - y)) / y.size
        theta -= lr * gradient
    print("loss:", cross_entropy_loss(np.dot(X, theta), sigmoid(z)))

Adapted from Martín Pellarolo.
```

---

# USE CASE

[IMAGE: Wrench icon]

```python
from sklearn.datasets import fetch_20newsgroups

tnews = fetch_20newsgroups(categories=['sci.space', 'alt.atheism'])
X, y = tnews['data'], tnews['target']
```

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
```

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(min_df=3)
X_train = tfidf.fit_transform(X_train)
```

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(n_jobs=-1)
lr.fit(X_train, y_train)
```

---

```python
from sklearn.metrics import accuracy_score

print(accuracy_score(lr.predict(X_train), y_train),
      accuracy_score(lr.predict(tfidf.transform(X_test)), y_test))
```

```
0.9950248756218906 0.966542750929368
```

```python
sorted(zip(tfidf.vocabulary_, lr.coef_[0]), key=lambda x: x[1], reverse=True)
```

```
('access', 1.1874025309892315),
('doppelganger', 1.0622629839976143),
('solar', 1.0527179075064501),
('wrap', 1.052644230973313),
('penalty', 1.0354513756064758),
('facility', 0.9258179949460602),
('conventional', 0.8998670263151941),
('velocity', 0.8760137340712848),
('c5ky9y', 0.8402055498256688)
```

---

```python
from lime import lime_text
from sklearn.pipeline import make_pipeline

c = make_pipeline(tfidf, lr)

from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=['alt.atheism', 'sci.space'])

exp = explainer.explain_instance(X[86], c.predict_proba, num_features=5)
exp.show_in_notebook(text=True)
```

---

# WHAT ARE SUPPORT VECTOR MACHINES?

[IMAGE: Pensive face emoji]

Mathematically complex, but two simple intuitions:

*   We want our decision boundaries to have an optimal distance from certain groups of classes $\rightarrow$ use support vectors.
*   We want to be able to classify non-linear relations $\rightarrow$ use kernel trick.

---

# SUPPORT VECTORS

[IMAGE: Handshake/fist bump emoji]

---

# KERNEL TRICK

[IMAGE: Magic wand icon]

[IMAGE: Two scatter plots illustrating the kernel trick. The left plot shows data in R^3 (separable with a hyperplane). The right plot shows the same data projected onto R^2, now separated by a circle (the hyperplane projection).]

---

# DISTANCE (AND MEMORY)-BASED DECISIONS: $k$-NN

[IMAGE: Brain icon]

[IMAGE: Abstract geometric pattern (like a cracked map) with two small white cat faces.]

---

# GENERAL ALGORITHM

[IMAGE: Robot icon]

*   Store the complete training matrix $X_{\text{train}}$ in memory.
*   Calculate distance metric between a given $\vec{x}_{\text{test}}$ and all $\vec{x}_{\text{train}} \in X_{\text{train}}$.
*   Choose the $k$ vectors from $X_{\text{train}}$ with the highest similarity to $\vec{x}_{\text{test}}$.
*   Look up the labels for these $k$ vectors, take majority label $\rightarrow$ this is the classification.

---

# DISTANCES

[IMAGE: Ruler icon]

*   Euclidean Distance: $\sqrt{\sum_{i=1}^n (x_i - y_i)^2}$
*   Manhattan Distance: $\sum_{i=1}^n |x_i - y_i|$
*   Minkowski Distance: $(\sum_{i=1}^n |x_i - y_i|^p)^{\frac{1}{p}}$
*   Cosine Distance: $1 - \frac{\vec{p} \cdot \vec{q}}{\sqrt{\vec{p} \cdot \vec{p}} \cdot \sqrt{\vec{q} \cdot \vec{q}}}$

> Note similarity / distance 'difference'.

---

# AUGMENTATIONS TO $k$-NN

[IMAGE: Thermometer icon]

*   Weight labels by:
    *   Frequency (majority preferred).
    *   Inverse frequency (rarer preferred).
    *   Distance (closer instances more important).

---

# $k$-NEAREST NEIGHBORS

[IMAGE: Houses/home icon]

---
*(Page 40 is blank)*
---

# GENERALIZATION

[IMAGE: Lizard/gecko icon]

[IMAGE: Close-up photo of a green lizard/chameleon perched on a bright green leaf.]

---

# SURFACE-LEVEL

[IMAGE: Bar chart icon]

---

# THEORETICAL

[IMAGE: Target/bullseye icon]

---

# ALGORITHMS

[IMAGE: Grid chart icon]

[IMAGE: Three adjacent line plots (35x35 grid) showing scatter plots and blue fitted lines, likely representing underfitting, optimal fit, and overfitting for regression models.]

---

# ! WHAT CAN CAUSE THESE ERRORS?

*   Model hyper-parameters.
    *   $k$ for $k$-NN, depth and pruning of decision trees, polynomial features for regression.
    *   Can be solved by proper tuning and validation
*   Data imbalance.
    *   Large class differences, or incorrectly split.
    *   Can be solved by stratification or adjusting the sample.

---

# HYPER-PARAMETERS & TUNING

[IMAGE: Screwdriver icon]

*   Split a validation set.

[IMAGE: Two diagrams showing parameter search space. The left diagram (Grid Layout) shows points evenly spaced in a 3x3 grid. The right diagram (Random Layout) shows randomly scattered points. Both diagrams have 'Unimportant parameter' on the Y-axis and 'Important parameter' on the X-axis, illustrating the benefit of random search when parameter importance is unknown.]

---

# SCHEMES FOR USING METRICS IN MODEL SELECTION

[IMAGE: School bag icon]

*   Hold-Out.
*   $k$-Fold Cross-validation.
*   Leave-One-Out.

---

# SAMPLING DATA (SPLITS)

[IMAGE: Bar chart icon]

[IMAGE: Four histograms (All samples, Training samples, Test samples) displaying the distribution of Sepal Width [cm] for three classes (Setosa, Virginica, Versicolor), illustrating how data is split for training and testing while preserving class frequencies.]

This work by Sebastian Raschka is licensed under a Creative Commons Attribution 4.0 International License.

---

# QUESTIONS

[IMAGE: Waving person emoji]

Post on the Discussion board and join class on Thursdays!