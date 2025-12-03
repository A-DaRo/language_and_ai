# 🏷️ Classification

| Property | Value |
| :--- | :--- |
| **Week** | Week 3 |
| **Book Chapters** | Chapters 3, 4, 5 |
| **Slides** | [week3.pdf (File)](./week3.pdf), [week3\_inperson\_slides.pdf (File)](./week3_inperson_slides.pdf) |
| **Recordings** | [tue.video.yuja.com/P/V…216706](https://tue.video.yuja.com/P/VideoManagement/MediaLibrary/search/WatchVideo/3216706) |
| **Solutions** | Empty |

***

> **‼️ This set of exercises is a lot, but I assume you are already familiar with most of it (it's mostly practice material for the ones who aren't). Task 1 covers the language modeling and smoothing components of the lecture. Of specific interest (**check these regardless of your experience**): Task 8 discusses the Multinomial implementation of Naive Bayes which is covered in the book, and is [different from NB discussed in the video lecture](https://nlp.stanford.edu/IR-book/pdf/13bayes.pdf). The video lecture discussed the Bernoulli variant (restricted to binary counts), which is not covered in detail in the book. Note that [smoothing also works differently](https://nlp.stanford.edu/IR-book/pdf/13bayes.pdf) for this variant (see Task 9). You can use [this](https://nlp.stanford.edu/IR-book/pdf/13bayes.pdf) chapter from Introduction to Information Retrieval as reference text if you need one.

## 🎽 Warm-up Questions

***

*   Last week, we talked about representing language as frequencies in a bag-of-words model. As these typically represent language in an unordered fashion, without relations between words, they miss important characteristics of the data (e.g., "good" relating to "movie"). Can you think of simple (or less simple) ways to improve upon simply using these single words as features?
*   In the previous question, you might have opted for feature-engineering some interesting properties of the sentence, but the most common way of making richer contextual representations is by including -context- through encoding sequences of multiple words as a single feature. These are then added to the term matrix, and we get counts when the this now longer string occurs in a document. Can you think of issues with taking these larger contexts as features? Hint: language is creative and the chances of you saying something unique increases with the length of a sequence.
*   The longer the context, the less likely it is to appear outside of initial documents. Let's suppose that we don't have counts for "I drove my blue bike to work", but we did see "I drove my bike to work". Can you think of a way to combine information from these different (partially overlapping) sequences? What makes a particular word 'valid' given a context, or sequence of tokens (e.g., why “I drove my bike” and not “I drove my cat”)? How large does a context need to be to give enough information, you think?
*   Finally, can you think of cultural or demographic differences that might affect the likelihood of a word occurring in a sentence? The sentence “I drove my bike to work” might be a good one to use as an example. 🙂 How do you think such idiosyncrasies affect classifiers, and models that have to interpret / produce language?

## 📺 Lecture Videos

***

1.  **Language Models**
    [Video Placeholder: tue.video.yuja.com/P/VideoManagement/MediaLibrary/search/WatchVideo/3216706]

2.  **Smoothing**
    [Video Placeholder: tue.video.yuja.com/P/VideoManagement/MediaLibrary/search/WatchVideo/3216706]

3.  **Naive Bayes**
    [Video Placeholder: tue.video.yuja.com/P/VideoManagement/MediaLibrary/search/WatchVideo/3216706]

4.  **ID3 Decision Trees**
    [Video Placeholder: tue.video.yuja.com/P/VideoManagement/MediaLibrary/search/WatchVideo/3216706]

5.  **Linear & Logistic Regression**
    [Video Placeholder: tue.video.yuja.com/P/VideoManagement/MediaLibrary/search/WatchVideo/3216706]

6.  **Use Case + SVMs**
    [Video Placeholder: tue.video.yuja.com/P/VideoManagement/MediaLibrary/search/WatchVideo/3216706]

7.  $k$-**Nearest Neighbors**
    [Video Placeholder: tue.video.yuja.com/P/VideoManagement/MediaLibrary/search/WatchVideo/3216706]

8.  **Generalization**
    [Video Placeholder: tue.video.yuja.com/P/VideoManagement/MediaLibrary/search/WatchVideo/3216706]

***

> 💡 Remember that $W$ to refers to a sequence (a tokenized string; could be a sentence, document, etc.), or array of an instance (aka an independent variable), and $w_i$ the numerical representation of the $i$-th element of the sequence (e.g., a single token, a bi-gram, etc.). A more general-purpose notation is $x$ and $x_i$ respectively. Similarly, we use $y$ for a list or array of a target, or label (aka the dependent variable) and $y_i$ for it's $i$-th entry.
>
> We denote the whole feature (or vector) space by $X$, and a vector of an instance in this space by (e.g., the first instance) $\vec{x}_1$. If we want to distinguish two general vectors in the same formula (e.g., for distances) we use $\vec{p}$ and $\vec{q}$ instead.
>
> The empirical mean of an array is denoted by $\bar{x}$ (or $\bar{y}$), which can be calculated using $\frac{1}{n} \sum^n_{i=1}{v_i}$ where $v_i$ is either $x_i$ or $y_i$. Here, $n$ (can also be denoted as $N$) refers to our sample size (or length of the arrays). Remember that $\sum^n_{i=1}$ iterates over all positions in $x$ or $y$, and sums the result of this. This can also be denoted pairwise; for example, given $x = [1, 2]$ and $y = [1, 4]$ the result of $\sum_{i=1}^{n}(x_{i} - y_{i})$ is $(1 - 1) + (2 - 4)$.

## 🌐 $n$-Gram Language Modeling

***

Remember that for bi-grams ($w_{n-1}, w_n$), we can estimate their probabilities from a particular corpus (the bigger, the more truthful to language as a whole) using maximum likelihood estimation:

$$
P(w_n\ |\ w_{n-1}) = \frac{C(w_{n-1},\ w_n)}{C(w_{n-1})}.
$$

With Laplace smoothing, this looks like:

$$
P_{\text{Laplace}}(w_n\ |\ w_{n-1}) = \frac{C(w_{n-1},\ w_n) + 1}{C(w_{n-1}) + V},
$$

where, $V$ is the size of the vocabulary. We can evaluate our language model using Perplexity:

$$
\operatorname{PP}(W)=\sqrt[N]{\prod_{i=1}^{N} \frac{1}{P\left(w_{i} \mid w_{i-1}\right)}}.
$$

Assume we are given the following corpus:

$$
\begin{array}{l}
\texttt{<s> A sentence </s>} \\
\texttt{<s> Another sentence </s>} \\
\texttt{<s> Yet another sentence </s>}
\end{array}
$$

Unigram counts:

| | $\texttt{<s>}$ | $\texttt{a}$ | $\texttt{sentence}$ | $\texttt{another}$ | $\texttt{yet}$ | $\texttt{</s>}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| count | 3 | 1 | 3 | 2 | 1 | 3 |

Bigram counts:

| | $\texttt{<s>}$ | $\texttt{a}$ | $\texttt{sentence}$ | $\texttt{another}$ | $\texttt{yet}$ | $\texttt{</s>}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| $\texttt{<s>}$ | 0 | 1 | 0 | 1 | 1 | 0 |
| $\texttt{a}$ | 0 | 0 | 1 | 0 | 0 | 0 |
| $\texttt{sentence}$ | 0 | 0 | 0 | 0 | 0 | 3 |
| $\texttt{another}$ | 0 | 0 | 2 | 0 | 0 | 0 |
| $\texttt{yet}$ | 0 | 0 | 0 | 1 | 0 | 0 |
| $\texttt{</s>}$ | 0 | 0 | 0 | 0 | 0 | 0 |

> ℹ️ Table above can be read as a bigrams with [left col, right col]; i.e., `<s> a` occurs once, and `another sentence` twice.

Probabilities:

| | $\texttt{<s>}$ | $\texttt{a}$ | $\texttt{sentence}$ | $\texttt{another}$ | $\texttt{yet}$ | $\texttt{</s>}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| $\texttt{<s>}$ | (0+1)/(3+6) | (1+1)/(3+6) | (0+1)/(3+6) | (1+1)/(3+6) | (1+1)/(3+6) | (0+1)/(3+6) |
| $\texttt{a}$ | (0+1)/(1+6) | (0+1)/(1+6) | (1+1)/(1+6) | (0+1)/(1+6) | (0+1)/(1+6) | (0+1)/(1+6) |
| $\texttt{sentence}$ | (0+1)/(3+6) | (0+1)/(3+6) | (0+1)/(3+6) | (0+1)/(3+6) | (0+1)/(3+6) | (3+1)/(3+6) |
| $\texttt{another}$ | (0+1)/(2+6) | (0+1)/(2+6) | (2+1)/(2+6) | (0+1)/(2+6) | (0+1)/(2+6) | (0+1)/(2+6) |
| $\texttt{yet}$ | (0+1)/(1+6) | (0+1)/(1+6) | (0+1)/(1+6) | (1+1)/(1+6) | (0+1)/(1+6) | (0+1)/(1+6) |

### 🚩 Task 1

***

*   Using $P_{\text{Laplace}}$, get all the bi-gram probabilities (using case folding).
*   Using these bi-gram probabilities, calculate the perplexity for the following sentence:
    $\texttt{<s> Sentence another sentence </s>}$

<details>
<summary>**Solution (Full)**</summary>

$$
\begin{array}{ll}
P(\texttt{sentence}\ |\ \texttt{<s>}) &= (0+1)/(3+6) \\
P(\texttt{another}\ |\ \texttt{sentence}) &= (0+1)/(3+6) \\
P(\texttt{sentence}\ |\ \texttt{another}) &= (2+1)/(2+6) \\
P(\texttt{</s>}\ |\ \texttt{sentence}) &= (3+1)/(3+6) \\
\end{array} \\ \quad \\
\operatorname{PP}(W)=\sqrt[N]{\prod_{i=1}^{N} \frac{1}{P\left(w_{i} \mid w_{i-1}\right)}} = \sqrt[4]{\frac{1}{.111} \cdot \frac{1}{.111} \cdot \frac{1}{.375} \cdot \frac{1}{.444}} = \sqrt[4]{487.461} \approx 4.699
$$
</details>

## 🏹 Regression Prediction

***

Remember that for a 2-d space (1 feature, 1 target), the prediction of a linear regression is $\hat{y} = \beta_0 + \beta_1 \cdot x_1$, where $\beta_0$ is the intercept (or bias coefficient) and $\beta_1$ the coefficient for the first feature (i.e., the slope of the line). Here, $x_1$ is the value of the first (and only feature), but coefficients and features up until $\beta_n \cdot x_n$ might be added. Let's assume we are given the following two documents represented by word frequencies:

$$
\begin{array}{r|rrr}
\hline
\text{doc (\#)} & \text{lit}\ (x_1) & \text{dank}\ (x_2) & \text{age}\ (x_3) \\
\hline
1 & 5 & 20 & 30 \\
2 & 7 & 19 & 25 \\
\hline
\end{array}
$$

> ‼️ Please note that these are arbitrary fits.

*   Predict $\hat{y}$ for the first instance using feature $x_1$ only, with $\beta_0 = 2$ and $\beta_1 = 0.5$.
*   Predict $\hat{y}$ for the first instance using all features, with $\beta_0 = 2$, $\beta_1 = 0.5$, $\beta_2 = 0.7$ and $\beta_3 = 0.9$.

<details>
<summary>**Solution (Full)**</summary>
*   $\hat{y} = \beta_0 + \beta_1 \cdot x_1 = 2 + 0.5 \cdot 5 = 4.5$
*   $\hat{y} = \beta_0 + \beta_1 \cdot x_1 + \beta_2 \cdot x_2 + \beta_3 \cdot x_3 = 2 + 0.5 \cdot 5 + 0.7 \cdot 20 + 0.9 \cdot 30 = 45.5$
</details>

*   Given the previous solution, what happens to $\hat{y}$ if the intercept decreases by 2?

<details>
<summary>**Solution**</summary>
$\hat{y}$ decreases by 2.
</details>

## 🔨 Fitting Regression

***

Remember that we can determine the regression line given this data analytically, through:

$$
\beta_1 = \frac{\sum_{i=1}^{n}(x_{i}-{\bar{x}})(y_{i}-{\bar{y}})}{\sum_{i=1}^{n}(x_{i}-{\bar{x}})^{2}},
$$

and:

$$
\beta_0 = \bar{y} - \beta_1 \cdot \bar{x},
$$

where $\bar{x}, \bar{y}$ are the averages for our feature $x$ and target $y$.

### 🚩 Task 3

***

*   Determine the regression line for this data snippet of YouTube comments, where $x$ is represented as word frequencies, and $y$ as meta-data counts:

$$
\begin{array}{ll}
\hline
\text{give-away}\ (x_1)\ & \text{upvotes}\ (y)  \\
\hline
5 & 119 \\
7 & 121 \\
\hline
\end{array}
$$

<details>
<summary>**Solution (Full)**</summary>

$$
\bar{x} = (5 + 7) / 2 = 6, \quad \bar{y} = (119 + 121) / 2 = 120
$$

$$
\beta_1 = \frac{(5 - 6)\cdot(119-120) + (7 - 6)\cdot(121-120)}{(5 - 6)^{2} + (7 - 6)^2} = \frac{2}{2} = 1
$$

$$
\beta_0 = 120 - 1 \cdot 6 = 114
$$

$$
f(X) = 114 + 1 \cdot{x_1}
$$

> 💡 You can confirm this is a good fit by plugging our 'train' values back in: (114 + 1 * 5) = 119 and (114 + 1 * 7) = 121.
</details>

*   Determine the regression line for this data snippet with emojis on Reddit, where both $x$ and $y$ are represented as token frequencies:

$$
\begin{array}{ll}
\hline
\text{💎}\ (x_1)\ & \text{🙌}\ (y)  \\
\hline
5 & 6 \\
7 & 10 \\
9 & 15 \\
\hline
\end{array}
$$

<details>
<summary>**Solution (Full)**</summary>

$$
\bar{x} = (5 + 7 + 9) / 3 = 7, \quad \bar{y} = (6 + 10 + 15) / 3 \approx 10.333
$$

$$
\beta_1 = \frac{(5 - 7)\cdot(6 - 10.333) + (7 - 7)\cdot(10 - 10.333) + (9 - 7) \cdot (15 - 10.333)}{(5 -7)^2 + (7 - 7)^2 + (9 - 7)^2} \\ = \frac{18}{8} = 2.25
$$

$$
\beta_0 = 10.333 - 2.25 \cdot 7 = -5.417
$$

$$
f(X) = -5.417 + 2.25 \cdot{x_1}
$$
</details>

***

You can create more solutions using the following code:

```python
from scipy import stats
x = [5, 7, 9]    # change
y = [6, 10, 15]  # change
slope, intercept, *misc = stats.linregress(x, y)
print(round(intercept, 3), '+', round(slope, 3), '* x_1')
```

## 📊 Evaluating (Linear) Regression

***

Remember that we have the following metrics to evaluate a linear regression model, where $N$ is the amount of instances we have labels for (i.e., equal to the length of $y$ and $\hat{y}$):

$$
\operatorname{MAE} = \frac{\sum^N_{i=1} | \hat{y}_i - y_i |}{N}
$$

$$
\operatorname{MSE} = \frac{\sum^N_{i=1} (\hat{y}_i - y_i)^2}{N}
$$

$$
\operatorname{RMSE} = \sqrt{\frac{\sum^N_{i=1} (\hat{y}_i - y_i)^2}{N}}
$$

$$
R² = 1 -\frac{\sum^N_{i=1} (y_i - \hat{y}_i)^2}{\sum^N_{i=1} (y_i - \bar{y})^2}
$$

### 🚩 Task 4

***

*   Given $y = [8, 3, 1]$ and $\hat{y} = [4, 3, 2]$, calculate MAE, MSE, RMSE, and R².

<details>
<summary>**Solution (Full)**</summary>
$$
\operatorname{MAE} = \frac{|4-8| + |3-3| + |2-1|}{3} = \frac{4 + 0 + 1}{3} \approx 1.667
$$

$$
\operatorname{MSE} = 
\frac{(4-8)^2 + (3-3)^2 + (2-1)^2}{3} \approx 5.667
$$

$$
\operatorname{RMSE} = \sqrt{\frac{(4-8)^2 + (3-3)^2 + (2-1)^2}{3}} 
\approx 2.38
$$

$$
R^2 = 1 - \frac{(8-4)^2 + (3-3)^2 + (1-2)^2}{(8 - 4^{\star})^2 + (3 - 4)^2 + (1 - 4)^2} = 1 - \frac{17}{26}
\approx 0.346 \\ \quad \\ {}^\star\bar{y} = 
(8+3+1)/3 = 4
$$

</details>

*   Given $y = [10, 22, 31, 7]$ and $\hat{y} = [20, 11, 7, 10]$, calculate MAE, MSE, and RMSE.

<details>
<summary>**Solution</summary>
$MAE = 12.0 \quad MSE = 201.5 \quad RMSE = 14.195 \quad R^2 = -1.184$
</details>

***

> ‼️ Unlike the other metrics, `r2_score` is **not a symmetric function** in Scikit-learn, so the order of `ŷ` and `y` matters. Be careful!

You can create more solutions using the following code:

```python
from sklearn import metrics
import numpy as np
y = [8, 3, 1]  # change
ŷ = [4, 3, 2]  # change
mae = metrics.mean_absolute_error(y, ŷ)
mse = metrics.mean_squared_error(y, ŷ)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y, ŷ)
print('mae:', round(mae, 3), 'mse:', round(mse, 3), 
	    'rmse:', round(rmse, 3), 'r2:', round(r2, 3))
```

## 🏷️ Evaluating Classification

***

Remember that for classification, we are generally interested in how our classifier confuses predictions. This is the goal of a confusion matrix:

$$
\begin{array}{l|ll}
      & \hat{y} = 1 & \hat{y} = 0   \\
\hline
y = 1 & TP          & FN   \\
y = 0 & FP          & TN   \\
\end{array}
$$

It might either classify something as negative ($\hat{y} = 0$) while the actual label is positive ($y = 1$), which we would refer to as a False (prediction validity) Negative (predicted label), and a positive prediction ($\hat{y} = 0$) while the actual label is negative ($y = 0$), as a False Positive. We can use these in ratios to produce various metrics:

$$
\text{Accuracy} = \frac{TP + TN}{TP + FN + FP + TN}
$$

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

$$
F_\beta\ \text{score} = \frac{(1 +\beta^2) \cdot TP}{(1+\beta^2) \cdot TP + \beta^2 \cdot FN + FP}
$$

### 🚩 Task 5

***

*   Given a collection of classifications and labels:
$$
y = [0, 1, 0, 1, 0, 1, 0, 1, 0, 0]
\\ \hat{y} = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1]
$$
You may assume 0 is the negative class, 1 the positive. Produce the confusion matrix.

<details>
<summary>**Solution (Full)**</summary>
We have two TPs, two TNs, four FPs, and two FNs, so:
$$
\begin{array}{l|ll}
      & \hat{y} = 1 & \hat{y} = 0   \\
\hline
y = 1 & 2          & 2   \\
y = 0 & 4          & 2   \\
\end{array}
$$
</details>

*   Given the same labels $y = [0, 1, 0, 1, 0, 1, 0, 1, 0, 0]$, do this for a majority baseline.

<details>
<summary>**Solution (Full)**</summary>
The majority class is 0 (6/10), so a majority baseline would produce the following predictions:
$$
y = [0, 1, 0, 1, 0, 1, 0, 1, 0, 0]
\\ \hat{y} = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
$$
Now we have six TNs and four FNs:
$$
\begin{array}{l|ll}
      & \hat{y} = 1 & \hat{y} = 0   \\
\hline
y = 1 & 0          & 4   \\
y = 0 & 0          & 6   \\
\end{array}
$$
</details>

*   Calculate accuracy, precision, recall, and $F_1$ score with respect to the positive class (1) given these two confusion matrices above (classifier and majority baseline).

<details>
<summary>**Solution (Full)**</summary>
**Classifier:**
$$
\text{accuracy} = \frac{TP + TN}{TP + FN + FP + TN} = \frac{2 + 2}{2+2+4+2} = 0.4
$$
$$
\text{precision} = \frac{TP}{TP + FP} = \frac{2}{2 + 4} = 0.333
$$
$$
\text{recall} = \frac{TP}{TP + FN} = \frac{2}{2 + 2} = 0.5
$$
$$
F_1\ \text{score} = \frac{(1 + \beta^2) \cdot TP}{(1+\beta^2) \cdot TP + \beta^2 \cdot FN + FP} = \frac{(1+ 1^2) \cdot 2 }{(1 + 1^2) \cdot 2 + 1^2 \cdot 2 + 4} = 0.4
$$

**Baseline:**
$$
\text{accuracy} = \frac{TP + TN}{TP + FN + FP + TN} = \frac{0 + 6}{0+4+0+6} = 0.6
$$
$$
\text{precision} = \frac{TP}{TP + FP} = \frac{0}{0 + 0} = 0
$$
$$
\text{recall} = \frac{TP}{TP + FN} = \frac{0}{0 + 4} = 0
$$
$$
F_1\ \text{score} = \frac{(1 + \beta^2) \cdot TP}{(1+\beta^2) \cdot TP + \beta^2 \cdot FN + FP} = \frac{(1 + 1^2) \cdot 0 }{(1 + 1^2) \cdot 0 + 1^2 \cdot 4 + 0} = 0
$$

</details>

***

You can create more solutions using the following code:

```python
from sklearn import metrics
y = [0, 1, 0, 1, 0, 1, 0, 1, 0, 0]   # change
ŷ = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1]   # change
ŷb = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # change
print("classfier\n")
print(metrics.confusion_matrix(y, ŷ))
print(metrics.classification_report(y, ŷ))
print("basline\n")
print(metrics.confusion_matrix(y, ŷb))
print(metrics.classification_report(y, ŷb))
```

## 🏡 $k$-Nearest Neighbors

***

Remember that the $k$-NN algorithm is as follows:

*   Given: $k$, and a distance metric $S$.
*   At training (fitting) time, given: a training set $X$, and labels $y$ we simply 'remember them'.
*   At testing (inference) time, given: a test set $T$, we do the following:
    *   For a given $\vec{t}$ in $T$, for all $\vec{x}_n$ in $X$, we calculate $S(\vec{t}, \vec{x}_n)$. This results in a list $S_{\vec{t}}$ with distances, and thus closest neighbors, for instance $\vec{t}$. We store each distance along with its original index $n$.
    *   We rank this $S_{\vec{t}}$ list, and take the top $k$ as the neighbors we are interested in,
    *   We look up the indices $n$ for the top neighbors in $y$ (so we know their labels).
    *   We choose the majority label of these top neighbors, and 'return' our prediction $\hat{y}$ for $\vec{t}$.

### 🚩 Task 6

***

*   Given $k=1$ and distance = [Euclidean](https://example.com/euclidean-distance), $X, y$ and $\vec{t}$ below, what is $\hat{y}$?

$$
X = 
\begin{bmatrix}
2 & 5 & 10 \\
3 & 2 & 6  \\
\end{bmatrix} \quad y = [0, 1] \quad \vec{t} = \langle 4, 3, 8 \rangle
$$

<details>
<summary>**Solution (Full)**</summary>
Remember that the Euclidean Distance is defined as:
$$
\sqrt{\sum_{i=1}^{n}\left(\vec{p}_{i}-\vec{q}_{i}\right)^{2}}
$$
$$
\text{E}(X_0, \vec{t}) = \sqrt{(2 - 4)^2 + (5 - 3)^2 + (10 - 8)^2} = 3.464 
$$
$$
\text{E}(X_1, \vec{t}) = \sqrt{(3 - 4)^2 + (2 - 3)^2 + (6 - 8)^2} = 2.450
$$
The ranked neighbors are: $[\text{E}(X_1, \vec{t}) = 2.450,\  \text{E}(X_0, \vec{t}) = 3.464 ]$, from which the top $k$ gives us $X_1$ with $y_1 = 1$.
</details>

***

You can create more solutions using the following code:

```python
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(
		n_neighbors=1,           # change
		metric='euclidean'       # change
)
X = [[2, 5, 10], [3, 2, 6]]  # change
y = [0, 1]                   # change
t = [[4, 3, 8]]              # change
knn.fit(X, y)
print(knn.predict(t)[0])
```

## 🎄 Information Gain (Decision Trees)

***

Decision trees algorithms recursively make splits of the data, until they hit some criterion. Through this, they produce 'rules', which, at the end of the tree, we predict the majority label for the leftovers. In the lecture, we restrict ourselves to a *single* split of the data, and looked at Information Gain:

$$
IG(Y,x_n) = \text{Entropy}(Y) - \sum_{v\in x_n} \frac{|Y_v|}{|Y|} \cdot \text{Entropy}(Y_v),
$$

where Entropy can be calculated using:

$$
\text{Entropy}(S) = - \sum^n_{i=1} p_i \cdot \log_2{p_i}.
$$

Here:

*   $Y$ = The target column (label) we related the split quality to.
*   $x_n$ = The feature (word / token / etc) we are testing our split on.
*   $v$ = Each unique value of $x$, so if $x = [0, 1, 2, 1, 1, 2, 3, 1]$, then $v$ might be 0, 1, 2, or 3.
*   $Y_v$ = The subset of $v$ with respect to $Y$, so if $y = [0, 1, 0, 1]$ and $x = [0, 1, 1, 0]$, then $Y_{v=1}$ gives us $y = [1, 0]$ and $x = [1, 1]$.
*   $|Y|$ and $|Y_v|$ = the sizes of the respective splits.
*   $p_i$ is the probability of drawing label value $i$ from $S$.

### 🚩 Task 7

***

*   Given the dataset below, calculate the information gain of feature $x_0$.

$$
\begin{array}{ll}
\hline
x_0 & y \\
\hline
1 & 0 \\
2 & 1 \\
0 & 1 \\
1 & 0 \\
2 & 0 \\
1 & 1 \\
\hline
\end{array}
$$

<details>
<summary>**Solution (Full)**</summary>
First, let's solve the left part — $\text{Entropy}(Y)$. This we do like so:
$$
\text{Entropy}(Y) = - \left( \frac{3}{6} \cdot \log_2{\frac{3}{6}} + \frac{3}{6} \cdot \log_2{\frac{3}{6}} \right) = 1
$$
Then, we solve the right part, we calculate the entropy values for all values of $x_0$:
$$
\text{Entropy}(x_0 = 0) = - \left( \frac{1}{1} \cdot \log_2{\frac{1}{1}} \right) = 0
$$
$$
\text{Entropy}(x_0 = 1) = - \left( \frac{2}{3} \cdot \log_2{\frac{2}{3}} + \frac{1}{3} \cdot \log_2{\frac{1}{3}} \right) = 0.918
$$
$$
\text{Entropy}(x_0 = 2) = - \left( \frac{1}{2} \cdot \log_2{\frac{1}{2}} + \frac{1}{2} \cdot \log_2{\frac{1}{2}} \right) = 1
$$
Then we multiply those values by the sizes of their respective splits divided by the total dataset size, and put all the elements together:
$$
IG(Y,x_n) = 1^\star -\left( \frac{1}{6} \cdot 0 + \frac{3}{6} \cdot 0.918 + \frac{2}{6} \cdot 1 \right) = 0.208 \\ {}^\star = \text{Entropy}(Y)
$$
</details>

## 🎛️ Naive Bayes

***

Remember that:

$$
\hat{y} = \arg \max_y P(y) \prod_{j=1}^J P(x_j\ |\ y).
$$

$P(x_j\ |\ y)$ implies that for $x_j$, we get its conditional probability given every value of $y$, so 'how much does feature value $x_j$ occur under (for example) positive labels, and under negative labels'. Or, based on the equation from the book:

$$
\hat{P}(x_j\ |\ y) = \frac{C(x_j,\ y)}{\sum_{x_i \in V} C(x_i, y)},
$$

where $C$ is a co-occurrence count of the two inputs, and $V$ is our vocabulary. We then multiply and sum these (product $\prod$), and multiply them by prior $P(y)$ — i.e., the probabilities of a value in $y$ occurring.

The algorithm follows the following steps:

*   Given: a smoothing factor $\epsilon$ (there are more hyperparameters such as the assumed distribution, but we don't discuss those):
*   At training (fitting) time, given: a dataset $X$, we do the following:
    *   We get all probabilities (right-hand part after argmax) for our current data, and store it.
    *   Any $C$, we optionally smooth with $\epsilon$ (typically +1). For the denominator this means that:
        $\sum_{x_i \in V} \left(C(x_i, y) + \epsilon\right)$, which is equal to $\left(\sum_{x_i \in V} C(x_i, y)\right) + |V| \cdot \epsilon$
*   At test (inference) time, given new instance $t$, we:
    *   Look up the pre-calculated probabilities for the features [present](https://example.com/features-present) in $t$ ($x_j$), with respect to each possible label in $y$.
    *   We return label with the highest probability ($\arg \max_y$) as $\hat{y}$.

### 🚩 Task 8

***

*   Given the dataset below, and the following instance: $d = \texttt{all was not great}$ — calculate $\hat{y}$, counts are smoothed by +1.

$$
\begin{array}{lllll}
\hline
\texttt{all} & \texttt{great} & \texttt{not} & \texttt{bad} & y \\
\hline
0 & 0 & 0 & 1 & - \\
0 & 1 & 0 & 0 & - \\
0 & 0 & 0 & 1 & - \\
1 & 0 & 0 & 1 & - \\
1 & 1 & 1 & 1 & + \\
0 & 0 & 1 & 1 & + \\
1 & 1 & 1 & 0 & + \\
\hline
\end{array}
$$

<details>
<summary>**Solution (Full)**</summary>
The counts are obtained by looking at a particular word, and summing their frequencies across **one** particular label: e.g., $\texttt{all}$ occurs two times under $y=+$ ($2$, and we add $1$ for Laplace smoothing). Note that $\texttt{not}$ never occurs under $y=-$, so $p(\texttt{not} | -) = 0 + 1$ — again, the +1 being Laplace smoothing (always one for the numerator). The denominator is smoothed as (sum of all frequencies under a label + $|V| \cdot \epsilon$, where $\epsilon$ is the smoothing factor, i.e., +1) — also see [here](https://example.com/smoothing-works-differently). So for $+$, we sum all the frequencies:
$$
\sum \left[ \begin{array}{llll}
1 & 1 & 1 & 1 \\
0 & 0 & 1 & 1 \\
1 & 1 & 1 & 0 \\ 
\end{array} \right]
\begin{array}{l}
+ \\
+ \\
+ \\
\end{array}  = 9
$$
Then $9 + |V| \cdot \epsilon = (9 + (4 \cdot 1))$ (we drop the $\cdot 1$). This gives for all probabilities:
$$
\begin{array}{lr}
p(\texttt{all}\quad   | +) = (2 + 1)/(9+4) \approx .213 & p(\texttt{all}\quad   | -) = (1 + 1)/(5 + 4) \approx 0.222 \\
p(\texttt{great} | +) = (2+1)/(9+4) \approx .213 & p(\texttt{great} | -) = (1+1)/(5 + 4) \approx 0.222 \\
p(\texttt{not}\quad   | +) = (3+1)/(9+4) \approx .308 & p(\texttt{not}\quad   | -) = (0+1)/(5 + 4) \approx 0.111 \\
p(\texttt{bad}\quad   | +) = (2+1)/(9+4) \approx.213 & p(\texttt{bad}\quad   | -) = (3+1)/(5 + 4) \approx 0.444 \\
p(\texttt{yes}) \quad\quad      = 3/7 \approx 0.429 & p(\texttt{no}) \quad\quad\ \        = 4/7 \approx 0.571 \\
\end{array} \\ \quad \\
\  \ \ \ \texttt{all}\   \ * \texttt{great}   * \texttt{not}\ \     *\text{prior} \\
p(- | x) = 0.222 * 0.222 * 0.111 * 0.571 \approx .003 \\
p(+ | x) = 0.213 * 0.213 * 0.308 * 0.429 \approx .006 \\

\hat{y} = \arg \max_y [.003, .006] = 1\quad (\text{i.e., +})
$$
> 💡 You can check if the nominators are correct by summing them for a particular label (vertically per column), this should give the denominator for all columns (e.g.,: $2 + 2+ 3 + 2 = 9$ and $1 + 1 + 1 + 1 = 4$).
</details>

## 📺 Additional Explanations

***

Task 7 and 8 were discussed in a bit more detail in the lectures of previous years. You can find the video below:

[Video Link Placeholder: tue.video.yuja.com]

## 🔨 Smoothing Across Techniques

***

As you saw in the previous Task, [smoothing](https://example.com/smoothing-across-techniques) pops up in different parts of NLP where we use fractions or (log) probabilities. Each have their 'own' implementation. Two of such cases that are relevant to the current material are Bernoulli Naive Bayes (NB), and $\text{tf}\cdot\text{idf}$. For Bernoulli NB (explained in the lecture), we have the following distinction from Multinomial NB: the Bernoulli model estimates $\hat{P}(w\ |\ c)$ as the *fraction of documents of class $c$ that contain word $w$*. In contrast, the Multinomial model estimates $\hat{P}(w\ |\ c)$ as the *fraction of tokens or fraction of positions in documents of class $c$ that contain word $w$*. Because of this distinction, we smooth the denominator of Bernoulli NB with a constant of 2, as this class only models occurrence under a label, and nonoccurence (rather than all counts of all features). So Laplace MLE for this model looks like:

$$
P_\text{Laplace}(w\ |\ c) = \frac{C(w, c) + 1}{C(c) + 2}
$$

For $\text{tf}\cdot\text{idf}$, we also apply smoothing to avoid the undefined $\log$ of zeroes, and/or division by zero. In [Exercise 1](https://example.com/exercise-1), the most rudimentary $\text{tf}$ smoothing version was applied. Typically, we either implement *sublinear term frequency scaling*, as a weighted term frequency ($\text{wf}\ $) like so:

$$
\mathrm{wf}_{t, d}= \begin{cases}1+\log \mathrm{tf}_{t, d} & \text { if } \mathrm{tf}_{t, d}>0 \\ 0 & \text { otherwise }\end{cases}
$$

Notice that, compared to [here](https://example.com/idf-smoothing-comparison), there is no zero smoothing, and we do not add-one inside the $\log$ (as the log of 1 is also undefined), but simply apply it to non-zero values. We leave the zeroes as zeroes. This gives us $\mathrm{wf}_{t, d} \cdot \mathrm{idf}_{t}$. Also, again compare to the equation in the [1st exercise](https://example.com/first-exercise), the $\text{idf}$ value can still give us zero divisions. Hence, add-one smoothing is typically applied in the $\text{idf}$ portion, like so:

$$
\text{idf}_t = 1 + \log_b \frac{N + 1}{ \text{df}_t + 1}
$$

Remember that $b=10$, i.e., $\log_b = \log_{10} = \lg$.

### 🚩 Task 9

***

*   Given the following example from Exercise 1:
$$
\begin{array}{lllllll}
\hline
\# & \texttt{the} & \texttt{cat} & \texttt{sat} & \texttt{on} & \texttt{mat} & \texttt{my} \\
\hline
1 & 2 & 1 & 1 & 1 & 1 & 0 \\
2 & 0 & 2 & 1 & 1 & 0 & 2 \\
3 & 1 & 2 & 1 & 2 & 1 & 2 \\
\hline
\end{array}
$$
Apply sublinear tf scaling and smooth the idf values.

<details>
<summary>**Solution**</summary>
For the second document, we for example have:
$$
\begin{array}{lr}
w_{\texttt{the},2} = \text{wf}_{t,d} \cdot 1 + \log_b \frac{N + 1}{ \text{df}_t + 1} &= 0 \cdot \left(1 + \lg \frac{3 + 1}{2 + 1}\right) = 0.000 \\ \\
w_{\texttt{cat},2} = \text{wf}_{t,d} \cdot 1 + \log_b \frac{N + 1}{ \text{df}_t + 1} &=  \left(1 + \ln \left(2 \right)\right) \cdot \left(1 + \lg \frac{3 + 1}{3 + 1}\right) \approx 1.693 \\ \\
w_{\texttt{my},2} = \text{wf}_{t,d} \cdot 1 + \log_b \frac{N + 1}{ \text{df}_t + 1} &= \left(1 + \ln \left(2 \right)\right) \cdot \left(1 + \lg \frac{3 + 1}{2 + 1}\right) \approx 1.905
\end{array}
$$
These are obtained as follows, taking $w_{\texttt{cat},2}$ as an example: $\texttt{cat}$ occurs two times in document 2 ($\# = 2$), this is the term frequency ($\operatorname{tf}_{t,d}$). This is not zero (like $\texttt{the}$), so we apply the natural $\log$ ($\ln$) to the term frequency. This gives us the left portion $\left(1 + \ln \left(2 \right)\right)$. The right portion is calculated using [this](https://example.com/idf-equation) equation. Here, cat occurs in all three documents, so its $\text{df}_t = 3$. $N$ is the number of documents, so 3. We smooth both the numerator and denominator according to the equation, and add it on top of 1 to avoid the whole right hand being 0. So we get: $\left(1 + \lg \frac{3 + 1}{3 + 1}\right)$. Multiplying the left and right-hand side gives the solution.
</details>