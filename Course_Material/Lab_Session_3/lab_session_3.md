--- START OF FILE index.html ---
# 🌳 Lab Session 3

| Property | Value |
| :--- | :--- |
| Week | Week 3 |
| Book Chapters | Empty |
| Slides | Empty |
| Recordings | Empty |
| Solutions | [lab\_3\_solutions.py (File Link Placeholder)](lab_3_solutions.py) |

> 💡 Solution functions to implement with the self-grader are given for Task 1 and 2.

## Table of Contents

*   [🏋️ Predictions from Scratch](#predictions-from-scratch)
    *   [💪 Information Gain](#information-gain)
        *   [🚩 Task 1](#task-1)
    *   [🌲 Splitting the Data](#splitting-the-data)
        *   [🚩 Task 2](#task-2)
*   [🤖 Painless Predictions: Scikit-learn](#painless-predictions-scikit-learn)
    *   [🚀 Getting Vectors For Our Data](#getting-vectors-for-our-data)
        *   [🚩 Task 3](#task-3)
    *   [🪛 Tuning Some Classifiers](#tuning-some-classifiers)
        *   [🚩 Task 4](#task-4)
        *   [⭐ Bonus Task: Improve the Pipeline](#bonus-task-improve-the-pipeline)

---

## 🏋️ Predictions from Scratch

So we've worked on representing text as vectors in [Practical 1](https://mctenthij.notion.site/Vector-Space-Models-9650fa70c9157247) (Link Placeholder), and we've prepared the data (see [Practical 2](https://mctenthij.notion.site/Feature-Engineering-521cc1417ad5a333) (Link Placeholder)), and got some intuitions about the task (classifying Subreddits by topic). Before we put things together, let's work on implementing a classifier we have seen in the Lecture—from scratch. We'll try the ID3 Decision Tree, and take the example from the Exercises this week to test:

```python
import numpy as np

X = np.array([
    [1, 2, 0, 1, 2, 1]
]).T  # transpose to make this a feature column
y = np.array([0, 1, 1, 0, 0, 1])
```

### 💪 Information Gain

---

Let's start out with calculating Information Gain for a particular feature first, as that is basically our recursive operation to determine the left/right/etc. splits of the tree. Above, I formatted the single feature as a matrix so that we can make the function more general as we go. Below, you can see a minimal implementation of the parts around calculating Information Gain. Let's break this down.

```python
def count(x):
    return NotImplemented

def entropy(x, y):
    return NotImplemented

def information_gain(x, y):
    IG = entropy(count(y), len(y))
    for value in np.unique(x):
        xy_split = y[np.where(x == value)]
        E = entropy(count(xy_split), len(xy_split))
        IG -=  # implement
    return IG

information_gain(X[:,0], y)
```

So the main thing we require is in this equation (from the Exercise)
$$
IG(Y,x_n) = \text{Entropy}(Y) - \sum_{v\in x_n} \frac{|Y_v|}{|Y|} \cdot \text{Entropy}(Y_v),
$$
from which we start with the left-hand side of the equation, i.e., the entropy of $Y$:
$$
\text{Entropy}(S) = - \sum^n_{i=1} p_i \cdot \log_2{p_i}.
$$
Note that:
$$
p_i = \frac{C(S_i)}{C(S)}
$$

So we need two things: a count per value of $S$, and the total $S$. The latter we can simply get with `len`, but for the former we need a `count` function (`NotImplemented`). Once we have those, we can submit them to `entropy`, which should sum the product of $p_i$ and $\log_2\ p_i$. You can do this using array operations, or in a loop. Once we have this functionality down, we can basically repeat the same steps for every $v \in x_n$; i.e., we need a summation loop for several entropy values. What is left, then, is taking 'splits' from the data; we want to know the counts for a particular $v$ split by label (to do $|Y_v|\ /\ |Y|$, **and** Entropy). This is handled by this particular line:

```python
xy_split = y[np.where(x == value)]
```

This indexes into $Y$ (`y[...]`), and takes only the values where $x_n = v$. To test what this does, you can try the following:

```python
x = X[:,0]  # our feature x_n
v = 1       # the feature value 1
print(x)
print(np.where(x == v))  # these are indices of x
print(y[np.where(x == v)])
```

The `IG` part requires `yv / y` and `entropy` to be implemented still.

### 🚩 Task 1

*   Implement the `count` function that as input receives an `array` of `int`s ($S$), and returns $C(S_i)$: an `array` with the **non-zero** counts per $i$ (sorted by index).

> **Hint**
> You can use `numpy` for both of the required operations (counting and nonzero filtering).

> **Hint**
> `unique` with `return_counts` gives us what we need with some selection, `bincount` can be combined with `np.nonzero` and array indexing to do the same (with less operations).

> **Solution**
```python
def count(x):
    c = np.bincount(x)
    return c[c.nonzero()]
```

*   Implement the `entropy` function that as input receives $C(S_i)$: an `array` of `int`s of `count` output, and $C(S)$: an `int` with the total count (length) of $S$. It returns a `float` (Entropy)

> **Hint**
> The denominator is given in `y`, there are multiple numerators, you can split those either in a loop, or use array operations to do it in one go.

> **Hint**
> If you use the formula [here](https://mctenthij.notion.site/Lab-Session-3-29d979eeca9f8170a934d8f919536672#56b5e1c4f58e4fa3b74c901a04ce23a4), $p_i$ is essentially x/y.

> **Solution**
```python
def entropy(x, y):
    return -(sum(x / y * np.log2(x / y)))
```

*   Complete the `IG` calculation by incorporating $|Y_v|\ /\ |Y|$ and `entropy`.

> **Hint**
> $|Y_v|$ can be extracted from `y_split`.

> **Hint**
> You can do this with `len`, same with $|Y|$ from `y`.

> **Solution**
```python
def information_gain(x, y):
    IG = entropy(count(y), len(y))
    for value in np.unique(x):
        y_split = y[np.where(x == value)]
        E = entropy(count(y_split), len(y_split))
        IG -= (len(y_split) / len(y)) * E
    return IG
```

---

### 🌲 Splitting the Data

---

So, we can calculate IG per feature (using `information_gain`), which means we can rank our features, determine which one we make our first split on, and make the split based on the values. We need an outer loop over `X.T` (to loop per feature), store, and sort the IGs, and return the column index of the feature (`best_split_feature`). The code for handling the splits once the best index is known (`NotImplemented`) is given below (`split_data`):

```python
def best_split_feature(X, y):
    return NotImplemented

def split_data(X, y):
    top_ix = best_split_feature(X, y)
    for v, split_ix in [(v, np.where(X[i, :] == v))
                        for v in np.unique(X[i,:])]:
        yield top_ix, v, X[split_ix], y[split_ix]
```

You can use this matrix for this part:

```python
X = np.array([
    [1, 0, 0, 1, 2],
    [2, 0, 1, 1, 0],
    [1, 1, 0, 0, 2],
    [1, 2, 0, 0, 1],
    [1, 0, 0, 1, 2],
    [0, 0, 0, 1, 0]
])
y = np.array([0, 1, 1, 0, 0, 1])
```

### 🚩 Task 2

*   Implement the `best_split_feature` function, which takes as input a matrix `X` and an array `y`. It returns an `int` which is the index of the feature with the most information gain.

> **Hint**
> You can `enumerate` the `X.T` loop to keep track of the feature index.

> **Hint**
> We had IG defined before, you here need to use it on $x_n$ (`x` in `X.T`) and $y$.

> **Hint**
> If you store these two components in a list of tuples, you can sort them (see first Practical).

> **Solution**
```python
def best_split_feature(X, y):
    scores = [(i, information_gain(x, y)) for i, x in enumerate(X.T)]
    return sorted(scores, key=lambda x: x[1], reverse=True)[0][0]
```

> 💡 Note that once we have `split_data`, the parts are in place to (theoretically) build our tree:
>
> *   The feature and feature value to create a Leaf node from.
> *   The new splits, for which we either:
    *   Return the $y$ value if the split is pure (you can check this with `len(np.unique(y)) > 1`), or we exceeded some criterion (like depth).
    *   Repeat `split_data` if neither of those conditions are met (recursion).

We just require a `Tree` class to keep track of the (nested) leaves and unroll their decisions if we have new data. We'll stop here for this part, though. Funnily enough, even though I assigned completely 'random' (humans are bad at randomness) integers to $X$ — the first split is already a pure one for all values. 🥳

---

## 🤖 Painless Predictions: Scikit-learn

---

Scikit-learn (sklearn) is a fantastic library to get started with several prediction problems, including text classification. We'll focus on 'the works', but—as I assume the majority of you are familiar with the library—mostly on integrating spaCy, Pipelines, and (fair) evaluation.

> ‼️ This part will mostly be library-driven, but I expect you to be familiar with the parts explained here, and to be able to apply them to new problems.

[Scikit-learn Bookmark Placeholder]

### 🚀 Getting Vectors For Our Data

---

Remember that in the previous Lab Session, we loaded text data from political Subreddits, using Pandas, like so:

```python
import pandas as pd

# mirror https://surfdrive.surf.nl/files/index.php/s/9ROTj6HWRAlvngn/download
df = pd.read_csv('https://onyx.uvt.nl/grabber/politics/reddit-5000.csv',
                 na_values=['[deleted]', '[removed]']).dropna()
df.head(5)
```

Scikit-learn has a compatibility layer for pandas, so we can get to work immediately. Let's start by splitting our data into a train and test set, using `train_test_split` ([docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) (Link Placeholder)). Note that our dataset is not nicely distributed:

```python
%matplotlib inline
df['subreddit'].value_counts().plot(kind='bar')
```

We have a few options:

1.  We **undersample** the majority class—randomly being the most naive approach. More elegant approaches generally rely on [distance](https://imbalanced-learn.org/dev/references/generated/imblearn.under_sampling.CondensedNearestNeighbour.html) (Link Placeholder): we'd probably first want to exclude instances that are very similar, rather than throwing potentially informative ones away at random. Generally this a good option if the other data is plentiful.
2.  We **oversample** the minority class (see e.g. [SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html) (Link Placeholder)) by generating synthetic data. This is a really challenging problem for text data, so I would tread carefully.
    > 💡 It's a bit out of scope for this practical and course, but Imbalanced-Learn is a nice fork of Scikit-learn for these kinds of datasets.
    >
    > [Imbalanced-Learn Bookmark Placeholder]
3.  We account for the class imbalance using **stratified sampling**, and class weights in classifiers (if they support this).
4.  We ignore this and hope for the best (wouldn't recommend this personally 🙂).

For this practical, we'll stratify and hope for the best. Once we have these sets, we can vectorize them. Scikit-learn offers a few different flavors ([here](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text) (Link Placeholder)). Note that these have different parameter settings, one of which is the ability to pass a [custom tokenizer](https://scikit-learn.org/stable/modules/feature_extraction.html#customizing-the-vectorizer-classes) (Link Placeholder).

### 🚩 Task 3

*   Use `train_test_split` to split the dataframe into 90% train and 10% test. Make sure to set a seed to control for randomness.

> **Hint**
> Read the documentation to confirm which parameters to set.

> **Hint**
> You can call this function on `df['text']` and `df['subreddit']`.

> **Hint**
> `stratify` need `df['subreddit']` too (or `y`)

> **Solution**
```python
X_train, X_test, y_train, y_test = \
	train_test_split(df['text'], df['subreddit'],
                   test_size=0.1, stratify=df['subreddit'])
```

*   Convert the training and test set into tf*idf vectors, with sublinear scaling on, and `spaCy` as tokenizer (see previous Lab Session). Think about how to handle each split separately with the sklearn API (reading the documentation will help).

> **Hint**
> `fit_transform` happens on one part of the set, `transform` on the other

> **Hint**
> You can use a class similar to the [Preprocessor](https://mctenthij.notion.site/Feature-Engineering-521cc1417ad5a333#86262d3945f94188a28781c5d31fc299) class from the previous practical, instantiate it before, (`proc = Preprocessor` and then pass `proc.transform` as tokenizer. Note that we implemented this class to work on X (a dataframe), whereas it now needs to work on a single `doc` (a string). So, `proc.transform(doc)` should return a single `list` of tokens.

> **Hint**
> Regarding the `fit` / `transform` question. You can think of tf*idf as a learner of a **vocabulary** and **document weights**. Sure, we can update them when we see new data, but that would require reweighing everything with every new instance (think of a live application for example), and add more features, thus making 'older' classifiers with less features incompatible. Hence, we only `fit` on the training data.

> **Solution**
```python
class Tokenizer(object):
    
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        
    def tokenize(self, doc):
        return [token.text for token in self.nlp(doc)]
    
    def lemmatize(self, doc):
        return [token.lemma_ for token in self.nlp(doc)]

tok = Tokenizer()
tfidf = TfidfVectorizer(sublinear_tf=True, tokenizer=tok.tokenize)
X_tr_tf = tfidf.fit_transform(X_train)
X_te_tf = tfidf.transform(X_test)
```

> 💡 A note on randomness: random processes creep into more places than you can see from library’s their API: probability-based classifiers rely on it, `train_test_split` actually does a (random) shuffle before splitting, Neural Networks usually randomly initialize weights, and so forth; even Python itself and `numpy` have seeds that need to be set to be extra certain that when you repeat an experiment, you’ll get the same results.

### 🪛 Tuning Some Classifiers

---

Here we'll test the very useful Pipeline environment; it allows us to define several steps, and as all have `fit` and `transform` methods, they chain easily under one class. You can see it in action [here](https://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html#sphx-glr-auto-examples-model-selection-grid-search-text-feature-extraction-py) (Link Placeholder), but I'll walk through setting this up for our current ask.

Before we defined tf*idf, but we might want to tune some options there as well, such as lemmatization instead of simple tokenization, $n$-gram ranges, case folding yes or no, you name it. We'll therefore also incorporate it as an element in our hyperparameter tuning.

As classifier, let's stick to Multinomial Naive Bayes for now; it tends to work well with tf*idf, and does not require any tuning. This whole lot (if you distill it from the page), looks like so:

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(sublinear_tf=True)),
    ("clf", MultinomialNB()),
])

parameters = {
    # "tfidf__tokenizer": [tok.tokenize, tok.lemmatize],
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_estimator_.get_params()

print("Best parameter settings:")
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

model = grid_search.best_estimator_
```

> ‼️ Note two impracticalities here: when passing `spaCy` as a tokenizer in Grid Search, it implies it does 'live' tokenization for every combination of parameters. It's slow as-is, so realistically, you might want to pre-tokenize the data (e.g., in the DataFrame), and just run this experiment twice (once with tokens, once with lemmas) to not blow up the computation time. Moreover, using a class like we did, avoids instantiating `spaCy` every parse (which is slow), but it also means that scikit-learn will not let you run multithreaded using the `n_jobs` parameter (because of Python's GIL, [long story](https://realpython.com/python-gil/) (Link Placeholder)). So: don't do this at home (commented out for this reason, you can try, but it's really slow, set `n_jobs` to 1).

So note that our pipeline is a list of tuples, where the first element is a `str` with a name identifier. You can also set hyperparameters you'd like to keep static in this step. Then, for the parameters, we use a `dict` where the key is "`str` name id + `__` + the parameter name" (e.g., `tfidf__tokenizer`), and then a list of parameters as value. You can see the tokenizer example in the code above. We then run Grid Search in a cross-validation setting ([docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) (Link Placeholder)). It takes these two elements, and has addition arguments. It is fitted similar to the classifiers/vectorizers. Via the class methods, we can extract the parameters for the best performing classifier in the grid. Finally, we print these best-performing parameters.

### 🚩 Task 4

*   Add lowercasing (y/n), unigram, and uni+bigram features to the grid search.

> **Hint**
> Read the tfidf and gridsearch documentation and provided example. 🙂

> **Hint**
> It's `"tfidf__lowercasing": [something, something],` that needs to be added.

> **Hint**
> It's a dict, don't forget the closing commas.

> **Solution**
```python
parameters = {
    "tfidf__ngram_range": ((1, 1), (1, 2)),  # unigrams or uni+bigrams
    "tfidf__lowercase": [True, False],       # lowercase yes / no
}
```

*   Evaluate the pipeline on `X_test` by using the model's `predict`. You can use `classification_report` ([docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) (Link Placeholder)) to evaluate the predicted labels against the originals.

> **Hint**
> Check the predict documentation [here](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB.predict).

> **Hint**
> Predict needs `X_test` and returns a list of predicted labels.

> **Solution**
```python
from sklearn.metrics import classification_report
ŷ = model.predict(X_test)  # ŷ can also be y_pred for older versions
print(classification_report(y_test, ŷ))
```

*   Compare these predictions to a majority baseline.

> **Hint**
> You can just extract the majority label from y\_train and copy it the length of `y_test`.

> **Hint**
> Something like `ŷ_base = [majority label] * len(y_test)` .

> **Solution**
```python
ŷ = ['Conservative'] * len(y_test)
print(classification_report(y_test, ŷ))
```

### ⭐ Bonus Task: Improve the Pipeline

---

*   Turns out this is a difficult problem. You can use more classifiers (k-NN, SVMs, RandomForests might be good ones, and all are in sklearn), different features, etc. Knock yourself out. 🙂