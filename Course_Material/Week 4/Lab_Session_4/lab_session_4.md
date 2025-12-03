# Lab Session 4

[Cover Image Placeholder]

| Property | Value |
| :--- | :--- |
| **Week** | Week 4 |
| **Book Chapters** | Empty |
| **Slides** | Empty |
| **Recordings** | Empty |
| **Solutions** | `lab_4_solutions.py` |

> 💡 Solution functions to implement with the self-grader are given for Task 1.

**Table of Contents**

- [📇 Last of the Count Features: PPMI](#last-of-the-count-features-ppmi)
  - [😎 Getting Contexts, the Cool Way](#getting-contexts-the-cool-way)
    - [⛳ Task 1](#task-1)
- [🛰️ Word Embeddings and `gensim`](#word-embeddings-and-gensim)
  - [🚩 Task 2](#task-2)
- [🧰 Plugging Custom Components into `sklearn`](#plugging-custom-components-into-sklearn)
  - [🚩 Task 3](#task-3)

***

## 📇 Last of the Count Features: PPMI
---
Last week, we have gone over the default setup for running 'vanilla' Machine Learning algorithms using count-based features for classification (including tf\*idf weighting), using Scikit-learn. This week, we are going to move away from simple features, and into the domain of more general representations of language: word embeddings. First, we need some comparison material. Since (weighted) PPMI is fairly doable to implement (and not in Scikit-learn as-is), we'll try to have a go at it. We can code it as a class with three different Counters:

```python
import numpy as np
from collections import Counter


class PPMIVectorizer(object):

    def __init__(self, alpha=0.75, context_window=5):
        self.word_freqs = Counter()
        self.context_freqs = Counter()
        self.co_occ_freqs = Counter()
        self.alpha = alpha
        self.context_window = context_window
        
    def get_counts(self, document):
        return NotImplemented
    
    def get_ppmi(self,):        
        return NotImplemented
```

For the co-occurrence calculations as described in the [Exercise](/7ad3c26fab864011adfc8d269f281beb), we need three things: the total count of a word (`word_freqs`), the total count of a context (`context_freqs`), and the co-occurrence count (`co_occ_freqs`). It's rather expensive to do summations over column/rows in matrices every time, so keeping them in something like a hash map (i.e., a `dict`) likely provides faster look-up (disclaimer: haven't confirmed). Either way, it's nicer to work with things we know than having to mess around with NumPy. 🙂

### 😎 Getting Contexts, the Cool Way
---
The annoying part of this is how to get the context windows in a not-super-convoluted fashion. I will spoil a very neat [one-liner](https://en.wikipedia.org/wiki/One-liner_program) (whaaat? 🤯 yes. — it's one of my favorite things in Python) for this. Here it is:

```python
zip(*[tokens[i:] for i in range(context_window)])
```

No way, right? Sure enough, if you run the following:

```python
tokens = ["hello", "this", "is", "a", "sentence", "and", "that", "is", "all"]
print(list(zip(*[tokens[i:] for i in range(5)])))
```

You will get all 5-grams in the text (🪄). It even returns an empty string if there are no grams of length $n$ (🧙). So how does it work? The slicing part is 'relatively' straight forward:

```python
print([tokens[i:] for i in range(5)])
```

It generates $n$ (here 5) sequences, each of which has a 0 + $n$ starting position, and then the rest of the sequence (`:`). If you try a very large $n$, you can see at some point it runs out of sequence to take $n$th → until the end from. Ok, so what. We now have a list of lists with potentially uneven sequences, but mostly of equal length. Where does the magic happen? This is in the `zip(*` part. What happens here? `zip` is typically used to concatenate two lists so that they are now a combination of tuples:

```python
a = [1, 2, 3]
b = [4, 5, 6]

for x in zip(a, b):
    print(x)
# or
for ai, bi in zip(a, b):
    print(ai, bi)
```

Notice how it does element-wise looping over multiple lists at once? What happens if we let it loop over two 1-off sequences of the same sentence?

```python
sentence = ["this", "is", "a", "sentence"]
part1 = ["this", "is", "a", "sentence"]  # or: sentence[0:]
part2 = ["is", "a", "sentence"]  # or: sentence[1:]

for x in zip(part1, part2):
    print(x)
```

We get bi-grams, etc. (🎊)! If we add another sequence:

```python
sentence = ["this", "is", "a", "sentence"]
part1 = ["this", "is", "a", "sentence"]  # or: sentence[0:]
part2 = ["is", "a", "sentence"]  # or: sentence[1:]
part3 = ["a", "sentence"]  # or: sentence[2:]

for x in zip(part1, part2, part3):
    print(x)
```

We get 3-grams, etc. But wait, they are uneven. It's not giving errors? Nope, `zip` stops iterating if the **smallest** element runs out of things to loop through. So what's the asterisk doing there? In our examples, we had separate lists to `zip`, but here we have a list of lists. The `*` maps them to different 'positions' (as if they were individual variables). A bit hacky, and not very interpretable, but still pretty amazing. Anyway, let's continue. 😄

### ⛳ Task 1
---
* Implement `get_counts` using the one-liner above. As input, it receives `document`: a `list` of tokens (`str`). Fill the `freqs` class attributes with word and context word frequencies, and a `tuple` of (word, context word) in the `co_occ` part. This function should return `self.co_occ_freqs` (only for checking, not functional). To test this, you can use:

```python
from sklearn.datasets import fetch_20newsgroups
X = fetch_20newsgroups(subset='train', categories=['sci.space']).data
```

And:

```python
from numpy.testing import assert_equal  # only have to import this once
from lab_4_solutions import PPMIVectorizer as PPMIVectorizer_solution

try:  # NOTE: make sure your class is in the same file / notebook
    assert_equal(PPMIVectorizer().get_counts(X).most_common(20),
                 PPMIVectorizer_solution().get_counts(X).most_common(20))
    print("Success!")
except AssertionError:
    print("Solution is not identical:")
    print("Your func output:",
          PPMIVectorizer().get_counts(X).most_common(20))
    print("Solutions output:",
          PPMIVectorizer_solution().get_counts(X).most_common(20))
```

<details>
<summary>**Hint**</summary>
While iterating through the one-liner, it's probably good to determine the 'middle' index of the window automatically, so it also works for longer context sizes (rather than just 5).
</details>
<details>
<summary>**Hint**</summary>
Now that you have the middle as an integer, you can either `pop` ([docs](https://www.w3schools.com/python/ref_list_pop.asp)) the middle from the window (need to convert it to a list first), immediately giving you the target word (popped), and the context (rest). Alternatively, you can use slicing (`[:something]` for beginning until something index, and `[something:]` for something index until the end of the sequence.
</details>
<details>
<summary>**Hint**</summary>
Add the context words and word individually to the counters: `counter_name[word] += 1`. Don't forget the `(word, context_word)` tuple.
</details>
<details>
<summary>**Solution**</summary>
```python
def get_counts(self, document):
    for window in zip(*[document[i:] for i in
                        range(self.context_window)]):
        middle = int((len(window) - 1) / 2)
        context = window[:middle] + window[middle + 1:]
        word = window[middle]
        self.word_freqs[word] += 1
        for context_word in context:
            self.context_freqs[context_word] += 1
            self.co_occ_freqs[(word, context_word)] += 1

    return self.co_occ_freqs
```
</details>

* Implement `get_ppmi`. This should use the co-occurrence, word, and context counts we used in the previous step. You want to weight every tuple in the co-occurrence counter using:

$$\text{PPMI}_{\alpha} (w, c) = \max \left(\log_{2} \frac{P(w, c)}{P(w) P_{\alpha}(c)}, 0\right) \quad
P_{\alpha}(c)=\frac{\operatorname{count}(c)^{\alpha}}{\sum_{c} \operatorname{count}(c)^{\alpha}}$$

Where $\alpha$ = `self.alpha`. Once this is done, you can overwrite the counts in `self.co_occ_freqs` with these PPMI values (acts like a `dict`). Finally, `return` the `self.co_occ_freqs`.

> ‼️ This is a pretty difficult task, but it's quite informative to implement it, and very close to how `word2vec` etc. are structured. It's here for those who like a challenge, and to see how incredibly sparse these matrices can get. 🙂 You can test this function similar to the one above.

<details>
<summary>**Hint**</summary>
You need the sum of all context words for $P_{\alpha}(c)$ and the sum of all words for PPMI. Get those first.
</details>
<details>
<summary>**Hint**</summary>
Now you want a loop over the co-occurrence frequencies. If you loop over the `items()`, this will give you `(w, c), wc_freq` (i.e., word, context word, and their co-occurrence frequency). This should be enough to look up all required parts in the class attributes.
</details>
<details>
<summary>**Hint**</summary>
You can get the required probabilities like so:
```python
P_wc = wc_freq / sum_total
P_w = self.word_freqs[w] / sum_total
P_alpha_c = (self.context_freqs[c] ** self.alpha /
             sum_context ** self.alpha)
```
</details>
<details>
<summary>**Solution**</summary>
```python
def get_ppmi(self):        
    sum_context = sum(self.context_freqs.values())
    sum_total = sum(self.word_freqs.values()) + sum_context
    
    for (w, c), wc_freq in self.co_occ_freqs.items():
        P_wc = wc_freq / sum_total
        P_w = self.word_freqs[w] / sum_total
        P_alpha_c = (self.context_freqs[c] ** self.alpha /
                     sum_context ** self.alpha)
        ppmi = max(np.log2(P_wc / (P_w * P_alpha_c)), 0)
        self.co_occ_freqs[(w, c)] = ppmi
        
    return self.co_occ_freqs
```
</details>

If you want to turn these pieces into a fully usable class, it looks like so:

<details>
<summary>**Full Code**</summary>
```python
import numpy as np
from collections import Counter


class PPMIVectorizer(object):

    def __init__(self, alpha=0.75, context_window=5):
        self.word_freqs = Counter()
        self.context_freqs = Counter()
        self.co_occ_freqs = Counter()
        self.alpha = alpha
        self.context_window = context_window
        self.word_cols = {}
        self.context_cols = {}
        
    def get_counts(self, document):
        # your answer
        for window in zip(*[document[i:] for i in
                            range(self.context_window)]):
            middle = int((len(window) - 1) / 2)
            context = window[:middle] + window[middle + 1:]
            word = window[middle]
            self.word_freqs[word] += 1
            for context_word in context:
                self.context_freqs[context_word] += 1
                self.co_occ_freqs[(word, context_word)] += 1

        return self.co_occ_freqs
    
    def get_ppmi(self):        
        # your answer
        sum_context = sum(self.context_freqs.values())
        sum_total = sum(self.word_freqs.values()) + sum_context
        
        for (w, c), wc_freq in self.co_occ_freqs.items():
            P_wc = wc_freq / sum_total
            P_w = self.word_freqs[w] / sum_total
            P_alpha_c = (self.context_freqs[c] ** self.alpha /
                         sum_context ** self.alpha)
            ppmi = max(np.log2(P_wc / (P_w * P_alpha_c)), 0)
            self.co_occ_freqs[(w, c)] = ppmi
            
        return self.co_occ_freqs

    def set_cols(self):
        # this is so we can look up context indices (to fill 0 vec)
        self.context_cols = {w: i for i, w in enumerate(
            sorted(self.context_freqs.keys()))}
		# here we save which words had which contexts (to fill vec)
        self.word_cols = {}
        for w, c in self.co_occ_freqs.keys():
            if not self.word_cols.get(w):
                self.word_cols[w] = []
            self.word_cols[w].append((w, c))
            
    def get_vec(self, word):
        # we initialize every word vector as empty with context length
        vec = [0.0] * len(self.context_cols)
        try:
            for (w, c) in self.word_cols[word]:
                # fill each context column index with ppmi from co_occ
                vec[self.context_cols[c]] = self.co_occ_freqs[(w, c)]
        except KeyError:  # if word is out of vocabulary we do nothing
            pass
        return vec

    def check_tokens(self, document):
        # if not tokenized, do arbitrarily
        if not isinstance(document, list):
            return document.split(' ')
        else:
            return document
        
    def fit(self, X):
        for document in X:
            self.get_counts(self.check_tokens(document))
        self.get_ppmi()
        self.set_cols()
            
    def transform(self, X):
        # this sums the PPMI embeddings per document (can also do mean)
        return [np.sum([self.get_vec(w) for w in
                     self.check_tokens(document)]) for document in X]
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
```
</details>

***

## 🛰️ Word Embeddings and `gensim`
---
Ok, back to practical stuff. This lab session, we will introduce the last tool (before we cover Deep Learning libraries)—`gensim`. Gensim is a collection well-implemented (C, so fast too) algorithms to do semantic representations of language: tf\*idf, LSA, topic models, word2vec, etc.

[Bookmark: Gensim: topic modelling for humans (https://radimrehurek.com/gensim/index.html)]

To install, you can do either of:

```bash
!python -m pip install --upgrade gensim
# or:
!conda install -c conda-forge gensim
```

> 💡 If this doesn't work from Jupyter, you can use Anaconda Prompt or another terminal.

Gensim also comes with a data [downloader](https://radimrehurek.com/gensim/downloader.html)! 🥳 The available datasets can be found here:

[Bookmark: GitHub - RaRe-Technologies/gensim-data: Data repository for pretrained NLP models and NLP corpora. (https://github.com/RaRe-Technologies/gensim-data)]

This practical, we'll use the Fake News dataset (‼️**contains fake news only, no 'legit' news**) to delve through word relations.

### 🚩 Task 2
---
* Use the `downloader` to load the fake news data and train word2vec on it.

<details>
<summary>‼️ Callout icon</summary>
Somehow the Gensim API for word2vec doesn't immediately process the dataset (words will not be in vocab, try it for fun). You can use this line to convert it to a list of documents that word2vec **does** accept:

```python
[simple_preprocess(t['text']) for t in dataset]
```

If you want to read the data you can loop through the object that `gensim` returns.
</details>

<details>
<summary>**Hint**</summary>
See the [documentation](https://radimrehurek.com/gensim/downloader.html) for an example. The ID for the fake news dataset is `fake-news` (see GitHub page).
</details>
<details>
<summary>**Solution**</summary>
```python
import gensim.downloader as api
from gensim.models import Word2Vec


dataset = api.load("fake-news")
model = Word2Vec([simple_preprocess(t['text']) for t in dataset])
```
</details>

* Let's try to extract some interesting word-associations like we did in [Lab 2](/b86e02fa993d49d487515d2f9f3895e6), but now using nicer representations. Read the word2vec [docs](https://radimrehurek.com/gensim/models/word2vec.html#module-gensim.models.word2vec) and [examples](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#sphx-glr-auto-examples-tutorials-run-word2vec-py). Extract `top_n` words for multiple (related) keywords of your choice (keep the task at hand in mind). Convert the output into a `list` of `dicts` (where words are keys, similarities values). Load this into a pandas DataFrame, and drop empty **columns** (it's rows by default).

<details>
<summary>**Hint**</summary>
The similarities can be obtained from `model.wv.most_similar`.
</details>
<details>
<summary>**Hint**</summary>
Pandas can directly load the dictionary into a DataFrame.
</details>
<details>
<summary>**Hint**</summary>
You can change the `axis` in `dropna` to make it drop column-wise.
</details>
<details>
<summary>**Solution**</summary>
```python
import pandas as pd
sims = [{w: sim for w, sim in model.wv.most_similar(word, topn=100)}
        for word in ["trump", "obama"]]

df = pd.DataFrame(sims)
df = df.dropna(axis=1)
```
⭐
</details>

* Run the above two pieces of code a few times and compare the output. What do you notice?

<details>
<summary>**Hint**</summary>
Look at the similarity weights for the same words.
</details>
<details>
<summary>**Solution**</summary>
They change per run! Word2vec not deterministic (so not reproducible) out of the box. This can be fixed by providing the function with a custom hash (not a great implementation here) and setting `workers` to 1 (multi-threading messes up single-program-determinism):

```python
def pseudo_hash(astring):
    return ord(astring[0])

dataset = api.load("fake-news")
model = Word2Vec([simple_preprocess(t['text']) for t in dataset],
                 workers=1, hashfxn=pseudo_hash)
```
> ‼️ This makes things really slow, so it might take a while to train.
</details>

---
You can find more things to try (such as sentence similarity, and analogies) [here](https://radimrehurek.com/gensim/models/keyedvectors.html). They are a great way to explore your word representations, so do give them a go. Keep in mind that word2vec is a **model**; while it does observe regularities, observations might be sensitive to parameter settings. Some fun work I did on Dutch word embeddings and dialects can be found here:

[Bookmark: Evaluating Unsupervised Dutch Word Embeddings as a Linguistic Resource (https://arxiv.org/abs/1607.00225)]

## 🧰 Plugging Custom Components into `sklearn`
---
So for the last part, let's see if word embeddings improve performance on the task we have been doing in all the previous lab sessions. For this, we need to turn it into a `Vectorizer`. `gensim` used to provide this functionality, but not anymore. No worries, we can just make our own class:

```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec

class Word2VecTransformer(TransformerMixin,BaseEstimator):
    def __init__(self):
        self.w2v = None

    def fit(self, X, y=None):
        self.w2v = Word2Vec([simple_preprocess(doc) for doc in X])
        return self

    def transform(self, X):
        vec_X = []
        for doc in X:
            vec = []
            for token in simple_preprocess(doc):
                try:
                    vec.append(self.w2v.wv[token])
                except KeyError:
                    pass
            if not vec:
                vec.append(self.w2v.wv['the'])
            vec_X.append(np.mean(vec, axis=1))
        return vec_X
    
# NOTE: Need to implement fit_transform for simplicity if not using Pipeline
# def fit_transform(self, X):
#     self.fit(X)
#     return self.transform(X)
```

Here, we inherit some base functionality from scikit-learn that will provide the correct class attributes to be compatible with their API (`TransformerMixin` for vectorizer classes, `BaseEstimator` for standard objects). Note that during `fit`, we train word2vec on our data. `transform` then converts them to vectors, and finally takes the mean over the vector dimensions (`axis=1`) to return it as one vector representation. We might then for example train a Multi-Layer Perceptron (Feedforward Network) on these embeddings:

```python
from sklearn.neural_network import MLPClassifier
w2v = Word2VecTransformer()
X = w2v.fit_transform([t['text'] for t in dataset])
# split the data into train and test
# NOTE: we don't have labels, this is just some pseudo-code example
clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
```

### 🚩 Task 3
---
* This is an open-ended task. Try plugging in the above word2vec class into the experiments of last week's lab.