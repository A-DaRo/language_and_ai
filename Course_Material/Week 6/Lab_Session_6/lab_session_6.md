# Lab Session 6

* **Week:** Week 6
* **Book Chapters:** Empty
* **Slides:** Empty
* **Recordings:** Empty
* **Solutions:** Empty

**(Page Cover Image Placeholder)**

***

> ‼️ **Please read:** this practical features a *lot* of packages. I can’t be sure if these will work for everyone, or that your hardware supports them. See it as a test. If you want to look at the output, I have provided this in the attachments above (under Solutions). ↖️

**Table of Contents**

- [📜 Introduction](#introduction)
    - [📊 Evaluation](#evaluation)
- [💽 Data](#data)
- [👎 Rule-Based Sentiment](#rule-based-sentiment)
    - [⛳ Task 1](#task-1)
- [👍 Transformer-based Sentiment](#transformer-based-sentiment)
    - [⛳ Task 2](#task-2)
- [⚔️ Comparing our Models](#comparing-our-models)
    - [⛳ Task 3](#task-3)
- [👥 Entities](#entities)
    - [⛳ Task 4](#task-4)
    - [🪟 Visualizing](#visualizing)
    - [🚀 Where to?](#where-to)

## 📜 Introduction
***
The last lab session will be a bit more free-form than the others. As we have somewhat depleted the set of algorithms that can do with minimal implementation, and we have to fit two topics into one (Information Extraction and Deep Learning), you’ll mostly be trying some things yourself. We are going to focus on using the libraries we have discussed thus far, and go through how to deploy them for a downstream application: extracting information encoded in text.

### 📊 Evaluation
***
Evaluation is quite important, even if we’re using things ‘out-of-the-box’; e.g., a pre-trained Named Entity Recognition (NER) module. As you will see during this lab, outputs might be quite different, and are certainly not always correct. To get a sense of how well something might perform, we have a few options:

* **Publication**: if there is one, read the associated paper or documentation to see what the model was tested on and the error.
* **Data**: we find a labeled dataset (different from what it was trained on), apply the model to the set, and look at the performance. We might also collect our own (but we’ll have to label it).
* **Qualitative**: we eyeball performance regarding the things that matter to us. This means we manually feed the model some data and look at the actual output, rather than just metrics.

This practical, for Sentiment Analysis (SA), we’re going to evaluate our models using **data**, and we’ll be inspecting them in a **qualitative analysis**. For NER, we’ll mostly do the latter.

## 💽 Data
***
We’ll use the same dataset as before:

```python
import pandas as pd

df = pd.read_csv('https://surfdrive.surf.nl/files/index.php/s/'\
                 '9ROTj6HWRAlvngn/download',
                 na_values=['[deleted]', '[removed]'])\
                 .dropna().reset_index(drop=True)
```

It’s probably good to get a bit more data than 5000 instances for this practical (see Practical 2). The tagging at the end takes about 4 minutes if it’s run single-core (does that by default). If any particular instances are of interest, they will be noted below.

## 👎 Rule-Based Sentiment
***
For the first classifier, we’ll use a ‘sentiment lexicon’. This is an annotated lexicon (vocabulary) with associated polarity scores (positive, neutral, negative). A popular option is VADER:

**(Bookmark: GitHub - cjhutto/vaderSentiment: VADER Sentiment Analysis...)**

You can install VADER via:

```
!python pip install vaderSentiment
```

### ⛳ Task 1
***
* Classify some texts from the `df` with VADER. Usage instructions can be found at the GitHub link. If you don’t want to read the politically-loaded messages, if you exactly copied [this] code, you can use indices `16` and `49`.
    > **Hint**
    >
    > You can select instances from the `df` using `df.iloc[some_index].text`.
    
    > **Hint**
    >
    > You can find usage instructions [here].
    
    > **Solution**
    >
    > ```python
    > from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    > analyzer = SentimentIntensityAnalyzer()
    > print(df.iloc[16].text, analyzer.polarity_scores(df.iloc[16].text))
    > print(df.iloc[49].text, analyzer.polarity_scores(df.iloc[49].text))
    > ```
    > ^This code should give you a dictionary with positive, neutral, negative, and compound scores (aggregated, with negation rules etc. applied, and normalized, see docs).

* Try some of your own sentences an input. Can you get it to fail?
    > **Solution**
    >
    > This receives a slightly positive polarity score:
    >
    > ```python
    > print(analyzer.polarity_scores("This classifier is so great. Not."))
    > ```
    >
    > If you try a bit more you can see that negation does work, but only within a sentence span:
    >
    > ```python
    > print(analyzer.polarity_scores("This classifier is not so great."))
    > ```

## 👍 Transformer-based Sentiment
***
For the next part, we’ll dive right into transformers. As I mentioned in the lectures, HuggingFace has some fantastic open-source libraries. One of those is, quaintly named, `transformers`:

**(Bookmark: 🤗 Transformers)**

The great thing about this library is, other than providing a ton of API to train your own models, that it allows you to upload those models to the [ModelHub]. Once models are uploaded to the ModelHub, everyone can use them with simple string identifiers. This means that we can search for a good sentiment model and apply it too! We first need PyTorch (Deep Learning library). If you don’t have a GPU, on both Mac and Windows, you run:

```
!python -m pip install torch
```

For Linux, see the docs linked in the block below.

> 💡 I wouldn’t recommend this as it takes quite a bit of time, but if you’d like to try your GPU (needs a decent amount of RAM, fairly modern GPU, and CUDA needs to be installed), you can follow the instructions here:
>
> **(Bookmark: CUDA Toolkit 10.2 Download)**
>
> and here:
>
> **(Bookmark: PyTorch)**

Then, you can install:

```
!python -m pip install transformers
```

Once that is (finally) done, we can start looking for models to try out! The syntax for doing sentiment classification using `transformers` is incredibly simplified using their `pipeline`. You can just run:

```python
from transformers import pipeline
model = pipeline("text-classification", "some-model-name")
model("a piece of very positive text, yay!") 
```

If you get a replacement for `"some-model-name"` from the ModelHub, it should be up and running (automatically downloads the model). This is quite the open-source software engineering marvel.

### ⛳ Task 2
***
* Find the most used sentiment model on the Hub (see below), and fill it out in the code above. Run some examples. Can you try to map the `LABEL_#` to pos/neg/neutral?
    **(Bookmark: Models - Hugging Face)**
    
    > **Solution**
    >
    > ```python
    > from transformers import pipeline
    > model = pipeline("text-classification",
    >                  "cardiffnlp/twitter-roberta-base-sentiment")
    > model("a piece of very positive text, yay!")  # LABEL_2 = positive
    > model("a piece of very negative text, boo!")  # LABEL_0 = negative
    > model("a piece of text")                      # LABEL_1 = neutral
    > ```

* Try the examples you tested for VADER. Does this transformer model get them correctly?
    > **Solution**
    >
    > My `"This classifier is so great. Not."` still produces a positive label, although the confidence `score` is quite low.

## ⚔️ Comparing our Models
***
To formally evaluate performance of our respective models, we’ll use `datasets` (as you can see by the name, HuggingFace too):

```
!python -m pip install datasets
```

This, too, has a Hub:

**(Bookmark: Hugging Face - The AI community building the future.)**

We’re going to use `sentiment-140` ([link]) — the website of which is [here]. You can load it like so:

```python
from datasets import load_dataset

dataset = load_dataset("sentiment140")
print(dataset)
```

You will notice this returns a `Dict`-type object.

### ⛳ Task 3
***
* Inspect the dataset. What can you infer from the labels? How do these map to pos/neg/neutral?
    > **Hint**
    >
    > You can access the labels via `dataset['train']['sentiment']`.
    
    > **Hint**
    >
    > For the label mapping, check the website.
    
    > **Solution**
    >
    > There are two labels: 0 for negative, and 4 for positive. The original has 2 for neutral but that is not part of our set. We therefore need to come up with some mapping function.

* Now we’ll have to come up with some mapping. Both of our models also predict neutral, so we’ll have to account for that too (for simplicity, you can always guess one class).
    > **Hint**
    >
    > For VADER you can use `max(some_dict, key=some_dict.get)`. You can remove the `compound`. Alternatively, you can use `compound` and see if it’s < or > 0.
    
    > **Hint**
    >
    > For our transformer, it’s simply mapping the string label to the integers for this dataset.
    
    > **Solution**
    >
    > ```python
    > def map_vader(prediction):
    >     if prediction['compound'] <= 0:
    >         return 0
    >     else:
    >         return 4
    > 
    > def map_trnsf(prediction):
    >     if prediction[0]['label'] == 'LABEL_0':
    >         return 0
    >     else:  # also for neutral
    >         return 4
    > ```

* Evaluate the accuracy of both models on the test set (you can use the training set for more data—we’re not using it for training anyways—but it will take quite some time). You can use `scikit-learn` for the evaluation.
    > **Hints**
    >
    > Probably want to loop through `dataset['test']['text']`, feed the output into the mapping functions, and append them to some list. `dataset['test']['sentiment']` has y_true.
    
    > **Solution**
    >
    > ```python
    > from sklearn.metrics import accuracy_score
    > 
    > ŷ_V = []
    > ŷ_T = []
    > for document in dataset['test']['text']:
    >     ŷ_V.append(map_vader(analyzer.polarity_scores(document)))
    >     ŷ_T.append(map_trnsf(model(document)))
    > 
    > accuracy_score(dataset['test']['sentiment'], ŷ_V)
    > accuracy_score(dataset['test']['sentiment'], ŷ_T)
    > ```

Transformers work a bit better. Not by an incredible margin though (but this is arguably a rather suboptimal evaluation), roughly 5.5%.

> 💡 So, from this we can conclude that simple rule-based models are not all that bad in comparison to state-of-the-art NLP systems. It really depends on the application.

## 👥 Entities
***
To detect mentions of people’s names, we’re going to apply Named Entity Recognition (NER). SpaCy provides entity recognition out of the box:

**(Bookmark: Linguistic Features · spaCy Usage Documentation)**

... and has a cool way of visualizing them:

**(Bookmark: Visualizers · spaCy Usage Documentation)**

If you’re using a notebook, you can replace `serve` with `render`.

> 💡 I wanted to include a transformer here too, but finding i) a similar tag set, with ii) a convenient span mapping, turned out to be quite the undertaking. Feel free to play around with it though! `transformers` has a `pipeline("ner", ...)` for this.

### ⛳ Task 4
***
* Using the information from the links above, apply SpaCy’s NER to `df.iloc[26].text` using displacy and `en_core_web_lg` as dataset for `load`. Does anything stand out to you?
    > **Solution**
    >
    > ```python
    > import spacy
    > from spacy import displacy
    > nlp = spacy.load("en_core_web_lg")
    > doc = nlp(df.iloc[26].text)
    > displacy.render(doc, style="ent")
    > ```
    >
    > SpaCy tags `Trump` at the start of the sentence as an `ORG`. If you call `spacy.explain('ORG')` you can see why that’s a bit odd.

* Extract the tags as a list per document in `df`. If there are no entities, you can leave it empty. If there are entities, only append person names.
    > **Hint**
    >
    > You can loop through `df` using `df.iterrows`, which gives a tuple of (index, series).
    
    > **Solution**
    >
    > ```python
    > import spacy
    > from spacy import displacy
    > nlp = spacy.load("en_core_web_lg")
    > 
    > ents = []
    > for document in df.iterrows():
    >     doc = nlp(document[1].text)
    >     doc_ents = []
    >     for ent in doc.ents:
    >         if ent.label_ == 'PERSON':
    >             doc_ents.append(ent.text)
    >     ents.append(doc_ents)
    > ```

### 🪟 Visualizing
***
So, for the last bit, let’s do something cool with this information we got. First, we’ll map all our entities into a co-occurrence matrix. You can code this yourself, but a hacky way to do this, which gets it straight into a DataFrame, is using `pandas`:

```python
import numpy as np

u = pd.get_dummies(pd.DataFrame(ents), prefix='', prefix_sep='')\
    .groupby(level=0, axis=1).sum()

v = u.T.dot(u)
v.values[(np.r_[:len(v)], ) * 2] = 0
```

Here we convert everything to one-hot vectors, group by name and sum. The co-occurence matrix is then done by transposing and taking the dot product. The last bit requires two libraries: `networkx` and `pyvis`. You can `pip` install them under those names.

```python
from pyvis.network import Network
import networkx as nx

nt = Network('500px', '500px')
nt.from_nx(nx.from_pandas_adjacency(v))
```

You can find the result of this below (will take some time to load):

**(Bookmark: s3-us-west-2.amazonaws.com - Network Visualization Placeholder)**

or render it yourself in a notebook (you'll get weightings too!):

```python
nt = Network('500px', '500px', notebook=True)
nt.from_nx(nx.from_pandas_adjacency(v))
nt.show("graph.html")
```

### 🚀 Where to?
***
Now, you can imagine that we could for example map sentiment weights as edges from entity to entity. This would give us a visualization of ‘persons mentioned in positive/negative’ contexts. However, it’s a graph, so we’d want to disambiguate who the sentiment is directed to (aspect-based sentiment). For that, we’d need a dependency parser. As we have information about both subreddit polarities, we could make a comparison how sentiment is directed between this different groups. The original dataset also includes time, so we could take that into account as well! Then we’d have to compare how SpaCy faces up against transformer models, and if our conclusions are still the same. Free paper ideas—but we’ll leave the lab sessions at that. 🙂