# 💾 Lab Session 2

**Page Properties**

| Property | Value |
| :--- | :--- |
| **Week** | Week 2 |
| **Book Chapters** | Empty |
| **Slides** | Empty |
| **Recordings** | Empty |
| **Solutions** | [Attachment: lab\_2\_solutions.py] |

---

> 💡 For this notebook, most tasks are a few lines, and all library-based. Hence, for this lab, there will only be grader scripts for the bonus question of Task 3, and Task 4.

**Table of Contents**

*   💽 Data Preliminaries
    *   🚩 Task 1
*   🎨 Descriptive Statistics
    *   🚩 Task 2
*   ✂️ Tokens
    *   🚩 Task 3
    *   🧠 Estimating Human Word Recognition from Data
*   💥 Preprocessing
    *   🚩 Task 4
    *   ✅ Testing the Class
*   ⭐ Bonus: Minimum Edit Distance

## 💽 Data Preliminaries

---

In this lab session, we will be working with Reddit data from [Pushshift](https://files.pushshift.io/reddit/). I have provided a URL interface which you can use to download the data in the size that you'd like (taking into account not everyone has time/computing power available to run on larger datasets). The safest way is the `right click link → save link as` option, but you can also open the link and `save page as` from there (might crash your browser).

For this week, the data can be found here:

> ⚠️ If onyx is down, please click the mirror link instead.

| Docs | Size | Link | Mirror |
| :--- | :--- | :--- | :--- |
| 5k | 1MB | <https://onyx.uvt.nl/grabber/politics/reddit-5000.csv> | <https://surfdrive.surf.nl/s/9ROTj6HWRAlvngn/download> |
| 10k | 2MB | <https://onyx.uvt.nl/grabber/politics/reddit-10000.csv> | <https://surfdrive.surf.nl/s/9ROTj6HWRAlvngn/download> |
| 50k | 11MB | <https://onyx.uvt.nl/grabber/politics/reddit-50000.csv> | <https://surfdrive.surf.nl/s/jBTUfgZQ5lXmZn9/download> |
| 100k (max) | 22MB | <https://onyx.uvt.nl/grabber/politics/reddit-100000.csv> | <https://surfdrive.surf.nl/s/4UiBcBPWoeBKuUb/download> |

> 💡 You can download any size you like (max 100,000—more is available, but let's not overload the server) by replacing `n` in `reddit-n.csv` with an integer.

Little background information: this is a filtered version of the Reddit JSONs which are stored in a BSON format within a [MongoDB](https://www.mongodb.com/) instance. Specifically, they are from political subreddits (r/Conservative, r/Liberal, r/democrats, r/Republican). A single (compressed) instance looks like so:

```
{'_id': ObjectId("608b10c285ff77323393b894"),
 'stickied': false,
 'link_id': 't3_5afnej',
 'body': "I think you are missing the point of being a Conservative. That is to conserve good ideas instead of throwing them away. \n\nAgain, you can accept it and adapt but you have no right to complain when things turn out worse and not the way you would like.\n\nSeems like you shouldn't really be involved or care about politics if you just allow the current climate to shape your view.",
 'controversiality': null,
 'subreddit': 'Conservative',
 'author': 'Latinenthusiast',
 'author_flair_css_class': null,
 'subreddit_id': 't5_2qh6p',
 'author_flair_text': null,
 'score': 1,
 'parent_id': 't1_d9g4zob',
 'created_utc': 1477958454,
 'edited': 'false',
 'id': 'd9g53pp',
 'retrieved_on': 1481097679,
 'distinguished': null,
 'gilded': 0}
```

What the download links provide, is a `.csv` format with just the `subreddit` and `body` fields. Only one preprocessing step has been conducted thus far: replacing `'\n'` (newline) with `' '` (space). If you'd code this up yourself, it'd look something like:

```python
import json

data = []
for line in open('reddit-dump-file.json'):
		obj = json.loads(line)  # these are typically JSON objects per line
		data.append((obj['subreddit'], obj['body'].replace('\n', ' ')))
return data  # this list of tuples can be directly loaded into e.g. pandas
```

> ‼️ **Data disclaimer**: before working with the data, please be aware that this is web data. Hence, I can't give a blanket list of trigger warnings for everything potentially discussed on these Subreddits; they talk about a variety of (sensitive) political topics, under 'Subreddit rules' promoting civility. Racism, antisemitism, misogyny, misandry, or other forms of hate speech are generally banned on all Subreddits (if moderators / flaggers have spotted this content).
>
> Viewing the first 7 (`.head(7)`) lines should give you enough content to do the lab assignment with strictly neutral messages.

### 🚩 Task 1

---

*   Import this file (can even do it directly by passing the URL if the library supports that) with, e.g., [Pandas](https://pandas.pydata.org/) (using [read\_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html), supports URLs).
*   Print/output the first 7 lines (see [here]()); there are 'missing values' specific to Reddit. Identify these values and remove them (you can do that with `read_csv` using `na_values` ([docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html)), and `drop_na` ([docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html)), or using a script of your choice).

    > ‼️ **Note:** when dropping the missing values, update `head(7)` to `head(5)`.

    **Solution**

    ```python
    import pandas as pd

    # mirror https://surfdrive.surf.nl/files/index.php/s/9ROTj6HWRAlvngn/download
    df = pd.read_csv('https://onyx.uvt.nl/grabber/politics/reddit-5000.csv',
                     na_values=['[deleted]', '[removed]']).dropna()
    df.head(5)
    ```

> 💡 **Quick note**: see how DataFrames are just a class? Here, `read_csv` is basically just a wrapper to load the data file, and return an initialized DataFrame (`__init__` for DataFrame requires data at the very least). In previous versions of Pandas, it was part of the DataFrame class (`DataFrame.from_csv`). Once we have this `df` class, we can call functions on its own data representation; e.g., show me the top-$n$ (`.head()`) or remove the NaNs (`.dropna()`). Typically, Pandas returns an updated version of `self` — but some functionality can be done 'in place' (so you don't have to rebind it to a variable). If you print `df.__dict__` you can see the internal representation of the class attributes.

## 🎨 Descriptive Statistics

---

Before we dive into more complex analyses. Let's first extract some surface-form descriptive statistics. The most basic one would be looking at our label distribution (which Subreddits are represented in what quantity). Furthermore, the data is from 2019, when the Trump administration was still in full swing. It might be interesting having a quick look at the mention of certain figures in the data!

### 🚩 Task 2

---

*   Which of the subreddits is most represented in the dataset? Is this expected? You can use `value_counts` ([docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.value_counts.html)) for this.

    **Hint**

    You can apply `value_counts` on a column (`df.column_name.value_counts()`).

    **Hint**

    You can visit the subreddits on [reddit.com](http://reddit.com) to get member estimates.

    **Solution**

    ```python
    df.subreddit.value_counts()
    ```

    Shows the distribution. It depends on your data split, what the most common one is. For 5000, it's Conservative. Based on the subreddit's members the 'true' user distribution is as follows:

    | **Subreddit** | **Members** | **Perc. (rounded)** | **Observed** |
    | :--- | :--- | :--- | :--- |
    | r/Conservative | 885k | 59.36% | ? (your data) |
    | r/Liberal | 109k | 7.31% | ? |
    | r/democrats | 322k | 22.27% | ? |
    | r/Republican | 175k | 11.74% | ? |
    | **Total** | 1491k | 100% | |

If you run:

```python
df.subreddit.value_counts(normalize=True)
```

These can directly be compared. This is posts vs. members (depends on activity of subreddit, nr. of posts, etc.), so exact correlation cannot be assumed.

For the second part, let's find mentions of Trump. There are at least two ways to go about it. Both use Pandas' `str` operators that work on [Series](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html) (i.e., the columns).

> ⚠️ **Note**: `append` has been [deprecated](https://github.com/pandas-dev/pandas/issues/35407) as of Pandas version 1.4 and `concat` should be used instead.

*   For one, you can use `count` ([docs](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.count.html)) to create a column, and `append` ([docs](https://pandas.pydata.org/pandas-docs/version/1.4/reference/api/pandas.DataFrame.append.html)) it to the original dataframe (or make a new one by using `concat` ([docs](https://pandas.pydata.org/docs/reference/api/pandas.concat.html))). Then, you can use `value_counts` like before.

    **Hint**

    `df.text.str.count('Trump')` returns a Series (an array, or a single column).

    **Hint**

    To concatenate, put two Series in a list; this makes a new dataframe. This has to be done along the vertical axis (`axis=1`). You can rename series for clarity (`my_series.name = "new_name`).

    **Hint**

    Note that `count` can be more than one per document. If you use them as-is, the `value_counts` will show counts per count, so to say. You can filter this by doing e.g. `counts >= 1` (which will break it down into no mention and mention).

    **Solution**

    ```python
    counts = df.text.str.count('Trump')
    counts.name = "counts"
    count_df = pd.concat([df.subreddit, counts >= 1], axis=1)
    print(count_df.value_counts(sort=False))  # not sorted is clearer (imo)
    ```

*   A bit more fancy, if you dare, is to use a `pivot_table` ([docs](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pivot_table.html)). For this, you need to pass `.str.count` as a `lambda` expression in `aggfunc` (🥳).

    **Hint**

    `text` goes in values, `subreddit` can go in either `index` or `columns`

    **Hint**

    The base for `aggfunc` is `lambda x: x.str.count('Trump')` but it needs to be aggregated.

    **Hint**

    You can use `sum` in the lambda expression, that will give more accurate counts than the solution above. To get the same, you need to incorporate `>= 1` too.

    **Solution**

    ```python
    df.pivot_table(values='text', index='subreddit',  # >= 1 is optional
               aggfunc=lambda x: sum(x.str.count('Trump') >= 1))
    ```

You are obviously welcome to try more. The main motivation is typically doing sanity checks of the data: does it meet our expectations. [Here](https://pandas.pydata.org/pandas-docs/stable/search.html?q=str) you can find more utilities to extract information from surface-level strings (`findall`, `len`), and operations to process the data (`casefold`, `replace`, `slice`, `join`). It's good to familiarize yourself with these. They can be useful to quickly run over large sets of data (Pandas is relatively fast with this), and even to extract some manual features for, e.g., classification ('is <certain person> mentioned in the document'). Before we go into further sanitization, we have a few more options to inspect our data samples; for that, we need to tokenize.

## ✂️ Tokens

---

As before, we'll have to identify the units of our text sequences (tokens). In the last lab session, we designed custom tokenizers using a rudimentary white space split. That worked well because our data was artificial. This is real data, and as you can hopefully see, white spaces aren't going to cut it. First, we'll try using RegEx (see this week's [exercises]() for primers and practice). You can use `re` in a script, or use Pandas' `str` operators for regex (`findall`, `replace`).

### 🚩 Task 3

---

*   Use a regular expression to find all tokens (words and characters) and split them by space.

    **Hint**

    Use `df.text.str.findall` to get a list of token matches.

    **Hint**

    You probably want to use `[^\w\s]` to match punctuation. Note that using `|` is greedy (matches the primary group first).

    **Hint**

    You can concatenate the words back to a string using `join` (this is optional).

    **Solution**

    ```python
    tokens = df.text.str.findall('\w+|[^\w\s]')  # optional .str.join(' ')
    print(tokens)
    ```

*   Use a `Counter` ([docs](https://docs.python.org/3/library/collections.html#collections.Counter), also see previous lab) to get the counts for these tokens. You can use tokens as an iterable of lists (don't use `join` in this case). Put the top 10,000 tokens back into a DataFrame (accepts lists of tuples), and `plot` ([docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html)) them using a `loglog` scale. What do we see here (discussed in the lecture 🙂)?

    **Tips on Plotting (Jupyter or IDE)**

    You can invoke `%matplotlib inline` in Jupyter (somewhere in the notebook, preferably at the top) to automatically show plots. If you are using an IDE, you have the option to `import matplotlib.pyplot as plt`, store the result of any plot further down in an `ax` variable and use `plt.show()` at the end.

    **Hint**

    You can instantiate counters with a list, and append counters to each other, like so:

    ```python
    from collections import Counter

    c = Counter()
    for a_list in some_list_of_lists:
    		c += Counter(a_list)
    ```

    **Hint**

    `pd.DataFrame(...)` is compatible with the output of `most_common`. You can call `plot` on this object and provide `loglog` as an argument.

    **Solution**

    ```python
    from collections import Counter

    tokens = df.text.str.findall('\w+|[^\w\s]')  # from previous part
    c = Counter()
    for x in tokens:
        c += Counter(x)
    ```

    Part 2 Jupyter:

    ```python
    %matplotlib inline 
    pd.DataFrame(c.most_common(10000),
                 columns=['word', 'freq']).plot(loglog=True)
    ```

    Part 2 IDE:

    ```python
    import matplotlib.pyplot as plt
    ax = pd.DataFrame(c.most_common(10000),
                      columns=['word', 'freq']).plot(loglog=True)
    plt.show()
    ```

    The result is a Zipfian (or Power Law) curve.

*   ⭐ **Bonus**: write a function `top_words_per_subreddit` that takes as input a `list` of (unique) label values (`str`, should be 4), and a tokenized `DataFrame` (you can overwrite the old column using `df['text'] = ...`). It should `return` a `dict` where the `key` is the subreddit (`str`), and the `value` is a `set` with which words from the subreddit's respective top 5000's were unique to that subreddit (so requires [set](https://docs.python.org/3/tutorial/datastructures.html#sets) operations). The self-grader solution is provided (see top to download). If you forgot about the grader script, an explanation is [here]().

    **Hint**

    An easy way to get the unique labels is with `df.subreddit.unique()`.

    **Hint**

    You can use `df.iterrows` which gives back an `index, row` tuple. From `row`, you can access `row.subreddit` and `row.text`.

    **Hint**

    Make a `dict` with label keys, and empty counters. In a loop, add to the counters. Third loop (also not nested), extract the most common terms (make sure you get the words only, in a set).

    **Hint**

    Now you should have a `top_words` dict, you should write two nested loops to make one big union set (without the current one), then you can subtract them.

    **Solution**

    ```python
    def top_words_per_subreddit(labels, df):
        counters = {label: Counter() for label in labels}
        for ix, row in tqdm(df.iterrows()):
            counters[row.label] += Counter(row.text)
        # NOTE: gets too messy if we do this in the loop below
        top_words = {label: set([w for w, i in c.most_common(5000)])
                     for label, c in counters.items()}
        unique_words = {}
        for i, set_i in top_words.items():
            master_set = set()
            for j, set_j in top_words.items():
                if i == j:
                    continue
                master_set = master_set.union(set_j)
            unique_words[i] = set_i - master_set
                
        return unique_words
    ```

    For the test function:

    ```python
    from numpy.testing import assert_equal  # only have to import this once
    from lab_2_solutions import top_words_per_subreddit as \
        top_words_per_subreddit_solution

    try:  # NOTE: make sure your function is in the same file / notebook
        assert_equal(top_words_per_subreddit(
                        df.subreddit.unique(), df),
                     top_words_per_subreddit_solution(
                        df.subreddit.unique(), df))
        print("Success!")
    except AssertionError:
        print("Solution is not identical:")
        print("Your func output:", tokenize(documents))
        print("Solutions output:",
              top_words_per_subreddit_solution(df.subreddit.unique(), df))
    ```

## 🧠 Estimating Human Word Recognition from Data

---

There is some interesting CogSci work relating word frequencies in popular media (like movies) to explaining vocabularies of subgroups of non-native language speakers. Moreover, these frequency ranks estimated from corpora are good predictors of word recognition response time experiments. For example, you are provided with word-by-word slides of non-words and complex words. You are tasked to press a button as quickly as possible if you recognize it as being a word. Papers below:

[Bookmark: Word knowledge in the crowd: Measuring vocabulary size and word prevalence in a massive online experiment - Emmanuel Keuleers, Michaël Stevens, Paweł Mandera, Marc Brysbaert, 2015 - https://journals.sagepub.com/doi/full/10.1080/17470218.2015.1022560]

[Bookmark: Subtlex-UK: A New and Improved Word Frequency Database for British English - Walter J. B. van Heuven, Pawel Mandera, Emmanuel Keuleers, Marc Brysbaert, 2014 - https://journals.sagepub.com/doi/full/10.1080/17470218.2013.850521]

## 💥 Preprocessing

---

Ok, so we've mostly worked with Pandas and a bit with RegEx. Let's get spaCy up and running.

> ‼️ Please check spaCy’s install page for install instructions if you have a setup that doesn’t match the ones below (e.g. a Mac with an M1 chip is already an issue). See below:

[Bookmark: Install spaCy · spaCy Usage Documentation - https://spacy.io/usage]

If you are using Jupyter, you can run:

```python
# assuming python 3 here, try without 3 if
# this does not work, or, drop the first line
!python3 -m pip install -U pip setuptools wheel
!python3 -m pip install -U spacy
!python3 -m spacy download en_core_web_sm
```

If all else fails, and you are using Anaconda, you can try the following in Anaconda Prompt (comes with it by default):

```python
python -m pip install spacy
python -m spacy download en_core_web_sm
```

Otherwise, instructions are available [here](https://spacy.io/usage). As shown in the previous lecture, spaCy allows for a very simple interface to apply advanced NLP models on your data. Some examples:

```python
import spacy
nlp = spacy.load('en_core_web_sm')

print([token.text for token in nlp("This is a text with some words.")])
```

A more comprehensive tutorial for spaCy (takes about an hour and a half or so—you can take it if you have the time, it's mostly not required for the current lab session), is available here:

[Bookmark: Chapter 1: Finding words, phrases, names and concepts · Advanced NLP with spaCy - https://course.spacy.io/en/chapter1]

We will be using spaCy more often in this course, but for now we'll keep things simple. We're going to set up for a prediction task. Namely, the Subreddit of our data given the text. To do so, we'll use a few different flavors of preprocessing. Once we have those, we can gauge their respective impact on classifier performance. First, we have our original dataframe, and tokenization using RegEx. I've implemented these in the following class:

```python
import spacy

class Preprocessor(object):
    
    def __init__(self, method='regex'):
        self.nlp = spacy.load('en_core_web_sm')
        if method == 'regex':
            self.proc = self.regex_tokens
        elif method == 'spacy':
            self.proc = self.spacy_tokens
        elif method == 'spacy-lemma':
            self.proc = self.spacy_lemma
        
    def regex_tokens(self, X):
        return X.str.findall('\w+|[^\w\s]').to_list()
    
    def spacy_tokens(self):
        return NotImplemented
    
    def spacy_lemma(self):
        return NotImplemented  
    
    def transform(self, X):
        return self.proc(X)

# mirror: https://surfdrive.surf.nl/files/index.php/s/9ROTj6HWRAlvngn/download
df = pd.read_csv('https://onyx.uvt.nl/grabber/politics/reddit-5000.csv',
                 na_values=['[deleted]', '[removed]']).dropna()  
proc = Preprocessor(method='regex')
proc.transform(df.text)
```

### 🚩 Task 4

---

*   As you can see, there are two `NotImplemented` functions using spaCy. One using regular tokens, the other getting lemmas. Implement these yourself.

    **Hint**

    You still need to add `X` to the respective functions.

    **Hint**

    For lemmas, spaCy has a `lemma_` token attribute.

    **Solution**

    ```python
    import spacy

    class Preprocessor(object):
        
        def __init__(self, method='regex'):
            self.nlp = spacy.load('en_core_web_sm')
            if method == 'regex':
                self.proc = self.regex_tokens
            elif method == 'spacy':
                self.proc = self.spacy_tokens
            elif method == 'spacy-lemma':
                self.proc = self.spacy_lemma
            
        def regex_tokens(self, X):
            return X.str.findall('\w+|[^\w\s]').to_list()
        
        def spacy_tokens(self, X):
            return [[token.text for token in nlp(text)] for text in X]
        
        def spacy_lemma(self, X):
            return [[token.lemma_ for token in nlp(text)] for text in X]  
        
        def transform(self, X):
            return self.proc(X)
    ```

### ✅ Testing the Class

---

Some minor alterations are required from the first script:

```python
from numpy.testing import assert_equal  # only have to import this once
from lab_2_solutions import Preprocessor as Preprocessor_solution

method_test = 'regex'  # NOTE: change this!
try:  # NOTE: make sure your class is in the same file / notebook
    assert_equal(Preprocessor(method=method_test).transform(df.text),
                 Preprocessor_solution(method=method_test).transform(df.text))
    print("Success!")
except AssertionError:
    print("Solution is not identical:")
    print("Your func output:",
          Preprocessor(method=method_test).transform(df.text))
    print("Solutions output:",
          Preprocessor_solution(method=method_test).transform(df.text))
```

Please note that spaCy can take quite a while as it runs single-threaded by default, depending on how big your data is. In the next practical, we'll be implementing these preprocessing steps for actual classification!

## ⭐ Bonus: Minimum Edit Distance

---

If you still have time (🙂), you can try implementing the minimum edit distance algorithm shown in the lectures. The algorithm is one step (constructing the matrix), though it is missing a backtrace operation to find the shortest sequence of actions to be performed. There are suggestions in the book to implement this. No solution (you can use the [exercises]() to check input); it's just for fun.