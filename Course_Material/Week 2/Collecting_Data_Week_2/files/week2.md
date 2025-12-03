# LANGUAGE & AI: COLLECTING DATA

[IMAGE: Large pile of multicolored LEGO bricks.]

Chris Emmery
Department of Cognitive Science & AI
Tilburg University

@cmry • @_cmry • @cmry • cmry.github.io

---

# COLLECTING LANGUAGE / TEXT

[IMAGE: Stack of two books ("STUNNING CSS3" and "JAVASCRIPT & JQUERY") with a white smartphone resting on top.]

---

# RECAP LAST LECTURE

[IMAGE: Recycling symbol icon]

*   We looked at some interesting language **features**.
*   We discussed some rudimentary ways of converting documents into **vectors**.
*   Using this space, we did some **similarity** calculations.

---

# PIPELINE

[IMAGE: Diagram illustrating a text processing pipeline: Data sources (CSV/Database) -> Preliminary formatting -> Tokenization (with Stemming, Stopword removal) -> Advanced analytics -> Visualization.]

> Image by Daniel Harris.

---

# WHAT DOES TEXT DATA LOOK LIKE?

[IMAGE: Monitor icon]

```html
<div data-test-id="post-content">
<div class="_23h0-EcaBUorIHC-JZyh6J" style="width: 40px; border-left: 4px s
<div class="_1E9mcoVn4MYnuBQSVDt1gC" id="vote-arrows-t3_apmsqk">
<button aria-label="upvote" aria-pressed="false" class="voteButton
<span class="_2q7IQ0BUOWEEZoeAxN555e _3SUsITjKNQ7Tp0Wi2jGxIM qù
</button>
<div class=" 1rZYMD_4xY3gRcSS3p8ODO 3a2ZHWaih05DgAOtvu6cIo " style
116k
</div>
<button aria-label="downvote" aria-pressed="false" class="voteButtc
<span class="_1iKd82bq_nqObFvSH1iC_Q Q0BxYHtCOJ_rNSPJMU2Y7 2fe
</button>
</div>
</div>
<div class="_14-YvdFiW5iVvfe5wdgmET">
<div class=" 2dr 3pZUCk8KfJ-x0txT 1">
<a data-click-id="subreddit" class=" 3ryJoIoycVkA88fy40qNJc" href='
<img style="background-color: rgb(87, 196, 199);" alt="Subreddi
</a>
</div>
<div class="cZPZhMe-UCZ8htPodMyJ5">
\λήτε ~1~~~" 2c+z~11m0ar7штерDin? IIA TOJ3_)CVC+IDADMV+0"/
```

---

# APIS & DUMPS

[IMAGE: Printer icon]

*   Datasets
*   Twitter
*   Wikipedia
*   Reddit

---

# STRUCTURED OBJECTS

[IMAGE: Folder icon]

e.g., JSON, XML

```json
{
"data": [
{
"id": "1212092628029698048",
"text": "We believe the best future version of our API will come fr",
"possibly_sensitive": false,
"referenced_tweets": [
{
"type": "replied_to",
"id": "1212092627178287104"
}
],
"entities": {
"urls": [
{
"start": 222,
"end": 245,
"url": "https://t.co/yvxdK6a0o2",
"expanded_url": "https://twitter.com/LovesNandos/status",
"display_url": "pic.twitter.com/yvxdK6a0o2"
}
]
}
```

---

# DOWN TO THE TEXT

[IMAGE: Spool of thread icon]

[IMAGE: Two examples of social media posts (Tweets/Reddit comments). The first is "These "maxi this maxi that" posts are sus af" and the second is "LMFFAOOOO HEELLPPP CUZ I TOLD HIM I DONT WANT TO GO OUT WITH HE CALLED ME دلوعه".]

---

# GARBAGE IN, GARBAGE OUT

[IMAGE: Trash can icon]

*   Can apply to many levels: data collection, sampling, sanitization, etc.
*   For language: user-generated text is creative, which makes it a nightmare to work with.
*   **Consider**: typos have a tremendous effect on the size of your vocabulary, and the representation of your documents (thus the similarity quality).

---

# LANGUAGE VARIATION

[IMAGE: Puzzle piece icon]

*   abbreviations, acronyms
*   capitalization
*   character flooding
*   concatenations
*   emoticons
*   dialect, slang
*   typos

---

# FINDING & FIXING ERRORS

[IMAGE: Pencil writing icon]

[IMAGE: Close-up of a blue Bic Wite-Out correction tape dispenser.]

---

# REGULAR EXPRESSIONS

[IMAGE: Three chickens icon]

*   A language to define string patterns.
*   Available in many programming languages (in Python `re`).
*   Can not only be used to **find** patterns but also to **replace** them.

```python
import re
text = "You there, did you what you were looking for?"
patt = re.compile("you")
for match in patt.finditer(text):
    print(match)
```

```
<re.match object; span=(15, 18), match='you'>
<re.match object; span=(24, 27), match='you'>
```

---

# REGEX: DISJUNCTIONS

[IMAGE: Regular expression icon (X)]

> this OR that

```python
text = "You there, did you what you were looking for?"
regex_find("[Yy]ou", text)
```

```
<re.match object; span=(0, 3), match='You'>
<re.match object; span=(15, 18), match='you'>
<re.match object; span=(24, 27), match='you'>
```

---

# REGEX: DISJUNCTIONS

[IMAGE: Regular expression icon (X)]

```python
text = "You there, did you what you were looking for?"
regex_find("[Yyou]", text)
```

```
<re.match object; span=(0, 1), match='Y'>
<re.match object; span=(1, 2), match='o'>
<re.match object; span=(2, 3), match='u'>
<re.match object; span=(15, 16), match='y'>
<re.match object; span=(16, 17), match='o'>
<re.match object; span=(17, 18), match='u'>
<re.match object; span=(24, 25), match='y'>
<re.match object; span=(25, 26), match='o'>
<re.match object; span=(26, 27), match='u'>
<re.match object; span=(34, 35), match='o'>
<re.match object; span=(35, 36), match='o'>
<re.match object; span=(42, 43), match='o'>
```

---

# REGEX: NEGATION

[IMAGE: Red X icon]

```python
text = "You there, did you what you were looking for?"
regex_find("[^a-z]", text)
```

```
<re.match object; span=(0, 1), match='Y'>
<re.match object; span=(3, 4), match=' '>
<re.match object; span=(9, 10), match=','>
<re.match object; span=(10, 11), match=' '>
<re.match object; span=(14, 15), match=' '>
<re.match object; span=(18, 19), match=' '>
<re.match object; span=(23, 24), match=' '>
<re.match object; span=(27, 28), match=' '>
<re.match object; span=(32, 33), match=' '>
<re.match object; span=(40, 41), match=' '>
<re.match object; span=(44, 45), match='?'>
```

---

# REGEX: KLEENE EXPRESSIONS

[IMAGE: Gold star icon]

[IMAGE: Black and white close-up of Stephen Kleene, an American mathematician.]

---

# REGEX: KLEENE EXPRESSIONS

[IMAGE: Gold star icon]

```python
text = "You there, did you what you were looking for?"
regex_find("o[a-z]*", text)
```

```
<re.match object; span=(1, 3), match='ou'>
<re.match object; span=(16, 18), match='ou'>
<re.match object; span=(25, 27), match='ou'>
<re.match object; span=(34, 40), match='ooking'>
<re.match object; span=(42, 44), match='or'>
```

---

# REGEX: KLEENE EXPRESSIONS

[IMAGE: Gold star icon]

```python
text = "You there, did you what you were looking for?"
regex_find("u[a-z]+", text)
```

---

# REGEX: KLEENE EXPRESSIONS

[IMAGE: Gold star icon]

```python
text = "You there, did you what you were looking for?"
regex_find(".e.", text)
```

```
<re.match object; span=(5, 8), match='her'>
<re.match object; span=(28, 31), match='wer'>
```

> Matches are **greedy** by default.

---

# SUBSTITUTION

[IMAGE: Substitution icon (arrows pointing in a circle)]

```python
re.sub('!+', '!', "Amazing!!!!!!!!!!")
```

```
'Amazing !'
```

> More practice: https://regexr.com/

---

# ELIZA: ROGERIAN THERAPIST

[IMAGE: Robot icon]

*   Men are all alike.
    *   **IN WHAT WAY**
*   They're always bugging us about something or other.
    *   **CAN YOU THINK OF A SPECIFIC EXAMPLE**
*   Well, my boyfriend made me come here.
    *   **YOUR BOYFRIEND MADE YOU COME HERE**
*   He says I'm sad much of the time.
    *   **I AM SORRY TO HEAR YOU ARE SAD**

---

# ELIZA: JUST ROGERIAN REGEX

[IMAGE: Robot icon]

```regex
s/.* I'M (anxious|sad) .*/I AM SORRY TO HEAR YOU ARE \1/
s/.* I AM (anxious|sad) .*/WHY DO YOU THINK YOU ARE \1?/
s/.* all .*/IN WHAT WAY?/
s/.* always .*/CAN YOU THINK OF A SPECIFIC EXAMPLE?/
```

---

# SYSTEM EVALUATION

[IMAGE: Boxing glove icon]

[IMAGE: Close-up of a young boy leaning over a wooden table, writing on a school paper with a pencil.]

---

# LABEL, SEQUENCE, RANKING?

[IMAGE: Police officer/security guard emoji]

*   $f(d) \rightarrow y$
*   $f(d_t) \rightarrow y_t$
*   $f(d; q) \rightarrow r$

---

# BUILDING REGEX

[IMAGE: Hammer icon]

Find all instances of "the".

---

# BUILDING REGEX

[IMAGE: Hammer icon]

Find all instances of "the".

`the`

---

# BUILDING REGEX

[IMAGE: Hammer icon]

Find all instances of "the".

`the`

Misses caps.

---

# BUILDING REGEX

[IMAGE: Hammer icon]

Find all instances of "the".

`the`

Misses caps.

`[tT]he`

---

# BUILDING REGEX

[IMAGE: Hammer icon]

Find all instances of "the".

`the`

Misses caps.

`[tT]he`

Matches words like 'there'.

---

# BUILDING REGEX

[IMAGE: Hammer icon]

Find all instances of "the".

`the`

Misses caps.

`[tT]he`

Matches words like 'there'.

`[^a-zA-Z][tT]he[^a-zA-Z]`

---

# EVALUATION

[IMAGE: Magnifying glass icon]

Two error types:

*   Matching ones that we shouldn't have (there, then): **false positives**.
*   Not matching ones that we should have (The): **false negative**.

> NLP / ML often face this as antagonistic goals to optimization.

---

# !? CONFUSION MATRICES

| | $\hat{y}=1$ | $\hat{y}=0$ |
| :---: | :---: | :---: |
| $y=1$ | TP | FN |
| $y=0$ | FP | TN |

---

# OTHER METRICS

[IMAGE: Bar chart icon]

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

$$
F_{\beta} \text{ score} = \frac{(1 + \beta^2) \cdot TP}{(1 + \beta^2) \cdot TP + \beta^2 \cdot FN + FP}
$$

---

# INFORMATION RETRIEVAL

[IMAGE: Books icon]

$$
\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}
$$

> Where $|Q|$ is the total number of queries, and $\text{rank}_i$ the rank of the 1st relevant result.

$$
\text{AP} = \frac{\sum_{k=1}^n (P(k) \ast \text{rel}(k))}{\text{number of relevant items}}
$$

> Where $\text{rel}(k)$ is an indicator function which is 1 when the item at rank $K$ is relevant, and $P(k)$ is the Precision@k metric.

---

# ANNOTATION

[IMAGE: Price tag icon]

*   Collect data.
*   Have humans **annotate** labels, relevance, etc.
*   Evaluate the **inter-rater agreement** (with e.g. Cohen's $\kappa$).

---

# NORMALIZATION

[IMAGE: Police officer/security guard emoji]

[IMAGE: Single orange LEGO brick centered on a blue LEGO baseplate.]

---

# VARIATION IN VIEW

[IMAGE: Sparkles icon]

| Corpus | Tokens ($N$) | Types ($|V|$) |
| :--- | :---: | :---: |
| Shakespeare | 884 K | 31 K |
| Brown corpus | 1 M | 38 K |
| Switchboard | 2.4 M | 20 K |
| COCA | 440 M | 2 M |
| Google $n$-gram | 1 T | 13 M |

> **Heap's Law**: $|V| = k \cdot N^{\beta}$ where $0 < \beta < 1$ (typically $.67-.75$).

---

# COMMON APPROACHES

[IMAGE: Hammer and wrench icon]

*   **Case folding**: The $\rightarrow$ the.
*   **Lemmatization**: He is reading sci-fi stories $\rightarrow$ He be read sci-fi story.
*   **Stemming**: cats $\rightarrow$ cat, accurate $\rightarrow$ accur, copy $\rightarrow$ copi (Porter stemmer).

---

# CORRECTION: LEVENSHTEIN EDIT DISTANCE

[IMAGE: Checkmark icon]

```
i n t e * n t i o n
| | | | | | | | | |
* e x e c u t i o n
d s s i s
```

```python
def min_edit_dist(source, target):
    n, m = len(source), len(target)
    D[n+1,m+1] = zeroes

    for i in range(1, n):
        D[i, 0] = D[i-1, 0] + del_cost(source[i])
    for j in range(1, m):
        D[0, j] = D[0, j-1] + ins_cost(target[j])

    for i in range(1, n):
        for j in range(1, m):
            d[i, j] = min(D[i-1, j] + del_cost(source[i]),
                          D[i-1, j-1] + sub_cost(source[i], target[j])
                          D[i, j-1] + ins_cost(target[i]))
    return D
```

---

# DISTANCES + TRACEBACK

[IMAGE: Map pin icon]

| \# | \# | e | x | e | c | u | t | i | o | n |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **\#** | 0 | $\leftarrow$1 | $\leftarrow$2 | $\leftarrow$3 | $\leftarrow$4 | $\leftarrow$5 | $\leftarrow$6 | $\leftarrow$7 | $\leftarrow$8 | $\leftarrow$9 |
| **i** | $\uparrow$1 | $\leftarrow \uparrow$2 | $\leftarrow \uparrow$3 | $\leftarrow \uparrow$4 | $\leftarrow \uparrow$5 | $\leftarrow \uparrow$6 | $\leftarrow \uparrow$7 | 6 | $\leftarrow$7 | $\leftarrow$8 |
| **n** | $\uparrow$2 | $\leftarrow \uparrow$3 | $\leftarrow \uparrow$4 | $\leftarrow \uparrow$5 | $\nwarrow \leftarrow \uparrow$6 | $\leftarrow \uparrow$7 | $\leftarrow \uparrow$8 | $\uparrow$7 | $\leftarrow \uparrow$8 | 7 |
| **t** | $\uparrow$3 | $\leftarrow \uparrow$4 | $\leftarrow \uparrow$5 | $\leftarrow \uparrow$6 | $\leftarrow \uparrow$7 | $\leftarrow \uparrow$8 | 7 | $\leftarrow \uparrow$8 | $\leftarrow \uparrow$9 | $\uparrow$8 |
| **e** | $\uparrow$4 | $\nwarrow$3 | $\leftarrow$4 | $\leftarrow$5 | $\leftarrow$6 | $\nwarrow \leftarrow$7 | $\leftarrow \uparrow$8 | $\leftarrow \uparrow$9 | $\leftarrow \uparrow$10 | $\uparrow$9 |
| **n** | $\uparrow$5 | $\uparrow$4 | $\leftarrow \uparrow$5 | $\leftarrow \uparrow$6 | $\leftarrow \uparrow$7 | $\leftarrow \uparrow$8 | $\nwarrow \uparrow$9 | $\leftarrow \uparrow$10 | $\leftarrow \uparrow$11 | $\uparrow$10 |
| **t** | $\uparrow$6 | $\uparrow$5 | $\leftarrow \uparrow$6 | $\leftarrow \uparrow$7 | $\leftarrow \uparrow$8 | $\leftarrow \uparrow$9 | $\nwarrow$8 | $\leftarrow$9 | $\leftarrow$10 | $\uparrow$11 |
| **i** | $\uparrow$7 | $\uparrow$6 | $\leftarrow \uparrow$7 | $\leftarrow \uparrow$8 | $\leftarrow \uparrow$9 | $\leftarrow \uparrow$10 | $\uparrow$9 | $\nwarrow$8 | $\leftarrow \uparrow$9 | $\leftarrow$10 |
| **o** | $\uparrow$8 | $\uparrow$7 | $\leftarrow \uparrow$8 | $\leftarrow \uparrow$9 | $\uparrow$10 | $\leftarrow \uparrow$11 | $\uparrow$10 | $\uparrow$9 | $\nwarrow$8 | $\leftarrow$9 |
| **n** | $\uparrow$9 | $\uparrow$8 | $\leftarrow \uparrow$9 | $\leftarrow \uparrow$10 | $\leftarrow \uparrow$11 | $\leftarrow \uparrow$12 | $\uparrow$11 | $\uparrow$10 | $\uparrow$9 | $\nwarrow$8 |

---

# ENCODINGS

[IMAGE: Monitor icon]

*   $n$-gram encoding (more next lecture).
*   Byte-Pair Encoding (BPE, count-based merges).
*   WordPiece Encoding (currently common, probability-based merges).

---

# QUESTIONS

[IMAGE: Waving hand emoji]

Post on the Discussion board and join class on Thursdays!