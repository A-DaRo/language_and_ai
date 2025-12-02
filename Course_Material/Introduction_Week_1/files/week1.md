# LANGUAGE & AI: INTRODUCTION

[IMAGE: Close-up of C-3PO, a gold-colored fictional robot character.]

Chris Emmery
Department of Cognitive Science & AI
Tilburg University

@cmry • @_cmry • @cmry • cmry.github.io

---

# WHY IS LANGUAGE DIFFICULT?

[IMAGE: View looking straight up through the sparse, dark branches of tall trees against a bright, overcast sky.]

---

# WHAT IS LANGUAGE?

> We describe objects with **nouns**, actions with **verbs**, and properties with **adjectives**.

---

# SURFACE FORM: SENTENCE

This is a simple sentence WORDS

---

# WORD FORM AND RELATIONS: MORPHOLOGY

This is a simple sentence WORDS

| | | be | MORPHOLOGY |
| :---: | :---: | :---: | :---: |
| | | 3sg | |
| | | present | |

---

# WORD CATEGORIES: PART OF SPEECH TAGS

| DT | VBZ | DT | JJ | NN | PART OF SPEECH |
| :---: | :---: | :---: | :---: | :---: | :---: |
| This | is | a | simple | sentence | WORDS |
| | be | | | | MORPHOLOGY |
| | 3sg | | | | |
| | present | | | | |

---

# PHRASE STRUCTURE: SYNTAX

[IMAGE: Parse tree diagram for the sentence "This is a simple sentence" showing phrase structure (S, VP, NP) and POS tags (DT, VBZ, JJ, NN).]

| | | | | | SYNTAX |
| :---: | :---: | :---: | :---: | :---: | :---: |
| | | | | | PART OF SPEECH |
| This | is | a | simple | sentence | WORDS |
| | be | | | | MORPHOLOGY |
| | 3sg | | | | |
| | present | | | | |

---

# WORD MEANING: SEMANTICS

[IMAGE: Parse tree diagram for the sentence "This is a simple sentence" showing phrase structure (S, VP, NP) and POS tags (DT, VBZ, JJ, NN).]

| | | | | | SYNTAX |
| :---: | :---: | :---: | :---: | :---: | :---: |
| | | | | | PART OF SPEECH |
| This | is | a | simple | sentence | WORDS |
| | be | | SIMPLE1 | SENTENCE1 | MORPHOLOGY |
| | 3sg | | having | string of words | SEMANTICS |
| | present | | few parts | satisfying the grammatical rules of a languauge | |

---

# CONTEXTUAL MEANING: DISCOURSE

[IMAGE: Parse tree diagram for the sentence "This is a simple sentence" connected to a subsequent sentence "But it is an instructive one." by an arrow labeled CONTRAST, illustrating Discourse context.]

| | | | | | SYNTAX |
| :---: | :---: | :---: | :---: | :---: | :---: |
| | | | | | PART OF SPEECH |
| This | is | a | simple | sentence | WORDS |
| | be | | SIMPLE1 | SENTENCE1 | MORPHOLOGY |
| | 3sg | | having | string of words | SEMANTICS |
| CONTRAST | present | | few parts | satisfying the grammatical rules of a languauge | DISCOURSE |
| | | | | | |
| | But it is an instructive one. | | | | |

---

# MORE AMBIGUITY

*   They saw a kid with a telescope.
*   Flying planes can be dangerous.
*   Time flies like an arrow.

[IMAGE: Two distinct parse tree diagrams for the sentence "They saw a kid with a telescope", illustrating structural ambiguity (syntactic ambiguity).]

---

# COMMON SENSE REASONING

> The purchase of Houston-based LexCorp by BMI for $2 Bn prompted widespread sell-offs by traders as they sought to minimize exposure. LexCorp had been an employee-owned concern since 2008.

*   H1: BMI acquired an American company.
*   H2: BMI bought employee-owned LexCorp for $3.4 Bn.
*   H3: BMI is an employee-owned concern.

---

# PRACTICAL CHALLENGES: SENTIMENT ANALYSIS

| | |
| :--- | :--- |
| **Negation** | "This movie is definitely not bad." |
| **Dependencies** | "Whoever thinks this film is incredibly bad is an idiot." |
| **Sarcasm** | "This is the best movie ever, lol." |
| **Context** | "This movie is only slightly better than Jaws." |

---

# CHALLENGES $\rightarrow$ TASKS

[IMAGE: Dependency parse diagram for the sentence "Whoever thinks this film is incredibly bad is an idiot." showing grammatical relations like nsubj, ccomp, acomp, advmod, csubj, and attr, along with POS tags (WP, VBZ, NN, RB, JJ).]

---

# CHALLENGES

*   Sarcasm detection.
*   World knowledge:

[IMAGE: Screenshot of a Google search result for "what is the average rating for jaws?". The result displays multiple high ratings for the 1975 film "Jaws, Reviews" (97% Rotten Tomatoes, 5/5 Common Sense Media, 8/10 IMDb).]

---

# HOW DO WE LEARN LANGUAGE?

[IMAGE: Young child sitting and reading a book titled "DERNIERS NOMADES GRAND NORD".]

---

# NAIVE ENGINEER

*   We learn from patterns in speech (text, dialogue)!
*   Chuck a whole bunch of text at a model.
*   Give it 'production' as goal (recognition, utterance).
*   >???
*   >Profit.

---

# LANGUAGE ACQUISITION RESEARCH

*   Kids need to learn how to **segment** speech first: stress, rhythm, spotting words.
*   Once they can, they start **mapping** tokens to objects.
*   Slow at first, around 2 years comes the **vocabulary spurt**.
*   Learn words in context of known words (**fast mapping**).

[IMAGE: Audio waveform segmented into individual German words: Wilhelm, hätte, Cora, heute, abend, folgen, sollen.]

> Image from Schmidt-Kassow et al., 2011.

---

# GROUNDED LEARNING

[IMAGE: Two people looking up toward a bright light source or screen, with one person pointing a finger.]

> Include vision, speech, etc.

---

# NASCENT FIELDS

*   **Linguistics**: study of morphology, syntax, pragmatics, semantics, phonetics, etc. How does language work? Largely unsolved.
*   **Psycholinguistics**: how does the brain process language (neurolinguistics, language acquisition). How does the brain process language? Largely unsolved.

---

# RELATED FIELDS

*   **Natural Language Processing**: building systems for specific tasks (machine translation, text-to-speech, summarization, language generation, question-answering etc.). Typically uses machine learning (nowadays).
*   **Computational Linguistics**: combining language technologies to do linguistically-motivated research.
*   **Text Mining**: combining language technologies to extract information from text data.

---

# THE BITTER LESSON

---

# A (VERY BRIEF) HISTORY OF NLP

*   Up until the 70s: everyone was really stoked on Chomsky's Language Faculty. Language is innate, so let's not learn from data, let's build grammars and ontologies.

[IMAGE: Black and white photograph of Noam Chomsky.]

---

# A (VERY BRIEF) HISTORY OF NLP

*   70-80s AI winter: Machine Translation is still more expensive than human translators. AI didn't deliver on its hype. Funding for AI research killed.
*   80s-10s: people less stoked on Chomsky, in come the statisticians, let's learn from data! Some really cool problem-specific algorithms are developed for NLP, everyone on their own island.

---

# A (VERY BRIEF) HISTORY OF NLP

*   10s-20s: Chris starts his PhD, neural networks take over NLP; everyone spends much time tweaking the same models for their specific tasks and they take weeks to run (not a fun time).
*   20s-now: Chris still hasn't received his PhD, neural networks still dominate NLP; we now have tranformers (much more fun)!

---

# SO NOW?

[IMAGE: XKCD comic strip (titled "The Bitter Lesson" in the context of the presentation). Panel 1: "OUR FIELD HAS BEEN STRUGGLING WITH THIS PROBLEM FOR YEARS." Panel 2: "STRUGGLE NO MORE! I'M HERE TO SOLVE IT WITH ALGORITHMS!" Panel 3: People sitting at computers. Panel 4 (SIX MONTHS LATER): "WOW, THIS PROBLEM IS REALLY HARD." "YOU DON'T SAY."]

---

# WHY (PSYCHO)LINGUISTICS FOR NLP?

Many modern day systems are:

*   **Black boxes**: they need creativity and linguistic questions to be interpreted.
    *   Communication: important from an industry perspective to communicate why a systems says $y$.
*   **Biased**: data is biased, fixes from from e.g. sociology.
    *   Risk: harmful content and inferences in demos.
*   **Imperfect**: debugging models is required, task-specific expert knowledge required for good test sets.
    *   Deployment: NLP are often put in 'heuristic prisions'.

> Respectively: interpretability, fairness, and consistency research.

---

# TEXT MINING PRELIMINARIES

[IMAGE: Abstract photo showing the spines of many books arranged in a circular, swirling pattern.]

---

# WARM-UP + RECAP

*   Language is complex:
    *   Representing language is complex.
    *   Mathematically interpreting language is complex.
    *   Inferring knowledge from language is complex.
    *   Understanding language is complex.
*   For both **classification** and **retrieval** (essential to Text Mining), we need good language representations.

---

# LANGUAGE AS A STRING

```
title,director,year,score,budget,gross,plot
"Dunkirk","Christopher Nolan",2017,8.4,100000000,183836652,"Allied soldiers fro
"Interstellar","Christopher Nolan",2014,8.6,165000000,187991439,"A team of exp
"Inception","Christopher Nolan",2010,8.8,160000000,292568851,"A thief, who stea
"The Prestige","Christopher Nolan",2006,8.5,40000000,53082743,"After a tragic a
"Memento","Christopher Nolan",2000,8.5,9000000,25530884,"A man juggles searchir
```

---

# TEXT TO VECTORS

[IMAGE: Open wooden drawer containing slips of paper with Chinese characters (part of a traditional indexing system or text sample).]

---

# CONVERTING TO NUMBERS

$$
d = \text{the cat sat on the mat} \rightarrow \vec{d} = \langle?\rangle
$$

---

# WORDS AS FEATURES

$$
d = \text{the cat sat on the mat} \rightarrow
$$

$$
\begin{bmatrix}
\text{cat} & \text{mat} & \text{on} & \text{sat} & \text{the} \\
1 & 1 & 1 & 1 & 1
\end{bmatrix}
$$

> Bag-of-Words Representation

---

# DOCUMENTS AS INSTANCES

$$
d_0 = \text{the cat sat on the mat}
$$
$$
d_1 = \text{my cat sat on my cat}
$$

$$
\begin{bmatrix}
\text{cat} & \text{mat} & \text{my} & \text{on} & \text{sat} & \text{the} \\
1 & 1 & 0 & 1 & 1 & 1 \\
1 & 0 & 1 & 1 & 1 & 0
\end{bmatrix}
$$

---

# DOCUMENTS $\ast$ TERMS

$$
V = [\text{cat} \quad \text{mat} \quad \text{my} \quad \text{on} \quad \text{sat} \quad \text{the}]
$$

$$
X = \begin{bmatrix}
1 & 1 & 0 & 1 & 1 & 1 \\
1 & 0 & 1 & 1 & 1 & 0
\end{bmatrix}
$$

*   $d$ = document
*   $V$ = vocabulary
*   $X$ = feature space

---

# DOCUMENT SIMILARITY

Wikipedia articles:

| data | language | learning | mining | text | vision | $y$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0 | 1 | 0 | 0 | 1 | CV |
| 1 | 1 | 1 | 0 | 1 | 0 | NLP |
| 1 | 0 | 1 | 1 | 1 | 0 | TM |

*   CV = Computer vision
*   NLP = Natural Language Processing
*   TM = Text Mining

---

# DOCUMENT SIMILARITY - JACCARD COEFFICIENT

$$
d_0 = \langle 1, 0, 1, 0, 0, 1 \rangle
$$
$$
d_1 = \langle 1, 1, 1, 0, 1, 0 \rangle
$$
$$
d_2 = \langle 1, 0, 1, 1, 1, 0 \rangle
$$

$$
J(d_0, d_1) = 2/5 = 0.4
$$
$$
J(d_0, d_2) = 2/5 = 0.4
$$
$$
J(d_1, d_2) = 3/5 = 0.6
$$

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

> words in $A$ **and** $B$ (**intersection**) / words in $A$ **or** $B$ (**union**)

---

# DOCUMENT RETRIEVAL

| data | language | learning | mining | text | vision |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0 | 1 | 0 | 0 | 1 |
| 1 | 1 | 1 | 0 | 1 | 0 |
| 1 | 0 | 1 | 1 | 1 | 0 |

$$
q = \text{learning language with text}
$$

| data | language | learning | mining | text | vision |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 1 | 1 | 0 | 1 | 0 |

---

# WORDS IN VECTOR SPACES

[IMAGE: Abstract image resembling a star field or a spatial warp effect, with bright lines radiating from a central point.]

---

# BINARY VS. FREQUENCY

*   (+) Binary is a very compact representation (in terms of memory).
*   (+) Algorithms like Decision Trees have a very straightforward and compact structure.
*   (-) Binary says very little about the weight of each word (feature).
*   (-) We can't use more advanced algorithms that work with Vector Spaces.

---

# TERM FREQUENCIES - SOME NOTATION

Let $\mathbf{D} = \{d_1, d_2, \dots, d_N\}$ be a set of documents, and $\mathbf{T} = \{t_1, t_2, \dots, t_M\}$ (previously $V$) a set of index terms for $\mathbf{D}$.

Each document $d_i \in \mathbf{D}$ can be represented as a frequency vector:

$$
\vec{d}_i = \langle \text{tf}(t_1, d_i), \dots, \text{tf}(t_M, d_i) \rangle
$$

where $\text{tf}(t, d)$ denotes the frequency of term $t_m \in \mathbf{T}$ for document $d_i$.

---

# TERM FREQUENCIES

$$
d_0 = \text{the cat sat on the mat}
$$
$$
d_1 = \text{my cat sat on my cat}
$$

$$
T = [\text{cat} \quad \text{mat} \quad \text{my} \quad \text{on} \quad \text{sat} \quad \text{the}]
$$

$$
X = \begin{bmatrix}
1 & 1 & 0 & 1 & 1 & 1 \\
2 & 0 & 2 & 1 & 1 & 0
\end{bmatrix}
$$

---

# TERM FREQUENCIES?

```python
d0 = 'natural-language-processing.wiki'
d1 = 'information-retrieval.wiki'
d2 = 'artificial-intelligence.wiki'
d3 = 'machine-learning.wiki'
d4 = 'text-mining.wiki'
d5 = 'computer-vision.wiki'
```

$$
t = [\text{learning}]
$$

$$
X_t = \begin{bmatrix}
27 \\
2 \\
46 \\
134 \\
6 \\
10
\end{bmatrix}
\quad
\ln(X_t) = \begin{bmatrix}
3.33 \\
1.10 \\
3.85 \\
4.91 \\
1.95 \\
2.40
\end{bmatrix}
$$

*   $\text{tf} = 10$ less important than $\text{tf} = 100$, but also $.10$?
*   Information Theory to the rescue!
*   Natural log: $\ln(\text{tf}(t, d) + 1)$
*   Note $+1$ smoothing to avoid $\ln(0) = -\text{inf}$

---

# !! SOME REMAINING PROBLEMS

*   The longer a document, the higher the probability a term will occur often, and will thus have more weight.
*   Rare terms should actually be informative, especially if they occur amongst few documents.
    *   If $d_1$ and $d_2$ both have the token `normalization` in their vectors, and all the other documents do not $\rightarrow$ strong similarity.

> Latter: Document Frequency

---

# (INVERSE) DOCUMENT FREQUENCY

$$
\text{idf}_t = \log_b \frac{N}{\text{df}_t} \quad N = \text{nr. of documents}
$$

$$
t = [\text{naive}]
$$

$$
X_t = \begin{bmatrix}
1 \\
0 \\
2 \\
3 \\
0 \\
0
\end{bmatrix}
\quad
\text{df}_t = 3 \quad \text{idf}_t = \log \frac{6}{3} = -0.30
$$

---

# PUTTING IT TOGETHER: $\text{tf} \ast \text{idf}$ WEIGHTING

$$
w_{t,d} = \ln(\text{tf}(t, d) + 1) \cdot \log \frac{N}{\text{df}_t}
$$

| $d$ | learning | text | language | intelligence |
| :---: | :---: | :---: | :---: | :---: |
| 0 | $5 \rightarrow 0.32$ | 1 | 10 | 0 |
| 1 | 2 | $21 \rightarrow 0.0$ | 6 | 0 |
| 2 | 0 | 3 | 0 | $1 \rightarrow 0.33$ |

---

# NORMALIZING VECTOR REPRESENTATIONS

*   We fixed the global information per **document $\ast$ term** instance.
*   Despite $\text{tf} \ast \text{idf}$, we still don't account for the **length** of documents (i.e. the amount of words in total).
*   Why is this an issue?

---

# EUCLIDEAN DISTANCE

$$
d(\vec{p}, \vec{q}) = \sqrt{\sum_{i=1}^n (\vec{p}_i - \vec{q}_i)^2}
$$

> Documents with many words are far away.

---

# $\ell_2$ NORMALIZATION

$$
\| \vec{p} \|_2 = \sqrt{\sum_i p_i^2}
$$

> Divide all feature values by norm.

---

# COSINE SIMILARITY

$$
\vec{p} \cdot \vec{q} = \sum_{i=1}^n \vec{p}_i \vec{q}_i = \vec{p}_1 \vec{q}_1 + \vec{p}_2 \vec{q}_2 + \dots + \vec{p}_n \vec{q}_n
$$

> Under the $\ell_2$ norm only (otherwise normalize vectors before)!

$$
\text{SIM} = \frac{\vec{p} \cdot \vec{q}}{\sqrt{\vec{p} \cdot \vec{p}} \cdot \sqrt{\vec{q} \cdot \vec{q}}}
$$