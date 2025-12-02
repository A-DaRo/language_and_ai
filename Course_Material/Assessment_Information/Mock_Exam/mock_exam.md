# 🏋️ Mock Exam

> **⚠️ Disclaimer**: this is (part of) an old exam (2020). It serves to illustrate the [style](https://example.com/mock-exam-style) of questions to expect (and for some practice). Content / difficulty and balance might differ from exam to exam. Newly introduced topics (e.g. LLMs) are not included. Some math questions for later topics are missing as examples, but the ones that are included should adequately demonstrate that they are either exact copies of the practice material, or questions about subcomponents involved.

### Which of the following sentences would you expect a sentiment classifier is least likely to accurately predict the label of?

a. I really hate multiple choice questions.
b. I don't generally like multiple choice questions at all.
c. Multiple choice questions are fun! Not.
d. I like multiple choice questions about as much as stepping on a piece of LEGOs.

<details>
<summary>**Correct answer**</summary>
d
</details>

### **Which of the following statements best describes Sutton's bitter lesson? 🍋**

a. Leveraging human knowledge beats leveraging of computation.
b. In the long run, general methods are more effective than clever shorter term tricks.
c. Moore's law indicates exponentially falling cost per unit of computation.
d. The complexity of language will never be captured by general methods, regardless of compute.

<details>
<summary>**Correct answer**</summary>
b
</details>

### What is the Jaccard Similarity between the following two vectors?

$\mathbf{v}_1 = [1, 0, 1, 0, 1, 0, 1]$
$\mathbf{v}_2 = [1, 1, 0, 1, 1, 0, 1]$

<details>
<summary>**Correct answer**</summary>
0.5
</details>

### You are given a Bag-of-Words representation with word frequencies. Three documents (#1, #2, and #3) have exactly the same frequencies for all words, except for one word: “language”. In document #3, this word has a much higher frequency than in the other two. How does this affect the Cosine Distance between the documents?

a. Document #3 is much further away from #1 and #2 on the “language” axis.
b. The Cosine Distance between document #1 and #3 will be high.
c. The Cosine Distance between document #1 and #3 will be low.
d. Document #3 is much further away from #1 and #2 on the axes other than that of “language”.

<details>
<summary>**Correct answer**</summary>
b
</details>

### **Which of the following characteristics belongs to the concept 'Garbage In, Garbage Out'?**

a. Typos increase the vocabulary and affect the quality of document similarity.
b. The same word with different meanings affects the quality of document similarity.
c. Emoticons can change the sentiment of a document.
d. Slang words might change the predictions of a classifier.

<details>
<summary>**Correct answer**</summary>
a
</details>

### Compute the Levenshtein distance between `bafg` and `gafb`.

<details>
<summary>**Correct answer**</summary>
4
</details>

### Which of these patterns will the regular expression: `ab[^c]c.{2}` find a match in?

a. aaaabbcb
b. abcabcab
c. ccaabcaba
d. baabbccba

<details>
<summary>**Correct answer**</summary>
d
</details>

### **Consider this table of two corpora, and their descriptive statistics. What can be concluded from their difference in types and tokens?**

[Image placeholder: Table comparing corpus statistics for Shakespeare and Wikipedia sample. Shakespeare has 884,647 tokens and 29,067 types. Wikipedia Sample has 1,000,000 tokens and 72,000 types.]

a. The Shakespeare corpus likely contains many out-of-vocabulary words.
b. The Wikipedia sample contains a lot of misspellings.
c. The vocabulary of Wikipedia is more varied than that of Shakespeare.
d. The texts from Shakespeare are longer than those from Wikipedia.

<details>
<summary>**Correct answer**</summary>
c
</details>

### What is the correct reason for applying add-one smoothing to a bi-gram count matrix?

a. Add-one smoothing adds probability mass to zeroes to penalize common words.
b. Add-one smoothing avoids calculations resulting in zero.
c. Add-one smoothing encodes the unigram in the bigram counts.
d. Add-one smoothing results in better estimations than Laplace smoothing.

<details>
<summary>**Correct answer**</summary>
b
</details>

### Which of the these metrics is used to evaluate a Language Model's performance on unseen data?

a. Mean Reciprocal Rank
b. Accuracy
c. Information Gain
d. Perplexity

<details>
<summary>**Correct answer**</summary>
d
</details>

### Consider a k-Nearest Neighbors text classifier.

[Image placeholder: Diagram showing a k-Nearest Neighbors classifier decision boundary, with points labelled as Class A and Class B in a 2D feature space.]

<details>
<summary>**Correct answer**</summary>
1
</details>

### When should one apply Bernoulli Naive Bayes rather than Multinomial Naive Bayes?

a. The features are binary.
b. The target is binary.
c. In both cases.
d. In neither case.

<details>
<summary>**Correct answer**</summary>
a
</details>

### Which of the statements below correctly describes the concept of Distributional Similarity?

a. The meaning of a word can be derived from its context.
b. Semantically similar words will often share similar contexts.
c. The probabilities for semantically similar words will be the same.
d. We can efficiently compute similarity in a distributed fashion.

<details>
<summary>**Correct answer**</summary>
b
</details>

### In Recurrent Neural Networks, how can we view the functionality of $\mathbf{U}$?

[Image placeholder: Diagram illustrating a simple Recurrent Neural Network calculation step: $h_{t} = f(W x_{t} + U h_{t-1} + b)$]

a. It maps the previous hidden states to the output.
b. It is an attention matrix over the context.
c. It weighs information from the previous hidden state.
d. It embeds the previous hidden state.

<details>
<summary>**Correct answer**</summary>
c
</details>

### The softmax is defined below. What happens to the output when there is very small variance in the output values?

[Image placeholder: Equation for softmax function: $p_{i} = \frac{e^{z_{i}}}{\sum_{j} e^{z_{j}}}$]

a. The absolute difference between the values is decreased through applying the softmax.
b. The relative difference between the values is decreased through applying the softmax.
c. The values are standardized through applying the softmax.
d. The small variance causes $\arg\max$ to fail.

<details>
<summary>**Correct answer**</summary>
a
</details>

### From the options below, what was the most important driver behind the rise of the transformers?

a. Optimus Prime.
b. The addition of attention to the models.
c. Recurrent nature of the models.
d. Memory-efficiency of the models.

<details>
<summary>**Correct answer**</summary>
b
</details>

### Which of the following statement(s) are characteristics of Part of Speech tagging with Hidden Markov Models?

```
1. Tags are determined using a sequential model, and minimum edit distance.
2. Due to the label distribution, majority baseline scores over types are often > 90%.
3. We can extract relevant features using Hearst Patterns.
```

a. 2
b. 2 and 3.
c. 1 and 3.
d. 3

<details>
<summary>**Correct answer**</summary>
a
</details>

### Consider the following emission matrix. Given only this information, what is the most ambiguous token for a Part-of-Speech tagging task?

[Image placeholder: Emission matrix table showing tokens (will, back, bill, the, saw, pitch) and POS tags (NN, VB, DT, VBP, MOD). Values are probabilities.]

a. will
b. back
c. the
d. bill

<details>
<summary>**Correct answer**</summary>
a
</details>

---

> **⚠️** Please note that the “Point Allocation” model is not exhaustive. Partial answers might be combined into +1 point, and correct answers that do not exactly match the model answer will be allocated points too (in the relevant category).

### Define edit distance and explain the role of dynamic programming (e.g. the Viterbi algorithm) in its calculation. Then consider its role in spelling correction: how are typos identified and how can edit distance help in replacing typos? Consider tweets as a case study, and discuss the relevance of this data to the task.

<details>
<summary>**Point Allocation Answer**</summary>
*   1 pt — mentions edit distance is the (minimum) amount of ‘edits’ (additions, deletions, replacements) required to change one string into another
*   1 pt — includes mention that dynamic programming can be used to efficiently compute this
*   1 pt — notes out-of-vocabulary words can be seen as typo’s
*   1 pt — explains MED calculation between word and candidates for replacement
*   1 pt — general mention of some of the concepts: noise (creative language), short-form, fast social platform, high typo yield, challenging data
</details>

### Break down the pipeline to calculate cosine similarity score between word vectors. Discuss how words vectors are constructed, using count-based and prediction-based approaches respectively, and explain how their pipelines differ when calculating document similarity.

<details>
<summary>**Point Allocation Answer**</summary>
*   1 pt — mentions raw text first needs to be tokenized.
*   1 pt — correctly identifies count-based techniques (tf\*idf, ppmi) AND notes vectors are constructed through frequencies and weightings
*   1 pt — correctly identifies prediction-based techniques (word2vec) AND notes vectors are constructed as a by-product of training a model
*   1 pt — generally explains that in count-based approaches, every document is often already represented by a vector words (normalize, dot product = cosine)
*   1 pt — generally explains that in prediction-based approaches, words are often represented by dense vectors, which still need to be summed/averaged to construct a document vector
</details>