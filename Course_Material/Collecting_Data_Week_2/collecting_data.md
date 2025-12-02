# 💽 Collecting Data

| Property | Value |
| :--- | :--- |
| **Week** | Week 2 |
| **Book Chapters** | Chapter 2 |
| **Slides** | [week2.pdf (File)](files/week2.pdf), [week2\_inperson\_slides.pdf (File)](files/week2_inperson_slides.pdf) |
| **Recordings** | [tue.video.yuja.com/P/V…464953](https://tue.video.yuja.com/P/VideoManagement/MediaLibrary/Users/u-oD0ckDKj/MyMediaCollections/88f56cbc-028a-44b3-9a70-c4212816b9c6/WatchVideo/5464953) |
| **Solutions** | Empty |

***

**Table of Contents**

1.  [🎽 Warm-up Questions](#warm-up-questions)
2.  [📺 Lecture Videos](#lecture-videos)
3.  [🧶 RegEx](#regex)
    *   [🚩 Task 1](#task-1)
    *   [🚩 Task 2](#task-2)
    *   [🚩 Task 3](#task-3)
4.  [📏 Minimum Edit Distance](#minimum-edit-distance)
    *   [🚩 Task 4](#task-4)

## 🎽 Warm-up Questions

***

> ℹ️ These questions are intended to prime your brain for the materials we will be covering in the videos below. They are optional, and there are no expectations wrt the answers.

*   Last lecture we talked about how language is complex: its constructions are inherently minimalistic for the sake of communication effectiveness, and dependent on (dialogue or ‘world knowledge’) context to be fully disambiguated. We, as humans, aren’t always (formally) effective communicators, though. Posts on social media, how we message friends and family; the way we express ourselves through the medium of text can be quite varied and creative. Can you think of examples of such ‘non-standard’ language use?
*   We also talked about how to split language into (computationally) meaningful units, how to improve the information encoded in their representation, and how to apply operations on this representation (similarity scores). We briefly discussed ways that variance might be introduced (see previous question, too), and how this is problematic for our representations of language. Can you articulate why?
*   Variance, or noise, can potentially be reduced: despite how creative we like to be in our language use, generally there are ways to make things look more ‘normal’ or ‘standardized’. For example: spell checking, or taking the ‘root form’ of words; any more standard form might work. Can you think of ways we might programmatically implement this process of correcting for language variance?
*   Can you think of social scenario’s where correcting for variance might have strong effects on how good your text representation is? For example, suppose you’re interested in classifying text in some live chat on an online platform to filter profanity (like YouTube does, for example). Would you remove or keep the variance?

## 📺 Lecture Videos

***

> 🖼️ Slides are available at the top of the page!

1.  **Noisy Text**
    [Video Placeholder: tue.video.yuja.com/P/VideoManagement/MediaLibrary/search/WatchVideo/5464953]

2.  **Regular Expressions**
    [Video Placeholder: tue.video.yuja.com/P/VideoManagement/MediaLibrary/search/WatchVideo/5464953]

3.  **System Evaluation**
    [Video Placeholder: tue.video.yuja.com/P/VideoManagement/MediaLibrary/search/WatchVideo/5464953]

4.  **Normalization**
    [Video Placeholder: tue.video.yuja.com/P/VideoManagement/MediaLibrary/search/WatchVideo/5464953]

## 🧶 RegEx

***

Let's practice some regular expressions. You can test them any way you like; in Python, using `re`, or [RegExr](https://regexr.com/) (both shown in the lecture). In general, you're expected to understand the syntax parts discussed in the book both ways (what syntax matches what, what syntax is required to match something). Note that for Python, a `/pattern/` can be expressed as `'pattern'`. Here's a basic cheat sheets to use:

<details>
<summary>**Cheat Sheet**</summary>

| RE | Meaning | Example | Match | No Match |
| :--- | :--- | :--- | :--- | :--- |
| `.` | Any single character | `n.p` | nop, nlp | noop |
| `[abc]` | Any one of these characters | `analy[zs]e` | analyze, analyse | analyce |
| `[a-z]` | Any characters in this range | `test[2-4][a-c]` | test2a, test3c | test1a, test2z |
| `[^abc]` | Not one of these characters | `analy[^s]e` | analyze | analyse |
| `[^a-z]` | Not a character in this range | `demo[1-3][^a-c]` | demo1d, demo2f | demo1a |
| `|` | Or | `test|demo` | demo, test | example |
| `^` | (string) Starts with | `^be` | be that as it may | to be |
| `$` | (string) Ends with | `be$` | that's where i'll be | be there |
| `?` | Zero or one time(s) (greedy) | `tests?[1-3]` | test1, tests2 | testx1 |
| `??` | Zero or one time(s) (lazy) | `<li?>.*?</li>?` | see comment | tests, tests2 |
| `*` | Zero or more time(s) (greedy) | `lo*ve it` | looooove it, lve it | luve it |
| `*?` | Zero or more time(s) (lazy) | `lo*?ve it` | love it | loove it |
| `{n}` | n times | `ba{2}` | baa | baaa |
| `{n,m}` | n to m times | `ba{1,2}` | ba, baa | baaa |
| `{n,}` | n times or more | `ba{3,}` | baaa, baaaa | baa |
| `()` | group | `(do|get) that` | do that, get that | make that |
| `(?:)` | passive group | `(?:ignore) this` | (ignore) this | |
| `(?=)` | lookahead | `we (?=ignore)` | we (ignore) | |
| `(?<=)` | lookbehind | `(?<=ignore) this` | (ignore) this | |
| `\` | escape character | `really\?` | really? | reall |
| `\s` | white space | `a\sspace` | a space | aspace |
| `\S` | not a white space | `\S*` | token | |
| `\d` | digit | `[a-z]\d` | a9 | aa |
| `\D` | not a digit | `[a-z]\D` | aa | a9 |
</details>

### 🚩 Task 1

***

Given the following text:

> **Natural language processing** (**NLP**) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language,
in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of "understanding" the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.

[Try without a regex tool.](https://example.com/no-regex-tool)

*   Match all letters (tokens).

<details>
<summary>**Solution**</summary>
`\w+`
</details>

*   Match any combination of two vowels.

<details>
<summary>**Solution**</summary>
`[aoeiu][aoeiu]`
</details>

*   Match words that end with an 's'.

<details>
<summary>**Solution**</summary>
`\b\w*s\b`
</details>

*   Match words that are four characters or less.

<details>
<summary>**Solution**</summary>
`\b\w{1,4}\b`

> ‼️ This only works in the RegExr. For Python we need [lookahead/behind statements](https://www.rexegg.com/regex-lookarounds.html). These are not supported for all browsers in RegExr.

`(?<=\b)\w{1,4}(?=\b)`
</details>

### 🚩 Task 2

***

Which of the following patterns will the given regex find matches in ([try without a regex tool](https://example.com/no-regex-tool))?

1. aaaabbbccc
2. cbacbacba
3. aabc
4. ccaaccaa

*   `a+`

<details>
<summary>**Solution**</summary>
1, 2, 3, 4
</details>

*   `[ab]{3,}`

<details>
<summary>**Solution**</summary>
1, 3
</details>

*   `c.ac?`

<details>
<summary>**Solution**</summary>
2, 4
</details>

*   `[^a]c`

<details>
<summary>**Solution**</summary>
1, 3, 4
</details>

### 🚩 Task 3

***

Given the following text:

> Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language,
in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of "understanding" the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.

And this suboptimal regex cooked up by your lecturer:

`[A-Z]\w+`

Assume we want to find all capitalized words that are **not** the start of a sentence (label 1, rest of the tokens in the text have label 0). The text is already tokenized (spaces delimit tokens). [You can use a regex tool.](https://example.com/use-regex-tool)

*   What is the accuracy of this regex?

<details>
<summary>**Solution**</summary>
If we run `[^ ]+` we can see there are 97 tokens. Of those, the following are positive: $\texttt{Georgetown, Russian, English, ALPAC}$ — we matched all of those correctly. We also matched the following tokens that were supposed to be negative (false positives): $\texttt{The, The, However, Little}$. Hence, the accuracy is 93 correct / 97 total = 95.88%.
</details>

*   What regex gets us 100% accuracy?

<details>
<summary>**Solution**</summary>
For this specific example, this works (there are probably more options):

`(?<=\w )[A-Z]\w+`

Very much tailored to this specific example, although it probably generalizes fairly well. Basically, "if a letter + space precedes the capitalized word" and then `?<=` not captured (so it will only return the original part in the output).

> ‼️ Please note that look behind/ahead is not supported for all browsers in RegExr.

`(?<=\b)\w{1,4}(?=\b)`
</details>

## 📏 Minimum Edit Distance

***

To calculate the distance between two strings, we start with an 'empty string' `#` and compute $D(i,j)$ for all $i\ (0 < i< n)$ and $j\ (0 < j < m)$, given the following—according to Levenshtein distance—operations:

$$
\min \left\{
	\begin{array}{lr}
D(i-1,j) + 1 \\
D(i,j-1) + 1 \\
D(i-1,j-1)\ + \left\{
	    \begin{array}{lr}
            2;\ \text{if}\ X(i) \neq Y(j) \\
            0;\ \text{if}\ X(i) = Y(j)
            \end{array}
            \right.
	\end{array}
\right.
$$

Assume we have the following two strings: `abcd` and `aced`. We get:

| | **#** | a | c | e | d |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **#** | **0** | 1 | 2 | 3 | 4 |
| b | **1** | 2 | 1 | 2 | 3 |
| c | **2** | 3 | 2 | 1 | 2 |
| d | **3** | 4 | 3 | 2 | 1 |
| e | **4** | 5 | 4 | 3 | 2 |

I have annotated the steps that we followed. We start with the first \# rows (bold), those always range from empty string to the length of the string. Then, we can start from the bottom of the 1st column (provided that our table lay-out is similar), from the cell that is not bold. There are three numbers around it (0 ↙️, 1 ⬅️ and 1 ⬇️). We first check ↙️: if the strings are the same, then we have a cost of 0, if they are not, then using the diagonal character costs 2 (substitution). If we use ⬅️, or ⬇️, the cost is 1. We have to **add** this costs to whatever is already in those respective cells (so 0 + either 0 (if similar char) or 2 (if not similar), 1 + 1 and 1 + 1). Then we select what the [minimum](https://example.com/minimum-value) original value + cost is as the current value. For this cell, that is zero (because similar characters). This operation we continue on applying. For the next cells upwards, nothing exiting happens (just adding ones). Then on to the second column, above the **2**. We have ↙️ 1 (+ 2 because not similar), ⬅️ 0 (+ 1) and ⬇️ 2 (+ 1). The [minimum](https://example.com/minimum-value-2) value of these is 0 + 1 = 1. etc etc.

Finally, if we trace the shortest path back from the top right corner, we get the operations we need to do. If we go ⬅️, that means an addition, if we go ⬇️, that means a deletion, if we go ↙️, it's either substitution or nothing, depending on the value. So: 2 → 2 (nothing, both d), 2 → 1 (insert e), 1 → 1 (nothing, both c), 1 → 0 (delete b), 0 → 0 (do nothing, both a).

### 🚩 Task 4

***

*   What is the minimum edit distance between `bcde` and `cdab`?

<details>
<summary>**Solution**</summary>
We can just try to eyeball this: delete `b` (1), replace `e` → `a` (2), insert `b` (1) = 4.
</details>

*   What is the backtrace path between `bcde` and `cdab`?

<details>
<summary>**Solution**</summary>
Now we have to calculate the whole matrix:

| | \# | c | d | a | b |
| :--- | :--- | :--- | :--- | :--- | :--- |
| e | 4 | 3 | 2 | 3 | 4 |
| d | 3 | 2 | 1 | 2 | 3 |
| c | 2 | 1 | 2 | 3 | 4 |
| b | 1 | 2 | 3 | 4 | 3 |
| \# | 0 | 1 | 2 | 3 | 4 |

So we get (starting right top corner) 4 → 2 → 1 → 1 → 1→ 0

</details>

> 💡 To generate additional practice examples, Peter Kleiweg has a great [demo](https://www.let.rug.nl/~kleiweg/lev/) (matrix is vertically flipped, though). Also, old lecture recordings have a full walk-through (starts at **30:04**). See below:

[Video Link Placeholder: tue.video.yuja.com]