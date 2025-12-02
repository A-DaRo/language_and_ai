```markdown
# 📚 Introduction

| Property | Value |
| :--- | :--- |
| **Week** | Week 1 |
| **Book Chapters** | Empty |
| **Slides** | [week1.pdf](week1.pdf), [week1_inperson_slides.pdf](week1_inperson_slides.pdf) |
| **Recordings** | [tue.video.yuja.com/P/V…448486](https://tue.video.yuja.com/P/VideoManagement/MediaLibrary/Users/u-oD0ckDKj/MyMediaCollections/e1da2362-5417-4be4-9f03-b74947ac789e/WatchVideo/5448486) |

***

**Table of Contents**

- [🎽 Warm-up Questions](#%F0%9F%8E%BD-warm-up-questions)
- [📺 Lecture Videos](#%F0%9F%93%BA-lecture-videos)
  - [1️⃣ Personal Introduction](#%F0%9F%87%B9%E2%83%A3-personal-introduction)
  - [2️⃣ Course Information](#%F0%9F%87%B8%E2%83%A3-course-information)
  - [3️⃣ Why is Language Difficult?](#%F0%9F%87%B7%E2%83%A3-why-is-language-difficult)
  - [4️⃣ Text Mining Preliminaries](#%F0%9F%87%B6%E2%83%A3-text-mining-preliminaries)
- [📏 Distances](#%F0%9F%92%AC-distances)
  - [🚩 Task 1](#%F0%9F%9A%A9-task-1)
- [⚖️ tf\*idf](#%E2%9A%96%EF%B8%8F-tfidf)
  - [🚩 Task 2](#%F0%9F%9A%A9-task-2)

***

## 🎽 Warm-up Questions
***

> ℹ️ These questions are intended to prime your brain for the materials we will be covering in the videos below. They are optional, and there are no expectations wrt the answers.

*   Text is (on a surface level at least) a fairly low-fi medium; there are some characters, generally limited to whatever is included in [unicode](https://unicode-table.com/en/). These characters form sequences, which we understand as words. Words form sentences, which form paragraphs, which form a text. If a computer were to process a text, a first step could be to split this text into meaningful units. This process is called tokenization. What do you think would be a good heuristic to tokenize a text? Or rather: how to best chop up this blob of characters that is a text?
*   Computers display sequences of characters (which you may know as strings) pretty well—as demonstrated by the text on this page). However, doing any sort of math to extract patterns from these strings is difficult. Can you think of a way to represent words in a particular text as numbers? Can you think of ways this representation might be shared between different pieces of text?
*   Let's evaluate your idea from the previous question: does it have a fixed size (meaning, will the size of the representation be the same if the sentences are longer or shorter)? Does it provide information about the structure (or order) of the sentences? Do you have an idea how to add these kind of features to your representation?

## 📺 Lecture Videos
***

> 🖼️ Slides will be made available at the top of the page!

### 1️⃣ **Personal Introduction**
[Video Placeholder]

### 2️⃣ **Course Information**
[Video Placeholder]

### 3️⃣ **Why is Language Difficult?**
[Video Placeholder]

### 4️⃣ **Text Mining Preliminaries**
[Video Placeholder]

> 💡 Please note that the Student Hours have been replaced by an in-person lecture on Thursdays.

## 📏 Distances
***

Remember that our Euclidean Distance function can be defined as 👇​
$$\sqrt{\sum_{i=1}^{n}\left(\vec{p}_{i}-\vec{q}_{i}\right)^{2}}$$

The Jaccard coefficient as 👇​
$$J(A,B) = \frac{| A \cap B |}{| A \cup B |}$$

And the Cosine Similarity as 👇​
$$\frac{\vec{p} \bullet \vec{q}}{ \sqrt{\vec{p} \bullet \vec{p}} \cdot \sqrt{\vec{q} \bullet \vec{q}}}$$

> 💡 Note that $\sqrt{\vec{p} \bullet \vec{p}}$ is the $\ell_2$ norm for vector $\vec{p}$. If you $\ell_2$ normalize the full space, the denominator drops, and it's simply $||\vec{p}||_2 \bullet ||\vec{q}||_2$ to get the cosine similarity.

Here $\bullet$ explicitly denotes the dot product 👇​
$$\sum_{i=1}^n \vec{p}_i \cdot \vec{q}_i$$

### 🚩 Task 1
***

Given the following set of vectors:
$$\begin{array}{l|ll}
\vec{x_1} & 6.6 & 6.2 \\
\vec{x_2} & 9.7 & 9.9 \\
\vec{x_3} & 1.3 & 2.7 \\
\vec{x_4} & 1.3 & 1.3 \\
\end{array}$$

*   Calculate the Euclidean Distance between vector 1 and vector 4.
<details>
<summary>**Solution (Full)**</summary>
$$\sqrt{(6.6 - 1.3)^2 + (6.2 - 1.3)^2} = 7.218$$
</details>

*   Calculate the Cosine Similarity between vector 2 and 3.
<details>
<summary>**Solution (Full)**</summary>
$$\frac{9.7 \cdot 1.3 + 9.9 \cdot 2.7}{\sqrt{9.7 \cdot 9.7 + 9.9 \cdot 9.9} \cdot \sqrt{1.3 \cdot 1.3 + 2.7 \cdot 2.7}} \approx \frac{39.34}{13.86 \cdot 2.997} \approx 0.947 $$
</details>

And given:
$$\begin{array}{l|llllllllll}
\vec{x_1} & 1 & 0 & 1 & 1 & 1 & 0 & 1 & 1 & 0 & 0 \\
\vec{x_2} & 0 & 1 & 0 & 1 & 1 & 0 & 1 & 1 & 0 & 0 \\
\end{array}$$

*   Calculate the Jaccard coefficient between vector 1 and 2.
<details>
<summary>**Solution (Full)**</summary>
$$\frac{4\ \text{(total shared words)}}{7\ \text{(total words in either)}} = 0.571$$
</details>

## ⚖️ tf*idf
***

Remember that:
$$w_{t,d} = \log \left( \text{tf}(t,d) + 1 \right) \cdot \log_b \frac{N}{\text{df}_t}$$

Here, $\text{tf}$ is the term frequency, or 'how many times does the word/term $t$ occur in a document $d$'. Then, $\text{df}$ is the document frequency, or 'how many documents does the word $t$ occur in'. So, $\text{tf}$ is inferred from instances, whereas $\text{df}$ is inferred from the dataset—i.e., the collection of documents. $N$ is the total amount of documents, $b$ is a base (typically 10), $\log$ is $\ln$, $\log_{10}$ is $\lg$.

### 🚩 Task 2
***

*   Given the three documents below, calculate the term frequency matrix.
    > the cat sat on the mat
    > my cat sat on my cat
    > my cat sat on the mat on my cat

<details>
<summary>**Solution (Full)**</summary>
First we determine all the unique words (our vocabulary): the, cat, sat, on, mat, my. These our our features. The documents are our instances. We list the term frequency per document for each feature (term), as such:
$$\begin{array}{lllllll}
\hline
\# & \texttt{the} & \texttt{cat} & \texttt{sat} & \texttt{on} & \texttt{mat} & \texttt{my} \\
\hline
1 & 2 & 1 & 1 & 1 & 1 & 0 \\
2 & 0 & 2 & 1 & 1 & 0 & 2 \\
3 & 1 & 2 & 1 & 2 & 1 & 2 \\
\hline
\end{array}$$
</details>

*   What is the $\text{df}$ for $\texttt{cat}$, $\texttt{the}$ and $\texttt{my}$ ?
<details>
<summary>**Solution (Full)**</summary>
Cat occurs in all 3 documents, the in 2, and my in 2 as well. 
</details>

*   Convert the $\text{tf}$ matrix from the previous question to a $\text{tf}\cdot\text{idf}$ matrix with base 10.
<details>
<summary>**Solution (Full)**</summary>
Given:
$$\begin{array}{lllllll}
\hline
\# & \texttt{the} & \texttt{cat} & \texttt{sat} & \texttt{on} & \texttt{mat} & \texttt{my} \\
\hline
1 & 2 & 1 & 1 & 1 & 1 & 0 \\
2 & 0 & 2 & 1 & 1 & 0 & 2 \\
3 & 1 & 2 & 1 & 2 & 1 & 2 \\
\hline
\end{array}$$
For the first row, we have, for example:
$$w_{\texttt{the},1} = \log \left( \text{tf}(t,d) + 1 \right) \cdot \log_b \frac{N}{\text{df}_t} = \ln \left(2 + 1 \right) \cdot \lg \frac{3}{2} = 0.193$$
$$w_{\texttt{cat},1} = \log \left( \text{tf}(t,d) + 1 \right) \cdot \log_b \frac{N}{\text{df}_t} = \ln \left(1 + 1 \right) \cdot \lg \frac{3}{3} = 0.000$$
$$w_{\texttt{mat},1} = \log \left( \text{tf}(t,d) + 1 \right) \cdot \log_b \frac{N}{\text{df}_t} = \ln \left(1 + 1 \right) \cdot \lg \frac{3}{2} = 0.122$$

> 💡 Tip: because $\text{df}$ is static across instances for the same term, you can typically quickly find the same values for these low frequency documents.

$$\begin{array}{lllllll}
\hline
\# & \texttt{the} & \texttt{cat} & \texttt{sat} & \texttt{on} & \texttt{mat} & \texttt{my} \\
\hline
1 & 0.194 & 0.000 & 0.000 & 0.000 & 0.122 & 0.000 \\
2 & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 & 0.194 \\
3 & 0.122 & 0.000 & 0.000 & 0.000 & 0.122 & 0.194 \\
\hline
\end{array}$$
</details>
```