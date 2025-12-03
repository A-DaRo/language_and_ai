# 📜 Lab Session 1

| Property | Value |
| :--- | :--- |
| **Week** | Week 1 |
| **Book Chapters** | Empty |
| **Slides** | Empty |
| **Recordings** | Empty |
***

> 💡 Welcome to the first lab session! [Before you start this practical session](https://example.com/before-starting): this introduction practical requires coding up most things yourself. Generally, we will mostly be working with libraries to do this.

***

**Table of Contents**

1.  [🐍 Python Preliminaries](#python-preliminaries)
    *   [📦 Object-Oriented Programming](#object-oriented-programming)
2.  [✂️ Tokenization](#tokenization)
    *   [🚩 Task 1](#task-1)
3.  [✅ Testing Your Function](#testing-your-function)
4.  [🚀 Vectorization](#vectorization)
    *   [🚩 Task 2](#task-2)
5.  [📊 Document Similarities](#document-similarities)
    *   [🚩 Task 3](#task-3)
    *   [🧰 In Practice](#in-practice)
6.  [⭐ Bonus: Putting Things Together, and Extending Them](#bonus-putting-things-together-and-extending-them)

## 🐍 Python Preliminaries

***

Based on the course survey, I assume many of you have a solid basis in Python. Before we start the lab sessions, and because this is the introduction lecture anyway, I would like to dedicate some time to explaining best practices for writing research code in ML (these draw from libraries such as [Scikit-learn](https://scikit-learn.org/stable/)). If you're interested in a primer on ML reproducibility, see the [course readings](https://example.com/ml-reproducibility). Here, you can read that one of the recommendations is not to use Notebooks for research code (🙂).

For this course, it's probably best to work in something you're comfortable with, but it wouldn't hurt to invest some time in using a nice IDE (e.g., [Atom](https://atom.io/), [VSCode](https://code.visualstudio.com/), etc.). One of the advantages is you can use [Linter](https://code.visualstudio.com/docs/python/linting) integration to check your code for style guide adherence, such as [PEP 8](https://pep8.org/). This is an important component to improving the readability (and consistency) of your code ([readability counts](https://www.python.org/dev/peps/pep-0020/)), which again is good for reproducibility (say you open-source your code on [GitHub](https://github.com/), you'd want people to be able to read it fairly well). Notebooks are not a great vehicle for this. They are a great vehicle for text describing code, though, so I'll provide them either way!

Research code can get quite convoluted. Machine Learning (and Text Mining) code typically tests combinations of different algorithms, ways to process the data, etc. and what influence they have on final results. Hence, it's beneficial to structure code according to these 'pipeline' components, which might re-use core functionality and build on each other only with select functionality. This is why many of the libraries that we will use, have an Object-Oriented Programming (OOP) design—which we'll adhere to as well. If you're unfamiliar with OOP, you can skip this section.

### 📦 Object-Oriented Programming

***

OOP in Python treats everything as an `object` or a `class`. There are many good tutorials that explain its workings and advantages (see, e.g., [here](https://realpython.com/python3-object-oriented-programming/) and [here](https://realpython.com/inheritance-composition-python/)). I'll provide a very brief overview here (this practical should illustrate its use fairly well). Typically, when learning Python, you know how to write functions. For some very arbitrary (not course-related) examples, we have:

```python
def add(x, value):
    return x + value

print(add(3, 2))
print(add(0, 1))
```

Now, let's assume we write a small application where we have a shopping cart. We want to be able to add stuff to it, remove things from the cart, and calculate a total price + VAT. In the end, we print the cart's current contents, and the total price. If we do this with functions, it might look something like so (really quick example, excuse running complexity etc. 🙂):

```python
def item_operation(shopping_cart, operation, name, price):
    if operation == 'add':
        shopping_cart.append((name, price))
        return shopping_cart
    elif operation == 'remove':
        shopping_cart.remove((name, price))
        return shopping_cart
    else:
        return shopping_cart

def total_price(shopping_cart, vat):
    return sum([item[1] for item in shopping_cart]) * (1 + vat)

shopping_cart = []
shopping_cart = item_operation(shopping_cart, 'add', 'milk', 0.79)
shopping_cart = item_operation(shopping_cart, 'add', 'milk', 0.79)
shopping_cart = item_operation(shopping_cart, 'remove', 'milk', 0.79)
shopping_cart = item_operation(shopping_cart, 'add', 'butter', 1.49)
shopping_cart = item_operation(shopping_cart, 'add', 'eggs', 2.89)
total_price = total_price(shopping_cart, 0.20)

print(shopping_cart, total_price)
```

This is fine, fully functional and doesn’t look too bad. Typically, you can do things with functions to a large extent without things getting too hairy (if you, e.g., use `global` scopes). Anyway, with OOP, the same code would look somewhat like: 		

```python
class ShoppingCart(object):

    def __init__(self, vat):
        self.contents = []
        self.vat = vat

    def add(self, name, price):
        self.contents.append((name, price))

    def remove(self, name, price):
        self.contents.remove((name, price))

    def total_price(self):
        return sum([item[1] for item in self.contents]) * (1 + self.vat)
        
cart = ShoppingCart(vat=0.20)
cart.add('milk', 0.79)
cart.add('milk', 0.79)
cart.remove('milk', 0.79)
cart.add('butter', 1.49)
cart.add('eggs', 2.89)

print(cart.contents, cart.total_price())
```

Classes have this special `__init__` function, which is called when the class is initialized (or instantiated) with `ShoppingCart()`. This is typically the place to define what are called class 'attributes'; class-wide variables we can access in its methods (functions) through `self` (which points to the class / object itself). All regular class method definitions therefore start with this (`def add(self, `). These attributes can also be accessed outside of the class (see `cart.contents`).

So there are a few advantages here: functions are neatly grouped with the object they belong (particularly for larger pieces of code, that makes for better readability). Furthermore, because we can rely on the class attributes through which we update `self.contents`, we don't need to return the cart to the user of the function all the time. Lastly, `vat` can be passed when initializing the class, so it might be used if we add functionality. It can be updated outside of the class as well (say `cart.vat = 0.30`). None of this is incredibly restricted to OOP, but it is much more convenient in syntax and keeps class attributes 'safe' (not strictly, just from yourself) from accidentally being updated by other functions, for example.

Classes can also inherit functionality from each other, like so:

```python
class A(object):

    def __init__(self):
        self.x = 2

    def add(self):
        return self.x + 1

    
class B(A):
    
    def add(self):
        return self.x + 2

    
class C(A):

    def __init__(self):
        self.x = 6

    def something(self):
        return None    


a, b, c = A(), B(), C()
print(a.add(), b.add(), c.add())
```

An extremely arbitrary example, but it's about as minimalistic as we can go. So `A` defines a class variable `x`, and method `add`. If we define a class as the 'child' of `A` (like `B(A)`) it inherits both the `__init__` and `add` functionality. We can overwrite them, leave them be, add different methods, whatever you want. `B` keeps `init` but overwrites `add`, `C` the other way around. Here, we would describe `A` as the base or parent class.

## ✂️ Tokenization

***

Okay, so let's put this into practice. In the lecture, we looked at converting text into vectors. For now, we'll use a few short test documents, so we can eyeball the results easily:

```python
documents = [
    "this is an example sentence, nothing special happening in this sentence",
    "again a very ordinary sentence without anything special happening",
    "this is a sentence as example. this is another example sentence",
    "yet another sentence, simple and ordinary"
]
```

### 🚩 Task 1

***

Write a function `tokenize` that takes as input a `list` of `documents` where the elements are `string`s. Separate the units of these text strings (e.g., words) into individual strings. As output, `return` them as a `list` of `list`s (where `documents` is the main list, and each element is a `list` of tokens; a list per document). You can use functionality from the `string` type ([docs](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)), and the `string` package ([docs](https://docs.python.org/3/library/string.html)).

<details>
<summary>**Hint 1**</summary>
`split` might help.
</details>

<details>
<summary>**Hint 2**</summary>
`string.punctuation` too.
</details>

<details>
<summary>**Hint 3**</summary>
You might want to use indexing and [slicing](https://www.w3schools.com/python/python_strings_slicing.asp) of strings.
</details>

<details>
<summary>**Hint 4**</summary>
You can append tokens to their own list. If they contain punctuation, you should make sure those are individual elements in that list.
</details>

<details>
<summary>**Partial Solution**</summary>
A 'naive', simple solution which simply splits on whitespace would look like:

```python
def tokenize(documents):
    tokenized_documents = []
    for document in documents:
        tokens = document.split(' ')
        tokenized_documents.append(tokens)
    return tokenized_documents
```

We loop through the documents, split strings by the character `' '` (whitespace), and append it to our list `tokenized_documents`, which we then return. We didn't account for the one `,` though.

<details>
<summary>**Solution**</summary>
The solution I wrote checks if the last character in a token `[-1]` is in the list of what Python considers punctuation `string.punctuation`, then concatenates them (`+=`) in their own list to the `token_list`.

```python
import string

def tokenize(documents):
    tokenized_documents = []
    for document in documents:
        token_list = []
        for token in document.split(' '):
            if token[-1] in string.punctuation:
                token_list += [token[:-1], token[-1]]
            else:
                token_list.append(token)
        tokenized_documents.append(token_list)
    return tokenized_documents
```
</details>
</details>

## ✅ Testing Your Function

***

So I mentioned we would have a self-grader. As the name implies, can run this yourself, without checking the solutions, like so:

*   Get the solutions `.py` from this page (🔝).
*   Put the `lab_1_solutions.py` into your notebook's directory.
*   Run the following code:

```python
from numpy.testing import assert_equal  # only have to import this once
from lab_1_solutions import tokenize as tokenize_solution

try:  # NOTE: make sure your function is in the same file / notebook
    assert_equal(tokenize(documents),
                 tokenize_solution(documents))
    print("Success!")
except AssertionError:
    print("Solution is not identical:")
    print("Your func output:", tokenize(documents))
    print("Solutions output:", tokenize_solution(documents))
```

Note that the functions in the solution file have the same name, so we import them using `_solution` to avoid variable name collision. This piece of code uses `numpy.testing` functionality to evaluate if two things are indeed equal; things, here, being the output of your `tokenize` function, and then the `tokenize_solution`. I added a `try/except` statement to catch the error that `assert_equal` throws if they are not the same, in which case it will print some useful information for you (namely, what the solution file has as output).

> 💡 The solutions for the other tasks will be in the same file, so you can re-use the above piece of code to test your other functions. I'll leave that up to you.

## 🚀 Vectorization

***

Now that the documents are neatly split into tokens, we can start turning them into a term \* document matrix. For now, we'll stick to simple term frequency counts. Remember that we need to construct the 'vocabulary' axis $T$ (for all terms / tokens in all documents), and $D$ for our documents. There are many ways to do this; computational efficiency is not the focus of this course, so do not worry about complexity too much.

### 🚩 Task 2

***

Write a function `vectorize` that takes as input the output of `tokenize` (i.e., `M`, `T`). It should `return` a term \* document matrix (as a `list` of `list`s), where each element is the associated term frequency $\text{tf}(t, d)$. Furthermore, it should `return` the vocabulary $T$ as a `list`.   [Collections](https://docs.python.org/3/library/collections.html) has some interesting functionality for this. Make sure to [sort](https://docs.python.org/3/howto/sorting.html) 𝑇 alphabetically for consistency (check updated solution file above). [Collections](https://docs.python.org/3/library/collections.html) has some interesting functionality for word frequencies.

<details>
<summary>**Hint 1**</summary>
For $T$: An easy way to get the unique elements of a list is to use `set`, you can convert it back to `list`.
</details>

<details>
<summary>**Hint 2**</summary>
To get easy frequencies: `Counter` can take a list as input. It returns a `dict` with the elements as keys and the frequencies as values.
</details>

<details>
<summary>**Hint 3**</summary>
`Counter` also returns zero if a key is not in there.
</details>

<details>
<summary>**Hint 4**</summary>
The easiest way to go about this is to first build $T$ by going through all the documents, and all its tokens. Then after, you can do the same loop, but just through the documents, and go over each element of T to construct a vector, looking up its frequencies in the `Counter`.
</details>

<details>
<summary>**Solution**</summary>
Because of the double `for` loops and no use of arrays, it's a bit of a suboptimal (yet Pythonic-ish) solution, but it works. We first construct $T$ through a one-liner [list comprehension](https://www.digitalocean.com/community/tutorials/understanding-list-comprehensions-in-python-3). Would be the same as writing two loops, then `add`ing everything to a `set`, then converting it to a `list`. Now, we have our `documents` and `T`, so the dimensions of our matrix are there, we just need to unroll the tokens in `documents` into frequencies. We loop, and simply shove our tokens from `document` into `Counter` which gives us a dictionary with count frequencies (you can inspect it with `print` or a debugger break point to confirm). We can then loop through $T$ and use each $t$ to look up the frequency; it will give a zero if it's not in the dictionary, so we end up with the same size for each vector.

```python
from collections import Counter

def vectorize(documents):
    M = []
    T = sorted(set([token for document in documents for token in document]))
    for document in documents:
        vector = []
        freq_dict = Counter(document)
        for t in T:
            vector.append(freq_dict[t])
        M.append(vector)
    return M, T
```
</details>

## 📊 Document Similarities

***

Given our newly constructed term-frequency matrix, it's fairly simple to calculate several distance functions. For now, we'll stick to cosine similarity, which we denote as:

$$
\frac{\vec{p} \bullet \vec{q}}{ \sqrt{\vec{p} \bullet \vec{p}} \cdot \sqrt{\vec{q} \bullet \vec{q}}}
$$

Where $\vec{p} \bullet \vec{q}$ is the dot product.

### 🚩 Task 3

***

Write a function `cossim` that takes as input the output of `vectorize` (i.e., `M`, `T`). It should `return` the indices (two `int`s) of the document pair with the highest cosine similarity (not itself). Note: i) cosine similarity is symmetrical (p\*q == q\*p), ii) you do not need to use `T` for the solution, this is just for provided for function compatibility. You can use `numpy` for the required math.

<details>
<summary>**Hint 1**</summary>
The cosine similarly formula uses [np.dot](http://np.dot) and `np.sqrt` (where `np` is `numpy`).
</details>

<details>
<summary>**Hint 2**</summary>
You need two `range` loops, which you can use to index into `M`.
</details>

<details>
<summary>**Hint 3**</summary>
To only use a triangular half of the matrix, you can use two `range(len(M))` loops, passing the current index of the first as starting index to the second.
</details>

<details>
<summary>**Hint 4**</summary>
You can use a `dict` with a `tuple` of indices as key, and the score as value. This is [sort](https://docs.python.org/3/howto/sorting.html)able. You can use a `lambda` expression to sort on the values.
</details>

<details>
<summary>**Solution**</summary>
The two loop here provide the minimum iterations needed to get the pairs. We use their index values (`i` and `j`) to index twice into `M` (for two respective vectors `p` and `q`), run the cosine similarity formula and add it to the `dict` as `dict[(i, j)] = sim`. This is then `sorted(dict.items(), key=lambda x: x[1], ...` where we added a reverse (so we can do 0, otherwise -1 would work as well) and extract the tuple key from there.

```python
import numpy as np

def cossim(M, T):
    sims = {}
    for i in range(len(M)):
        for j in range(i, len(M)):
            if i == j:
                continue
            p, q = M[i], M[j]
            sims[(i, j)] = np.dot(p, q) / (np.sqrt(np.dot(p, p)) * 
                                           np.sqrt(np.dot(q, q)))
    return sorted(sims.items(), key=lambda x: x[1], reverse=True)[0][0]
```
</details>

### 🧰 In Practice

***

Now, that was all rather inconvenient and a lot of code work. There are various libraries that we will get into in this course that make life much easier. For example, the above pipeline of three functions in `spacy` is shown below. Note that for this practical, it's not required to install `spacy` yet, but you can go ahead if you want to run the code:

```python
import spacy
nlp = spacy.load('en_core_web_sm')

sims = {}
for i in range(len(documents)):
    for j in range(i, len(documents)):
        if i == j:
            continue
        sims[(i, j)] = nlp(documents[i]).similarity(nlp(documents[j]))

print(sorted(sims.items(), key=lambda x: x[1], reverse=True)[0][0])
```

Bam. No need for tokenizing or vectorizing… all out of the box, even the similarity calculations. Just needed to deal with our task of providing the documents in the right order and sorting. Either way, it's good to think about, and learn what's going on behind the scenes. So we'll do a bit of both, generally. Next practical, we'll start working with some real data.

***

## ⭐ Bonus: Putting Things Together, and Extending Them

***

Now that we've written up these three nice functions that all work in unison, we can neatly structure them into a single class, or multiple classes, depending on where we expect to extend functionality. If you have time left, try to implement them OOP style. Even better, try to implement binary vectorization and a different similarity metric. That's when the use of OOP is displayed best. Below is my implementation in two flavors (examples with and without inheritance) if you just want to take a peek. 🙂

<details>
<summary>**OOP Example**</summary>
Binary and TF vectorizers with inheritance:

```python
from collections import Counter
import string


class BaseVectorizer(object):
    
    def __init__(self):
        self.T = set()

    def tokenize(self, document):
        tokens = []
        for token in document.split(' '):
            if token[-1] in string.punctuation:
                tokens += [token[:-1], token[-1]]
            else:
                tokens.append(token)
        self.T.update(tokens)
        return tokens
    
    def get_vector(self, document):
        return NotImplemented

    def vectorize(self, documents):
        M = []
        for document in documents:
            M.append(self.get_vector(document))
        return M
    
    def fit(self, documents):
        return [self.tokenize(doc) for doc in documents]
    
    def transform(self, documents):
        assert self.T  # vocab not fitted
        return self.vectorize(documents)
    
    def fit_transform(self, documents):
        return self.transform(self.fit(documents))
    

class TFVectorizer(BaseVectorizer):
    
    def get_vector(self, document):
        vector = []
        freq_dict = Counter(document)
        for t in self.T:
            vector.append(freq_dict[t])
        return vector
    

class BinaryVectorizer(BaseVectorizer):
    
     def get_vector(self, document):
        vector = []
        for t in self.T:
            if t in document:
                vector.append(1)
            else:
                vector.append(0)
        return vector
    
    
vect = BinaryVectorizer()
print(vect.fit_transform(documents))

vect = TFVectorizer()
print(vect.fit_transform(documents))
```

</details>
---