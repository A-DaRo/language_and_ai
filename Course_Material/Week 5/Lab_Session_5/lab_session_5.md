```markdown
# 🪜 Lab Session 5

| Property | Value |
| :--- | :--- |
| **Week** | Week 5 |
| **Book Chapters** | Self-study |
| **Slides** | Empty |
| **Recordings** | Empty |
| **Solutions** | Empty |

***

> 💡 Given that this is a follow-along session (turned into a ton of reading, too, actually), no solutions are available.

**Table of Contents**

- [🏃‍♂️ The Perils of Rushed Projects](#%F0%9F%8F%83%E2%80%8D%E2%99%82%EF%B8%8F-the-perils-of-rushed-projects)
- [🧱 Structuring Research Code](#%F0%9F%A7%B1-structuring-research-code)
  - [🚪 Starting Out and Knowing When To Quit Prototyping](#%F0%9F%9A%AA-starting-out-and-knowing-when-to-quit-prototyping)
    - [🚩 Task 1](#%F0%9F%9A%A9-task-1)
  - [🛋️ The `README.md`](#%F0%9F%9B%8B%EF%B8%8F-the-readme.md)
    - [🚩 Task 2](#%F0%9F%9A%A9-task-2)
  - [🏯 Example Structures](#%F0%9F%8F%81-example-structures)
    - [📰 The Section Model](#%F0%9F%93%B0-the-section-model)
    - [🏭 The Factory Model](#%F0%9F%8F%8D-the-factory-model)
    - [🚩 Task 3](#%F0%9F%9A%A9-task-3)
- [🏗️ Structuring Research Output](#%F0%9F%8F%81%E2%80%8D%E2%9A%99%EF%B8%8F-structuring-research-output)
  - [🪛 Hyperparameter Tuning](#%F0%9F%AA%9B-hyperparameter-tuning)
    - [🚩 Task 4](#%F0%9F%9A%A9-task-4)
  - [🪵 Experimental Logging](#%F0%9F%A6%9B-experimental-logging)
    - [🚩 Task 5](#%F0%9F%9A%A9-task-5)

***

## 🏃‍♂️ The Perils of Rushed Projects
***

During the interim assignment (or any ML-related project really, especially when it’s a group-based one) you’ll need to maintain your sanity by making sure your code is structured. General software development and project management practices provide fairly high-level standards and frameworks that either declare structures or provide tooling for it. Unfortunately, due to the generally small scale of operations that make up [individual research projects](individual%20research%20projects), and the heavy reliance on libraries, applying such standards to research code typically causes massive overhead (time vs pay-off). Hence, researchers often forego applying any of such practices.

Another somewhat unique property of writing academic code is that, more often than not, the end state of a project is a moving target. Moving, but also non-linear. There are often no certain building blocks that determine an acceptable result. Given that in the interim assignment you’re not necessarily dependent on positive outcomes (e.g.: we found an effect! or: our ideas produced a state-of-the-art model!), you may experience this less. However, imagine you run all your ideas and you get nothing as a result. That’s weeks of time thinking, coding, and running models to result in “well, that clearly did not work”. You can stop there for the assignment, however, in reality this means back to the drawing table. Identifying if you missed something, can add something, can improve something, debugging, you name it. Your code is going to do this:

[Image Placeholder: "THE HORSE" by Ali Bati.]
“THE HORSE” by [Ali Bati](https://www.alibati.com/work/horse).

ML researchers, very much like students, are tied down by conference deadlines. If you have a functional lab (read: some ivy league or FAANG lab) you coordinate and slowly roll out your work at different venues during the year. However, from the point of view of a PhD student, having to generally produce/ideally publish North of four papers for a dissertation, you only get about two shots a year to make a deadline. Why is this monologue of mine important? Knowing under what conditions code is written (stress, lack of sleep, last minute panic after finding research-breaking bugs, inevitably being scooped at least once while working on idea) tells you a lot about why academic code is often so bad, or not shared at all. After making a deadline it’s often not polishing time (as such time is seldomly rewarded in any way in academia), it’s off to the next part of the rat race—on to the next conference deadline.

Replace conference with assignment and it should be somewhat relatable, right? Now, what if I told you (insert Morpheus here), life doesn’t have to be like this? I’d be gravely misrepresenting what to expect! Not everyone has the luxury or the (personal) workflow to structure everything beforehand and keep the kitchen spotless. Best practices are the first to go when time is limited. There is no hack to this when you’re in this situation. So, what [can] we reasonably do? Much like this cooking analogy, you can set up a particular way, and clean afterwards. In between, we can apply some small heuristics to make sure we don’t blow up the blender and have marinara sauce splattered around to make for a culinary horror scene nobody wants to do cleanup on.

So, I’ll try to characterize a typical modern workflow. No judgement on my end, I have definitely done my fair share of projects that follow this for a little while:

*   You get an idea, some data, and your first idea is to explore. Fun times ahead!
*   Jupyter, or anything implementing it (like Google Colab), seems like a good place to start; we can load and look at the data, write some preprocessing code.
*   Little scripts and library snippets start accumulating.
*   Alright, data is prepped, write it all out to files.
*   At this point, we’re working in Jupyter anyway, seems like hassle to move everything over. Let’s try some models for now and see if they work.
*   They don’t work. Maybe add some more. Nope, also not really working. What was the eval setup again?
*   Hmm, I need to tweak the pipeline a bit, but I don’t want to lose my old code. Duplicate the notebook. Make changes.
*   There’s clearly something wrong in the data loading, maybe patch a few blocks in to those files and run everything again.
*   Ok it seems to work now. I should have documented a bit because it took way too long to reverse-puzzle the required execution order of our notebooks together.
*   Let’s plot a few things now that we have the models cached in the memory of our Jupyter instance.
*   Oops, this model is clearly overfitting. Let me check all the components again.
*   That took way too much time again. Let me write some docs in markdown around the cells. That took a lot of time too!
*   I fixed it. But now my model doesn’t beat the baseline. I’ll try again some other time.
*   Okay I just ran everything again with the same configuration and it now beats the baseline. What?
*   Guess we’ll just write this up and pretend like this was the idea all along.
*   Okay so I guess this is the point where we… structure everything? *looks at codebook mess*.

If even part of this is relatable (apologies for the long bit), hopefully this lab session will be useful. If it isn’t, you’re clearly a seasoned coding veteran and I applaud your practices; they are better than mine. For the others, let’s discuss a few points that improve this process. We don’t need to do away with it altogether! That’d be pretending we don’t have our own preferences and everyone ought to be doing the same thing, which is some heated purist train I stay away from.

## 🧱 Structuring Research Code
***

Jupyter is great in some ways, and horrible in many others. If you have time (some other time, likely), I very much recommend watching this talk by Joel Grus (a fantastic scientific developer who at the time worked for AllenNLP) which very nicely illustrates my, and additional concerns:

[Video Placeholder: Can't reproduce the talk here]

### 🚪 Starting Out and Knowing When To Quit Prototyping
***

Whether or not you use Jupyter, you may start out prototyping in small files without much thought behind their structure (they might be one workflow per file, for example).

If you do work with Jupyter, some good initial practices are the following:

*   Create a separate directory for Jupyter files. Your objective should be to clear this directory out as much as you can during the project by constantly merging things into `.py` files.
*   Keep your exploratory phases short, and move into an IDE after as many as a few cells.
*   Keep workflows that lend themselves for Jupyter (such as plotting) in another, separate directory. These files should really only work with output created elsewhere, and only provide a ‘view’ of things (for those familiar with [model-view-controller patterns](https://en.wikipedia.org/wiki/Model%E2%80%93view%E2%80%93controller)).

> 💡 A good replacement for Jupyter that works in things like the VSCode’s console (or a simple terminal) is [IPython](https://ipython.readthedocs.io/en/stable/) (`python -m IPython` or just `ipython` after installing). It’s more sequential, allows for fast prototyping, and it can run and import variables / objects from a file using `%run myfile.py`.

A core practice that’s worth internalizing when prototyping is the following:

*   Determine which things you implement now may be made more ‘general’ later. Some examples: loading your data; why only make it work on one file? Tease out the functions that might also work on other files. Put it into a class! Splitting the data? Same deal. Applying the same splitting method on different types of data assures they are processed similarly. Running preprocessing, vectorization, and models separate? Why not put them in a pipeline?
*   Think about what a user may reasonably want to run from your entire experiment. Presumably, they wouldn’t want to string ten functions together to process your data. Wrap things classes that control common processes; let the user manipulate the configurations of those processes only. Good modularization also allows for the user to add their own stuff. Several common classes might for example be:
    *   `Dataloader` — reads from a file, preprocesses, splits the data. Outputs splits.
    *   `Descriptives` — reads a data state (raw or preprocessed / split) and reports some corpus statistics etc. Bonus points if it outputs [LaTeX tables](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_latex.html).
    *   `Modeler` — provided with a set of (baseline) models, hyper-parameter settings, and the data. Fits models, optionally makes predictions, and outputs the states of the best (and maybe intermediate) model(s) and the predictions (e.g. on test).
    *   `Logger` — optional; usually ‘wraps’ around the `Modeler` class or is called within it. Keeps track of experimental state (especially if models run for a long time). Writes log files for you to check progress / issues.
    *   `Evaluator` — using the prediction files, runs eval metrics and qualitative evaluations (optionally their plots too, but if you want to do those in Jupyter that’s fine too).
    *   `Tester` — highly optional but very good practice. You might want to write ([nose](https://nose.readthedocs.io/en/latest/index.html)) tests that run some dummy data / models through your pipeline. The tests should ideally test assumptions about data / model states via `assert` statements and such (minimal tutorial [here](https://nose.readthedocs.io/en/latest/writing_tests.html)). For example: if I run tf\*idf over a dataset and only want 80,000 features; the return object should be data\_size\*80000. If it isn’t, something has gone wrong, and we should throw warnings. This is especially useful to run after every change you make in the code above. It (hopefully) prevents your code from breaking in unforeseen ways while working.
    *   `Experiment` — this can serve as a configuration class that calls all above elements in the correct order and passes user settings to your experiment. You can have the user call the class with arguments to pass those, make the user pass a `settings.yml` or `settings.json` to the Python file, or use [argparse](https://docs.python.org/3/library/argparse.html).
*   Ideally, these are all in separate files, but you approach may vary (we’ll cover some more concrete variations later).
*   Use small `#` comments while working, merge them into docstrings later. If you are nice about naming your variables so they make sense, most 10-20 line functions should be readable without a comment per line.
*   Keep a `.md` (or Notion) file somewhere in which you can write high-level documentation **while writing your code**. Don’t treat this as an afterthought. It will cost way too much time to write docstrings and documentation later, and it can be a nice procrastination break from trying to debug your own coding.

### 🚩 Task 1
***

Below are several code cells (taken from `Untitled8.ipynb` on my server) that have arrived at a point where prototyping has gone far enough (imo). Use the principles laid out above to clean up and structure this into a project with several `.py` files and ideally classes. Try to focus on which parts are too specific and could be generalized.

```python
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups(subset="train")
y, X = data.target, data.data
y, X = zip(*[(yi, xi) for yi, xi in zip(y, X) 
           if yi in [2, 4, 3]])
```

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

tfidf = CountVectorizer()
X_tf = tfidf.fit_transform(X)
nb = MultinomialNB()
nb.fit(X_tf.todense() * 1000, y)
```

```python
feature_names = tfidf.get_feature_names()
coefs_with_fns = sorted(zip(nb.feature_log_prob_[0], feature_names))
top = zip(coefs_with_fns[:50], coefs_with_fns[:- (50 + 1):-1])
for (coef_1, fn_1), (coef_2, fn_2) in top:
    print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))
```

```python
coefs_with_fns[:10]
```

```python
coefs_with_fns[-10:]
```

```python
coefs_with_fns = sorted(zip(nb.feature_log_prob_[1], feature_names))
coefs_with_fns[-10:]
```

```python
%matplotlib inline
import matplotlib.pyplot as plt

values, features = zip(*coefs_with_fns[50000:50050])
plt.figure(figsize=(15, 2)) 
plt.bar(features, [-1 * x for x in values])


plt.xlabel('Features')
plt.ylabel('Values')
plt.title('Coefficients')
plt.xticks(rotation=45)

plt.show()
```

### 🛋️ The `README.md`
***

In the following sections, we’ll go over 3 of my research repositories so you can get a sense of what’s expected from ‘functioning’ open-source code bases. These are all on GitHub. You don’t have to use GitHub (or git) to do this of course, it’s an industry standard we don’t have time for to cover in this course. A few online tutorials should get you quite far if you’re interested though!

The repositories we’ll cover have a similar `README` structures, so let’s focus on the more ‘advanced’ one for this section, which can be found here:

[GitHub Link Placeholder: Repository for the experiments described in "Adversarial Stylometry in the Wild: Transferable Lexical Substitution Attacks on Author Profiling" presented at EACL 2021.](https://github.com/cmry/reap)

You can ignore the code, we just focus on the `README.md` file. You can open `README.md` if you actually want to see the markdown syntax for the file, but reading it from the main page is enough for now. A rough description of the information that’s conveyed here:

*   A header with vital information. People using your code (generally rare, but definitely not inconceivable) should know the what/who/where/why answers fairly quickly. Hence, for academic code, I generally always include the following:
    *   A link to the paper.
    *   Where it was/will be published (gives more credibility to the source).
    *   The [distinct] software license(s) for the code AND the data (if provided).
    *   The `.bib` file to cite the work (important for those h-index gains).
*   A section with some generally useful points for reproduction:
    *   A tl;dr which highlights some points why someone who found your research code should care about this repository.
    *   Instructions on how to reproduce the results in the paper (and how to get the data to do so), and what system it was built on (I generally provide Python version and OS, could be better, but it’s something).
    *   Dependencies and their versions. This is generally best formatted in a `requirements.txt` but I like to put it here too.
    *   Resources required. What kind of CPU/GPU it was ran on, and how long that took. Bonus points if you calculate CO2 emissions (see e.g. [here](https://mlco2.github.io/impact/#compute)).
*   A section dedicated to experimental manipulation. What elements can be changed to change the experiment? Where do we change those? As you can see I even have specific line numbers in these (it’d probably be better if they were linked, but anyway).
*   Ideally: a section on how to add to the research code. Which components are modular and can be swapped out? How does one do that?

I wouldn’t want to claim that this is how you make ‘successful’ repositories. Clearly, mine are not widely starred nor used (with some minor exception being our Dutch [embeddings](https://github.com/clips/dutchembeddings)). However, it will definitely beat the cr\*p out of the vast majority of what is available! 🙂​

### 🚩 Task 2
***

Write a `README.md` for either the code from Task 1, or your interim assignment (probably more useful) by applying the suggested structure above. You may also be able to find other tutorials / suggestions online.

### 🏯 Example Structures
***

Generally, both of these amount to one thing (also a component of the interim assignment): repositories that can run ‘end-to-end’; i.e., provided with data, they can produce experimental results that (closely) match what is in a paper.

#### 📰 The Section Model
***

[GitHub Link Placeholder: Repository for the experiments and dataset described in "Simple Queries as Distant Labels for Predicting Gender on Twitter" presented at W-NUT 2017.](https://github.com/cmry/simple-queries)

For this 👆 repository, I structured the different functions and classes according to their section in the paper. Let’s go over them file-by-file:

*   `misc_keys.py` — this is where my own Twitter (RIP) API keys went (which I of course removed), and a user of the repository could provide their own.
*   `sec3_data.py` — this file collects the data. It retrieves timelines from users that posted gender-based self-reports. It writes the collected data to a simple file ‘database’ which writes JSON lines out to files. The `DB` class just provides some convenient syntax for it. As you can see, it also incorporates some data readers from papers that have hand-labeled Twitter corpora for gender classification. It inherits the main reader (that queries the Twitter API), and the classes just have specific methods to specify what user IDs to collect.
*   `sec3_proc.py` — starts with a `AnnotationStats` class for descriptives that would probably have been better in its own file, but anyway. Other than that, this file mostly batches tweets into single instances of up to 200 tweets, and applies a bunch of preprocessing to those batches. It then splits them into sets, so the result are cleaned data splits. Note that these splits have already been prepped in fastText format (with `__label__n` prepended everywhere, see line 183).
*   `sec4_exp.sh` — this is just a wrapper for running fastText when they didn’t have a Python library yet. 🙂 It accepts the directory of the model, the output, and the data directory as arguments (it’s a bash/‘shell’ file, which runs on Unix systems).
*   `sec5_res.py` — at the bottom you can see that the boilerplate code at the bottom initiates a majority baseline, a lexicon-based classifier, and calls the fastText scripts, and then runs these models on the different pre-prepared splits in `./data`.

#### 🏭 The Factory Model
***

[GitHub Link Placeholder: Repository for the experiments described in "Current Limitations in Cyberbullying Detection: on Evaluation Criteria, Reproducibility, and Data Scarcity" submitted as pre-print to arXiv.](https://github.com/cmry/amica)

For this 👆 repository, I used what is called a ‘factory’ class (multiple, actually), which calls other classes with required parameters and in the right order. You can see that we’re getting a bit more into the vibe from [here](#29d979eeca9f81d1b740db7ee2fb933f). Let’s go over this file-by-file:

*   `evaluation.py` — the `Evaluation` class handles a lot of things: cross-validation, tuning, reporting on feature importances, oversampling, storing / logging the results, and yes, doing the actual scoring and reporting!
*   `experiments.py` — this is the big factory file. It receives a bunch of command-line arguments (through `argparse`) and then initiates one of the `Experiment` factory classes (you can see it also does some sklearn `Pipeline` selection beforehand, which starts at line 141). We can focus on `EnglishCompare` for now, where you can for example see that `Reader` and `Evaluation` are called, and some training set merging is going on (`_cross_data`).
*   `models.py` — this file specifies all models used in `experiments.py`; it structures things like a majority baseline and WordEmbedding classifiers into the sklearn API so they can be used with the sklearn `Pipeline` and their tuning / evaluation.
*   `neural.py` — same deal as `models.py` but for neural models specifically. Why in a separate file you may ask? This part was used for the infamous reproduction portion (covered in the lecture), and I therefore wanted to have these split.
*   `reader.py` — has the `Reader` class and a `Dataset` class. Does data loading, preprocessing, splitting, and merging (so all data-related operations). Note that because this is supposed to run end-to-end, there is no writing to file here.
*   `utils.py` — this file is mostly used to run the ‘debugger’ (i.e., the unit tests) which apparently couldn’t use an imported function so this file also features the code for sklearn’s classification report…

> 💡 If you want some minimal examples of unit (nose) tests, you can find those [here](https://github.com/cmry/reap/tree/main/debug). Note that these all involve importing classes and then applying some operations on small / fake data and inputs.

### 🚩 Task 3
***

Decide on an implementation style. You can apply its structure to the code from Task 1, or (at least conceptually) think about how this would work for your interim assignment project. Does your code end up running end-to-end?

## 🏗️ Structuring Research Output
***

Alright, enough rambling about code. Let’s get some tools going that help us live a more comfortable research life! These are suggestions, and I will point you to parts that are relevant for the interim assignment. You do not [need] to use these.

### 🪛 Hyperparameter Tuning
***

Annoying to implement yourself even when using scikit-learn (especially if you want to keep it fast). Grid Search is slow. Why not use a library? There are several that only need a few lines to do more fancy hyper-parameter tuning. Some popular ones are:

[Link Placeholder: Ray Tune: Hyperparameter Tuning — Ray 2.8.1]
Works with Scikit Learn, XGBoost, transformers, and can even implement other hyperparameter tuners!

[Link Placeholder: Optuna - A hyperparameter optimization framework]
Works with Scikit Learn and XGBoost (no transformers apparently). Has some neat extensions.

[Link Placeholder: Hyperopt Documentation]
Used to be popular but the ones above definitely caught up and improve upon hyperopt.

### 🚩 Task 4
***

Follow / try any of the scikit-learn tutorials:

[Link Placeholder: Tune’s Scikit Learn Adapters — Ray 2.8.1]

[Link Placeholder: github.com/optuna/optuna-examples/blob/main/sklearn/sklearn_simple.py](files/18-sklearn_simple.py)

[Link Placeholder: GitHub - hyperopt/hyperopt-sklearn: Hyper-parameter optimization for sklearn]
This requires a separate package which is a bit annoying.

## 🪵 Experimental Logging
***

When running experiments, you have to deal with an exponential amount of settings and associated results. It’s easy to lose track of those, and what worked under which conditions. To help you with this, there is a multitude of tools available. My personal favorite over the last few years has been Weights & Biases (wandb for short):

[Weights & Biases Link Placeholder: wandb.ai]

You can create a free personal account (student / researcher licenses are also available). Linking existing code from things like Scikit-Learn, PyTorch (Deep Learning software), HuggingFace libraries, and XGBoost to their interface only requires a few lines of code and if you are careful with structuring all meta-data around your experiments, this can prove a super useful tool.

They provide tooling to store experimental results, models (and their versions), and very useful hyper-parameter tuning and visualization (via Sweeps). Most of this will be quite the time commitment, but it will pay off in the end if you start with it for a project.

### 🚩 Task 5
***

**Optional; if you don’t want to register for their service: no problem**.

Walk through the quickstart:

[Link Placeholder: Quickstart | Weights & Biases Documentation](https://docs.wandb.ai/quickstart)

Follow the scikit-learn tutorial:

[Link Placeholder: Scikit-Learn | Weights & Biases Documentation](https://docs.wandb.ai/guides/integrations/scikit)

It’s also available as a notebook:

[Link Placeholder: Google Colaboratory Notebook for Simple Scikit Integration](files/23-Simple_Scikit_Integration.ipynb)
```