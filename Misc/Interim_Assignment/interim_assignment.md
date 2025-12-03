# Interim Assignment

[Cover Image Placeholder]

> 💡 Writing advice is largely adapted from [Christopher Potts](https://github.com/cgpotts)’ excellent NLU course.

> ❗ Make sure you have read the [section](/29d979eeca9f8162b84fe74bfbe424fc?pvs=25#29d979eeca9f81b0afedd73a92a554f8) on the assessment of the interim assignment first. The data and task of the assignment will be discussed in the in-person lecture.

**Table of Contents**

- [🥅 Learning Goals](#learning-goals)
- [🔭 Scope](#scope)
- [📝 Research Proposal](#research-proposal)
  - [🔍 Research Questions / Hypotheses](#research-questions--hypotheses)
  - [📚 Literature Review](#literature-review)
    - [🌐 Finding Papers](#finding-papers)
    - [👀 What To Pay Attention To](#what-to-pay-attention-to)
    - [🎽 Starting / Example Papers](#starting--example-papers)
  - [💽 Data](#data)
  - [📊 Evaluation (Metrics)](#evaluation-metrics)
  - [🤖 Models](#models)
  - [💬 General Reasoning](#general-reasoning)
  - [⏳ Progress summary](#progress-summary)
- [📜 Research Paper](#research-paper)
  - [🖌️ Formatting](#formatting)
  - [🎨 Organization](#organization)
    - [🗜️ Abstract](#abstract)
    - [📣 Introduction](#introduction)
    - [🗄️ Related Work](#related-work)
    - [💾 Data](#data-1)
    - [🔬 (Method and) Experimental Setup](#method-and-experimental-setup)
    - [📈 Results](#results)
    - [💭 Discussion and Conclusion](#discussion-and-conclusion)
  - [🖊️ Authorship Statement](#authorship-statement)
  - [💡 Advice on Scientific Writing](#advice-on-scientific-writing)
- [👾 Suggestions for Code Delivery](#suggestions-for-code-delivery)
- [❓ FAQ](#faq)

***

## 🥅 Learning Goals
---
1. Evaluate the suitability of natural language data sources for a data science problem.
2. Implement NLP methods to analyze and transform natural language data.
3. Apply machine learning to language data.
4. Explain the limitations of NLP techniques.

## 🔭 Scope
---
The topical scope for this year is twofold:
* The assignment should use the provided data, which focuses on stylometry, and specifically author profiling. You may deviate from this as long as the second criterion is upheld.
* The version of the data provided for the assignment has data pollution issues across the various tasks. This means there are pieces of text in there that directly relate to the labels (e.g., in-text mentions of gender and age). Your goal is to investigate, and potentially mitigate these issues with the knowledge provided in the course.

To avoid overlap with last year, solely predicting the labels is insufficient. If you have questions about the above, please use the discussion board or practical sessions.

## 📝 Research Proposal
---
This is a short, structured report designed to help you establish your core experimental framework. This component is purely **formative** and **optional**. Making the deadline means you will receive feedback to better guide you to a passing research paper. Please see below for what to expect in terms of feedback.

* **Formatting**: your call. We recommended to already start working in the template for the research paper. File type to be handed in is `.pdf`.
* **Length**: 1-2 pages, with a strong preference for **one**.
* **Deadline**: between weeks 3 & 4. Multiple submissions are possible for quick error corrections only. We generally assume submissions are final; feedback may not cover updated versions.

The required sections for this assignment are as follows:

### 🔍 Research Questions / Hypotheses
---
A statement of the project's core research question, or hypothes(i/e)s. These may vary widely and can be oriented towards issues not just in NLP but also in machine learning, social sciences, digital humanities, and any other field that fits the data.

You may argue you just want to see how a new model does on a particular task, that this is a goal, and there are no questions or hypotheses. However, your work will benefit greatly from framing a more precise research question or hypothesis. For example, one might identify a particular preprocessing step and hypothesize that it is crucial for success. This resolves the issue of which models to compare – we need to see results from a model with this component and a minimally different one without it, so that we can isolate the effects of this component. You can greatly improve your research paper’s Introduction section by being clear about your questions or hypotheses.

### 📚 Literature Review
---
The literature review is an essential part of an academic workflow and seeks to summarize and synthesize several papers in the area relevant to the provided dataset. It's meant serve as the intellectual foundation for your project. Very little of the actual work makes it in writing. Your goal for the proposal is to outline what you think are the **three** (minimum) **closely-related papers** and why they are relevant.

#### 🌐 Finding Papers
Papers for your literature review can come from anywhere. There is no requirement that they be NLP papers. Indeed, for interdisciplinary projects, the papers really must come from multiple disciplines, else the lit review won't provide a good foundation for your project. Breakthrough ideas in science often come from finding connections across fields, and we want to facilitate that!

> ℹ️ Even finding "papers" might be too restrictive. The items in your lit review could be books (ambitious!), exceptionally good and detailed blog posts, government reports, etc. – we're even open-minded about including things like talks and interview transcripts if they are very rich and important. Relevant ideas are expressed and discussed in lots of ways, and our primary interest is in collecting those ideas and synthesizing them.

All that said, projects for this course are restricted to having a main NLP component (due to the data), so it will pay to spend time in the [ACL Anthology](https://aclweb.org/anthology/). The ACL community has been exceptionally good about collecting all of its published work going back for decades, so searching the Anthology can very quickly give you a sense of the intellectual landscape and take you to the important papers.

For more general searches of the scientific literature, [Google Scholar](https://scholar.google.com/) and [Semantic Scholar](https://www.semanticscholar.org/) have very broad coverage, and their citation counts provide some useful guidance as to which papers are the most important in a given area.

> ‼️ Citation counts don't guarantee quality, but they do suggest influence, and so papers with many citations are probably worth a look. Using arXiv-only papers may also mean rolling the die on paper quality (as those papers are not peer-reviewed).

#### 👀 What To Pay Attention To
While trying to get a sense of the field, and related ones, try to focus on these bits:
1. *General problem/task definition*: What are these papers trying to solve, and why?
2. *Concise summaries of the articles*: Do not simply copy the article texts in full, also not in your notes. If they end up in your paper this would be considered plagiarism, even with attribution and local paraphrasing (i.e., replacing a few words does not make it your own writing). Also, we can read the papers ourselves. Try to make notes for each paper in your own words, focusing on the major contributions of each article.
3. *Compare and contrast*: Focus on describing the similarities and differences of the papers. Do they agree with each other? Are results seemingly in conflict? If the papers address different subtasks, how are they related? (If they are not related, then you may have made suboptimal choices for a lit review...). This part is probably the most valuable for the research paper, as it can become the basis for a related work section.
4. *References section*: The entries should appear alphabetically and give at least full author name(s), year of publication, title, and outlet if applicable (e.g., journal name or proceedings name). BibTeX (part of Beyond that, we are not picky about the format. Electronic references are fine but need to include the above information in addition to the link.

#### 🎽 Starting / Example Papers
[Bookmark: The role of personality, age, and gender in tweeting about mental illness (https://aclanthology.org/W15-1203)]
[Bookmark: Author Age Prediction from Text using Linear Regression (https://aclanthology.org/W11-1515)]
[Bookmark: TwiSty: A Multilingual Twitter Stylometry Corpus for Gender and Personality Profiling (https://aclanthology.org/L16-1258)]

Note that these are all 8 pages and might also offer new data / labels (different from your paper).

### 💽 Data
---
A description of the labels / split setup that the project will use for evaluation. This mostly serves to check if this is aligned with the research questions / hypotheses. If this isn't immediately evident, you will receive feedback to rethink the setup.

### 📊 Evaluation (Metrics)
---
A description of the metrics that will form the basis for evaluation (see course material for references). We sort of expect these to be familiar quantitative metrics, but we're open-minded. We do, however, require that the research paper include a quantitative evaluation of some kind. We will primarily be focused on determining whether the metrics are appropriate given the data and research questions / hypotheses. Especially if you depart from what's standard, or you want to propose your own metric, then you'll need to justify these decisions.

Qualitative evaluations are important as well, of course. Please express your ideas for such evaluations here. These ideas can eventually be incorporated into [your research paper’s results](https://github.com/cgpotts/cs224u/blob/main/projects.md#Analysis).

### 🤖 Models
---
A description of the models that you'll be using as baselines, and a preliminary description of the model or models that will be the focus of your investigation.

At this early stage, some aspects of these models might not yet be worked out, so preliminary descriptions are fine. Your focus should be on making it clear how the models interact with the data’s task and metrics to provide a clear test of your hypothesis. If we can't put the pieces together in this way, you'll be requested to write a more precise description in your paper.

Identifying your baseline models is crucial. As will be discussed in the course, baseline systems provide essential context to evaluate against.

### 💬 General Reasoning
---
An explanation of how the data and models come together to inform your core research question(s) or or hypothes(i/e)s. You might feel that you've already given this explanation in the above sections. That's good! It suggests that the ideas are coming together. Please restate the general reasoning in this separate section. This will likely be part of your abstract in the research paper.

### ⏳ Progress summary
---
What you have done, what you still need to do, and any obstacles or concerns that might prevent your project from coming to fruition.

The more precise you are here, the more useful feedback we can provide. One of the most productive things we can do at this stage is help you define the scope of your project and then set priorities on that basis. For example, they might identify a core set of experiments that need to be run and suggest that you set the others aside until you have drafted your entire paper.

It's also very helpful to identify potential obstacles, so that we may help you overcome them or find a strategy for avoiding them entirely.

## 📜 Research Paper
---
The research paper implements the ideas put forward in the research proposal.

At this point we’d like to emphasize that we will never evaluate a project based on how "good" the results are. Publication venues do this, because they have additional constraints on space that lead them to favor positive evidence for new developments over negative results. In this course, we are not subject to this constraint, so we can do the right and good thing of valuing positive results, negative results, and everything in between. Please find details on what we **do** evaluate your work on in the rubric below.

### 🖌️ Formatting
---
The paper should be 4 pages long (typical for a short paper in NLP venues), in ACL submission format and adhering to ACL guidelines concerning references, layout, supplementary materials, and so forth (file type `.pdf`). See the [assessment page](/29d979eeca9f8162b84fe74bfbe424fc?pvs=25#29d979eeca9f81f8a008cfa97dab6cb1) for templates (Word and $\LaTeX$).

### 🎨 Organization
---
Papers in the field tend to use a common structure. You are not required to follow this structure, but doing so is likely to help you write a paper that is easily understood by practitioners, so it's strongly encouraged. These are as follows:

#### 🗜️ Abstract
Ideally a quarter-column in length. It's good to give some context for the work right at the start, define the current proposal and situate it in that context, summarize the core findings, and close by identifying the broader significance of the work. The "General reasoning" section of your proposal is likely to provide good material for the abstract.

#### 📣 Introduction
0.5–1 column. This is an extremely important part of the paper. In this section, the reader is likely to form their expectations for the work and begin to form their opinions of it. The introduction should basically tell the full story of the paper, in a way that is accessible to most people with a similar background:
1. Where are we? That is, what area of the field are you working in? Answering this question is important for orienting the reader.
2. What research question/hypothesis is being pursued? It's a good sign if you have a sentence that starts with a phrase like "The central [research question/hypothesis] of this paper is ...". You don't need to be this explicit, but, on the other hand, this is a way of ensuring that you don't end up saying only vague things about what your research question/hypothesis is. Also, being direct about this can expose a lack of clarity in your own thinking that you can then work through.
3. What concepts does the former depend on? You can't require your reader to fill in the gaps. Try to place all the building blocks of your research question/hypothesis in a way that the only logical next step in the reader’s thinking is formulating the research question/hypothesis themselves.
4. Why *this* research question/hypothesis? What broader issues does it address? This will provide further context for your ideas and help motivate your work.
5. What steps does the paper take to address your research question/hypothesis?
6. What are the central findings of the paper, and how do they inform the core question/hypothesis?

#### 🗄️ Related Work
0.75–1 column. This section can draw heavily on the lit review you did for the proposal. However, in addition to recent work, it should also highlight a portion of relevant seminal papers. A good strategy for this section is to first organize the papers you want to cover into general groups that relate to your own work in important ways. For each such group, articulate its thematic unity, briefly discuss what each paper achieves, and then, crucially, relate this work to your own project, as a way of providing context for your work and differentiating it from prior work. In this way, you carve out a place for your own contribution. The more relevant / recent the work, the more detail is expected. Space is limited, so don’t go too overboard.

#### 💾 Data
Length highly variable. This section should describe the properties of the dataset you were provided. This typically means giving some actual examples (or descriptions of them, if the examples are very long) as well as quantitative summaries (e.g., number of examples, number of examples per class, vocabulary size, etc.). With this discussion, you are trying to convey what your task is like, and build up context for your later analyses.

#### 🔬 (Method and) Experimental Setup
Length highly variable. Your research proposal should provide basic descriptions that you can expand and polish for this section.

We assume you are just comparing familiar models, and hence the method part can be quite brief. It is still crucial to briefly describe *why* you opted for your selection models. To the extent possible, it's good to separate the model descriptions from particular choices that you made for your experiments, as those are really part of the experimental setup.

Hence, the experimental setup portion should explain how the data and models work together for your experiments. The reader should get a clear picture of which models were evaluated, how they were trained, how the data were pre-processed and subdivided for the experiments, and which metrics were used.

#### 📈 Results
Length highly variable. A concise review of the experimental outcomes. No interpretation necessary at this point. Your view on the results is typically left for the discussion as to maintain focus in writing in the results section. This section should primarily guide the reader along any tables, figures, etc. that describe the outcomes. You may draw the reader’s attention to parts of the results as they are important to the central argument (e.g., model X outperforms the baseline).

#### 💭 Discussion and Conclusion
Length highly variable. This section is more open-ended. First and foremost, it should say what the experimental results mean. It is often fruitful to support this core conclusion with error analyses and qualitative trends in the predictions that provide deeper insights into where your favored model is succeeding and failing.

The conclusion portion is typically a very brief (quarter-column in length) summary of what the paper did and why. You may see this as a succinct encapsulation of the paper itself, similar to the abstract but with a more technical scope, since it can build on the paper itself.

### 🖊️ Authorship Statement
---
At the end of your paper (instead of the 'Acknowledgments' section in the template), please include a brief authorship statement, explaining how the individual authors contributed to the project. **This section, in addition to the References / Bibliography, does not count towards the page limit**. You are free to include whatever information you deem important to convey. For guidance, see the second page, right column, of [this guidance for PNAS authors](http://blog.pnas.org/iforc.pdf). We are requiring this largely because we think it is a good policy in general. Only in extreme cases, and after discussion with the team, would we consider giving separate grades to team members based on this statement.

### 💡 Advice on Scientific Writing
---
Here is a really nice paper on scientific writing:
[Bookmark: Novelist Cormac McCarthy’s tips on how to write a great science paper (https://www.nature.com/articles/d41586-019-02918-5)]

## 👾 Suggestions for Code Delivery
---
Please see the material of week 5’s lab session:
[Link to page: Lab Session 5 (../Lab_Session_5_Week_5_Self-study/index.html)]

## ❓ FAQ
---
To be filled when asked questions become frequent. 🙂