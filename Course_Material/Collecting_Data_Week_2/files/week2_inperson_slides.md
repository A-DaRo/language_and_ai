# Language & AI
## Lecture 2

Dr. ir. Marijn ten Thij, Nov 20st 2025

[IMAGE: Manicured garden or park setting]

## Video Lecture
Collecting Data (etc.)

* What is / can be data?
* What is / can be noise?
* Denoising data
    * RegEx (find and replace)
    * Eval (models + corrections).
    * Normalization (spelling correction, case folding, stemming, lemmatization).
    * Encodings.

[IMAGE: Abstract black and white graphic resembling light beams or a futuristic structure, with the text '100%']

## This Lecture
Thinking About Data

* Interim assignment questions.
* Data collection considerations.
* Lot about data annotation!
* Contextualizing preprocessing.

# Research Components
Empirical, Quantitative, Qualitative

* Empirical questions: answers given through experiments.
* Experiments/analyses: quantitative vs. qualitative. In ML/NLP:
    * Quantitative always involves a metric:
        * We have models.
        * These are tested under certain experimental conditions (different splits, preprocessing, hyper-parameters, etc.).
        * We compare effects using evaluation metric differences.

# Research Components (Cont.)
Empirical, Quantitative, Qualitative

* So what makes something a qualitative analysis in ML/NLP?
    * Usually involves actually looking at the data.
    * Can also be plotting certain things and comparing them yourself.
    * Generally this is to personally infer relations / make observations that aren't rooted in numbers; you don't have 'proof'.
    * Scribbr has some decent high level descriptions for more context.

[IMAGE: Black and white photo of two silhouetted people walking across a steep ridge/hill]

# Text Data
What They Don't Tell You

[IMAGE: Stacked Western Digital 4TB hard drives, showing serial numbers and branding, including the text: Western Digital. 4TB, WD40PURZ, SATA 6Gb/s, Designed for 24 x 7 systems]

## Pre-Made
Is Unrealistic

* API's and platforms (HuggingFace Data, Kaggle, etc.) are convenient.
* Working with .csv's (tabular) and otherwise structured data is a breeze.
* This is (often) not what reality looks like. Even structured data can be a mess.

# Real Text Data
Where to Get It and Why Doesn't Everyone?

* Scraping (e.g. via Python).
* In essence very simple.
* But:
    * Detection Systems.
    * Scale (processing time, storage, relevancy).
    * Maintenance.

```
">
<div class="...">
<a class="..." style="color: rgb(129, 131, 132);" href="/user/ManGood2002/">
u/ManGood2002
</a>
<div id="UserInfoTooltip--t3_apmsqk--lightbox"></div>
</div>
<a class="..." data-click-id="timestamp"
href="/r/Showerthoughts/comments/apmsqk/the_syllables_in_on_your_mark_get_set_go_are_a/
">
3 years ago
</a>
</div>
<!-- ..
->
<div class="..." >
<span class="..." id="">
<span id="PostAwardBadges--t3_apmsqk--lightbox-gid_3">
<img alt="Platinum" class="..." id="..."
src="https://www.redditstatic.com/gold/awards/icon/platinum 32.png">
</span>
<span></span>
</span>
</div>
<div class="...">
<div class="...">
<div class="..."
style="--posttitletextcolor: #D7DADC;">
<h1 class=""..." >
The syllables in “on your mark, get set, go" are a countdown
<!--
-->
</h1>
</div>
</div>
```

# Processing API Data

Errors?

commented on How do you see the future of China?r/AskALiberal. Posted by
u/Winston_Duarte
anfortunately I'm not when it comes to this. The CCP has
tightly consolidated its power, and I don't see them losing that grip. Though who knows,
maybe China invading Taiwan can be a blessing in disguise. If they decide to actually do
it, I hope it goes disastrously for them, and maybe the grip of the CCP could loosen as
they show signs of weakness. But I'm not holding my breath.
Reply Share

6 hr. ago
In Finnish 'junk mail' is 'roskaposti' (lit. 'trash mail')
'Spam' is 'spämmi' (self explanatory, informal)
Vote
Reply Share
France
6 hr. ago
in
6 hr. ago
i don't think i'v encountered other phrase than "spam" in polish, perhaps "reklamy" (ads), but
that's more specific term
Vote
Reply Share
5 hr. ago
Poland
commented on Which language would you like to see on Duolingo? Discussion
r/duolingo. Posted by u/Electrical-Force-805
1 point 57 minutes ago
Albanian, Serbo-Croatian, Afrikaans, Estonian, Luxembourgish, Slovene, Northern Sámi,
Frisian, Persian, Bulgarian, Maori
Reply Share
Luxia33 4 points. 2 hours ago
Maori! It seems like theyve abandonded it :(
1 point 1 hour ago
That's because they got rid of the Contributor program. As far as I know, their
process to create courses is entirely internal nowadays.
Reply Share
-in
5 hr. ago
Spam or junkmail most commonly used but ongevraagde/ongewenste e-mail is the Dutch term,
it literally means unasked for or unwanted e-mails
Vote
Reply Share
6 hr. ago
Vestland, Norway
The "official" word is søppelpost (trash/garbage mail), IIRC, but I don't think I've ever heard
somebody say that. It's usually just called spam.
commented on Turkey borders 7 countries with 7 different languages i.redd.it/5knhgc...
r/MapPorn. Posted by u/l1qmaballs
63 points 1 hour ago
Congratulations to get the title wrong: it's 7 different alphabets.
16 points 1 hour ago
I'm sorry to tell you, but you got it wrong as well. Turkey is bordered by only 5
alphabets. The Arabic/Persian writing system is primarily an abjad, not an alphabet.
Reply Share
YellowOnline 9 points 1 hour ago
I'm saying what's on the map that OP posted. I am not responsible for the qu see more
4 points 1 hour ago
Yes yes, I'm just playing around
Reply Share

# Data Operations
After Retrieval

[IMAGE: Abstract turquoise/green graphic featuring stylized circuit board patterns and digital dots, framed by a border.]

* Query time (training, stats).
    * What to index on?
    * What to split on?
* Redundancy (i.e., samples).
* Sharing.
* Database, dataframe, data file, output file, log file, ...

[IMAGE: Large, round, opened bank vault door]

# Text as Value
Why the Internet is Closing

* Rising issues:
    * X has restricted research access.
    * Reddit API being restricted.
    * Data relevancy?
* Text data is monetized and access controlled.
* 'Fair use' often an issue.
* Don't forget: this is your data!

[IMAGE: Stylized teal and white robot sitting in a chair in front of a small laptop, surrounded by stacks of coins]

# Automated Content
NLP Undoing NLP

* Bots. All the bots.
* What is real and what isn't?
    * Reviews.
    * Comments.
    * Articles.
    * Most interesting: Al trained on 'the Internet', generating content. And after?

[IMAGE: Large wooden question mark illuminated by light bulbs, lying on its side]

# Any Questions?
(So far)

# Considerations When Collecting Data
Things Other Than Computational Limitations

* Text in a structured format? Expect issues? Sensitive? Can it be shared?
* How much meta-data is required (in the future) and how can it be stored?
* Collect labels directly, using a heuristic (distant labeling), or annotators?
* Latter case: how to recruit annotators? Quality control?
    * MechanicalTurk, CrowdFlower, etc.
    * Experts. Your friends? WEIRD group.
    * Pay (how much?). What's the incentive? Inter-rater agreement?
* Sample bias (on everything). How to mitigate, and is that desirable?

[IMAGE: Book cover for 'THE WEIRDEST PEOPLE IN THE WORLD', showing categories like Western, Industrialized, Educated, Rich, Democratic, and 'HOW THE WEST BECAME']

# Cohen's Kappa
Inter-rater Reliability Scoring

(No exercises so
equation not
on the exam 😄)

$$ \kappa = \frac{p_o - p_e}{1 - p_e} = 1 - \frac{1 - p_o}{1 - p_e} $$

where $p_o$ is the relative observed agreement among raters, and $p_e$ is:

$$ p_e = \frac{1}{N^2} \sum_k n_{k1} n_{k2} $$

where $n_{k1}$ is the number of items (total $N$) classified as $k$ by rater 1.

📄 -> 🧑‍💻 | 👍 | 👎 | 🧑‍💻 | 👍

# Annotations
And What to Pay Attention To

* How diverse is your team?
    * Not just the annotators!
* How are you instructing the annotators?
* If all fails: can we hide the information somehow?
* Might be used for post-hoc error analysis!

[IMAGE: Snippets of three academic papers related to annotator bias and data quality]

**Paper 1 (Top Right):**
Don't Blame the Annotator:
Bias Already Starts in the Annotation Instructions
Mihir Parmar¹ Swaroop Mishra¹* Mor Geva²⁺ Chitta Baral¹
Arizona State University ²Allen Institute for AI
{mparmar3, srmishr1, chitta)@asu.edu, pipek@google.com
(Dasigi et al., 2019; Zhou
i et al., 2020).
ess of this method, past stud-
a collected through crowd-
various biases that lead to
el performance (Schwartz
et al., 2018; Poliak et al.,
e Bras et al., 2020; Mishra
d Arunkumar, 2021; Het-

**Paper 2 (Middle Left):**
cs.CL] 28 Aug 2019
Are We Modeling the Task or the Annotator? An Investigation of
Annotator Bias in Natural Language Understanding Datasets
Mor Geva
Tel Aviv University,
Allen Institute for AI
morgeva@mail.tau.ac.
Yoav Goldberg
Bar-Ilan University,
Allen Institute for AI
-Amail.com

**Paper 3 (Bottom Right):**
Annotators with Attitudes:
How Annotator Beliefs And Identities Bias Toxic Language Detection
Maarten Sap Swabha Swayamdipta Laura Vianna
Xuhui Zhou Yejin Choi Noah A. Smith
Paul G. Allen School of Computer Science, University of Washington, Seattle, WA, USA
Allen Institute for AI, Seattle, WA, USA
Department of Psychology, University of Washington, Seattle, WA, USA
Georgia Institute of Technology, Atlanta, GA, USA
{maartens, swabhas}@allenai.org
Abstract
Warning: this paper discusses and contains
content that is offensive or upsetting.
The perceived toxicity of language can vary
based on someone's identity and beliefs, but
that variation is often ignored when collecting
toxic language datasets, resulting in datar
and Tsvetkov, 2020) or backfiring against minori-
ties (Yasin, 2018; Are, 2020, i.a.). For exam-
ple, racial biases have been uncovered in toxic
language detection where text written in African
American English (AAE) is falsely flagged as
toxic (Sap et al., 2019; Davidson et al., 2019).
The crux of the is

[IMAGE: Large wooden question mark illuminated by light bulbs, lying on its side]

# Any Questions?
(So far)

[IMAGE: Silhouette of a directional signpost against a sunset sky]

# What to Input?
Keep? Replace? Remove?

# What Are Tokens?
And Why Does Marijn Care So Much?

* The alcoholic content of this beer is high. It's good.
* The A.B.V. of this IPA is 7.8%. Quite strong.
* This Triple from La Trappe is fairly strong. I will have something else.
* La Trappe's Triple IPA. (if only)

[IMAGE: Chalkboard with handwritten text examples and NLP labels]

# Stylometry Example

Until his sixteenth work Aus Italien (From Italy) (1886), his first tone
poem, he did not depart from the classic forms, although there
were a few signs of change in style in a violin sonata which he
wrote just before the tone poem. In fact, he was so much against
Wagner and his innovations, that no one could have guessed that
later he himself would be considered an innovator and would be
accused of imitating Wagner.
During his youth, after hearing Siegfried he wrote to a friend about
the music of Mime: "It would have killed 411a cat and the horror of
musical dissonances would melt rocks into omelettes."

```/\[A-Z]\w+/g```
```/\[^\w]+/g```

# Stylometry Example II

Until his sixteenth work Aus Italien (From Italy) (1886), his first tone
poem, he did not depart from the classic forms, although there
were a few signs of change in style in a violin sonata which he
wrote just before the tone poem. In fact, he was so much against
Wagner and his innovations, that no one could have guessed that
later he himself would be considered an innovator and would be
accused of imitating Wagner.
During his youth, after hearing Siegfried he wrote to a friend about
the music of Mime: "It would have killed 411a cat and the horror of
musical dissonances would melt rocks into omelettes.”

LI||| ||| ||||||||| |||I LII LIIIIII (LIII LIIII)
(nnnn), III IIIII |||| ||||, || ||| ||| |||||| |||||||
|||| |||, |||||||| ||||||||||||||||||||||
||||| || | |||||| |||||||||||||||||||||||||||||||||||
||||. LI ||||, || ||| ||||||||||||||||||||||
||||||||||, |||||| ||| ||| |||||||||||
||||||| ||||| || |||||||| || |||||||| || ||||||||
|| ||||||||| |||||.
LI|||| ||| |||||, ||||| ||||||| ||||||||| |
||||||||||| ||| ||||| || LIII: “LI ||||| |||| ||||||
nnnl III III III
|||||||||.”

# Stylometry Example III

[IMAGE: Matrix plot visualization (a)] (a) Sharing Her Crime, M. A. Fleming
[IMAGE: Matrix plot visualization (b)] (b) The Actress' Daughter, M. А. Fleming
[IMAGE: Matrix plot visualization (c)] (c) King Lear, W. Shakespeare
[IMAGE: Matrix plot visualization (d)] (d) Hamlet, W. Shakespeare

! * ( ) , . : ; ? ...

[IMAGE: Matrix plot visualization (e)] (e) The History of Mr. Polly, H. G. Wells
[IMAGE: Matrix plot visualization (f)] (f) The Wheels of Chance, H. G. Wells

Plots under MIT license (via GitHub).

Euro. Jnl of Applied Mathematics (2021), vol. 32, pp. 1069-1105 © The Author(s), 2020. Published by 1069
Cambridge University Press.
doi:10.1017/S0956792520000157

Pull out all the stops: Textual analysis via
punctuation sequences
ALEXANDRA N. M. DARMON¹, MARYA BAZZI1,2,3, SAM D. HOWISON¹
and MASON A. PORTER 1,4
¹Oxford Centre for Industrial and Applied Mathematics, Mathematical Institute, University of Oxford, Oxford OX2
6GG, UK
²The Alan Turing Institute, London NW1 2DB, UK
³Warwick Mathematics Institute, University of Warwick, Coventry CV4 7AL, UK
⁴Department of Mathematics, University of California, Los Angeles, Los Angeles, California 90095, USA
emails: alexandra.darmon@hotmail.fr, mbazzi@turing.ac.uk, howison@maths.ox.ac.uk, mason@math.ucla.edu

(Received 31 December 2018; revised 16 January 2020; accepted 12 May 2020;
first published online 21 September 2020)

# Is Preprocessing Essential?
A Debate for the Ages

Language Resources and Evaluation (2023) 57:257-291
https://doi.org/10.1007/s10579-022-09620-5
ORIGINAL PAPER
The impact of preprocessing on word embedding quality:
a comparative study
Zahra Rahimi¹. Mohammad Mehdi Homayounpour¹
Accepted: 16 September 2022/Published online: 18 October 2022
Check for
updates
© The Author(s), under exclusive licence to Springer Nature B.V. 2022
Check for updates
Article
Improving the performance of sentiment analysis ext Preprocessing for Text
using enhanced preprocessing technique and ning in Organizational
Artificial Neural Network
Ankit Thakkar, Senior Member, IEEE, Dhara Mungra, Anjali Agrawal, and Kinjal Chaudhari
Abstract-With the presence of a maseiun
manual
In
N
search: Review
d Recommendations
uti Thapa¹, Louis Tay²,
Padmini Srinivasan³
Organizational Research Methods
2022, Vol. 25(1) 114-146
The Author(s) 2020
Article reuse guidelines:
sagepub.com/journals-permissions
DOI: 10.1177/1094428120971683
journals.sagepub.com/home/orm
SSAGE
PLOS ONE
Check for
updates
RESEARCH ARTICLE
The influence of preprocessing on text
classification using a bag-of-words
representation
Yaakov HaCohen-Kerner, Daniel Miller, Yair Yigal
Dept. of Computer Science, Jerusalem College of Technology - Lev Academic Center, Jerusalem, Israel
*kerner@jct.ac.il
Abstract
Text classification (TC) is the task of automatically assigning documents to a fixed
number of categories. TC is an important component in many text applications. Many
of these applications perform preprocessing. There are different types of text prepro-
cessing, e.g., conversion of uppercase letters into lowercase letters, HTML tag removal,
stopword removal, punctuation mark removal, lemmatization, correction of common
misspelled words, and reduction of replicated characters. We hypothesize that the
annlication of different combinations of preprocessing methods can improve TC results.
in and systematic set of TC experiments (and this is
of five/
uti
have provided new methods for capitalizing on the voluminous
d by organizations, their employees, and their customers. Although
during text preprocessing affect whether the content and/or style
tistical power of subsequent analyses, and the validity of insights
methodological articles have described the general process of
but recommendations for preprocessing text data were incon-
es use and report different preprocessing techniques. To address
ary reviews of computational linguistics and organizational text
ally grounded text preprocessing decision-making recommen-
f text mining conducted (i.e., open or closed vocabulary), the
1, and the data set's characteristics (i.e., corpus size and average
ns from these recommendations will be appropriate and, at
characteristics of one's text data. We also provide recom-
to promote transparency and reproducibility.

> The paper that is cited to back up the
> assumed improvement shows that it
> indeed increases performance for
> various NN setups, but decreases
> performance for ROBERTa. In case this
> requires clarification: modern LLMs use
> subword tokenizers that make ‘classic’
> preprocessing steps unnecessary at
> best, but generally damage the
> performance (due to incorrect pre-
> tokenization).

[IMAGE: Brick wall with two separate stone plaques reading 'YES' and 'NO']

# The Task
Decides

* Ask yourself:
    * Do I think my preprocessing steps will add anything or remove information?
* Consider using both types of representations!
* Assess quantitatively, and diligently!

[IMAGE: Large wooden question mark illuminated by light bulbs, lying on its side]

# Any Questions?