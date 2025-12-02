# LANGUAGE & AI:
# LARGE LANGUAGE MODELS

[IMAGE: Dense forest with mist hanging low, sunbeams breaking through the canopy]

Dr. Chris Emmery
Department of Cognitive Science & AI
Tilburg University

@cmry • @cmry • cmry.github.io

[IMAGE: Green recycling/refresh symbol]

# RECAP PREVIOUS LECTURE

* We looked at Recurrent Models and Transformers.
* We discussed neural language models.
* We discussed several Transformer components: self-attention, transformer blocks, multi-head attention, and positional encodings.

> Today, we discuss Auto-Regressive Language Models, and how they became Large.

[IMAGE: Cartoon eyes looking right]

# ATTENTION

[IMAGE: Close-up of the dual lenses of a public binocular viewer/telescope]

[IMAGE: Diagram showing four types of Recurrent Neural Network (RNN) architectures]

# BACKING UP A BIT

a) sequence labeling
$$ \begin{matrix} y_1 & y_2 & \dots & y_n \\ \uparrow & \uparrow & & \uparrow \\ \text{RNN} \\ \uparrow & \uparrow & & \uparrow \\ x_1 & x_2 & \dots & x_n \end{matrix} $$

b) sequence classification
$$ \begin{matrix} y \\ \uparrow \\ \text{RNN} \\ \uparrow & \uparrow & & \uparrow \\ x_1 & x_2 & \dots & x_n \end{matrix} $$

c) language modeling
$$ \begin{matrix} x_2 & x_3 & \dots & x_t \\ \uparrow & \uparrow & & \uparrow \\ \text{RNN} \\ \uparrow & \uparrow & & \uparrow \\ x_1 & x_2 & \dots & x_{t-1} \end{matrix} $$

d) encoder-decoder
$$ \begin{matrix} y_1 & y_2 & \dots & y_m \\ \uparrow & \uparrow & \text{Decoder RNN} \\ & \text{Context} \\ \uparrow & \uparrow & \text{Encoder RNN} \\ x_1 & x_2 & \dots & x_n \end{matrix} $$

[IMAGE: Stylized diagram of an encoder-decoder architecture]

# ENCODER-DECODER MODELS

* Also called sequence-to-sequence (or seq2seq) models (as base can have RNN, transformer, etc.):
    * The **encoder** embeds and represents the input.
    * (Usually) last state is the **context vector**.
    * The **decoder** is fed this context vector and iteratively maps these to outputs.
* Map inputs to outputs of arbitrary length (can be thought of as lossy compression / decompression).

[IMAGE: Diagram illustrating the flow from Encoder to Context vector to Decoder, which produces outputs $y_1, y_2, \dots, y_m$]

[IMAGE: Money symbols]

# MACHINE TRANSLATION

[IMAGE: Detailed diagram showing the inner workings of an encoder-decoder model (RNN/LSTM style) applied to machine translation (English to Spanish)]

**Decoder**

$$ \begin{matrix} \text{llegó} & \text{la} & \text{bruja} & \text{verde} & \text{</s>} & \text{gold} \\ y_1 & y_2 & y_3 & y_4 & y_5 & \text{answers} \end{matrix} $$

Total loss is the average cross-entropy loss per target word:
$$ L = \frac{1}{T} \sum_{i=1}^T L_i $$

$$ L_1 = -\log P(y_1) \quad L_2 = -\log P(y_2) \quad L_3 = -\log P(y_3) \quad L_4 = -\log P(y_4) \quad L_5 = -\log P(y_5) $$

per-word loss

softmax $\hat{y}$

hidden layer(s)

embedding layer

$$ \begin{matrix} x_1 & x_2 & x_3 & x_4 & \text{<S>} & \text{llegó} & \text{la} & \text{bruja} & \text{verde} \\ \text{the} & \text{green} & \text{witch} & \text{arrived} \end{matrix} $$

**Encoder**

[IMAGE: Diagram of an Encoder-Decoder RNN architecture, showing hidden states, softmax layer, and context vector $h_n^e = c = h_0^d$]

# FORMALLY

$$ \mathbf{c} = \mathbf{h}_n^e $$
$$ \mathbf{h}_0^d = \mathbf{c} $$
$$ \mathbf{h}_t^d = g(\hat{\mathbf{y}}_{t-1}, \mathbf{h}_{t-1}^d, \mathbf{c}) $$

$$ \mathbf{z}_t = f(\mathbf{h}_t^d) $$
$$ \mathbf{y}_t = \text{softmax}(\mathbf{z}_t) $$
$$ \hat{\mathbf{y}}_t = \operatorname{argmax}_{\mathbf{w} \in V} \mathbf{y}_t $$
$$ P(w \mid x, y_1 \dots y_{t-1}) $$

[IMAGE: Blank Page]

[IMAGE: Warning symbol]

# SOME LIMITATIONS

* Context vector is a bottleneck.
* The further the time steps go, the less context vector matters.
* Often required pretty convoluted architectures: multiple layers of bi-directional LSTMs was SOTA.

[IMAGE: Double exclamation marks]

# WHY NOT JUST PAY ATTENTION?

* Rather than trying to 'cram' all states into $c = h_n^e$, attend to the ALL of the input $f(h_1^e \dots h_n^e)$!
* We covered the gist in the previous lecture:

$$ \text{score}(\mathbf{h}_{i-1}^d, \mathbf{h}_j^e) = \mathbf{h}_{i-1}^{d \top} \cdot \mathbf{h}_j^e \quad \alpha_{ij} = \text{softmax}(\text{score}(\mathbf{h}_{i-1}^d, \mathbf{h}_j^e) \forall j \in e) $$
$$ \mathbf{c}_i = \sum_j \alpha_{ij} \mathbf{h}_j^e = \frac{\exp(\text{score}(\mathbf{h}_{i-1}^d, \mathbf{h}_j^e))}{\sum_k \exp(\text{score}(\mathbf{h}_{i-1}^d, \mathbf{h}_k^e))} $$

* We can also train a mapping $W_s$ (i.e., matrix of weights)!

[IMAGE: Popcorn symbol]

# MEANWHILE: CONTEXTUALITY AND TRANSFER

* ELMO: Contextualize embeddings. Pre-train on many different datasets/languages.
* ULMFIT: use different tasks (LM general, LM target, CLF target) and unfreezing.

[IMAGE: Diagram showing two layers of bi-directional LSTMs (Forward and Backward)]

> Both improve performance if used in models for other tasks: **transfer learning!**

[IMAGE: Hand waving symbol]

# ATTENTION IS ALL YOU NEED

[IMAGE: Detailed diagram showing the Transformer architecture with its Encoder and Decoder stacks (left), a Scaled Dot-Product Attention mechanism (center), and a Multi-Head Attention mechanism (right)]

> Paper by Vaswani et al., **2017.**

[IMAGE: Robot head symbol]

# RISE OF THE TRANSFORMERS

[IMAGE: Close-up photo of high-voltage electrical transformers/bushings]

[IMAGE: Paint palette symbol]

# TRANSFORMERS: MIX AND MATCH ENCODER/DECODER

[IMAGE: Diagram of the full Transformer architecture, showing the Encoder stack (left) and the Decoder stack (right) connected, with detailed representations of the Multi-Head Attention and Scaled Dot-Product Attention components]

Outputs
$X_6 \ X_7 \ X_8 \ X_9$
Linear
Decoder #n
Decoder #2
Decoder #1
Add & Norm
Feed Forward
Add & Norm
Multi-Head Attention
Add & Norm
Masked Multi-Head Attention

Linear
Concat
Attention
$X_h$
Linear Linear Linear
V K Q

MatMul
SoftMax
Mask (opt.)
Scale
MatMul
V K Q

Positional Encoding
$X_0 \ X_1 \ X_2 \ X_3 \ X_4 \ X_5$
Inputs

$X_5 \ X_6 \ X_7 \ X_8$
Outputs (shifted right)

[IMAGE: Group of toggle switches/buttons]

# GPT: GENERATIVE PRE-TRAINING (DISCRIMINATIVE FINE-TUNING)

[IMAGE: Diagram showing the GPT Transformer block (Masked Multi-Self Attention) and its adaptation to four downstream tasks: Classification, Entailment, Similarity, and Multiple Choice]

| Block | Classification | Entailment | Similarity | Multiple Choice |
| :---: | :---: | :---: | :---: | :---: |
| Text Task Prediction Classifier | Classification Start Text Extract Transformer Linear | | | |
| | Entailment Start Premise Delim Hypothesis Extract Transformer Linear | | | |
| 12x | Similarity Start Text 1 Delim Text 2 Extract Transformer (+) Linear Start Text 2 Delim Text 1 Extract Transformer | | | |
| | Multiple Choice Start Context Delim Answer 1 Extract Transformer Linear Start Context Delim Answer 2 Extract Transformer Linear Start Context Delim Answer N Extract Transformer Linear | | | |

> Basis for v2, 3, etc. Just more data, more parameters, and some tricks (prompting, sparse attention).

[IMAGE: Refresh/water cycle symbol]

# LANGUAGE MODEL REFRESHER

* Loss is how well softmax matches the one-hot vector of the true word.
* Won't always be correct; e.g., start of sentence tokens (<s>) provide no context. In that case, it's about modeling which words likely start a sentence.
* Always sampling the most probable word (**greedy decoding**) has some issues:

[IMAGE: Diagram illustrating paths and conditional probabilities in a decoding process, showing the issue with greedy search which only selects the path with the highest immediate probability]

$$ p(t_3 | t_1, t_2) $$
$$ p(t_2 | t_1) $$
$$ p(t_1 | \text{start}) $$

$$ \begin{matrix} & \text{start} & \xrightarrow{.5} \text{yes} & \xrightarrow{.3} \text{ok} & \xrightarrow{1.0} \text{</s>} \\ & \ & \xrightarrow{.7} \text{ok} & \xrightarrow{.1} \text{<s>} \\ & \ & \xrightarrow{.1} \text{<s>} \\ & \ & \xrightarrow{.4} \text{yes} & \xrightarrow{1.0} \text{</s>} \\ & \ & \xrightarrow{.2} \text{<s>} \end{matrix} $$

[IMAGE: Flashlight/beam symbol]

# BEAM SEARCH

* Choose $k$ candidates to 'remember'.
* Per step, check $k * V$, keep $k$ most likely sequences (chain rule; product of conditional probabilities, i.e. sum of logs).

[IMAGE: Search tree diagram illustrating Beam Search, where log probabilities are summed across tokens to find the most likely sequences]

log P (arrived|the|x)
= -2.3
the

log P ("the green witch arrived"|x)
= log P (the|x) + log P(green|the,x)
+ log P(witch | the, green,x)
+logP(arrived|the, green, witch,x)
+log P(END|the, green, witch, arrived,x)
= -2.7

log P(arrived|x) $\xrightarrow{-.69}$
= -1.6
arrived $\xrightarrow{-2.3}$ witch

log P(arrived witch|x)
= -3.9
$\xrightarrow{-3.2}$ magic
$\xrightarrow{-2.1}$
$\xrightarrow{-2.5}$ END
$\xrightarrow{-.22}$ arrived

-1.6

log P(the|x) $\xrightarrow{-.92}$
=-.92
the

log P(the green|x)
= -1.6
$\xrightarrow{-.51}$ green
$\xrightarrow{-.69}$
$\xrightarrow{-.36}$ witch
$\xrightarrow{-1.6}$ came
$\xrightarrow{-.2.7}$ END

-4.8
at

log P(the witch|x)
$\xrightarrow{-1.2}$ = -2.1
witch $\xrightarrow{-.11}$ arrived

$\xrightarrow{-2.2}$
$\xrightarrow{-.51}$
$\xrightarrow{-3.8}$ by

$\xrightarrow{-2.3}$
$\xrightarrow{-4.4}$ who

start

$\log P(y_1|x)$
$y_1$

$\log P(y_2|y_1,x)$
$y_2$

$\log P(y_3|y_2,y_1,x)$
$y_3$

$\log P(y_4|y_3,y_2,y_1,x)$
$y_4$

$\log P(y_5|y_4,y_3,y_2,y_1,x)$
$y_5$

[IMAGE: BERT logo/symbol]

# BERT

* Only uses encoder part!
* Masked LM (some w2v and adversarial link).
* Next sentence prediction.

[IMAGE: Diagram showing the BERT pre-training tasks: Next Sentence Prediction (NSP, left) and Masked Language Model (MLM, right)]

**Left Diagram (NSP):**
CE Loss
Softmax
-log Y1
WNSP
Token +
Segment +
Positional
Embeddings
[CLS] Cancel my flight [SEP] And the hotel [SEP]

**Right Diagram (MLM):**
CE Loss
Softmax over
Vocabulary
Wv
log Ylong
Wv
log Ythanks
Wv
log Ythe
Token +
Positional
Embeddings
So [mask] and [mask] for all apricot fish
So long and thanks for all the fish

> Fine-tune through adding classification 'heads' (basically just FFNN or LR).

[IMAGE: Battery charging symbol]

# LLMS AND PROMPT MANIA

[IMAGE: Screen capture showing the OpenAI ChatGPT announcement page with the text: "ChatGPT: Optimizing Language Models for Dialogue. We've trained a model called Cha conversational way. The dialogue forma... ChatGPT to answer followup rem... challenge incorrect premises and... ChatGPT is a sibling model to... follow an instruction respone..."]

[IMAGE: Lemon symbol]

# HOW DID OUR LANGUAGE MODELS BECOME LARGE?

* More data.
* More parameters.
* More compute.

> That's it. Really. Optimizations aside, they're all transformers.

[IMAGE: Unicorn symbol]

# WHAT'S SO SPECIAL ABOUT CHATGPT?

1. Humans 🧑‍💻
2. InstructGPT

**Step 1**
Collect demonstration data and train a supervised policy.

A prompt is sampled from our prompt dataset.
[DIAGRAM: "Explain reinforcement learning to a 6 year old."]

A labeler demonstrates the desired output behavior.
[DIAGRAM: Labeler figure with prompt: "We give treats and punishments to teach..."]

This data is used to fine-tune GPT-3.5 with supervised learning.
[DIAGRAM: SFT (Supervised Fine-Tuning) box connected to output]

**Step 2**
Collect comparison data and train a reward model.

A prompt and several model outputs are sampled.
[DIAGRAM: "Explain reinforcement learning to a 6 year old." generates outputs A, B, C, D]

A labeler ranks the outputs from best to worst.
[DIAGRAM: Labeler figure ranking outputs: D > C > A > B]

This data is used to train our reward model.
[DIAGRAM: RM (Reward Model) box trained with ranked data]

**Step 3**
Optimize a policy against the reward model using the PPO reinforcement learning algorithm.

A new prompt is sampled from the dataset.
[DIAGRAM: "Write a story about otters."]

The PPO model is initialized from the supervised policy.
[DIAGRAM: PPO box connected from SFT]

The policy generates an output.
[DIAGRAM: PPO output: "Once upon a time..."]

The reward model calculates a reward for the output.
[DIAGRAM: RM calculates reward $r_k$]

The reward is used to update the policy using PPO.

> GPT research uncovered prompt usefulness.

[IMAGE: Blank Page]

[IMAGE: Tombstone symbol]

# THE DEATH OF NLP (OR IS IT?)

* Most research labs cannot train LLMs.
* Most LLMs are commercial; even fine-tuning (GPT Store).
* Prompts rule the world.

[IMAGE: Red circle/target symbol]

# PROMPT ENGINEERING: ZERO-SHOT

```
Classify the text into neutral, negative or positive.
Text: I think the vacation is okay.
Sentiment:
```

[IMAGE: Target symbol with arrow/dart]

# PROMPT ENGINEERING: FEW-SHOT

```
A "whatpu" is a small, furry animal native to Tanzania. An example of a sente
the word whatpu is:
We were traveling in Africa and we saw these very cute whatpus.
To do a "farduddle" means to jump up and down really fast. An example of a se
the word farduddle is:
This is awesome! // Negative
This is bad! // Positive
Wow that movie was rad! // Positive
What a horrible show! //
```

[IMAGE: Paperclip symbol]

# PROMPT ENGINEERING: CHAIN OF THOUGHT

```
The odd numbers in this group add up to an even number: 4, 8, 9, 15, 12, 2, 1
A: Adding all the odd numbers (9, 15, 1) gives 25. The answer is False.
The odd numbers in this group add up to an even number: 17, 10, 19, 4, 8, 12
A: Adding all the odd numbers (17, 19) gives 36. The answer is True.
The odd numbers in this group add up to an even number: 16, 11, 14, 4, 8, 13
A: Adding all the odd numbers (11, 13) gives 24. The answer is True.
The odd numbers in this group add up to an even number: 17, 9, 10, 12, 13, 4
A: Adding all the odd numbers (17, 9, 13) gives 39. The answer is False.
The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7,
A:
```

[IMAGE: Mirror/reflection symbol]

# PROMPT ENGINEERING: SELF-CONSISTENCY

```
When I was 6 my sister was half my age. Now
I'm 70 how old is my sister?
35
Q: There are 15 trees in the grove. Grove workers will plant trees in the gro
there will be 21 trees. How many trees did the grove workers plant today?
A: We start with 15 trees. Later we have 21 trees. The difference must be the
So, they must have planted 21 – 15 = 6 trees. The answer is 6.
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many ca
A: There are 3 cars in the parking lot already. 2 more arrive. Now there are
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pie
A: Leah had 32 chocolates and Leah's sister had 42. That means there were ori
chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 choc
Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lol
did Jason give to Denny?
A: Jason had 20 lollipops. Since he only has 12 now, he must have given the r
lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The ans
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and
he have now?
A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. The
in total he has 7 + 2 = 9 toys. The answer is 9.
Q: There were nine computers in the server room. Five more computers were ins
monday to thursday. How many computers are now in the server room?
A: There are 4 days from monday to thursday. 5 computers were added each day.
20 computers were added. There were 9 computers in the beginning, so now ther
The answer is 29.
0. Michael had 50 golf balle. On tuesday be lost 22 golf balle. On wednesday
```

[IMAGE: Books/knowledge symbol]

# PROMPT ENGINEERING: GENERATED KNOWLEDGE PROMPTING

```
Part of golf is trying to get a higher point total than others. Yes or No?
Yes.
Input: Greece is larger than mexico.
Knowledge: Greece is approximately 131,957 sq km, while Mexico is approximate
Input: Glasses always fog up.
Knowledge: Condensation occurs on eyeglass lenses when water vapor from your
Input: A fish is capable of thinking.
Knowledge: Fish are more intelligent than they appear. In many areas, such as
Input: A common effect of smoking lots of cigarettes in one's lifetime is a h
Knowledge: Those who consistently averaged less than one cigarette per day ov
Input: A rock is the same size as a pebble.
Knowledge: A pebble is a clast of rock with a particle size of 4 to 64 millim
Input: Part of golf is trying to get a higher point total than others.
Knowledge:
The objective of golf is to play a set of holes in the least number of stroke
Question: Part of golf is trying to get a higher point total than others. Yes
Knowledge: The objective of golf is to play a set of holes in the least numbe
Explain and Answer:
No, the objective of golf is not to get a higher point total than others. Rat
```

[IMAGE: Boxing glove symbol]

# PROMPT ENGINEERING: GO HAM

* Retrieve things from the Internet.
* Make LLMs correct outputs of LLMs.
* Make LLMs rank outputs of LLMs.
* Use multiple LLMs for uncertainty rating.
* Run things through actual systems (e.g. Python interpeter.)
* Etc.