# LEACE: Perfect linear concept erasure in closed form

**Nora Belrose**$^1$ **David Schneider-Joseph**$^1$ **Shauli Ravfogel**$^2$ **Ryan Cotterell**$^3$
**Edward Raff**$^4$ **Stella Biderman**$^{1,4}$
$^1$EleutherAI $^2$Bar-Ilan University $^3$ETH Zürich $^4$Booz Allen Hamilton
{nora,stella}@eleuther.ai david@davidsj.com

### Abstract

Concept erasure aims to remove specified features from an embedding. It can improve fairness (e.g. preventing a classifier from using gender or race) and interpretability (e.g. removing a concept to observe changes in model behavior). We introduce LEAst-squares Concept Erasure (LEACE), a closed-form method which provably prevents all linear classifiers from detecting a concept while changing the embedding as little as possible, as measured by a broad class of norms. We apply LEACE to large language models with a novel procedure called concept scrubbing, which erases target concept information from *every* layer in the network. We demonstrate our method on two tasks: measuring the reliance of language models on part-of-speech information, and reducing gender bias in BERT embeddings. Our code is available at https://github.com/EleutherAI/concept-erasure.

---

## 1 Introduction

The ability to prevent a machine learning system from using a specified concept is important for fairness and interpretability. Popular notions of fairness require that protected attributes should not causally affect predictions [22, 26], and interpretability research often estimates the causal effect of a concept by attempting to remove it from a model’s internal activations [10, 30, 25, 5, 18].

What it means for a model $\mathcal{M}$ to “use” a concept $Z$ is often vague and application-specific, but a necessary condition is that its outputs—and therefore its inputs and hidden states—should have significant *mutual information* with $Z$.$^1$ **Concept erasure** leverages this fact to limit $\mathcal{M}$’s use of $Z$ *without* finetuning or inspecting its parameters. Instead, we edit the input or hidden states $X$ used by $\mathcal{M}$ to minimize the predictive $\mathcal{V}$-information $I_{\mathcal{V}}(X \to Z)$ [43], a tractable lower bound on the mutual information $I(X; Z)$ which measures the degree to which classifiers from the family $\mathcal{V}$ can predict $Z$. Intuitively, if no classifier in $\mathcal{V}$ can outperform a constant function at predicting $Z$—a condition known as **guardedness**—then $\mathcal{M}$ can’t use $Z$ either, at least if $\mathcal{V}$ is expressive enough relative to $\mathcal{M}$.

In this work, we improve upon existing concept erasure techniques using a theory-driven approach. We focus on the case where $\mathcal{V}$ is the set of linear classifiers, and prove a previously unnoticed equivalence: a classification task is linearly guarded if and only if every class has exactly the same mean feature vector (§ 3). Leveraging this equivalence, we derive a simple necessary and sufficient condition for an affine transformation to produce linearly guarded features. We then identify the unique *surgical* transformation in this family—the one that minimizes the mean squared distance from the original features with respect to all norms induced by inner products, including the popular Euclidean and Mahalanobis norms. We name it **LEAst-squares Concept Erasure (LEACE)** (§ 4).

While prior work has focused on preventing linear models from leveraging $Z$, we aim to erase concepts from deep neural networks as well. Interpretability research has shown that networks can be usefully described as encoding features in linear subspaces [11, 24, 41], suggesting that fundamentally nonlinear methods may not be necessary for successful erasure in DNNs. In light of this, we introduce a simple procedure called **concept scrubbing** (§ 6), which sequentially applies LEACE to the activations at each layer of a deep network.

We empirically validate our proposals, demonstrating the superiority of LEACE for erasing gender bias from BERT embeddings (§ 5.2), and using concept scrubbing to measure the extent to which large language models use part-of-speech information (§ 6).

---
$^1$This follows from the fact that causal dependence is a special kind of statistical dependence [28]. By the data processing inequality, $\mathcal{M}$’s output can’t have any more information about $Z$ than its input or hidden states.

## 2 Preliminaries

Consider a $k$-class classification task over jointly defined random vectors $X$ (the input data) and $Z$ (the one-hot labels), with $X$ of finite first moment and taking values in $\mathbb{R}^d$, and $Z$ taking values in $\mathcal{Z} = \{z \in \{0, 1\}^k \mid \|z\|_1 = 1\}^2$ with each $\mathbb{P}(Z = j) > 0$. Let $\eta(\cdot; \theta) : \mathbb{R}^d \to \mathbb{R}^k$ be a predictor chosen from a function class $\mathcal{V} = \{\eta(\cdot; \theta) \mid \theta \in \Theta\}$ (presumed to contain all constant functions) so as to minimize the expectation $\mathbb{E}[\mathcal{L}(\eta(X), Z)]$ of some $\mathcal{L} : \mathbb{R}^k \times \mathcal{Z} \to [0, \infty)$ in a class $\mathfrak{L}$ of loss functions.

We borrow the concept of **guardedness** from Ravfogel et al. [33], who define it in terms of $\mathcal{V}$-information [43]. We opt for a slightly more general definition here, which is equivalent to theirs in the case of cross-entropy loss (see Appendix G).

**Definition 2.1 (Guardedness).** *Let $X, Z, \mathcal{V}$, and $\mathfrak{L}$ be as defined above, and let $\chi$ be the set of all random vectors of finite first moment taking values in $\mathbb{R}^d$, jointly defined with $Z$.*
*We say $X (\mathcal{V}, \mathfrak{L})$-**guards** $Z$ if, for all losses $\mathcal{L} \in \mathfrak{L}$, it maximizes the minimum expected loss:*
$$X \in \operatorname*{argmax}_{X' \in \chi} \inf_{\theta \in \Theta} \mathbb{E}[\mathcal{L}(\eta(X'; \theta), Z)].$$
*In other words, its conditional distribution $\mathbb{P}(X \mid Z = \cdot)$ is among the worst possible distributions for predicting $Z$ from $X$ using a predictor of the form $\eta(\cdot; \theta) \in \mathcal{V}$ and a loss function in $\mathfrak{L}$.*

**Definition 2.2 (Trivially Attainable Loss).** *The **trivially attainable loss** for labels $Z$ and loss $\mathcal{L}$ is the lowest possible expected loss available to a constant predictor $\eta(x) = \mathbf{b}$: $L_\tau = \inf_{\mathbf{b} \in \mathbb{R}^k} \mathbb{E}[\mathcal{L}(\mathbf{b}, Z)]$.*

*We will sometimes write it $L_\tau^{(Z,\mathcal{L})}$ in cases of possible ambiguity. If there is a specific constant predictor actually achieving this loss, we call it the **trivial predictor** $\eta_\tau = \eta_\tau^{(Z,\mathcal{L})}$.*

We examine this problem in the important case of loss functions $\mathcal{L} : \mathbb{R}^k \times \mathcal{Z} \to [0, \infty)$ which are convex in the prediction $\eta(x)$, and linear predictors that take the functional form $\eta(x; \mathbf{b}, \mathbf{W}) = \mathbf{b} + \mathbf{W}x$, for some bias $\mathbf{b} \in \mathbb{R}^k$ and weight matrix $\mathbf{W} \in \mathbb{R}^{k \times d}$.

**Definition 2.3 (Linear Guardedness).** *If $X (\mathcal{V}, \mathfrak{L})$-guards $Z$, where $\mathfrak{L}$ is the class of nonnegative loss functions which are convex in their first argument, and $\mathcal{V}$ is the class of linear predictors $\eta(x) = \mathbf{b} + \mathbf{W}x$, we say that $X$ **linearly guards** $Z$.*

---
$^2$We frequently use the integer $j \leq k$ to refer to the element of $\mathcal{Z}$ which is 1 at the $j^{\text{th}}$ index and 0 elsewhere.

## 3 Theoretical Results

Our primary theoretical result is that the following conditions are all equivalent:

1. The data $X$ linearly guards the labels $Z$. (Definition 2.3)
2. For all convex losses $\mathcal{L}$, the trivially attainable loss is optimal on $(X, Z)$. (Definition 2.2)
3. The class-conditional mean vectors $\mathbb{E}[X \mid Z = i]$ are equal to the unconditional mean $\mathbb{E}[X]$.
4. Every component of $X$ has zero covariance with every component of $Z$.
5. Every linear classifier evaluated on $X$ exhibits statistical parity w.r.t. $Z$. (App. C)

The equivalence of conditions 1, 2, and 5 is relatively straightforward to show, and the relevant theorems can be found in Appendices B and C. The other equivalences are proven below (cond. 3 $\leftrightarrow$ cond. 2 in § 3.1 and § 3.2); cond. 3 $\leftrightarrow$ 4 in § 3.3).

### 3.1 Equality of Class Centroids Implies Linear Guardedness

The following result establishes the implication from condition 3 to condition 2.

**Theorem 3.1.** *Suppose $\mathcal{L}$ is convex in the linear prediction $\eta$. Then if each class-conditional mean $\mathbb{E}[X \mid Z = i]$ is equal to $\mathbb{E}[X]$, the trivially attainable loss cannot be improved upon.*

*Proof.* Let $\eta(x) = \mathbf{b} + \mathbf{W}x$ be any linear predictor. By Jensen’s inequality,$^3$ the loss with $\eta$ evaluated on $X$ is lower bounded by the loss with $\eta$ evaluated on the unconditional mean of the data $\mathbb{E}[X]$:
$$\begin{aligned} \mathbb{E}[\mathcal{L}(\eta, Z)] &= \mathbb{E}_Z[\mathbb{E}[\mathcal{L}(\eta, Z) \mid Z]] \\ &\geq \mathbb{E}_Z[\mathcal{L}(\mathbb{E}[\eta \mid Z], Z)] & \text{(Jensen's inequality)} \\ &= \mathbb{E}_Z[\mathcal{L}(\mathbf{b} + \mathbf{W}\mathbb{E}[X \mid Z], Z)] & \text{(linearity of } \eta\text{)} \\ &= \mathbb{E}_Z[\mathcal{L}(\mathbf{b} + \mathbf{W}\mathbb{E}[X], Z)]. & \text{(by assumption)} \end{aligned}$$
This in turn is the loss of the constant predictor $\eta'(x) = \mathbf{b} + \mathbf{W}\mathbb{E}[X]$. Since the trivially attainable loss is the best that can be achieved by a constant predictor, and every predictor’s loss is lower bounded by that of some constant predictor, we cannot improve upon the trivially attainable loss. $\square$

Intuitively, this shows that the classifier’s expected loss is lower-bounded by the loss it would receive if each data point were replaced with the centroid of its class. But, if these centroids are all equal, the loss can’t be any lower than what we’d get if every data point were replaced with the *global* mean $\mathbb{E}[X]$. In that case, the data points are indistinguishable and we can’t do better than $\mathbf{W} = \mathbf{0}$.

### 3.2 Linear Guardedness Implies Equality of Class Centroids

We now prove the implication from condition 2 to condition 3. Condition 2 applies when the trivially attainable loss is optimal for *all* convex losses, including cross-entropy loss in particular. And if it holds for cross-entropy loss, we now show that condition 3—the class centroids are equal—must follow. First a more general lemma:

**Lemma 3.2.** *Suppose $\mathcal{L}$ has bounded partial derivatives, which when off-category never vanish and do not depend on the category, i.e. $\partial \mathcal{L}(\eta, z_1)/\partial \eta_i = \partial \mathcal{L}(\eta, z_2)/\partial \eta_i \neq 0$ for all categories $z_1, z_2 \neq i$. If $\mathbb{E}[\mathcal{L}(\eta, Z)]$ is minimized among linear predictors by the constant predictor $\eta(x) = \mathbf{b}^* + \mathbf{W}^*x$ with $\mathbf{W}^* = \mathbf{0}$, then each class-conditional mean $\mathbb{E}[X \mid Z = i]$ is equal to $\mathbb{E}[X]$.*

*Proof.* The first-order optimality condition on the $i^{\text{th}}$ component of our parameters $\mathbf{b}$ and $\mathbf{W}$ yields the equations:
$$\mathbb{E} \left[ \frac{\partial \mathcal{L}(\eta, Z)}{\partial \eta_i} \cdot \frac{\partial \eta_i}{\partial b_i} \right] = 0 \quad \text{and} \quad \mathbb{E} \left[ \frac{\partial \mathcal{L}(\eta, Z)}{\partial \eta_i} \cdot \frac{\partial \eta_i}{\partial \mathbf{W}_i} \right] = \mathbf{0}, \tag{1}$$
where we have used the boundedness of $\mathcal{L}$’s partial derivative and the finite first moment of $\frac{\partial \eta_i}{\partial b_i} = 1$ and $\frac{\partial \eta_i}{\partial \mathbf{W}_i} = X$ to justify (via the Dominated Convergence Theorem) interchanging the derivative with the expectation.

Since $\eta$ is constant over all values of $X$, and $\frac{\partial \eta_i}{\partial b_i} = 1$, the first equation in (1) reduces to:
$$\mathbb{P}(Z = i) \frac{\partial \mathcal{L}(\eta, i)}{\partial \eta_i} + \mathbb{P}(Z \neq i) \frac{\partial \mathcal{L}(\eta, \neq i)}{\partial \eta_i} = 0, \tag{2}$$
where $\frac{\partial \mathcal{L}(\eta, \neq i)}{\partial \eta_i}$ is an abuse of notation denoting the off-category partial derivative, emphasizing its independence of the category $Z$.

---
$^3$Specifically, its generalization to convex functions over $\mathbb{R}^k$. See [12] p. 76.

Similarly, the constancy of $\eta$ and the fact that $\frac{\partial \eta_i}{\partial \mathbf{W}_i} = X$ reduces the second equation in (1) to:
$$\mathbb{P}(Z = i) \frac{\partial \mathcal{L}(\eta, i)}{\partial \eta_i} \cdot \mathbb{E}[X \mid Z = i] + \mathbb{P}(Z \neq i) \frac{\partial \mathcal{L}(\eta, \neq i)}{\partial \eta_i} \cdot \mathbb{E}[X \mid Z \neq i] = \mathbf{0}. \tag{3}$$
Solving for $\mathbb{P}(Z = i) \frac{\partial \mathcal{L}(\eta, i)}{\partial \eta_i}$ in (2) and substituting in (3) gives us:
$$\mathbb{P}(Z \neq i) \frac{\partial \mathcal{L}(\eta, \neq i)}{\partial \eta_i} \cdot \left( \mathbb{E}[X \mid Z \neq i] - \mathbb{E}[X \mid Z = i] \right) = \mathbf{0}.$$
If $\mathbb{P}(Z \neq i) = 0$, then $\mathbb{E}[X] = \mathbb{E}[X \mid Z = i]$ is trivially true. Otherwise, using the non-vanishingness of the off-category partial derivative $\frac{\partial \mathcal{L}(\eta, \neq i)}{\partial \eta_i}$, division yields the equivalence of $\mathbb{E}[X \mid Z = i]$ to $\mathbb{E}[X \mid Z \neq i]$, and hence to the unconditional mean $\mathbb{E}[X]$. $\square$

We now show that Lemma 3.2 applies to the widely used cross entropy loss:

**Theorem 3.3.** *If the class probabilities $\mathbb{P}(Z = j)$ are all nonzero, and the trivially obtainable loss is optimal when $\mathcal{L}(\eta, z) = -\log \frac{\exp(\eta_z)}{\sum_{i=1}^k \exp(\eta_i)}$, then each class has the same mean $\mathbb{E}[X \mid Z = z]$.*

*Proof.* In this case, the trivial predictor $\eta_\tau(Z)_j = \log(\mathbb{P}(Z = j))$ exists, achieving the trivially obtainable loss, which we have assumed optimal. Furthermore, $\mathcal{L}$ has on-category partial derivative $\partial \mathcal{L}(\eta, i)/\partial \eta_i = \exp(\eta_i)/\sum_{j=1}^k \exp(\eta_j) - 1 \in (-1, 0]$, and nonvanishing off-category partial derivative $\partial \mathcal{L}(\eta, \neq i)/\partial \eta_i = \exp(\eta_i)/\sum_{j=1}^k \exp(\eta_j) \in (0, 1)$, both bounded, so the conditions of Lemma 3.2 apply. $\square$

### 3.3 Linearly Guarded Labels Have Zero Covariance with the Features

The next theorem establishes the equivalence of conditions 3 and 4.

**Theorem 3.4.** *Let $X$ be a random vector taking values in $\mathbb{R}^d$ with finite first moment, and $Z$ a random vector taking values in $\{0, 1\}^k$ with one-hot encoding, with each class probability $\mathbb{P}(Z = j)$ being nonzero. Then the class-conditional means $\mathbb{E}[X \mid Z = j]$ are all equal to the unconditional mean $\mathbb{E}[X]$ if and only if every component of $X$ has zero covariance with every component of $Z$, i.e. the cross-covariance matrix $\mathbf{\Sigma}_{XZ}$, whose $(i, j)^{\text{th}}$ entry is $\operatorname{Cov}(X_i, Z_j)$, is the zero matrix.*

*Proof.* Since $Z$ is one-hot, we can rewrite the $(i, j)^{\text{th}}$ entry of $\mathbf{\Sigma}_{XZ}$ as:
$$\mathbb{E}[X_i Z_j] - \mathbb{E}[X_i]\mathbb{E}[Z_j] = \mathbb{P}(Z = j) (\mathbb{E}[X_i \mid Z = j] - \mathbb{E}[X_i]).$$
As $\mathbb{P}(Z = j) > 0$, it follows that $\mathbb{E}[X_i \mid Z = j] = \mathbb{E}[X_i]$ if and only if $\operatorname{Cov}(X_i, Z_j) = 0$. $\square$

We have thus established the equivalence of the first four conditions stated earlier. See Appendix C for the last one, on statistical parity.

## 4 Least-Squares Concept Erasure

In Section 3 we saw that $X$ linearly guards $Z$ if and only if each component of $X$ has zero covariance with each component of $Z$. We will now characterize the set of affine transformations $r(x) = \mathbf{P}x + \mathbf{b}$ such that $r(X)$ linearly guards $Z$.

**Theorem 4.1.** *Let $X$ and $Z$ be random vectors taking values in $\mathbb{R}^d$ and $\mathbb{R}^k$ respectively, with $X$ of finite first moment. Then given some affine function $r(x) = \mathbf{P}x + \mathbf{b}$, the modified random vector $r(X)$ linearly guards $Z$ if and only if the columns of the cross-covariance matrix $\mathbf{\Sigma}_{XZ}$ are contained in the null space of $\mathbf{P}$.*

*Proof.* From Theorem 3.4 we know that $r(X)$ linearly guards $Z$ if and only if $\operatorname{Cov}(r(X), Z)$ is the zero matrix. By the linearity property of cross-covariance, we have:
$$\operatorname{Cov}(r(X), Z) = \operatorname{Cov}(\mathbf{P}X + \mathbf{b}, Z) = \mathbf{P}\operatorname{Cov}(X, Z) = \mathbf{P}\mathbf{\Sigma}_{XZ}.$$
Therefore, $r(X)$ linearly guards $Z$ if and only if $\operatorname{ker}(\mathbf{P}) \supseteq \operatorname{colsp}(\mathbf{\Sigma}_{XZ})$. $\square$

**Implications for prior work.** Notably, the above theorems imply that three previously proposed methods in the literature, Spectral Attribute Removal (SAL) [36], Mean Projection [17], and Fair PCA [20], are guaranteed to achieve linear guardedness given suitable hyperparameters. See Appendix D for further discussion.

### 4.1 Derivation of LEACE

Theorem 4.1 is a very weak condition, which is far from identifying unique values for $\mathbf{P}$ and $\mathbf{b}$. In most applications, however, we’d like to make a “small” edit to $X$ so that useful information contained in $X$ is maximally preserved. We operationalize the notion of a small edit in terms of the mean squared norm $\mathbb{E}\|r(X) - X\|^2_{\mathbf{M}}$ defined by some positive-definite inner product $\mathbf{M}$,$^4$ which can be thought of as a local quadratic approximation to *any* measure of divergence between $X$ and $r(X)$ (such as Kullback–Leibler divergence, for example). While we are primarily interested in the Euclidean ($\mathbf{M} = \mathbf{I}$) and Mahalanobis ($\mathbf{M} = \mathbf{\Sigma}_{XX}^+$) norms, it will turn out that there is a *single* erasure function that minimizes *all* such norms simultaneously. We will see in Section 6 that ensuring edits are small in this sense provides substantial benefit to downstream task performance as compared to other methods which also guard the labels $Z$.

Below, we derive the optimal eraser under the assumption that $X$ and $Z$ are centered.

**Theorem 4.2.** *Let $X$ and $Z$ be centered random vectors taking values in $\mathbb{R}^d$ and $\mathbb{R}^k$ respectively, each of finite second moment. Let $\mathbf{M} \in \mathbb{R}^{d \times d}$ be a p.s.d. matrix defining a (possibly degenerate) inner product on $\mathbb{R}^d$: $\langle \mathbf{x}, \mathbf{y} \rangle_{\mathbf{M}} = \mathbf{x}^T \mathbf{M} \mathbf{y}$. Let $\mathbf{\Sigma}_{XX} \in \mathbb{R}^{d \times d}$ be $X$’s covariance matrix, and $\mathbf{\Sigma}_{XZ} \in \mathbb{R}^{d \times k}$ be the cross-covariance matrix of $X$ and $Z$. Let $\mathbf{A}^+$ denote the Moore-Penrose pseudoinverse of a matrix $\mathbf{A}$, and let $\mathbf{A}^{1/2}$ be the p.s.d. square root of a p.s.d. matrix $\mathbf{A}$. Then the objective*
$$\operatorname*{argmin}_{\mathbf{P} \in \mathbb{R}^{d \times d}} \mathbb{E} \left[ \|\mathbf{P}X - X\|_{\mathbf{M}}^2 \right] \quad \text{subject to } \operatorname{Cov}(\mathbf{P}X, Z) = \mathbf{0}$$
*has the following solution:*
$$\mathbf{P}^* = \mathbf{I} - \mathbf{W}^+ \mathbf{P}_{\mathbf{W}\mathbf{\Sigma}_{XZ}} \mathbf{W},$$
*where $\mathbf{W}$ is the whitening transformation $(\mathbf{\Sigma}_{XX}^{1/2})^+$ and $\mathbf{P}_{\mathbf{W}\mathbf{\Sigma}_{XZ}} = (\mathbf{W}\mathbf{\Sigma}_{XZ})(\mathbf{W}\mathbf{\Sigma}_{XZ})^+$ is the orthogonal projection matrix onto $\operatorname{colsp}(\mathbf{W}\mathbf{\Sigma}_{XZ})$.*

*Proof.* See Appendices E.1 and E.2 for two independent proofs of Theorem 4.2. $\square$

The above theorem assumes that the random vectors $X$ and $Z$ are centered, and does not include a bias term. Below we extend our results to the uncentered case, and derive the optimal bias $\mathbf{b}^*$.

**Theorem 4.3.** *Let $X$ and $Z$ be random vectors taking values in $\mathbb{R}^d$ and $\mathbb{R}^k$ respectively, each of finite second moment. Define $\mathbf{M}$ and $\mathbf{P}^*$ as in Theorem 4.2 and $\mathbf{b}^* = \mathbb{E}[X] - \mathbf{P}^*\mathbb{E}[X]$. Then $(\mathbf{P}^*, \mathbf{b}^*)$ minimizes $\mathbb{E}\|\mathbf{P}X + \mathbf{b} - X\|^2_{\mathbf{M}}$, subject to $\operatorname{Cov}(\mathbf{P}X + \mathbf{b}, Z) = \mathbf{0}$.*

*Proof.* Let $\mathbf{P} \in \mathbb{R}^{d \times d}$ and define $\tilde{X} = X - \mathbb{E}[X]$ and $\mathbf{c} = \mathbf{P}\mathbb{E}[X] + \mathbf{b} - \mathbb{E}[X]$. Then,
$$\begin{aligned} \mathbb{E} \|\mathbf{P}X + \mathbf{b} - X\|_{\mathbf{M}}^2 &= \mathbb{E} \|(\mathbf{P}\tilde{X} - \tilde{X}) + \mathbf{c}\|_{\mathbf{M}}^2 \\ &= \mathbb{E} \|\mathbf{P}\tilde{X} - \tilde{X}\|_{\mathbf{M}}^2 + 2\mathbb{E} [\mathbf{P}\tilde{X} - \tilde{X}]^T \mathbf{M}\mathbf{c} + \mathbf{c}^T \mathbf{M}\mathbf{c} \\ &= \mathbb{E} \|\mathbf{P}\tilde{X} - \tilde{X}\|_{\mathbf{M}}^2 + \mathbf{c}^T \mathbf{M}\mathbf{c}, \end{aligned}$$
where we have eliminated the middle term because $\mathbf{P}$ is linear and $\mathbb{E}[\tilde{X}] = \mathbf{0}$. Since $\mathbf{M}$ is p.s.d., our objective is minimized for $\mathbf{c} = \mathbf{0}$, i.e. $\mathbf{b} = \mathbb{E}[X] - \mathbf{P}\mathbb{E}[X]$. The problem thus reduces to choosing $\mathbf{P}$ so as to minimize $\mathbb{E} \|\mathbf{P}\tilde{X} - \tilde{X}\|_{\mathbf{M}}^2$ subject to $\operatorname{Cov}(\mathbf{P}X + \mathbf{b}, Z) = \operatorname{Cov}(\mathbf{P}\tilde{X}, Z) = \mathbf{0}$, which Theorem 4.2 shows occurs when $\mathbf{P} = \mathbf{P}^*$. $\square$

---
$^4$Our proofs also include degenerate “inner products” where $\mathbf{M}$ is singular, and the associated seminorms.

[IMAGE: Figure 1: LEACE projection in 3 steps. First the data is whitened, ensuring equal variance in all directions. It is then orthogonally projected onto colsp(WΣXZ)⊥, guaranteeing linear guardedness. Finally, we unwhiten the data so that its covariance structure mimics the original.]

Putting together Theorems 4.2 and 4.3 and rearranging, we arrive at the LEACE formula:
$$r_{\text{LEACE}}(\mathbf{x}) = \mathbf{x} - \mathbf{W}^+ \mathbf{P}_{\mathbf{W}\mathbf{\Sigma}_{XZ}} \mathbf{W}(\mathbf{x} - \mathbb{E}[X]) \tag{1}$$
Intuitively, LEACE de-means and whitens $\mathbf{x}$, projects onto the subspace responsible for correlations between $X$ and $Z$, then unwhitens the result. Finally, it subtracts this value from $\mathbf{x}$, thereby surgically removing the linearly available information about $Z$.

### 4.2 Oblique Projections are Least-Squares Optimal

Prior work on linear concept erasure has assumed that erasure functions should be orthogonal projections [29, 32, 36], appealing to the well-known fact that an orthogonal projection of a point $\mathbf{x}$ onto a subspace $U$ yields the nearest point in $U$ to $\mathbf{x}$. But even in the case where $X$ is centered, $r_{\text{LEACE}}$ is *not* an orthogonal projection in general. Orthogonal projection matrices are symmetric, and $\mathbf{I} - \mathbf{W}^+ \mathbf{P}_{\mathbf{W}\mathbf{\Sigma}_{XZ}} \mathbf{W}$ is only symmetric in the special case where $\mathbf{P}_{\mathbf{W}\mathbf{\Sigma}_{XZ}}$ and $\mathbf{W}$ commute. It is an **oblique** projection however, since applying $\mathbf{P}^*$ twice yields the same result as applying it once: $(\mathbf{P}^*)^2 = \mathbf{I} - 2\mathbf{W}^+\mathbf{P}_{\mathbf{W}\mathbf{\Sigma}_{XZ}}\mathbf{W} + \mathbf{W}^+\mathbf{P}_{\mathbf{W}\mathbf{\Sigma}_{XZ}}\mathbf{W}\mathbf{W}^+\mathbf{P}_{\mathbf{W}\mathbf{\Sigma}_{XZ}}\mathbf{W} = \mathbf{P}^*$.

Orthogonal projections are generally not least-squares optimal for concept erasure because the necessary and sufficient condition for linear guardedness, $\mathbf{P}\mathbf{\Sigma}_{XZ} = \mathbf{0}$, is a constraint on the *nullspace* of $\mathbf{P}$, and not on its range. We may freely choose the range of the projection to minimize the mean squared distance, as long as we zero out $\operatorname{colsp}(\mathbf{\Sigma}_{XZ})$. In Figure 1, an orthogonal projection would map all points onto the the dashed line, thereby preserving less of the variance of the original data than LEACE does (green line). See Appendix F for a concrete example.

### 4.3 Extension to Continuous $Z$

While not a focus of this work, it’s worth noting that LEACE can also be applied to the setting where $Z$ takes arbitrary values in $\mathbb{R}^k$, as long as we restrict ourselves to the ordinary least squares regression loss $\mathcal{L}(\eta, \mathbf{z}) = \|\eta - \mathbf{z}\|_2^2$. In particular, the proofs of equivalence between conditions 1 and 2 given in Appendix B make no categorical assumption on $Z$, and the equivalence between the optimality of a zero weight matrix (condition 2) and zero cross-covariance (condition 4) is well known in the OLS setting. We can then apply Theorems 4.2 and 4.3, which also make no categorical assumption, to derive the same optimal affine eraser as in the categorical case.

## 5 Evaluation

### 5.1 Intrinsic Evaluation

Following Ravfogel et al. [31] we evaluate the ability of our method to remove gender information from the last hidden layer of a frozen BERT model. We use the biographies dataset of De-Arteaga et al. [6], composed of short biographies annotated by both binary gender and profession. We embed each biography with the [CLS] embedding in the last layer of BERT, enforce the same conditional-mean constraint to remove gender information from the [CLS], and then evaluate the performance of the model, after the intervention, on the main task of profession prediction. We compare our intervention with RLACE [31], which uses gradient-based optimization to solve a linear concept-erasure adversarial game.

**Concept erasure results.** First, we evaluate the ability of logistic regression classifiers to recover the removed information. The results, presented in Fig. 2, show that our method is the only to achieve random accuracy (perfect erasure) with a small edit, although RLACE (but not INLP) comes close. At the same time, our method is around 2 orders of magnitude faster, and does not require gradient-based optimization.

### 5.2 Downstream Fairness

How does our intervention affect the behavior of the model on the main classification task of profession prediction? We fit a logistic regression profession-prediction classifier over the projected [CLS] embeddings.

To measure the bias in a classifier, we follow De-Arteaga et al. [6] and use the TPR-GAP measure, which quantifies the bias in a classifier by considering the difference (GAP) in the true positive rate (TPR) between individuals with different protected attributes (e.g. race or gender). We use the notation $\text{GAP}_{z,y}^{\text{TPR}}$ to denote the TPR-gap in some main-class label $y$ (e.g. “nurse” prediction) for some protected group $z$ (e.g. “female”), we also consider $\text{GAP}_z^{\text{TPR,RMS}}$, the RMS of the TPR-gap across all professions for a protected group $z$:
$$\text{GAP}_z^{\text{TPR,RMS}} = \sqrt{\frac{1}{|\mathcal{C}|} \sum_{y \in \mathcal{C}} (\text{GAP}_{z,y}^{\text{TPR}})^2}$$
To calculate the relation between the bias the model exhibits and the bias in the data, we also calculate $\rho_{(\text{GAP}^{\text{TPR}},\% \text{Women})}$, the correlation between the TPR gap in a given profession and the percentage of women in that profession.

[IMAGE: Figure 3: The correlation between GAP_female,y^TPR and the relative proportion of women in profession y, for BERT embedding, before (left; R=0.867) and after (right; R=0.392) the projection.]

[IMAGE: Figure 2: Gender prediction accuracy after bias-removal projection versus the mean squared distance from the original embedding for INLP, RLACE, and LEACE on BERT embeddings.]

**Results.** The main-task classifier achieves profession-prediction accuracy of 77.3% on the projected embeddings (compared with 79.3% over the original embeddings), indicating that the intervention minimally affects the ability to predict the profession of a person from the embedding of their biography. At the same time, the TPR gap drops significantly from 0.198 to 0.084, indicating a sharp drop in the biased behavior of the profession classifier. Indeed, inspecting the correlation $\rho_{(\text{GAP}^{\text{TPR}},\% \text{Women})}$ between the gap (per profession) and the embedding of women in this profession, we see that this correlation plummets from 0.867 to 0.392 after erasure. Re-fitting the main-task logistic regression classifier over the projected embeddings yields a slightly higher main-task accuracy of 78.1%, at the price of significantly increasing the TPR gap to 0.158.$^5$

### 5.3 Revisiting Amnesic Probing

Elazar et al. [10] have introduced the idea of *amnesic probing* as a causal intervention that aims to test the importance of a given concept (e.g. part-of-speech tag) to some main task (e.g. language modeling). They applied Iterative Nullspace Projection (INLP) to remove different concepts from the activations of the model, and assessed the degree to which its behavior changed when performing masked language modeling. Since INLP often requires dozens of iterations to completely erase the concept, its usage in this context raises concerns of collateral damage due to magnitude of the intervention and the non-exhaustive nature of INLP removal. Here, we replicate their experiments on the `bert-base-uncased` model with our interventions.

**Experimental setup.** We use part-of-speech (POS) tags as our concept of interest. We collect sentences and their coarse POS tags (“Noun”, “Verb” etc.; 18 in total) from the English Universal Dependencies dataset [27]. We tokenize the sentences with the BERT tokenizer and map each wordpiece to the POS tag of the word to which it belongs. We collect the unmasked BERT embeddings for each layer, intervene to linearly erase the POS concept from that layer, and continue the forward pass until the last layer, from which we compute the distribution of the MLM over the vocabulary. Note that in each experiment we intervene on a single layer. We quantify the decrease in accuracy following the intervention, as well as the increase in the loss. We compare with a baseline intervention of a random orthogonal projection whose null space has the same rank as the label space (18). For INLP, we perform 20 iterations. This is needed because INLP does not effectively remove the concept; even after 20 iterations, classification accuracy is above majority accuracy. As a result, INLP reduces the rank of the embedding by 360. By contrast, our method decreases the rank just by 17.

[IMAGE: Figure 4: Amnesic probing results on bert-base-uncased.]

**Results.** The results are shown in Fig. 4b. Our intervention only mildly changes BERT LM accuracy and loss until layer 8, with the highest drop recorded in layer 11. INLP, in contrast, shows maximum effect at layer 6. Since it removes hundreds of dimensions, it is difficult to attribute this effect to the erasure of the concept. These results suggest that the *causal* effect of the POS concept on the language model is concentrated in layer 11. Interestingly, this stands in contrast with POS linear probing results, which are optimal at earlier layers [38]. As Elazar et al. [10] have noted, probing does not generally correlate with intervention-based analysis techniques.

---
$^5$The softmax probabilities of a multiclass logistic regression classifier can leak the removed information if another classifier is stacked on top of it [33], though this setup is not linear.

## 6 Concept Scrubbing

Unfortunately, Elazar et al. [10] were forced to limit their interventions to a single layer due to the limitations of INLP. INLP often requires the deletion of several dozen dimensions before linear guarding is achieved—as demonstrated in Figure 2. Kumar et al. [21] show empirically and theoretically that INLP causes needless “collateral damage” to useful parts of the embedding that are orthogonal to the concept being erased. Because of this collateral damage, it’s impossible to apply INLP to multiple layers of a transformer without causing its outputs to collapse into gibberish.

Instead, we would like to erase all linear information about a concept in the activations at *every* layer, which we term **concept scrubbing**. LEACE makes concept scrubbing possible and eminently practical. It causes minimal collateral damage, induces little computational overhead, and the covariance statistics it relies on can be computed in a *streaming* fashion, without ever storing all the hidden states in memory or on disk.

**Algorithm.** Any intervention on the model at layer $\ell$ changes the distribution of hidden states at layers $\ell' > \ell$. Because of this, the naive approach of independently fitting LEACE parameters $(\mathbf{P}, \mathbf{b})$ for all layers of the clean model, then applying them all at once, may fail to fully erase the target concept. Instead, we fit LEACE parameters *sequentially*, starting from the first layer and proceeding to the final layer. After we compute $(\mathbf{P}, \mathbf{b})$ for a layer, we immediately use them to scrub the hidden states for that layer, then feed these scrubbed embeddings to the next layer (Algorithm 1).

| **Algorithm 1** Concept scrubbing |
| :--- |
| **Require:** Model with $\ell$ layers $f = f_\ell \circ \dots \circ f_1$ |
| **Require:** Design matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$ |
| **Require:** Label matrix $\mathbf{Z} \in \mathbb{R}^{n \times k}$ |
| **Ensure:** LEACE parameters for each layer in $f$ |
| 1: $\mathbf{H}_1 \leftarrow \text{Embed}(\mathbf{X})$ |
| 2: $L \leftarrow \text{list()}$ |
| 3: **for** $l \in 1 \dots \ell$ **do** |
| 4: $\quad \text{Fit } (\mathbf{P}, \mathbf{b}) \text{ on } \mathbf{H}_l \text{ and } \mathbf{Z}$ |
| 5: $\quad \text{Append } (\mathbf{P}, \mathbf{b}) \text{ to } L$ |
| 6: $\quad \mathbf{H}_l \leftarrow \mathbf{P}(\mathbf{H}_l - \mu_{\mathbf{H}_l}) + \mu_{\mathbf{H}_l}$ (Eq. 1) |
| 7: $\quad \mathbf{H}_{l+1} \leftarrow f_l(\mathbf{H}_l)$ |
| 8: **return** $L$ |

| | | LLaMA | | | Pythia | | |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Condition | 7B | 13B | 30B | 160M | 1.4B | 6.9B | 12B |
| No intervention | 0.69 | 0.66 | 0.62 | 0.90 | 0.70 | 0.64 | 0.62 |
| Random erasure | 0.69 | 0.66 | 0.62 | 0.99 | 0.72 | 0.66 | 0.63 |
| LEACE | 1.73 | 1.84 | 1.96 | 2.79 | 2.25 | 3.57 | 3.20 |
| SAL | 3.24 | 3.26 | 3.16 | 3.53 | 3.44 | 4.17 | 4.69 |
| unigram entropy | 2.90 | 2.90 | 2.90 | 2.66 | 2.66 | 2.66 | 2.66 |

**Table 1:** Perplexity in autoregressive language models when removing linearly available part-of-speech information from the input to each transformer layer. Units are bits per UTF-8 byte. The unigram baseline assigns probabilities to tokens based only on their frequency and not on the context.

### 6.1 Experimental Details

**Dataset.** For each model family, we use a sample from the respective pretraining distribution: the validation split of the Pile [13] for the Pythia models [2], and the RedPajama replication of the LLaMA pretraining corpus for the LLaMA family [39]. sampling a slice of $2^{22}$ tokens for fitting the LEACE parameters and another slice of $2^{22}$ tokens for evaluation. Since neither corpus comes with part-of-speech tags, we use the model from the SpaCy library [19] to automatically generate Universal Dependency tags [23].

**Baseline method.** We also run concept scrubbing using full-rank SAL [36], which is similar to our method but lacks a bias term and does not adjust for correlations between features (Appendix D).

**Architecture.** We focus on autoregressive language models. We evaluate our method on EleutherAI’s Pythia 160M, 1.4B, 6.9B, and 12B models [2], and Meta’s LLaMA 7B, 13B, and 30B [39]. We apply concept erasure to the input of each transformer block, immediately after normalization is applied (LayerNorm or RMSNorm).

**Randomized erasure.** Almost any intervention on a neural network will cause its performance to degrade to some extent. Following Elazar et al. [10], we isolate the effect of the concept erasure by comparing it to a control condition in which we orthogonally project onto a *random* linear subspace of the same rank as the cross-covariance matrix. To reduce the variance of our results, we sample a fresh subspace for each minibatch, and erase that subspace at each layer, reporting the cross-entropy loss averaged over subspaces.

**Training efficiency.** Algorithm 1 avoids redundant computation by caching the layer $i$ hidden states for *every* data point, then using them to run layer $i + 1$. This approach has the downside of requiring a large amount of memory or disk space during training (up to 500GB in our experiments). It’s possible to avoid caching any hidden states and instead recompute them as needed, at the expense of increasing the total compute cost from $O(\ell)$ to $O(\ell^2)$.

### 6.2 Results

We find strong evidence that autoregressive language models heavily rely on linearly encoded part-of-speech information. While erasing a randomly selected subspace has little to no effect on language modeling performance, scrubbing away part-of-speech information induces a large increase in perplexity across all models (Table 1).

The specific numbers, however, depend on the erasure method used: SAL induces significantly larger increases in perplexity for all models we tested. We take this to mean that SAL inflicts more collateral damage on other useful features in the embedding than LEACE does. In other words, interventions made with LEACE are more *surgical* than those made with prior work; they more closely approximate the ideal of a perfect intervention which only erases the target concept and keeps everything else fixed [40, 15]. If this experiment were conducted with SAL alone, we would have *overestimated* the causal effect of part-of-speech.

---

## 7 Limitations and Future Work

Much work remains to be done to validate concept scrubbing. Specifically, we’d like to see experiments that target concepts much narrower than part-of-speech, and use behavioral metrics to determine whether scrubbing changes the network in the ways we’d intuitively expect. If these experiments succeed, an exciting next step would be the incorporation of concept scrubbing into the pretraining and/or finetuning process. This may make it possible to train deep neural networks subject to *conceptual constraints*. It remains to be seen if gradient-based optimizers will be able to “circumvent” such constraints by encoding protected attributes in completely nonlinear ways.

In this work, we focused exclusively on *linear* concept erasure due to its simplicity and tractability. Some authors have proposed nonlinear concept erasure techniques based on kernel methods, but have found that erasure functions fit using one kernel do not generalize well to other kernels [32, 36]. We conjecture that it is intractable to nondestructively edit $X$ so as to prevent a general nonlinear adversary from recovering $Z$, unless the data generating process for $X$ is known in detail.$^6$

A major motivation of concept erasure is that it promises to prevent models from using a concept in a *post hoc*, model-agnostic fashion. But if our concept scrubbing procedure turns out to yield unsatisfactory results in practical use cases, the most promising research direction might then be to improve *model-specific* techniques, such as those that modify the training procedure [8, 9, 14].

---

## 8 Acknowledgements

We are grateful to CoreWeave for providing the compute resources used in Section 6. Shauli Ravfogel is grateful to be supported by the Bloomberg Data Science PhD Fellowship.

---
$^6$We suspect erasing a concept is at least as hard as extracting it from the original embedding. But in the worst case, information about $Z$ could be encoded *cryptographically* in $X$, which would be intractable to decode given standard computational complexity assumptions. If the data is generated by a known algorithm, however, it may be possible to efficiently eliminate mutual information between $Z$ and $X$ by simply breaking the links in the causal graph that connect them.

## References

[1] UC Berkeley. The Hilbert space of random variables. Lecture Notes Electrical Engineering 126, 2018. URL https://inst.eecs.berkeley.edu/~ee126/sp18/projection.pdf.

[2] Stella Biderman, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. *arXiv preprint arXiv:2304.01373*, 2023.

[3] Tolga Bolukbasi, Kai-Wei Chang, James Y. Zou, Venkatesh Saligrama, and Adam T. Kalai. Man is to computer programmer as woman is to homemaker? Debiasing word embeddings. *Advances in Neural Information Processing Systems*, 29:4349–4357, 2016. URL https://proceedings.neurips.cc/paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf.

[4] Xilun Chen, Yu Sun, Ben Athiwaratkun, Claire Cardie, and Kilian Weinberger. Adversarial deep averaging networks for cross-lingual sentiment classification. *Transactions of the Association for Computational Linguistics*, 6:557–570, 2018. URL https://aclanthology.org/Q18-1039.

[5] Verna Dankers, Christopher Lucas, and Ivan Titov. Can transformer be too compositional? Analysing idiom processing in neural machine translation. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 3608–3626, 2022.

[6] Maria De-Arteaga, Alexey Romanov, Hanna Wallach, Jennifer Chayes, Christian Borgs, Alexandra Chouldechova, Sahin Geyik, Krishnaram Kenthapadi, and Adam Tauman Kalai. Bias in bios: A case study of semantic representation bias in a high-stakes setting. In *Proceedings of the Conference on Fairness, Accountability, and Transparency*, FAT* ’19, page 120–128, New York, NY, USA, 2019. Association for Computing Machinery. ISBN 9781450361255. doi: 10.1145/3287560.3287572. URL https://doi.org/10.1145/3287560.3287572.

[7] Sunipa Dev, Tao Li, Jeff M. Phillips, and Vivek Srikumar. OSCaR: Orthogonal subspace correction and rectification of biases in word embeddings. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 5034–5050, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.411. URL https://aclanthology.org/2021.emnlp-main.411.

[8] Harrison Edwards and Amos Storkey. Censoring representations with an adversary. In *International Conference in Learning Representations*, pages 1–14, May 2016. URL https://arxiv.org/abs/1511.05897.

[9] Yanai Elazar and Yoav Goldberg. Adversarial removal of demographic attributes from text data. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 11–21, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1002. URL https://aclanthology.org/D18-1002.

[10] Yanai Elazar, Shauli Ravfogel, Alon Jacovi, and Yoav Goldberg. Amnesic probing: Behavioral explanation with amnesic counterfactuals. *Transactions of the Association for Computational Linguistics*, 9:160–175, 2021. doi: 10.1162/tacl_a_00359. URL https://aclanthology.org/2021.tacl-1.10.

[11] Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Hatfield Zac Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. A mathematical framework for transformer circuits. *Transformer Circuits Thread*, 2021.

[12] Thomas S. Ferguson. *Mathematical Statistics*. Academic Press, Cambridge, MA, 1967.

[13] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. The Pile: An 800GB dataset of diverse text for language modeling. *arXiv preprint arXiv:2101.00027*, 2020.

[14] Atticus Geiger, Zhengxuan Wu, Hanson Lu, Josh Rozner, Elisa Kreiss, Thomas Icard, Noah Goodman, and Christopher Potts. Inducing causal structure for interpretable neural networks. In *International Conference on Machine Learning*, pages 7324–7338. PMLR, 2022.

[15] Christopher Grimsley, Elijah Mayfield, and Julia R.S. Bursten. Why attention is not explanation: Surgical intervention and causal reasoning about neural models. In *Proceedings of the Twelfth Language Resources and Evaluation Conference*, pages 1780–1790, Marseille, France, May 2020. European Language Resources Association. URL https://aclanthology.org/2020.lrec-1.220.

[16] Pantea Haghighatkhah, Wouter Meulemans, Bettina Speckmann, Jérôme Urhausen, and Kevin Verbeek. Obstructing classification via projection. In Filippo Bonchi and Simon J. Puglisi, editors, *46th International Symposium on Mathematical Foundations of Computer Science*, Leibniz International Proceedings in Informatics, LIPIcs. Schloss Dagstuhl - Leibniz-Zentrum für Informatik, 2021.

[17] Pantea Haghighatkhah, Antske Fokkens, Pia Sommerauer, Bettina Speckmann, and Kevin Verbeek. Better hit the nail on the head than beat around the bush: Removing protected attributes with a single projection. pages 8395–8416, December 2022. doi: 10.18653/v1/2022.emnlp-main.575. URL https://aclanthology.org/2022.emnlp-main.575.

[18] Evan Hernandez and Jacob Andreas. The low-dimensional linear geometry of contextualized word representations. In *Proceedings of the 25th Conference on Computational Natural Language Learning*, pages 82–93, 2021.

[19] Matthew Honnibal, Ines Montani, Sofie Van Landeghem, and Adriane Boyd. spaCy: Industrial-strength Natural Language Processing in Python, 2020.

[20] Matthäus Kleindessner, Michele Donini, Chris Russell, and Muhammad Bilal Zafar. Efficient fair PCA for fair representation learning. In *International Conference on Artificial Intelligence and Statistics*, pages 5250–5270. PMLR, 2023.

[21] Abhinav Kumar, Chenhao Tan, and Amit Sharma. Probing classifiers are unreliable for concept removal and detection. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, *Advances in Neural Information Processing Systems*, volume 35, pages 17994–18008. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper_files/paper/2022/file/725f5e8036cc08adeba4a7c3bcbc6f2c-Paper-Conference.pdf.

[22] Matt J. Kusner, Joshua Loftus, Chris Russell, and Ricardo Silva. Counterfactual fairness. *Advances in Neural Information Processing Systems*, 30, 2017.

[23] Ryan McDonald, Joakim Nivre, Yvonne Quirmbach-Brundage, Yoav Goldberg, Dipanjan Das, Kuzman Ganchev, Keith Hall, Slav Petrov, Hao Zhang, Oscar Täckström, et al. Universal dependency annotation for multilingual parsing. In *Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, pages 92–97, 2013.

[24] Neel Nanda. Actually, Othello-GPT has a linear emergent world model, Mar 2023. URL <https://neelnanda.io/mechanistic-interpretability/othello>.

[25] Vassilina Nikoulina, Maxat Tezekbayev, Nuradil Kozhakhmet, Madina Babazhanova, Matthias Gallé, and Zhenisbek Assylbekov. The rediscovery hypothesis: Language models need to meet linguistics. *Journal of Artificial Intelligence Research*, 72:1343–1384, 2021.

[26] Hamed Nilforoshan, Johann D. Gaebler, Ravi Shroff, and Sharad Goel. Causal conceptions of fairness and their consequences. In *International Conference on Machine Learning*, pages 16848–16887. PMLR, 2022.

[27] Joakim Nivre, Marie-Catherine de Marneffe,