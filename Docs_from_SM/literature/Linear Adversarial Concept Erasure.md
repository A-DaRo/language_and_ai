# Linear Adversarial Concept Erasure

**Shauli Ravfogel**$^{1 \ 2}$ **Michael Twiton**$^3$ **Yoav Goldberg**$^{1 \ 2}$ **Ryan Cotterell**$^4$

### Abstract

Modern neural models trained on textual data rely on pre-trained representations that emerge without direct supervision. As these representations are increasingly being used in real-world applications, the inability to *control* their content becomes an increasingly important problem. We formulate the problem of identifying and erasing a linear subspace that corresponds to a given concept, in order to prevent linear predictors from recovering the concept. We model this problem as a constrained, linear maximin game, and show that existing solutions are generally not optimal for this task. We derive a closed-form solution for certain objectives, and propose a convex relaxation, RLACE, that works well for others. When evaluated in the context of binary gender removal, the method recovers a low-dimensional subspace whose removal mitigates bias by intrinsic and extrinsic evaluation. We show that the method is highly expressive, effectively mitigating bias in deep nonlinear classifiers while maintaining tractability and interpretability.

[https://github.com/shauli-ravfogel/rlace-icml](https://github.com/shauli-ravfogel/rlace-icml)

---

### 1. Introduction

We are interested in the question of removing information from a given real-valued vector representation, e.g., representations that are obtained via neural language encoders of text (Melamud et al., 2016; Peters et al., 2018; Howard and Ruder, 2018; Devlin et al., 2019). Specifically, we ask the following: Given a set of vectors $\mathbf{x}_1, \dots, \mathbf{x}_N \in \mathbb{R}^D$

$^1$Department of Computer Science, Bar Ilan University $^2$Allen Institute for Artificial Intelligence $^3$Independent researcher $^4$ETH Zürich. Correspondence to: Shauli Ravfogel <shauli.ravfogel@gmail.com>, Michael Twiton <mtwito101@gmail.com>, Yoav Goldberg <yoav.goldberg@gmail.com>, Ryan Cotterell <ryan.cotterell@inf.ethz.ch>.

Proceedings of the $39^{\text{th}}$ International Conference on Machine Learning, Baltimore, Maryland, USA, PMLR 162, 2022. Copyright 2022 by the author(s).

[IMAGE: Figure 1. Removal of gender information from GloVe representations using RLACE, after PCA (Experiment § 5.1). Left: original space; Right: after a rank-1 RLACE projection. Word vectors are colored according to their being male-biased or female-biased.]

and a labeling $y_1, \dots, y_N$, with each $y_n \in \mathcal{Y}$,$^1$ a label set of concepts, can we derive a function $r(\cdot)$ such that the resulting vectors $r(\mathbf{x}_1), \dots, r(\mathbf{x}_N)$ are *not* predictive of the concept labels $y_1, \dots, y_N$, but $r(\mathbf{x}_n)$ preserves the information found in $\mathbf{x}_n$ as much as possible? However, unlike methods (Edwards and Storkey, 2015; Chen et al., 2018; Xie et al., 2017; Elazar and Goldberg, 2018; Zhang et al., 2018) that require a modification to the training process, here we are interested in *post-hoc* methods, which assume a fixed, pre-trained encoder (such as GloVe (Pennington et al., 2014), BERT (Devlin et al., 2019), or GPT-2 (Radford et al., 2019) and aim to learn an additional function $r(\cdot)$ that removes information from the fixed representations. Thus, this problem generalizes bias mitigation, e.g., by removing the concept of gender, from word representations (Bolukbasi et al., 2016).

In this article, we focus on the special case where the function $r(\cdot)$ is a linear transformation—specifically, we aim to identify and remove a linear concept subspace from the representation using an orthogonal projection matrix in such a manner that prevents any linear classifier from recovering the value of the concept. By restricting ourselves to the linear case, we obtain a practical solution while also enjoying the increased interpretability of linear methods. However, by imposing the constraint that the linear transformation is a non-low-rank orthogonal projection matrix, we enforce that the linear transformation is minimally invasive.

Isolating a linear concept space in representations of text was pioneered by Bolukbasi et al. (2016), who used principal component analysis to identify a linear gender bias subspace in static word representations. After identifying the bias subspace, Bolukbasi et al. (2016) gave a recipe for mitigating gender bias from the word representations.

---
$^1$Throughout this paper, we take $\mathcal{Y} = \{0, 1\}$.

However, Gonen and Goldberg (2019) later demonstrated that Bolukbasi et al.’s (2016) method does not exhaustively remove gender bias. Indeed, a linear classifier trained on the modified representations can still recover the gender labels initially associated with each representation. In an attempt to improve upon Bolukbasi et al. (2016), (INLP; Ravfogel et al., 2020) introduced iterative nullspace projection (INLP), which estimates a linear gender subspace by first training a classifier to predict gender, and then projecting onto the nullspace of learned classifier’s weights; Ravfogel et al. (2020) found their method competitive.

We provide a thorough analysis of the problem of identifying and neutralizing linear concept subspaces formalized as a maximin game (Neumann and Morgenstern, 1944). We contend our formalization of linear adversarial concept removal offers us the best of two worlds. On one hand, in some cases, we can maintain the superior performance often witnessed in the adversarial paradigm. On the other, we maintain a more interpretable concept space due to our linearity assumption. In several cases, such as linear regression and Rayleigh quotient maximization, we are able to derive a closed-form solution to the maximin problem. For the case of classification loss, e.g., logistic regression, we develop a convex relaxation, **Relaxed Linear Adversarial Concept Erasure (RLACE)**, that allows us to find a good solution in practice. For concreteness, in our experiments, we follow the motivating example of removing information predictive of binary gender, and find the method effective in mitigating bias in both contextualized and static representations.

### 2. Linear Maximin Games

This section focuses on the mathematical preliminaries necessary to develop linear adversarial concept removal, a formulation that allows us to impose a structure on the adversarial intervention. Specifically, we formulate the problem as a maximin game between a predictor that aims to predict a quantity that operationalizes the concept (e.g., binary gender) and an adversary that tries to hinder the prediction by projecting the representations onto a subspace of predefined dimensionality. By constraining the adversarial intervention to a linear projection, we maintain the advantages of linear methods—interpretability and transparency—while directly optimizing an expressive objective that aims to prevent *any* linear model from predicting the concept of interest.

#### 2.1. Notation and Generalized Linear Modeling

We overview generalized linear modeling (Nelder and Wedderburn, 1972) as a framework for concept erasure.

**Notation.** Assume we are given a dataset $\mathcal{D} = \{(\mathbf{x}_n, y_n)\}_{n=1}^N$ of $N$ response–representation pairs, where the response variables $y_n$ represent the information to be neutralized, e.g., binary gender. In this article, we take $y_n \in \mathcal{Y} \stackrel{\text{def}}{=} \{0, 1\}$ and $\mathbf{x}_n \in \mathbb{R}^D$ to be a $D$-dimensional real column vector.$^2$ We use the notation $\mathbf{X} = [\mathbf{x}_1; \dots; \mathbf{x}_N]^\top \in \mathbb{R}^{N \times D}$ to denote a matrix containing the inputs, and $\mathbf{y} = [y_1, \dots, y_n]^\top \in \mathbb{R}^N$ to denote a column vector containing all the dependent variables.

**Generalized Linear Models.** Let $\Theta \subset \mathbb{R}^D$ be a compact set of parameters, and let $\theta \in \Theta$ be a real column vector of parameters. A generalized linear model consists of a linear predictor of the form $\theta^\top \mathbf{x}_n$ coupled with a **link function** $g(\cdot)$, which allows us to relate a linear prediction to the response in a more nuanced (perhaps non-linear) way. Denoting the link function’s inverse as $g^{-1}(\cdot)$, we write the prediction of a generalized linear model as $\hat{y}_n \stackrel{\text{def}}{=} g^{-1}(\theta^\top \mathbf{x}_n)$. We additionally assume a **loss function** $\ell(y_n, \hat{y}_n) \geq 0$, a non-negative function of the true response $y_n$ and a predicted response $\hat{y}_n$, that is to be minimized. By changing the link and loss functions, we obtain different problems such as linear regression, Rayleigh quotient minimization, logistic regression classification, and others. Using the above notation, this paper considers the generalized linear classification objective
$$\sum_{n=1}^N \ell(y_n, \hat{y}_n) = \sum_{n=1}^N \ell(y_n, g^{-1}(\theta^\top \mathbf{x}_n)). \tag{1}$$
We seek to minimize Eq. (1) with respect to $\theta \in \Theta$ in order to learn a good predictor of the concept labels.

#### 2.2. The Linear Bias Subspace Hypothesis

Consider a collection $\{\mathbf{x}_m\}_{m=1}^M$ of $M$ representations where each $\mathbf{x}_m \in \mathbb{R}^D$. The linear bias subspace hypothesis (Bolukbasi et al., 2016; Vargas and Cotterell, 2020) posits that there exists a linear subspace $\mathcal{B} \subseteq \mathbb{R}^D$ that (fully) contains gender bias information within representations $\{\mathbf{x}_m\}_{m=1}^M$.$^3$ It follows from this hypothesis that one strategy for the removal of gender information from representations is to i) identify the subspace $\mathcal{B}$, and ii) project the representations onto the orthogonal complement of $\mathcal{B}$, i.e., re-define every representation $\mathbf{x}_m$ in our collection as $\text{proj}_{\mathcal{B}^\perp}(\mathbf{x}_m)$. Basic linear algebra tells us that the operation $\text{proj}_{\mathcal{B}^\perp}$ may be represented by an **orthogonal projection matrix**, i.e., a symmetric matrix $\mathbf{P}$ such that $\mathbf{P}^2 = \mathbf{P}$ and $\text{proj}_{\mathcal{B}^\perp}(\mathbf{x}_m) = \mathbf{P}\mathbf{x}_m$. In this formulation, we have that $\text{null}(\mathbf{P})$ is the linear subspace $\mathcal{B}$ that encodes the bias, and $\text{range}(\mathbf{P})$ is its orthogonal complement, i.e., the non-bias subspace. Intuitively, an orthogonal projection matrix onto

---
$^2$We could have also formulated the problem where $y_n$ was also a vector. We have omitted this generalization for simplicity.
$^3$While Bolukbasi et al. (2016) and Vargas and Cotterell (2020) focused on bias mitigation, their notion of a bias subspace can be extended to any concept, and we do so here.

a subspace maps a vector to its closest neighbor in the subspace. In our case, the projection maps a vector to the closest vector in the subspace that excludes the bias subspace.

#### 2.3. Linear Maximin Games

We are now in a position to define a linear maximin game that adversarially identifies and removes a linear bias subspace. Following Ravfogel et al. (2020), we search for an orthogonal projection matrix $\mathbf{P}$ that projects onto $\mathcal{B}^\perp$, i.e., the orthogonal complement of the bias subspace $\mathcal{B}$. We define $\mathcal{P}_K$ as the set of all $D \times D$ orthogonal projection matrices of rank $D - K$. More formally, we have that $\mathbf{P} \in \mathcal{P}_K \iff \mathbf{P} = \mathbf{I}_D - \mathbf{W}^\top \mathbf{W}, \mathbf{W} \in \mathbb{R}^{K \times D}, \mathbf{W}\mathbf{W}^\top = \mathbf{I}_K$, where $\mathbf{I}_D$ denotes the $D \times D$ identity matrix and $\mathbf{I}_K$ denotes the $K \times K$ identity matrix. The matrix $\mathbf{P}$’s kernel is the $K$-dimensional subspace $\mathcal{B} = \text{range}(\mathbf{W}^\top \mathbf{W})$.

We now define the following sequential maximin game:$^4$
$$\max_{\mathbf{P} \in \mathcal{P}_K} \min_{\theta \in \Theta} \sum_{n=1}^N \ell(y_n, g^{-1}(\theta^\top \mathbf{P} \mathbf{x}_n)), \tag{2}$$
where $K$, the dimensionality of the bias subspace, is a hyperparameter. Recall that $\mathcal{P}_K$ is the set of all $D \times D$ orthogonal projection matrices of rank $D - K$. We say that pair $(\theta^\star, \mathbf{P}^\star)$ is a solution to Eq. (2) if
$$\max_{\mathbf{P} \in \mathcal{P}_K} \min_{\theta \in \Theta} \sum_{n=1}^N \ell(y_n, g^{-1}(\theta^\top \mathbf{P} \mathbf{x}_n)) = \sum_{n=1}^N \ell(y_n, g^{-1}(\theta^{\star\top} \mathbf{P}^\star \mathbf{x}_n)). \tag{3}$$
Eq. (3) can be thought of as a sequential game where the first player chooses an orthogonal matrix $\mathbf{P}$ and the second player chooses a parameter vector $\theta \in \Theta$ with knowledge of the orthogonal matrix $\mathbf{P}$. Such a solution to the sequential game, due to the assumed compactness of $\Theta$ and the compactness of $\mathcal{P}_K$, always exists in our setting. However, it is NP-hard to solve in general (Daskalakis et al., 2021). Eq. (2) is a special case of the general adversarial training algorithm (Goodfellow et al., 2014), but where the adversary is constrained to interact with the input only via an orthogonal projection matrix of rank at most $D - K$. This constraint enables us to derive principled solutions,

---
$^4$**Correction:** An earlier version of this paper erroneously claimed that Eq. (2) is a convex–concave game (Kneser, 1952; Tuy, 2004). The game is actually convex–convex. We thank David Schneider-Joseph for pointing out this mistake. If the game given in Eq. (13) had been convex–concave, the order of the min and max would not have been relevant. However, given that it is not, we have adjusted the formulation such that max precedes the min, which differs from the earlier version of this paper, but has the semantics we originally intended.

while *minimally* changing the input.$^5$

We now spell out several instantiations of common generalized linear models within the framework of adversarial generalized linear modeling: (i) linear regression, (ii) partial least squares regression, and (iii) logistic regression.

**Example (Linear Regression).** Consider the loss function $\ell(y_n, \hat{y}_n) = \|y_n - \hat{y}_n\|^2$, and the inverse link function $g^{-1}(z) = z$. Then Eq. (2) corresponds to
$$\max_{\mathbf{P} \in \mathcal{P}_K} \min_{\theta \in \Theta} \sum_{n=1}^N \|y_n - \theta^\top \mathbf{P} \mathbf{x}_n\|^2 = \max_{\mathbf{P} \in \mathcal{P}_K} \min_{\theta \in \Theta} \|\mathbf{y} - \mathbf{X}\mathbf{P}\theta\|^2_F. \tag{4}$$

**Example (Partial Least Squares Regression).** Consider the loss function $\ell(y_n, \hat{y}_n) = (y_n\hat{y}_n)^2$ and inverse link function $g^{-1}(z) = z$. Then Eq. (2) corresponds to
$$\max_{\mathbf{P} \in \mathcal{P}_K} \min_{\substack{\theta \in \Theta, \\ \|\mathbf{P}\theta\|^2=1}} \sum_{n=1}^N \|\theta^\top \mathbf{P} \mathbf{x}_n y_n\|^2 = \max_{\mathbf{P} \in \mathcal{P}_K} \min_{\substack{\theta \in \Theta, \\ \|\mathbf{P}\theta\|^2=1}} \|\theta^\top \mathbf{P} \mathbf{X}^\top \mathbf{y}\|^2_F. \tag{5}$$
where we have additionally placed a constraint on the parameter space on $\theta$ post-multiplied by the projection matrix $\mathbf{P}$, which is, strictly speaking, not part of the formalism. This means that partial least squares, strictly speaking, does not fit into our paradigm of general linear modeling.

**Example (Logistic Regression).** Consider the loss function $\ell(y_n, \hat{y}_n) = -y_n \log \hat{y}_n - (1 - y_n) \log(1 - \hat{y}_n)$, and the inverse link function $g^{-1}(z) = \frac{\exp z}{1 + \exp z}$. Then Eq. (2) corresponds to
$$\max_{\mathbf{P} \in \mathcal{P}_K} \min_{\theta \in \Theta} -\left(\sum_{n=1}^N y_n \log \frac{\exp(\theta^\top \mathbf{P} \mathbf{x}_n)}{1 + \exp(\theta^\top \mathbf{P} \mathbf{x}_n)} + (1 - y_n) \log \frac{1}{1 + \exp(\theta^\top \mathbf{P} \mathbf{x}_n)} \right). \tag{6}$$

### 3. Solving the Linear Maximin Game

At the technical level, this paper asks a simple question: For which pairs of $\ell(\cdot, \cdot)$ and $g^{-1}(\cdot)$ can we solve the objective given in § 2? We find a series of satisfying answers. In the case of linear regression (Example 1) and Rayleigh quotient optimization, e.g., partial least squares regression (Example 2), we derive a closed-form solution. And, in the case of logistic regression (Example 3), we derive a convex relaxation that allows us to solve it efficiently in practice with a gradient-based optimization method.

---
$^5$Note that an orthogonal projection of a point onto a subspace gives the closest point on that subspace in terms of $L_2$ distance.

#### 3.1. Linear Regression

We begin with the case of linear regression (Example 1). We show that there exists an optimal solution to Eq. (4) in the following proposition.

**Proposition 3.1.** *For $K = 1$, the maximin game given in Eq. (4) has a solution at $(\mathbf{P}^\star, \mathbf{0})$ where*
$$\mathbf{P}^\star = \mathbf{I}_D - \frac{\mathbf{X}^\top \mathbf{y}\mathbf{y}^\top \mathbf{X}}{\mathbf{y}^\top \mathbf{X}\mathbf{X}^\top \mathbf{y}}. \tag{7}$$
*At this point, the objective evaluates to $\|\mathbf{y}\|^2$.*

*Proof.* See App. B.1 for a proof. ■

Note that $\mathbf{P}^\star = \mathbf{I}_D - \frac{\mathbf{X}^\top \mathbf{y}\mathbf{y}^\top \mathbf{X}}{\mathbf{y}^\top \mathbf{X}\mathbf{X}^\top \mathbf{y}}$ is an orthogonal projection matrix whose null space spans the direction $\mathbf{X}^\top \mathbf{y}$, the covariance between the representations and responses. Moreover, because linear regression aims to explain covariance, it suffices to consider a one-dimensional bias subspace.

#### 3.2. Partial Least Squares

We now turn to partial least squares regression (Wold, 1973) as a representative of a special class of objectives based on the Rayleigh quotient. The loss function described in Example 2 is *not* convex due to the constraint that the parameters have unit norm. However, we can still efficiently minimize making use of basic results in linear algebra (Horn and Johnson, 2012) Other techniques in this framework include principal component analysis (Pearson, 1901) and canonical correlation analysis (Hotelling and Pabst, 1936). Recall that we omit the bias term from consideration when analyzing partial least squares.

We now state a lemma about maximin games in the form of a Rayleigh quotient. This lemma allows us to show that Example 2 can be solved exactly with an eigendecomposition.

**Lemma 3.2.** *Let $\mathbf{A} \in \mathbb{R}^{D \times D}$ be a symmetric matrix, and let $\mathbf{A} = \mathbf{V}^\top \mathbf{\Lambda} \mathbf{V}$ be its eigendecomposition. We order $\mathbf{A}$’s orthonormal eigenbasis $\{\mathbf{v}_1, \dots, \mathbf{v}_D\}$ in an ascending fashion according to the eigenvalues $\lambda_1 \leq \lambda_2 \dots \leq \lambda_D$. Then, the maximin game*
$$\max_{\mathbf{P} \in \mathcal{P}_K} \min_{\theta \in \Theta} \frac{\theta^\top \mathbf{P}^\top \mathbf{A} \mathbf{P} \theta}{\|\mathbf{P} \theta\|^2}, \tag{8}$$
*has the solution*
$$\mathbf{P}^\star = \mathbf{I}_D - \sum_{d=1}^K \mathbf{v}_d \mathbf{v}_d^\top \tag{9a}$$
$$\theta^\star = \mathbf{v}_{K+1}. \tag{9b}$$
*At this point, the objective Eq. (8) evaluates to $\lambda_{K+1}$.*

*Proof.* The proof is provided in App. B.2. ■

**Proposition 3.3.** *The partial least squares objective, given in Eq. (5), is a special case of Eq. (8) where we define*
$$\mathbf{A} = \mathbf{X}^\top \mathbf{y}\mathbf{y}^\top \mathbf{X}. \tag{10}$$
*Thus, its solution is given by (Eq. (9b), Eq. (9a)).*

*Proof.* The adversarial partial least squares objective Example 2 is scale invariant. It can be equivalently expressed as
$$\begin{aligned} \max_{\mathbf{P} \in \mathcal{P}_K} &\min_{\substack{\theta \in \Theta, \\ \|\mathbf{P} \theta\|^2=1}} \sum_{n=1}^N \|\theta^\top \mathbf{P} \mathbf{x}_n y_n\|^2 \tag{11a} \\ &= \max_{\mathbf{P} \in \mathcal{P}_K} \min_{\substack{\theta \in \Theta, \\ \|\mathbf{P} \theta\|^2=1}} \theta^\top \mathbf{P} \mathbf{X}^\top \mathbf{y}\mathbf{y}^\top \mathbf{X}\mathbf{P} \theta \tag{11b} \\ &= \max_{\mathbf{P} \in \mathcal{P}_K} \min_{\theta \in \Theta} \frac{\theta^\top \mathbf{P} \mathbf{X}^\top \mathbf{y}\mathbf{y}^\top \mathbf{X}\mathbf{P} \theta}{\|\mathbf{P} \theta\|^2}. \tag{11c} \end{aligned}$$
Now, define $\mathbf{A} = \mathbf{X}^\top \mathbf{y}\mathbf{y}^\top \mathbf{X}$ and the result follows. ■

#### 3.3. Logistic Regression

We now turn to the most practical setting where we consider logistic regression. In this case, we propose a practical convex relaxation of the problem. Note that while our exposition focuses on logistic regression, any other convex loss, e.g., hinge loss, may be substituted in.

We now describe **Relaxed Linear Adversarial Concept Erasure (RLACE)**, an effective method to solve the objective Eq. (2) for classification problems. To overcome the need to search over all orthogonal projection matrices, we propose to relax $\mathcal{P}_K$ to its **convex hull**. In the case of a rank-constrained orthogonal projection matrix, the convex hull is called the **Fantope** (Boyd and Vandenberghe, 2014):
$$\mathcal{F}_K = \{\mathbf{A} \in \mathcal{S}^D \mid \mathbf{I}_D \succeq \mathbf{A} \succeq 0, \text{tr}(\mathbf{A}) = K\}, \tag{12}$$
where $\mathcal{S}^D$ is the of all $D \times D$ real symmetric matrices, $\text{tr}$ is the trace operator, and $\succeq$ refers to the eigenvalues of the matrix $\mathbf{A}$. This yields the following relaxation of Eq. (2):
$$\max_{\mathbf{P} \in \mathcal{F}_K} \min_{\theta \in \mathbb{R}^D} \sum_{n=1}^N \ell(y_n, g^{-1}(\theta^\top \mathbf{P} \mathbf{x}_n)), \tag{13}$$
where the relaxation is highlighted in grey.

We solve the relaxed objective Eq. (13) with alternate minimization and maximization over $\theta$ and $\mathbf{P}$, respectively.$^6$ Concretely, we alternate between: (i) holding $\mathbf{P}$ fixed

---
$^6$This procedure is a form of gradient descent–ascent and, thus, is not guaranteed to converge. However, given promising empirical results, we did not explore more complex algorithms that come with a convergence guarantee.

```python
def INLP({(xn, yn)}^N_{n=1}, K, l(., .), g^-1(.)):
    # param: {(xn, yn)}^N_{n=1} training data
    # param: K bias subspace dimension
    # param: l(., .) loss function
    # param: g^-1(.) inverse link function
    P_0 = I_D
    for k in range(K):
        # ties are broken arbitrarily
        theta*_k = argmin_{theta in Theta} sum^N_{n=1} l(yn, g^-1(theta^T P_{k-1} xn))
        P_k = P_k - (theta*_k theta*_k^T) / (theta*_k^T theta*_k)
    return P_K
```
**Listing 1:** A Python-esque implementation of INLP.

and taking an unconstrained gradient step over $\theta$ towards minimizing the objective, (ii) holding $\theta$ fixed and taking an unconstrained gradient step towards maximizing the objective, and (iii) enforcing the constraint by projecting $\mathbf{P}$ onto the Fantope (Eq. (12)), using the algorithm given by Vu et al. (2013). And, see App. C for more details.

### 4. Relation to Interative Nullspace Projection

In this section, we provide an analysis of iterative nullspace projection (INLP; Ravfogel et al., 2020), a recent linear concept erasure method that attempts to mitigate bias in pre-trained presentations in a seemingly similar manner to our maximin formulation. INLP constructs an orthogonal projection matrix $\mathbf{P}$ by iteratively training a generalized linear model and projection on the complement of the subspace spanned by the parameter vector. We give complete pseudocode in Listing 1. If runs for $K$ iterations, INLP returns an orthogonal projection matrix of $D - K$. See Ravfogel et al. (2020) for more details.

Given our formulation § 2, we ask the following question: For what pairs of loss and link functions $\ell(\cdot, \cdot)$ and $g^{-1}(\cdot)$, does INLP return an exact solution to the objective given in Eq. (2)? In the case of linear regression, we give a counter-example that shows that INLP is not optimal in § 4.1. However, we are able to show that INLP optimally solves problems with a Rayleigh quotient loss in § 4.2.

#### 4.1. Linear Regression

In the following proposition, we show that INLP does not result in the optimal solution given in Proposition 3.1. While INLP will eventually damage the ability to perform linear regression on the task, it may remove an unnecessarily large number of dimensions.

**Proposition 4.1.** *INLP (Listing 1) applied to linear regression (Example 1) does not return Eq. (7), the orthogonal projection matrix found that is part of the solution found in Proposition 3.1 after $K = 1$ iterations.*

*Proof.* First, stack the training data $\{(\mathbf{x}_n, y_n)\}_{n=1}^N$ input to INLP into an $N \times D$ matrix $\mathbf{X}$ and an $N$-dimensional column vector $\mathbf{y}$. Then, INLP returns
$$\mathbf{P}_2 = \mathbf{I}_D - \frac{(\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}\mathbf{y}^\top \mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-1}}{\mathbf{y}^\top \mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-\top}(\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}}, \tag{14}$$
applying the standard result about the minimizer of linear regression. However, in general, Eq. (14) is not equal to Eq. (7), which proves the result. ■

#### 4.2. Partial Least Squares

Interestingly, we find that INLP does recover a solution when applied to partial least squares.

**Proposition 4.2.** *INLP (Listing 1) applied to partial least squares (Example 2) returns the orthogonal projection matrix found that is part of the solution found in Proposition 3.3 after $K = 1$ iterations.*

*Proof.* First, stack the training data $\{(\mathbf{x}_n, y_n)\}_{n=1}^N$ input to INLP into an $N \times D$ matrix $\mathbf{X}$ and an $N$-dimensional column vector $\mathbf{y}$. Observe that $\mathbf{X}^\top \mathbf{y}\mathbf{y}^\top \mathbf{X}$ is real symmetric and of rank 1. The rest of the proof is effectively a recapitulation of a standard argument of the spectral theorem.$^7$ Now, consider the first iteration of INLP:
$$\theta_1 \in \operatorname*{argmin}_{\theta \in \Theta} \frac{\theta^\top \mathbf{X}^\top \mathbf{y}\mathbf{y}^\top \mathbf{X} \theta}{\|\theta\|^2}. \tag{15}$$
Observe $\theta_1$ is an eigenvector associated with the largest eigenvalue (Roch, 2020). Moreover, $\theta_1$ is the only non-zero eigenvector. INLP returns $\mathbf{P}_2 = \mathbf{I}_D - \frac{\theta_1 \theta_1^\top}{\theta_1^\top \theta_1}$ which is of the form in Eq. (9a) in Lemma 3.2. This shows the result. ■

Proposition 4.2 is little more than a recapitulation of the fact that Rayleigh quotient problems can be solved by a singular value decomposition, which itself, can be performed iteratively (Wold, 1966).

#### 4.3. Logistic Regression

In § 5, we empirically demonstrate that INLP does not return a similar $\mathbf{P}$ to the orthogonal projection matrix found by INLP. And, moreover, optimizing Eq. (13) generally results in a $\mathbf{P}$ of lower rank than the one found by INLP that is just as effective at neutralizing the target concept. Indeed, in all experiments, we were able to identify a 1-dimensional subspace whose removal completely neutralized the concept using Eq. (13), while INLP requires more than one direction.

---
$^7$Note that as an implicit assumption of the proposition, we require $\Theta$ contains the eigenvectors of $\mathbf{X}^\top \mathbf{y}\mathbf{y}^\top \mathbf{X}$.

[IMAGE: Figure 2. Gender prediction accuracy after bias-removal projection against the dimensionality of the neutralized subspace for INLP and RLACE,, on GloVe representations (Experiment § 5.1). Error bars indicate standard deviation.]

### 5. Experiments

In this section, we consider mitigating gender associations in static word representations (§5.1) and increasing fairness in multi-class classification over contextualized representations (§ 5.2). Additionally, we qualitatively demonstrate the impact of RLACE on the input space by linearly removing different concepts from images (§ 5.3).$^8$

#### 5.1. Static Word Representations

We replicate the experimental design of Ravfogel et al. (2020) and Gonen and Goldberg (2019), which allows the experiment to ascertain the effectiveness of bias mitigation in static word representations. Our experiments focus on the static word representations given by the uncased version of the GloVe (Pennington et al., 2014), which Ravfogel et al.’s (2020) dataset annotates with a binary label for whether it is male- or female-biased. See App. C.2 for further details about our experimental setting. We perform 10 runs of RLACE and INLP with random initializations and report mean and standard deviations.

**Classification.** Before projection, a logistic regressor can recover the gender label of a word representation with near-perfect accuracy. However, this accuracy drastically drops after an application of RLACE for all the different values of $K$ examined. Indeed, post-projection accuracy drops to nearly chance even when we set $K = 1$; see Fig. 2. This finding suggests that there exists a 1-dimensional subspace whose removal fully hinders gender classification.$^9$

---
$^8$In App. F.2, we demonstrate that our method identifies a matrix that is close to a projection matrix, i.e., a vertex of the Fantope.
$^9$This result was later proven by Belrose et al. (2024).

| | WEAT’s d ↓ | $p$-value |
| :--- | :--- | :--- |
| **Math-art.** | | |
| Original | 1.57 | 0.000 |
| PCA | 1.46 ± 0.00 | 0.000 ± 0.000 |
| RLACE | 0.80 ± 0.01 | 0.062 ± 0.002 |
| INLP | 1.11 ± 0.10 | 0.015 ± 0.008 |
| **Professions-family.** | | |
| Original | 1.69 | 0.000 |
| PCA | 1.11 ± 0.00 | 0.005 ± 0.000 |
| RLACE | 0.79 ± 0.01 | 0.071 ± 0.003 |
| INLP | 1.11 ± 0.08 | 0.012 ± 0.007 |
| **Science-art.** | | |
| Original | 1.63 | 0.000 |
| PCA | 1.16 ± 0.00 | 0.003 ± 0.000 |
| RLACE | 0.77 ± 0.01 | 0.072 ± 0.004 |
| INLP | 1.01 ± 0.15 | 0.028 ± 0.020 |

**Table 1.** WEAT bias association results.

INLP, in contrast, does not reach majority-class accuracy even after the removal of a 20-dimensional subspace. We also examined the PCA-based approach of Bolukbasi et al. (2016), where the subspace neutralized is defined by the first $K$ principal components of the subspace spanned by the difference of the representations for gendered words.$^{10}$ However, for all $K \in \{1, \dots, 10\}$ examined, Bolukbasi et al.’s (2016) method did not significantly influence gender prediction accuracy after applying the orthogonal projection. In Ravfogel et al. (2020), it was shown that high-dimensional representation space tends to be (approximately) linearly separable into binary gender by multiple different orthogonal linear classifiers. Our results, surprisingly, show that there is a 1-dimensional subspace whose removal exhaustively removes the gender concept. Importantly, as expected given that we removed a linear subspace, non-linear classifiers are still able to recover gender: Both RBF-SVM and a ReLU MLP with 1 hidden layer of size 128 predict gender with above 90% accuracy.

**Clustering by Gender.** We now explore how RLACE influences the geometry of the representation space. Using RLACE with $K = 1$ iterations, we estimate an orthogonal projection matrix that hinders a logistic regression from classifying the representations by their associated gender. We perform principal components analysis on the GloVe representations before and after applying the orthogonal projection matrix, coloring the points by gender. As can be seen in Fig. 1, the original representation clusters by gender, however, this clustering significantly decreases post-projection. See App. E for a quantitative analysis of this effect.

---
$^{10}$We used the following pairs of words to compute such a difference: (“woman”, “man”), (“girl”, “boy”), (“she”, “he”), (“mother”, “father”), (“daughter”, “son”), (“gal”, “guy”), (“female”, “male”), (“her”, “his”), (“herself”, “himself”), (“mary”, “john”).

[IMAGE: Figure 3. Gender prediction accuracy after bias-removal projection against the dimensionality of the neutralized subspace, for INLP and RLACE, finetuned BERT representations (Experiment § 5.2).]

**Word Association Tests.** Islam et al. (2016) introduce the Word Embedding Association Test (WEAT), a measure for the association of similarity between male- and female-biased words as well as stereotypically gender-biased professions. The test examines, for example, whether a group of words denoting STEM professions is more similar, on average, to male names than to female ones. We measure the association between stereotypically male and female names and (1) career- and family-related terms, (2) art and mathematics words, and (3) artistic and scientific fields. We report the test’s statistic, WEAT’s d, and the associated $p$-values after applying a rank-1 projection in Tab. 1. RLACE is the most effective method.

**Influence on Semantic Content.** We now test to what extent RLACE damages the semantic content of the word representations. We evaluate our word representations using SimLex-999 (Hill et al., 2015), a test that measures the quality of the representation space by comparing word similarity in that space to lexical similarity, as judged by human annotators. The test is composed of pairs of words, and we calculate the Pearson correlation between the cosine similarity before and after projection, and the similarity score that human annotators gave to each pair. Similarly to Ravfogel et al. (2020), we find no significant influence on correlation with human judgments. Specifically, we observe a Pearson correlation of 0.399 with the cosine similarity over the original representations, 0.392 after applying a rank-1 orthogonal projection, estimated by RLACE, and 0.395 after 1 iteration of INLP. See App. F for the neighbors of randomly chosen words before and after RLACE.

#### 5.2. Profession Classification

We now evaluate the impact of RLACE on the fairness of a profession classifier. We consider De-Arteaga et al.’s (2019) dataset of short biographies collected from the web, annotated with both binary gender and profession. We represent each biography with the [CLS] representation in the last layer of BERT (Devlin et al., 2019), apply RLACE to erase gender from the [CLS] representation, and then evaluate the performance of the classifier, after applying the orthogonal projection to the input representations, on the downstream task of profession prediction.

We consider several profession classifiers.
*   A multiclass logistic regression profession classifier that operates on the frozen representations of pre-trained BERT (BERT-frozen).
*   A pretrained BERT model finetuned to the profession classification task (BERT-finetuned).
*   A pretrained BERT model finetuned to the profession classification task, trained adversarially for gender removal with the gradient-reversal layer method of Ganin and Lempitsky (2015) (BERT-adv). We consider (1) a linear adversary, and (2) an MLP adversary with 1 hidden layer of size 300 and ReLU activations.

We run RLACE on the representations of BERT-frozen and BERT-finetuned. We treat BERT-adv as a baseline; we report the results of 3 independent runs with random initialization. See App. D for more details.

To measure the bias in the downstream classifier, we follow De-Arteaga et al. (2019) and apply the TPR-GAP measure, a quantification of the bias in a classifier by considering the difference (GAP) in the true positive rate (TPR) between individuals with different protected attributes, e.g., race or gender. We use the notation $\text{GAP}_{y,z}^{\text{TPR}}$ to denote the TPR gap in profession $z$, e.g., NURSE for a protected group $y$, e.g., FEMALE. We also consider $\text{GAP}_z^{\text{TPR,RMS}}$, the RMS of the TPR-gap across all professions for a protected group $y$. See App. D and De-Arteaga et al. (2019) for the formal definitions. To calculate the relation between the bias the model exhibits and the bias in the data, we also compute the correlation between the TPR gap in a given profession and the percentage of women in that profession, denoted as $\sigma_{(\text{GAP}^{\text{TPR}},\% \text{WOMEN})}$.

The results are summarized in Tab. 2. RLACE effectively hinders the ability to predict gender from the representations using a rank-1 projection because INLP does not completely remove the ability to predict gender from the finetuned model even after 100 iterations (Fig. 3). Both methods have a moderate negative impact on the task of profession prediction in the finetuned model, while in the frozen model, INLP—but not RLACE—also significantly damages the task, degrading performance from 79.91% to 71.27% accuracy. Bias, as measured by $\text{GAP}_{y,z}^{\text{TPR,RMS}}$ is mitigated by both methods to a similar degree, while INLP has some ad-

| Setting | Accuracy (gender) ↓ | Accuracy (Profession) ↑ | $\text{GAP}_{\text{MALE},y}^{\text{TPR,RMS}}$ ↓ | $\sigma_{(\text{GAP}^{\text{TPR}},\% \text{WOMEN})}$ ↓ |
| :--- | :--- | :--- | :--- | :--- |
| BERT-frozen | 99.84 | 79.91 | 0.029 | 0.840 |
| BERT-frozen + RLACE (rank 1) | 52.16 ± 0.13 | 79.21 ± 0.00 | 0.020 ± 0.000 | 0.463 ± 0.005 |
| BERT-frozen + RLACE (rank 50) | 53.24 ± 0.73 | 76.73 ± 1.03 | 0.021 ± 0.001 | 0.426 ± 0.043 |
| BERT-frozen + INLP (rank 1) | 99.30 ± 0.00 | 79.58 ± 0.01 | 0.028 ± 0.000 | 0.779 ± 0.014 |
| BERT-frozen + INLP (rank 50) | 51.95 ± 0.25 | 71.27 ± 0.09 | 0.022 ± 0.000 | 0.338 ± 0.030 |
| BERT-finetuned | 85.42 ± 0.05 | 84.71 ± 0.09 | 0.026 ± 0.001 | 0.816 ± 0.005 |
| BERT-finetuned + RLACE (rank 1) | 53.61 ± 0.72 | 83.42 ± 0.10 | 0.022 ± 0.001 | 0.705 ± 0.022 |
| BERT-finetuned + RLACE (rank 100) | 53.87 ± 1.32 | 80.93 ± 1.04 | 0.024 ± 0.001 | 0.658 ± 0.030 |
| BERT-finetuned + INLP (rank 1) | 96.30 ± 0.63 | 85.41 ± 0.06 | 0.026 ± 0.000 | 0.820 ± 0.007 |
| BERT-finetuned + INLP (rank 100) | 62.76 ± 1.31 | 83.74 ± 0.09 | 0.021 ± 0.001 | 0.579 ± 0.048 |
| BERT-finetuned-adv (MLP adversary) | 98.01 ± 1.73 | 83.72 ± 1.69 | 0.024 ± 0.003 | 0.707 ± 0.079 |
| BERT-finetuned-adv (Linear adversary) | 99.40 ± 0.07 | 84.68 ± 0.16 | 0.026 ± 0.001 | 0.803 ± 0.015 |
| Majority | 53.52 | 30.0 | - | - |

**Table 2.** Results from § 5.2.

vantage in decreasing $\sigma_{(\text{GAP}^{\text{TPR}},\% \text{WOMEN})}$. For the finetuned model, the decrease in $\sigma_{(\text{GAP}^{\text{TPR}},\% \text{WOMEN})}$ is moderate.

Interestingly, the adversarially finetuned models (BERT-adv)—both with a linear and an MLP adversary—show somewhat decreased bias as measured by $\text{GAP}_{y,z}^{\text{TPR,RMS}}$, but do not hinder the ability to predict gender at all.$^{11}$ Beyond the effectiveness of RLACE for selective information removal, we conclude that the connection between the ability to predict gender from the representation, and the TPR-gap metric, is not clear-cut, and requires further study.

#### 5.3. Erasing Concepts in Image Data

Our empirical focus so far has lain on erasing concepts from textual data. We now turn to visual data, which has the advantage of allowing the experiment to be able to clearly inspect the influence of RLACE on the input. To qualitatively assess this effect, we use face images from the CelebsA dataset (Yang et al., 2015), which is composed of faces annotated with different concepts, such as SUNGLASSES and SMILE. We downscale all data to 50-pixel-by-50-pixel,

---
$^{11}$During training, the adversaries converged to close-to-random gender prediction accuracy; but this did not generalize to *new* adversaries at test time. This phenomenon was observed—albeit to a lesser degree—by Elazar and Goldberg (2018).

[IMAGE: Figure 4. Application of RLACE to the raw pixels of image data. We present images (before and after a rank-1 projection) for the concepts SMILE and GLASSES.]

grey-scale images, flatten them to 2,500-dimensional vectors, and run our method on the pixels and hinder a linear classifier from classifying, for instance, whether a person has sunglasses based on the pixels of their image.$^{12}$ We experiment with the following visual concepts: GLASSES, SMILE, MUSTACHE, BEARD, BALD and HAT.

**Results.** See Fig. 4 and App. F.1 for randomly sampled images before and after erasure. In all cases, a rank-1 orthogonal projection matrix, discovered by RLACE, is enough to remove the ability to classify the images into their concepts. Indeed, we achieve a classification accuracy of less than 1% above majority-class accuracy. We observe that the intervention changes the images by manipulating the features one would expect to be associated with the concepts of interest, e.g., erasing the concept SUNGLASSES results in an image with sunglasses superimposed on the pixels.$^{13}$ Because the intervention is constrained to be a projection, it is limited in expressivity, and it is easier to remove features than add new ones.

### 6. Related Work

Current approaches to concept erasure are predominantly based on adversarial approaches applied at training time (Ganin and Lempitsky, 2015; Edwards and Storkey, 2015; Chen et al., 2018; Xie et al., 2017; Zhang et al., 2018; Wang et al., 2021). However, such methods have proven themselves unstable and were shown by Elazar and Goldberg (2018) to not completely remove the concept present in the representations. To the authors’ knowledge, the first post-

---
$^{12}$Modern architectures for computer vision rely on deep models. We focus on linear classification in order to better understand the effect of RLACE.
$^{13}$Note that, in contrast to regular style transfer, we prevent classification of the concept. At times, (e.g., in the SUNGLASSES case), we converge to a solution that always adds the concept. However, this need not be the case.

hoc linear concept erasure method was given by Bolukbasi et al. (2016), who used PCA to identify a gender subspace spanned by a few presupposed gender directions. Building on the criticism of Gonen and Goldberg (2019), several authors have proposed alternative linear formulations (Dev and Phillips, 2019; Ravfogel et al., 2020; Dev et al., 2021; Kaiser et al., 2021). Closest to our work is Sadeghi et al. (2019), who study a different linear adversarial formulation. Their analysis is focused on the special case of linear regression, and they considered a general linear adversary, i.e., one that is not constrained to apply an orthogonal projection matrix, which is more expressive.

Beyond bias mitigation, concept subspaces have been used as an interpretability tool (Kim et al., 2018) in the causal analysis of neural networks (Elazar et al., 2021; Ravfogel et al., 2021) and in the study of the geometry of neural networks’ representations (Celikkanat et al., 2020; Gonen et al., 2020; Hernandez and Andreas, 2021). Moreover, our linear concept erasure objective is different than past work, which is based on subspace clustering (Parsons et al., 2004), because we focus on hindering the ability of a linear classifier to predict the concept from the representations; we do not assume that the data lives in a linear subspace.

### 7. Conclusion

We have formulated the task of concept erasure from the representation space as a constrained version of a general maximin game. In the constrained game, the adversary is limited to a fixed-rank orthogonal projection. This constrained formulation allows us to derive closed-form solutions to this problem for certain objectives and to devise a more general convex relaxation that works well in practice for others. We empirically show that the relaxed optimization recovers a 1-dimensional subspace whose removal is enough to mitigate linearly encoded concepts. As a downside, the method proposed here only protects against linear classifiers. Effectively removing non-linear information while maintaining the advantages of the constrained, linear approach remains an open challenge.

### References

George B. Arfken, Hans J. Weber, and Frank E. Harris. 2011. *Mathematical Methods for Physicists: A Comprehensive Guide*. Academic Press.

Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Edward Raff, and Stella Biderman. 2024. LEACE: Perfect linear concept erasure in closed form. In *Proceedings of the 37th International Conference on Neural Information Processing Systems*.

Tolga Bolukbasi, Kai-Wei Chang, James Y. Zou, Venkatesh Saligrama, and Adam T. Kalai. 2016. Man is to computer programmer as woman is to homemaker? Debiasing word embeddings. *Advances in Neural Information Processing Systems*, 29:4349–4357.

Stephen P. Boyd and Lieven Vandenberghe. 2014. *Convex Optimization*. Cambridge University Press.

Hande Celikkanat, Sami Virpioja, Jorg Tiedemann, and Marianna Apidianaki. 2020. Controlling the imprint of passivization and negation in contextualized representations. In *Proceedings of the Third BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP*, pages 136–148.

Xilun Chen, Yu Sun, Ben Athiwaratkun, Claire Cardie, and Kilian Weinberger. 2018. Adversarial deep averaging networks for cross-lingual sentiment classification. *Transactions of the Association for Computational Linguistics*, 6:557–570.

Constantinos Daskalakis, Stratis Skoulakis, and Manolis Zampetakis. 2021. The complexity of constrained minimax optimization. Association for Computing Machinery.

Maria De-Arteaga, Alexey Romanov, Hanna M. Wallach, Jennifer T. Chayes, Christian Borgs, Alexandra Chouldechova, Sahin Cem Geyik, Krishnaram Kenthapadi, and Adam Tauman Kalai. 2019. Bias in bios: A case study of semantic representation bias in a high-stakes setting. *CoRR*, abs/1901.09451.

Sunipa Dev, Tao Li, Jeff M Phillips, and Vivek Srikumar. 2021. OSCaR: Orthogonal subspace correction and rectification of biases in word embeddings. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 5034–5050. Association for Computational Linguistics.

Sunipa Dev and Jeff Phillips. 2019. Attenuating bias in word vectors. In *The 22nd International Conference on Artificial Intelligence and Statistics*, pages 879–887. PMLR.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 4171–4186. Association for Computational Linguistics.

Harrison Edwards and Amos Storkey. 2015. Censoring representations with an adversary. *arXiv preprint arXiv:1511.05897*.

Yanai Elazar and Yoav Goldberg. 2018. Adversarial removal of demographic attributes from text data. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 11–21. Association for Computational Linguistics.

Yanai Elazar, Shauli Ravfogel, Alon Jacovi, and Yoav Goldberg. 2021. Amnesic probing: Behavioral explanation with amnesic counterfactuals. *Transactions of the Association for Computational Linguistics*, 9:160–175.

Yaroslav Ganin and Victor Lempitsky. 2015. Unsupervised domain adaptation by backpropagation. In *International Conference on Machine Learning*, pages 1180–1189. PMLR.

Hila Gonen and Yoav Goldberg. 2019. Lipstick on a pig: Debiasing methods cover up systematic gender biases in word embeddings but do not remove them. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 609–614. Association for Computational Linguistics.

Hila Gonen, Shauli Ravfogel, Yanai Elazar, and Yoav Goldberg. 2020. It’s not Greek to mBERT: Inducing word-level translations from multilingual BERT. In *Proceedings of the Third BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP*, pages 45–56. Association for Computational Linguistics.

Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Joshua Bengio. 2014. Generative adversarial nets. In *Proceedings of the 27th International Conference on Neural Information Processing Systems*, page 2672–2680. MIT Press.

Moritz Hardt, Eric Price, and Nati Srebro. 2016. Equality of opportunity in supervised learning. In *Advances in Neural Information Processing Systems 29: Annual Conference on Neural Information Processing*, pages 3315–3323.

Evan Hernandez and Jacob Andreas. 2021. The low-dimensional linear geometry of contextualized word representations. In *Proceedings of the 25th Conference on Computational Natural Language Learning*, pages 82–93. Association for Computational Linguistics.

Felix Hill, Roi Reichart, and Anna Korhonen. 2015. SimLex-999: Evaluating semantic models with (genuine) similarity estimation. *Comput. Linguistics*, 41(4):665–695.

Roger A. Horn and Charles R. Johnson. 2012. *Matrix Analysis*. Cambridge University Press.

Harold Hotelling and Margaret Richards Pabst. 1936. Rank correlation and tests of significance involving no assumption of normality. *The Annals of Mathematical Statistics*, 7(1):29–43.

Jeremy Howard and Sebastian Ruder. 2018. Universal language model fine-tuning for text classification. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 328–339. Association for Computational Linguistics.

Aylin Caliskan Islam, Joanna J. Bryson, and Arvind Narayanan. 2016. Semantics derived automatically from language corpora necessarily contain human biases. *CoRR*, abs/1608.07187.

Jens Kaiser, Sinan Kurtyigit, Serge Kotchourko, and Dominik Schlechtweg. 2021. Effects of pre- and post-processing on type-based embeddings in lexical semantic change detection. In *Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume*, pages 125–137. Association for Computational Linguistics.

Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, and Rory Sayres. 2018. Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (TCAV). In *Proceedings of the 35th International Conference on Machine Learning*, volume 80 of *Proceedings of Machine Learning Research*, pages 2668–2677.

Hellmuth Kneser. 1952. Sur un théorème fondamental de la théorie des jeux. *Comptes Rendus Hebdomadaires des Séances de l’Académie des Sciences*, 234(25):2418–2420.

Oren Melamud, Jacob Goldberger, and Ido Dagan. 2016. Context2vec: Learning generic context embedding with bidirectional LSTM. In *Proceedings of The 20th SIGNLL Conference on Computational Natural Language Learning*, pages 51–61. Association for Computational Linguistics.

J. A. Nelder and Robert W. M. Wedderburn. 1972. Generalized linear models. *Journal of the Royal Statistical Society: Series A (General)*, 135(3):370–384.

John von Neumann and Oskar Morgenstern. 1944. *Theory of Games and Economic Behavior*. Princeton University Press.

Lance Parsons, Ehtesham Haque, and Huan Liu. 2004. Subspace clustering for high dimensional data: a review. *ACM SIGKDD Explorations Newsletter*, 6(1):90–105.

Karl Pearson. 1901. On lines and planes of closest fit to systems of points in space. *The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science*, 2(11):559–572.

F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. 2011. Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12:2825–2830.

Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global vectors for word representation. In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing*, pages 1532–1543.

Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. 2018. Deep contextualized word representations. In *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)*, pages 2227–2237. Association for Computational Linguistics.

Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners.

Shauli Ravfogel, Yanai Elazar, Hila Gonen, Michael Twiton, and Yoav Goldberg. 2020. Null it out: Guarding protected attributes by iterative nullspace projection. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 7237–7256. Association for Computational Linguistics.

Shauli Ravfogel, Grusha Prasad, Tal Linzen, and Yoav Goldberg. 2021. Counterfactual interventions reveal the causal effect of relative clause representations on agreement prediction. In *Proceedings of the 25th Conference on Computational Natural Language Learning*, pages 194–209. Association for Computational Linguistics.

Sebastien Roch. 2020. *Spectral theorem*. University of Wisconsin.

Andrew Rosenberg and Julia Hirschberg. 2007. V-measure: A conditional entropy-based external cluster evaluation measure. In *Proceedings of the 2007 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning*, pages 410–420. Association for Computational Linguistics.

Bashir Sadeghi, Runyi Yu, and Vishnu Boddeti. 2019. On the global optima of kernelized adversarial representation learning. In *2019 IEEE/CVF International Conference on Computer Vision*, pages 7970–7978. IEEE.

Hoang Tuy. 2004. Minimax theorems revisited. *Acta Mathematica Vietnamica*, 29(3):217–229.

Francisco Vargas and Ryan Cotterell. 2020. Exploring the linear subspace hypothesis in gender bias mitigation. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing*, pages 2902–2913. Association for Computational Linguistics.

Vincent Q. Vu, Juhee Cho, Jing Lei, and Karl Rohe. 2013. Fantope projection and selection: A near-optimal convex relaxation of sparse PCA. In *Advances in Neural Information Processing Systems*, pages 2670–2678.

Liwen Wang, Yuanmeng Yan, Keqing He, Yanan Wu, and Weiran Xu. 2021. Dynamically disentangling social bias from task-oriented representations with adversarial attack. In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 3740–3750. Association for Computational Linguistics.

Herman Wold. 1966. Estimation of principal components and related models by iterative least squares. *Multivariate Analysis*, pages 391–420.

Herman Wold. 1973. Nonlinear iterative partial least squares (NIPALS) modelling: Some current developments. In *Multivariate Analysis–III*, pages 383–407.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. 2020. Transformers: State-of-the-art natural language processing. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pages 38–45. Association for Computational Linguistics.

Qizhe Xie, Zihang Dai, Yulun Du, Eduard Hovy, and Graham Neubig. 2017. Controllable invariance through adversarial feature learning. In *Proceedings of the 31st International Conference on Neural Information Processing Systems*, page 585–596.

Shuo Yang, Ping Luo, Chen-Change Loy, and Xiaoou Tang. 2015. From facial parts responses to face detection: A deep learning approach. In *Proceedings of the 2015 IEEE International Conference on Computer Vision (ICCV)*, page 3676–3684. IEEE Computer Society.

Brian Hu Zhang, Blake Lemoine, and Margaret Mitchell. 2018. Mitigating unwanted biases with adversarial learning. In *Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society*, page 335–340. Association for Computing Machinery.

---

### A. Ethical Considerations

The experiments discussed in this paper revolved around the removal of binary gender information from a representation of text. First, we would like to acknowledge that gender is a non-binary concept. Beyond this point, however, erasure of binary gender has real-world applications—in particular, as it relates to fairness in machine learning. However, we would like to caution the readers to take the results with a grain of salt and be careful when deploying RLACE in a practical setting. Despite our formal analysis, care should be taken to measure the effectiveness of the approach in the context in which RLACE is to be deployed, considering, among other things, the exact data to be used, the exact fairness metrics under consideration, and the overall application. We further urge practitioners not to regard this method as a solution to the problem of bias in representation, but rather as a preliminary research effort towards mitigating certain aspects of the problem. Unavoidably, we only consider a limited set of datasets in our experiments, and they may not reflect all the subtle and implicit ways in which gender bias may manifest itself. As such, it is likely that different forms of bias still exist in the representations after applying RLACE.

### B. Proofs

#### B.1. Proof of Proposition 3.1

**Proposition 3.1.** *For $K = 1$, the maximin game given in Eq. (4) has a solution at $(\mathbf{P}^\star, \mathbf{0})$ where*
$$\mathbf{P}^\star = \mathbf{I}_D - \frac{\mathbf{X}^\top \mathbf{y}\mathbf{y}^\top \mathbf{X}}{\mathbf{y}^\top \mathbf{X}\mathbf{X}^\top \mathbf{y}}. \tag{7}$$
*At this point, the objective evaluates to $\|\mathbf{y}\|^2$.*

*Proof.* We lower- and upper-bound the maximin game to prove the result.

**Lower Bound.** First, note that
$$\min_{\theta \in \Theta} \|\mathbf{y} - \mathbf{X}\mathbf{P}_0\theta\|^2 \leq \max_{\mathbf{P} \in \mathcal{P}_K} \min_{\theta \in \Theta} \|\mathbf{y} - \mathbf{X}\mathbf{P}\theta\|^2, \tag{16}$$
$\forall \mathbf{P}_0 \in \mathcal{P}_K$. Choose
$$\mathbf{P}_0 = \mathbf{I}_D - \frac{\mathbf{X}^\top \mathbf{y}\mathbf{y}^\top \mathbf{X}}{\mathbf{y}^\top \mathbf{X}\mathbf{X}^\top \mathbf{y}}. \tag{17}$$
Then, we seek to solve the following minimization problem
$$\min_{\theta \in \Theta} \|\mathbf{y} - \mathbf{X}\mathbf{P}_0\theta\|^2. \tag{18}$$
Because Eq. (18) is convex, we only check the first-order optimality condition
$$\begin{aligned} \frac{\partial}{\partial \theta} \|\mathbf{y} - \mathbf{X}\mathbf{P}_0\theta\|^2 &= -2\mathbf{P}_0\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\mathbf{P}_0\theta) \tag{19a} \\ &= -2 \underbrace{\mathbf{P}_0^\top \mathbf{X}^\top \mathbf{y}}_{=0} + 2\mathbf{P}_0^\top \mathbf{X}^\top \mathbf{X}\mathbf{P}_0\theta \tag{19b} \\ &= 0, \tag{19c} \end{aligned}$$
which implies we have
$$0 = \mathbf{P}_0^\top \mathbf{X}^\top \mathbf{X}\mathbf{P}_0\theta = (\mathbf{P}_0\mathbf{X})^2\theta. \tag{20}$$
Thus, we have a solution at $\theta^\star = 0$.$^{14}$ Plugging $\mathbf{P}_0$ back into the objective, we arrive at
$$\|\mathbf{y} - \mathbf{X}\mathbf{P}_0\mathbf{0}\|^2 = \|\mathbf{y}\|. \tag{21}$$
Thus, we achieve the following lower bound
$$\|\mathbf{y}\| \leq \max_{\mathbf{P} \in \mathcal{P}_K} \min_{\theta \in \Theta} \|\mathbf{y} - \mathbf{X}\mathbf{P}\theta\|^2. \tag{22}$$

---
$^{14}$Note that there are other solutions, e.g., we could take $\theta = \mathbf{X}^\top \mathbf{y}$. However, because the objective is convex, $\mathbf{0}$ is a global minimum.

**Upper bound.** Moreover, because the second player, who seeks to minimize the objective, can always choose $\theta^\star = 0$, we have the following upper bound
$$\max_{\mathbf{P} \in \mathcal{P}_K} \min_{\theta \in \Theta} \|\mathbf{y} - \mathbf{X}\mathbf{P}\theta\|^2 \leq \|\mathbf{y}\|^2. \tag{23}$$

**Putting it together.** Putting the bounds together, we have that $\left(\mathbf{I}_D - \frac{\mathbf{X}^\top \mathbf{y}\mathbf{y}^\top \mathbf{X}}{\mathbf{y}^\top \mathbf{X}\mathbf{X}^\top \mathbf{y}}, \mathbf{0}\right)$ is an solution, as desired. Moreover, we see the value of the objective at $\left(\mathbf{I}_D - \frac{\mathbf{X}^\top \mathbf{y}\mathbf{y}^\top \mathbf{X}}{\mathbf{y}^\top \mathbf{X}\mathbf{X}^\top \mathbf{y}}, \mathbf{0}\right)$ is $\|\mathbf{y}\|^2$. ■

#### B.2. Proof of Lemma 3.2

**Lemma 3.2.** *Let $\mathbf{A} \in \mathbb{R}^{D \times D}$ be a symmetric matrix, and let $\mathbf{A} = \mathbf{V}^\top \mathbf{\Lambda} \mathbf{V}$ be its eigendecomposition. We order $\mathbf{A}$’s orthonormal eigenbasis $\{\mathbf{v}_1, \dots, \mathbf{v}_D\}$ in an ascending fashion according to the eigenvalues $\lambda_1 \leq \lambda_2 \dots \leq \lambda_D$. Then, the maximin game*
$$\max_{\mathbf{P} \in \mathcal{P}_K} \min_{\theta \in \Theta} \frac{\theta^\top \mathbf{P}^\top \mathbf{A} \mathbf{P} \theta}{\|\mathbf{P} \theta\|^2}, \tag{8}$$
*has the solution*
$$\mathbf{P}^\star = \mathbf{I}_D - \sum_{d=1}^K \mathbf{v}_d \mathbf{v}_d^\top \tag{9a}$$
$$\theta^\star = \mathbf{v}_{K+1}. \tag{9b}$$
*At this point, the objective Eq. (8) evaluates to $\lambda_{K+1}$.*

*Proof.* First, we manipulate the objective as follows
$$\begin{aligned} \frac{(\mathbf{P} \theta)^\top \mathbf{A}(\mathbf{P} \theta)}{(\mathbf{P} \theta)^\top(\mathbf{P} \theta)} &= \frac{(\mathbf{P} \theta)^\top \mathbf{V}^\top \mathbf{\Lambda} \mathbf{V} (\mathbf{P} \theta)}{(\mathbf{P} \theta)^\top(\mathbf{P} \theta)} \tag{24a} \\ &= \frac{(\mathbf{P} \theta)^\top \mathbf{V}^\top \mathbf{\Lambda} \mathbf{V} (\mathbf{P} \theta)}{(\mathbf{P} \theta)^\top \mathbf{V}^\top \mathbf{V} (\mathbf{P} \theta)}. \tag{24b} \end{aligned}$$
where we enforce the constraint that $\|\mathbf{P} \theta\|^2 = 1$. Note that we are ensured the existence of an eigendecomposition of $\mathbf{A}$ because we assume $\mathbf{A}$ symmetric.

**Lower Bound.** To construct a lower bound, choose
$$\mathbf{P}_0 = \mathbf{I}_D - \sum_{k=1}^K \mathbf{v}_d \mathbf{v}_d^\top. \tag{25}$$
Then, consider
$$\min_{\theta \in \Theta} \frac{\theta^\top \mathbf{P}_0^\top \mathbf{V}^\top \mathbf{\Lambda} \mathbf{V} \mathbf{P}_0 \theta}{\|\mathbf{P}_0 \theta\|^2} \leq \max_{\mathbf{P} \in \mathcal{P}_K} \min_{\theta \in \Theta} \frac{\theta^\top \mathbf{P}^\top \mathbf{V}^\top \mathbf{\Lambda} \mathbf{V} \mathbf{P} \theta}{\|\mathbf{P} \theta\|^2}. \tag{26}$$
The left-hand side of Eq. (26), however, is minimized with $\theta = \mathbf{v}_{K+1}$ and achieves a value of $\lambda_{K+1}$. Thus, we have
$$\lambda_{K+1} \leq \max_{\mathbf{P} \in \mathcal{P}_K} \min_{\theta \in \Theta} \frac{\theta^\top \mathbf{P}^\top \mathbf{V}^\top \mathbf{\Lambda} \mathbf{V} \mathbf{P} \theta}{\|\mathbf{P} \theta\|^2}. \tag{27}$$

**Upper Bound.** We next argue for an upper bound on the objective. Choose
$$\max_{\mathbf{P} \in \mathcal{P}_K} \min_{\theta \in \Theta} \frac{(\mathbf{P} \theta)^\top \mathbf{V}^\top \mathbf{\Lambda} \mathbf{V} (\mathbf{P} \theta)}{(\mathbf{P} \theta)^\top \mathbf{V}^\top \mathbf{V} (\mathbf{P} \theta)} \leq \max_{\mathbf{P} \in \mathcal{P}_K} \min_{\theta \in \{e_1, \dots, e_{D-k+1}\}} \frac{(\mathbf{P} \theta)^\top \mathbf{V}^\top \mathbf{\Lambda} \mathbf{V} (\mathbf{P} \theta)}{(\mathbf{P} \theta)^\top \mathbf{V}^\top \mathbf{V} (\mathbf{P} \theta)}. \tag{28}$$
We have now reduced the inner continuous minimization problem to a discrete one with $D - K + 1$ choices. We can consider each of these choices individually. Inspection reveals that
$$\max_{\mathbf{P} \in \mathcal{P}_K} \frac{(\mathbf{P} \mathbf{e}_j)^\top \mathbf{V}^\top \mathbf{\Lambda} \mathbf{V} (\mathbf{P} \mathbf{e}_j)}{(\mathbf{P} \mathbf{e}_j)^\top \mathbf{V}^\top \mathbf{V} (\mathbf{P} \mathbf{e}_j)} = \begin{cases} \lambda_j & \text{if } \mathbf{e}_j \in \text{range}(\mathbf{P}) \\ 0 & \text{otherwise}. \end{cases} \tag{29}$$

Because we can choose $\mathbf{P}$ to have a range of dimension at most $D - K$, it follows that we should choose it to span $\{\mathbf{v}_1, \dots, \mathbf{v}_K\}$, the eigenvectors that correspond to the $K$ smallest eigenvalues. This implies that the right-hand side of Eq. (28) has the solution
$$\mathbf{P}^\star = \mathbf{I}_D - \sum_{k=1}^K \mathbf{v}_k \mathbf{v}_k^\top \tag{30a}$$
$$\theta^\star = \mathbf{v}_{K+1}, \tag{30b}$$
and, thus, we arrive at the upper-bound
$$\max_{\mathbf{P} \in \mathcal{P}_K} \min_{\theta \in \Theta} \frac{(\mathbf{P} \theta)^\top \mathbf{V}^\top \mathbf{\Lambda} \mathbf{V} (\mathbf{P} \theta)}{(\mathbf{P} \theta)^\top \mathbf{V}^\top \mathbf{V} (\mathbf{P} \theta)} \leq \lambda_{K+1}. \tag{31}$$

**Putting it Together.** Given that we have upper and lower bounded the problem with $\mathbf{v}_{K+1}$, we conclude that $(\mathbf{P}^\star, \theta^\star)$ is a solution with
$$\mathbf{P}^\star = \mathbf{I}_D - \sum_{k=1}^K \mathbf{v}_k \mathbf{v}_K^\top \tag{32a}$$
$$\theta^\star = \mathbf{v}_{K+1}, \tag{32b}$$
and, moreover, that the value of the objective if $\lambda_{K+1}$, as desired. ■

### C. Optimizing the Relaxed Objective

In this appendix, we describe the optimization of the relaxed objective given in Eq. (13).

#### C.1. Alternate Optimization with Projected Gradient Descent

To optimize the relaxed objective Eq. (13), we alternate minimization and maximization sticks over $\theta$ and $\mathbf{P}$, respectively. On the one hand, we update $\theta$ using descent:
$$\theta_{t+1} \leftarrow \theta_t - \alpha_t \nabla_\theta \sum_{n=1}^N \ell(y_n, g^{-1}(\theta^\top \mathbf{P} \mathbf{x}_n)). \tag{33}$$
On the other hand, we update $\mathbf{P}$ with projected gradient ascent:
$$\mathbf{P}_{t+1} \leftarrow \Pi_{\mathcal{F}_K} \left( \mathbf{P}_t + \alpha_t \nabla_{\mathbf{P}} \sum_{n=1}^N \ell(y_n, g^{-1}(\theta^\top \mathbf{P} \mathbf{x}_n)) \right), \tag{34}$$
where $\alpha_t$ is the learning rate, and $\Pi_{\mathcal{F}_K}$ is the projection operation onto the Fantope, given in Vu et al. (2013). The following lemma describes how to calculate that projection:

**Lemma C.1 (Vu et al. (2013)).** *Let $\mathcal{F}_K$ be the $K$-dimensional fantope; see Eq. (12), and let $\mathbf{P} = \sum_{d=1}^D \lambda_d \mathbf{v}_d \mathbf{v}_d^\top$ be the eigendecomposition of $\mathbf{P}$ where $\lambda_d$ is $\mathbf{P}$’s $d^{\text{th}}$ eigenvalue and $\mathbf{v}_d$ is its corresponding eigenvector. The projection of $\mathbf{P}$ onto the fantope is given by $\Pi_{\mathcal{F}_K}(\mathbf{P}) = \sum_{d=1}^D \lambda_d^+(\gamma) \cdot \mathbf{v}_d \mathbf{v}_d^\top$, where $\lambda_d^+(\gamma) = \min(\max(\lambda_d - \gamma, 0), 1)$ and $\gamma$ satisfies the equation $\sum_{d=1}^D \lambda_d^+(\gamma) = k$.*

Lemma C.1 specifies that finding the projection entails performing an eigendecomposition of $\mathbf{P}$ and finding $\gamma$ that satisfies a set of monotone, piecewise linear equations. Becauase we can easily find $\gamma$ where $\sum_{d=1}^D \lambda_d^+(\gamma) > K$ and $\gamma$ where $\sum_{d=1}^D \lambda_d^+(\gamma) < K$, we can solve the system of equations using bisection.

Upon termination of the optimization process, we return the closest vertex of the Fantope. To do so, we perform a spectral decomposition of $\mathbf{P}$ and return the orthogonal projection matrix $\mathbf{P}_{\text{final}}$ whose range spans the first $D - K$ eigenvectors. The process is discussed in more detail in App. C.

#### C.2. Experimental Setup

In this appendix, we describe the experimental setting for those experiments involving static word representations § 5.1. We conduct experiments on 300-dimensional uncased GloVe vectors. Following (Ravfogel et al., 2020), to approximate the gender annotation for the vocabulary, we project all vectors on the $\vec{he} - \vec{she}$ direction, and take the 7,500 most male-biased and female-biased words.$^{15}$ Note that unlike (Bolukbasi et al., 2016), we use the $\vec{he} - \vec{she}$ direction only to induce approximate gender labels to train RLACE.

We use the same train–dev–test split as Ravfogel et al. (2020), but discard the gender-neutral words, i.e., we cast the problem as a binary classification. We end up with a training set, evaluation set, and test set of sizes 7,350, 3,150, and 4,500, respectively. We run this procedure for 50,000 iterations with the cross-entropy loss, alternating between an update to the adversary and to the classifier after each iteration.

The inner optimization problem described in the Fantope projection operation is solved with the bisection method (Arfken et al., 2011). We train with a simple SGD, with a learning rate of 0.005, chosen by experimenting with the development set. We use a batch size of 128. After each 1000 batches, we freeze the adversary, train the classifier to convergence, and record its loss. Finally, we return the adversary which yields the *highest* classification loss. At test time, we evaluate the ability to predict gender using logistic regression classifiers trained in scikit-learn (Pedregosa et al., 2011) For the dimensionality of the neutralized subspace, we experiment with the values $K = 1 \dots 20$ for INLP and RLACE. We perform 10 runs and report mean ± standard deviation.

### D. Experimental Setting: Deep Classification

In this appendix, we describe the experimental setting for the deep classification experiments § 3.3. We use the same train–dev–test split of the biographies dataset considered by Ravfogel et al. (2020), resulting in training, evaluation, and test sets of sizes 255,710, 39,369, and 98,344, respectively. We run a simple stochastic gradient descent optimization procedure, with a learning rate of 0.005 and a weight decay of $1\text{e}^{-4}$, chosen by experimenting with the development set. We consider a batch size of 256, and, again, choose the adversary which yields the highest classification loss. As the dimensionality of the bias subspace, we run both RLACE and INLP with $K = 1 \dots 50$ on BERT-frozen and $K = 1 \dots 100$ on BERT-finetuned. We perform 3 runs of the entire experimental pipeline (classifier training, applying INLP and RLACE) and report mean ± the standard deviation.

**Classifier training.** We experiment with several profession classifiers, as detailed in § 3.3. For BERT-frozen, we use the HuggingFace implementation (Wolf et al., 2020). For BERT-finetuned, we finetune the pre-trained BERT on the profession classification task, using an SGD optimizer with a learning rate of 0.0005, weight decay of $1\text{e}^{-6}$ and momentum of 0.9. We train for 70, 000 batches of size 10 and choose the model that achieved the lowest loss on the development set. For BERT-adv, we perform the same training procedure but add an additional classification head which is trained to predict gender, and whose gradient is reversed (Ganin and Lempitsky, 2015). This procedure creates an encoder that generates hidden representations which are predictive of the professions but are not predictive of gender. The adversary always converged to a low gender classification accuracy (below 55%), which is commonly interpreted as the success of the removal process.

**Fairness Measure: TPR-GAP.** We informally describe the fairness measures used in § 5.2. The TPR-GAP is tightly related to the notion of fairness by equal opportunity (Hardt et al., 2016): a fair binary classifier is expected to show similar success in predicting the task label $\mathbf{y}$ for two populations when conditioned on the true class. We refer the reader to Hardt et al. (2016) for more information.

### E. V -Measure

To quantify the effect of our intervention on the GloVe representation space in § 5.1, we perform $k$-means clustering with different values of $k$, and use $V$-measure (Rosenberg and Hirschberg, 2007) to quantify the association between cluster identity and the gender labels, after a projection that removes rank-1 subspace. The results are presented in Fig. 5. $V$-measure for the original representations is 1.0, indicating strong alignment between cluster identity and gender label. The

---
$^{15}$Note that $\vec{he}$ is the static representation for the word “he” and $\vec{she}$ is the static representation for the word “she”.

[IMAGE: Figure 5. V-measure between gender labels and cluster identity, for different numbers of clusters on the x-axis (lower values are better). Error bars are standard deviations from 10 random runs.]

score drastically drops after a rank-1 relaxed projection, while INLP projection and the PCA-based method of (Bolukbasi et al., 2016) have a smaller effect.

### F. Influence on Neighbors in Embedding Space

In § 5.1, we showed that the SimLex999 evaluation does not indicate that our intervention damages the general semantics encoded in the GloVe embedding space. To qualitatively demonstrate this, we provide in Tab. 3 the closest-neighbors to 15 randomly sampled words from the vocabulary, before and after our intervention.

#### F.1. Additional results on the CelebsA dataset

We present here randomly sampled outputs for the 6 concepts we experimented with the following concepts: GLASSES, SMILE, MUSTACHE, BEARD, BALD, and HAT; see the experimental designs in § 5.3.

[IMAGE: Figure 6. GLASSES]

#### F.2. Relaxation Quality

We now investigate to what extent the optimization of the relaxed objective Eq. (13) results in a matrix $\mathbf{P}$ that is a valid rank-$K$ orthogonal projection matrix. Recall that orthogonal projection matrix have eigenvalues that are either 0 or 1, and, accordingly, their sum is the rank of the matrix. In Fig. 12, we present the eigenvalues spectrum of $\mathbf{P}$ after optimization with $K = 6$ on the static word representation dataset (§ 5.1). We find that the top 6 eigenvalues are indeed close to 1, and the rest are close to 0—suggesting the approximation is tight: the resulting matrix is close to a valid rank-$K$ orthogonal projection matrix.

| Word | Neighbors before | Neighbors after |
| :--- | :--- | :--- |
| “ocean” | “waters”, “atlantic”, “sea” | “waters”, “atlantic”, “sea” |
| “museum” | “heritage”, “art”, “exhibition” | “heritage”, “art”, “exhibition” |
| “lol” | “:p”, “:d”, “haha” | “:p”, “:d”, “haha” |
| “twenty” | “five”, “ten”, “hundred” | “five”, “ten”, “hundred” |
| “sample” | “free”, “test”, “samples” | “example”, “test”, “samples” |
| “storm” | “weather”, “wind”, “rain” | “weather”, “wind”, “rain” |
| “state” | “ohio”, “government”, “states” | “ohio”, “california” “states” |
| “electrical” | “electricity”, “mechanical”, “electric” | “electricity”, “mechanical”, “electric” |
| “papers” | “essay”, “essays”, “paper” | “essay”, “essays”, “paper” |
| “contributions” | “participation”, “contribute”, “contribution” | “participation”, “contribute”, “contribution” |
| “lab” | “research”, “science”, “laboratory” | “research”, “science”, “laboratory” |
| “joke” | “laugh”, “stupid”, “funny” | “laugh”, “stupid”, “funny” |
| “hear” | “tell”, “listen”, “heard” | “tell”, “listen”, “heard” |
| “detail” | “description”, “detailed”, “details” | “description”, “detailed”, “details” |
| “extreme” | “hardcore”, “severe”, “intense” | “hardcore”, “severe”, “intense” |

**Table 3.** Neighbors to random words in GloVe space before and other rank-1 RLACE projection.

[IMAGE: Figure 7. SMILE]

[IMAGE: Figure 8. MUSTACHE]

[IMAGE: Figure 9. BEARD]

[IMAGE: Figure 10. BALD]

[IMAGE: Figure 11. HAT]

[IMAGE: Figure 12. The spectrum of the approximate solution to the relaxed optimization problem Eq. (13) with K = 7.]