---
author: "Vien Vuong"
title: "Notes on Probabilistic Machine Learning [Ongoing]"
date: "2022-09-14"
description: "Chapter notes on Kevin Murphy's Probabilistic Machine Learning: An Introduction (2022), one of the most comprehensive textbooks for Machine Learning."
tags: ["notes", "ml"]
comments: false
socialShare: false
toc: true
math: true
cover:
  src: /probml-notes/probml-cover-cropped.jpg
  alt: Probabilistic Machine Learning Cover
---

Chapter notes on Kevin Murphy's Probabilistic Machine Learning: An Introduction (2022). Ongoing work so expect constant updates as I continue to read/re-read the chapters.

Some sections taken from course notes for CS 446 - Machine Learning and ECE 544 - Pattern Recognition at UIUC. Many thanks to Prof. Matus Telgarsky and Prof. Alexander Schwing for your wonderful lectures.

## Chapter 1: Introduction

- Definition

  - A computer program is said to learn
    from experience E
    with respect to some class of tasks T,
    and performance measure P,
  - If
    its performance at tasks in T,
    as measured by P,
    improves with experience E.

**Supervised Learning**

- Task is to learn a mapping function from inputs to outputs
- Inputs are vectors called features, covariates, predictors
- Output is called as label, target, response

- Classification

  - Output is a set of unordered, mutually exclusive labels
  - Goal is to design a model with minimum misclassification rate
  - $L(\theta) = {1 \over N} \sum I\{y \ne f(x, \theta)\}$
  - Empirical Risk is the average loss on training set
  - Model fitting minimizes empirical risk (ERM)
  - Overall goal is generalization

- Uncertainty

  - Model cannot map inputs to outputs with 100% certainty
  - Model Uncertainty due to lack of knowledge between input and output
  - Data Uncertainty due to stochasticity in labels
  - Good model assigns high probability to true output for each input
  - Intuition for minimizing NLL (negative log-likelihood)
  - $NLL(\theta) = {1 \over N} \sum p(y | f(x, \theta))$
  - Optimal parameters give the MLE estimate

- Regression

  - Output is a real valued quantity
  - Model fitting often involves minimizing the quadratic loss or MSE
  - $L(\theta) = {1 \over N} \sum (y - f(x, \theta))^2$
  - Data uncertainty. For example: if the output distribution is Gaussian
  - $p(y | x, \theta) = \mathcal N(y | f(x, \theta), \ \sigma^2)$
  - $NLL(\theta) \propto MSE(\theta)$
  - Linear Regression is an affine transformation between inputs and outputs
  - Polynomial Regression improves the fit by considering higher order interactions
  - FNNs do feature extraction by nesting the functions

- Overfitting and Generalization

  - A model that perfectly fits training data but is too complex suffers from overfitting
  - Population risk the theoretical expected loss on the true data generating process
  - Generalization gap is the difference between empirical risk and population risk
  - High generalization gap is a sign of overfitting
  - Population risk is hard to estimate. Approximate using test risk. Expected error on unseen data points.
  - Test error has U-shaped curve wrt model’s degree of freedom

- No Free Lunch Theorem: No single best model that works optimally for all kinds of problems

**Unsupervised Learning**

- Learn an unconditional model of the data $p(x)$ rather than $p(y | x)$
- In clustering, the goal is to partition the input space into regions with homogenous points.
- Dimensionality reduction projects input data from high dimension to lower dimension subspace.
- Self-Supervised Learning involves creating proxy supervised tasks from unlabled data
- Evaluation is done by measuring the probability assigned by the model to unseen data
- This treats the problem as one of density estimation
- Unsupervised learning also aims to increase sample efficiency in downstream supervised learning tasks

**Reinforcement Learning**

- A system or an agent learns how to interact with its environment
- Goal is learn a policy that specifies optimal action given a state
- Unlike, Supervised Learning, the reward signal is occasional and delayed.
- Learning with critic vs learning with teacher.

**Data Preprocessing**

- Text Data

  - Bag of Words transforms a document to a term frequency matrix
  - Frequent words have undue influence (pareto distribution)
  - Log of counts assuage some of the problems
  - Inverse Document Frequency: $\text{IDF}_i = \log {N \over 1 + \text{DF}_i}$
  - N is the total number of documents and DF is the documents with term i
  - $\text{TF-IDF} = \log(1 + TF) \times IDF$
  - Word embeddings map sparse vector representation of word to lower dimension dense vector
  - UNK token can help capture OOV words
  - Sub-word units or word pieces created using byte-pair encoding perform better than UNK token and help in reducing the vocabulary.

- Missing Data
  - MCAR: missing completely at random
  - MAR: missing at random
  - NMAR: not missing at random
  - Handling of the missing values depends to the type

## Chapter 2: Probability — Univariate Models

**Definitions**

- Frequentist View: Long run frequencies of events that can happen multiple times
- Bayesian View: Quantify the uncertainty
  - Model Uncertainty: Ignorance of underlying process
  - Data Uncertainty: Stochasticity
  - Data uncertainty can’t be reduced with more data
- Event: Some state of the world (A) that either holds or doesn’t hold.
  - $0 \le P(A) \le 1$
  - $P(A) + P(\bar A) = 1$
- Joint Probability: If two events happen simultaneously
  - $P(A,B)$
  - If A and B are independent: $P(A,B) = P(A)P(B)$
  - $P(A \cup B) = P(A) + P(B) - P(A \cap B)$
- Conditional Probability: Event B happens given A has already happened
  - $P(A | B) = P(A \cap B) | P(A)$
- A random variable represents unknown quantity of interest whose value cannot be determined.
- Sample space denotes the set of possible values of a random variable.
- Event is a set of outcomes from a given sample space.
  - If the sample is finite or countably finite, it’s discrete random variable
  - If the sample space is real valued, it’s continuous random variable
- Probability Mass Function computes the probability of events of a given random variable
  - $0 \le p(x) \le 1$
  - $\sum_x p(x) = 1$
- Cumulative Distribution Function are monotonically non-decreasing functions.
  - $\text{CDF}(x) = P(X \le x)$
  - $P(A \le X \le B) = \text{CDF}(B) - \text{CDF}(A)$
- Probability Density Function is the derivative of CDF
- Inverse CDF or Quantile Function
  - $P^{-1}(0.5)$ is the median
  - $P^{-1}(0.25); P^{-1}(0.75)$ are lower and upper quartiles
- Marginal Distribution of an random variable
  - $p(X=x) = \sum_y p(X=x, Y=y)$
- Conditional Distribution of a Random Variable
  - $p(Y=y | X=x) = {p(Y=y, X=x) \over p(X=x)}$
- Product Rule
  - $p(x,y) = p(y|x)p(x) = p(x|y) p(y)$
- Chain Rule
  - $p(x1,x2,x3) = p(x1) p(x2 | x1) p(x3 | x1, x2)$
- X and Y are independent
  - $X \perp Y \Rightarrow p(X,Y) = p(X) p(Y)$
- X and Y are conditionally independent of Z
  - $X \perp Y | Z \Rightarrow p(X,Y | Z) = p(X|Z) p(Y | Z)$

**Moments of a Distribution**

- Mean or Expected Value

  - First moment around origin
  - $\mathbf E(X) = \sum xp(x) \; \text{OR} \; \int_x xp(x) dx$
  - Linearity of Expectation: $\mathbf E(aX + b) = a \mathbf E(X) + b$

- Variance of a distribution

  - Second moment around mean
  - $\mathbf E(X-\mu)^2 = \sigma^2$
  - $\text{Var}(aX + b) = a^2 Var(X)$

- Mode of a distribution

  - Value with highest probability mass or probability density

- Law of Total / Iterated Expectation

  - $E(X) = E(E(X|Y))$

- Law of Total Variance
  - $V(X) = E(V(X | Y)) + V(E(X | Y))$

**Bayes’ Rule**

- Compute probability distribution over some unknown quantity H given observed data Y
- $P(H | Y) = {P(Y |H) P(H) \over P(Y)}$
- Follows from product rule
- p(H) is the prior distribution
- p(Y | H) is the observation distribution
- p(Y=y | H=h) is the likelihood
- Bayesian Inference: $\text{posterior} \propto \text{prior} \times \text{likelihood}$

**Distributions**

- Bernoulli and Binomial Distribution

  - Describes a binary outcome
  - $Y \sim Ber(\theta)$
  - $Y = \theta^y (1 - \theta)^{1-y}$
  - Binomial distribution is N repeatitions of Bernoulli trials
  - $Bin(p | N,\theta) = {N \choose p} \theta^p (1 - \theta)^{1-p}$

- Logistic Distribution

  - If we model a binary outcome using ML model, the range of f(X) is [0,1]
  - To avoid this constraint, use logistic function: $\sigma(a) = {1 \over 1 + e^{-a}}$
  - The quantity a is log-odds: log(p | 1-p)
  - Logistic function maps log-odds to probability
  - $p(y=1|x, \theta) = \sigma(f(x, \theta))$
  - $p(y=0|x, \theta) = \sigma( - f(x, \theta))$
  - Binary Logistic Regression: $p(y|x, \theta) = \sigma(wX +b)$
  - Decision boundary: $p(y|x, \theta) = 0.5$
  - As we move away from decision boundary, model becomes more confident about the label

- Categorical Distribution

  - Generalizes Bernoulli to more than two classes
  - $\text{Cat}(y | \theta) = \prod \theta_c ^ {I(y=C)} \Rightarrow p(y = c | \theta) = \theta_c$
  - Categorical distribution is a special case of multinomial distribution. It drops the multinomial coefficient.
  - The categorical distribution needs to satisfy
    - $0 \le f(X, \theta) \le 1$
    - $\sum f(X, \theta) = 1$
  - To avoid these constraints, its common to pass the raw logit values to a softmax function
    - ${e^x_1 \over \sum e^x_i} , {e^x_2 \over \sum e^x_i}....$
  - Softmax function is “soft-argmax”
    - Divide the raw logits by a constant T (temperature)
    - If T → 0 all the mass is concentrated at the most probable state, winner takes all
  - If we use categorical distribution for binary case, the model is over-parameterized.
    - $p(y = 0 | x) = {e^{a_0} \over e^{a_0} + e^{a_1}} = \sigma(a_0 - a_1)$

- Log-Sum-Exp Trick

  - If the raw logit values grow large, the denominator of softmax can enounter numerical overflow.
  - To avoid this:
    - $\log \sum \exp(a_c) = m + \log \sum \exp(a_c - m)$
    - if m is arg max over a, then we wont encounter overflow.
  - LSE trick is used in stable cross-entropy calculation by transforming the sigmoid function to LSE(0,-a).

- Gaussian Distribution

  - CDF of Gaussian is defined as
    - $\Phi(y; \mu, \sigma^2) = {1 \over 2} [ 1 + \text{erf}({z \over \sqrt(2)})]$
    - erf is the error function
  - The inverse of the CDF is called the probit function.
  - The derivative of the CFD gives the pdf of normal distribution
  - Mean, Median and Mode of gaussian is $\mu$
  - Variance of Gaussian is $\sigma$
  - Linear Regression uses conditional gaussian distribution
    - $p(y | x, \theta) = \mathcal N(y | f_\mu(x, \theta); f_\sigma(x, \theta))$
    - if variance does not depend on x, the model is homoscedastic.
  - Gaussian Distribution is widely used because:
    - parameters are easy to interpret
    - makes least number of assumption, has maximum entropy
    - central limit theorem: sum of independent random variables are approximately gaussian
  - Dirac Delta function puts all the mass at the mean. As variance approaches 0, gaussian turns into dirac delta.
  - Gaussian distribution is sensitive to outliers. A robust alternative is t-distribution.
    - PDF decays as polynomial function of distance from mean.
    - It has heavy tails i.e. more mass
    - Mean and mode is same as gaussian.
    - Variance is $\nu \sigma^2 \over \nu -2$
    - As degrees of freedom increase, the distribution approaches gaussian.

- Exponential distribution describes times between events in Poisson process.
- Chi-Squared Distribution is sum-squares of Gaussian Random Variables.

**Transformations**

- Assume we have a deterministic mapping y = f(x)
- In discrete case, we can derive the PMF of y by summing over all x
- In continuous case:

  - $P_y(y) = P(Y \le y) = P(f(X) \le y) = P(X \le f^{-1}(y)) = P_x(f^{-1}(y))$
  - Taking derivatives of the equation above gives the result.
  - $p_y(y) = p_x(x)|{dy \over dx}|$
  - In multivariate case, the derivative is replaced by Jacobian.

- Convolution Theorem

  - y = x1 + x2
  - $P(y \le y^*) = \int_{-\infty}^{\infty}p_{x_1}(x_1) dx_1 \int_{-\infty}^{y^* - x1}p_{x_2}(x_2)dx_2$
  - Differentiating under integral sign gives the convolution operator
  - $p(y) = \int p_1(x_1) p_2(y - x_1) dx_1$
  - In case x1 and x2 are gaussian, the resulting pdf from convolution operator is also gaussian. → sum of gaussians results in gaussian (reproducibility)

- Central Limit Theorem

  - Suppose there are N random variables that are independently identically distributed.
  - As N increases, the distribution of this sum approaches Gaussian with:
    - Mean as Sample Mean
    - Variance as Sample Variance

- Monte-Carlo Approximation
  - It’s often difficult ti compute the pdf of transformation y = f(x).
  - Alternative:
    - Draw a large number of samples from x
    - Use the samples to approximate y

## Chapter 3: Probability — Multivariate Models

- Covariance measures the degree of linear association
  - $\text{COV}[X,Y] = E[(X - E(X))(Y - E[Y])]$
- Covariance is unscaled measure. Correlation scales covariance between -1, 1.
  - $\rho = {\text{COV}[X,Y] \over \sqrt{V(X)} \sqrt{V(Y)}}$
- Independent variables are uncorrelated. But, vice-versa is not true.
- Correlation doesn’t imply causation. Can be spurious.

- Simpson’s Paradox

  - Statistical Trend that appears in groups of data can disappear or reverse when the groups are combined.

- Mixture Models

  - Convex combination of simple distributions
  - $p(y|\theta) = \sum \pi_k p_k(y)$
  - First sample a component and then sample points from the component
  - GMM: $p(y) = \sum_K \pi_k \mathcal N(y | \mu_k, \sigma_k)$
  - GMMs can be used for unsupervised soft clustering.
  - K Means clustering is a special case of GMMs
    - Uniform priors over components
    - Spherical Gaussians with identity matrix variance

- Markov Chains
  - Chain Rule of probability
  - $p(x1,x2,x3) = p(x1) p(x2 | x1) p(x3 | x1, x2)$
  - First-order Markov Chain: Future only depends on the current state.
  - y(t+1:T) is independent of y(1:t)
  - $p(x1,x2,x3) = p(x1) p(x2 | x1) p(x3 | x2)$
  - The p(y | y-1) function gives the state transition matrix
  - Relaxing these conditions gives bigram and trigram models.

## Chapter 4: Statistics

- Inference is the process of quantifying uncertainty about an unknown quantity estimated from finite sample of data

**Maximum Likelihood Estimation**

- Pick parameters that assign highest probability to training data
  - $\theta_{MLE} = \arg \max p(D | \theta) = \prod p(y | x, \theta)$
- MLE can be factorized because of IID assumption
- Maximizing MLE is equivalent to minimizing NLL
  - $\text{NLL}(\theta) = -\log p(D | \theta)$
- For unsupervised learning MLE is unconditional.
  - $\theta_{MLE} = \arg\max p( x | \theta)$
- Justification for MLE
  - Bayesian MAP estimate with uninformative uniform prior
    - $\theta_{MAP} = \arg\max p(\theta | D) = \arg \max p(D | \theta) + p(\theta)$
  - KL Divergence: MLE brings predicted distribution close to empirical ditribution
    - $KL(p||q) = H(p) - H(p,q)$
    - Cross-entropy term in KL-Divergence corresponds to KL-Divergence
- Sufficient Statistics of the data summarize all the information needed.
  <!-- - N0 (negative \#samples) and N1 (positive \#samples) in case of Bernoulli Distribution -->
  - N0 (negative #samples) and N1 (positive #samples) in case of Bernoulli Distribution
- MLE Examples
  - Bernoulli Distribution
    - $NLL(\theta) = N_1 \log(\theta) - N_0 \log(1-\theta)$
    - $\Delta NLL \Rightarrow \theta = N_1 / (N_0 + N_1)$
  - Categorical DIstribution
    - Add unity contraint as Lagrangian
    - $NLL(\theta) = \sum N_k \log(\theta) + \lambda (\sum \theta_k -1))$
  - Gaussian Distribution
    - $NLL(\theta) = {1 \over 2\sigma^2 }\sum \log(y - \mu)^2 + {N \over 2} log (2\pi \sigma^2)$
    - Sample mean and sample variance become sufficient statistics
  - Linear Regression
    - $p(y | x; \theta) = \mathcal N (y | wx +b, \sigma^2)$
    - $NLL \propto \sum (y - wx - b) ^ 2$
    - Quadratic Loss is a good choice for linear regression

**Empirical Risk Minimization**

- Empirical Risk Minimization is the expected loss where the expectation is taken wrt to empirical distribution
- ERM generalizes MLE by replacing log-loss with any loss function
  - $L(\theta) = {1 \over N} \sum l(y, x, \theta)$
  - Loss could be miss-classification rate as an example
- Surrogate losses devised to make optimization easier.

  - Log-Loss, Hinge-Loss etc.

- Method of Moments (MoM) compares theoretical moments of a distribution with to the empirical ones.

  - Moments are quantitative measures related to the shape of the function's graph

- In batch learning, entire dataset is available before training.
- In online learning, dataset arrives sequentially.
  - $\theta_t = f(x_t, \theta_{t-1})$
  - Recursive updates are required. For example MA, or EWMA
    - $\mu_t = \mu_{t-1} + {1 \over t}(x_t - \mu_{t-1})$
    - $\mu_t = \beta \mu_{t-1} + (1 - \beta) y_t$

**Regularization**

- MLE/ERM picks parameters that minimize loss on training set.
- Empirical distribution may not be same as true distribution.
- Model may not generalize well. Loss on unseen data points could be high. Overfitting.
- Regularization helps reduce overfitting by adding a penalty on complexity.
  - In-built in MAP estimation
  - $L(\theta) = NLL + \lambda \log p(\theta)$
  - Add-one smoothing in Bernoulli to solve zero count problem is regularization.
  - The extra one term comes from Beta priors.
- In linear regression, assume parameters from standard gaussian.
  - $L(\theta) = NLL + \lambda \log w^2$
  - L2 Penalty in MAP estimation
- Regularization strength is picked by looking at validation dataset
  - Validation risk is estimate for population risk.
  - Cross-Validation in case of small size of training dataset
- One Standard Error Rule
  - Select the model with loss within one SE of the baseline / simple model
- Early Stopping prevents too many steps away from priors. Model doesn’t memorize too much.
- Using more suitable informative data samples also prevents overfitting.
  - Bayes’ Error is inherent error due to stochasticity.
  - With more data, learning curve approaches Bayes’ Error.
  - If we start with very few observations, adding more data may increase the error as model uncovers new data patterns.

**Bayesian Statistics**

- Start with prior distribution
- Likelihood reflects the data for each setting of the prior
- Marginal Likelihood shows the average probability of the data by marginalizing over model parameters
- Posterior Predictive Distribution: is Bayes Model Averaging
  - $p(y | x, D) = \int p(y | x, \theta) p(\theta | D) d\theta$
  - Multiple parameter values considered, prevents overfitting
  - Plug-in Approximation: Uses dirac delta to pul all the weight on MLE
  - This simplifies the calculations
- Conjugate Priors
  - posterior = prior x likelihood
  - Select prior in a form that posterior is closed form and has same family as prior
  - Bernoulli-Beta
  - Gaussian-Gaussian

**Frequentist Statistics**

- Data is a random sample drawn from some underlying distribution
- Induces a distribution over the test statistic calculated from the sample.
- Estimate variation across repeated trials.
- Uncertainty is calculated by quantifying how the estimate would change if the data was sampled again.
- Sampling Distribution
  - Distribution of results if the estimator is applied multiple times to different datasets sampled from same distribution
- Bootstrap
  - If the underlying distribution is complex, approximate it by a Monte-Carlo technique
  - Sample N data points from original dataset of size N with replacement
  - Bootstrap Sample is 0.633 x N on average
    - Probability the point is selected atleast once
    - $1 - (1 - {1 \over N})^N \approx 1 - {1 \over e}$
- 100 (1 - a) % CI is the probability that the true value of the parameter lies in the range.

**Bias-Variance Tradeoff**

- Bias of an estimator
  - $bias(\hat \theta) = E[\hat \theta] - \theta^*$
    - Measures how much the estimate will differ from true value
    - Sample variance is not an unbiased estimator for variance
  - $\mathbf V[\hat \theta] = E[\hat \theta ^ 2] - E[\hat \theta]^2$
    - Measures how much will the estimate vary is data is resampled
  - Mean Squared Error
    - $E[(\hat \theta - \theta^*)^2] = \text{bias}^2 + \text{variance}$
    - It’s okay to use a biased estimator if the bias is offset by decrease in variance.

## Chapter 5: Decision Theory

- Optimal Policy specifies which action to take for each possible observation to minimize risk or maximize utility
- Implicit assumption is that agents are risk neutral. 50 vs 0.5 \* 100
- Zero-One loss: miss-classification rate in binary classifier
  - $l_{01}(y, \hat y) = I\{y \ne \hat y\}$
  - Optimal policy is to choose most probable label to minimize risk
    - $R(y | x) = p(y \ne \hat y | x) = 1 - p(y = \hat y | x)$
    - $\pi(x) = \arg \max p(y | x)$
  - In case the errors are cost-sensitive
    - FP is not same as FN
    - $l_{01} = c \times l_{10}$
    - Choose the label 1 if expected loss is lower:
      - $p0 \times l_{01} < p1 \times c \times l_{10}$
    - c will trade-off the decision boundary
  - In case reject or abstain is also a possible action
    - Assume the cost of error $\lambda_e$
    - Assume the cost of rejection: $\lambda_r$
    - No decision when model confidence is below $1 - {\lambda_e \over \lambda _r}$

**ROC Curves**

- Summarize performance across various thresholds

- Confusion Matrix
  - Give a threshold $\tau$
  - Confusion Matrix
    - Positive, Negative: Model Prediction
    - True, False: Actual Labels
    - TP, TN: Correct Predictions
    - FP: Model predicts 1, Ground Truth is 0
    - FN: Model predicts 0, Ground Truth is 1
  - Ratios from Confusion Matrix
    - TPR, Sensitivity, Recall
      - TP / (TP + FN)
      - Accuracy in positive predictions
    - FPR, Type 1 Error rate
      - FP / (FP + TN)
      - Error in Negative Predictions
  - ROC Curve is a plot between FPR (x-axis) and TPR (y-axis) across various thresholds
  - AUC is a numerical summary of ROC
  - Equal Error Rate is where ROC crosses -45 degree line.
  - ROC Curve is insensitive to class imbalance
    - FPR consists of TN in denominator
    - If TN >> TP, metric becomes insensitive to FPR
  - Precision-Recall Curves
    - The negatives are not model specific but system specific
    - For a search query, retrieve 50 vs 500 items. (or tiles vs list)
    - Precision
      - TP / TP + FP
    - Recall
      - TP / TP + FN
    - There is no dependency on TN
    - Precision curve has distortions. Smooth it out by interpolation.
    - To summarize the performance by a scalar
      - Precision @ K
      - Average Precision: Area under interpolated precision curve
      - mAP or Mean Average Precision is mean of AP across different PR curves (say different queries)
    - F-Score
      - Weighted harmonic mean between precision and recall
      - ${1 \over F} = {1 \over 1 + \beta^2} {1 \over P} + {\beta^2 \over 1 + \beta^2} {1 \over R}$
      - Harmonic mean imposes more penalty if either precision or recall fall to a very low level
  - Class Imbalance
    - ROC curves are not sensitive to class imbalance. Does not matter which class is defined as 1 or 0.
    - PR curves are sensitive to class imbalance. Switching classes impacts performance.
      - $P = {TP \over TP + FP}$
      - $P = {TPR \over TPR + r^{-1} FPR}$
      - r = positive / negative
    - F-Score is also affected by class imbalance.

**Regression Metrics**

- L2 Loss

  - $l(h,a) = (h-a)^2$
  - Risk Estimate
  - $R(a|x) = E[(h-a)^2| x] = E[h^2|x] -2aE[h|x] + a^2$
  - To minimize risk, set the derivative of risk to zero.
  - $\pi(x) \Rightarrow E[h|X] = a$
  - Optimal action is to set the prediction to posterior conditional mean.

- L1 Loss

  - L2 Loss is sensitive to outliers.
  - L1 is more robust to alternatives
  - $l(h,a) = |h-a|$

- Huber Loss
  - Middle ground between L1 and L2 loss
  - Set a threshold $\delta$
    - If error exceeds thresholds → L1 loss
    - If error below threshold → L2 loss

**Probabilistic Metrics**

- Estimate probabilistic distribution over labels

- KL Divergence
  - $KL(p||q) = \sum p log(p|q)$
  - $KL(p||q) = H(p,q) - H(p)$
  - Always >= 0. Equality holds when p == q
  - H(p) is the entropy.
  - H(p,q) is the cross entropy.
  - Cross entropy measures the bits required to encode data coming from p encoded via q.
  - KL divergence measures the extra bits needed to compress information using wrong distribution q instead of p.
  - H(p) is independent of q. Hence, minimizing KL divergence is equivalent to minimizing the cross-entropy.
  - Extending cross-entropy to multiple labels leads to log-loss.
  - KL divergence is sensitive to errors at low probability events.

**A/B Testing**

- Test and Roll approach to business decisions
- Randomly assign different actions to different populations
- Incurs opportunity cost. Exploration-Exploitation tradeoff.

- Bayesian Approach
- Bandits
- Marginal Log-Likelihood

<INCOMPLETE>

**Information Criteria**

- Marginal Likelihood difficult to compute.
- ICs incorporate model complexity penalty without the use of validation set.
- ICs are of the form deviance + some form of complexity.
  - $\text{deviance} = -2 \sum \log p + C$
- Bayesian Information Criterion

  - $C = \log(N) \times \text{dof}$
  - dof is degrees of freedom or number of free parameters
  - log of marginal likelihood of the gaussian approximation to the posterior

- Akaike Information Criterion
  - Penalizes model less heavily compared to BIC
  - $C = 2 \times \text{dof}$

**Frequentist Decision Theory**

- Risk of an estimator is the expected loss when applying the estimator to data sampled from likelihood function $p( y,x | \theta)$
- Bayes Risk
  - True generating function unknown
  - Assume a prior and then average it out
- Maximum Risk
  - Minimize the maximum risk
- Consistent Estimator
  - Recovers true parameter in the limit of infinite data
- Empirical Risk Minimization
  - Population Risk
    - Expectation of the loss function w.r.t. true distribution
    - True distribution is unknown
    - $R(f, \theta^*) = \mathbf{E}[l(\theta^*, \pi(D))]$
  - Empirical Risk
    - Approximate the expectation of loss by using training data samples
    - $R(f, D) = \mathbf{E}[l(y, \pi(x))]$
  - Empirical Risk Minimizaiton
    - Optimize empirical risk over hypothesis space of functions
    - $f_{ERM} = \arg \min_H R(f,D)$
  - Approximation Error
    - Risk that the chosen true parameters don’t lie in the hypothesis space
  - Estimation Error
    - Error due to having finite training set
    - Difference between training error and test error
    - Generalization Gap
  - Regularized Risk
    - Add complexity penalty
    - $R_\lambda(f,D) = R(f,D) + \lambda C(f)$
    - Complexity term resembles the prior term in MAP estimation
  - Structural Risk
    - Empirical underestimates population risk
    - Structural risk minimization is to pick the right level of model complexity by minimizing regularized risk and cross-validation

**Statistical Learning Theory**

- Upper bound on generalization error with certain probability
- PAC (probably approximately correct) learnable
- Hoeffding’s Inequality
  - Upper bound on generalization error
- VC Dimension
  - Measures the degrees of freedom of a hypothesis class

**Frequentist Hypothesis Testing**

- Null vs Alternate Hypothesis
- Likelihood Ratio Test
  - $p(D| H_0) / p(D| H_1)$
- Null Hypothesis Significance Testing
  - Type-1 Error
    - P(Reject H0 | H0 is True)
    - Significance of the test
    - $\alpha$
  - Type-2 Error
    - P(Accept H0 | H1 is True)
    - $\beta$
    - Power of the test is $1 - \beta$
  - Most powerful test is the one with highest power given a level of significance
  - Neyman-Pearson lemma: Likelihood ratio test is the most powerful test
  - p-value
    - Probability, under the null hypothesis, of observing a test statistic larger that that actually observed

## Chapter 6: Information Theory

- Entropy is a measure of uncertainty or lack of predictability associated with a distribution
- If entropy is high, it’s difficult to predict the value of observation
- $H(X) = - \sum p \log p = - E(\log p)$
- Uniform distribution has maximum entropy
- Dirac Delta distribution has minimum entropy

- Cross Entropy

  - $H(p,q) = -\sum p \log q$

- Joint Entropy

  - $H(X,Y) = -\sum p(x,y) \log p(x,y)$

- Conditional Entrpoy

  - $H(Y | X) = \sum_x p(X) H(p(Y | X=x))$
  - $H(Y|X) = H(X,Y) - H(X)$
  - Reduction in joint uncertainty (X,Y) given we observed X

- Perplexity

  - Exponentiated cross-entropy
  - Geometric mean of inverse probabilities
  - $\text{perplexity}(p) = 2^{H(p)} = \sqrt{ \prod {1 \over p}}$
  - Used to evaluate the quality of generative language models
  - Weighted average of branching factor
    - Number of possible words that can follow a given word
    - Given vocab size is K
    - If some words are more frequent, perplexity is lower than K

- KL Divergence

  - Relative Entropy
  - Distance between two distribution
  - $KL(p||q) = H(p,q) - H(p)$
  - Extra bits needed when compressing data generated from p using q
  - Suppose objective is to minimize KL divergence
  - Empirical distribution puts probability mass on training data and zero mass every where else
    - $p = {1 \over N} \sum {\delta (x - x_n)}$
  - This reduces KL divergence to cross entropy or negative log likelihood.
    - $KL(p||q) = {1 \over N} \sum \log q(x_n)$
  - Data augmentation perturbs data samples to reflect natural variations. This spreads the probability mass over larger space. Prevents overfitting.

- Mutual Information

  - $I(X,Y) = KL(p(x,y) || p(x)p(y))$
  - KL Divergence between joint and factored marginal distribution
  - $I(X,Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)$
  - Reduction in uncertainty about X after observing Y
  - Generalized correlation coefficient that can capture non-linear trends.
  - Can be normalized to reduce scale effect
  - Data Processing Inequality: Transformation cannot increase the amount of information

- Fano’s Inequality
  - Feature selection via high mutual information
  - Bounds probability of misclassification in terms of mutual information between features

## Chapter 8: Optimization

- Optimization Problem: Try to find values for a set of variables that minimize/maximize a scalar valued objective function
  - $\arg \min_{\theta}L(\theta)$
- The point that satisfies the optimization problem is called global optimum
- Local optimum is a point that has optimal objective value compared to nearby points.
- Optimality Conditions
  - gradient $g(\theta) = \Delta L(\theta)$ is zero
  - hessain $H(\theta) = \Delta^2 L(\theta)$ is positive definite
- Unconstrained Optimization: Finding any value in parameter space that minimizes the loss
- Constrained Optimization: Finding optimal value in a feasible set that is subset of the parameter space. $\mathit C \in \{\theta : g_j(\theta) \le 0 : j \in I, h_k(\theta)= 0 : k \in E \}$
  - I is the set of ineuqliaty constraints
  - K is the set of equality constraints
  - If there are too many constraints the feasible set may become empty.
- Smooth Optimization: Objective and constraints are continuously differentiable
- Lipschitz Constant: $|f(x_1) - f(x_2)| \le L|x_1 - x_2|$
  - Function cannot change by more than L units if input changes by 1 unit
- Non-smooth Optimization: Some points where gradient of the objective or the constraints is not well defined
- Composite Objective: Contains both smooth and non-smooth terms.
- Subgradient: Generalized notion of derivative to work with functions having local discontinuities.

- First-Order Optimization Methods

  - Leverage first-order derivatives of the objective function
  - Ignore the curvature information
  - $\theta_t = \theta_{t-1} + \eta_t d_t$
  - d is the descent direction, $\eta$ is the step size
  - Steepest Descent: direction opposite to the gradient g
  - Step Size: controls the amount to move in the descent direction
    - Constant Step Size
      - incorrect values can lead to oscillations, slow convergence
    - Line Search
      - set as a 1d minimization problem to select the optimal value
    - Learning rate schedule must respect Robbins-Monro condition
      - ${\sum \eta^2 \over \sum \eta} \rightarrow 0 \, \text{as} \, \eta \rightarrow 0$
  - Momentum
    - Gradient Descent slow across lat regions of the loss landscape
    - Heavy Ball or Momentum helps move faster along the directions that were previously good.
    - $m_t = \beta m_{t-1} + g_{t-1}$
    - $\theta_t = \theta_{t-1} + \eta_t m_t$
    - Momentum is essentially EWMA of gradients
  - Nestrov Momentum
    - Momentum may not slow down enough at the bottom causing oscillation
    - Nestrov solves for that by adding a lookahead term
    - $m_{t+1} = \beta m_t - \eta_t \Delta L(\theta_t + \beta m_t)$
    - It updates the momentum using gradient at the predicted new location

- Second-Order Optimization Methods

  - Gradients are cheap to compute and store but lack curvature information
  - Second-order methods use Hessian to achieve faster convergence
  - Newton’s Method:
    - Second-order Taylor series expansion of objective
    - $L(\theta) = L(\theta_t) + g(\theta - \theta_t) + {1 \over 2} H (\theta - \theta_t)^2$
    - Descent Direction: $\theta = \theta_t - H^{-1} g$
  - BFGS:
    - Quasi-Newton method
    - Hessian expensive to compute
    - Approximate Hessian by using the gradient vectors
    - Memory issues
    - L-BFGS is limited memory BFGS
    - Uses only recent gradients for calculating Hessian

- Stochastic Gradient Descent

  - Goal is to minimize average value of a function with random inputs
  - $L(\theta) = \mathbf E_z[L(\theta, z)]$
  - Random variable Z is independent of parameters theta
  - The gradient descent estimate is therefore unbiased
  - Empirical Risk Minimization (ERM) involves minimizing a finite sum problem
    - $L(\theta) = {1 \over N}\sum l(y, f(x(\theta))$
  - Gradient calculation requires summing over N
  - It can be approximated by summing over minibatch B << N in case of random sampling
  - This will give unbiased approximation and results in faster convergence

- Variance Reduction

  - Reduce the variance in gradient estimates by SGD
  - Stochastic Variance Reduced Gradient (SVRG)
    - Adjust the stochastic estimates by those calculated on full batch
  - Stochastic Averaged Gradient Accelerated (SAGA)
    - Aggregate the gradients to calculate average values
    - $g_t = \Delta L(\theta) - g_{local} + g_{avg}$

- Optimizers

  - AdaGrad (Adaptive Gradient)
    - Sparse gradients corresponding to features that are rarely present
    - $\theta_{t+1} = \theta_t -\eta_t {1 \over \sqrt{s_t +\epsilon}} g_t$
    - $s_t = \sum g^2$
    - It results in adaptive learning rate
    - As the denominator grows, the effective learning rate drops
  - RMSProp
    - Uses EWMA instead of sum in AdaGrad
    - $s_t = \beta s_{t-1} + (1-\beta)g^2_t$
    - Prevents from s to grow infinitely large
  - AdaDelta

    - Like RMSProp, uses EWMA on previous gradients
    - But also uses EWMA on updates
    - $\delta_t = \beta \delta_{t-1} + (1 - \beta) (\Delta \theta^2)$
    - $\theta_{t+1} = \theta_t -\eta_t {\sqrt{\delta_t +\epsilon} \over \sqrt{s_t +\epsilon}} g_t$

  - Adam
    - Adaptive Moment Estimation
    - Combines RMSProp with momentum
    - $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
    - $s_t = \beta_1 s_{t-1} + (1 - \beta_1) g_t^2$
    - $\Delta \theta = \eta {1 \over \sqrt s_t + e} m_t$

- Constrained Optimization

  - Lagrange Multipliers
    - Convert constrained optimization problem (with equality constraints) to an unconstrained optimization problem
    - Assume the constraint is $h(\theta) = 0$
    - $\nabla h(\theta)$ is orthogonal to the plane $h(\theta) = 0$
      - First order Taylor expansion
    - Also, $\nabla L(\theta)$ is orthogonal to the plane $h(\theta) = 0$ at the optimum
      - Otherwise, moving along the constraint can improve the objective value
    - Hence, at the optimal solution: $\nabla L(\theta) = \lambda \nabla h(\theta)$
      - $\lambda$ is the Lagrangian multiplier
    - Convert the above identity to an objective
      - $L(\theta, \lambda) = L(\theta) - \lambda h(\theta)$

- KKT Conditions

  - Generalize the concept of Lagrange multiplier to inequality constraints
  - Assume the inequality constraint: $g(\theta) < 0$
  - $L(\theta, \mu) = L(\theta) + \mu g(\theta)$
  - $\min L(\theta) \rightarrow \min_{\theta} \max_{\mu \ge 0} L(\theta, \mu)$
    - Competing objectives
    - $\mu$ is the penalty for violating the constraint.
    - If $g(\theta) > 0$, then the objective becomes $\infty$
  - Complementary Slackness
    - If the constraint is active, $g(\theta) = 0, \mu > 0$
    - If the constraint is inactive, $g(\theta) < 0, \mu = 0$
    - $\mu * g = 0$

- Linear Programming

  - Feasible set is a convex polytope
  - Simplex algorithm moves from vertex to vertex of the polytope seeking the edge that improves the objective the most.

- Proximal Gradient Descent

  - Composite objective with smooth and rough parts
  - Proximal Gradient Descent calculates the gradients of the smooth part and projects the update into a space the respects the rough part
  - L1 Regularization is sparsity inducing. Can be optimized using proximal gradient descent. (0,1) is preferred vs $1 \over \sqrt 2$, $1 \over \sqrt 2$. L2 is agnostic between the two.

- Expectation Maximization Algorithm

  - Compute MLE / MAP in cases where there is missing data or hidden variables.
  - E Step: Estimates hidden variables / missing values
  - M Step: Uses observed data to calculate MLE / MAP
  - $LL(\theta) = \sum \log p( y | \theta) = \sum \log \sum p(y, z | \theta)$
  - z is the hidden / latent variable
  - Using Jensen’s inequality for convex functions
    - $LL(\theta) \ge \sum \sum q(z) \log p (y | \theta, z)$
    - q(z) is the prior estimate over hidden variable
    - log(p) is the conditional likelihood
    - Evidence lower bound or ELBO method
  - EMM for GMM
    - E Step: Compute the responsibility of cluster k for generating the data point
    - M Step: Maximize the computed log-likelihood

- Simulated Annealing
  - Stochastic Local Search algorithm that optimizes black-box functions whose gradients are intractable.

## Chapter 9: Linear Discriminant Analysis

- Generative Models specify a way to generate features for each class
  - $p(y = c |x, \theta) \propto p(x | y = c, \theta) \times p(y = c)$
  - $p(x | y, \theta)$ is the class conditional density
  - $p(y)$ is the prior over class labels
- Discriminative Models estimate the posterior class probability

  - $p(y | x, \theta)$

- Gaussian Discriminant Analysis

  - Class Conditional Densities are multivariate Gaussians
  - $p(x | y=c, \theta) = N(\mu_c, \Sigma_c)$
  - $\log p(y = c | x, \theta )$ will be quadratic in $\mu_c$, $\Sigma_c$ (QDA)
  - If the covariance matrices are shared across class labels, the decision boundary will become linear in $\mu_c$ (LDA)
  - LDA can be refactored to be similar to logistic regression
  - Models are fitted via MLE.
    - $\Sigma_c$ estimates often lead to overfitting
    - Tied covariances i.e. LDA solve this problem
    - MAP estimation can introduce some regularization
  - Class assignment is based on nearest centroid based on the estimates of $\mu_c$

- Naive Bayes Classifiers

  - Work on Naive Bayes Assumption
  - Features are conditionally independent given the class label
    - $p(\mathbf x | y = c) = \prod p(x_d | y = c)$
  - The assumption is naive since it will rarely hold true
  - $p(y = c | x, \theta) \propto \pi(y = c) \prod p(x_d | y = c, \theta_{dc})$
  - Model has very few parameters and is easy to estimate
  - The distribution of $p(x_d | y = c)$ is
    - Bernoulli for binary
    - Categorical for categorical
    - Gaussian for continuous

- Generative Classifiers are better at handing missing data or unlabeled data
- Discriminative Models give more robust estimates for posterior probabilities

## Chapter 10: Logistic Regression

- Discriminative classification model

  - Estimate $p(y | x, \theta)$
  - $y \in \{1,2,...,C\}$

- Binary Logisitc Regression

  - y is binary {0,1}
  - $p(y | x, \theta) = Ber(y | \sigma(w^Tx + b))$
  - $\sigma$ is the sigmoid function
  - $p(y = 1 | x, \theta) = \sigma(w^Tx +b)$
  - Alternative equivalent notation y is {-1, +1}
  - Compact notation:
    - $p(\tilde y | x, \theta) = \sigma(\tilde y \times (w^tx + b))$
  - If the misclassification cost is same across classes, optimal decision rule
    - predict y = 1 if class 1 is more likely
    - $p(y = 1 | x) > p (y = 0 | x)$
    - $\log {p(y = 1 |x) \over p(y = 0 | x)} > 1$
    - $w^Tx + b > 0$
    - $w^Tx + b$ is the linear decision boundary of the classifier
  - Maximum Likelihood Estimation
    - Minimize the NLL
    - $\text{NLL} = - \log \prod \text{Ber}(y| \sigma(w^Tx +b))$
    - $\text{NLL} = -\sum y \log(\hat y) = H(y, \hat y)$ i.e. binary cross-entropy
    - If compact notation is used
      - $\text{NLL} = \sum \log \sigma (\tilde y (w^Tx+b))$
      - $\text{NLL} = \sum \log ( 1 + \exp (\tilde y (w^Tx +b))))$
    - Optimization:
      - $\Delta \text{NLL} =0$ is the first order condition
      - $\Delta \sigma(x) = \sigma(x) \times (1 - \sigma(x))$
      - $\Delta NLL = \sum (\hat y - y) x$
      - Sum of residuals weighted by inputs
      - Hessian is positive definite making the optimization convex
      - Minimization of NLL can be achieved by first order methods like SGD
      - Second order methods like Newton’s method can result in faster convergence
        - IRLS (iteratively weighted least squares) is the equivalent
  - MAP Estimation
    - MLE estimation leads to overfitting
    - Use zero mean Gaussian priors over w
    - $p(w) = N(0, \lambda ^{-1} I)$
    - $L(w) = NLL(w) + \lambda ||w||^2$
    - $\lambda$ is the L2 regularization which penalizes the weights from growing large
    - Given the gaussian priors assume zero mean in MAP, it’s important to standardize the input features to make sure they are on the same scale

- Multinomial Logistic Regression assumes categorical distribution instead of Bernoulli

  - $p(y=c|x, \theta) = { \exp^{a_c} \over \sum \exp ^a}$
  - If features are made class dependent, the model is called maximum entropy classifier
    - Commonly used in NLP
    - $p(y=c|w, x) \propto exp(w^T\phi(x,c))$

- Hierarchical Classification

  - Labels follow a taxonomy
  - Label Smearing: Label is propagated to all the parent nodes
  - Set up as multi-label classification problem

- Handling Many Classes

  - Hierarchical Softmax
    - Faster computation of normalization constant in softmax
    - Place the output nodes in tree structure with frequent classes sitting on top
  - Class Imbalance
    - Long-tail has little effect on loss and model may ignore these classes
    - Use sampling startegies
    - $p_c = N_c^q / \sum N_c^q$
    - Instance based sampling: q = 1
    - Class balanced sampling: q = 0
    - Square root sampling: q = 0.5

- Robust Logistic Regression
  - Robust to outliers
  - Mixture Model
    - Smoothen the likelihood with uniform Bernoulli prior
    - $p(y | x) = \pi Ber(0.5) + (1 - \pi) Ber(y |x, \theta)$
  - Bi-tempered Logistic Loss
    - Tempered Cross-Entropy
      - Handles mislabeled outliers away from the decision boundary
    - Tempered Softmax
      - Handles mislabeled points near the decision boundary
  - Probit Approximtion
    - Sigmoid function is similar in shape to Gaussian CDF
    - Using it gives the probit approximation

## Chapter 11: Linear Regression

- Predict real valued output
- $p(y | x, \theta) = N(y | w^Tx +b, \sigma^2)$
- Simple Linear regression has i feature vector
- Multiple Linear Regression has many feature vectors
- Multivariate Linear Regression has multiple outputs
- Feature extractor helps in improving the fit of the model

- Least Square Estimate

  - Minimize the negative log likelihood (NLL)
  - $\text{NLL}(w, \sigma^2) = {1 \over 2\sigma^2} \sum (y - \hat y)^2 + {N \over 2} \log(2\pi\sigma^2)$
  - First term is referred as Residual Sum Squares (RSS)
  - Ordinary Least Squares
    - $\Delta_w RSS = 0$
    - $X^TXw = X^Ty$
    - Normal Equation because $Xw - y$ is orthogonal to $X$
    - $w = (X^TX)^{-1}X^Ty$
    - Hessian is $X^TX$ i.e. positive definite if X is full rank
  - Inverting $X^TX$ is not easy for numerical reasons as it may be ill-conditioned or singular
  - A better approach is to compute pseudo-inverse using SVD
  - If the variance is heteroskedastic, the model becomes weighted least squares.
    - $p(y|x, \theta) = N(y| wx +b, \sigma^2(x))$
  - In case of if simple linear regression:
    - $w = C_{xy} / C_{xx}$, i.e. ratio of covariances
    - $b = \bar y - w \bar x$
  - In case of two inputs with no correlation:
    - $W_{X1} = R_{YX2.X1}$
    - $W_{X2} = R_{YX1.X2}$
    - Partial Regression Coefficients Y on X1 keeping X2 constant
  - The estimate of variance from NLL is MSE of residuals
    - $\hat \sigma^2 = {1 \over N}\sum (y - \hat y)^2$

- Goodness of Fit

  - Residual Plots: Check if the residuals are normally distributed with zero mean
  - Prediction Accuracy: RMSE $\sqrt{ {1\over N} RSS}$ measures prediction error
  - Coefficient of Determination: $R^2 = 1 - {RSS \over TSS}$
    - TSS: Prediction from baseline model: average of Y
    - TSS - RSS: Reduction in variance / betterment in fit

- Ridge Regression

  - MLE / OLS estimates can result in overfitting
  - MAP estimation with zero mean Gaussian Prior
    - $p(w) = N(0, \lambda^{-1}\sigma^2)$
    - $L(w) = RSS + \lambda ||w||^2$
  - $\lambda$ is the L2 regularization or weight decay
  - Ridge Regression is connected to PCA
    - The eigenvectors, eigenvalues of $X^TX$ matrix
    - Ridge regression shrinks the eigenvectors corresponding to smaller eigenvalues.
    - $\lambda$ is sometimes referred as shrinkage parameter
    - Alternate way is to run PCA on X and then run regression
    - Ridge is a superior approach

- Robust Linear Regression

  - MLE/MAP is sensitive to outliers
  - Solutions
    - Replace Gaussian with Student-t distribution which has heavy tails
      - The model does not get obsessed with outliers
      - Tails have more mass which gets factored in while maximizing MLE
    - Compute MLE using EM
      - Represent Student-t distribution as Gaussian scale mixture
    - Using Laplace Distribution which is robust to outliers
    - Using Huber Loss
      - L2 loss for small errors
      - L1 loss for large erros
      - Loss function is differentiable
    - RANSAC
      - Random Sample Concensus
      - Identify outliers from fitted models

- Lasso Regression

  - Least absolute shrinkage and selection operator
  - Case where we want the parameters to be zero i.e. sparse models
  - Used for feature selection
  - MAP formulation with Laplace priors
  - L1 regularization
  - Rationale for sparsity
    - Consider Lagrange Formulation with constraint
    - L1 formulation: $||w|| \le B$
    - L2 formulation: $||w||^2 \le B$
    - L1 constraint is a rhombus
    - L2 constraint is a sphere
    - The objective is more likely to intersect L1 constraint at an point corner
    - At the corners the parameters for some dimensions are 0
  - Regularization Path
    - Start with very high value of regularization
    - Gradually decrease the regularization strength
    - The set of parameters that get swept out is known as regularization path
    - Performs variable selection

- Elastic Net

  - Combination of Ridge and Lasso
  - Helpful in dealing with correlated variables
  - Estimates of highly correlated variables tend be equal

- Coordinate Descent
  - Basis for glmnet library
  - Solve for jth coefficient while all others are fixed
  - Cycle through the coordinates

## Chapter 13: Neural Networks for Structured Data (FFNN)

- Linear Models do an affine transformation of inputs $f(x, \theta) = Wx + b$
- To increase model flexibility, perform feature transformation $f(x, \theta) = W \phi(x) + b$
- Repeatedly nesting the transformation functions results in deep neural networks
  - $f(x, \theta) = f_L(f_{L-1}(f_{L-2}.....(f_1(x)....))$
  - Composed of differentiable functions in any kind of DAG (directed acyclic graphs)
- Multilayer Perceptrons
  - XOR (Exclusive OR) Problem: Inputs are note linearly separable
  - Stacking multiple functions on top of each other can overcome this problem
- Stacking linear activation functions results in linear model
  - $f(x, \theta) = W_L(W_{L-1}(W_{L-2}.....(W_1(x)....)) = \tilde Wx$
- Activation functions are differentiable non-linear functions
  - Sigmoid: (0,1)
  - TanH: (-1,+1) (e2x -1 / e2x + 1)
  - ReLU: max(a, 0), non-saturating activation function
- Universal Function Approximator

  - MLP with one hidden layer is universal function approximator
  - Can form a suitable smooth function given enough hidden units

- Backpropagation Algorithm

  - Compute gradient of a loss function wrt parameters in each layer
  - Equivalent to repeated application of chain rule
  - Autodiff: Automatic Differentiation on Computation Graph
  - Suppose $f = f_1 \circ f_2 \circ f_3 \circ f_4$
    - Jacobain $\mathbf J_f$ needs to be calculated for backprop
    - Row Form: $\triangledown f_i(\mathbf x)$ is the ith row of jacobian
      - Calculated efficiently using forward mode
    - Column Form: $\delta \mathbf f \over \delta x_i$ is the ith column of jacobian
      - Calculated efficiently using the backward mode

- Derivatives

  - Cross-Entropy Layer
    - $z = \text{CrossEntropyWithLogitsLoss(y,x)}$
    - $z = -\sum_c y_c \log(p_c)$
    - $p_c = {\exp x_c \over \sum_c \exp x_c}$
    - ${\delta z \over \delta x_c} = \sum_c {\delta z \over \delta p_i} \times {\delta p_i \over \delta x_c}$
    - When i = c
      - ${\delta z \over \delta x_c} = {-y_c \over p_c} \times p_c (1 - p_c) = - y_c ( 1 - p_c)$
    - When i <> c
      - ${\delta z \over \delta x_c} = {-y_c \over p_c} \times - p_i p_c = -y_c p_c$
    - Adding up
      - $-y_c(1-p_c) + \sum_{i \ne c} y_c p_i = p_c \sum_c y_c - y_c = p_c - y_c$
  - ReLU
    - $\phi(x) = \max(x,0)$
    - $\phi'(x,a) =   I\{x > 0\}$
  - Adjoint
    - ${\delta o \over \delta x_j} = \sum_{children} {\delta o \over \delta x_i} \times {\delta x_i \over \delta x_j}$

- Training Neural Networks

  - Maximize the likelihood: $\min  L(\theta) = -\log p(D|\theta)$
  - Calculate gradients using backprop and use an optimizer to tune the parameters
  - Objective function is not convex and there is no guarantee to find a global minimum
  - Vanishing Gradients
    - Gradients become very small
    - Stacked layers diminish the error signals
    - Difficult to solve
    - Modify activation functions that don’t saturate
    - Switch to architectures with additive operations
    - Layer Normalization
  - Exploding Gradients
    - Gradients become very large
    - Stacked layers amplify the error signals
    - Controlled via gradient clipping
  - Exploding / Vanishing gradients are related to the eigenvalues of the Jacobian matrix
    - Chain Rule
    - ${\delta  L \over \delta z_l} = {\delta  L \over \delta z_{l+1}} \times {\delta z_{l+1}  \over \delta z_{l}}$
    - ${\delta  L \over \delta z_l} = J_l \times g_{l+1}$

- Non-Saturating Activations

  - Sigmoid
    - $f(x) = 1 / (1 + \exp^{-x}) = z$
    - $f'(x) = z (1 - z)$
    - If z is close to 0 or 1, the derivative vanishes
  - ReLU
    - $f(x) = \max(0, x)$
    - $f'(x) =  I \{x > 0\}$
    - Derivative will exist as long as the input is positive
    - Can still encounter dead ReLU problem when weights are large negative/positive
  - Leaky ReLU
    - $f(x,\alpha) = max(\alpha x, x); \,\,\, 0< \alpha < 1$
    - Slope is 1 for for positive inputs
    - Slope is alpha for negative inputs
    - If alpha is learnable, then we get parametric ReLU
  - ELU, SELU are smooth versions of ReLU
  - Swish Activation
    - $f(x) = x \sigma(x)$
    - $f'(x) = f(x) + \sigma(x) (1 - f(x))$
    - The slope has additive operations

- Residual Connections
  - It’s easier to learn small perturbations to inputs than to learn new output
  - $F_l(x) = x + \text{activation}_l(x)$
  - Doesn’t add more parameters
  - $z_L = z_l + \sum_{i=l}^L F_i(z_i)$
  - ${\delta L \over \delta \theta_l} = {\delta L \over \delta z_l} \times {\delta z_l \over \delta \theta_l}$
  - ${\delta L \over \delta \theta_l} = {\delta z_l \over \delta \theta_l} \times {\delta L \over \delta z_L} (1 + \sum f'(z_i))$
  - The derivative of the layer l has a term that is independent of the network
- Initialization

  - Sampling parameters from standard normal distribution with fixed variance can result in exploding gradients
  - Suppose we have linear activations sampled from standard Normal Distribution
    - $o = \sum w_j x_ij$
    - $E(o) =  \sum E(w_j)E(x_{ij}) = 0$
    - $V(o) \propto n_{in} \sigma^2$
  - Similarly for gradients:
    - $V(o') \propto n_{out} \sigma^2$
  - To prevent the expected variance from blowing up
    - $\sigma^2 = {1 \over (n_{in} + n_{out})}$
    - Xavier Initialization, Glorot Initialization
    - He/LeCun Initialization equivalent if n_in = n_out

- Regularization
  - Early Stopping
    - Stop training when error on validation set stops reducing
    - Restricts optimization algorithm to transfer information from the training examples
  - Weight Decay
    - Impose prior on parameters and then use MAP estimation
    - Encourages smaller weights
  - Sparse DNNs
    - Model compression via quantization
  - Dropout
    - Turnoff outgoing connections with probability p
    - Prevents complex co-adaptation
    - Each unit should learn to perform well on its own
    - At test time, turning on dropout is equivalent to ensemble of networks (Monte Calo Dropout)

## Chapter 14: Neural Networks for Images (CNN)

- MLPs not effective for images
  - Different sized inputs
  - Translational invariance difficult to achieve
  - Weight matrix prohibitive in size
- Convolutional Neural Networks

  - Replace matrix multiplication with convolution operator
  - Divide image into overlapping 2d patches
  - Perform template matching based on filters with learned parameters
  - Number of parameters significantly reduced
  - Translation invariance easy to achieve

- Convolution Operators

  - Convolution between two functions
    - $\[f \star g\](z) = \int f(u) g(z-u) du$
  - Similar to cross-correlation operator
    - $\[w \star x\](i) = \sum_u^{L-1} w_ux_{i+u}$
  - Convolution in 2D
    - $\[W \star X\](i,j) = \sum_{u=0}^{H-1}\sum_{v=0}^{W-1} w_{u,v}x_{i+u,j+v}$
    - 2D convolution is template matching, feature detection
    - The output is called feature map
  - Convolution is matrix multiplication
    - The corresponding weight matrix is Toeplitz like
    - $y = Cx$
    - $C = [[w_1, w_2,0|w_3, w_4, 0|0,0,0],[0, w_1, w_2 | 0, w_3, w_4 | 0,0,0],....]$
    - Weight matrix is sparse in a typical MLP setting
  - Valid Convolution
    - Filter Size: $(f_h, f_w)$
    - Image Size: $(x_h, x_w)$
    - Output Size : $(x_h - f_w + 1, x_w - f_w + 1)$
  - Padding
    - Filter Size: $(f_h, f_w)$
    - Image Size: $(x_h, x_w)$
    - Padding Size: $(p_h, p_w)$
    - Output Size : $(x_h + 2p_h - f_w + 1, x_w + 2p_w - f_w + 1)$
    - If 2p = f - 1, then output size is equal to input size
  - Strided Convolution
    - Skip every sth input to reduce redundancy
    - Filter Size: $(f_h, f_w)$
    - Image Size: $(x_h, x_w)$
    - Padding Size: $(p_h, p_w)$
    - Stride Size: $(s_h, s_w)$
    - Output Size: $\lbrack {x_h + 2p_h -f_h +s_h \over s_h}, {x_w + 2p_w -f_w + s_w \over s_w} \rbrack$
  - Mutiple channels
    - Input images have 3 channels
    - Define a kernel for each input channel
    - Weight is a 3D matrix
    - $z_{i,j} = \sum_H \sum_W \sum_C x_{si + u, sj+v, c} w_{u,v,c}$
  - In order to detect multiple features, extend the dimension of weight matrix
    - Weight is a 4D matrix
    - $z_{i,j,d} = \sum_H \sum_W \sum_C x_{si + u, sj+v, c} w_{u,v,c,d}$
    - Output is a hyper column formed by concatenation of feature maps
  - Special Case: (1x1) point wise convolution
    - Filter is of size 1x1.
    - Only the number of channels change from input to output
    - $z_{i,j,d} = \sum x_{i,j,c}w_{0,0,c,d}$
  - Pooling Layers
    - Convolution preserves information about location of input features i.e. equivariance
    - To achieve translational invariance, use pooling operation
    - Max Pooling
      - Maximum over incoming values
    - Average Pooling
      - Average over incoming values
    - Global Average Pooling
      - Convert the (H,W,D) feature maps into (1,1,D) output layer
      - Usually to compute features before passing to fully connected layer
  - Dilated Convolution
    - Convolution with holes
    - Takes every rth input (r is the dilation rate)
    - The filters have 0s
    - Increases the receptive field
  - Transposed Convolution
    - Produce larger output form smaller input
    - Pad the input with zeros and then run the filter
  - Depthwise

- Normalization - Vanishing / Exploding gradient issues in deeper models - Add extra layers to standardize the statistics of hidden units - Batch Normalization - Zero mean and unit variance across the samples in a minibatch - $\hat z_n = {z_n - \mu_b \over \sqrt{\sigma^2_b +\epsilon}}$ - $\tilde z_n = \gamma \hat z_n + \beta$ - $\gamma, \beta$ are learnable parameters - When applied to input layer, BN is close to unsual standardization process - For other layers, as model trains, the mean and variance change - Internal Covariate Shift - At test time, the inference may run on streaming i.e. one example at a time - Solution: After training, re-compute the mean and variance across entire training batch and then freeze the parameters - Sometimes, after recomputing, the BN parameters are fused to the hidden layer. This results in fused BN layer - BN struggles when batch size is small - Layer Normalization - Pool over channel, height and width - Match on batch index - Instance Normalization - Pool over height and width - Match over batch index - Normalization Free Networks - Adaptive gradient clipping
- Common Architectures - ResNet - Uses residula blocks to learn small perturbation in inputs - Residual Block: conv:BN:ReLU:conv:BN - Use padding, 1x1 convolution to ensure that additive operation is valid - DenseNet - Concatenate (rather than add) the output with the input - $x \rightarrow [x, f_1(x), f_2(x, f_1(x)), f_3(x, f_1(x), f_2(x))]$ - Computationally expensive - Neural Architecture Search - EfficeintNetV2
- Adversarial Exmaples
  - White-Box Attacks
    - Gradient Free
    - Add small perturbation to input that changes the prediction from classifier
    - Targeted attack
  - Black-Box Attack
    - Gradient Free
    - Design fooling images as apposed to adversarial images

## Chapter 15: Neural Networks for Sequences (RNN)

- RNN maps input sequences to output space in a stateful way
- Output y(t) not only depends on x(t) but also a hidden state h(t)
- Hidden state gets updated over time as the sequence is processed

- Vec2Seq (Sequence Generation)

  - Input is a vector
  - Output is a sequence of arbitrary length
  - Output sequence is generated one token at a time
    - $p(y_{1:T} | x) = \sum p(y_{1:T}, h_{1:T} | x)$
    - $p(y_{1:T} | x) = \sum \prod p(y_t | h_t) \times p(h_t | h_{t-1} , y_{t-1}, x)$
    - $p(y_t | h_t)$ can be:
      - Categorical
      - Gaussian
    - $h_t = \phi( W_{xh}[x;y_{t-1}] + W_{hh}h_{t-1} + b_h)$
      - W(x,h) are input to hidden weights
      - W(h,h) are hidden to hidden weights
  - RNNs can have unbounded memory unlike Markov models

- Seq2Vec (Sequence Classification)

  - Input is a variable length sequence
  - Output is a fixed dimension vector
  - For example: Classification Task
    - $p(y|x_{1:T}) = \text{Cat}(y|S(WH_T))$
  - Results can be improved if model can depend on both past and future context
    - Apply bidirectional RNN
    - $h^{\rightarrow} = \phi(W_{xh}^{\rightarrow}x_t + W_{hh}^{\rightarrow}h_t)$
    - $h^{\leftarrow} = \phi(W_{xh}^{\leftarrow}x_t + W_{hh}^{\leftarrow}h_t)$
    - Input to the linear layer is concatenation of the two hidden states

- Seq2Seq (Sequence Translation)

  - Input is a variable length sequence
  - Output is a variable length sequence
  - Aligned Case:
    - If input and output length are the same
    - One label prediction per step
    - $p(y_{1:T}|h_{1:T}) = \sum \prod p(y_t | h_t) I\{h_t = f(h_{t-1},x_t)\}$
  - Unaligned Case
    - If input and output length are not the same
    - Encoder-Decoder architecture
    - Encode the sequence to get the context vector
    - Generate the output sequence using the decoder
    - Teacher Forcing
      - While training the model, ground truth is fed to the model and not the labels generated by the model
      - Teacher’s values are force fed to the model
      - Sometimes results in poor test time performance
      - Scheduled Sampling
        - Start with teacher forcing
        - At regular intervals feed the samples generated from the model

- Backpropagation through Time (BPTT)

  - Unrolling the computation graph along time axis
  - $h_t = W_{hx}x_t + W_{hh}h_{t-1} = f(x_t, h_{t-1}, w_h)$
  - $o_t = W_{ho}h_t = g(h_t, w_{oh})$
  - $L = {1 \over T}\sum l(y_t, o_t)$
  - ${\delta L \over \delta w_h} = {1 \over T} \sum {\delta l \over \delta w_h}$
  - ${\delta L \over \delta w_h} = {1 \over T} \sum {\delta l \over \delta o_t} {\delta o_t \over \delta h_t} {\delta h_t \over \delta w_h}$
  - ${\delta h_t \over \delta w_h} = {\delta h_t \over \delta w_h} + {\delta h_t \over \delta h_{t-1}} {\delta h_{t-1} \over \delta w_h}$
  - Common to truncate the update to length of the longest subsequence in the batch
  - As the sequence goes forward, the hidden state keeps getting multiplied by W(hh)
  - Gradients can decay or explode as we go backwards in time
  - Solution is to use additive rather than multiplicative updates

- Gated Recurrent Units

  - Learn when to update the hidden state by using a gating unit
  - Update Gate: Selectively remember important pieces of information
    - $Z_t = \sigma(W_{xz} X_t + W_{hz} H_{t-1})$
  - Reset Gate: Forget things and reset the hidden state when information is no longer useful
    - $R_t = \sigma(W_{rx} X_t + W_{rh} H_{t-1})$
  - Candidate State
    - Combine old memories that are not reset
    - $\tilde H_t = \tanh ( W_{xh} X_t + W_{hh} R_t \times H_{t-1})$
    - If reset is close to 1, standard RNN
    - If reset close to 0, standard MLP
    - Captures new short term information
  - New State
    - $H_t = Z_t H_{t-1} + (1 - Z_t) \tilde H_t$
    - Captures long term dependecies
    - If Z is close to 1, the hidden state carries as is and new inputs are ignored

- Long Short Term Memory (LSTM)

  - More sophisticated version of GRU
  - Augment the hidden state with memory cell
  - Three gates control this cell
    - Input: $I_t = \sigma( W_{ix} X_t + W_{ih} H_{t-1})$, what gets read in
    - Output: $O_t = \sigma(W_{ox} X_t + W_{oh} H_{t-1})$, what gets read out
    - Forget: $F_t = \sigma (W_{fx} X_t + W_{fh} H_{t-1})$, when the cell is reset
  - Candidate Cell State
    - $\tilde C_t = \tanh ( W_{cx} X_t + W_{ch} H_{t-1})$
  - Actual Candidate:
    - $C_t = F_{t} \times C_{t-1} + I_t  \times  \tilde C_{t}$
  - Hidden State
    - $H_t = O_t \times \tanh(C_t)$
    - Both output and hidden state for next time step
    - Hence, captures short term memory
  - The memory cell state captures long term memory
  - Peephole Connections
    - Pass cell state as additional input to the gates
  - _How does LSTM solve vanishing gradients problem?_

- Decoding

  - Output is generated one token at a time
  - Simple Solution: Greedy Decoding
    - Argmax over vocab at each step
    - Keep sampling unless <EOS> token output
  - May not be globally optimal path
  - Alternative: Beam Search
    - Compute top-K candidate outputs at each step
    - Expand each one in V possible ways
    - Total VK candidates generated
  - GPT used top-k and top-p sampling
    - Top-K sampling: Redistribute the probability mass
    - Top-P sampling: Sample till the cumulative probability exceeds p

- Attention

  - In RNNs, hidden state linearly combines the inputs and then sends them to an activation function
  - Attention mechanism allows for more flexibility.
    - Suppose there are m feature vectors or values
    - Model decides which to use based on the input query vector q and its similarity to a set of m keys
    - If query is most similar to key i, then we use value i.
  - Attention acts as a soft dictionary lookup
    - Compare query q to each key k(i)
    - Retrieve the corresponding value v(i)
    - To make the operation differentiable:
      - Compute a convex combination
    - $Attn(q,(k_1,v_1),(k_2, v_2)...,(k_m,v_m)) = \sum_{i=1}^m \alpha_i (q, \{k_i\}) v_i$
      - $\alpha_i (q, \{k_i\})$ are the attention weights
    - Attention weights are computed from an attention score function $a(q,k_i)$
      - Computes the similarity between query and key
    - Once the scores are computed, use soft max to impose distribution
    - Masking helps in ignoring the index which are invalid while computing soft max
    - For computational efficiency, set the dim of query and key to be same (say d)
      - The similarity is given by dot product
      - The weights are randomly initialized
      - The expected variance of dot product will be d.
      - Scale the dot product by $\sqrt d$
      - Scaled Dot-Product Attention
        - Attention Weight: $a(q,k) = {q^Tk \over \sqrt d}$
        - Scaled Dot Product Attention: $Attn(Q,K,V) =  S({QK^T \over \sqrt d})V$
    - Example: Seq2Seq with Attention
      - Consider encoder-decoder architecture
      - In the decoder:
        - $h_t = f(h_{t-1}, c)$
        - c is the context vector from encoder
        - Usually the last hidden state of the encoder
      - Attention allows the decoder to look at all the input words
        - Better alignment between source and target
      - Make the context dynamic
        - Query: previous hidden state of the decoder
        - Key: all the hidden states from the encoder
        - Value: all the hidden states from the encoder
        - $c_t = \sum_{i=1}^T \alpha_i(h_{t-1}^d, \{h_i^e\})h_i^e$
      - If RNN has multiple hidden layers, usually take the top most layer
      - Can be extended to Seq2Vec models

- Transformers

  - Transformers are seq2seq models using attention in both encoder and decoder steps
  - Eliminate the need for RNNs
  - Self Attention:
    - Modify the encoder such that it attends to itself
    - Given a sequence of input tokens $[x_1, x_2, x_3...,x_n]$
    - Sequence of output tokens: $y_i = Attn(x_i, (x_1,x_1), (x_2, x_2)...,(x_n, x_n))$
      - Query is xi
      - Keys and Values are are x1,x2…xn (all valid inputs)
    - In the decoder step:
      - $y_i = Attn(y_{i-1}, (y_1,y_1), (y_2, y_2)...(y_{i-1}, y_{i-1}))$
      - Each new token generated has access to all the previous output
  - Multi-Head Attention
    - Use multiple attention matrices to capture different nuances and similarities
    - $h_i = Attn(W_i^q q_i, (W_i^k k_i, W_i^v v_i))$
    - Stack all the heads together and use a projection matrix to get he output
    - Set $p_q h = p_k h = p_v h = p_o$ for parallel computation \*\*How?
  - Positional Encoding
    - Attention is permutation invariant
    - Positional encodings help overcome this
    - Sinusoidal Basis
    - Positional Embeddings are combined with original input X → X + P
  - Combining All the Blocks
    - Encoder
      - Input: $ Z = LN(MHA(X,X,X) + X$
      - Encoder: $E = LN(FF(Z) + Z)$
        - For the first layer:
          - $ Z = \text{POS}(\text{Embed}(X))$
    - In general, model has N copies of the encoder
    - Decoder
      - Has access to both: encoder and previous tokens
      - Input: $ Z = LN(MHA(X,X,X) + X$
      - Input $ Z = LN(MHA(Z,E,E) + Z$

- Representation Learning
  - Contextual Word Embeddings
    - Hidden state depends on all previous tokens
    - Use the latent representation for classification / other downstream tasks
    - Pre-train on a large corpus
    - Fine-tune on small task specific dataset
    - Transfer Learning
  - ELMo
    - Embeddings from Language Model
    - Fit two RNN models
      - Left to Right
      - Right to Left
    - Combine the hidden state representations to fetch embedding for each word
  - BERT
    - Bi-Directional Encoder Representations from Transformers
    - Pre-trained using Cloze task (MLM i.e. Masked Language Modeling)
    - Additional Objective: Next sentence Prediction
  - GPT
    - Generative Pre-training Transformer
    - Causal model using Masked Decoder
    - Train it as a language model on web text
  - T5
    - Text-to-Text Transfer Transformer
    - Single model to perform multiple tasks
    - Tell the task to perform as part of input sequence

## Chapter 16: Exemplar-based Methods

- Non-parametric Models

  - Keep the training data around
  - Effective number of model parameters grow with |D|

- Instance-based Learning

  - Models keep training examples around test time
  - Define similarity between training points and test input
  - Assign the label based on the similarity

- KNN

  - Classify the ew input based on K closest examples in the training set
  - $p(y = c | x, D) = {1 \over K} \sum  I\{y=c\}$
  - The closest point can be computed using Mahalanobis Distance
  - $d_M(x,\mu) = \sqrt{(x-\mu)^TM(x-\mu)}$
  - M is positive definite matrix
  - If M = I, then distance reduces to Euclidean matrix

- Curse of Dimensionality

  - Space volume grows exponentially with increase in dimension
    - Suppose inputs are uniformly distributed
    - As we move from square to cube, 10% edge covers less region

- Speed and Memory Requirements

  - Finding K nearest neighbors slow
  - KD Tree / LSH to speed up approximate neighbor calculation
    - KD Tree:
      - K dimensional binary search tree
    - LSH: Similar objects go to same hash bucket
      - Shingling, Minhash, LSH

- Open Set recognition

  - New classes appear at test time
    - Person Re-identification
    - Novelty Detection

- Learning Distance Matrix

  - Treat M is the distance matrix as a parameter
  - Large Margin Nearest Neighbors
  - Find M such that
    - $M = W^T W$ (Positive Definite)
    - Similar points have minimum distance
    - Dissimilar points are at least m units away (margin)

- Deep Metric Learning

  - Reduce the curse of dimensionality
  - Project he input from high dimension space to lower dimension via embedding
  - Normalize the embedding
  - Compute the distance
    - Euclidean or Cosine, both are related
    - $|e_1 - e_2|^2 = |e1|^2 + |e_2|^2 - 2e_1 e_2$
    - Euclidean = 2 ( 1 - Cosine)
    - $\cos \theta = {a \dot b \over ||a|| ||b||}$
    - Derivation via trigonometry
      - $\ cos \theta = a^2 + b ^ 2 - c^2 / 2 a b$
  - Learn an embedding function such that similar examples are close and dissimar examples are far
  - Loss functions:
    - Classification Losses
      - Only learn to push examples on correct side of the decision boundary
    - Pairwise Loss
      - Simaese Neural Network
      - Common Backbone to embed the inputs
      - $L(\theta, x_i, x_j) =  I \{y_i =y_j\} d(x_i, x_j) +  I \{y_i \ne y_j\} [m - d(x_i, x_j)]_+$
      - If same class, minimize the distance
      - If different class, maximize the distance with m margin (Hinge Loss)
    - Triplet Loss
      - In Pairwise Loss: positive and negative examples siloed
      - $L(\theta, x_i, x^+, x^-) = [m + d(x_i, x_+) - d(x_i, x_-)]_+$
      - Minimize the distance between anchor and positive
      - Maximize the distance between anchor and negative
      - m is the safety margin
      - Need to mine hard negative examples that are close to the positive pairs
      - Computationally slow
      - Use proxies to represent each class and speed up the training

- Kernel Density Estimation
  - Density Kernel
    - Domain: R
    - Range: R+
    - $\int K(x)dx = 1$
    - $K(-x) = K(x)$
    - $\int x K(x-x_n) dx = x_n$
  - Gaussian Kernel
    - $K(x) = {1 \over \sqrt{2\pi}} \exp(-{1\over2}x^2)$
    - RBF: Generalization to vector valued inputs
  - Bandwitdth
    - Parameter to control the width of the kernel
  - Density Estimation
    - Extend the concept of Gaussian Mixture models to the extreme
    - Each point acts as an individual cluster
      - Mean $x_n$
      - Constant variance
      - No covariance
      - Var-Cov matrix is $\sigma^2 I$
      - $p(x|D) = {1 \over N}\sum K_h(x - x_n)$
      - No model fitting is required
  - KDE vs KNN
    - KDE and KNN are closely related
    - Essentially, in KNN we grow the volume around a point till we encounter K neighbors.

## Chapter 18: Trees, Forests, Bagging, and Boosting

- Recursively partition the input space and define a local model in the resulting region of the input space

  - Node i
  - Feature dimension d_i is compared to threshold t_i
    - $R_i = \{x : d_1 \le t1, d_2 \le t_2\}$
    - Axis parallel splits
  - At leaf node, model specifies the predicted output for any input that falls in the region
    - $w_1 = {\sum_{N} y_n  I \{x \in R_1\} \over \sum_{N}  I \{x \in R_1\}}$
  - Tree structure can be represented as
    - $f(x, \theta) = \sum_J w_j  I\{x \in R_j\}$
    - where j denotes a leaf node

- Model Fitting

  - $L(\theta) = \sum_J \sum_{i \in R_j} (y_i, w_j)$
  - The tree structure is non-differentiable
  - Greedy approach to grow the tree
  - C4.5, ID3 etc.
  - Finding the split
    - $L(\theta) = {|D_l \over |D|} c_l + {|D_r \over |D|} c_r$
    - Find the split such that the new weighted overall cost after splitting is minimized
    - Looks for binary splits because of data fragmentation
  - Determining the cost
    - Regression: Mean Squared Error
    - Classification:
      - Gini Index: $\sum \pi_ic (1 - \pi_ic)$
      - $\pi_ic$ probability that the observation i belongs to class c
      - $1 - \pi_ic$ probability of misclassification
      - Entropy: $\sum \pi_{ic} \log \pi_{ic}$
  - Regularization
    - Approach 1: Stop growing the tree according to some heuristic
      - Example: Tree reaches some maximum depth
    - Approach 2: Grow the tree to its maximum possible depth and prune it back
  - Handling missing features
    - Categorical: Consider missing value as a new category
    - Continuous: Surrogate splits
      - Look for variables that are most correlated to the feature used for split
  - Advantages of Trees
    - Easy to interpret
    - Minimal data preprocessing is required
    - Robust to outliers
  - Disadvantages of Trees
    - Easily overfit
    - Perform poorly on distributional shifts

- Ensemble Learning

  - Decision Trees are high variance estimators
  - Average multiple models to reduce variance
  - $f(y| x) = {1 \over M} \sum f_m (y | x)$
  - In case of classification, take majority voting
    - $p = Pr(S > M/2) = 1 - \text{Bin}(M, M/2, \theta)$
    - Bin(.) if the CDF of the binomial distribution
    - If the errors of the models are uncorrelated, the averaging of classifiers can boost the performance
  - Stacking
    - Stacked Generalization
    - Weighted Average of the models
    - $f(y| x) = {1 \over M} \sum w_m f_m (y | x)$
    - Weights have to be learned on unseen data
    - Stacking is different from Bayes averaging
      - Weights need not add up to 1
      - Only a subset of hypothesis space considered in stacking

- Bagging

  - Bootstrap aggregation
  - Sampling with replacement
    - Start with N data points
    - Sample with replacement till N points are sampled
    - Probability that a point is never selected
      - $(1 - {1 \over N})^N$
      - As N → $\infty$, the value is roughly 1/e (37% approx)
  - Build different estimators of these sampled datasets
  - Model doesn’t overly rely on any single data point
  - Evaluate the performance on the 37% excluded data points
    - OOB (out of bag error)
  - Performance boost relies on de-correlation between various models
    - Reduce the variance is predictions
    - The bias remains put
    - $V = \rho \sigma ^ 2 + {(1 - \rho) \over B} \sigma ^2$
    - If the trees are IID, correlation is 0, and variance is 1/B
  - Random Forests
    - De-correlate the trees further by randomizing the splits
    - A random subset of features chosen for split at each node
    - Extra Trees: Further randomization by selecting subset of thresholds

- Boosting

  - Sequentially fitting additive models
    - In the first round, use original data
    - In the subsequent rounds, weight data samples based on the errors
      - Misclassified examples get more weight
  - Even if each single classifier is a weak learner, the above procedure makes the ensemble a strong classifier
  - Boosting reduces the bias of the individual weak learners to result in an overall strong classifier
  - Forward Stage-wise Additive Modeling
    - $(\beta_m, \theta_m) = \arg \min \sum l(y_i, f_{m-1}(x_i, \theta_{m-1}) + \beta_m F_m(x_i, \theta))$
    - $f_m(x_i, \theta_m) = f_{m-1}(x_i, \theta_{m-1}) + \beta_m F_m(x_i, \theta_m)$
  - Example: Least Square Regression
    - $l(y_i, f_{m-1}(x_i) + \beta_m F_m(x_i)) =  (y_i - f_{m-1}(x_i) - \beta_m F_m(x_i))^2$
    - $l(y_i, f_{m-1}(x_i) + \beta_m F_m(x_i)) = (r_im - \beta_m F_m(x_i))^2$
    - Subsequent Trees fit on the residuals from previous rounds
  - Example: AdaBoost
    - Classifier that outputs {-1, +1}
    - Loss: Exponential Loss
      - $p(y=1|x) = {\exp F(x) \over \exp -F(x) + \exp F(x)}$
      - $l(y_i, x_i) = \exp(- \tilde y F(x_i))$
    - $l_m = \sum \exp ( - \tilde y_i f_{m-1} (x_i) - \tilde y_i \beta F_m(x_i)) = \sum w_{im} \exp (- \tilde y_i \beta F_m(x_i))$
    - $l_m = \exp^{-\beta} \sum_{\tilde y = F(x)} w_{im} + \exp^\beta \sum_{\tilde y != F(x)} w_{im}$
    - $F_m = \arg \min \sum w_{im}  I\{y_i \ne F(x)\}$
    - Minimize the classification error on re-weighted dataset
    - The weights are exponentially increased for misclassified examples
    - LogitBoost an extension of AdaBoost
      - Newton update on log-loss

- Gradient Boosting

  - No need to derive different algorithms for different loss functions
  - Perform gradient descent in the space of functions
  - Solve for: $ f = \arg \min L(f)$
    - Functions have infinite dimensions
    - Represent them by their values on the training set
    - Functon: $f = (f(x_1), f(x_2)...,f(x_n))$
    - Gradient: $g_{im} = [ {\delta l(y_i, f(x_i)) \over \delta f(x_i)}]$
    - Update: $f_m = f_{m-1} - \beta_m g_m$
  - In the current form, the optimization is limited to the set of training points
  - Need a function that can generalize
  - Train a weak learner that can approximate the negative gradient signal
    - $F_m = \arg\min \sum (-g_m -F(x_i))^2$
    - Use a shrinkage factor for regularization
  - Stochastic Gradient Boosting
    - Data Subsampling for faster computation and better generalization

- XGBoost

  - Extreme Gradient Boosting
  - Add regularization to the objective
  - $L(f) = \sum l(y_i, f(x_i)) + \Omega(f)$
  - $\Omega(f) = \gamma J + {1 \over 2} \lambda \sum w_j^2$
  - Consider the forward stage wise additive modeling
  - $L_m(f) = \sum l(y_i, f_{m-1}(x_i) + F(x_i)) + \Omega(f)$
  - Use Taylor’s approximation on F(x)
  - $L_m(f) = \sum l(y_i, f_{m-1}(x_i)) + g_{im} F_m(x_i) + {1 \over 2} h_{im} F_m(x_i)^2) + \Omega(f)$
    - g is the gradient and h is the hessian
  - Dropping the constant terms and using a decision tree form of F
  - $F(x_{ij}) = w_{j}$
  - $L_m = \sum_j (\sum_{i \in I_j} g_{im}w_j) + (\sum_{i \in I_j} h_{im} w_j^2) + \gamma J + {1 \over 2} \lambda \sum w_j^2$
  - Solution to the Quadratic Equation:
    - $G_{jm} = \sum_{i \in I_j} g_{im}$
    - $H_{jm} = \sum_{i \in I_j} h_{im}$
    - $w^* = {- G \over H + \lambda}$
    - $L(w^*) = - {1 \over 2} \sum_J {G^2_{jm} \over H_{jm} + \lambda} + \gamma J$
  - Condition for Splitting the node:
    - $\text{gain} = [{G^2_L \over H_L + \lambda} + {G^2_R \over H_R + \lambda} - {G^2_L + G^2_R \over H_R + H_L + \lambda}] - \gamma$
    - Gamma acts as regularization
    - Tree wont split if the gain from split is less than gamma

- Feature Importance

  - $R_k(T) = \sum_J G_j  I(v_j = k)$
  - G is the gain in accuracy / reduction in cost
  - I(.) returns 1 if node uses the feature
  - Average the value of R over the ensemble of trees
  - Normalize the values
  - Biased towards features with large number of levels

- Partial Dependency Plot
  - Assess the impact of a feature on output
  - Marginalize all other features except k

## Chapter 19: Learning with Fewer Labeled Examples

- Data Augmentation
  - Artificially modified versions of input vectors that may appear in real world data
  - Improves accuracy, makes model robust
  - Empirical risk minimization to vicinical risk minimization
  - Minimizing risk in the vicinity of input data point

-Transfer Learning

- Some data poor tasks may have structural similarity to other data rich tasks
- Transferring information from one dataset to another via shared parameters of a model
- Pretrain the model on a large source dataset
- Fine tune the model on a small target dataset
- Chop-off the the head of the pretrained model and add a new one
- The parameters may be frozen during fine-tuning
- In case the parameters aren't frozen, use small learning rates.
- Adapters

  - Modify the model structure to customize feature extraction
  - For example: Add MLPs after transformer blocks and initialize them for identity mappings
  - Much less parameters to be learned during fine-tuning

- Pre-training

  - Can be supervised or unsupervised.
  - Supervised
    - Imagenet is supervised pretraining.
    - For unrelated domains, less helpful.
    - More like speedup trick with a good initialization.
  - Unsupervised
    - Use unlabeled dataset
    - Minimize reconstruction error
  - Self-supervised
    - Labels are created from ulabeled dataset algorithmically
    - Cloze Task
      - Fill in the blanks
    - Proxy Tasks
      - Create representations
      - Siamese Neural Networks
      - Capture relationship between inputs
    - Contrastive Tasks
      - Use data augmentation
      - Ensure that similar inputs have closer representations
  - SimCLR
    - Simple Contrastive Learning for Visual Representations
    - Generate two augmented views (random crops)
    - Maximize the similarity of similar views and minimize the similarity of different views
    - Use large batch training to ensure diverse set of negatives
    - Forces the model to learn and focus on local features / views
  - Clip
    - Contrastive Language-Image Pretraining
    - Image with matching text
    - Maximize the similarity of the image embedding to that of the embedding of the matching text
    - Works well in case of zero-shot learning
    - Requires prompt engineering to convert image metadata to labels for text embeddings

- Domain Adaption

  - Different domains but same output labels
  - Example - Product reviews and movie reviews
  - Domain adversarial learning
  - Auxiliary task: Model has to learn the source of the input
  - Minimize the loss on desired classification task
  - Maximize the loss on auxiliary task
  - Gradient reversal trick

- Semi-Supervised Learning

  - Leverage large amount of unlabeled data
  - Learn high level structure of data distribution from unlabeled data
  - Learn fine-grained details of given task using labeled data
  - Avoid labeling all of the dataset
  - Works on clusterassumption.
    - A good decision boundary will not pass through data dense regions
  - Pseudo Labeling / Self-Training
    - Use the model itself to infer predictions on unlabeled data
    - Treat the predictions as labels for subsequent training
    - Inferred labels are pseudo-correct in comparison to ground truth
    - Suffers from Confirmation Bias
    - If the original predictions are wrong, model becomes progressievely worse
    - Use soft labels from model predicitons and scaled using softmax temperture
  - Co-training
    - Two complementary views (say two sets of features) for each data point
    - Train two models independently
    - Score the unlabled data point using each of the models
    - If model scores high from one model and low from another - add it to the training dataset for the low scoring model
    - Repeat until convergence
  - Label Propagation
    - Transductive learning
    - Manifold Assumption: Similar data points should share the same label
    - Construct a graph of the dataset
      - Nodes: Data points
      - Edges: Similarity between the data points
      - Model Labels: Labels of the data points
    - Propagate the labels in such a way that there is minimal label disagreement between node and it's neighbours
    - Label guesses for unlabeled data that can be used for superised learning
    - Details:
      - M labeled points, N unlabeled points
      - T: (M+N) x (M+N) transition matrix of normalized edge weights
      - Y: Label matrix for class distribution of (M+N) x C dimension
      - Use transition matrix to propagate labels Y = TY until convergence
    - Success depends on calculating similarity between data points
  - Consistency Regularization
    - Small perturbation to input data point should not change the model predicitons

- Generative Models

  - Natural way of using unlabeled data by learning a model of data generative process.
  - Variational Autoencoders
    - Models joint distribution of data (x) and latent variables (z)
    - First sample: $z \sim p(z)$ and then sample $x\sim p(x|z)$
    - Encoder: Approximate the posterior
    - Decoder: Approximate the likelihood
    - Maximize evidence lower bound of the data (ELBO) (derived from Jensen's ineuqlity)
    - Use VAEs to learn representations for downstream tasks
  - Generative Adversarial Netwworks
    - Generator: Maps latent distribution to data space
    - Discriminator: Distinguish between outputs of generator and true distribution
    - Modify discriminator to predict class labels and fake rather than just fake

- Active Learning

  - Identify true predictive mapping by quering as few data points as possible
  - Query Synthesis: Model asks output for any input
  - Pool Based: Model selects the data point from a pool of ulabeled data points
  - Maximum Entropy Sampling
    - Uncertainty in predicted label
    - Fails when examples are ambiguous of mislabeled
  - Bayesian Active Learning by Disagreement (BALD)
    - Select examples where model makes predictions tht are highly diverese

- Few-Shot Learning

  - Learn to predict from very few labeled example
  - One-Shot Learning: Learn to predict from single example
  - Zero-Shot Lerning: Learn to predict without labeled examples
  - Model has to generalize for unseen labels during traning time
  - Works by learning a distance metric

- Weak Supervision
  - Exact label not aviabale for data points
  - Distribution of labels for each case
  - Soft labels / label smoothing

## Chapter 22: Recommender Systems

- Feedback

  - Explicit
    - Rating, Like or Dislike from users
    - Sparse, Values missing at at random
  - Implicit
    - Monitor user actions - click vs no-click, watch vs skip etc.
    - Sparse positive-only ratings matrix

- Collaborative Filtering

  - Users collaborate on recmmending items
  - Impute values by looking at how other similar users have rated the item
  - $\hat Y_{ui} = \sum sim(u_i, u') Y_{u', i}$
  - Typical approach to calculate similarity is to compare the available ratings from both users
  - Data Sprsity makes it challenging
  - Matrix Factorization
    - View the problem as that of matrix completion
    - Predict all the missing entries of the ratings matrix
    - Optimization: $L = ||Z - Y||^2$
    - Break up Z into low rank matrices: $Z = U^TV$
    - U is the matrix corresponding to users, V is the matrix corresponding to items
    - Since Y is incomplete, Z can't be found using SVD. (unless missing imputation)
    - Use ALS (Alternating Least Squares). Estimate U given V and V given U.
    - Add user-specific and item specific biases
    - $\hat y_{ui} = \mu + b_u + c_i + u_u^T v_i$
    - $L = \sum (y_{ui} - \hat y_{ui})^2$
  - Probabilistic Matrix Factorization
    - Convert the ratings to a probabilistic model
    - $p(Y=y) = N(\mu + b_u + c_i + u_u^T v_i, \sigma^2)$

- Bayesian Personalized Ranking

  - Implicit Feedback
  - Ranking Loss: Model ranks item i (positive set) ahead of j (negative set) for user u
    - $p(y = (u,i,j) | \theta) =  \sigma(f(u,i;\theta) - f(u,j;\theta))$
  - Use hinge-loss to estimate the parameters

- Factorization Machines

  - Predict the rating for each user-item pair using one-hot encodings
    - $x = \text{concat}[user, item]$
  - $f(x) = \mu + \sum w_i x_i + \sum \sum (v_i v_j) x_i x_j$
  - The dot product captures the interactions between users, items and user-items.
  - Using a low-rank matrix it reduces the number of parameters to be estimated.
  - Adding contextual information to solve for cold-start is straight forward.
  - If explicit feedback, use MSE Loss.
  - If implicit feedback, use Ranking Loss.

- Cold-Start Problem

  - Difficult to generate predictions for new user or new item.
  - Leverage side information. Item and user metadata.

- Exploration-Exploitation Tradeoff
  - Counterfactual - Is there an item the user would have liked but wasn't shown?
  - Recommend items for which response is uncertain.
