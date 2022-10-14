---
author: "Vien Vuong"
title: "A Deep Dive Into Meta-Learning [Ongoing]"
date: "2022-09-13"
description: "Meta-learning in ML is used to improve the results and performance of a learning algorithm by changing some aspects of the learning algorithm based on experiment results. This guide explores almost every sub-domain of meta-learning by summarizing at least 3 seminal papers per sub-domain."
tags: ["meta-learning", "ml"]
comments: false
socialShare: false
toc: true
math: true
---

Meta learning, also known as “learning to learn”, is a subset of machine learning in computer science. It is used to improve the results and performance of a learning algorithm by changing some aspects of the learning algorithm based on experiment results. Meta learning helps researchers understand which algorithm(s) generate the best/better predictions from datasets.

Machine learning algorithms have some challenges such as:

- Need for large datasets for training
- High operational costs due to many trials/experiments during the training phase
- Experiments/trials take long time to find the best model which performs the best for a certain dataset.

Meta learning can help machine learning algorithms to tackle these challenges by optimizing learning algorithms and finding learning algorithms which perform better.

This guide explores almost every sub-domain of meta-learning by summarizing at least 3 seminal papers per sub-domain.

## Metric Learning

### Prototypical networks for few shot learning (NeurIPS, 2017)

**Authors:** J. Snell, K. Swersky, and R. S. Zemel

### Matching networks for one shot learning (NeurIPS, 2016)

**Authors:** O. Vinyals, C. Blundell, T. Lillicrap, K. Kavukcuoglu, and D. Wierstra

### Meta-learning with differentiable convex optimization (CVPR, 2019)

**Authors:** K. Lee, S. Maji, A. Ravichandran, and S. Soatto

## Parameter Initialization

### Model-agnostic meta-learning for fast adaptation of deep networks (ICML, 2017)

**Authors:** C. Finn, P. Abbeel, and S. Levine

This paper (MAML) addresses a variety of problems associated with few-shot transfer learning and multi-task learning. Classical pre-training trains an optimal policy over the distribution of tasks, which is then fine-tuned over a specific dataset for a specific task. There are two problems. First, the amount of fine-tuning data is still large. Second, the pre-trained initial parameters are optimized to maximize the average return over all tasks, which does not guarantee fast adaptation to fine-tuning. MAML aims to instead optimize for models that are easy and fast to fine-tune. In other words, MAML can be viewed as maximizing the sensitivity of the loss functions of new tasks with respect to the parameters.

MAML achieves this through a two-step process:

1. For a given set of tasks, we sample multiple data point (or trajectories) using θ and update the parameter using one (or multiple) gradient step(s). This is the meta-optimization step, and is called the "inner loop".
2. For the same tasks, we sample multiple trajectories from the updated parameters θ’, calculate the gradient, and backpropagate to θ. This is called the "outer loop," and it directly updates the objective.

MAML is optimized so that one or small number of gradient steps will produce maximally effective behavior on a certain task. This solves both the data problem (not many examples required), and the adaptivity problem (by design).

MAML achieves very good results over a variety of tasks including supervised regression and classification, and reinforcement learning. In many classification and regression tasks, MAML achieves results comparable to SOTA at the time (2017), and consistently beat classic pre-trained models in few-shot learning. MAML also showed to perform well with a variety of models.

MAML is also relatively simple, and does not introduce any additional learned parameters like previous meta-learning approaches.

The MAML meta-gradient update involves a gradient through a gradient, which requires computing an expensive Hessian (2nd-order derivative). Fortunately, the paper offers an more efficient alternative called First-Order MAML (FOMAL) that also achieves very good result.

MAML still requires you to do a full gradient descent step over all the weights/parameters, which can be expensive. This full GD step is also run over only a few data points, which can lead to overfitting. MAML also does not fully solve the problem of maximizing the average return over all tasks, as it does not distinguish between task-specific and task-independent neural net components (parameters).

### Probabilistic model-agnostic meta-learning (NeurIPS, 2018)

**Authors:** C. Finn, K. Xu, and S. Levine

### Meta-learning with latent embedding optimization (ICLR, 2019)

**Authors:** A. A. Rusu, D. Rao, J. Sygnowski, O. Vinyals, R. Pascanu, S. Osindero, and R. Hadsell

## Parameter and Hyperparameter Optimization

### Learning to learn by gradient descent by gradient descent (NeurIPS, 2016)

**Authors:** M. Andrychowicz, M. Denil, S. G. Colmenarejo, M. W. Hoffman, D. Pfau, T. Schaul, and N. de Freitas

The seminal paper ‘No Free Lunch Theorems for Optimization’ tells us that no general-purpose optimisation algorithm can dominate all others. So to get the best performance, we need to match our optimisation technique to the characteristics of the problem at hand. This realization leads to a variety of hand-designed deep-learning optimization update rules, such as momentum, Rprop, Adagrad, RMSprop, and ADAM. However, it takes a long time to develop these algorithms that perform well only on certain problems. This paper, "Learning to learn by gradient descent by gradient descent," attempts to replicate the success with learned feature representation to the optimization step, developing a model that learns how to best optimize the objective function on any particular class of optimization problem. Specifically, a 2-layer LSTM with a forget-gate architecture is used as the learned optimizer. To keep the optimizer small and fast, the LSTM optimizer operates coordinate-wise on the parameters of the objective function. In addition to the single coordinate, the LSTM optimizer also takes in the previous hidden state and outputs the update for the corresponding optimise parameter. The paper outlines a variety of benchmarks, including approximating 10-dimensional quardratic functions, classifying MNIST and CIFAR-10, and generating neural arts. In all of these benchmark, the LSTM optimizer dominates state-of-the-art optimization methods in both convergence rate and accuracy. Not only do these learned optimizers perform very well, they also display great potential for transfer learning. According to the authors, "[...] the LSTM optimizer trained on 12,288 parameter neural art tasks being able to generalize to tasks with 49,512 parameters, different styles, and different content images all at the same time. We observed similar impressive results when transferring to different architectures in the MNIST task." In conclusion, this paper marks a crossover point where another class of learned algorithms, in this case optimizers, once again outperform those of the best human designers.

Using a neural network, especially an LSTM, as an optimizer is much more computationally expensive than nonparametric 1st-order methods such as ADAM or RMSprop. This time-expensive property means that the LSTM optimizer cannot be used for real-time applications (e.g., automated driving). However, the idea of the paper can extend to other kinds of parameterized functions that are much faster and lighter than neural nets. The results presented in the paper, while impressive, are also quite contrived and does not cover meta-learning scenarios extensively.

### Optimization as a model for few-shot learning (ICLR, 2016)

**Authors:** S. Ravi and H. Larochelle

### Gradient-based hyperparameter optimization through reversible learning (ICML 2015)

**Authors:** D. Maclaurin, D. Duvenaud, and R. Adams

## Parameter Prediction

### HyperNetworks (ICLR, 2017)

**Authors:** D. Ha, A. Dai, and Q. V. Le

If done properly, weight-sharing can reduce the size of a neural network drastically while retaining its performance characteristics. However, most state-of-the-art architectures at the time (2016) falls in either one extreme or another. Specifically, recurrent networks (RNNs) allow layers to have the same weights (weight-tying), while convolutional nets (CNNs) do not permit any weight-sharing between layers at all. This paper explores a middle ground between the two - to enforce a relaxed version of weight-tying. A HyperNetwork is just a small network that generates the weights of a much larger network, effectively parameterizing the weights of each layer of the larger network. This paper also explores the tradeoff between the model’s expressivity versus how much we tie the weights, i.e., how much we can compress the parameters. However, the greatest contribution of this paper by far is in the context of RNNs. It experiments with allowing the weights of an RNN to be different at each time step (like a CNN), and also for each individual input sequence. The paper embeds a small LSTM cell (called the HyperLSTM) into each standard LSTM cell whose weights are generated by the HyperLSTM. Impressively, the HyperNetwork-RNN hybrid is able to achieve state-of-the-art results on two highly competitive benchmarks: Character-Level Penn Treebank and Hutter Prize Wikipedia.

Since the paper's main focus is in the exploration of parameter generation, the SOTA result was easily surpassed not too long after. However, the ability to "generate a generative model" remains an extremely important contribution to the field. While the result shown by dynamic hypernetworks used with RNNs are very impressive, the author was not able to achieve to the same success with static hypernetworks used with CNNs. As a result, CNNs are de-prioritized despite being the inspiration in the first place.

### Dynamic few-shot visual learning without forgetting (CVPR, 2018)

**Authors:** S. Gidaris and N. Komodakis

### From red wine to red tomato: Composition with context (CVPR, 2017)

**Authors:** I. Misra, A. Gupta, and M. Hebert

## Modular Meta-learning

### Modular meta-learning (CoRL, 2018)

**Authors:** F. Alet, T. Lozano-Perez, and L. P. Kaelbling

### Learning multiple visual domains with residual adapters (NeurIPS, 2017)

**Authors:** S.-A. Rebuffi, H. Bilen, and A. Vedaldi

### A universal representation transformer layer for few-shot image classification (arXiv:2006.11702, 2020)

**Authors:** L. Liu, W. Hamilton, G. Long, J. Jiang, and H. Larochelle

The paper addresses the problem of multi-domain few-shot image classification (where unseen classes and examples come from diverse data sources), and proposes a Universal Representation Transformer (URT) layer, which learns to transform a universal representation into task-adapted representations. The method proposed builds on top of SUR [Dvornik et al 2020], where a universal representation is extracted from the outputs of a collection of pre-trained and domain-specific backbones and a selection procedure infers how to weight each backbone for a given task at hand. While SUR inferred those weights by optimising a loss on the support set (the few examples provided in a task), the authors in this paper introduce an attention-based layer (inspired by Vaswani et al Transformer) that learns to weight the appropriate backbones for each task. This layer has the main advantage that it can be learned across few-shot tasks from many domains so it can support transfer across these tasks.

The method and contributions are very well motivated and introduced. The paper is also very well written and very well presented. This new proposed URT layer is a very interesting and novel contribution for this specific task. The experimental section is good, which includes comparison with other state-of-the-art methods and an ablation study that analyses the contribution of the different components of the proposed approach. Section 4.3 is particularly interesting as the attention scores produced by the network are visualised on the test tasks, which gives a better understanding of how this URT layer works.

The URT architecture in this paper is very similar to SUR, the only difference and novelty being the way the weights for the different backbones are computed. Thus, it would have been interesting to see how does SUR compare to URT with a single head, specially since the performance gap is quite significative from 1 to 2 layers. First, because it would give a deeper insight about the contribution of the different components of URT (attention layer vs multi-head). Second, because that’d be a bit more fair comparison between SUR and URT given that SUR only uses a single representation head: two heads means double dimensionality of the representation, and multi-head could also be applied to SUR using a similar approach.

## Architecture Search

### Neural architecture search with reinforcement learning (ICLR, 2017)

**Authors:** B. Zoph and L. Quoc

Although most popular and successful model architectures are designed by human experts, it doesn’t mean we have explored the entire network architecture space and settled down with the best option. We would have a better chance to find the optimal solution if we adopt a systematic and automatic way of learning high-performance model architectures. This paper by Zoph & Le is one of the pioneering works in the field of Neural Architecture Search (NAS), leading to many interesting ideas for better, faster and more cost-efficient NAS methods.

The initial design of NAS involves a RL-based controller for proposing child model architectures for evaluation. The controller is implemented as a RNN, outputting a variable-length sequence of tokens used for configuring a network architecture. The controller is trained as a RL task using REINFORCE where:

- The action space is a list of tokens for defining a child network predicted by the controller
- The accuracy of a child network that can be achieved at convergence is the reward for training the controller.
- NAS optimizes the controller parameters with a REINFORCE loss. We want to maximize the expected reward (high accuracy) with the gradient. The nice thing here with policy gradient is that it works even when the reward is non-differentiable.

The authors apply this approach to evolve a convolutional neural network on CIFAR-10 and a recurrent neural network cell on Penn Treebank. Impressively, the generated model is able to chieve SOTA on the Penn Treebank dataset and almost SOTA on CIFAR-10 with a smaller and faster network. The cell found on Penn Treebank also beats LSTM baselines on other language modeling datasets and on machine translation

While the paper was groundbreaking at the time of publish, it certainly has some flaws that has been improved on by subsequent papers. The drawbacks include a very long training time for the "parent" generator model, even with a lot of computing resources. The experiments also provided some, but not extensive, data on the generality of the generated architectures.

### DARTS: Differentiable architecture search (ICLR, 2019)

**Authors:** H. Liu, K. Simonyan, and Y. Yang

### Smash: one-shot model architecture search through hypernetworks (ICLR, 2018.)

**Authors:** A. Brock, T. Lim, J. Ritchie, and N. Weston

## Data Selection & Reweighting in Meta-learning

### Mentornet: Learning data-driven curriculum for very deep neural networks on corrupted labels (ICML, 2018)

**Authors:** L. Jiang, Z. Zhou, T. Leung, L.-J. Li, and L. Fei-Fei

Problem: Modern networks are capable of memorizing the entire dataset. This becomes a problem when the dataset contains corrupted labels, which are also memorized. To make the model more robust against noisy labels, this paper proposes a Mentor network that generate a curriculum which controls the order and attention to learn each sample. This is within the domain of Curriculum Training (proposed by Bengio et al., 2009) which hypothesizes that a reasonable curriculum can help the model focus on samples whose labels are likely to be correct. A reasonable curriculum would provide the model first with "easy" and then "complex" samples so as to increase learning entropy. However, compared to existing CL methods, MentorNet is different in two aspects. First, the curriculum is learned from data instead of tuned by human experts. It takes into account of the feedback from StudentNet and can be dynamically adjusted during training. Second, the learning objective is jointly minimized using MentorNet and StudentNet via SGD, which makes the algorithm highly efficient.

Impressively, MentorNet paired with a deep CNN StudentNet is able to achieve SOTA result on WebVision (in 2017), a large benchmark with 2.2 million images of real-world noisy labels. It outperforms the previous SOTA top-5 accuracy (Lee el al., 2017) by 3.4%. MentorNet also performs very well on CIFAR-100 (and other large image classification datasets). Define p as the noise fraction of CIFAR-100. The full data-driven version of MentorNet overperforms the bare ResNet101 by 13% when p=0.2, 23% when p=0.4, and 26% when p=0.8. Furthermore, when paired with a MentorNet, ResNet101 is shown to not only converge faster (for the most part), but also is significantly more robust against overfitting (the corrupted labels), as test error approaches 0 as training continues instead of increasing as in the bare model.

The transfer learning application of MentorNet is consistently worse than the direct version. For the aforementioned example in CIFAR-100, MentorNet PD is the transfer version trained on a subset of the CIFAR-10 training set compared to MentorNet DD which is directly trained on CIFAR-100. For p=0.4, PD lags behind DD by 12%, and for p=0.8, PD barely outperforms the bare network, and lags behind by a significant 21%. Therefore, transfer learning application of MentorNet is quite dubious. Nevertheless, MentorNet is still proved to be extremely effective in combatting noisy labeled data.

### Learning to reweight examples for robust deep learning (ICML, 2018)

**Authors:** M. Ren, W. Zeng, B. Yang, and R. Urtasun

### Meta-weight-net: Learning an explicit mapping for sample weighting (NeurIPS, 2019)

**Authors:** J. Shu, Q. Xie, L. Yi, Q. Zhao, S. Zhou, Z. Xu, and D. Meng

## Data Augmentation

I presented this in a CS 598 Meta-learning lecture. Check out the [**slide deck on my Google Drive**](https://drive.google.com/file/d/1JPjf32L8Xgicr1gcXYJ1YlJqRQhLBi4G/view?usp=sharing)

Data Augmentation is a technique that can be used to artificially expand the size of a training set by creating modified data from the existing one. It is a good practice to use DA if you want to prevent overfitting, or the initial dataset is too small to train on, or even if you want to squeeze better performance from your model.

Data Augmentation is not only used to prevent overfitting. In general, having a large dataset is crucial for the performance of both ML and Deep Learning (DL) models. However, we can improve the performance of the model by augmenting the data we already have. It means that Data Augmentation is also good for enhancing the model’s performance.

This article focuses on 3 seminal papers in Data Augmentation: AutoAugment (2019), DADA (2020), and DAGAN (2017).

### AutoAugment: learning augmentation policies from data (CVPR, 2019)

**Authors:** E. D. Cubuk, B. Zoph, D. Mane, V. Vasudevan, and Q. V. Le.

- Problem: Data augmentation implementations are manually designed
- AutoAugment autoamtically search for improved data augmentation policies
- Search space: policy with many sub-policies
  - One randomly chosen for each mini-batch
- Sub-policy consists of 2 operations:
  1. An image processing function (translation, rotation, shearing)
  2. The probabilities and magnitudes of the function
- Search to find the best policy such that the neural network yields the highest validation accuracy on a target dataset
- Very impressive results: Achieves SOTA on CIFAR-10, CIFAR-100, SVHN, and ImageNet (w/o additional data)

- Data augmentation: translating the image by a few pixels, flipping the image horizontally or vertically.
- Goal: reduce overfitting and teach model about invariances in the data domain
- CNN has baked-in invariances (convolution filters)
- However, data augmentation allows more flexibility and ease vs. architectural invariances
- Data augmentation does not transfer easily between datasets (flipping images makes sense in CIFAR-10, but not MNIST)
- This paper aims to automate the process of finding an effective data augmentation policy.
- Each AutoAugment policy consists of:
  1. Several choices and orders of possible operations
  2. Probabilities of applying the functions
  3. Magnitudes with which they are applied
- Uses RL to search for optimal policy (there might be better alternatives)

- 2 use cases:
  1. Apply directly on dataset to find best augmentation policy (AutoAugment-direct)
  2. Learned policies can be transferred to new datasets (AutoAugment-transfer)
- Performance:
  1. Direct: SOTA on CIFAR-10, CIFAR-100, SVHN, reduced SVHN, ImageNet (w/o additional data)
  2. Transfer: Policies on one task can generalize well across different models and datasets
     - Significant improvements on a variety of FGVC datasets
     - Even on datasets with smaller improvements (Stanford Cars, FGVC Aircraft), training with ImageNet still reduces test error by ~1.5%
     - Transferring augmentation is an alternative to standard weight transfer learning
- Best augmentations are dataset-specific:

  - MNIST: elastic distortions, scale, translation, rotation
  - Natural images (CIFAR-10, ImageNet): random cropping, mirroring, color shifting / whitening
  - Though so far, only manually designed

- This approach using RL is able to overcome the 2% error-rate barrier that was impossible through architectural search alone.
- Previous attempts:
  1. Smart Augmentation: augmentation through merging two or more samples from the same class
  2. Tran et al.: Bayesian approach based on distribution learned from training set
  3. DeVries and Taylor: augment data with simple transformations in the learned feature space
  4. Also GANs
- Key difference vs. generative models: this method generates symbolic transformation operations vs. generating the augmented data directly (except Ratner et al.)

Formulate as discrete search problem with 2 components:

1. Search algorithm (controller RNN)
2. Search space

Search space details:

- Each policy consists of 5 sub-policies
  - Each sub-policy consisting of 2 image operations applied in sequence.
  - Each operation is associated with 2 hyper-parameters:
    1. Probability of applying operation
    2. Magnitude of operation
  - Application of sub-policies are stochastic: even with the same sub-policy, image can be transformed differently in different mini-batches
- Figure 2: Example of a policy with 5 sub-policies in search space
- Discretize range of magnitudes into 10 values (.1, .2, ..., 1.) and probability into 11 values (0, .1, ..., 1.) to use a discrete search algorithm.

Search algorithm details:

```
while training:
  Controller RNN samples strategy S (operation type, probability, magnitude)
  Train child network with strategy S to get validation accuracy R
  Use R to update the controller
```

- Uses RL, has 2 components:
  1. Controller (RNN): predicts a decision produced by a softmax
  2. Training algorithm (Proximal Policy Optimization)
     For each policy (5 sub-policies, each with 2 operations, each 3 attributes) so \(5 \* 2 \* 3 = 30\) softmax predictions
- Controller trained with reward signal (how good the policy is at improving the generalization of the "child model")

Vs. Ratner et al.: similar to GANs, generator learns to propose augmentation policy (sequence of image processing operations) such that augmented images can fool a discriminator.

- Difference: this method tries to optimize classification accuracy directly, whereas their method tries to make sure the augmented images are similar to the current training images
- On both ResNet-56 and 32, AutoAugment leads to higher improvement of ~3%

Relation between training steps and number of sub-policies:

- 2 layers of stochasticity: each image is transformed by 1 of the 5 sub-policies, each with its own probability of application.
- Stochasticity requires a minimum number of epochs per sub-policy to be effective.
  - Child models trained with 5 sub-policies, they need to be trained more than 80-100 epochs to be effective.
  - In this paper, child models are trained for 120 epochs
  - After the policy is learned, the full model is trained for longer (much more than 100 epochs)

Transferability across datasets and architectures:

- Policies described above transfer well to many model architectures and datsets.
- Despite this, policies learned on data distributions closest to target yield the best performance (transfer less effective than direct).

### DADA: Differentiable automatic data augmentation (ECCV, 2020)

**Authors:** Y. Li, G. Hu, Y. Wang, T. Hospedales, N. M. Robertson, and Y. Yang

- Problem: AutoAugment is extremely computationally expensive, limiting its wide applicability.
- Followup works (PBA, Fast AutoAugment) improve the speed, but still suffer from the same bottleneck.
- Differentiable Automatic Data Augmentation (DADA) is proposed to dramatically reduce the cost.
  - Relaxes the DA policy selection to a differentiable optimization problem via Gumbel-Softmax.
  - Introduces an unbiased gradient estimator, RELAX, leading to an efficient and accurate DA poilcy.
- DADA is at least one order of magnitude faster than SOTA while achieving comparable accuracy.
- DA particularly important when data is not readily available, e.g. medical image analysis.

Bottleneck: optimization (selecting discrete augmentation functions) is intrinsically non-differentiable, making joint optimization of network weights and DA parameters impossible. - Have to resort to multi-pass RL, BayesOpt, and evolutionary strategies -> slow. - Much more efficient if we can relax the optimization to be differentiable and jointly optimize the network weights and DA parameters in a single-pass way. - Motivated by differentiable neural architecture search.

Same search space as AA:

- Policy contains many sub-policies, each with 2 operations (probability and magnitude).
- DADA first reformulates the discrete search space to a joint distribution that encodes sub-policies and operations.
  - Sub-policy selection same as sampling from a Categorical distribution.
  - Augmentation application same as sampling from a Bernoulli distribution.
- So DA optimization becomes a Monte Carlo gradient estimate problem.
- However, Categorical and Bernoulli distributions are not differentiable!
- So we have to relax the 2 distributions to be differentiable through Gumbel-Softmax gradient estimator (a.k.a. concrete distribution).
- Furthermore, DADA minimizes the loss on validation set rather than the accuracy (used by AutoAugment) to facilitate gradient computation.

Need 2 things:

1. Efficient optimization strategy
2. Accurate gradient predictor

Solutions:

1. Naive solution for sampling-based optimization: iterate 2 sub-optimizations until convergence

   1. Optimize DA policies
   2. Train neural network

   - Problem: sequential optimization very slow.
   - Better solution: like DARTS, jointly optimize parameters of DA and network through SGD.

2. Gumbel-Softmax estimator
   - Problem: it is biased
   - Better solution: RELAX estimator, which can provide an unbiased gradient estimator

### Data augmentation generative adversarial networks (arXiv:1711.04340, 2017)

**Authors:** A. Antoniou, A. Storkey, and H. Edwards

Techniques to combat overfitting: dropout, batch norm, batch renorm, layer norm, etc.

- However, in low data regime, these techniques fall short since the flexibility of network is so high.
- Not able to capitalize on known input invariances that might form good prior knowledge

Typical data augmentation techniques use a very limited set of known invariances. You can learn a much larger invariance space through training a form of conditional GAN in a different domain called the source domain.

Can then be applied in the low-data domain of interest, the target domain. DAGAN enables effective neural network training even in low-data target domains. As it does not depend on the classes themselves, it captures the cross-class transformations, moving data-points to other points of equivalent class. So DAGAN can be applied to unseen classes.

Main contributions:

1. Using GAN to learn a representation and process for DA
2. Demonstrate realistic DA samples from a single data-point
3. Use DAGAN to augment standard classifier in low-data regime, demonstrating significant improvements in the generalization performance on all tasks.
4. The application of DAGAN in the meta-learning space, achieving SOTA in EMNIST and Omniglot and beating all other general meta-learning models.
5. Efficient 1-shot augmentation of matching networks.

## Data Synthesis

### Dataset distillation (arXiv:1811.10959, 2018)

**Authors:** T. Wang, J.-Y. Zhu, A. Torralba, and A. Efros

Dataset distillation aims to synthesize a small number of data points that will, when used as input data, be able to approximate the model trained on the original dataset. This paper main focuses on computer vision applications, and the synthesized images are called distilled images. The goal here is to "compress" and encode as much data as possible into a few distilled images.

As opposed to conventional wisdom, Dataset Distillation (DD) proves it is possible to train an image classification model with images from synthetic images from outside the manifold.

### Image deformation meta-networks for one-shot learning (CVPR, 2019)

**Authors:** Z. Chen, Y. Fu, Y.-X. Wang, L. Ma, W. Liu, and M. Hebert

### Learning to simulate (ICLR, 2019)

**Authors:** N. Ruiz, S. Schulter, and M. Chandraker

This paper prosposes a RL-based learning method for automatically adjusting the parameters of any (non-differentiable) simulator. Traditional methods mostly attempt to hand-craft parameters or adjust only parts of the parameters. Their objectives focus on mimicking the real data distribution or randomly generating a large volume of data. In contrast, this approach proposes to automatically determine simulation parameters to directly maximize the performance (accuracy) of the main model. This boils down to a bi-level optimization problem where the upper-level task is a meta-learner that learns how to generate data to minimize the validation loss, and the lower-level task is the main task model (MTM) that learns to solve the actual task at hand. To actually optimize this, the paper uses a policy gradient method to optimize the two objectives. A gradient-based optimization is not applicable because the simulator distribution is parameterized by simulator parameters which makes it non-differentiable, and the actual simulation process, e.g. image rendering, is assumed to be non-differentiable. Now, the simulator parameters learned are unconstrained continuous variables, which can be used to parameterize a multivariate Gaussian acting as the simulator.

The simulator performs very well:

1. On the toy experiment on Gaussian mixtures: While the simulator converged on a different distribution than the real GMM, the learned SVM decision boundary still performs very well on the test data.
2. On the parameterized traffic simulator: The simulated traffic scene is able to generate different types of intersections, with various car models, road layouts, buildings on the sides, and various weather conditions.
3. On high-level vision tasks: Learning to Simulate (LTS) outperforms random parameter intialization on both the car counting task and the semantic segmentation tasks on real and simulated data.

There is a slight problem with this paper, however. While the experiments conducted are very intuitive, the experimentation could have been more extensive. The problems chosen are great, and the search space is also quite interesting. However, LTS was mainly compared against random simulator parameters, and not any SOTA computer vision model without simulated data. Simulator performance is compared against the distribution of the validation set, which is a big plus. Still, a lack of comparison to other models lessened the impact of this paper.

## Other Meta-Objectives

### Episodic training for domain generalization (ICCV, 2019)

**Authors:** D. Li, J. Zhang, Y. Yang, C. Liu, Y.-Z. Song, and T. Hospedales

### Learning to learn from noisy labeled data (CVPR, 2019)

**Authors:** J. Li, Y. Wong, Q. Zhao, and M. Kankanhalli

### Adversarial attacks on graph neural networks via meta learning (ICLR, 2019)

**Authors:** D. Zügner and S. Günnemann

Graph neural networks have displayed impressive SOTA results in various tasks (e.g., text classification). However, little is known about their robustness. This paper studies the problem of learning a better poisoned graph parameters that can maximize the loss of a graph neural network. The paper finds that by computing the meta-gradient, the authors are able to solve the bilevel problem underlying training-time attacks. As a result, they are able to generate small perturbations in the input data that cause the model not only to perform significantly worse, but also worse than a simple baseline that ignores all relational information.

Previous works on adversarial attacks on GNNs are generally sparse. Dai et al. (2018) consider test-time (evasion) attacks, but not training-time (poisoning) attacks, which this paper addresses. A previous paper by the same authors, Zugner et al. (2018) consider both types of attacks, but their algorithm is suited only to targeted attack on single nodes. This paper will make sure the attack is able to increase misclassification error globally for every node.

The paper proposes multiple approaches to calculating the meta-gradients. The first is to compute the second-order derivatives over the backpropagation step, essentially treating the graph as a hyperparameter to optimize. This approach displays great results. However, second-order derivatives are costly to compute, so the authors also propose approximate methods which are shown to be more efficient. The attack objective here is a bilevel problem:

$$
\underset{\widehat{G} \in \Phi(G)}{\min} \mathcal{L}_{\text{atk}} (f_{\theta^{*}} (\widehat{G}) \\
\text{ s.t. } \theta^{*} = \underset{\theta}{\argmin} \mathcal{L_{\text{train}}} (f_{\theta} (\widehat{G}))
$$

While we are trying to undermine the generalization performance of the GNN, we do not have the validation labels. The paper remediates this problem by trying instead to on misclassification rate on the training set: $\mathcal{L_{\text{atk}}} = -\mathcal{L_{\text{train}}}$

In both cases for meta-gradient computation (direct and approximate), the experimental results on three graph datasets show that the proposed model could improve the misclassification rate of the unlabeled nodes.

For the meta-gradient to be computed, the proposed attack model assumes the graph structure are accessible to the attackers. This is not always true in practice, which might limit the method's applicability. The paper does provide a joint study with the graph features on Citseer, and the impact of the combined attack is comparable but lower than the structure attack. This ablation is insightful, but it still doesn't address the case where the graph structure is not available to the attacker at all. Overall, this paper is very well written and comprehensively considers multiple algorithms to generate the meta-gradients used as the attack vector.

## Meta-Reinforcement Learning

### Learning to reinforcement learn (CogSci, 2017)

**Authors:** J. Wang, Z. Kurth-Nelson, D. Tirumala, H. Soyer, J. Leibo, R. Munos, C. Blundell, D. Kumaran, and M. Botvinick

Recent advances have allowed to attain human-, and sometimes superhuman-level performance in certain complex and large-scale task environments such as Atari and Go. However, overall, RL still lags significantly behind human performance on most tasks because of two main limitations:

1. Deep RL typically requires a massive volume of training data, whereas human learners can attain reasonable performance on any of a wide range of tasks with comparatively little experience.
2. Deep RL systems typically specialize on one restricted task domain, whereas human learners can flexibly adapt to changing task conditions.

The authors propose a framework to overcome these factors which they call deep meta-reinforcement learning. The concept is to train a recurrent neural network (in this case an LSTM) as a reinforcement learning algorithm (in this case A2/3C), so that the LSTM eventually becomes its own reinforcement learning algorithm.

![L2RL Algorithm](/meta-learning/l2rl-alg.png)

Using an actor-critic with recurrence architecture, they trained the agent on several tasks from the same family of tasks to prove their agent can learn a new task even if the weights are fixed, after a training period.

The agent is trained in three different categories of tasks:

1. Variations of the multi-armed bandit problem. With a distribution $D$ of Markov decision process (MDP), the authors sample a task at the beginning of an episode and the internal states of the agent are reset. The training process then goes on for a certain number of episodes, where a new task is drawn from $D$ at each new episode.

   After training is completed, the agent’s policy is fixed and it is tested over new tasks drawn from $D$ or slight modifications of $D$ to test its generalization capabilities. The authors show that the agent is performing well on new MDPs even when its weights are fixed.

2. A problem where a bandit had one informative arm (with a fixed position) and multiple rewarding-arm including one optimal arm (with varying positions). The informative arm provided no reward but informed the agent as to where the optimal arm was. The above graph shows that the algorithm was able to prefer losing immediate reward for information which then provided optimal reward, as opposed to classic solvers which did not use this arm.

   They also trained the agent to see if it would learn the underlying task structure where needed. To do so, they trained the agent on the MDP above. Without going into details into the experiment, the authors discovered that the agent has the same behavior as a model-based agent, while being model-free, which implies that the agent learned the underlying task structure.

3. Finally, the authors then wanted to see if the agent was able to learn an abstract task representation and give rise to one-shot learning. To do so, they adapted an experiment from a study on animal behavior, where the agent had to do two consecutive steps (see paper for more details).

   Each episode had its own sets of images, and after a couple of episodes, the agent always selected the “target” image after the first try, which indicates that the agent is able to “one-shot-learn” the target image.

   The authors also reported an experiment where, after some training, is able to return to the goal confidently within an episode even with a random starting point. This means that the agent is able to “one-shot-learn” the position of the goal and is generally able to learn how to navigate in a maze, independently of its structure.

The paper marks the beginning of a resurgence of "modern" meta-learning, and it proposes many novel ideas. However, the authors did not try to compare their agent to “classic” feed-forward deep reinforcement-learning agents (except for the reported maze experiment), so we cannot know if the behaviors shown by the LSTM agent are unique to it or general to all deep RL agents.

### RL^ 2: Fast reinforcement learning via slow reinforcement learning (ICLR, 2017)

**Authors:** Y. Duan, J. Schulman, X. Chen, P. Bartlett, I. Sutskever, and P. Abbeel

### Meta-reinforcement learning of structured exploration strategies (NeurIPS, 2018)

**Authors:** A. Gupta, R. Mendonca, Y. Liu, P. Abbeel, and S. Levine

## Semi-supervised Learning & Active Learning & Domain Shift

### Meta-learning for semi-supervised few-shot classification (ICLR, 2018)

**Authors:** M. Ren, E. Triantafillou, S. Ravi, J. Snell, K. Swersky, J. Tenenbaum, H. Larochelle, and R. Zemel

### Learning algorithms for active learning (ICML, 2017)

**Authors:** P. Bachman, A. Sordoni, and A. Trischler

### Cross-domain few-shot classification via learned feature-wise transformation (ICLR, 2020)

**Authors:** H.-Y. Tseng, H.-Y. Lee, J.-B. Huang, and M.-H. Yang

Computer vision benefits from a large amount of data, but high quality labeled data is not always available. This sparks a wave of interest in few-shot classification task, where the model has to recognizes instances from novel tasks with only a few labeled samples per class (support set). Focusing on image classification, existing methods have found success with learning a metric function that approximate the distance between feature embeddings of unlabeled and labeled images. However, these methods don't generalize well because of large feature discrepancy across domains. This paper proposes using a feature-wise transformations to augment input images with affine transforms to simulate various feature distributions. Futhermore, the authors propose a meta-learning approach to automate searching hyperparameters for the feature-wise transformation layer.

The goal of this method is one of domain generalization, i.e., from $N$ seen domains $\{ \mathcal{T}^{\text{seen}}_1, ..., \mathcal{T}^{\text{seen}}_N \}$, learn a metric-based few-shot classification model so that it perform well on unseen domain $\mathcal{T}^{\text{unseen}}$. The core idea of metric-based learning is sample two labeled sets: support set and query set. The feature encoder $E$ then extracts the feature embeddings of the sample images. The metric function $M$ then predicts the categories of query images based on labels of support images, and the feature embeddings of the support and query images. Now, the training objective is to minimize classification error of images in the query set.

A problem arises when the feature space discrepancy between seen and unseen domains is too large. This can cause the metric function $M$ to overfit to the seen domains and fail to generalize to the unseen domains. To address the problem, the authors propose to integrate a feature-wise transformation to augment the intermediate feature activations with affine transformations into the feature encoder $E$. Intuitively, this combined architecture can produce more diverse feature distributions which improve the generalization ability of the metric function $M$.

Now, to learn the hyperparameters of the feature-wise transformation, we split the training set into a pseudo-seen and pseudo-unseen domains. This is a alternating optimization problem:

1. Optimize classification loss to update neural network parameters with the pseudo-seen samples like in metric-based methods.
2. Optimize generalization loss to update hyperparameters of the feature-wise transformation $f$. Generalization loss is the difference in classification error when feature-wise transformation is turned on and when it's removed.

This paper is one of the pioneering works in the field of few-shot learning. It suggests metric-based models might be overfitting to the training set, and proposes a novel approach that might boost generalizability. Unfortunately, it seems like the learned feature-transform only manages to increase accuracy by an average of 2% across datasets. However, the proposed approach has a lot of potential, and a more sophisticated one-pass bilevel optimization approach might be able to perform much better.

## Retrospection: Concept of Learning to Learn

### Evolutionary principles in self-referential learning. On learning how to learn: The meta-meta-... hook. (1987)

**Authors:** J. Schmidhuber

### Using fast weights to deblur old memories (Conference Of The Cognitive Science Society, 1987)

**Authors:** G. E. Hinton and D. C. Plaut

The paper proposes two weights on each connection:

1. A slow-changing, plastic weight for long-term knowledge
2. A fast-changing, elastic weight for temporary knowledge



### On the optimization of a synaptic learning rule (Conf. Optimality in Artificial and Biological Neural Networks, 1992)

**Authors:** S. Bengio, Y. Bengio, J. Cloutier, and J. Gecsei

### Learning to control fast-weight memories: An alternative to dynamic recurrent networks (Neural Computation, 1992)

**Authors:** J. Schmidhuber

### Is learning the n-th thing any easier than learning the first? (NeurIPS, 1996)

**Authors:** S. Thrun

### Learning to learn: Introduction and overview (Learning To Learn, 1998)

**Authors:** S. Thrun and L. Pratt

### Learning to learn using gradient descent (International Conference on Artificial Neural Networks, 2001)

**Authors:** S. Hochreiter, A. Younger, and P. Conwell

### Meta-learning with backpropagation (IJCNN, 2001)

**Authors:** A. S. Younger, S. Hochreiter, and P. R. Conwell

### A perspective view and survey of meta-learning (Artificial Intelligence Review, 2002)

**Authors:** R. Vilalta and Y. Drissi

### Meta-learning in reinforcement learning (Neural Networks, 2003)

**Authors:** N. Schweighofer and K. Doya
