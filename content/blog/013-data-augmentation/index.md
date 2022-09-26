---
author: "Vien Vuong"
title: "Data Augmentation in Meta-learning"
date: "2022-09-17"
description: "Data Augmentation is a technique that can be used to artificially expand the size of a training set by creating modified data from the existing one. This article focuses on 3 seminal papers in Data Augmentation: AutoAugment (2019), DADA (2020), and DAGAN (2017)."
tags: ["meta-learning", "ml"]
comments: false
socialShare: false
toc: true
math: true
---

I presented this in a CS 598 Meta-learning lecture. Check out the [slide deck on my Google Drive](https://drive.google.com/file/d/1JPjf32L8Xgicr1gcXYJ1YlJqRQhLBi4G/view?usp=sharing)

Data Augmentation is a technique that can be used to artificially expand the size of a training set by creating modified data from the existing one. It is a good practice to use DA if you want to prevent overfitting, or the initial dataset is too small to train on, or even if you want to squeeze better performance from your model.

Data Augmentation is not only used to prevent overfitting. In general, having a large dataset is crucial for the performance of both ML and Deep Learning (DL) models. However, we can improve the performance of the model by augmenting the data we already have. It means that Data Augmentation is also good for enhancing the model’s performance.

This article focuses on applications of Data Augmentation in Meta-learning. Specifically, I cover two Auto-Augment-based methods (AutoAugment and DADA), and one GAN-based method (DAGAN).

## AutoAugment: learning augmentation policies from data (CVPR, 2019)

Authors: E. D. Cubuk, B. Zoph, D. Mane, V. Vasudevan, and Q. V. Le.

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

## DADA: Differentiable automatic data augmentation (ECCV, 2020)

Authors: Y. Li, G. Hu, Y. Wang, T. Hospedales, N. M. Robertson, and Y. Yang

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

## Data augmentation generative adversarial networks (arXiv:1711.04340, 2017)

Authors: A. Antoniou, A. Storkey, and H. Edwards

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
