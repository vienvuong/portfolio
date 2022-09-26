---
author: "Vien Vuong"
title: "Data Selection & Reweighting in Meta-learning"
date: "2022-09-15"
description: ""
tags: ["meta-learning", "ml"]
comments: false
socialShare: false
toc: true
math: true
---

## Mentornet: Learning data-driven curriculum for very deep neural networks on corrupted labels (ICML, 2018)

Authors: L. Jiang, Z. Zhou, T. Leung, L.-J. Li, and L. Fei-Fei

### Overview and Main Contributions

Problem: Modern networks are capable of memorizing the entire dataset. This becomes a problem when the dataset contains corrupted labels, which are also memorized. To make the model more robust against noisy labels, this paper proposes a Mentor network that generate a curriculum which controls the order and attention to learn each sample. This is within the domain of Curriculum Training (proposed by Bengio et al., 2009) which hypothesizes that a reasonable curriculum can help the model focus on samples whose labels are likely to be correct. A reasonable curriculum would provide the model first with "easy" and then "complex" samples so as to increase learning entropy. However, compared to existing CL methods, MentorNet is different in two aspects. First, the curriculum is learned from data instead of tuned by human experts. It takes into account of the feedback from StudentNet and can be dynamically adjusted during training. Second, the learning objective is jointly minimized using MentorNet and StudentNet via SGD, which makes the algorithm highly efficient.

Impressively, MentorNet paired with a deep CNN StudentNet is able to achieve SOTA result on WebVision (in 2017), a large benchmark with 2.2 million images of real-world noisy labels. It outperforms the previous SOTA top-5 accuracy (Lee el al., 2017) by 3.4%. MentorNet also performs very well on CIFAR-100 (and other large image classification datasets). Define p as the noise fraction of CIFAR-100. The full data-driven version of MentorNet overperforms the bare ResNet101 by 13% when p=0.2, 23% when p=0.4, and 26% when p=0.8. Furthermore, when paired with a MentorNet, ResNet101 is shown to not only converge faster (for the most part), but also is significantly more robust against overfitting (the corrupted labels), as test error approaches 0 as training continues instead of increasing as in the bare model.

### Potential Improvements and Criticisms

The transfer learning application of MentorNet is consistently worse than the direct version. For the aforementioned example in CIFAR-100, MentorNet PD is the transfer version trained on a subset of the CIFAR-10 training set compared to MentorNet DD which is directly trained on CIFAR-100. For p=0.4, PD lags behind DD by 12%, and for p=0.8, PD barely outperforms the bare network, and lags behind by a significant 21%. Therefore, transfer learning application of MentorNet is quite dubious. Nevertheless, MentorNet is still proved to be extremely effective in combatting noisy labeled data.

## Learning to reweight examples for robust deep learning (ICML, 2018)

Authors: M. Ren, W. Zeng, B. Yang, and R. Urtasun

## Meta-weight-net: Learning an explicit mapping for sample weighting (NeurIPS, 2019)

Authors: J. Shu, Q. Xie, L. Yi, Q. Zhao, S. Zhou, Z. Xu, and D. Meng