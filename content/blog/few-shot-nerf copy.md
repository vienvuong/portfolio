---
author: "Vien Vuong"
title: "Instant Semantically Consistent Few Shot View Synthesis [Ongoing]"
date: "2022-09-30"
description: ""
tags: ["nerf", "3d", "computer-vision", "ml", "project"]
comments: false
socialShare: false
toc: true
math: true
draft: false
---

This is part of my final project for CS 598: Learning to Learn at UIUC.

## Abstract

Recent advances in machine learning has led to increased interest in employing coordinate-based neural networks as a promising tool for computer graphics for tasks such as view synthesis ([NeRF](https://www.matthewtancik.com/nerf)), radiance caching ([instant-ngp](https://nvlabs.github.io/instant-ngp/)), geometry representations ([DeepSDF](https://arxiv.org/abs/1901.05103)), and more (see survey paper [Neural Fields](https://neuralfields.cs.brown.edu/)). These methods, now called neural fields, approximate continuous 3D space with a countinuous, parametric function. Often, this parametric function is an MLP which takes in coordinates as input and output a vector (such as color or occupancy). Neural fields differ from previous signal representations like pixel images or voxels which are discrete and approximate continuous signals with regularly spaced samples of the signal. Neural fields have sparked an explosion of research interest, which have led to widespread success in problems such as 3D shape and image synthesis, animation of human bodies, 3D reconstruction, and pose estimation.

- Note: Neural fields are also known as implicit neural representations, neural implicits, or coordinate-based neural networks.

Of the neural field techniques, neural radiance fields (NeRF) is one of the most prominent and well-studied. Since 2020, significant improvements have been made to address limitations in every aspect of the original [NeRF](https://www.matthewtancik.com/nerf) implementation. While subsequent papers have achieved impressive results, little effort has been invested into combining and contrasting these very diverse approaches. This paper focuses on three state-of-the-art papers ([instant-ngp](https://nvlabs.github.io/instant-ngp/), [VQAD](https://nv-tlabs.github.io/vqad/) and [TODO: 2nd paper here](https://to.do/)) that apply computer graphics and meta-learning optimizations to maximize NeRF performance in two aspects:

1. Training and rendering performance
2. Generalizability

Specifically, our goal is to maximize NeRF's quality, capability, and generalizabilty on few-shot novel view synthesis tasks. We will conduct experiments on datasets **TODO**. We will evaluate synthesis quality and training/rendering time compared to numbers given by other methods. Finally, we will study how generalizable our model is by pre-training on one dataset, and then fine-tuning and testing on a different dataset. We will evaluate the 0-shot, 1-shot, and 5-shot transfer-learning cases.

This paper consists of 2 sections:

- [**Part I**](#part-i-an-overview-of-neural-field-techniques) — Historical overview of progress leading up to instant-NGP, VQAD, and **TODO**. We will identify common trends and components of past methods (different conditioning, representation, forward map, architecture, and manipulation methods), and consolidate discovered knowledge in neural field research.

- [**Part II**](#part-ii-fast-semantically-consistent-few-shot-view-synthesis) — Implementation and performance evaluation of **TODO**.

## Part II: Fast Semantically Consistent Few Shot View Synthesis

This project plans to combine and implement subsequent works that propose applying computer graphics and meta-learning approaches to problems 1, 4, and 5. Our goal is to maximize NeRF quality, capability, and generalizabilty on few-shot novel view synthesis tasks.

## Method

### Framework

#### Kaolin Wisp

I will use NVIDIA Kaolin Wisp to implement my enhanced NeRF model.

Kaolin Wisp is a PyTorch library powered by NVIDIA Kaolin Core to work with neural fields (including NeRFs, NGLOD, instant-ngp and VQAD). Kaolin Wisp provides a PyTorch interface to the custom CUDA kernels proposed by [VQAD](https://nv-tlabs.github.io/vqad/). This allows me to implement my version of VQAD entirely in Python and not have to worry about low-level CUDA code.

Kaolin Wisp also provides:

1. A set common utility functions (datasets, image I/O, mesh processing, and ray utility functions)
2. Building blocks like differentiable renderers and differentiable data structures (octrees, hash grids, triplanar features)
3. Debugging visualization tools, interactive rendering and training, logging, and trainer classes.

These features are incredibly useful for building complex neural fields, and will certainly speed up and simplify my prototyping process.

## Experiments

### Evaluation Metrics

- PSNR (Peak Signal-to-Noise Ratio): higher PSNR, lower MSE. Lower MSE implies less difference between the ground truth image and the rendered image. Thus, higher PSNR, better the model.

- SSIM (Structural Similarity Index): Checks the structural similarity with the ground truth image model. Higher SSIM, better the model.

- LPIPS (Learned Perceptual Image Patch Similarity): Determines the similarity with the view of perception using VGGNet. Lower LPIPS, better the model.

### Datasets

#### NeRF Synthetic Dataset

Introduced by the original NeRF paper, Mildenhall et al., ECCV 2020. This will be used as a baseline to compare quality metrics (PSNR, SSIM, LPIPS) against other NeRF papers.

The dataset contains three parts with the first 2 being synthetic renderings of objects called Diffuse Synthetic 360◦ and Realistic Synthetic 360◦ while the third is real images of complex scenes. We will test our model on the second and third parts of the dataset (the first part is too simple).

#### ShapeNetCore

Introduced by ShapeNet: An Information-Rich 3D Model Repository, Chang et al., 2015. This will also be used as a baseline to compare quality metrics (PSNR, SSIM, LPIPS) against other NeRF papers. Since this dataset has many objects of the same classes, it can be used as a pretraining dataset before testing the model on the NeRF dataset or another novel view synthesis dataset.

ShapeNetCore is a subset of the full ShapeNet dataset with single clean 3D models and manually verified category and alignment annotations. It covers 55 common object categories with about 51,300 unique 3D models. The 12 object categories of PASCAL 3D+, a popular computer vision 3D benchmark dataset, are all covered by ShapeNetCore.

## Sources

### Papers

[State of the Art on Neural Rendering](https://arxiv.org/abs/2004.03805) (Tewari et al., EUROGRAPHICS 2020)

[Neural Fields in Visual Computing and Beyond](https://neuralfields.cs.brown.edu/) (Xie et al., EUROGRAPHICS 2022)

[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://www.matthewtancik.com/nerf) (Mildenhall et al., ECCV 2020)

[Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://bmild.github.io/fourfeat/) (Tancik et al., NeurIPS 2020)

[Learned Initializations for Optimizing Coordinate-Based Neural Representations](https://www.matthewtancik.com/learnit) (Tancik et al., CVPR 2021)

[pixelNeRF: Neural Radiance Fields from One or Few Images](https://alexyu.net/pixelnerf/) (Yu et al., CVPR 2021)

[AutoInt: Automatic Integration for Fast Neural Volume Rendering](https://arxiv.org/abs/2012.01714) (Lindell et al., CVPR 2021)

[DONeRF: Towards Real-Time Rendering of Compact Neural Radiance Fields using Depth Oracle Networks](https://depthoraclenerf.github.io/) (Neff et al., EGSR 2021)

[Putting NeRF on a Diet: Semantically Consistent Few-Shot View Synthesis](https://ajayj.com/dietnerf/) (Jain et al., ICCV 2021)

[Plenoxels: Radiance Fields without Neural Networks](https://alexyu.net/plenoxels/) (Yu et al., CVPR 2022)

[Direct Voxel Grid Optimization: Super-fast Convergence for Radiance Fields Reconstruction](https://sunset1995.github.io/dvgo/) (Sun et al., CVPR 2022)

[Improved Direct Voxel Grid Optimization for Radiance Fields Reconstruction](https://arxiv.org/abs/2206.05085) (Sun et al., 2022)

[TensoRF: Tensorial Radiance Fields](https://apchenstu.github.io/TensoRF/) (Chen et al., ECCV 2022)

[FastNeRF: High-Fidelity Neural Rendering at 200FPS](https://arxiv.org/abs/2103.10380) (Garbin et al., ICCV 2021)

[Instant Neural Graphics Primitives with a Multiresolution Hash Encoding](https://nvlabs.github.io/instant-ngp/) (Müller et al., SIGGRAPH 2022)

[Variable Bitrate Neural Fields](https://nv-tlabs.github.io/vqad/) (Takikawa et al., SIGGRAPH 2022)

[DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation](https://arxiv.org/abs/1901.05103) (Park et al., CVPR 2019)

[Real-time Neural Radiance Caching for Path Tracing](https://research.nvidia.com/publication/2021-06_real-time-neural-radiance-caching-path-tracing) (Müller et al., SIGGRAPH 2021)

[ShapeNet: An Information-Rich 3D Model Repository](https://shapenet.org/) (Chang et al., 2015)

### Websites

**Frank Dellaert's Blog Posts**

- [NeRF at CVPR 2022](https://dellaert.github.io/NeRF22/)
- [NeRF at ICCV 2021](https://dellaert.github.io/NeRF21/)
- [NeRF Explosion 2020](https://dellaert.github.io/NeRF/)

**GitHub Repositories**

- [NVIDIA Kaolin Wisp](https://github.com/NVIDIAGameWorks/kaolin-wisp)
- [Awesome Neural Radiance Fields](https://github.com/yenchenlin/awesome-NeRF)
