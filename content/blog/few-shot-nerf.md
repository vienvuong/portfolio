---
author: "Vien Vuong"
title: "Fast Semantically Consistent Few Shot View Synthesis [Ongoing]"
date: "2022-09-30"
description: ""
tags: ["nerf", "3d", "computer-vision", "ml", "research"]
comments: false
socialShare: false
toc: true
math: true
draft: false
---

This is part of my final project for CS 598: Learning to Learn at UIUC.

## Abstract

Recent advances in machine learning has led to increased interest in employing coordinate-based neural networks as a promising tool for computer graphics for tasks such as view synthesis ([NeRF](https://www.matthewtancik.com/nerf)), radiance caching ([Instant-NGP](https://nvlabs.github.io/instant-ngp/)), geometry representations ([DeepSDF](https://arxiv.org/abs/1901.05103)), and more (see survey paper [Neural Fields](https://neuralfields.cs.brown.edu/)). These methods, now called neural fields, approximate continuous 3D space with a countinuous, parametric function. Often, this parametric function is an MLP which takes in coordinates as input and output a vector (such as color or occupancy). Neural fields differ from previous signal representations like pixel images or voxels which are discrete and approximate continuous signals with regularly spaced samples of the signal. Neural fields have sparked an explosion of research interest, which have led to widespread success in problems such as 3D shape and image synthesis, animation of human bodies, 3D reconstruction, and pose estimation.

Since the seminal [NeRF paper](https://www.matthewtancik.com/nerf) in 2020, significant improvements have been made to address limitations in every aspects of the original implementation. While subsequent papers have achieved impressive results, little effort has been invested into combining and contrasting their very diverse approaches. This project focuses on two state-of-the-art papers ([VQAD](https://nv-tlabs.github.io/vqad/) and [TODO: 2nd paper here](https://to.do/)) that apply computer graphics and meta-learning principles to maximize NeRF's performance in two aspects:

1. Training and rendering performance
2. Generalizability

Specifically, our goal is to maximize NeRF's quality, capability, and generalizabilty on few-shot novel view synthesis tasks. We will conduct experiments on datasets **TODO**, and we will study how our model transfer between these datasets **TODO**.

This paper consists of 2 sections:

- [**Part I**](#part-i-an-overview-of-neural-field-techniques) — Historical overview of progress leading up to VQAD and **TODO**. We will identify common trends and components of past methods (different conditioning, representation, forward map, architecture, and manipulation methods), and consolidate discovered knowledge in neural field research.

- [**Part I**](#part-ii-fast-semantically-consistent-few-shot-view-synthesis) — Implementation and performance evaluation of VQAD+**TODO**.

## Part I: An Overview of Neural Field Techniques

### Section 1: Introduction to Neural Radiance Fields (NeRF)

<video src="/nerf-guide/nerf-demo.mp4" autoplay="true" controls="false" loop="true"></video>

After causing a big splash in ECCV 2020, the impressive [NeRF paper](https://www.matthewtancik.com/nerf) by Mildenhall et al. has kickstarted an explosion in interest in the field of neural volume rendering. It is a novel, data-driven approach that provides an efficient synthesis of visually compelling novel scenes from input images or videos.

NeRF also allows explicit or implicit control of scene properties such as illumination, camera parameters, pose, geometry, appearance, and semantic structure, which has been impossible with previous photogrammetry or GAN-based approaches.

Arguably, the greatest contribution of the paper is its novel approach to storing 3D scenes as implicit representations. 3D geometry, texture, and lighting data can now be generated with a neural network. This volume rendering process is fully differentiable.

- Input: a single contious 5D coordinate (spatial location $(x, y, z)$ and viewing direction $(\theta, \phi)$)
- Output: volume density and RGB "color" (i.e., view-dependent emitted radiance)

This approach proves to be much more space-efficient and high-fidelity than discrete mesh-based or voxel-based representations (but much slower, see below).

#### Algorithm

![NeRF Overview](/nerf-guide/nerf-overview.png)

The full algorithm to compute a neural radiance field is as such:

1. 5D coordinates (location and viewing direction) are sampled along camera rays casted onto the 3D object.
2. Optional: Hierarchical volume sampling (HVS) can be used here reduce sampling density in free space and occluded regions to improve sampling efficiency. The overall network architecture is composed of two networks: the "coarse" network (naive stratified sampling) and the "fine" network (informed biased sampling based on output of the "coarse" network). This affects how the loss function is computed.
3. Optional: Positional encoding (also see gamma encoding in [Fourier Features](https://bmild.github.io/fourfeat/)) can be used here to lift 5D input coordinates to a higher-dimensional space. The paper shows this can allow the MLP to better represent higher frequency variation in color and geometry. This encoding lifts the input coordinates from 5D to a higher-dimensional space.
4. The encoded coordinates are now pass into an MLP to produce a color and volume density. The network can be represented as a mapping $F_{\Theta}: (x, d) \rightarrow (c, \sigma)$, where $c = (r, g, b)$ is the emitted color, $\sigma$ is the volume density, $x$ is the 3D spatial location, and $d$ is the unit vector of the camera ray's viewing direction.
5. These color and volume density values can now be transformed into an image by a fully differentiable volume rendering procedure.
6. This differentiability allows end-to-end backpropagation from the rendering loss through the fully connected layers (MLP). The model is then optimized to minimize the residual between the synthesized and ground truth observed images.

To encourage viewpoint-free image synthesis, the NeRF restricts the network to predict the volume density $\sigma$ as a function of only the location $x$, while the RGB color $c$ is predicted as a function of both the location $x$ and the viewing direction $d$.

#### Loss Function

![NeRF Loss Function](/nerf-guide/nerf-loss.png)

The ultimate goal of this network is to predict the expected color value for the ray correctly. Since we can estimate the ground truth ray color with the ground truth 3D model, we can use L2-distance with the RGB values as a loss.

If HVS is used, the loss is the sum of the L2-norms of the coarse network and the fine network, such that both networks are well-optimized.

#### Limitations of NeRF and Subsequent Works

Unfortunately, the original implementation of NeRF has various limitations:

1. Long training and rendering time (compared to other 3D representation methods)
2. Can only represent static scenes and the generated views are not interactable.
3. Lighting is fixed, no support for relighting
4. A trained NeRF representation does not generalize to other scenes/objects. It tends to overfit to training views.
5. NeRF needs to observe the geometry of every part of the scene to generate novel views. It cannot infer the geometry of occluded parts without help of priors.

Many of these drawbacks have been quickly identified and addressed by subsequent papers. We will evaluate important papers from 2020-2022 that focus on problems 1, 4, and 5. Finally, these papers will lead up to a discussion about VQAD and **TODO** in the [State of the Art](#section-4-state-of-the-art) section.

### Section 2: Improving Performance

- NeRF main bottleneck is having to train and query an MLP that maps 5D spatial coordinate to color and opacity.
- Training and rendering time has been significantly reduced with approaches such as:

  - [AutoInt](https://arxiv.org/abs/2012.01714) restructures the coordinatebased MLP to compute ray integrals exactly, for >10x faster rendering with a small loss in quality.
  - [Learned Initializations](https://www.matthewtancik.com/learnit) employs meta-learning on many scenes to start from a better MLP initialization, for both 10-20x faster training and better priors when per-scene data is limited.
  - [DONeRF](https://depthoraclenerf.github.io/) achieves speedup by predicting a surface with a depth oracle network and sampling near the surface. This reduces the number of samples necessary for rendering each ray. Achieves 15-78x faster rendering time.

- Even with these optimizations, NeRF still proves to be too slow compared to classical volume rendering methods like voxel grids and multi-plane images (MPIs). It seems that to achieve any further improvement, the MLP itself will have to be modified.

- There are four main types of data structure being used to replace MLP.
  1. Sparse voxel grid (no MLP): [Plenoxels](https://alexyu.net/plenoxels/)
  2. Dense voxel grid (with shallow MLP): [DVGO](https://sunset1995.github.io/dvgo/) and [DVGOv2](https://arxiv.org/abs/2206.05085)
  3. Hash: [Instant-NGP](https://nvlabs.github.io/instant-ngp/)
  4. Factorized components: [TensoRF](https://apchenstu.github.io/TensoRF/)
- Note that Plenoxels, Instant-NGP, and DVGOv2 require custom CUDA kernels, while DVGO and TensoRF use standard PyTorch implementation.

#### [Plenoxels](https://alexyu.net/plenoxels/) (Yu et al., CVPR 2022)

- Plenoxels proposes to replace the NeRF MLP with a sparse voxel ("plenoxel") grid with density and spherical harmonic coefficients at each voxel.

  ![Plenoxels Framework](/nerf-guide/plenoxels-framework.png)

- To render a ray, we compute the color and density via trilinear interpolation of the neighboring voxel coefficients. Then we integrate to render, and optimize using the standard MSE reconstruction loss relative to the training images, along with a total variation regularizer.

- Achieves >100x training time speedup. And while not optimized for fast rendering, Plenoxels can generate views at 15fps (>450x rendering time speedup).

  ![Plenoxels Speed](/nerf-guide/plenoxels-speed.png)

- Key take-aways:
  - The key component in NeRF is the differentiable volumetric rendering, not the neural network.
  - Trilinear interpolation is key to achieve high resolution and better convergence (vs. nearest neighbor).
  - Regularization is important to prevent artifacts.

#### [DVGO](https://sunset1995.github.io/dvgo/) (Sun et al., CVPR 2022)

![DVGO Speed](/nerf-guide/dvgo-speed.png)

#### [DVGOv2](https://arxiv.org/abs/2206.05085) (Sun et al., Preprint 2022)

#### [Instant-NGP](https://nvlabs.github.io/instant-ngp/) (Müller et al., SIGGRAPH 2022)

#### [TensoRF](https://apchenstu.github.io/TensoRF/) (Chen et al., ECCV 2022)

### Section 3: Improving Generalizability

#### [Learned Initializations](https://www.matthewtancik.com/learnit) (Tancik et al., CVPR 2021)

- Optimizing a coordinate-based network, such as NeRF, from randomly initialized weights for each new signal is inefficient.

- Using a MAML-based algorithm to learn initial weights enables faster convergence during optimization and can serve as a strong prior over the signal class being modeled.

  ![Learned Initializations Overview](/nerf-guide/learned-inits-overview.png)

- Learned intializations result in more than 10x faster training time and greatly improved generalization when only partial observations of a given signal are available.

  ![Learned Initializations Demo](/nerf-guide/learned-inits-demo.webp)

#### [pixelNeRF](https://alexyu.net/pixelnerf/) (Yu et al., CVPR 2021)

- Conditions a NeRF on image inputs in a fully convolutional manner.
- Allows the network to be trained across multiple scenes to learn a scene prior, enabling it to perform novel view synthesis in a feed-forward manner from a sparse set of views (as few as one).
- Every 5D spatial coordinate $(x, d)$ is combined with a corresponding image feature extracted from the feature volume $W$ via projection and interpolation. This combined feature is the input to the main NeRF MLP.

  ![pixelNeRF Framework](/nerf-guide/pixelnerf-framework.png)

#### [DietNeRF](https://ajayj.com/dietnerf/) (Jain et al., ICCV 2021)

- To improve generalizability in few-shot context, DietNeRF employs a pre-trained image encoder and a "semantic consistency loss" $\mathcal{L}_{SC}$ to guide the optimization process.
- Seeks to address two major problems with NeRF:
  1. NeRF tends to overfit on training views.
  2. Regularization fixes geometry, but hurts fine-detail.
- Uses [pixelNeRF](https://alexyu.net/pixelnerf/) as the main neural network

### Section 4: State of the Art

#### [VQAD](https://nv-tlabs.github.io/vqad/) (Takikawa et al., SIGGRAPH 2022)

#### [TODO: Paper here](https://to.do/)

## Part II: Fast Semantically Consistent Few Shot View Synthesis

This project plans to combine and implement subsequent works that propose applying computer graphics and meta-learning approaches to problems 1, 4, and 5. Our goal is to maximize NeRF's quality, capability, and generalizabilty on few-shot novel view synthesis tasks.

## Method

## Datasets

## Experiments

### Evaluation Metrics

- PSNR (Peak Signal-to-Noise Ratio): higher PSNR, lower MSE. Lower MSE implies less difference between the ground truth image and the rendered image. Thus, higher PSNR, better the model.

- SSIM (Structural Similarity Index): Checks the structural similarity with the ground truth image model. Higher SSIM, better the model.

- LPIPS (Learned Perceptual Image Patch Similarity): Determines the similarity with the view of perception using VGGNet. Lower LPIPS, better the model.

## Sources

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

[Improved Direct Voxel Grid Optimization for Radiance Fields Reconstruction](https://arxiv.org/abs/2206.05085) (Sun et al., Preprint 2022)

[Instant Neural Graphics Primitives with a Multiresolution Hash Encoding](https://nvlabs.github.io/instant-ngp/) (Müller et al., SIGGRAPH 2022)

[TensoRF: Tensorial Radiance Fields](https://apchenstu.github.io/TensoRF/) (Chen et al., ECCV 2022)

[Variable Bitrate Neural Fields](https://nv-tlabs.github.io/vqad/) (Takikawa et al., SIGGRAPH 2022)

[DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation](https://arxiv.org/abs/1901.05103) (Park et al., CVPR 2019)
