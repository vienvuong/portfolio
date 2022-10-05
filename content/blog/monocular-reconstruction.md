---
author: "Vien Vuong"
title: "NeRF in Monocular Human Reconstruction [Ongoing]"
date: "2022-09-29"
description: ""
tags: ["nerf", "3d", "computer-vision", "ml", "project"]
comments: false
socialShare: false
toc: true
math: true
draft: false
---

This is part of my final project for ECE 544: Pattern Recognition at UIUC.

## Introduction

### Neural Radiance Fields (NeRFs)

<video src="/nerf-guide/nerf-demo.mp4" autoplay="true" controls="false" loop="true"></video>

After causing a big splash in ECCV 2020, the impressive NeRF paper by Mildenhall et al. has kickstarted an explosion in interest in the field of neural volume rendering. It is a novel, data-driven approach that provides an efficient synthesis of visually compelling novel scenes from input images or videos.

NeRF also allows explicit or implicit control of scene properties such as illumination, camera parameters, pose, geometry, appearance, and semantic structure, which has been impossible with previous photogrammetry or GAN-based approaches.

Arguably, the greatest contribution of the paper is its approach to representing 3D scenes as 5D neural radiance fields:

- Input: a single contious 5D coordinate (spatial location $(x, y, z)$ and viewing direction $(\theta, \phi)$)
- Output: volume density and RGB "color" (i.e., view-dependent emitted radiance)

This approach proves to be much more space-efficient and high-fidelity than discrete mesh-based or voxel-based representations.

#### Algorithm

![NeRF Overview](/nerf-guide/nerf-overview.png)

The full algorithm to compute a neural radiance field is as such:

1. 5D coordinates (location and viewing direction) are sampled along camera rays casted onto the 3D object.
2. Optional: Hierarchical volume sampling (HVS) can be used here reduce sampling density in free space and occluded regions to improve sampling efficiency. The overall network architecture is composed of two networks: the "coarse" network (naive stratified sampling) and the "fine" network (informed biased sampling based on output of the "coarse" network). This affects how the loss function is computed.
3. Optional: Positional encoding (also called gamma encoding) can be used here to lift 5D input coordinates to a higher-dimensional space. The paper shows this can allow the MLP to better represent higher frequency variation in color and geometry. This encoding lifts the input coordinates from 5D to a higher-dimensional space.
4. The encoded coordinates are now pass into an MLP to produce a color and volume density. The network can be represented as a mapping $F_{\Theta}: (x, d) \rightarrow (c, \sigma)$, where $c = (r, g, b)$ is the emitted color, $\sigma$ is the volume density, $x$ is the 3D spatial location, and $d$ is the unit vector of the camera ray's viewing direction.
5. These color and volume density values can now be transformed into an image by a fully differentiable volume rendering procedure.
6. This differentiability allows end-to-end backpropagation from the rendering loss through the fully connected layers (MLP). The model is then optimized to minimize the residual between the synthesized and ground truth observed images.

To encourage viewpoint-free image synthesis, the NeRF restricts the network to predict the volume density $\sigma$ as a function of only the location $x$, while the RGB color $c$ is predicted as a function of both the location $x$ and the viewing direction $d$.

#### Loss Function

![NeRF Loss Function](/nerf-guide/nerf-loss.png)

The ultimate goal of this network is to predict the expected color value for the ray correctly. Since we can estimate the ground truth ray color with the ground truth 3D model, we can use L2-distance with the RGB values as a loss.

If HVS is used, the loss is the sum of the L2-norms of the course network and the fine network, such that both networks are well-optimized.

#### Discussion

However, the original implementation of NeRF has various drawbacks, many of which have been quickly identified and addressed by subsequent papers.

1. Slow training and slow rendering
2. Can only represent static scenes
3. Lighting is fixed, no support for relighting
4. A trained NeRF representation does not generalize to other scenes/objects

[Source: Frank Dellaert's blog]


## Dataset

I use a subset of the [SAIL-VOS 3D dataset](http://sailvos.web.illinois.edu/_site/_site/index.html) (Hu et al., CVPR 2021) provided by Prof. Alexander Schwing for the final project.

