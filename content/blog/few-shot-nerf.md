---
author: "Vien Vuong"
title: "Instant Semantically Consistent Few Shot View Synthesis [Ongoing]"
date: "2022-09-23"
description: ""
tags: ["nerf", "3d", "computer-vision", "ml", "project"]
comments: false
socialShare: false
toc: true
math: true
draft: false
---

This is part of my final project for CS 543: Computer Vision at UIUC.

Group of 1: Vien Vuong (vienv2)

> <video src="/nerf-guide/nerf-demo.mp4" autoplay="true" controls="false" loop="true"></video> \
> Source: Mildenhall et al., ECCV 2020

## Abstract

Recent advances in machine learning has led to increased interest in employing coordinate-based neural networks as a promising tool for computer graphics for tasks such as view synthesis ([NeRF](https://www.matthewtancik.com/nerf)), radiance caching ([instant-ngp](https://nvlabs.github.io/instant-ngp/)), geometry representations ([DeepSDF](https://arxiv.org/abs/1901.05103)), and more (see survey paper [Neural Fields](https://neuralfields.cs.brown.edu/)). These methods, now called neural fields, approximate continuous 3D space with a countinuous, parametric function. Often, this parametric function is an MLP which takes in coordinates as input and output a vector (such as color or occupancy). Neural fields differ from previous signal representations like pixel images or voxels which are discrete and approximate continuous signals with regularly spaced samples of the signal. Neural fields have sparked an explosion of research interest, which have led to widespread success in problems such as 3D shape and image synthesis, animation of human bodies, 3D reconstruction, and pose estimation.

- Note: Neural fields are also known as implicit neural representations, neural implicits, or coordinate-based neural networks.

Of the neural field techniques, neural radiance fields (NeRF) is one of the most prominent and well-studied. Since 2020, significant improvements have been made to address limitations in every aspect of the original [NeRF](https://www.matthewtancik.com/nerf) implementation. While subsequent papers have achieved impressive results, little effort has been invested into combining and contrasting these very diverse approaches. This paper focuses on three state-of-the-art papers ([instant-ngp](https://nvlabs.github.io/instant-ngp/), [VQAD](https://nv-tlabs.github.io/vqad/) and [MonoSDF](https://to.do/)) that apply computer graphics and meta-learning optimizations to maximize neural volume rendering performance in two aspects:

1. Training and rendering performance
2. Generalizability

This paper consists of 2 sections:

- [**Part I**](#part-i-an-overview-of-neural-field-techniques) — Historical overview of progress leading up to Instant-NGP, VQAD, and MonoSDF. We will identify common trends and components of past methods (different conditioning, representation, forward map, architecture, and manipulation methods), and consolidate discovered knowledge in neural field research.

- [**Part II**](#part-ii-fast-semantically-consistent-few-shot-view-synthesis) — Implementation and performance evaluation of Instant-NGP + VQAD + MonoSDF.

## Goals

### Minimum Goal

NeRF has proven to be highly capable in novel view synthesis, being able to capture fine-grained details from lighting, texture, and shapes. However, the original NeRF implementation takes a significant amount of time and input views to train and render visually compelling novel views. Our goal with this project is to combine state-of-the-art techniques (Instant-NGP and VQAD) to maximize NeRF's quality, capability, and generalizabilty on few-shot novel view synthesis tasks.

We will conduct experiments on datasets [NeRF Synthetic Dataset](#nerf-synthetic-dataset) and [ShapeNetCore](#shapenetcore). We will evaluate novel view synthesis quality and training/rendering time compared to numbers given by other methods (see [Evaluation Metrics](#evaluation-metrics)).

Finally, we will study how generalizable our model is by pre-training on ShapeNetCore dataset, and then fine-tuning and testing on the NeRF dataset (or another [more complex dataset](#complex-datasets)). Specifically, we will consider the 1-shot (monocular), and 5-shot transfer-learning cases.

### Maximum Goal

If possible, we also plan to expand NeRF's performance to 3D reconstruction tasks. While NeRF is able to capture a scene very well, it is unclear how one would extract the 3D geometry. One could argue the volumetric density values NeRF generates are the geometry. However, inferring surfaces from densities proves to be rather non-trivial. Fortunately, there is a better representation for 3D shape called the signed distance function (SDF). Methods have been developed that allow NeRF to simultaneously predict SDF and perform volumetric rendering, while still training end-to-end. MonoSDF combines this approach with 3D geometric priors learned using a meta-learning approach to generate extremely detailed 3D scenes.

As a reach goal, we plan to conduct experiments on 3D reconstruction and object segmentation on the ScanNet dataset. Again, we will consider the 1-shot (monocular), and 5-shot learning cases. We would like to compare our methods with the best occupancy-networks- and voxel-grid-based methods.

## Part I: An Overview of Neural Field Techniques

### Section 1: Introduction to Neural Radiance Fields (NeRF)

After causing a big splash in ECCV 2020, the impressive [NeRF paper](https://www.matthewtancik.com/nerf) by Mildenhall et al. has kickstarted an explosion in interest in neural field research.

> ![NeRF Explosion](/nerf-guide/nerf-explosion.png) \
> Source: Xie et al., EUROGRAPHICS 2022

Neural radiance fields (NeRF) is a novel, data-driven approach that provides an efficient synthesis of visually compelling novel scenes from input images or videos. Impressively, NeRF allows explicit or implicit control of scene properties such as illumination, camera parameters, pose, geometry, appearance, and semantic structure, which has been impossible with previous photogrammetry or GAN-based approaches.

Arguably, the greatest contribution of the paper is its novel approach to storing 3D scenes as implicit representations. 3D geometry, texture, and lighting data can now be generated with a neural network. This volume rendering procedure is fully differentiable.

- Input: a single contious 5D coordinate (spatial location $(x, y, z)$ and viewing direction $(\theta, \phi)$)
- Output: volume density and RGB "color" (i.e., view-dependent emitted radiance)

This approach proves to be much more space-efficient and high-fidelity than discrete mesh-based, voxel-based, or multiplane image representations (but much slower, see below).

#### Algorithm

> ![NeRF Overview](/nerf-guide/nerf-overview.png) \
> Source: Mildenhall et al., ECCV 2020

The full algorithm to compute a neural radiance field is as such:

1. 5D coordinates (location and viewing direction) are sampled along camera rays casted onto the 3D object.
2. Optional: Hierarchical volume sampling (HVS) can be used here reduce sampling density in free space and occluded regions to improve sampling efficiency. The overall network architecture is composed of two networks: the "coarse" network (naive stratified sampling) and the "fine" network (informed biased sampling based on output of the "coarse" network). This affects how the loss function is computed.
3. Optional: Positional encoding (see [Fourier Features](https://bmild.github.io/fourfeat/)) can be used here to lift 5D input coordinates to a higher-dimensional space. The paper shows this can allow the MLP to better represent higher frequency variation in color and geometry. This encoding lifts the input coordinates from 5D to a higher-dimensional space.
4. The encoded coordinates are now pass into an MLP to produce a color and volume density. The network can be represented as a mapping $F_{\Theta}: (x, d) \rightarrow (c, \sigma)$, where $c = (r, g, b)$ is the emitted color, $\sigma$ is the volume density, $x$ is the 3D spatial location, and $d$ is the unit vector of the camera ray's viewing direction.
5. These color and volume density values can now be transformed into an image by a fully differentiable volume rendering procedure.
6. This differentiability allows end-to-end backpropagation from the rendering loss through the fully connected layers (MLP). The model is then optimized to minimize the residual between the synthesized and ground truth observed images.

To encourage viewpoint-free image synthesis, the NeRF restricts the network to predict the volume density $\sigma$ as a function of only the location $x$, while the RGB color $c$ is predicted as a function of both the location $x$ and the viewing direction $d$.

#### Loss Function

> ![NeRF Loss Function](/nerf-guide/nerf-loss.png) \
> Source: Mildenhall et al., ECCV 2020

The ultimate goal of this network is to predict the expected color value for the ray correctly. Since we can estimate the ground truth ray color with the ground truth 3D model, we can use L2-distance with the RGB values as a loss.

If HVS is used, the loss is the sum of the L2-norms of the coarse network and the fine network, such that both networks are well-optimized.

#### Limitations of NeRF and Subsequent Works

Unfortunately, the original implementation of NeRF has various limitations, some of which are:

1. Long training and rendering time (because of the neural network)
2. Can only represent static scenes and the generated views are not interactable.
3. Lighting is fixed, no support for relighting
4. NeRF needs to observe the geometry of every part of the scene to generate novel views. It cannot infer the geometry of occluded parts without help of priors.
5. A trained NeRF representation does not generalize to other scenes/objects. It tends to overfit to training views.

Most of these problems have been quickly identified and addressed by subsequent papers. For more details, see the comprehensive survey paper [Neural Fields](https://neuralfields.cs.brown.edu/). In the following [Section 2](#section-2-improving-performance) and [Section 3](#section-3-improving-generalizability), we will evaluate important papers from 2020-2022 that focus on problems 1, 4, and 5. Finally, these papers will lead up to a discussion about VQAD and **TODO** in [Section 4](#section-4-state-of-the-art), both of which will be combined and implemented in [Part II](#part-ii-fast-semantically-consistent-few-shot-view-synthesis).

### Section 2: Improving Performance

NeRF main bottleneck is having to train and query an MLP that maps 5D spatial coordinate to color and opacity. The MLP has to be trained for hundreds of thousands of iterations, and then queried millions of times to render new images (hundreds of forward passes for each of millions of pixels). Training and rendering time has been significantly reduced with early approaches such as:

- [AutoInt](https://arxiv.org/abs/2012.01714) restructures the coordinatebased MLP to compute ray integrals exactly, for >10x faster rendering with a small loss in quality.
- [Learned Initializations](https://www.matthewtancik.com/learnit) employs meta-learning on many scenes to start from a better MLP initialization, for both 10-20x faster training and better priors when per-scene data is limited.
- [DONeRF](https://depthoraclenerf.github.io/) achieves speedup by predicting a surface with a depth oracle network and sampling near the surface. This reduces the number of samples necessary for rendering each ray. Achieves 15-78x faster rendering time.

However, even with these optimizations, NeRF still proves to be too slow compared to classical volume rendering methods like voxel grids and multi-plane images (MPIs). It is clear that for the goal of instant training and real-time rendering to be achieved, the MLP itself will have to be modified.

In this section, we will go over six approaches that attempt to speed up NeRF training and/or rendering time. Some papers (Instant-NGP and VQAD) also manage to significantly reduce storage size and training time memory overhead. We will discuss methods that propose using some kind of specialized data structure to store the neural representation data and/or replacing the MLP altogether.

- Coninuous volume representation (same as original NeRF)

  1. Factorizing MLP into two MLPs: [FastNeRF](https://arxiv.org/abs/2103.10380)

- Discretized volume representation

  2. Sparse voxel octree with shallow MLP: [NGLOD](https://nv-tlabs.github.io/nglod/)
  3. Dense voxel grid with thousands of tiny MLPs: [KiloNeRF](https://nvlabs.github.io/instant-ngp/)
  4. Sparse voxel grid without any MLP: [Plenoxels](https://alexyu.net/plenoxels/)
  5. Dense voxel grid with shallow MLP: [DVGO](https://sunset1995.github.io/dvgo/) and [DVGOv2](https://arxiv.org/abs/2206.05085)
  6. Factorized components: [TensoRF](https://apchenstu.github.io/TensoRF/)
  7. Hash: [Instant-NGP](https://nvlabs.github.io/instant-ngp/)
  8. Hash: [VQAD](https://nv-tlabs.github.io/vqad/)

Instant-NGP and VQAD, will be included in [Section 4: State of the Art](#section-4-state-of-the-art). While VQAD's data structure is much faster compared to Instant-NGP, the _instant_ neural rendering algorithm remains the same and thus will be included in our implementation in [Part II](#part-ii-fast-semantically-consistent-few-shot-view-synthesis).

Note that Plenoxels, Instant-NGP, and DVGOv2 require custom CUDA kernels, while NGLOD, KiloNeRF, DVGO, and TensoRF achieve their speedup with standard PyTorch implementation. For the sake of simplicity, this paper will use the Instant-NGP's and VQAD's official PyTorch implementations available in NVIDIA Kaolin Wisp (see [Method](#method)).

#### [NGLOD](https://nv-tlabs.github.io/nglod/) (Takikawa et al., CVPR 2021)

- Neural signed distance functions (SDFs) are an alternative type of neural implicit representation of 3D scenes to NeRF. SDFs encode 3D surfaces with a function of position that returns the closest distance to a surface.

  > Neural SDFs \
  > ![Signed Distance Functions](/nerf-guide/nglod-sdf.jpg) \
  > Source: Takikawa et al., CVPR 2021

- Just like NeRF, SDFs have the same bottleneck where the MLP has to queried millions of times to render images.
- NGLOD proposes representing implicit surfaces using an octree-based feature volume which adaptively fits shapes with multiple discrete levels of detail (LODs), and multiple continuous LOD with SDF interpolation.
- Develops an efficient algorithm to directly render our novel neural SDF representation in real-time by querying only the necessary LODs with sparse octree traversal.

  > ![NGLOD](/nerf-guide/nglod-framework.jpg) \
  > Source: Takikawa et al., CVPR 2021

- Achieves 2-3 orders of magnitude more efficient in terms of rendering speed compared to previous works ().
- Produces SOTA reconstruction quality for complex shapes under both 3D geometric and 2D image-space metrics.

#### [FastNeRF](https://arxiv.org/abs/2103.10380) (Garbin et al., ICCV 2021)

- The first NeRF-based system capable of rendering high fidelity photorealistic images at 200fps on a high-end consumer GPU.
- Proposes factorizing NeRF into two neural networks to maximize caching:

  1. A position-dependent network that produces a deep radiance map
  2. A direction-dependent network that produces weights

  > ![FastNeRF](/nerf-guide/fastnerf-framework.png) \
  > Source: Garbin et al., ICCV 2021

- The inner product of the weights and the deep radiance map estimates the color in the scene at the specified position and as seen from the specified direction.
- Graphics-inspired factorization to allow for:

  1. Compactly caching a deep radiance map at each position in space.
  2. Efficiently querying that map using ray directions to estimate the pixel values in the rendered image.

- Achieves 3000 times faster rendering time compared to the original NeRF, and at least an order of magnitude faster than existing work on accelerating NeRF.

#### [KiloNeRF](https://nvlabs.github.io/instant-ngp/) (Reiser et al., SIGGRAPH 2022)

- Proposes utilizing thousands of tiny MLPs instead of one single large MLP.

  > ![FastNeRF](/nerf-guide/kilonerf-framework.png) \
  > Source: Reiser et al., SIGGRAPH 2022

- Each individual MLP only needs to represent parts of the scene, thus smaller and faster-to-evaluate MLPs can be used.
- Achieves three orders of magnitude in rendering time speedup compared to the original NeRF model.
- Further, using teacher-student distillation for training, we show that this speedup can be achieved without sacrificing visual quality.

#### [Plenoxels](https://alexyu.net/plenoxels/) (Yu et al., CVPR 2022)

- Plenoxels proposes to replace the NeRF MLP with a sparse voxel ("plenoxel") grid with density and spherical harmonic coefficients at each voxel.

  > ![Plenoxels Framework](/nerf-guide/plenoxels-framework.png) \
  > Source: Yu et al., CVPR 2022

- To render a ray, we compute the color and density via trilinear interpolation of the neighboring voxel coefficients. Then we integrate to render, and optimize using the standard MSE reconstruction loss relative to the training images, along with a total variation regularizer.

- Achieves >100x training time speedup. And while not optimized for fast rendering, Plenoxels can generate views at 15fps (>450x rendering time speedup).

  > ![Plenoxels Speed](/nerf-guide/plenoxels-speed.png) \
  > Source: Yu et al., CVPR 2022

- Key take-aways:
  - The key component in NeRF is the differentiable volumetric rendering, not the neural network.
  - Trilinear interpolation is key to achieve high resolution and better convergence (vs. nearest neighbor).
  - Regularization is important to prevent artifacts.

#### [DVGO](https://sunset1995.github.io/dvgo/) (Sun et al., CVPR 2022)

- Proposes representation consisting of a density voxel grid for scene geometry and a feature voxel grid with a shallow network for complex view-dependent appearance. 
- Modeling with explicit and discretized volume representations is not new, but we propose two simple yet non-trivial techniques that contribute to fast convergence speed and high-quality output. 
- First, we introduce the post-activation interpolation on voxel density, which is capable of producing sharp surfaces in lower grid resolution. 
- Second, direct voxel density optimization is prone to suboptimal geometry solutions, so we robustify the optimization process by imposing several priors. 
- Finally, evaluation on five inward-facing benchmarks shows that our method matches, if not surpasses, NeRF’s quality, yet it only takes about 15 minutes to train from scratch for a new scene.

> ![DVGO Speed](/nerf-guide/dvgo-speed.png) \
> Source: Sun et al., CVPR 2022

#### [DVGOv2](https://arxiv.org/abs/2206.05085) (Sun et al., 2022)

#### [TensoRF](https://apchenstu.github.io/TensoRF/) (Chen et al., ECCV 2022)

### Section 3: Improving Generalizability

The other major problem is NeRF needs to observe the geometry of every part of the scene to generate novel views; it cannot infer the geometry of occluded parts. This problem is exacerbated in low-data regime where not many views of the orginal scene is available.

Another related problem in this space is the poor generalizability of NeRF. By design, NeRF is supposed to overfit on a particular scene to be generate novel views as faithfully as possible. This means plain-old NeRF has no mechanism to easily generalize to new objects.

A potential solution to both these problems is to condition the model on some priors using assumptions about symmetry, sparseness, smoothness, etc. of objects. Priors

To generate visually comepelling novel views, we need a suitable prior over 3D surfaces. Handcrafting priors is limited in the complexity of heuristics that we can conceive. It is much better to learn such a prior from data, and these can be encoded within the parameters and architecture of a neural network.

#### [Learned Initializations](https://www.matthewtancik.com/learnit) (Tancik et al., CVPR 2021)

- Optimizing a coordinate-based network, such as NeRF, from randomly initialized weights for each new signal is inefficient.

- Using a MAML-based algorithm to learn initial weights enables faster convergence during optimization and can serve as a strong prior over the signal class being modeled.

  > ![Learned Initializations Overview](/nerf-guide/learned-inits-overview.png) \
  > Source: Tancik et al., CVPR 2021

- Learned intializations result in more than 10x faster training time and greatly improved generalization when only partial observations of a given signal are available.

  > ![Learned Initializations Demo](/nerf-guide/learned-inits-demo.webp) \
  > Source: Tancik et al., CVPR 2021

#### [pixelNeRF](https://alexyu.net/pixelnerf/) (Yu et al., CVPR 2021)

- Conditions a NeRF on image inputs in a fully convolutional manner.
- Allows the network to be trained across multiple scenes to learn a scene prior, enabling it to perform novel view synthesis in a feed-forward manner from a sparse set of views (as few as one).
- Every 5D spatial coordinate $(x, d)$ is combined with a corresponding image feature extracted from the feature volume $W$ via projection and interpolation. This combined feature is the input to the main NeRF MLP.

  > ![pixelNeRF Framework](/nerf-guide/pixelnerf-framework.png) \
  > Source: Yu et al., CVPR 2021

#### [DietNeRF](https://ajayj.com/dietnerf/) (Jain et al., ICCV 2021)

- To improve generalizability in few-shot context, DietNeRF employs a pre-trained image encoder and a "semantic consistency loss" $\mathcal{L}_{SC}$ to guide the optimization process.
- Seeks to address two major problems with NeRF:
  1. NeRF tends to overfit on training views.
  2. Regularization fixes geometry, but hurts fine-detail.
- Uses [pixelNeRF](https://alexyu.net/pixelnerf/) as the main neural network.

#### [VolSDF](https://lioryariv.github.io/volsdf/) (Yariv et al., NeurIPS 2021)

#### [NeuS](https://lingjie0206.github.io/papers/NeuS/) (Wang et al., NeurIPS 2021)

### Section 4: State of the Art

We consider the three papers Instant-NGP, VQAD, and Mono-SDF, each of which is considered "state-of-the-art" in their respective domain. We will go on to combine and implement them in [Part II](#part-ii-fast-semantically-consistent-few-shot-view-synthesis) of this paper. Due to the size and complexity of neural field research, there are many orthogonal methods which we are not able to address, yet can (and should) be considered "state-of-the-art" in their own right. These specific papers are chosen because we believe in the following reasons:

1. Their features are mutually compatible.
2. Their combined features optimize generalizability and training/rendering performance for few-shot novel view synthesis.
3. Individually, they have garnered a considerable amount of attention, which is a good heuristic for long-term development and support.

#### [Instant-NGP](https://nvlabs.github.io/instant-ngp/) (Müller et al., SIGGRAPH 2022)

TLDR: Using multiresolution hash to significantly reduce training and rendering time.

- To reduce training and testing time, Instant-NGP proposes a new input encoding that permits the use of a smaller network without sacrificing quality.
- In practice, this is a small NN augmented by a multiresolution hash table of trainable feature vectors whose values are optimized through SGD.
- The authors trust the multiresolution structure to be able to disambiguate hash collisions.
- Also creates a custom fully-fused CUDA kernel that parallelize this process while minimizing wasted bandwidth and compute operations.
- The commonality in these approaches is an encoding that maps model inputs to higher-dimensional space, which maximizes information extraction and enables smaller, more efficient MLP (NSVF, NGLOD).
- However, previous approaches rely on heuristics and structural modifications, which can complicate the training process, limit the method to a specific task, or limit performance on GPUs where control flow and pointer chasing is expensive.
- Instant-NGP addresses these concerns with multiresolution hash encoding configured by 2 values: number of parameters $T$ and the desired finest resolution $N_{\max}$.
- With the multiresolution hash tables, the paper is able to achieve task-independent adaptivity and efficiency:
  - Adaptivity: mapping a cascade of grids to corresponding fixed-size arrays of feature vectors. At coarse resolutions, the mapping is 1:1. At fine resolutions, multiple grid points can point to the same array entry. However, this is beneficial, as it causes the colliding training gradients to average, which would be dominated by the largest gradients. This means the hash table automatically prioritizes sparse areas with the most fine scale detail without the need for costly structural updates.
  - Efficiency: since the array is treated as a hash table, the lookups are $\mathcal{O}(1)$ and do not require control flow. This helps GPUs bypass execution divergence, serial pointer-chasing, and allows for all resolutions to be queried in parallel.
- Positional encodings with a multiresolution sequence of sine and cosine functions are used in the attention layer of transformers. Now they are adopted by NeRF to encode the spatio-directionally varying light field and volume density in the NeRF algorithm.
- Key idea: parametric encodings. Arranges additional trainable parameters (beyond weights and biases) in an auxiliary data structure, and interpolates these parameters depending on the input vector $x$.
  - Trades larger memory footprint for smaller computational cost: instead of optimizing instead every weight in an MLP with backprop, a trilinearly interpolated 3D grid only needs to update 8 grid points per backprop.
  - Another parametric approach: trains a large auxiliary coordinate encoder neural network (ACORN) to output dense feature grids in the leaf nodes around $x$. Better adaptivity at greater computational cost which can be amortized when sufficiently many inputs $x$ fall into each leaf node.
- Key idea: sparse parametric encoding. There are several problems with dense grids of trainable features:
  - It consumes much more memory than NN weights.
  - It allocates as many features to empty space as it does to near-surface areas.
  - Multiresolution decomposition can much better model smoothness in natural scenes.
- NSVF adopts a multi-stage coarse to fine strategy in which regions of the feature grid are progressively refined and culled away as necessary, which requires updating the sparse data structure and complicating the training process.
- Instant-NGP combines both ideas to reduce waste:
  - Stores the trainable feature vectors in a compact spatial hash table, whose hyperparameter $T$ can be adjusted to trade number of parameters for reconstruction quality.
  - Uses multiple separate hash tables indexed at different resolution whose interpolated outputs are concatenated before being passed through the MLP. This results in comparable reconstruction quality to dense grids while using 20x fewer parameters.
- Do not explicitly handle collisions of the hash functions (probing, bucketing, chaining). Instead, the authors rely on the NN to learn to disambiguate hash collisions itself.
- The multiresolution hash encoding algorithm is as such:
  1. For an input coordinate $x$, find surrounding voxels at $L$ resolution levels. Assign indices to the corners by hashing their integer coordinates (for coarse resolutions, this mapping is 1:1 as the corners are merged).
  2. Look up corresponding $F$-dimensional feature vectors from hash tables $\theta_{l}$.
  3. Linearly interpolate them based on the relative location of input $x$ within the respective $l$-th voxel.
  4. Concatenate the results of each level (and auxiliary inputs $\xi$) to produce the encoded MLP input $y$.
  5. Encoding is trained via backprop through the MLP, the concatenation, the linear interpolation, and then accumulated in the looked-up feature vectors.
- This encoding lifts model input $x \in \mathbb{R}^{d}$ to MLP input $y \in \mathbb{R}^{LF+E}$ where $L$ is the number of levels, $F$ is the dimension of the feature vector, and $E$ is the dimension of the auxiliary inputs $\xi \in \mathbb{R}^{\xi}$. However, only two encoding parameters out of these have to be tuned: hash table size $T$ and max resolution $N_{\max}$ (since adequate optima can be found for the other hyperparameters).
  - An example of auxiliary inputs $\xi$ are the view direction and material density and texture. They can be encoded using more efficient methods like one-blob encoding in NRC and spherical harmonics basis in NeRF.
  - Inputs $x$ are encoded into $y = enc(x; \theta)$, which are inputs to the MLP $m(y; \Phi)$. The model thus not only has trainable weights $\Phi$ but also trainable encoding parameters $\theta$.
- Hash table size $T$ provides a trade-off between performance, memory, and quality. Higher $T$ result in higher quality and lower performance. Memory footprint scales linearly in $T$, while quality and performance scale sub-linearly (diminishing returns).
- It seems counter-intuitive that encoding is able to faithfully reconstruct scenes in the presence of hash collisions. However, the authors found that different resolution levels have different strengths that complement each other. Coarse levels observe no collision but are low-resolution, while fine levels can capture small features but suffer from collisions. Luckily, collisions are pseudo-randomly scattered across space, so it is statiscally unlikely for them to occur simultaneously at every level for a given pair of points. Furthermore, hash collisions are equivalent to averaging the gradients of the training samples, which means they tend to favor high visibility and high density surfaces as opposed to empty space. So while hash collision hurts reconstruction quality, the end result is not catastrophic.
- If inputs $x$ are more concentrated in a small region, finer grid levels will experience fewer collisions and a more accurate function can be learned. The multiresolution hash encoding is able to automatically adapt to the input distribution like tree-based encodings, while not suffering from costly data structure modifications and discrete jumps during training.
- Instant-NGP keeps the hash tables differentiable through trilinear interpolation, which results in sufficient smoothness and prevents blocky appearance. However, if higher-order smoothness must be guaranteed like in SDFs, an alternative interpolation scheme can be used with a small decrease in quality.
- NOTE: Instant-NGP uses a modified PCG32 RNG hash function. As a possible improvement, the hash function can be learned instead of hand-crafted, turning the method into a dictionary-learning approach. Two possible avenues are:

  1. Continuous formulation of indexing that is amenable to analytic differentiation.
  2. Evolutionary optimization that efficiently explores the discrete function space.

- Instant-NGP is special compared to previously mentioned methods: its multiresolution hash encoding and specialized kernels are task-agnostic! Experiments show near SOTA performance in 4 different tasks at a fraction of the computational cost:

  1. Gigapixel: representing a gigapixel image by a neural network.
  2. SDF: learning a signed distance function in 3D space whose zero level-set represents a 2D surface.
  3. Neural radiance caching [(NRC)](https://research.nvidia.com/publication/2021-06_real-time-neural-radiance-caching-path-tracing) [Müller et al., 2021]: employing a neural network that is trained in real-time to cache costly lighting calculations.
  4. [NeRF](https://www.matthewtancik.com/nerf) [Mildenhall et al. 2020]: uses 2D images and their camera poses to reconstruct a volumetric radiance-and-density field that is visualized using ray marching.

  - Instant-NGP uses the same implementation and hyperparameters across all tasks and only vary the hash table size which trades off quality and performance.

- Some optimizations specific to NeRF:
  - For the NeRF implementation, the model consists of 2 concatenated MLPs: a density MLP (3D input) which feeds into a color MLP (5D input). The output of the color MLP is an RGB triplet, and its input is the concatenation of:
    - The 16 output values of the density MLP
    - The view direction projected onto the first 16 coefficients of the spherical harmonics basis (i.e. up to degree 4). This is a natural frequency encoding over unit vectors.
  - For ray marching, Instant-NGP also employs an occupancy grid that coarsely marks empty vs. dense space to minimize wasted computation. In large scenes, the occupancy grid can also be cascaded to distribute samples exponentially instead of uniformly along the ray.
- For NeRF rendering, Instant-NGP achieves more-or-less real-time performance: at 1080p resolution, Instant-NGP finishes training in a matter of seconds, and render novel scene in tens of milliseconds. Compared to Plenoxels, one of the fastest model at the time of publish, Instant-NGP achieves:
  - 4x faster rendering at a much higher resolution (1920x1080 60fps vs 800x800 15fps)
  - Comparable PSNR (see [Evaluation Metrics](#evaluation-metrics)) at 5s of training time compared to Plenoxels at 11 mins.
  - Superior PSNR vs. Plenoxel when trained for 1 min or longer.
  - Superior PSNR vs. mip-NeRF (one of the highest fidelity NeRF) when trained for 5 min or longer. Note that mip-NeRF takes multiple hours to finish training.
  - This is a massive improvement over Plenoxels, especially considering Plenoxels is submitted merely a month before Instant-NGP.
- Again for NeRF rendering, the author conduct an ablation where the entire NN is replaced with a single linear matrix multiplication similar to concurrent direct voxel-based NeRF (DVGO). The results show that quality is significantly compromised compared to the MLP, which is better able to capture specular effects and to resolve hash collisions across the interpolated multiresolution hash tables. The MLP is also shown to be only 15% more expensive than the linear layer, thanks to its small size and efficient implementation.
- For SDF rendering, Instant-NGP achieves almost identical fidelity against an optimized NGLOD model, which achieves SOTA in both speed and quality. The difference in quality can be attributed to hash collisions in Instant-NGP.
  - However, Instant-NGP's SDF is defined everywhere within the training volume as opposed to NGLOD, which is only defined within the octree (i.e. close to the surface). This permits the use of certain SDF rendering techniques such as approximate soft shadows.
  - The results also show that for Instant-NGP, the rendered colors are sensitive to slight changes in the surface normal, resulting in undesired microstructures on the scale of the finest grid resolution (a kind of "overfitting"). Since these artifacts are absent in NGLOD, the authors attribute them to hash collisions. They could potentially be eliminated by filtering hash table lookups or by imposing an additional smoothness prior on the loss.
- NOTE: In generative settings, parametric input encodings typically arrange their features in a dense grid which can then be populated by a separate generator network such as StyleGAN. There is a problem: the hash encoding adds an additional layer of complexity in that the features are not bijective with a regular grid of points.

- Last but not least, one of the most important contributions of Instant-NGP is its fast fully-fused CUDA kernels for the MLP, which allows for massive parallelization.

> Both videos are in real time. The training is just that quick!
> <video src="/nerf-guide/ingp-demo1.mp4" autoplay="true" controls="false" loop="true"></video> \
> <video src="/nerf-guide/ingp-demo2.mp4" autoplay="true" controls="false" loop="true"></video> \
> Source: Müller et al., SIGGRAPH 2022

- In conclusion, Instant-NGP has the best trade-off in terms of training and rendering time, and the closest thing we have to real-time interactive neural rendering.

#### [VQAD](https://nv-tlabs.github.io/vqad/) (Takikawa et al., SIGGRAPH 2022)

#### [MonoSDF](https://niujinshuchong.github.io/monosdf/) (Yu et al., NeurIPS 2022)

**TLDR:** Using the monocular geometric priors generated by general-purpose monocular estimators to improve reconstruction quality and optimization time.

- Current neural implicit-based surface reconstruction approaches achieve impressive results for simple scenes. However, they struggle in the presence of few input views and for scene that contain large textureless regions (e.g. a blank wall).
- A key reason for this is that these model are optimized using a per-pixel RGB reconstruction loss. Using only RGB images as input leads to an underconstrained problem as there are countless photo-consistent ways you could inpaint the sparsely sampled regions.
- This paper proposes using monocular geometric cues such as depths and normals, which can be computed efficiently, to aid in neural implicit surface reconstruction methods. As such, the estimated depths and normals of each image are used as additional supervision signal during optimization together with the RGB image reconstruction loss.

> ![MonoSDF](/nerf-guide/monosdf-framework.png) \
> Source: Yu et al., NeurIPS 2022

- Photometric consistency used by surface reconstruction methods and recognition cues used by monocular networks are complementary. Photoconsistency works better for textured regions and fine details, while recognition cues, such as surface normals and relative depth cues, excel in large textureless regions like walls.

- The paper also conducts extensive experiments on SOTA coordinate-based neural representations in the context of implicit surface reconstruction.
  - MLPs act globally and exhibit an inductive smoothness bias while being computationally expensive to optimize and evaluate.
  - Grid-based representations benefit from locality during training and rendering, hence computationally more efficient. However, reconstructions are noisier for sparse views or less-observed areas.
- Including monocular geometric priors improves implicit reconstruction across the board while exhibiting very fast convergence.

- Signed distance function (SDF) is a continuous function that for any given 3D point, returns the point's distance to the closest surface.
- There are multiple ways we could represent SDF:
  - Explicit as a dense grid of learnable SDF values. Use interpolation to query the SDF value $\hat{s}$ for an arbitrary point $x$ from the dense SDF grid $\mathcal{G}\_{\theta}$.
    $$\hat{s} = \text{interp}(x, \mathcal{G}\_{\theta})$$
  - Implicit as a single MLP. SDF values are the outputs of the MLP $f\_{\theta}$ where $\gamma(x)$ is the positional encoding of $x$.
    $$\hat{s} = f\_{\theta}(\gamma(x))$$
  - Hybrid using a (feature-conditioned) MLP decoder in combination with single- or multi-resolution feature grids.
    - Single-resolution: Each cell of the grid stores a feature vector instead of directly storing SDF values. The input to the MLP $f\_{\theta}$ is the interpolated local feature vector from the 3D feature grid $\Phi\_{\theta}$.
      $$\hat{s} = f\_{\theta}(\gamma(x), \text{interp}(x, \Phi\_{\theta}))$$
    - Multi-resolution: Multiple feature grids $\lbrace \Phi\_{\theta}^{l} \rbrace \_{l=1}^{L}$ for multiple resolutions $R\_{l}$. The resolutions are sampled in geometric space to combine feature at different frequencies. The input to the MLP is the concatenated value of all the interpolated feature vectors at every resolution. As the number of grid cells grow cubically, one can use a spatial hash function to index the feature vector at finer levels (see [Instant-NGP](#instant-ngp-müller-et-al-siggraph-2022)).
      $$\hat{s} = f\_{\theta}(\gamma(x), \lbrace \text{interp}(x, \Phi\_{\theta}^{l}) \rbrace \_{l})$$
- In addition to 3D geometry (SDF), one can also predict the color values optimized with reconstruction loss. We define a second function:
  $$\hat{c} = c\_{\theta}(x, v, \hat{n}, \hat{z})$$
  - $c_{\theta}$ is parameterized with a two-layer MLP with network weights $\theta$
  - If same as NeRF, the MLP takes in a 5D coordinate consisting of the 3D location $x$ and the camera viewing direction $v$ (two angles).
  - In addition, the MLP can also takes in the 3D unit normal $\hat{n}$ (analytical gradient of SDF function), and the feature vector $\hat{z}$.
    - If SDF parameterized with an MLP: $\hat{z}$ is the output of a second linear head of the SDF network.
    - If SDF parameterized with a dense feature grid: $\hat{z}$ is the interpolated vector from the grid.
- Our goal here is to unify volume rendering (NeRF) with monocular geometric priors to improve implicit surface generation. We expect the 2 types of geometric priors, depth and normal, to complement each other.
  - Monocular Depth Cues: Obtained with monocular depth predictor pretrained on the Omnidata dataset. This model predict a depth map $\bar{D}$ for each input RGB image.
    - Absolute scale is difficult to estimate, so $\bar{D}$ must be considered as a relative cue.
    - This relative depth information is provided over larger distances in the image.
  - Monocular Normal Cues: Same pretrained Omnidata model predicting a normal map $\bar{N}$ for each RGB image.
    - Unlike depth cues that provide semi-local relative information, normal cues are local and capture geometric detail.

## Part II: Fast Semantically Consistent Few Shot View Synthesis

This project plans to combine and implement subsequent works that propose applying computer graphics and meta-learning approaches to problems 1, 4, and 5. Our goal is to maximize NeRF quality, capability, and generalizabilty on few-shot novel view synthesis tasks.

## Method

### Framework

#### Kaolin Wisp

I will use NVIDIA Kaolin Wisp to implement my enhanced NeRF model.

> ![Kaolin Wisp](/nerf-guide/kaolin-wisp.jpg) \
> Source: [NVIDIA Kaolin Wisp](https://github.com/NVIDIAGameWorks/kaolin-wisp)

Kaolin Wisp is a PyTorch library powered by NVIDIA Kaolin Core to work with neural fields (including NeRFs, NGLOD, Instant-NGP and VQAD). Kaolin Wisp provides a PyTorch interface to the custom CUDA kernels proposed by [VQAD](https://nv-tlabs.github.io/vqad/). This allows me to implement my version of VQAD entirely in Python and not have to worry about low-level CUDA code.

Kaolin Wisp also provides:

1. A set common utility functions (datasets, image I/O, mesh processing, and ray utility functions)
2. Building blocks like differentiable renderers and differentiable data structures (octrees, hash grids, triplanar features)
3. Debugging visualization tools, interactive rendering and training, logging, and trainer classes.

These features are incredibly useful for building complex neural fields, and will certainly speed up and simplify my prototyping process.

### Reservations

## Experiments

### Evaluation Metrics

- PSNR (Peak Signal-to-Noise Ratio): higher PSNR, lower MSE. Lower MSE implies less difference between the ground truth image and the rendered image. Thus, higher PSNR, better the model.

- SSIM (Structural Similarity Index): Checks the structural similarity with the ground truth image model. Higher SSIM, better the model.

- LPIPS (Learned Perceptual Image Patch Similarity): Determines the similarity with the view of perception using VGGNet. Lower LPIPS, better the model.

### Datasets

#### NeRF Synthetic Dataset

Introduced by the original NeRF paper, Mildenhall et al., ECCV 2020. This will be used as a baseline to compare quality metrics (PSNR, SSIM, LPIPS) against other NeRF papers.

The dataset contains three parts with the first 2 being synthetic renderings of objects called Diffuse Synthetic 360◦ and Realistic Synthetic 360◦ while the third is real images of complex scenes. We will test our model on the second and third parts of the dataset (the first part is too simple).

- Realistic Synthetic 360◦ consists of eight objects of complicated geometry and realistic non-Lambertian materials.

  > <video src="/nerf-guide/nerf-demo.mp4" autoplay="true" controls="false" loop="true"></video>

- The real images of complex scenes consist of 8 forward-facing scenes captured with a cellphone at a size of 1008x756 pixels.

  > <video src="/nerf-guide/nerf-data-realistic.mp4" autoplay="true" controls="false" loop="true"></video>

#### ShapeNetCore

Introduced by ShapeNet: An Information-Rich 3D Model Repository, Chang et al., 2015. This will also be used as a baseline to compare quality metrics (PSNR, SSIM, LPIPS) against other NeRF papers. Since this dataset has many objects of the same classes, it can be used as a pretraining dataset before testing the model on the NeRF dataset or another novel view synthesis dataset.

> ![ShapeNetCore](/nerf-guide/shapenetcore.png)
> Source: Chang et al., 2015

#### Complex Datasets

These are more complex datasets that can potentially be used to test the ability of our model to infer 3D geometry.

- [ScanNet](http://www.scan-net.org/): realistic indoor scene dataset with 3D camera poses, surface reconstructions, and instance-level semantic segmentations.
- [Tanks and Temples](https://www.tanksandtemples.org/): realistic indoor and outdoor scenes captured with an industrial laser scanner at very high fidelity.

## Reflection

This is my first time working with neural radiance fields. I have experience working with PyTorch and 2D inapainting tasks, however, 3D computer vision is one of my long-time passion. This project is tangentially related to my Master thesis research, which focuses more on 3D geometry extraction (potentially by) using occupancy networks, as oppsed to novel view synthesis. However, if I observe good results from my experiments in this project, it might steer the direction of my research. Refer to [Goals](#goals) for information on the scope of the project.

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

[MonoSDF: Exploring Monocular Geometric Cues for Neural Implicit Surface Reconstruction](https://niujinshuchong.github.io/monosdf/) (Yu et al., NeurIPS 2022)

[Volume Rendering of Neural Implicit Surfaces](https://lioryariv.github.io/volsdf/) (Yariv et al., NeurIPS 2021)

[NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction](https://lingjie0206.github.io/papers/NeuS/) (Wang et al., NeurIPS 2021)

### Websites

**Frank Dellaert's Blog Posts**

- [NeRF at CVPR 2022](https://dellaert.github.io/NeRF22/)
- [NeRF at ICCV 2021](https://dellaert.github.io/NeRF21/)
- [NeRF Explosion 2020](https://dellaert.github.io/NeRF/)

**GitHub Repositories**

- [NVIDIA Kaolin Wisp](https://github.com/NVIDIAGameWorks/kaolin-wisp)
- [Awesome Neural Radiance Fields](https://github.com/yenchenlin/awesome-NeRF)
