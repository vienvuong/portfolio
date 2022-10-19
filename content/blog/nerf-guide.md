---
author: "Vien Vuong"
title: "A Deep Dive Into Neural Radiance Fields (NeRFs) [Ongoing]"
date: "2022-09-24"
description: "A chronicle of the explosive growth of neural radiance fields from 2020-2022. I will attempt to summarize around 50 important NeRF-related papers and outline how they improve upon the original paper."
tags: ["nerf", "3d", "computer-vision", "ml", "research"]
comments: false
socialShare: false
toc: true
math: true
cover:
  src: /nerf-guide/nerf-cover.jpg
  alt: Neural Radiance Fields Cover
---

## Introduction

Recent advances in machine learning has led to increased interest in employing coordinate-based neural networks as a promising tool for computer graphics for tasks such as view synthesis ([NeRF](https://www.matthewtancik.com/nerf)), radiance caching ([instant-ngp](https://nvlabs.github.io/instant-ngp/)), geometry representations ([DeepSDF](https://arxiv.org/abs/1901.05103)), and more (see survey paper [Neural Fields](https://neuralfields.cs.brown.edu/)). These methods, now called neural fields, approximate continuous 3D space with a countinuous, parametric function. Often, this parametric function is an MLP which takes in coordinates as input and output a vector (such as color or occupancy). Neural fields differ from previous signal representations like pixel images or voxels which are discrete and approximate continuous signals with regularly spaced samples of the signal. Neural fields have sparked an explosion of research interest, which have led to widespread success in problems such as 3D shape and image synthesis, animation of human bodies, 3D reconstruction, and pose estimation.

- Note: Neural fields are also known as implicit neural representations, neural implicits, or coordinate-based neural networks.

### Neural Radiance Fields (NeRFs)

<video src="/nerf-guide/nerf-demo.mp4" autoplay="true" controls="false" loop="true"></video>

After causing a big splash in ECCV 2020, the impressive NeRF paper by Mildenhall et al. has kickstarted an explosion in interest in the field of neural volume rendering. It is a novel, data-driven approach that provides an efficient synthesis of visually compelling novel scenes from input images or videos.

NeRF also allows explicit or implicit control of scene properties such as illumination, camera parameters, pose, geometry, appearance, and semantic structure, which has been impossible with previous photogrammetry or GAN-based approaches.

Arguably, the greatest contribution of the paper is its approach to representing 3D scenes as 5D neural radiance fields:

- Input: a single contious 5D coordinate (spatial location $(x, y, z)$ and viewing direction $(\theta, \phi)$)
- Output: volume density and RGB "color" (i.e., view-dependent emitted radiance)

This approach proves to be much more space-efficient and high-fidelity than discrete mesh-based, voxel-based, or point-cloud-based representations.

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

### Follow-Up Works

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

#### [VolSDF](https://lioryariv.github.io/volsdf/) (Yariv et al., NeurIPS 2021)

#### [NeuS](https://lingjie0206.github.io/papers/NeuS/) (Wang et al., NeurIPS 2021)

#### Fundamentals

###### [Mip-NeRF](https://jonbarron.info/mipnerf/)

- Address the severe aliasing artifacts from vanilla NeRF by adapting the mip-map idea from graphics and replacing sampling the light field by integrating over conical sections along a the viewing rays.

###### [MVSNeRF](https://apchenstu.github.io/mvsnerf/)

- Trains a model across many scenes and then renders new views conditioned on only a few posed input views, using intermediate voxelized features that encode the volume to be rendered.

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

###### [UNISURF](https://arxiv.org/abs/2104.10078)

- Propose to replace the density in NeRF with occupancy, and hierarchical sampling with root-finding, allowing to do both volume and surface rendering for much improved geometry.

###### [NerfingMVS](https://weiyithu.github.io/NerfingMVS/)

- Use a sparse depth map from an SfM pipeline to train a scene-specific depth network that subsequently guides the adaptive sampling strategy in NeRF.

#### Priors

###### [Learned Initializations for Optimizing Coordinate-Based Neural Representations](https://arxiv.org/abs/2012.02189) (Tancik et al., 2020)

- Optimizing a coordinate-based network, such as NeRF, from randomly initialized weights for each new signal is inefficient.
- Using meta-learned initial weights enables faster convergence during optimization and can serve as a strong prior over the signal class being modeled, resulting in better generalization when only partial observations of a given signal are available.

![Learned Initializations Overview](/nerf-guide/learned-inits-overview.png)

![Learned Initializations Demo](/nerf-guide/learned-inits-demo.webp)

###### [Dense Depth Priors for NeRF](https://barbararoessle.github.io/dense_depth_priors_nerf/)

- Estimates depth using a depth completion network run on the SfM point cloud in order to constrain NeRF optimization, yielding higher image quality on scenes with sparse input images.

Neural radiance fields (NeRF) encode a scene into a neural representation that enables photo-realistic rendering of novel views. However, a successful reconstruction from RGB images requires a large number of input views taken under static conditions — typically up to a few hundred images for room-size scenes. Our method aims to synthesize novel views of whole rooms from an order of magnitude fewer images. To this end, we leverage dense depth priors in order to constrain the NeRF optimization. First, we take advantage of the sparse depth data that is freely available from the structure from motion (SfM) preprocessing step used to estimate camera poses. Second, we use depth completion to convert these sparse points into dense depth maps and uncertainty estimates, which are used to guide NeRF optimization. Our method enables data-efficient novel view synthesis on challenging indoor scenes, using as few as 18 images for an entire scene.

###### [Depth-supervised NeRF](https://www.cs.cmu.edu/~dsnerf/)

- Also uses a depth completion network on structure-from-motion point clouds to impose a depth-supervised loss for faster training time on fewer views of a given scene.

###### [InfoNeRF](http://cvlab.snu.ac.kr/research/InfoNeRF)

- Penalizes the NeRF overfitting ray densities on scenes with limited input views through ray entropy regularization, resulting in higher quality depth maps when rendering novel views.

###### [RapNeRF](https://arxiv.org/abs/2205.05922)

- Focuses on view-consistency to enable view extrapolation, using two new techniques: random ray casting and a ray atlas. [(pdf)](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Ray_Priors_Through_Reprojection_Improving_Neural_Radiance_Fields_for_Novel_CVPR_2022_paper.pdf)

###### [RegNeRF](https://m-niemeyer.github.io/regnerf/)

- Enables good reconstructions from a view images by renders patches in _unseen_ views and minimizing an appearance and depth smoothness prior there.

#### Performance

###### [Neural Sparse Voxel Fields](https://github.com/facebookresearch/NSVF) (Liu et al., 2020)

![NSVF Pipeline](/nerf-guide/nsvf-pipeline.jpg)

- Hybrid scene representation that combines neural implicit fields with an explicit sparse voxel structure
- Defines a set of voxel-bounded implicit fields organized in a sparse voxel octree to model local properties in each cell
- Utilizes the sparse voxel structure to achieve efficient rendering by skipping the voxels containing no relevant scene content
- Progressive training strategy that efficiently learns the underlying sparse voxel structure with a differentiable ray-marching operation from a set of posed 2D images in an end-to-end manner
- 10 times faster than NeRF
- Applications: scene editing, scene composition, multi-scene learning, free-viewpoint rendering of a moving human, and large-scale scene rendering

###### [NeRF++: Analyzing and Improving Neural Radiance Fields](https://github.com/Kai-46/nerfplusplus) (Zhange et al., 2020)

- Analyzes the shape-radiance ambiguity: from certain viewpoints, NeRF can recover the wrong geometry from the radiance information.
- NeRF's MLP is quite robust against the shape-radiance ambiguity because of 2 reasons:
  1. Incorrect geometry forces the radiance field to have higher intrinsic complexity (i.e., much higher frequencies w.r.t $d$). Higher complexity required for incorrect shapes is more difficult to represent with a limited capacity MLP.
  2. NeRF's specific MLP structure favors $x$ more than $d$. This means the model encodes an implicit prior favoring smooth surface reflectance functions where $c$ is smooth with respect to $d$ at any given surface point $x$.
- Scene background that is very far away can cause resolution problems.
- Presents a novel spatial parameterization scheme (inverted sphere parameterization) that integrates over the normalized device coordinates (NDC) space instead of the Euclidean space.

I am most interested in applications of NeRF in 3D object reconstruction and depth estimation problems.

###### [DeRF: Decomposed Radiance Fields](https://ubc-vision.github.io/derf/) (Rebain et al., 2020)

- Propose to spatially decompose a scene into “soft Voronoi diagrams” and dedicate smaller networks for each decomposed part to take advantage of accelerator memory architectures.
- Achieves near-constant inference time regardless of the number of decomposed parts
- Provides up to 3x more efficient inference than NeRF (with the same rendering quality)

###### [AutoInt: Automatic Integration for Fast Neural Volume Rendering](https://www.computationalimaging.org/publications/automatic-integration/) (Lindell et al., 2020)

- NeRF's volume integrations along the rendered rays during training and inference have high computational and memory requirements
- Proposes automatic integration to directly learn the volume intergral using implicit neural representation networks
  > For training, we instantiate the computational graph corresponding to the derivative of the implicit neural representation. The graph is fitted to the signal to integrate. After optimization, we reassemble the graph to obtain a network that represents the antiderivative. By the fundamental theorem of calculus, this enables the calculation of any definite integral in two evaluations of the network.
- Greater than 10x less computation requirements than NeRF.

![AutoInt Framework](/nerf-guide/autoint-framework.png)

###### [JaxNeRF](https://github.com/google-research/google-research/tree/master/jaxnerf) (GitHub Repo)

- Uses [JAX](https://github.com/google/jax) to dramatically speed up training, from days to hours.

###### [FastNeRF](https://arxiv.org/abs/2103.10380)

- Factorizes the NeRF volume rendering equation into two branches that are combined to give the same results as NeRF, but allow for much more efficient caching, yielding a 3000x speed up.

###### [SNeRG](https://arxiv.org/abs/2103.14645)

- Precompute and "bake" a NeRF into a new Sparse Neural Radiance Grid (SNeRG) representation, enabling real-time rendering.

###### [RtS](https://arxiv.org/abs/2108.04886)

- Focuses on rendering derivatives efficiently and correctly for a variety of surface representations, including NeRF, using a fast "Surface NeRF" or sNerF renderer.

<!-- #### Dynamic

###### [Nerfies: Deformable Neural Radiance Fields](https://nerfies.github.io/) (Park et al., 2020)

<video src="/nerf-guide/nerfies-demo.mp4" autoplay="true" controls="false" loop="true" ></video>

- A casually captured “selfie video” can be turned into free-viewpoint videos, by fitting a deformation field in addition to the usual NeRF density/color representation.
- Underlying D-NeRF model deformable videos using a second MLP applying a deformation for each frame of the video.

###### [Space-Time Neural Irradiance Fields for Free-Viewpoint Video](https://video-nerf.github.io/)

- Simply use time as an additional input. Carefully selected losses are needed to successfully train this method to render free-viewpoint videos (from RGBD data!).

###### [Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes](https://www.cs.cornell.edu/~zl548/NSFF/)

- Instead train from RGB but use monocular depth predictions as a prior, and regularize by also outputting scene flow, used in the loss.

###### [D-NeRF: Neural Radiance Fields for Dynamic Scenes](https://www.albertpumarola.com/research/D-NeRF/index.html)

- Is quite similar to the Nerfies paper and even uses the same acronym, but seems to limit deformations to translations.

###### [NeRFlow: Neural Radiance Flow for 4D View Synthesis and Video Processing](https://yilundu.github.io/nerflow/)

- Is the latest dynamic NeRF variant to appear on Arxiv, and also uses a Nerfies style deformation MLP, with a twist: it integrates scene flow across time to obtain the final deformation.

###### [NR-NeRF: Non-Rigid Neural Radiance Fields: Reconstruction and Novel View Synthesis of a Dynamic Scene From Monocular Video](https://vcai.mpi-inf.mpg.de/projects/nonrigid_nerf/)

- Also uses a deformation MLP to model non-rigid scenes. It has no reliance on pre-computed scene.

###### [AD-NeRF](https://yudongguo.github.io/ADNeRF/)

- Train a conditional nerf from a short video with audio, concatenating DeepSpeech features and head pose to the input, enabling new audio-driven synthesis as well as editing of the input clip.

###### [DynamicVS](https://free-view-video.github.io)

- Is attacking the very challenging free-viewpoint video synthesis problem, and uses scene-flow prediction along with _many_ regularization results to produce impressive results.

- Is attacking the very challenging free-viewpoint video synthesis problem, and uses scene-flow prediction along with many regularization results to produce impressive results.

#### Multi-View

#### Large-Scale

#### Relighting

[NeRF-W: NeRF in the Wild](https://nerf-w.github.io/)

- Was one of the first follow-up works on NeRF, and optimizes a latent appearance code to enable learning a neural scene representation from less controlled multi-view collections.

[Neural Reflectance Fields](https://arxiv.org/abs/2008.03824)

- Improve on NeRF by adding a local reflection model in addition to density. It yields impressive relighting results, albeit from single point light sources.

[NeRV: Neural Reflectance and Visibility Fields for Relighting and View Synthesis](https://pratulsrinivasan.github.io/nerv/)

- Uses a second “visibility” MLP to support arbitrary environment lighting and “one-bounce” indirect illumination.

[NeRD: Neural Reflectance Decomposition from Image Collections](https://markboss.me/publication/2021-nerd/)

- Is another effort in which a local reflectance model is used, and additionally a low-res spherical harmonics illumination is inferred for a given scene.

#### Shape

[GRAF]()

- i.e., a “Generative model for RAdiance Fields”is a conditional variant of NeRF, adding both appearance and shape latent codes, while viewpoint invariance is obtained through GAN-style training.

[pi-GAN]()

- Is similar to GRAF but uses a SIREN-style implementation of NeRF, where each layer is modulated by the output of a different MLP that takes in a latent code.

[pixelNeRF]()

- Is closer to image-based rendering, where N images are used at test time. It is based on PIFu, creating pixel-aligned features that are then interpolated when evaluating a NeRF-style renderer.

[GRF]()

- Is pretty close to pixelNeRF in setup, but operates in a canonical space rather than in view space.

#### Composition

###### [EditNeRF](http://editnerf.csail.mit.edu/)

- Learns a category-specific conditional NeRF model, inspired by GRAF but with an instance-agnostic branch, and show a variety of strategies to edit both color and shape interactively.

###### [ObjectNeRF](https://zju3dv.github.io/object_nerf/)

- Trains a voxel embedding feeding two pathways: scene and objects. By modifying the voxel embedding the objects can be moved, cloned, or removed.

###### [AutoRF](https://sirwyver.github.io/AutoRF/)

- Learns appearance and shape priors for a given class of objects to enable single-shot reconstruction for novel view synthesis.

###### [PNF](https://abhijitkundu.info/projects/pnf/)

- Fits a separate NeRF to individual object instances, creating a panoptic-radiance field that can render dynamic scenes by composing multiple instance-NeRFs and a single "stuff"-NeRF.

#### Portrait

[NerFace: Dynamic Neural Radiance Fields for Monocular 4D Facial Avatar Reconstruction](https://gafniguy.github.io/4D-Facial-Avatars/)

- Is focused on 4D avatars and hence impose a strong inductive bias by including a deformable face model into the pipeline. This gives parametric control over the dynamic NeRF.

[Portrait Neural Radiance Fields from a Single Image](https://portrait-nerf.github.io/)

- Creates static NeRF-style avatars, but does so from a single RGB headshot. To make this work, light-stage training data is required.

#### Articulated

#### Pose/Pose-Free

[iNeRF: Inverting Neural Radiance Fields for Pose Estimation](https://yenchenlin.me/inerf/)

- Uses a NeRF MLP in a pose estimation framework, and is even able to improve view synthesis on standard datasets by fine-tuning the poses. However, it does not yet handle illumination.

[BARF](https://chenhsuanlin.bitbucket.io/bundle-adjusting-NeRF/) optimizes for the scene and the camera poses simultaneously, as in "bundle adjustment", in a coarse-to-fine manner.

[SCNeRF](https://postech-cvlab.github.io/SCNeRF/) is similar to BARF, but additionally optimizes over intrinsics, including radial distortion and per-pixel non-linear distortion.

[GNeRF](https://arxiv.org/abs/2103.15606) distinguishes itself from other the pose-free NeRF efforts by virtue of a "rough initial pose" network, which uses GAN-style training a la GRAF, which solves the (hard) initialization problem.

#### Conditional

[GRF](https://github.com/alextrevithick/GRF) is, like PixelNeRF and IBRNet at CVPR, closer to image-based rendering, where only a few images are used at test time. Unlike PixelNeRF GRF operates in a canonical space rather than in view space.

[GSN](https://apple.github.io/ml-gsn/) is a generative model for _scenes_: it takes a global code that is translated into a grid of local codes, each associated with a local radiance model. A small convnet helps upscaling the final output.

[GANcraft](https://nvlabs.github.io/GANcraft/) translates a semantic block world into a set of voxel-bound NeRF-models that allows rendering of photorealistic images corresponding to this “Minecraft” world, additionally conditioned a style latent code.

[CodeNeRF](https://sites.google.com/view/wbjang/home/codenerf) Trains a GRAF-style conditional NeRF (a shape and appearance latent code) and then optimizes at inference time over both latent codes _and_ the object pose.

#### Editable

#### Other

[IMAP](https://edgarsucar.github.io/iMAP/) is an awesome paper that uses NeRF as the scene representation in an online visual SLAM system, learning a 3D scene online and tracking a moving camera against it.

[MINE](https://vincentfung13.github.io/projects/mine/) learns to predict a density/color multi-plane representation, conditioned on a single image, which can then be used for NeRF-style volume rendering.

[NeRD](https://markboss.me/publication/2021-nerd/) or “Neural Reflectance Decomposition” uses physically-based rendering to decompose the scene into spatially varying BRDF material properties, enabling re-lighting of the scene.

[Semantic-NERF](https://shuaifengzhi.com/Semantic-NeRF/) add a segmentation renderer before injecting viewing directions into NeRF and generate high resolution semantic labels for a scene with only partial, noisy or low-resolution semantic supervision.

[CO3D](https://github.com/facebookresearch/co3d) contributes an _amazing_ dataset of annotated object videos, and evaluates 15 methods on single-scene reconstruction and learning 3D object categories, including a new SOTA “NerFormer” model.

Finally, [CryoDRGN2](https://openaccess.thecvf.com/content/ICCV2021/html/Zhong_CryoDRGN2_Ab_Initio_Neural_Reconstruction_of_3D_Protein_Structures_From_ICCV_2021_paper.html) attacks the challenging problem of reconstructing protein structure _and_ pose from a "multiview" set of cryo-EM _density_ images. It is unique among NeRF-style papers as it works in the Fourier domain. -->

## Sources

**Papers**

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
