---
author: "Vien Vuong"
title: "A Deep Dive Into Neural Radiance Fields (NeRFs) [Ongoing]"
date: "2022-09-28"
description: "We will explore and implement (at least) 3 papers that propose "
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

### Neural Radiance Fields (NeRFs)

<video src="/nerf-guide/nerf-demo.mp4" autoplay="true" controls="false" loop="true"></video>

After causing a big splash in ECCV 2020, the impressive NeRF paper by Mildenhall et al. has kickstarted an explosion in research in the field of neural volume rendering. It is a novel, data-driven approach that provides an efficient synthesis of visually compelling novel scenes from input images or videos.

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
5. These color and volume density values can now be transformed into an image by a fully differentiable volume rendering procedure (and a hierarchical sampling strategy).
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

#### Fundamentals

##### [Mip-NeRF](https://jonbarron.info/mipnerf/)

- Address the severe aliasing artifacts from vanilla NeRF by adapting the mip-map idea from graphics and replacing sampling the light field by integrating over conical sections along a the viewing rays.

[MVSNeRF](https://apchenstu.github.io/mvsnerf/) trains a model across many scenes and then renders new views conditioned on only a few posed input views, using intermediate voxelized features that encode the volume to be rendered.

##### [DietNeRF](https://arxiv.org/abs/2104.00677)

- Is a very out-of-the box method that supervises the NeRF training process by a semantic loss, created by evaluating arbitrary views using CLIP, so it can learn a NeRF from a single view for arbitrary categories.

[UNISURF](https://arxiv.org/abs/2104.10078) propose to replace the density in NeRF with occupancy, and hierarchical sampling with root-finding, allowing to do both volume and surface rendering for much improved geometry.

##### [NerfingMVS](https://weiyithu.github.io/NerfingMVS/)

- Use a sparse depth map from an SfM pipeline to train a scene-specific depth network that subsequently guides the adaptive sampling strategy in NeRF.

#### Performance

##### [Neural Sparse Voxel Fields](https://github.com/facebookresearch/NSVF) (Liu et al., 2020)

![NSVF Pipeline](/nerf-guide/nsvf-pipeline.jpg)

- Hybrid scene representation that combines neural implicit fields with an explicit sparse voxel structure
- Defines a set of voxel-bounded implicit fields organized in a sparse voxel octree to model local properties in each cell
- Utilizes the sparse voxel structure to achieve efficient rendering by skipping the voxels containing no relevant scene content
- Progressive training strategy that efficiently learns the underlying sparse voxel structure with a differentiable ray-marching operation from a set of posed 2D images in an end-to-end manner
- 10 times faster than NeRF
- Applications: scene editing, scene composition, multi-scene learning, free-viewpoint rendering of a moving human, and large-scale scene rendering

##### [NeRF++: Analyzing and Improving Neural Radiance Fields](https://github.com/Kai-46/nerfplusplus) (Zhange et al., 2020)

- Analyzes the shape-radiance ambiguity: from certain viewpoints, NeRF can recover the wrong geometry from the radiance information.
- NeRF's MLP is quite robust against the shape-radiance ambiguity because of 2 reasons:
  1. Incorrect geometry forces the radiance field to have higher intrinsic complexity (i.e., much higher frequencies w.r.t $d$). Higher complexity required for incorrect shapes is more difficult to represent with a limited capacity MLP.
  2. NeRF's specific MLP structure favors $x$ more than $d$. This means the model encodes an implicit prior favoring smooth surface reflectance functions where $c$ is smooth with respect to $d$ at any given surface point $x$.
- Scene background that is very far away can cause resolution problems.
- Presents a novel spatial parameterization scheme (inverted sphere parameterization) that integrates over the normalized device coordinates (NDC) space instead of the Euclidean space.

I am most interested in applications of NeRF in 3D object reconstruction and depth estimation problems.

##### [DeRF: Decomposed Radiance Fields](https://ubc-vision.github.io/derf/) (Rebain et al., 2020)

- Propose to spatially decompose a scene into “soft Voronoi diagrams” and dedicate smaller networks for each decomposed part to take advantage of accelerator memory architectures.
- Achieves near-constant inference time regardless of the number of decomposed parts
- Provides up to 3x more efficient inference than NeRF (with the same rendering quality)

##### [AutoInt: Automatic Integration for Fast Neural Volume Rendering](https://www.computationalimaging.org/publications/automatic-integration/) (Lindell et al., 2020)

- NeRF's volume integrations along the rendered rays during training and inference have high computational and memory requirements
- Proposes automatic integration to directly learn the volume intergral using implicit neural representation networks
  > For training, we instantiate the computational graph corresponding to the derivative of the implicit neural representation. The graph is fitted to the signal to integrate. After optimization, we reassemble the graph to obtain a network that represents the antiderivative. By the fundamental theorem of calculus, this enables the calculation of any definite integral in two evaluations of the network.
- Greater than 10x less computation requirements than NeRF.

![AutoInt Framework](/nerf-guide/autoint-framework.png)

##### [Learned Initializations for Optimizing Coordinate-Based Neural Representations](https://arxiv.org/abs/2012.02189) (Tancik et al., 2020)

- Optimizing a coordinate-based network, such as NeRF, from randomly initialized weights for each new signal is inefficient.
- Using meta-learned initial weights enables faster convergence during optimization and can serve as a strong prior over the signal class being modeled, resulting in better generalization when only partial observations of a given signal are available.

![Learned Initializations Overview](/nerf-guide/learned-inits-overview.png)

![Learned Initializations Demo](/nerf-guide/learned-inits-demo.webp)

##### [JaxNeRF](https://github.com/google-research/google-research/tree/master/jaxnerf) (GitHub Repo)

- Uses [JAX](https://github.com/google/jax) to dramatically speed up training, from days to hours.

##### [FastNeRF](https://arxiv.org/abs/2103.10380)

- factorizes the NeRF volume rendering equation into two branches that are combined to give the same results as NeRF, but allow for much more efficient caching, yielding a 3000x speed up.

##### [KiloNeRF](https://github.com/creiser/kilonerf)

- replaces a single large NeRF-MLP with thousands of tiny MLPs, accelerating rendering by 3 orders of magnitude.

##### [PlenOctrees](https://alexyu.net/plenoctrees/)

- introduce NeRF-SH that uses spherical harmonics to model view-dependent color, and then compresses that into a octree-like data-structure for rendering the result 3000 faster than NeRF.

##### [SNeRG](https://arxiv.org/abs/2103.14645)

- precompute and "bake" a NeRF into a new Sparse Neural Radiance Grid (SNeRG) representation, enabling real-time rendering.

##### [RtS](https://arxiv.org/abs/2108.04886)

- focuses on rendering derivatives efficiently and correctly for a variety of surface representations, including NeRF, using a fast "Surface NeRF" or sNerF renderer.

#### Dynamic

##### [Nerfies: Deformable Neural Radiance Fields](https://nerfies.github.io/) (Park et al., 2020)

<video src="/nerf-guide/nerfies-demo.mp4" autoplay="true" controls="false" loop="true" ></video>

- We present the first method capable of photorealistically reconstructing a non-rigidly deforming scene using photos/videos captured casually from mobile phones.

Our approach augments neural radiance fields (NeRF) by optimizing an additional continuous volumetric deformation field that warps each observed point into a canonical 5D NeRF. We observe that these NeRF-like deformation fields are prone to local minima, and propose a coarse-to-fine optimization method for coordinate-based models that allows for more robust optimization. By adapting principles from geometry processing and physical simulation to NeRF-like models, we propose an elastic regularization of the deformation field that further improves robustness.

We show that Nerfies can turn casually captured selfie photos/videos into deformable NeRF models that allow for photorealistic renderings of the subject from arbitrary viewpoints, which we dub "nerfies". We evaluate our method by collecting data using a rig with two mobile phones that take time-synchronized photos, yielding train/validation images of the same pose at different viewpoints. We show that our method faithfully reconstructs non-rigidly deforming scenes and reproduces unseen views with high fidelity.

##### [Space-Time Neural Irradiance Fields for Free-Viewpoint Video](https://video-nerf.github.io/)

- Simply use time as an additional input. Carefully selected losses are needed to successfully train this method to render free-viewpoint videos (from RGBD data!).

##### [Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes](https://www.cs.cornell.edu/~zl548/NSFF/)

- Instead train from RGB but use monocular depth predictions as a prior, and regularize by also outputting scene flow, used in the loss.

##### [D-NeRF: Neural Radiance Fields for Dynamic Scenes](https://www.albertpumarola.com/research/D-NeRF/index.html)

- Is quite similar to the Nerfies paper and even uses the same acronym, but seems to limit deformations to translations.

##### [NeRFlow: Neural Radiance Flow for 4D View Synthesis and Video Processing](https://yilundu.github.io/nerflow/)

- Is the latest dynamic NeRF variant to appear on Arxiv, and also uses a Nerfies style deformation MLP, with a twist: it integrates scene flow across time to obtain the final deformation.

##### [NR-NeRF: Non-Rigid Neural Radiance Fields: Reconstruction and Novel View Synthesis of a Dynamic Scene From Monocular Video](https://vcai.mpi-inf.mpg.de/projects/nonrigid_nerf/)

- Also uses a deformation MLP to model non-rigid scenes. It has no reliance on pre-computed scene.

##### [AD-NeRF](https://yudongguo.github.io/ADNeRF/) train a conditional nerf from a short video with audio, concatenating DeepSpeech features and head pose to the input, enabling new audio-driven synthesis as well as editing of the input clip.

##### [DynamicVS](https://free-view-video.github.io) is attacking the very challenging free-viewpoint video synthesis problem, and uses scene-flow prediction along with _many_ regularization results to produce impressive results.

- Is attacking the very challenging free-viewpoint video synthesis problem, and uses scene-flow prediction along with many regularization results to produce impressive results.

#### Priors

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

Finally, [CryoDRGN2](https://openaccess.thecvf.com/content/ICCV2021/html/Zhong_CryoDRGN2_Ab_Initio_Neural_Reconstruction_of_3D_Protein_Structures_From_ICCV_2021_paper.html) attacks the challenging problem of reconstructing protein structure _and_ pose from a "multiview" set of cryo-EM _density_ images. It is unique among NeRF-style papers as it works in the Fourier domain.

## Method

_TBD_

## Experiments

_TBD_

### Evaluation Metrics

- PSNR (Peak Signal-to-Noise Ratio): higher PSNR, lower MSE. Lower MSE implies less difference between the ground truth image and the rendered image. Thus, higher PSNR, the better model.

- SSIM (Structural Similarity Index): Checks the structural similarity with the ground truth image model. Higher SSIM, the better model.

- LPIPS (Learned Perceptual Image Patch Similarity): Determines the similarity with the view of perception; using VGGNet. Lower LPIPS, the better model.

## Milestones

_TBD_

## Sources

**Papers**

[State of the Art on Neural Rendering](https://arxiv.org/abs/2004.03805) (Tewari et al., Eurographics 2020)

[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934) (Mildenhall et al., ECCV 2020)

- [Website](https://www.matthewtancik.com/nerf)

[DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation](https://arxiv.org/abs/1901.05103) (Park et al., CVPR 2019)

[Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://bmild.github.io/fourfeat/) (Tancik, NeurIPS 2020)

[Neural Sparse Voxel Fields](https://arxiv.org/abs/2007.11571) (Liu et al., NeurIPS 2020)

- [Website](https://lingjie0206.github.io/papers/NSVF/)

[AutoRF: Learning 3D Object Radiance Fields from Single View Observations](https://arxiv.org/abs/2204.03593) (Muller et al., CVPR 2022)

- [Website](https://sirwyver.github.io/AutoRF/)

**Frank Dellaert's Blog Posts**

- [NeRF at CVPR 2022](https://dellaert.github.io/NeRF22/)

- [NeRF at ICCV 2021](https://dellaert.github.io/NeRF21/)

- [NeRF Explosion 2020](https://dellaert.github.io/NeRF/)

**Other**

[Awesome Neural Radiance Fields](https://github.com/yenchenlin/awesome-NeRF) (GitHub Repo)
