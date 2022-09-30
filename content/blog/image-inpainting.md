---
author: "Vien Vuong"
title: "Web App Implementation of LaMa Image Inpainting"
date: "2022-06-04"
description: "Modern image inpainting systems, despite the significant progress, often struggle with large missing areas, complex geometric structures, and high-resolution images. The authors find that one of the main reasons for that is the lack of an effective receptive field in both the inpainting network and the loss function. To alleviate this issue, this paper proposes a new method called large mask inpainting (LaMa). LaMa is based on i) a new inpainting network architecture that uses fast Fourier convolutions (FFCs), which have the image-wide receptive field; ii) a high receptive field perceptual loss; iii) large training masks, which unlocks the potential of the first two components. Our inpainting network improves the state-of-the-art across a range of datasets and achieves excellent performance even in challenging scenarios, e.g. completion of periodic structure. This is a web app implementation of the LaMa inpainting paper with a user-friendly canvas for removing objects from images."
tags: ["computer-vision", "ml", "project"]
comments: false
socialShare: false
toc: true
math: true
cover:
  src: /image-inpainting/lama.png
  alt: Web App Implementation of LaMa Image Inpainting
---

[**Check out this project on my GitHub**](https://github.com/vienvuong/inpainting)

## Introduction

Modern image inpainting systems, despite the significant progress, often struggle with large missing areas, complex geometric structures, and high-resolution images. The authors find that one of the main reasons for that is the lack of an effective receptive field in both the inpainting network and the loss function. To alleviate this issue, this paper proposes a new method called large mask inpainting (LaMa). LaMa is based on i) a new inpainting network architecture that uses fast Fourier convolutions (FFCs), which have the image-wide receptive field; ii) a high receptive field perceptual loss; iii) large training masks, which unlocks the potential of the first two components. Our inpainting network improves the state-of-the-art across a range of datasets and achieves excellent performance even in challenging scenarios, e.g. completion of periodic structure.

This is a web app implementation of the LaMa inpainting paper with a user-friendly canvas for removing objects from images. This project is created using TypeScript, React, Flask, and Docker.

## Selected Results

![LaMa Demo GIF](/lama-demo1.gif)
![LaMa Demo GIF](/lama-demo2.gif)

## Sources

[LaMa Official Implementation](https://github.com/saic-mdal/lama)
[Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://arxiv.org/abs/2109.07161) (Suvorov et al., WACV 2022)
