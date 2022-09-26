---
author: "Vien Vuong"
title: "Colorizing the Prokudin-Gorskii Collection With OpenCV [Ongoing]"
date: "2022-08-27"
description: "This project takes the digitized Prokudin-Gorskii glass plate images and automatically produces a color image with as few visual artifacts as possible. To achieve this, I extract the three color channel images, place them on top of each other, and align them so that they form a single RGB color image."
tags: ["computer-vision", "ml", "project"]
comments: false
socialShare: false
toc: true
math: true
cover:
  src: assets/report/prokudin_gorskii.jpg
  alt: Colorizing the Prokudin-Gorskii Collection With OpenCV
---

[**Check out this project on my GitHub**](https://github.com/vienvuong/colorizing-pg)

## Introduction

Sergei Mikhailovich Prokudin-Gorskii (1863-1944) was a photographer who, between the years 1909-1915, traveled the Russian empire and took thousands of photos of everything he saw. He used an early color technology that involved recording three exposures of every scene onto a glass plate using a red, green, and blue filter. Back then, there was no way to print such photos, and they had to be displayed using a special projector. Prokudin-Gorskii left Russia in 1918. His glass plate negatives survived and were purchased by the Library of Congress in 1948. Today, a digitized version of the Prokudin-Gorskii collection is available online.

This project takes the digitized Prokudin-Gorskii glass plate images and automatically produces a color image with as few visual artifacts as possible. To achieve this, I extract the three color channel images, place them on top of each other, and align them so that they form a single RGB color image.

## Cropping the Monochromatic Images

To allow manual cropping, the program prompts the user to manually crop each of the 3 monochromatic images corresponding to the Blue, Green, and Red channels in that order.

![Cropping](assets/report/cropping.png)

Now that we have the 3 channels, we have to align two of the channels to the third to construct the color image.

## Basic Alignment (Naive Search)

The easiest way to align the parts is to exhaustively search over a window of possible displacements (say [-15,15] pixels independently for the x and y axis), score each one using some image matching metric, and take the displacement with the best score.

There is a number of possible metrics that one could use to score how well the images match. The most basic one is the L2 norm of the pixel differences of the two channels, also known as the sum of squared differences (SSD) for images loaded as NumPy arrays.

**SSD (Sum of Squared Differences):**

$$ SSD(\vec{x}, \vec{y}) = \sum*{i}\sum*{j}(\vec{x}\_{i,j} - \vec{y}\_{i,j})^2 $$

Note that in our case, the images to be matched do not actually have the same brightness values (they are different color channels), so a cleverer metric might work better. One such possibility is normalized cross-correlation (NCC), which is simply the dot product between the two images normalized to have zero mean and unit norm. NCC has the advantage of being more robust to change in global brightness, but it is also slower than SSD.

**NCC (Normalized Cross Correlation):**

$$ NCC(\vec{x}, \vec{y}) = \langle\frac{\vec{x}}{||\vec{x}||}, \frac{\vec{y}}{||\vec{y}||}\rangle $$

## Multiscale Alignment (Pyramid Search)

For the high-resolution glass plate scans, exhaustive search over all possible displacements will become prohibitively expensive. To deal with this case, I implemented a faster search procedure using an image pyramid. An image pyramid represents the image at multiple scales (usually scaled by a factor of 2) and the processing is done sequentially starting from the coarsest scale (smallest image) and going down the pyramid, updating your estimate as you go.

![Pyramid 1](assets/report/pyramid1.png)

A more intuitive view of the image pyramid:

![Pyramid 2](assets/report/pyramid2.png)

We would start at the smallest scale (1/8 resolution) and align each of the Green and Red channels to Blue. We save the 2 displacement (alignment) vectors, and upscales them so they can be applied to the next level in the pyramid (1/4 resolution). Then we recursively align the next smallest level until we get to the full resolution images.

Multiscale alignment also results in a significant speed up. Naive search for a high-resolution image with a (-30, 30) search space in both X-Y directions takes around 12 minutes and produces a bad alignment. Pyramid search takes 5 minutes and produces a much better alignment.

### Problems with this approach

While this stereo matching method works well for some images:

![Good Result](assets/report/good_result1.jpg)

It completely fails for others:

![Bad Result](assets/report/bad_result1.jpg)

## Improvement: Only align edges

Current method focuses on comparing the brightnesses between the channels. However, since brightness is an issue, we can pass the image through an edge detection filter to remove brightness from the equation. There are multiple ways to find edges in an image. I choose to use the Canny edge detector because it is quite sophisticated.

![Mono Image](assets/report/mono1.jpg)
![Canny Image](assets/report/canny1.jpg)

Apply the same Pyramid Search algorithm:

![Canny Pyramid 1](assets/report/canny_pyramid1.png)
![Canny Pyramid 2](assets/report/canny_pyramid2.png)

Now if I colorize the image again, the result is much better:

![Good Result 2](assets/report/good_result2.jpg)

## Improvement: Crop out the strange color border

The borders of the photograph will have strange colors since the three channels won't exactly align. I solved this problem by computing the mean brightness of each channel for each row and column. At the border rows and columns, if one or more of the channel is almost completely dark (mean < 0.5), I crop out that row or column.

Now let's try with our previous result:

![Good Result 3](assets/report/good_result3.jpg)

## Improvement: Use cv2.matchTemplate() to speed up alignment

cv2.matchTemplate() is much faster than our Multiscale Search method and more robust to monochromatic channel images with large discrepencies.

![Match Template 1](assets/report/mt1.png)
![Match Template 2](assets/report/mt2.png)
![Match Template 3](assets/report/mt3.png)
![Match Template 4](assets/report/mt4.png)

## More Results

Green: $(21, -23)$, Red: $(36, 19)$

![Good Result 4](assets/results/01657u.jpg)

Green: $(28, -105)$, Red: $(39, -50)$

![Good Result 5](assets/results/01861a.jpg)

Green: $(2, -9)$, Red: $(1, -4)$

![Good Result 6](assets/results/00125v.jpg)

Green: $(11, 0)$, Red: $(11, 7)$

![Good Result 7](assets/results/00149v.jpg)

Green: $(12, -9)$, Red: $(14, -4)$

![Good Result 8](assets/results/00153v.jpg)

Green: $(3, -14)$, Red: $(3, -9)$

![Good Result 9](assets/results/00351v.jpg)

Green: $(4, -14)$, Red: $(3, -8)$

![Good Result 10](assets/results/00398v.jpg)

Green: $(-2, -16)$, Red: $(-4, -10)$

![Good Result 11](assets/results/01112v.jpg)
