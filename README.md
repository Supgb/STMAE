# STMAE
## A Spatiotemporal Mask Autoencoder for One-shot Video Object Segmentation
[Baiyu Chen](supgb.github.io) $^1$, Li Zhao $^1$, Sixian Chan $^2$

$^1$ Key Laboratory of Intelligent Informatics for Safety & Emergency of Zhejiang Province, Wenzhou University<br>
$^2$ The College of Computer Science and Technology, Zhejiang University of Technology

[FAIML 2024](www.faiml.org)

## Features
- A label-efficient VOS network based on autoencoding.
- Achieve comparable results with fully-supervised VOS approaches using only the first frame annotations of videos.

## Table of Contents
1. [Introduction](#introduction)
1. [Results](#results)
1. [Getting Started](#getting-started)
1. [Training](#training)
1. [Inference](#inference)

## Introduction
![STMAE](docs/stmae_method.png)
The above figure depicts the structure of STMAE, consisting of a key encoder and a mask autoencoder (mask encoder & decoder). The key encoder captures the spatiotemporal correspondences between reference frames and the query frame, and aggregates a coarse mask for the query frame according to the captured correspondences. Next, the mask autoencoder is responsible for reconstruct a clear prediction mask from the coarse one.

![One-shot Training](docs/stmae_idea.png)
The above figure illustrates the simple idea of the One-shot Training strategy. A forward reconstruction operation is first taken to obtain the predictions of subsquent frames under the *stop gradient* setting, and with gradients being calculated a backward reconstruction operation is used to rebuild the first frame mask by using predictions of subsquent frames.

## Results
$^\dag$ *The results here are improved caused we've updated our implementation.*

TBD

## Getting Started
To reproduce our results, you can either train the model following [Training](#training) or evaluate our [pretrained model](https://github.com/Supgb/STMAE/releases/tag/v1.0) following the instruction in [Inference](#inference). Before you start, the experiments environment can be configured using `conda` and `pip`.

### Clone the repository
```bash
git clone https://github.com/Supgb/STMAE.git && cd STMAE
```
### Create the environment using `conda`
```bash
conda create -n stmae python=3.8
conda activate stmae
```
### Install the dependencies using `pip`
```bash
pip install -r requirements.txt
```

## Training
TBD 

## Inference
TBD
