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
1. [Citation](#citation)

## Introduction
![STMAE](docs/stmae_method.png)
The above figure depicts the structure of STMAE, consisting of a key encoder and a mask autoencoder (mask encoder & decoder). The key encoder captures the spatiotemporal correspondences between reference frames and the query frame, and aggregates a coarse mask for the query frame according to the captured correspondences. Next, the mask autoencoder is responsible for reconstruct a clear prediction mask from the coarse one.

![One-shot Training](docs/stmae_idea.png)
The above figure illustrates the simple idea of the One-shot Training strategy. A forward reconstruction operation is first taken to obtain the predictions of subsquent frames under the *stop gradient* setting, and with gradients being calculated a backward reconstruction operation is used to rebuild the first frame mask by using predictions of subsquent frames.

## Results
$^*$ *The results here are improved caused we've updated our implementation.*

| Dataset |  J&F | J | F | Label % | Train on
| --- | :--:|:--:|:---:|:---:|:---:|
| DAVIS 2016 val. | 91.7 | 90.4 | 93.0 | 3.5 | DAVIS 2017 + YouTube-VOS 2018 |
| DAVIS 2017 val. | 85.3 | 82.0 | 88.6 | 3.5 | DAVIS 2017 + YouTube-VOS 2018 |

| Dataset | Overall Score | J-Seen | F-Seen | J-Unseen | F-Unseen | Label % | Train on
| --- | :--:|:--:|:---:|:---:|:---:|:---:|:---:|
| YouTubeVOS 18 val. | 84.3 | 83.2 | 87.9 | 79.0 | 87.2 | 3.5 % | DAVIS 2017 + YouTube-VOS 2018 |

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
### Download the datasets
```bash
python -m scripts.download_datasets
```
If datasets are already in your machine, you should use softlink (`ln -s`) to organize their structures as following:
```bash
├── STMAE
├── DAVIS
│   ├── 2016
│   │   ├── Annotations
│   │   └── ...
│   └── 2017
│       ├── test-dev
│       │   ├── Annotations
│       │   └── ...
│       └── trainval
│           ├── Annotations
│           └── ...
├── static
│   ├── BIG_small
│   └── ...
├── YouTube
│   ├── all_frames
│   │   └── valid_all_frames
│   ├── train
│   ├── train_480p
│   └── valid
└── YouTube2018
    ├── all_frames
    │   └── valid_all_frames
    └── valid
```
## Training
Our experiments are conducted using a batch size of 32 with 4 NVIDIA Tesla V100 GPUs. But we have tested that using a smaller batch size (incorperating the linear learning rate scaling) or less GPUs can deliver similar performances. The following command can be used to train our model from scratch:
```bash
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=[address:port] train.py --stage 3 --s3_batch_size 32 --s3_lr 2e-5 --s3_num_frames 8 --s3_num_ref_frames 3 --exp_id [identifier_for_exp] --val_epoch 5 --total_epoch 350
```
If you prefer a pretrained model for fine-tuning, please use the flag `--load_network` followed by the `path-to-the-pretrained-model`

## Inference
TBD

## Citation
TBD

## Acknowledgements
We thank PyTorch contributors and [Ho Kei Cheng](https://hkchengrex.github.io/) for releasing their implementation of [XMem](https://github.com/hkchengrex/XMem) and [STCN](https://github.com/hkchengrex/STCN).