# KTNet: A Few-Shot Learning Network for Out-of-Distribution Image Classification
[![Python](https://img.shields.io/badge/python-2.7%20%7C%203.5-blue.svg?style=flat-square&logo=python&color=3776AB)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-1.14.0-orange.svg?style=flat-square&logo=tensorflow&color=FF6F00)](https://github.com/y2l/meta-transfer-learning/tree/master/tensorflow)

This repository has the Tensorflow implementation for the paper "KTNet: A Few-Shot Learning Network for Out-of-Distribution Image Classification" by Islam Osman and Mohamed S. Shehata

#### Summary
* [Introduction](#introduction)
* [Datasets](#datasets)
* [Results](#results)
* [Installation](#installation)
* [How to use](#how-to-use)

## Introduction
This paper proposes a novel hybrid meta-learning network called KTNet. KTNet autonomously increases the number of labeled images, improving accuracy in data-constrained situations while also outperforming other state-of-the-art few-shot learning models. To accomplish this, KTNet leverages hybrid meta-learning to detect and cluster OOD images. This is done using self-supervised learning to learn a feature space that facilitates the clustering of OOD images. KTNet subsequently assigns pseudo labels to the clustered OOD images. One image is selected from each detected cluster and is added to the support set. The new support set has at least one labeled image for each class in the query set, including the in-distribution (ID) classes and OOD classes. Finally, KTNet applies few-shot learning on the pseudo labeled support set to classify the query set. To summarize, the novelties of this paper are as follows: 

* This paper presents a network that can detect and accurately classify OOD images into different distinguished classes in a few-shot learning context. To the best of the authors' knowledge, this approach is the first of its kind, as the current state-of-the-art that detects all OOD classes as one class. 
    
* KTNet is the first hybrid meta-learning network in the literature that combines metric-based learning and optimization-based learning. This hybrid approach takes advantage of each meta-learning method to boost overall performance.
* KTNet improves performance (even during the testing phase) by autonomously increasing the number of labeled images and adding them to the support set.

<p align="center">
    <img src="https://github.com/islamosmanubc/KTNet/blob/main/figures/ktnet.png" width="1200"/>
</p>

> Figure: The proposed hybrid meta-learning network (KTNet) architecture. The green arrow represents the training phase (supervised learning using the support set). The yellow arrow represents the testing phase (classifying the query set). The purple arrow represents the selection of samples from the query set and adding them to the support set (to increase the number of shots). The whole process is repeated until no more samples from the query set are added to the support set.


## Datasets

### Mini-ImageNet
mini-imagenet is a subset from ImageNet \cite{imagenet}. The dataset consists of 100 different classes, with 60,000 colored images of size 84 $\times$ 84, 600 per class. The dataset is divided into three uncorrelated sets: training, validation, and testing, with 64, 16, and 20 classes.

### Fewshot-CIFAR
Fewshot-CIFAR is constructed from CIFAR100 \cite{cifar100}. The dataset consists of 100 different classes, with 60,000 colored images of low resolution (32 $\times$ 32). The number of images per class is 600. One hundred classes are grouped into 20 superclasses. The dataset is divided by the superclasses to prevent overlapping between training, validation, and testing sets, with 12, 4, and 4 superclasses.

### Blood Cell Count and Detection (BCCD)
BCCD is a small-scale dataset for white blood cell detection and classification. It contains 5 different classes. Each class has 700 images. We used this dataset as an out-of-domain classification to test the generalization of the proposed work.

## Results
|          (%)           |  mini-imagenet 1-shot  | mini-imagenet 5-shot  | FC100 1-shot | FC100 5-shot |  BCCD 5-shot | 
| ---------------------- | ---------------------- | --------------------- | ------------ | ------------ | ------------ |
| `KTNet_base`           |      `57.7 ± 1.8`      |     `67.4 ± 1.7`      | `41.8 ± 1.3` | `54.6 ± 1.3` | `38.0 ± 0.5` |
| `KTNet`                |      `65.1 ± 1.1`      |     `78.5 ± 0.9`      | `49.3 ± 0.6` | `61.3 ± 0.5` | `44.8 ± 0.1` |
| `KTNet-RESIST`         |      `74.4 ± 0.8`      |     `82.6 ± 0.3`      | `57.9 ± 0.4` | `65.1 ± 0.4` | `50.6 ± 0.1` |


## Installation

To run the code, you need to install:
* python 2.7 or 3.5
* Tensorflow-gpu v1.14.0
Also, you need to install some extra packages:
* scipy
* scikit
* opencv
* pillow
* matplot
* tqdm

## How to use

You can do everything by manupilating the arguments in the file `mainKTNet.py`. Some of the important arguments are:
* `Phase` has 4 possible values `pre` for pre-training, `meta` for meta-training/testing, `disc` for training the metric-based learning branch (MBB), and `zero` for zero shot testing
* `metatrain` is a boolean variable, `True` runs the training and `False` runs the testing
* `way_num` is the number of ways of each episode
* `shot_num` is the number of shots (i.e., size of the support set in each episode)
* `metatrain_dir` is the directory for the training dataset (e.g., the directory of mini-imagenet training classes)
* `metatest_dir` the same but for testing dataset
* `metaval_dir` the same but for validation dataset

To train from scratch you need to run `mainKTNet.py` 3 times with this order:
* `python mainKTNet.py --phase=pre`
* `python mainKTNet.py --phase=meta`
* `python mainKTNet.py --phase=disc`

Then, for testing:
* `python mainKTNet.py --phase=meta`   for the image classification without OOD
* `python mainKTNet.py --phase=zero`   for the image classification with OOD
* `python mainKTNet.py --phase=zero --shot-num=0`   for the pure OOD image classification 
