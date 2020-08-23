# The Group Loss for Deep Metric Learning
Official PyTorch implementation of [The Group Loss for Deep Metric Learning" paper (ECCV 2020)]
(https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520273.pdf)
published at European Conference on Computer Vision (ECCV) 2020.

## Installation

Please download the code:

To use our code, first download the repository:
````
git clone https://github.com/dvl-tum/The_Group_Loss_for_Deep_Metric_Learning.git
````

To install the dependencies:

````
pip install -r requirements.txt
````

## Datasets

The code assumes that the CUB-200-2011 dataset is given in the format:

````
CUB_200_2011/images/001
CUB_200_2011/images/002
CUB_200_2011/images/003
...
CUB_200_2011/images/200
````

The code assumes that the CARS-196 dataset is given in the format:

````
CARS/images/001
CARS/images/002
CARS/images/003
...
CARS/images/198
````

The code assumes that the Stanford Online Products dataset is given in the format:

````
Stanford/images/00001
Stanford/images/00002
Stanford/images/00003
...
Stanford/images/22634
````

## Training

In order to train, evaluate and save a model, run the following command:

````
python train.py
````

For convenience, we provide models trained in the classification task. For three datasets (CUB-200-2011, CARS-196, and Stanford Online Products, in addition to ImageNet) they can be found at:

````
net/bn_inception_weights_pt04.pt
net/finetuned_cub_bn_inception.pth
net/finetuned_cars_bn_inception.pth
net/finetuned_Stanford_bn_inception.pth
````

Please see the file:

````
train_finetune.py
````

on how to pretrain the networks for the classification task (if you want to use some other type of network). For DenseNets, please email us to send you the pretrained networks (bear in mind though, the difference in performance is minimal, so you can skip the pretraining).

For convenience (in case you only want to use networks for feature extraction), we provide trained networks in the task of Group Loss, that reach similar results to those in the paper. They can be found at:

````
net/trained_cub_bn_inception.pth
net/trained_cars_bn_inception.pth
net/trained_stanford_bn_inception.pth
````


## Citation

If you find this code useful, please consider citing the following paper:

````
@InProceedings{Elezi_2020_ECCV,
author = {Elezi, Ismail and Vascon, Sebastiano and Torcinovich, Alessandro and Pelillo, Marcello and Leal-Taixe, Laura},
title = {The Group Loss for Deep Metric Learning},
booktitle = {European Conference on Computer Vision (ECCV)},
month = {August},
year = {2020}
}
````
