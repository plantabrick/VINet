# VINet: A Visual Interpretable Image Diagnosis Network

## Installation

The code was developed with python 2.7, opencv and pytorch 0.4.

get VINet source

`git clone https://github.com/plantabrick/VINet.git`

install pytorch and other dependencies

`pip2 install torch==0.4`

`pip2 install torchvision opencv-python shutil`

## Download model parameters

The trained model parameters can be downloaded from the following link：

Link:https://pan.baidu.com/s/1sxpA5d0QPZnJOkhFsB_lhQ 

Extraction code: 3lrg 

This link consists of two files, i.e ckpt_class.pkl and ckpt_cover.pkl, corresponding to two subnetworks.

These two files are placed directly in the directory of Vinet: "./VINet/ckpt_class.pkl" and "./VINet/ckpt_cover.pkl"

## Run the demo

`python test.py`


## Dataset

Complete data set:

Our dataset is a two-dimensional slice cut from luna16 dataset, which contains six types of tumors with different severity.
The following links can download the sliced data directly.

Link: https://pan.baidu.com/s/1aio8CDyQ0BgfcdMomjJt-Q 

Extraction code: 22vb

Compressed file password：vinet

Demo data:

The data used in the demo is already included in Git and does not need to be downloaded separately.


## Network Architecture

![](https://github.com/plantabrick/VINet/blob/master/overall.png)

## Visualization results

![](https://raw.githubusercontent.com/plantabrick/VINet/master/visualization.png)
