# VINet: A Visual Interpretable Image Diagnosis Network

## Installation

The code was developed with python 2.7, opencv and pytorch 0.4.

get VINet source

`git clone https://github.com/plantabrick/VINet.git`

install pytorch and other dependencies

`pip2 install torch==0.4`

`pip2 install torchvision opencv-python shutil`

## Run the demo

`python main.py`

## Dataset

Complete data set:

Our dataset is a two-dimensional slice cut from luna16 dataset, which contains six types of tumors with different severity.
The following links can download the sliced data directly.

Link: https://pan.baidu.com/s/1qd5d5fdx6yywww8lipihq

Extraction code: y4ia

Demo data:

The data used in the demo is already included in Git and does not need to be downloaded separately.


## Network Architecture

![](https://github.com/plantabrick/VINet/blob/master/model2.jpg)

## Visualization results

![](https://raw.githubusercontent.com/plantabrick/VINet/master/visualization.png
