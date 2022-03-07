<a href="https://tensorlayerx.readthedocs.io/">
    <div align="center">
        <img src="https://git.openi.org.cn/hanjr/tensorlayerx-image/raw/branch/master/tlx-LOGO--02.jpg" width="50%" height="30%"/>
    </div>
</a>

<!--- [![PyPI Version](https://badge.fury.io/py/tensorlayer.svg)](https://pypi.org/project/tensorlayerx/) --->
<!--- ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tensorlayer.svg)) --->

![GitHub last commit (branch)](https://img.shields.io/github/last-commit/tensorlayer/tensorlayerx/main.svg)
[![Documentation Status](https://readthedocs.org/projects/tensorlayerx/badge/)]( https://tensorlayerx.readthedocs.io/en/latest/)
[![Build Status](https://travis-ci.org/tensorlayer/tensorlayerx.svg?branch=master)](https://travis-ci.org/tensorlayer/tensorlayerx)
[![Downloads](http://pepy.tech/badge/tensorlayerx)](http://pepy.tech/project/tensorlayerx)
[![Downloads](https://pepy.tech/badge/tensorlayerx/week)](https://pepy.tech/project/tensorlayerx/week)
[![Docker Pulls](https://img.shields.io/docker/pulls/tensorlayer/tensorlayer.svg)](https://hub.docker.com/r/tensorlayer/tensorlayer/)

🇬🇧 TensorLayerX is a multi-backend AI framework, which can run on almost all operation systems and AI hardwares, and support hybrid-framework programming. The currently version supports TensorFlow, MindSpore, PaddlePaddle and PyTorch(partial) as the backends.[layer list](https://shimo.im/sheets/kJGCCTxXvqj99RGV/F5m5Z). 

🇨🇳 TensorLayerX 是一个跨平台开发框架，可以运行在各类操作系统和AI硬件上，并支持混合框架的开发。目前支持TensorFlow、MindSpore、PaddlePaddle框架常用神经网络层以及算子，PyTorch支持特性正在开发中，[支持列表](https://shimo.im/sheets/kJGCCTxXvqj99RGV/F5m5Z)。

# News
🔥 **TensorLayerX has been released, it supports TensorFlow、MindSpore and PaddlePaddle backends, and supports some PyTorch operator backends, allowing users to run the code on different hardware like Nvidia-GPU and Huawei-Ascend. Feel free to use it and make suggestions.**

🔥 **We need more people to join the dev team, if you are interested, please email hao.dong@pku.edu.cn**


# Design Features

Compare with [TensorLayer](https://github.com/tensorlayer/TensorLayer), TensorLayerX （TLX) is a brand new seperated project for platform-agnostic purpose. 

Comparison of TensorLayer version

<p align="center"><img src="https://git.openi.org.cn/hanjr/tensorlayerx-image/raw/branch/master/version.png" width="840"\></p>

🔥TensorLayerX inherits the features of the previous verison, including ***Simplicity***, ***Flexibility*** and ***Zero-cost Abstraction***. ***Compare with [TensorLayer](https://github.com/tensorlayer/TensorLayer), TensorLayerX supports more backends, such as TensorFlow, MindSpore, PaddlePaddle and PyTorch. It allows users to run the same code on different hardwares like Nvidia-GPU and Huawei-Ascend.*** In addition, more features are ***under development***.

- ***Model Zoo***: Build a series of model Zoos containing classic and sota models,covering CV, NLP, RL and other fields.

- ***Deploy***: In feature, TensorLayerX will support the ONNX protocol, supporting model export, import and deployment.

- ***Parallel***: In order to improve the efficiency of neural network model training, parallel computing is indispensable. 

# Quick Start

- Installation
```bash
# install from pypi
pip3 install tensorlayerx 
# install from Github
pip3 install git+https://github.com/tensorlayer/tensorlayerx.git 
```
For more installation instructions, please refer to [Installtion](https://tensorlayerx.readthedocs.io/en/latest/user/installation.html)

- Define a model

You can immediately use tensorlayerx to define a model, using your favourite framework in the background, like so:
```python
import os
os.environ['TL_BACKEND'] = 'tensorflow' # change to any framework!

import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import Dense
class CustomModel(Module):

  def __init__(self):
      super(CustomModel, self).__init__()

      self.dense1 = Dense(n_units=800, act=tlx.ReLU, in_channels=784)
      self.dense2 = Dense(n_units=800, act=tlx.ReLU, in_channels=800)
      self.dense3 = Dense(n_units=10, act=None, in_channels=800)

  def forward(self, x, foo=False):
      z = self.dense1(x)
      z = self.dense2(z)
      out = self.dense3(z)
      if foo:
          out = tlx.softmax(out)
      return out

MLP = CustomModel()
MLP.set_eval()
```

# Document
TensorLayer has extensive documentation for both beginners and professionals. 

[![English Documentation](https://img.shields.io/badge/documentation-english-blue.svg)](https://tensorlayerx.readthedocs.io/en/latest/)

# Examples

- [Basic Examples](https://github.com/tensorlayer/TensorLayerX/tree/main/examples)
- [TLCV]**Coming soon!**


# Contact
 - hao.dong@pku.edu.cn
 - tensorlayer@gmail.com

# Citation

If you find TensorLayerX useful for your project, please cite the following papers：

```
@article{tensorlayer2017,
    author  = {Dong, Hao and Supratak, Akara and Mai, Luo and Liu, Fangde and Oehmichen, Axel and Yu, Simiao and Guo, Yike},
    journal = {ACM Multimedia},
    title   = {{TensorLayer: A Versatile Library for Efficient Deep Learning Development}},
    url     = {http://tensorlayer.org},
    year    = {2017}
}

@inproceedings{tensorlayer2021,
  title={TensorLayer 3.0: A Deep Learning Library Compatible With Multiple Backends},
  author={Lai, Cheng and Han, Jiarong and Dong, Hao},
  booktitle={2021 IEEE International Conference on Multimedia \& Expo Workshops (ICMEW)},
  pages={1--3},
  year={2021},
  organization={IEEE}
}
```


