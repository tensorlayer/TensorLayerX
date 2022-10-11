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
[![Docker Pulls](https://img.shields.io/docker/pulls/tensorlayer/tensorlayerx.svg)](https://hub.docker.com/r/tensorlayer/tensorlayerx/)

🇬🇧 TensorLayerX is a multi-backend AI framework, which supports TensorFlow, Pytorch, MindSpore, PaddlePaddle, OneFlow and Jittor as the backends, allowing users to run the code on different hardware like Nvidia-GPU and Huawei-Ascend. 
This project is maintained by researchers from Peking University, Imperial College London, Princeton, Stanford, Tsinghua, Edinburgh and Peng Cheng Lab.
[supported layers](https://shimo.im/sheets/kJGCCTxXvqj99RGV/F5m5Z). 

🇨🇳 TensorLayerX 是一个跨平台开发框架，支持TensorFlow, Pytorch, MindSpore, PaddlePaddle, OneFlow和Jittor，用户不需要修改任何代码即可以运行在各类操作系统和AI硬件上（如Nvidia-GPU 和 Huawei-Ascend），并支持混合框架的开发。这个项目由北京大学、鹏城实验室、爱丁堡大学、帝国理工、清华、普林斯顿、斯坦福等机构的研究人员维护。
[支持列表](https://shimo.im/sheets/kJGCCTxXvqj99RGV/F5m5Z)。


# Document
TensorLayerX has extensive documentation for both beginners and professionals. 

[![English Documentation](https://img.shields.io/badge/documentation-english-blue.svg)](https://tensorlayerx.readthedocs.io/en/latest/)

# DeepLearning course  
🔥We have opened a video course for introductory learning deep learning, with example codes based on TensorLayerX.  
[Bilibili link](https://www.bilibili.com/video/BV1xB4y1h7V2?share_source=copy_web&vd_source=467c17f872fcde378494433520e19999)
# Design Features

Compare with [TensorLayer](https://github.com/tensorlayer/TensorLayer), [TensorLayerX](http://tensorlayerx.readthedocs.io)（TLX) is a brand new seperated project for platform-agnostic purpose. 

Compare to TensorLayer version:

<p align="center"><img src="https://git.openi.org.cn/hanjr/tensorlayerx-image/raw/branch/master/version.png" width="840"\></p>

🔥TensorLayerX inherits the features of the previous verison, including ***Simplicity***, ***Flexibility*** and ***Zero-cost Abstraction***. ***Compare with [TensorLayer](https://github.com/tensorlayer/TensorLayer), TensorLayerX supports more backends, such as TensorFlow, MindSpore, PaddlePaddle and PyTorch. It allows users to run the same code on different hardwares like Nvidia-GPU and Huawei-Ascend.*** In addition, more features are ***under development***.

- ***Model Zoo***: Build a series of model Zoos containing classic and sota models,covering CV, NLP, RL and other fields.

- ***Deploy***: In feature, TensorLayerX will support the ONNX protocol, supporting model export, import and deployment.

- ***Parallel***: In order to improve the efficiency of neural network model training, parallel computing is indispensable. 

# Resources

- [TLX2ONNX](https://github.com/tensorlayer/TLX2ONNX/) ONNX Model Exporter for TensorLayerX. ✅
- [Examples](https://github.com/tensorlayer/TensorLayerX/tree/main/examples) for tutorials✅
- [GammaGL](https://github.com/BUPT-GAMMA/GammaGL) is a multi-backend graph learning library based on TensorLayerX.✅
- OpenIVA an easy-to-use product-level deployment framework✅
- [TLXZoo](https://github.com/tensorlayer/TLXZoo) pretrained models/backbones🚧
- TLXCV a bunch of Computer Vision applications🚧
- TLXNLP a bunch of Natural Language Processing applications🚧
- TLXRL a bunch of Reinforcement Learning applications, check [RLZoo](https://github.com/tensorlayer/RLzoo) for the old version✅

More resources can be found [here](https://github.com/tensorlayer)


# Quick Start

## Installation

### Via docker

Docker is an open source application container engine. In the [TensorLayerX Docker Repository](https://hub.docker.com/repository/docker/tensorlayer/tensorlayerx), a specific version of TensorLayerX has been installed in docker images.

```bash
# pull from docker hub
docker pull tensorlayer/tensorlayerx:tagname
```


### Via pip
```bash
# install from pypi
pip3 install tensorlayerx 
```

### Build from source
```bash
# install from Github
pip3 install git+https://github.com/tensorlayer/tensorlayerx.git 
```
For more installation instructions, please refer to [Installtion](https://tensorlayerx.readthedocs.io/en/latest/user/installation.html)

## Define a model

You can immediately use tensorlayerx to define a model, using your favourite framework in the background, like so:
```python
import os
os.environ['TL_BACKEND'] = 'tensorflow' # modify this line, switch to any framework easily!
#os.environ['TL_BACKEND'] = 'mindspore'
#os.environ['TL_BACKEND'] = 'paddle'
#os.environ['TL_BACKEND'] = 'torch'
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import Linear
class CustomModel(Module):

  def __init__(self):
      super(CustomModel, self).__init__()

      self.linear1 = Linear(out_features=800, act=tlx.ReLU, in_features=784)
      self.linear2 = Linear(out_features=800, act=tlx.ReLU, in_features=800)
      self.linear3 = Linear(out_features=10, act=None, in_features=800)

  def forward(self, x, foo=False):
      z = self.linear1(x)
      z = self.linear2(z)
      out = self.linear3(z)
      if foo:
          out = tlx.softmax(out)
      return out

MLP = CustomModel()
MLP.set_eval()
```

# Contributing
Join our community as a code contributor, find out more in our [Help wanted list](https://github.com/tensorlayer/TensorLayerX/issues/5) and [Contributing](https://tensorlayerx.readthedocs.io/en/latest/user/contributing.html) guide!


# Contact
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


