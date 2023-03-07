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

[TensorLayerX](https://tensorlayerx.readthedocs.io) is a multi-backend AI framework, supports TensorFlow, Pytorch, MindSpore, PaddlePaddle, OneFlow and Jittor as the backends, allowing users to run the code on different hardware like Nvidia-GPU, Huawei-Ascend, Cambricon and more.
This project is maintained by researchers from Peking University, Imperial College London, Princeton, Stanford, Tsinghua, Edinburgh and Peng Cheng Lab.


- GitHub: https://github.com/tensorlayer/TensorLayerX  
- OpenI: https://openi.pcl.ac.cn/OpenI/TensorLayerX
- Homepage: [English](http://www.tensorlayerx.com/index_en.html?chlang=&langid=2) [中文](http://tensorlayerx.com)
- Document: https://tensorlayerx.readthedocs.io
- Previous Project: https://github.com/tensorlayer/TensorLayer

# Document
TensorLayerX has extensive documentation for both beginners and professionals. 

[![English Documentation](https://img.shields.io/badge/documentation-english-blue.svg)](https://tensorlayerx.readthedocs.io/en/latest/)

# Deep Learning course  
We have video courses for deep learning, with example codes based on TensorLayerX.  
[Bilibili link](https://www.bilibili.com/video/BV1xB4y1h7V2?share_source=copy_web&vd_source=467c17f872fcde378494433520e19999) (chinese)

# Design Features

<!-- <p align="center"><img src="https://git.openi.org.cn/hanjr/tensorlayerx-image/raw/branch/master/version.png" width="840"\></p> -->

- ***Compatibility***: Support worldwide frameworks and AI chips, enabling one code runs on all platforms.

- ***Model Zoo***: Provide a series of applications containing classic and SOTA models, covering CV, NLP, RL and other fields.

- ***Deployment***: Support ONNX protocol, model export, import and deployment.

# Multi-backend Design

You can immediately use TensorLayerX to define a model via Pytorch-stype, and switch to any backends easily.

```python
import os
os.environ['TL_BACKEND'] = 'tensorflow' # modify this line, switch to any backends easily!
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



# Resources

- [Examples](https://github.com/tensorlayer/TensorLayerX/tree/main/examples) for tutorials
- [GammaGL](https://github.com/BUPT-GAMMA/GammaGL) is series of graph learning algorithm
- [TLXZoo](https://github.com/tensorlayer/TLXZoo) a series of pretrained backbones
- [TLXCV](https://github.com/tensorlayer/TLXCV) a series of Computer Vision applications
- [TLXNLP](https://github.com/tensorlayer/TLXNLP) a series of Natural Language Processing applications
- [TLX2ONNX](https://github.com/tensorlayer/TLX2ONNX/) ONNX model exporter for TensorLayerX.
- [Paddle2TLX](https://github.com/tensorlayer/paddle2tlx) model code converter from PaddlePaddle to TensorLayerX.  

More official resources can be found [here](https://github.com/tensorlayer)


# Installation

- The latest TensorLayerX compatible with the following backend version

| TensorLayerX | TensorFlow | MindSpore | PaddlePaddle | PyTorch | OneFlow | Jittor|
| :-----:| :----: | :----: |:-----:|:----:|:----:|:----:|
|  v0.5.8  | v2.4.0 | v1.8.1 | v2.2.0 | v1.10.0 | -- | -- |
| v0.5.7 | v2.0.0 | v1.6.1 | v2.0.2 | v1.10.0 | -- | -- |

- via pip for the stable version
```bash
# install from pypi
pip3 install tensorlayerx 
```

- build from source for the latest version (for advanced users)
```bash
# install from Github
pip3 install git+https://github.com/tensorlayer/tensorlayerx.git 
```
For more installation instructions, please refer to [Installtion](https://tensorlayerx.readthedocs.io/en/latest/user/installation.html)


- via docker

Docker is an open source application container engine. In the [TensorLayerX Docker Repository](https://hub.docker.com/repository/docker/tensorlayer/tensorlayerx), 
different versions of TensorLayerX have been installed in docker images.

```bash
# pull from docker hub
docker pull tensorlayer/tensorlayerx:tagname
```

# Contributing
Join our community as a code contributor, find out more in our [Help wanted list](https://github.com/tensorlayer/TensorLayerX/issues/5) and [Contributing](https://tensorlayerx.readthedocs.io/en/latest/user/contributing.html) guide!


# Getting Involved

We suggest users to report bugs using Github issues. Users can also discuss how to use TensorLayerX in the following slack channel.

<br/>

<a href="https://join.slack.com/t/tensorlayer/shared_invite/enQtODk1NTQ5NTY1OTM5LTQyMGZhN2UzZDBhM2I3YjYzZDBkNGExYzcyZDNmOGQzNmYzNjc3ZjE3MzhiMjlkMmNiMmM3Nzc4ZDY2YmNkMTY" target="\_blank">
	<div align="center">
		<img src="https://github.com/tensorlayer/TensorLayer/blob/bdc2c14ff9ed9bd3ec7004d625e15683df7b530d/img/join_slack.png?raw=true" width="40%"/>
	</div>
</a>

# Contact
 - tensorlayer@gmail.com

# Citation

If you find TensorLayerX useful for your project, please cite the following papers：

```
@inproceedings{tensorlayer2021,
  title={TensorLayer 3.0: A Deep Learning Library Compatible With Multiple Backends},
  author={Lai, Cheng and Han, Jiarong and Dong, Hao},
  booktitle={2021 IEEE International Conference on Multimedia \& Expo Workshops (ICMEW)},
  pages={1--3},
  year={2021},
  organization={IEEE}
}
@article{tensorlayer2017,
    author  = {Dong, Hao and Supratak, Akara and Mai, Luo and Liu, Fangde and Oehmichen, Axel and Yu, Simiao and Guo, Yike},
    journal = {ACM Multimedia},
    title   = {{TensorLayer: A Versatile Library for Efficient Deep Learning Development}},
    url     = {http://tensorlayer.org},
    year    = {2017}
} 
```


