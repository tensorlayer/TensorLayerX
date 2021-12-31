<a href="https://tensorlayerx.readthedocs.io/">
    <div align="center">
        <img src="https://git.openi.org.cn/hanjr/tensorlayerx-image/raw/branch/master/tlx-LOGO-04.png" width="50%" height="30%"/>
    </div>
</a>

<!--- [![PyPI Version](https://badge.fury.io/py/tensorlayer.svg)](https://pypi.org/project/tensorlayerx/) --->
<!--- ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tensorlayer.svg)) --->

[![GitHub last commit (branch)](https://img.shields.io/github/last-commit/tensorlayer/tensorlayerx/master.svg)](https://github.com/tensorlayer/TensorLayerX)
[![Documentation Status](https://readthedocs.org/projects/tensorlayerx/badge/)]( https://tensorlayerx.readthedocs.io/en/latest/)
[![Build Status](https://travis-ci.org/tensorlayer/tensorlayerx.svg?branch=master)](https://travis-ci.org/tensorlayer/tensorlayerx)
[![Downloads](http://pepy.tech/badge/tensorlayerx)](http://pepy.tech/project/tensorlayerx)
[![Downloads](https://pepy.tech/badge/tensorlayerx/week)](https://pepy.tech/project/tensorlayerx/week)
[![Docker Pulls](https://img.shields.io/docker/pulls/tensorlayer/tensorlayerx.svg)](https://hub.docker.com/r/tensorlayer/tensorlayerx/)

🇬🇧 TensorLayerX is a multi-backend AI framework, which can run on almost all operation systems and AI hardwares, and support hybrid-framework programming. The currently version supports TensorFlow, MindSpore, PaddlePaddle and PyTorch(partial) as the backends.[layer list](https://shimo.im/sheets/kJGCCTxXvqj99RGV/F5m5Z). 

🇨🇳 TensorLayerX 是一个跨平台开发框架，可以运行在各类操作系统和AI硬件上，并支持混合框架的开发。目前支持TensorFlow、MindSpore、PaddlePaddle框架常用神经网络层以及算子，PyTorch支持特性正在开发中，[支持列表](https://shimo.im/sheets/kJGCCTxXvqj99RGV/F5m5Z)。


<details>
    <summary>🇷🇺 TensorLayerX</summary>
input text here.
</details>

<details>
    <summary>🇸🇦 TensorLayerX</summary>
input text here.
</details>

# TensorLayerX

Compare with [TensorLayer](https://github.com/tensorlayer/TensorLayer), TensorLayerX （TLX) is a brand new seperated project for platform-agnostic purpose. 

Comparison of TensorLayer version

<p align="center"><img src="https://git.openi.org.cn/hanjr/tensorlayerx-image/raw/branch/master/version.png" width="840"\></p>

🔥TensorLayerX inherits the features of the previous verison, including ***Simplicity***, ***Flexibility*** and ***Zero-cost Abstraction***. ***Compare with [TensorLayer](https://github.com/tensorlayer/TensorLayer), TensorLayerX supports more backends, such as TensorFlow, MindSpore, PaddlePaddle and PyTorch. It allows users to run the same code on different hardwares like Nvidia-GPU and Huawei-Ascend.*** In addition, more features are ***under development***.

- ***Model Zoo***: Build a series of model Zoos containing classic and sota models,covering CV, NLP, RL and other fields.

- ***Deploy***: In feature, TensorLayerX will support the ONNX protocol, supporting model export, import and deployment.

- ***Parallel***: In order to improve the efficiency of neural network model training, parallel computing is indispensable. 

🔥**Feel free to use TensorLayerX and make suggestions. We need more people to join the dev team, if you are interested, please email hao.dong@pku.edu.cn**

# Examples

- [Basic Examples](https://github.com/tensorlayer/TensorLayerX/tree/main/examples)
- [TLCV]**Coming soon!**



# Quick Start

- Installation
```bash
# install from pypi
pip3 install tensorlayerx 
# install from Github
pip3 install git+https://github.com/tensorlayer/tensorlayerx.git 
# install from OpenI
pip3 install git+https://git.openi.org.cn/OpenI/tensorlayerX.git
```
If you want to use TensorFlow backend, you should install TensorFlow：
```bash
pip3 install tensorflow # if you want to use GPUs, CUDA and CuDNN are required.
```


If you want to use MindSpore backend, you should install mindspore>=1.2.1
```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.2.1/MindSpore/gpu/ubuntu_x86/cuda-10.1/mindspore_gpu-1.2.1-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

If you want to use paddlepaddle backend, you should install paddlepaddle>=2.1.1
```bash
python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```

If you want to use PyTorch backend, you should install PyTorch>=1.8.0
```bash
pip3 install torch==1.8.2+cu102 torchvision==0.9.2+cu102 torchaudio===0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```

- [Tutorial](https://github.com/tensorlayer/TensorLayerX/tree/main/examples/basic_tutorials)

- Discussion: [Slack](https://join.slack.com/t/tensorlayer/shared_invite/enQtODk1NTQ5NTY1OTM5LTQyMGZhN2UzZDBhM2I3YjYzZDBkNGExYzcyZDNmOGQzNmYzNjc3ZjE3MzhiMjlkMmNiMmM3Nzc4ZDY2YmNkMTY) , [QQ-Group] , [WeChat-Group]



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


