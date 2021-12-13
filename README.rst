|TENSORLAYER-LOGO|

TensorLayer3 is a deep learning library designed for researchers and engineers that is compatible with multiple deep learning frameworks such as TensorFlow,
MindSpore and PaddlePaddle, allowing users to run the code on different hardware like Nvidia-GPU and Huawei-Ascend.
It provides popular DL and RL modules that can be easily customized and assembled for tackling real-world machine learning problems.
More details can be found here. TensorLayer3 will support TensorFlow, MindSpore, PaddlePaddle, and PyTorch backends in the future.

Install
=======

TensorLayer3 has some prerequisites that need to be installed first, including TensorFlow ,
MindSpore, PaddlePaddle,numpy and matplotlib.For GPU support CUDA and cuDNN are required.

.. code:: bash

    # for last stable version
    pip install --upgrade tensorlayer3

    # for latest release candidate
    pip install --upgrade --pre tensorlayer3

    # if you want to install the additional dependencies, you can also run
    pip install --upgrade tensorlayer3[all]              # all additional dependencies
    pip install --upgrade tensorlayer3[extra]            # only the `extra` dependencies
    pip install --upgrade tensorlayer3[contrib_loggers]  # only the `contrib_loggers` dependencies

Alternatively, you can install the latest or development version by directly pulling from OpenI:

.. code:: bash

    pip3 install git+https://git.openi.org.cn/TensorLayer/tensorlayer3.0.git

Containers with CPU support
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    # for CPU version and Python 2
    docker pull tensorlayer/tensorlayer:latest
    docker run -it --rm -p 8888:8888 -p 6006:6006 -e PASSWORD=JUPYTER_NB_PASSWORD tensorlayer/tensorlayer:latest

    # for CPU version and Python 3
    docker pull tensorlayer/tensorlayer:latest-py3
    docker run -it --rm -p 8888:8888 -p 6006:6006 -e PASSWORD=JUPYTER_NB_PASSWORD tensorlayer/tensorlayer:latest-py3

Containers with GPU support
~~~~~~~~~~~~~~~~~~~~~~~~~~~

NVIDIA-Docker is required for these containers to work: `Project
Link <https://github.com/NVIDIA/nvidia-docker>`__

.. code:: bash

    # for GPU version and Python 2
    docker pull tensorlayer/tensorlayer:latest-gpu
    nvidia-docker run -it --rm -p 8888:88888 -p 6006:6006 -e PASSWORD=JUPYTER_NB_PASSWORD tensorlayer/tensorlayer:latest-gpu

    # for GPU version and Python 3
    docker pull tensorlayer/tensorlayer:latest-gpu-py3
    nvidia-docker run -it --rm -p 8888:8888 -p 6006:6006 -e PASSWORD=JUPYTER_NB_PASSWORD tensorlayer/tensorlayer:latest-gpu-py3

Contribute
==========

Please read the `Contributor
Guideline <https://git.openi.org.cn/TensorLayer/tensorlayer3.0/src/branch/master/CONTRIBUTING.md>`__
before submitting your PRs.

Cite
====

If you find this project useful, we would be grateful if you cite the
TensorLayer papers.

::

    @article{tensorlayer2017,
        author  = {Dong, Hao and Supratak, Akara and Mai, Luo and Liu, Fangde and Oehmichen, Axel and Yu, Simiao and Guo, Yike},
        journal = {ACM Multimedia},
        title   = {{TensorLayer: A Versatile Library for Efficient Deep Learning Development}},
        url     = {http://tensorlayer.org},
        year    = {2017}
    }
    @inproceedings{tensorlayer2021,
        title={Tensorlayer 3.0: A Deep Learning Library Compatible With Multiple Backends},
        author={Lai, Cheng and Han, Jiarong and Dong, Hao},
        booktitle={2021 IEEE International Conference on Multimedia \& Expo Workshops (ICMEW)},
        pages={1--3},
        year={2021},
        organization={IEEE}
    }

License
=======

TensorLayer3 is released under the Apache 2.0 license.

.. |TENSORLAYER-LOGO| image:: https://git.openi.org.cn/TensorLayer/tensorlayer3.0/src/branch/master/img/tl_transparent_logo.png
   :target: https://tensorlayer3.readthedocs.io/