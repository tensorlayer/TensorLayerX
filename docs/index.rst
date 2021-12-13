Welcome to TensorLayer3
=======================================


.. image:: user/my_figs/tl_transparent_logo.png
  :width: 30 %
  :align: center
  :target: https://git.openi.org.cn/TensorLayer/tensorlayer3.0

**Documentation Version:** |release|

`TensorLayer3`_ is a deep learning library designed for researchers and engineers that is compatible with multiple deep learning frameworks
such as TensorFlow, MindSpore and PaddlePaddle, allowing users to run the code on different hardware like Nvidia-GPU and Huawei-Ascend.
It provides popular DL and RL modules that can be easily customized and assembled for tackling real-world machine learning problems.
More details can be found `here <https://git.openi.org.cn/TensorLayer/tensorlayer3.0>`_.
TensorLayer3 will support TensorFlow, MindSpore, PaddlePaddle, and PyTorch backends in the future.

.. note::
   If you got problem to read the docs online, you could download the repository
   on `OpenI`_, then go to ``/docs/_build/html/index.html`` to read the docs
   offline. The ``_build`` folder can be generated in ``docs`` using ``make html``.

User Guide
------------

The TensorLayer3 user guide explains how to install TensorFlow, CUDA and cuDNN,
how to build and train neural networks using TensorLayer3, and how to contribute
to the library as a developer.

.. toctree::
  :maxdepth: 2

  user/installation
  user/examples
  user/contributing
  user/get_involved
  user/faq

.. toctree::
  :maxdepth: 2
  :caption: Getting started

  user/get_start_model
  user/get_start_advance

API Reference
-------------

If you are looking for information on a specific function, class or
method, this part of the documentation is for you.

.. toctree::
  :maxdepth: 2
  :caption: Stable Functionalities

  modules/activation
  modules/cost
  modules/dataflow
  modules/prepro
  modules/files
  modules/iterate
  modules/layers
  modules/models
  modules/pretrain_models
  modules/nlp
  modules/vision
  modules/initializers
  modules/visualize
  modules/backend_ops

.. toctree::
  :maxdepth: 2
  :caption: Alpha Version Functionalities

  modules/db
  modules/optimizers
  modules/distributed
  
Command-line Reference
----------------------

TensorLayer3 provides a handy command-line tool `tl` to perform some common tasks.

.. toctree::
  :maxdepth: 2
  :caption: Command Line Interface

  modules/cli


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _OpenI: https://git.openi.org.cn/TensorLayer/tensorlayer3.0
.. _TensorLayer3: https://git.openi.org.cn/TensorLayer/tensorlayer3.0
