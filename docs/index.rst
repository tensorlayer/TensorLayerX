Welcome to TensorLayerX
=======================================


.. image:: user/my_figs/tlx_transparent_logo.png
  :width: 30 %
  :align: center
  :target: https://github.com/tensorlayer/TensorLayerX

**Documentation Version:** |release|

`TensorLayerX`_ is a deep learning library designed for researchers and engineers that is compatible with multiple deep learning frameworks
such as TensorFlow, MindSpore and PaddlePaddle, allowing users to run the code on different hardware like Nvidia-GPU and Huawei-Ascend.
It provides popular DL and RL modules that can be easily customized and assembled for tackling real-world machine learning problems.
More details can be found `here <https://github.com/tensorlayer/TensorLayerX>`_.

TensorLayerX is a multi-backend AI framework, which can run on almost all operation systems and AI hardwares, and support hybrid-framework programming. The currently version supports TensorFlow, MindSpore, PaddlePaddle and PyTorch(partial) as the backends.

.. note::
   If you got problem to read the docs online, you could download the repository
   on `TensorLayerX`_, then go to ``/docs/_build/html/index.html`` to read the docs
   offline. The ``_build`` folder can be generated in ``docs`` using ``make html``.

User Guide
------------

The TensorLayerX user guide explains how to install TensorFlow, CUDA and cuDNN,
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
  modules/losses
  modules/dataflow
  modules/files
  modules/nn
  modules/model
  modules/vision
  modules/initializers
  modules/ops
  modules/optimizers

Command-line Reference
----------------------

TensorLayerX provides a handy command-line tool `tlx` to perform some common tasks.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _OpenI: https://git.openi.org.cn/OpenI/TensorLayerX
.. _TensorLayerX: https://github.com/tensorlayer/TensorLayerX
