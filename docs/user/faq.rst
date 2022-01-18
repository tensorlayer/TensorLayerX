.. _faq:

============
FAQ
============


How to effectively learn TensorLayerX
========================================

No matter what stage you are in, we recommend you to spend just 10 minutes to
read the source code of TensorLayerX and the `Understand layer / Your layer <https://tensorlayerx.readthedocs.io/en/stable/modules/nn.html>`__
in this website, you will find the abstract methods are very simple for everyone.
Reading the source codes helps you to better understand TensorFlow, MindSpore, PaddlePaddle and allows
you to implement your own methods easily. For discussion, we recommend
`Gitter <https://gitter.im/tensorlayer/Lobby#?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge>`__,
`Help Wanted Issues <https://waffle.io/tensorlayer/tensorlayer>`__,
`QQ group <https://github.com/tensorlayer/tensorlayer/blob/master/img/img_qq.png>`__
and `Wechat group <https://github.com/shorxp/tensorlayer-chinese/blob/master/docs/wechat_group.md>`__.

Beginner
-----------
For people who new to deep learning, the contributors provided a number of tutorials in this website, these tutorials will guide you to understand  convolutional neural network, recurrent neural network, generative adversarial networks and etc. If your already understand the basic of deep learning, we recommend you to skip the tutorials and read the example codes on `Github <https://github.com/tensorlayer/TensorLayerX>`__ , then implement an example from scratch.

Engineer
------------
For people from industry, the contributors provided mass format-consistent examples covering computer vision, natural language processing and reinforcement learning. Besides, there are also many TensorFlow users already implemented product-level examples including image captioning, semantic/instance segmentation, machine translation, chatbot and etc., which can be found online.
It is worth noting that a wrapper especially for computer vision `Tf-Slim <https://github.com/tensorflow/models/tree/master/slim#Pretrained>`__ can be connected with TensorLayerX seamlessly.
Therefore, you may able to find the examples that can be used in your project.

Researcher
-------------
For people from academia, TensorLayerX was originally developed by PhD students who facing issues with other libraries on implement novel algorithm. Installing TensorLayer in editable mode is recommended, so you can extend your methods in TensorLayerX.
For research related to image processing such as image captioning, visual QA and etc., you may find it is very helpful to use the existing `Tf-Slim pre-trained models <https://github.com/tensorflow/models/tree/master/slim#Pretrained>`__ with TensorLayerX (a specially layer for connecting Tf-Slim is provided).


Install Master Version
========================

To use all new features of TensorLayerX, you need to install the master version from Github.
Before that, you need to make sure you already installed git.

.. code-block:: bash

  [stable version] pip3 install tensorlayerX
  [master version] pip3 install git+https://github.com/tensorlayer/TensorLayerX.git

Editable Mode
===============

- 1. Download the TensorLayerX folder from OpenI.
- 2. Before editing the TensorLayerX ``.py`` file.

 - If your script and TensorLayerX folder are in the same folder, when you edit the ``.py`` inside TensorLayerX folder, your script can access the new features.
 - If your script and TensorLayerX folder are not in the same folder, you need to run the following command in the folder contains ``setup.py`` before you edit ``.py`` inside TensorLayerX folder.

  .. code-block:: bash

    pip install -e .


Load Model
===========

Note that, the ``tl.files.load_npz()`` can only able to load the npz model saved by ``tl.files.save_npz()``.
If you have a model want to load into your TensorLayerX network, you can first assign your parameters into a list in order,
then use ``tl.files.assign_params()`` to load the parameters into your TensorLayerX model.



.. _GitHub: https://github.com/tensorlayer/TensorLayerX
.. _Deeplearning Tutorial: http://deeplearning.stanford.edu/tutorial/
.. _Convolutional Neural Networks for Visual Recognition: http://cs231n.github.io/
.. _Neural Networks and Deep Learning: http://neuralnetworksanddeeplearning.com/
.. _TensorFlow tutorial: https://www.tensorflow.org/versions/r0.9/tutorials/index.html
.. _Understand Deep Reinforcement Learning: http://karpathy.github.io/2016/05/31/rl/
.. _Understand Recurrent Neural Network: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
.. _Understand LSTM Network: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
.. _Word Representations: http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/
