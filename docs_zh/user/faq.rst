.. _faq:

============
FAQ
============


如何高效地学习TensorlayerX
========================================

无论您处于哪一个学习阶段，我们都建议您花费十分钟时间去阅读TensorlayerX的源码
以及 `理解网络层/您的网络层 <https://tensorlayerx.readthedocs.io/en/stable/modules/nn.html>`__
在这个页面您会发现抽象方法对每个人来说都很简单。
阅读源码可以帮助您更好的理解 TensorFlow, MindSpore, PaddlePaddle 等深度学习框架，并让您可以轻松地实现自己的
算法。
至于讨论，我们推荐如下渠道
`Gitter <https://gitter.im/tensorlayer/Lobby#?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge>`__,
`Help Wanted Issues <https://waffle.io/tensorlayer/tensorlayer>`__,
`QQ group <https://github.com/tensorlayer/tensorlayer/blob/master/img/img_qq.png>`__
and `Wechat group <https://github.com/shorxp/tensorlayer-chinese/blob/master/docs/wechat_group.md>`__.

初学者
-----------
对于新接触深度学习的人，开发者们在网站上提供了一系列教程，这些教程会引导您理解卷积神经网络、循环神经网络、对抗生成网络等。
如果您已经理解了深度学习基础，那我们推荐您跳过教程并阅读示例源码 `Github <https://github.com/tensorlayer/TensorLayerX>`__ ，然后从头开始实现一个示例。

工程师
------------
对于工业界人员，贡献者提供了大量格式一致的示例，涵盖计算机视觉、自然语言处理和强化学习。此外，还有很多 TensorFlow 用户已经实现了产品级的示例，包括图像描述、语义/实例分割、机器翻译、聊天机器人等，这些都可以在网上找到。
值得注意的是，一个针对计算机视觉的封装器，可以和TensorlayerX无缝对接。
因此，您能够找到可以在您的项目中使用的示例。

研究者
-------------
对于学术界人士来说，TensorLayerX 最初是由博士生们开发的，因为他们在使用其他库实现新算法遇到了问题。 
建议以可编辑模式安装 TensorLayerX，以便您可以在 TensorLayerX 中扩展您的方法。
对于与图像处理相关的研究，例如图像描述、视觉 QA 等，您会发现结合TensorLayerX使用 `Tf-Slim 预训练模型 <https://github.com/tensorflow/models/tree/master/slim#Pretrained>`__ 非常有帮助(提供了对接TF-Slim的特殊网络层)。


安装Master版本
========================

要使用 TensorLayerX 的所有最新功能，您需要从 Github 安装Master版本。
在这之前，请确认您已经安装了Git。

.. code-block:: bash

  [稳定版] pip3 install tensorlayerX
  [master 版本] pip3 install git+https://github.com/tensorlayer/TensorLayerX.git

可编辑模式
===============

- 1. 从OpenI下载TensorlayerX目录.
- 2. 在编辑 TensorLayerX的 ``.py`` 文件前，

 - 如果您的脚本和TensorlayerX在同一个目录，当你在TensorlayerX目录内编辑 ``.py`` 文件时，你的脚本即可访问最新的功能。
 - 如果您的脚本和TensorlayerX不在同一个目录，你需要在编辑 ``.py`` 文件前，在包含 ``setup.py`` 文件的目录运行以下命令 

  .. code-block:: bash

    pip install -e .

加载模型
===========

注意， ``tlx.files.load_npz()`` 只能加载由 ``tlx.files.save_npz()`` 保存的 ``.npz`` model.
如果您想要将一系列模型参数加载进您的TensorlayerX模型, 您可以首先将模型参数按顺序赋值给一个 ``list`` ,
然后调用 ``tlx.files.assign_params()`` 将参数加载进你的TensorlayerX模型



.. _GitHub: https://github.com/tensorlayer/TensorLayerX
.. _Deeplearning Tutorial: http://deeplearning.stanford.edu/tutorial/
.. _Convolutional Neural Networks for Visual Recognition: http://cs231n.github.io/
.. _Neural Networks and Deep Learning: http://neuralnetworksanddeeplearning.com/
.. _TensorFlow tutorial: https://www.tensorflow.org/versions/r0.9/tutorials/index.html
.. _Understand Deep Reinforcement Learning: http://karpathy.github.io/2016/05/31/rl/
.. _Understand Recurrent Neural Network: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
.. _Understand LSTM Network: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
.. _Word Representations: http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/
