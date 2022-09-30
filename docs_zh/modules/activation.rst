API - Activations
=========================

为了让TensorLayerX更简洁，我们尽可能的减少预置激活函数的数量。因此我们鼓励您使用自定义的激活函数。
而对于参数化的激活函数选项，请参阅layer APIs。

您的激活函数
-------------------

TensorlayerX中自定义激活函数非常轻松。
下面的例子实现了一个将输入乘2的激活函数.

.. code-block:: python

  from tensorlayerx.nn import Module
  class DoubleActivation(Module):
    def __init__(self):
        pass
    def forward(self, x):
        return x * 2
  double_activation = DoubleActivation()

对于更复杂的激活函数，需要使用TensorFlow(MindSpore, PaddlePaddle, PyTorch) 的API。

.. automodule:: tensorlayerx.nn.activation

激活函数列表
----------------

.. autosummary::

   ELU
   PRelu
   PRelu6
   PTRelu6
   ReLU
   ReLU6
   Softplus
   LeakyReLU
   LeakyReLU6
   LeakyTwiceRelu6
   Ramp
   Swish
   HardTanh
   Tanh
   Sigmoid
   Softmax
   Mish

TensorLayerX 激活函数
--------------------------------

ELU
------
.. autoclass:: ELU

PRelu
------
.. autoclass:: PRelu

PRelu6
------------
.. autoclass:: PRelu6

PTRelu6
------------
.. autoclass:: PTRelu6

ReLU
-----------------
.. autoclass:: ReLU

ReLU6
-----------------
.. autoclass:: ReLU6

Softplus
-----------------
.. autoclass:: Softplus

LeakyReLU
-----------------
.. autoclass:: LeakyReLU

LeakyReLU6
------------
.. autoclass:: LeakyReLU6

LeakyTwiceRelu6
---------------------
.. autoclass:: LeakyTwiceRelu6

Ramp
---------------------
.. autoclass:: Ramp

Swish
--------------------
.. autoclass:: Swish

HardTanh
----------------
.. autoclass:: HardTanh

Mish
---------
.. autoclass:: Mish

Tanh
---------
.. autoclass:: Tanh

Sigmoid
---------
.. autoclass:: Sigmoid

Softmax
---------
.. autoclass:: Softmax


Parametric activation
------------------------------
See ``tensorlayerx.nn``.
