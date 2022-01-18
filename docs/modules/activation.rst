API - Activations
=========================

To make TensorLayerX simple, we minimize the number of activation functions as much as
we can. So we encourage you to use Customizes activation function.
For parametric activation, please read the layer APIs.

Your activation
-------------------

Customizes activation function in TensorLayerX is very easy.
The following example implements an activation that multiplies its input by 2.

.. code-block:: python

  from tensorlayerx.nn import Module
  class DoubleActivation(Module):
    def __init__(self):
        pass
    def forward(self, x):
        return x * 2
  double_activation = DoubleActivation()

For more complex activation, TensorFlow(MindSpore, PaddlePaddle, PyTorch) API will be required.

.. automodule:: tensorlayerx.nn.activation

activation list
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

TensorLayerX Activations
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
