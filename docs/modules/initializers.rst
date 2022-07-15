API - Initializers
=========================

To make TensorLayerX simple, TensorLayerX only warps some basic initializers.
For more complex activation, TensorFlow(MindSpore, PaddlePaddle, PyTorch) API will be required.

.. automodule:: tensorlayerx.nn.initializers

.. autosummary::

   Initializer
   Zeros
   Ones
   Constant
   RandomUniform
   RandomNormal
   TruncatedNormal
   HeNormal
   HeUniform
   deconv2d_bilinear_upsampling_initializer
   XavierNormal
   XavierUniform

Initializer
------------
.. autoclass:: Initializer

Zeros
------------
.. autoclass:: Zeros

Ones
------------
.. autoclass:: Ones

Constant
-----------------
.. autoclass:: Constant

RandomUniform
--------------
.. autoclass:: RandomUniform

RandomNormal
---------------------
.. autoclass:: RandomNormal

TruncatedNormal
---------------------
.. autoclass:: TruncatedNormal

HeNormal
------------
.. autoclass:: HeNormal

HeUniform
------------
.. autoclass:: HeUniform

deconv2d_bilinear_upsampling_initializer
------------------------------------------
.. autofunction:: deconv2d_bilinear_upsampling_initializer

XavierNormal
---------------------
.. autoclass:: XavierNormal

XavierUniform
---------------------
.. autoclass:: XavierUniform