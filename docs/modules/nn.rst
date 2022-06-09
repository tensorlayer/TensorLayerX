API - NN
============

.. automodule:: tensorlayerx.nn

.. -----------------------------------------------------------
..                        Layer List
.. -----------------------------------------------------------

Layer list
----------

.. autosummary::

   Module
   Sequential
   ModuleList
   ModuleDict
   Parameter
   ParameterList
   ParameterDict

   Input

   OneHot
   Word2vecEmbedding
   Embedding
   AverageEmbedding

   Linear
   Dropout
   GaussianNoise
   DropconnectLinear

   UpSampling2d
   DownSampling2d

   Conv1d
   Conv2d
   Conv3d
   ConvTranspose1d
   ConvTranspose2d
   ConvTranspose3d
   DepthwiseConv2d
   SeparableConv1d
   SeparableConv2d
   DeformableConv2d
   GroupConv2d

   PadLayer
   PoolLayer
   ZeroPad1d
   ZeroPad2d
   ZeroPad3d
   MaxPool1d
   MeanPool1d
   MaxPool2d
   MeanPool2d
   MaxPool3d
   MeanPool3d
   GlobalMaxPool1d
   GlobalMeanPool1d
   GlobalMaxPool2d
   GlobalMeanPool2d
   GlobalMaxPool3d
   GlobalMeanPool3d
   AdaptiveMeanPool1d
   AdaptiveMaxPool1d
   AdaptiveMeanPool2d
   AdaptiveMaxPool2d
   AdaptiveMeanPool3d
   AdaptiveMaxPool3d
   CornerPool2d

   SubpixelConv1d
   SubpixelConv2d

   BatchNorm
   BatchNorm1d
   BatchNorm2d
   BatchNorm3d
   LayerNorm

   RNNCell
   LSTMCell
   GRUCell
   RNN
   LSTM
   GRU
   MultiheadAttention
   Transformer
   TransformerEncoder
   TransformerDecoder
   TransformerEncoderLayer
   TransformerDecoderLayer


   Flatten
   Reshape
   Transpose
   Shuffle

   Concat
   Elementwise

   ExpandDims
   Tile

   Stack
   UnStack

   Scale
   BinaryLinear
   BinaryConv2d
   TernaryLinear
   TernaryConv2d
   DorefaLinear
   DorefaConv2d

   MaskedConv3d

.. -----------------------------------------------------------
..                        Basic Layers
.. -----------------------------------------------------------

Base Layer
-----------

Module
^^^^^^^^^^^^^^^^
.. autoclass:: Module

Sequential
^^^^^^^^^^^^^^^^
.. autoclass:: Sequential

ModuleList
^^^^^^^^^^^^^^^^
.. autoclass:: ModuleList

ModuleDict
^^^^^^^^^^^^^^^^
.. autoclass:: ModuleDict

Parameter
^^^^^^^^^^^^^^^^
.. autofunction:: Parameter

ParameterList
^^^^^^^^^^^^^^^^
.. autoclass:: ParameterList

ParameterDict
^^^^^^^^^^^^^^^^
.. autoclass:: ParameterDict

.. -----------------------------------------------------------
..                        Input Layer
.. -----------------------------------------------------------

Input Layers
---------------

Input Layer
^^^^^^^^^^^^^^^^
.. autofunction:: Input

.. -----------------------------------------------------------
..                        Embedding Layers
.. -----------------------------------------------------------


One-hot Layer
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: OneHot

Word2Vec Embedding Layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Word2vecEmbedding

Embedding Layer
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Embedding

Average Embedding Layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: AverageEmbedding


.. -----------------------------------------------------------
..                  Convolutional Layers
.. -----------------------------------------------------------

Convolutional Layers
---------------------

Convolutions
^^^^^^^^^^^^^^^^^^^^^^^^^^

Conv1d
"""""""""""""""""""""
.. autoclass:: Conv1d

Conv2d
"""""""""""""""""""""
.. autoclass:: Conv2d

Conv3d
"""""""""""""""""""""
.. autoclass:: Conv3d

Deconvolutions
^^^^^^^^^^^^^^^^^^^^^^^^^^

ConvTranspose1d
"""""""""""""""""""""
.. autoclass:: ConvTranspose1d

ConvTranspose2d
"""""""""""""""""""""
.. autoclass:: ConvTranspose2d

ConvTranspose3d
"""""""""""""""""""""
.. autoclass:: ConvTranspose3d


Deformable Convolutions
^^^^^^^^^^^^^^^^^^^^^^^^^^

DeformableConv2d
"""""""""""""""""""""
.. autoclass:: DeformableConv2d


Depthwise Convolutions
^^^^^^^^^^^^^^^^^^^^^^^^^^

DepthwiseConv2d
"""""""""""""""""""""
.. autoclass:: DepthwiseConv2d


Group Convolutions
^^^^^^^^^^^^^^^^^^^^^^^^^^

GroupConv2d
"""""""""""""""""""""
.. autoclass:: GroupConv2d


Separable Convolutions
^^^^^^^^^^^^^^^^^^^^^^^^^^

SeparableConv1d
"""""""""""""""""""""
.. autoclass:: SeparableConv1d

SeparableConv2d
"""""""""""""""""""""
.. autoclass:: SeparableConv2d


SubPixel Convolutions
^^^^^^^^^^^^^^^^^^^^^^^^^^

SubpixelConv1d
"""""""""""""""""""""
.. autoclass:: SubpixelConv1d

SubpixelConv2d
"""""""""""""""""""""
.. autoclass:: SubpixelConv2d

MaskedConv3d
""""""""""""""""""""
.. autoclass:: MaskedConv3d


.. -----------------------------------------------------------
..                        Linear Layers
.. -----------------------------------------------------------

Linear Layers
-------------

Linear Layer
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Linear

Drop Connect Linear Layer
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DropconnectLinear


.. -----------------------------------------------------------
..                       Dropout Layer
.. -----------------------------------------------------------

Dropout Layers
-------------------
.. autoclass:: Dropout

.. -----------------------------------------------------------
..                        Extend Layers
.. -----------------------------------------------------------

Extend Layers
-------------------

Expand Dims Layer
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: ExpandDims


Tile layer
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Tile

.. -----------------------------------------------------------
..                  Image Resampling Layers
.. -----------------------------------------------------------

Image Resampling Layers
-------------------------

2D UpSampling
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: UpSampling2d

2D DownSampling
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DownSampling2d

.. -----------------------------------------------------------
..                      Merge Layer
.. -----------------------------------------------------------

Merge Layers
---------------

Concat Layer
^^^^^^^^^^^^^^^^^^^
.. autoclass:: Concat

ElementWise Layer
^^^^^^^^^^^^^^^^^^^
.. autoclass:: Elementwise

.. -----------------------------------------------------------
..                      Noise Layers
.. -----------------------------------------------------------

Noise Layer
---------------
.. autoclass:: GaussianNoise

.. -----------------------------------------------------------
..                  Normalization Layers
.. -----------------------------------------------------------

Normalization Layers
--------------------

Batch Normalization
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: BatchNorm

Batch Normalization 1D
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: BatchNorm1d

Batch Normalization 2D
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: BatchNorm2d

Batch Normalization 3D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: BatchNorm3d

.. -----------------------------------------------------------
..                     Padding Layers
.. -----------------------------------------------------------

Padding Layers
------------------------

Pad Layer (Expert API)
^^^^^^^^^^^^^^^^^^^^^^^^^
Padding layer for any modes.

.. autoclass:: PadLayer

1D Zero padding
^^^^^^^^^^^^^^^^^^^
.. autoclass:: ZeroPad1d

2D Zero padding
^^^^^^^^^^^^^^^^^^^
.. autoclass:: ZeroPad2d

3D Zero padding
^^^^^^^^^^^^^^^^^^^
.. autoclass:: ZeroPad3d

.. -----------------------------------------------------------
..                     Pooling Layers
.. -----------------------------------------------------------

Pooling Layers
------------------------

Pool Layer (Expert API)
^^^^^^^^^^^^^^^^^^^^^^^^^
Pooling layer for any dimensions and any pooling functions.

.. autoclass:: PoolLayer

1D Max pooling
^^^^^^^^^^^^^^^^^^^
.. autoclass:: MaxPool1d

1D Mean pooling
^^^^^^^^^^^^^^^^^^^
.. autoclass:: MeanPool1d

2D Max pooling
^^^^^^^^^^^^^^^^^^^
.. autoclass:: MaxPool2d

2D Mean pooling
^^^^^^^^^^^^^^^^^^^
.. autoclass:: MeanPool2d

3D Max pooling
^^^^^^^^^^^^^^^^^^^
.. autoclass:: MaxPool3d

3D Mean pooling
^^^^^^^^^^^^^^^^^^^
.. autoclass:: MeanPool3d

1D Global Max pooling
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GlobalMaxPool1d

1D Global Mean pooling
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GlobalMeanPool1d

2D Global Max pooling
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GlobalMaxPool2d

2D Global Mean pooling
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GlobalMeanPool2d

3D Global Max pooling
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GlobalMaxPool3d

3D Global Mean pooling
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GlobalMeanPool3d

1D Adaptive Max pooling
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: AdaptiveMaxPool1d

1D Adaptive Mean pooling
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: AdaptiveMeanPool1d

2D Adaptive Max pooling
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: AdaptiveMaxPool2d

2D Adaptive Mean pooling
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: AdaptiveMeanPool2d

3D Adaptive Max pooling
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: AdaptiveMaxPool3d

3D Adaptive Mean pooling
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: AdaptiveMeanPool3d

2D Corner pooling
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: CornerPool2d

.. -----------------------------------------------------------
..                    Quantized Layers
.. -----------------------------------------------------------

Quantized Nets
------------------

This is an experimental API package for building Quantized Neural Networks. We are using matrix multiplication rather than add-minus and bit-count operation at the moment. Therefore, these APIs would not speed up the inferencing, for production, you can train model via TensorLayer and deploy the model into other customized C/C++ implementation (We probably provide users an extra C/C++ binary net framework that can load model from TensorLayer).

Note that, these experimental APIs can be changed in the future.


Scale
^^^^^^^^^^^^^^
.. autoclass:: Scale

Binary Linear Layer
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: BinaryLinear

Binary (De)Convolutions
^^^^^^^^^^^^^^^^^^^^^^^^^^

BinaryConv2d
"""""""""""""""""""""
.. autoclass:: BinaryConv2d

Ternary Linear Layer
^^^^^^^^^^^^^^^^^^^^^^^^^^

TernaryLinear
"""""""""""""""""""""
.. autoclass:: TernaryLinear

Ternary Convolutions
^^^^^^^^^^^^^^^^^^^^^^^^^^

TernaryConv2d
"""""""""""""""""""""
.. autoclass:: TernaryConv2d

DorefaLinear
"""""""""""""""""""""
.. autoclass:: DorefaLinear

DoReFa Convolutions
^^^^^^^^^^^^^^^^^^^^^^^^^^

DorefaConv2d
"""""""""""""""""""""
.. autoclass:: DorefaConv2d


.. -----------------------------------------------------------
..                  Recurrent Layers
.. -----------------------------------------------------------

Recurrent Layers
---------------------

Common Recurrent layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RNNCell layer
""""""""""""""""""""""""""
.. autoclass:: RNNCell
    :members:

LSTMCell layer
""""""""""""""""""""""""""""""""""
.. autoclass:: LSTMCell
    :members:

GRUCell layer
""""""""""""""""""""""""""""""""""
.. autoclass:: GRUCell
    :members:

RNN layer
""""""""""""""""""""""""""""""""""
.. autoclass:: RNN
    :members:

LSTM layer
"""""""""""""""""""""""""""""""""
.. autoclass:: LSTM
    :members:

GRU layer
"""""""""""""""""""""""""""""""""
.. autoclass:: GRU
    :members:

.. -----------------------------------------------------------
..                  Transformer Layers
.. -----------------------------------------------------------

Transformer Layers
---------------------

Transformer layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


MultiheadAttention layer
"""""""""""""""""""""""""""""""""""
.. autoclass:: MultiheadAttention
    :members:

Transformer layer
""""""""""""""""""""""""""""""""""
.. autoclass:: Transformer
    :members:

TransformerEncoder layer
""""""""""""""""""""""""""""""""""
.. autoclass:: TransformerEncoder
    :members:

TransformerDecoder layer
""""""""""""""""""""""""""""""""""
.. autoclass:: TransformerDecoder
    :members:

TransformerEncoderLayer layer
"""""""""""""""""""""""""""""""""""""""
.. autoclass:: TransformerEncoderLayer
    :members:

TransformerDecoderLayer layer
"""""""""""""""""""""""""""""""""""""""
.. autoclass:: TransformerDecoderLayer
    :members:

.. -----------------------------------------------------------
..                      Shape Layers
.. -----------------------------------------------------------

Shape Layers
------------

Flatten Layer
^^^^^^^^^^^^^^^
.. autoclass:: Flatten

Reshape Layer
^^^^^^^^^^^^^^^
.. autoclass:: Reshape

Transpose Layer
^^^^^^^^^^^^^^^^^
.. autoclass:: Transpose

Shuffle Layer
^^^^^^^^^^^^^^^^^
.. autoclass:: Shuffle

.. -----------------------------------------------------------
..                      Stack Layers
.. -----------------------------------------------------------

Stack Layer
-------------

Stack Layer
^^^^^^^^^^^^^^
.. autoclass:: Stack

Unstack Layer
^^^^^^^^^^^^^^^
.. autoclass:: UnStack

