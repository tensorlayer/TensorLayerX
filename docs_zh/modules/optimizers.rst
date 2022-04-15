API - Optimizers
================

.. automodule:: tensorlayerx.optimizers

TensorLayerX provides simple API and tools to ease research, development and reduce the time to production.
Therefore, we provide the latest state of the art optimizers that work with Tensorflow, MindSpore, PaddlePaddle and PyTorch.
The optimizers functions provided by Tensorflow, MindSpore, PaddlePaddle and PyTorch can be used in TensorLayerX.
We have also wrapped the optimizers functions for each framework, which can be found in tensorlayerx.optimizers.
In addition, we provide the latest state of Optimizers Dynamic Learning Rate that work with Tensorflow, MindSpore, PaddlePaddle and PyTorch.

Optimizers List
---------------

.. autosummary::

   Adadelta
   Adagrad
   Adam
   Adamax
   Ftrl
   Nadam
   RMSprop
   SGD
   Momentum
   Lamb
   LARS

.. automodule:: tensorlayerx.optimizers.lr

Optimizers Dynamic Learning Rate List
--------------------------------------

.. autosummary::

   LRScheduler
   StepDecay
   CosineAnnealingDecay
   NoamDecay
   PiecewiseDecay
   NaturalExpDecay
   InverseTimeDecay
   PolynomialDecay
   LinearWarmup
   ExponentialDecay
   MultiStepDecay
   LambdaDecay
   ReduceOnPlateau


Adadelta
^^^^^^^^^^
.. autoclass:: tensorlayerx.optimizers.Adadelta

Adagrad
^^^^^^^^^^^^^^^^
.. autoclass:: tensorlayerx.optimizers.Adagrad

Adam
^^^^^^^^^^^^^^^^
.. autoclass:: tensorlayerx.optimizers.Adam

Adamax
^^^^^^^^^^^^^^^^
.. autoclass:: tensorlayerx.optimizers.Adamax

Ftrl
^^^^^^^^^^^^^^^^
.. autoclass:: tensorlayerx.optimizers.Ftrl

Nadam
^^^^^^^^^^^^^^^^
.. autoclass:: tensorlayerx.optimizers.Nadam

RMSprop
^^^^^^^^^^^^^^^^
.. autoclass:: tensorlayerx.optimizers.RMSprop

SGD
^^^^^^^^^^^^^^^^
.. autoclass:: tensorlayerx.optimizers.SGD

Momentum
^^^^^^^^^^^^^^^^
.. autoclass:: tensorlayerx.optimizers.Momentum

Lamb
^^^^^^^^^^^^^^^^
.. autoclass:: tensorlayerx.optimizers.Lamb

LARS
^^^^^^^^^^^^^^^^
.. autoclass:: tensorlayerx.optimizers.LARS




LRScheduler
^^^^^^^^^^^^^^^^
.. autoclass:: LRScheduler

StepDecay
^^^^^^^^^^^^^^^^
.. autoclass:: StepDecay

CosineAnnealingDecay
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: CosineAnnealingDecay

NoamDecay
^^^^^^^^^^^^^^^^
.. autoclass:: NoamDecay

PiecewiseDecay
^^^^^^^^^^^^^^^^
.. autoclass:: PiecewiseDecay

NaturalExpDecay
^^^^^^^^^^^^^^^^
.. autoclass:: NaturalExpDecay

InverseTimeDecay
^^^^^^^^^^^^^^^^^^^
.. autoclass:: InverseTimeDecay

PolynomialDecay
^^^^^^^^^^^^^^^^
.. autoclass:: PolynomialDecay

LinearWarmup
^^^^^^^^^^^^^^^^
.. autoclass:: LinearWarmup

ExponentialDecay
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: ExponentialDecay

MultiStepDecay
^^^^^^^^^^^^^^^^
.. autoclass:: MultiStepDecay

LambdaDecay
^^^^^^^^^^^^^^^^
.. autoclass:: LambdaDecay

ReduceOnPlateau
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: ReduceOnPlateau

