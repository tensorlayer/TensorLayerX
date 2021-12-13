API - Optimizers
================

.. automodule:: tensorlayer.optimizers

TensorLayer3 provides simple API and tools to ease research, development and reduce the time to production.
Therefore, we provide the latest state of the art optimizers that work with Tensorflow, MindSpore and PaddlePaddle.
The optimizers functions provided by TensorFlow, MindSpore and PaddlePaddle can be used in TensorLayer3.
We have also wrapped the optimizers functions for each framework, which can be found in tensorlayer.optimizers.
In addition, we provide the latest state of Optimizers Dynamic Learning Rate that work with Tensorflow, MindSpore and PaddlePaddle.

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

.. automodule:: tensorlayer.optimizers.lr

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
.. autoclass:: tensorlayer.optimizers.Adadelta

Adagrad
^^^^^^^^^^^^^^^^
.. autoclass:: tensorlayer.optimizers.Adagrad

Adam
^^^^^^^^^^^^^^^^
.. autoclass:: tensorlayer.optimizers.Adam

Adamax
^^^^^^^^^^^^^^^^
.. autoclass:: tensorlayer.optimizers.Adamax

Ftrl
^^^^^^^^^^^^^^^^
.. autoclass:: tensorlayer.optimizers.Ftrl

Nadam
^^^^^^^^^^^^^^^^
.. autoclass:: tensorlayer.optimizers.Nadam

RMSprop
^^^^^^^^^^^^^^^^
.. autoclass:: tensorlayer.optimizers.RMSprop

SGD
^^^^^^^^^^^^^^^^
.. autoclass:: tensorlayer.optimizers.SGD

Momentum
^^^^^^^^^^^^^^^^
.. autoclass:: tensorlayer.optimizers.Momentum

Lamb
^^^^^^^^^^^^^^^^
.. autoclass:: tensorlayer.optimizers.Lamb

LARS
^^^^^^^^^^^^^^^^
.. autoclass:: tensorlayer.optimizers.LARS




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

