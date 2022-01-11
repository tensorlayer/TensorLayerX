#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import tensorflow as tf

__all__ = ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD', 'Momentum', 'Lamb', 'LARS']


class Adadelta(tf.optimizers.Adadelta):
    """Optimizer that implements the Adadelta algorithm. Equivalent to tf.optimizers.Adadelta.

    References
    ----------
    - https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers/Adadelta?hl=en

    Parameters
    ----------
    learning_rate : A Tensor, floating point value
        The learning rate. Defaults to 0.001.
    rho : float or constant float tensor
        A Tensor or a floating point value. The decay rate.
    epsilon : float
        A small constant for numerical stability.Defaults to 1e-7.

    Examples
    --------
    With TensorLayer

    >>> import tensorlayerx as tl
    >>> optimizer = tl.optimizers.Adadelta(0.0001)
    >>> optimizer.apply_gradients(zip(grad, train_weights))

    """

    def __init__(self, learning_rate=0.001, rho=0.95, epsilon=1e-07, *args, **kwargs):
        super().__init__(learning_rate, rho, epsilon, *args, **kwargs)


class Adagrad(tf.optimizers.Adagrad):
    """Optimizer that implements the Adagrad algorithm. Equivalent to tf.optimizers.Adagrad.

    References
    ----------
    - https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers/Adagrad?hl=en

    Parameters
    ----------
    learning_rate : A Tensor, floating point value
        The learning rate. Defaults to 0.001.
    initial_accumulator_value : float
        Floating point value. Starting value for the accumulators (per-parameter momentum values).
        Must be non-negative.Defaults to 0.95.
    epsilon : float
        A small constant for numerical stability.Defaults to 1e-7.

    Examples
    --------
    With TensorLayer

    >>> import tensorlayerx as tl
    >>> optimizer = tl.optimizers.Adagrad(0.0001)
    >>> optimizer.apply_gradients(zip(grad, train_weights))

    """

    def __init__(self, learning_rate=0.001, rho=0.95, epsilon=1e-07, *args, **kwargs):
        super().__init__(learning_rate, rho, epsilon, *args, **kwargs)


class Adam(tf.optimizers.Adam):
    """Optimizer that implements the Adam algorithm. Equivalent to tf.optimizers.Adam.

    References
    ----------
    - https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers/Adam?hl=en

    Parameters
    ----------
    learning_rate : A Tensor, floating point value
        The learning rate. Defaults to 0.001.
    beta_1 : float or constant float tensor
        The exponential decay rate for the 1st moment estimates. Defaults to 0.9.
    beta_2 : float or constant float tensor
        The exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
    epsilon : float
        A small constant for numerical stability.Defaults to 1e-7.

    Examples
    --------
    With TensorLayer

    >>> import tensorlayerx as tl
    >>> optimizer = tl.optimizers.Adam(0.0001)
    >>> optimizer.apply_gradients(zip(grad, train_weights))

    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, *args, **kwargs):
        super().__init__(learning_rate, beta_1, beta_2, epsilon, *args, **kwargs)


class Adamax(tf.optimizers.Adamax):
    """Optimizer that implements the Adamax algorithm. Equivalent to tf.optimizers.Adamax.

    References
    ----------
    - https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers/Adamax?hl=en

    Parameters
    ----------
    learning_rate : A Tensor, floating point value
        The learning rate. Defaults to 0.001.
    beta_1 : float or constant float tensor
        The exponential decay rate for the 1st moment estimates. Defaults to 0.9.
    beta_2 : float or constant float tensor
        The exponential decay rate for the exponentially weighted infinity norm. Defaults to 0.999.
    epsilon : float
        A small constant for numerical stability.Defaults to 1e-7.

    Examples
    --------
    With TensorLayer

    >>> import tensorlayerx as tl
    >>> optimizer = tl.optimizers.Adamax(0.0001)
    >>> optimizer.apply_gradients(zip(grad, train_weights))

    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, *args, **kwargs):
        super().__init__(learning_rate, beta_1, beta_2, epsilon, *args, **kwargs)


class Ftrl(tf.optimizers.Ftrl):
    """Optimizer that implements the FTRL algorithm. Equivalent to tf.optimizers.Ftrl.

    References
    ----------
    - https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers/Ftrl?hl=en

    Parameters
    ----------
    learning_rate : A Tensor, floating point value
        The learning rate. Defaults to 0.001.
    learning_rate_power : float
        Controls how the learning rate decreases during training. Use zero for a fixed learning rate.
    initial_accumulator_value : float
        The starting value for accumulators. Only zero or positive values are allowed.
    l1_regularization_strength : float
        A float value, must be greater than or equal to zero. Defaults to 0.0.
    l2_regularization_strength : float
        A float value, must be greater than or equal to zero. Defaults to 0.0.
    l2_shrinkage_regularization_strength : float
        This differs from L2 above in that the L2 above is a stabilization penalty, whereas this L2 shrinkage is a magnitude penalty.
        When input is sparse shrinkage will only happen on the active weights.
    beta : float
        A float value, representing the beta value from the paper. Defaults to 0.0.

    Examples
    --------
    With TensorLayer

    >>> import tensorlayerx as tl
    >>> optimizer = tl.optimizers.Ftrl(0.0001)
    >>> optimizer.apply_gradients(zip(grad, train_weights))

    """

    def __init__(
        self, learning_rate=0.001, learning_rate_power=-0.5, initial_accumulator_value=0.1,
        l1_regularization_strength=0.0, l2_regularization_strength=0.0, beta=0.0,
        l2_shrinkage_regularization_strength=0.0, **kwargs
    ):
        super().__init__(
            learning_rate, learning_rate_power, initial_accumulator_value, l1_regularization_strength,
            l2_regularization_strength, beta, l2_shrinkage_regularization_strength, **kwargs
        )


class Nadam(tf.optimizers.Nadam):
    """Optimizer that implements the NAdam algorithm. Equivalent to tf.optimizers.Nadam.

    References
    ----------
    - https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers/Nadam?hl=en

    Parameters
    ----------
    learning_rate : A Tensor, floating point value
        The learning rate. Defaults to 0.001.
    beta_1 : float or constant float tensor
        The exponential decay rate for the 1st moment estimates. Defaults to 0.9.
    beta_2 : float or constant float tensor
         The exponential decay rate for the exponentially weighted infinity norm. Defaults to 0.999.
    epsilon : float
        A small constant for numerical stability.Defaults to 1e-7.

    Examples
    --------
    With TensorLayer

    >>> import tensorlayerx as tl
    >>> optimizer = tl.optimizers.Nadam(0.0001)
    >>> optimizer.apply_gradients(zip(grad, train_weights))

    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, *args, **kwargs):
        super().__init__(learning_rate, beta_1, beta_2, epsilon, *args, **kwargs)


class RMSprop(tf.optimizers.RMSprop):
    """Optimizer that implements the RMSprop algorithm. Equivalent to tf.optimizers.RMSprop.

    References
    ----------
    - https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers/RMSprop?hl=en

    Parameters
    ----------
    learning_rate : A Tensor, floating point value
        The learning rate. Defaults to 0.001.
    rho : float
        Discounting factor for the history/coming gradient. Defaults to 0.9.
    momentum : float
         A scalar or a scalar Tensor. Defaults to 0.0.
    epsilon : float
        A small constant for numerical stability.Defaults to 1e-7.
    centered : bool
        If True, gradients are normalized by the estimated variance of the gradient; if False, by the uncentered second moment.
        Setting this to True may help with training, but is slightly more expensive in terms of computation and memory.
        Defaults to False.

    Examples
    --------
    With TensorLayer

    >>> import tensorlayerx as tl
    >>> optimizer = tl.optimizers.RMSprop(0.0001)
    >>> optimizer.apply_gradients(zip(grad, train_weights))

    """

    def __init__(self, learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False, *args, **kwargs):
        super().__init__(learning_rate, rho, momentum, epsilon, centered, *args, **kwargs)


class SGD(tf.optimizers.SGD):
    """Gradient descent (with momentum) optimizer. Equivalent to tf.optimizers.SGD.

    References
    ----------
    - https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers/SGD?hl=en

    Parameters
    ----------
    learning_rate : A Tensor, floating point value
        The learning rate. Defaults to 0.001.
    momentum : float
        float hyperparameter >= 0 that accelerates gradient descent in the relevant direction and dampens oscillations.
        Defaults to 0, i.e., vanilla gradient descent.
    nesterov : bool
        Whether to apply Nesterov momentum. Defaults to False.

    Examples
    --------
    With TensorLayer

    >>> import tensorlayerx as tl
    >>> optimizer = tl.optimizers.SGD(0.0001)
    >>> optimizer.apply_gradients(zip(grad, train_weights))

    """

    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False, *args, **kwargs):
        super().__init__(learning_rate, momentum, nesterov, *args, **kwargs)


class Momentum(tf.compat.v1.train.MomentumOptimizer):
    """Optimizer that implements the Momentum algorithm. Equivalent to tf.compat.v1.train.MomentumOptimizer

    References
    ----------
    - https://tensorflow.google.cn/api_docs/python/tf/compat/v1/train/MomentumOptimizer?hl=en&version=nightly

    Parameters
    ----------
    learning_rate : A Tensor, floating point value
        The learning rate. Defaults to 0.001.
    momentum : float
        A Tensor or a floating point value. The momentum. Defaults to 0
    use_locking : bool
        If True use locks for update operations.
    use_nesterov : bool
        If True use Nesterov Momentum. See (Sutskever et al., 2013).
        This implementation always computes gradients at the value of the variable(s) passed to the optimizer.
        Using Nesterov Momentum makes the variable(s) track the values called theta_t + mu*v_t in the paper.
        This implementation is an approximation of the original formula, valid for high values of momentum.
        It will compute the "adjusted gradient" in NAG by assuming that the new gradient will be estimated
        by the current average gradient plus the product of momentum and the change in the average gradient.

    Examples
    --------
    With TensorLayer

    >>> import tensorlayerx as tl
    >>> optimizer = tl.optimizers.Momentum(0.0001)
    >>> optimizer.apply_gradients(zip(grad, train_weights))

    """

    def __init__(self, learning_rate, momentum=0.0, *args, **kwargs):
        super().__init__(learning_rate, momentum, *args, **kwargs)


class Lamb(object):
    """Optimizer that implements the Layer-wise Adaptive Moments (LAMB).

    References
    ----------
    - https://tensorflow.google.cn/addons/api_docs/python/tfa/optimizers/LAMB?hl=en

    """

    def __init__(self):
        raise NotImplementedError('Optimizer that not implemented the Layer-wise Adaptive Moments (LAMB).')


class LARS(object):
    """ LARS is an optimization algorithm employing a large batch optimization technique. Refer to paper LARGE BATCH TRAINING OF CONVOLUTIONAL NETWORKS.

    References
    ----------
    - https://www.mindspore.cn/docs/api/zh-CN/r1.5/api_python/nn/mindspore.nn.LARS.html?highlight=lars#mindspore.nn.LARS

    """

    def __init__(self):
        raise NotImplementedError('Optimizer that not implemented the LARS.')
