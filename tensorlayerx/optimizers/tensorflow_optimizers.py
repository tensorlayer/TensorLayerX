#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorlayerx as tlx

__all__ = ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD', 'Momentum', 'Lamb', 'LARS']


class Adadelta(object):
    """Optimizer that implements the Adadelta algorithm. Equivalent to tf.optimizers.Adadelta.

    References
    ----------
    - https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers/Adadelta?hl=en

    Parameters
    ----------
    lr : A Tensor, floating point value
        The learning rate. Defaults to 0.001.
    rho : float or constant float tensor
        A Tensor or a floating point value. The decay rate.
    eps : float
        A small constant for numerical stability.Defaults to 1e-7.
    weight_decay : float
        weight decay (L2 penalty) (default: 0.0)
    grad_clip : GradientClip or None
        Gradient cliping strategy.There are three cliping strategies
        ( `tlx.ops.ClipGradByValue` ,
        `tlx.ops.ClipGradByNorm`,
        `tlx.ops.ClipByGlobalNorm`  ).
        Default None, meaning there is no gradient clipping.

    Examples
    --------
    With TensorLayerx

    >>> import tensorlayerx as tlx
    >>> optimizer = tlx.optimizers.Adadelta(0.001)
    >>> optimizer.apply_gradients(zip(grad, train_weights))

    """

    def __init__(self, lr=0.001, rho=0.95, eps=1e-07, weight_decay=0.0, grad_clip=None):
        self.lr = lr
        self.rho = rho
        self.eps = eps
        if weight_decay < 0.0:
            raise ValueError("weight_decay should not smaller than 0.0, but got {}".format(weight_decay))
        self.weight_decay = tf.convert_to_tensor(float(weight_decay))
        self.grad_clip = grad_clip
        self.adadelta = tf.optimizers.Adadelta(learning_rate=self.lr, rho=self.rho, epsilon=self.eps)

    def apply_gradients(self, grads_and_vars):
        if grads_and_vars is None:
            raise ValueError('grads_and_vars is not set.')
        if self.weight_decay != 0.0 or self.grad_clip is not None:
            grads, vars = zip(*grads_and_vars)
            if self.weight_decay != 0.0:
                new_grads = []
                for grad, var in zip(grads, vars):
                    grad = grad + self.weight_decay * var
                    new_grads.append(grad)
                grads = new_grads
            if self.grad_clip is not None:
                if isinstance(self.grad_clip, tlx.ops.ClipByGlobalNorm):
                    new_grads, _ = self.grad_clip(grads)
                else:
                    new_grads = []
                    for g in grads:
                        new_grads.append(self.grad_clip(g))
                grads = new_grads
            grads_and_vars = zip(grads, vars)
        self.adadelta.apply_gradients(grads_and_vars)


class Adagrad(object):
    """Optimizer that implements the Adagrad algorithm. Equivalent to tf.optimizers.Adagrad.

    References
    ----------
    - https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers/Adagrad?hl=en

    Parameters
    ----------
    lr : A Tensor, floating point value
        The learning rate. Defaults to 0.001.
    initial_accumulator_value : float
        Floating point value. Starting value for the accumulators (per-parameter momentum values).
        Must be non-negative.Defaults to 0.95.
    eps : float
        A small constant for numerical stability.Defaults to 1e-7.
    weight_decay : float
        weight decay (L2 penalty) (default: 0.0)
    grad_clip : GradientClip or None
        Gradient cliping strategy.There are three cliping strategies
        ( `tlx.ops.ClipGradByValue` ,
        `tlx.ops.ClipGradByNorm`,
        `tlx.ops.ClipByGlobalNorm`  ).
        Default None, meaning there is no gradient clipping.

    Examples
    --------
    With TensorLayerx

    >>> import tensorlayerx as tlx
    >>> optimizer = tlx.optimizers.Adagrad(0.001)
    >>> optimizer.apply_gradients(zip(grad, train_weights))

    """

    def __init__(self, lr=0.001, initial_accumulator=0.1, eps=1e-07, weight_decay=0.0, grad_clip=None):
        self.lr = lr
        self.initial_accumulator = initial_accumulator
        self.eps = eps
        if weight_decay < 0.0:
            raise ValueError("weight_decay should not smaller than 0.0, but got {}".format(weight_decay))
        self.weight_decay = tf.convert_to_tensor(float(weight_decay))
        self.grad_clip = grad_clip
        self.adagrad = tf.optimizers.Adagrad(
            learning_rate=self.lr, initial_accumulator_value=self.initial_accumulator, epsilon=self.eps
        )

    def apply_gradients(self, grads_and_vars):
        if grads_and_vars is None:
            raise ValueError('grads_and_vars is not set.')
        if self.weight_decay != 0.0 or self.grad_clip is not None:
            grads, vars = zip(*grads_and_vars)
            if self.weight_decay != 0.0:
                new_grads = []
                for grad, var in zip(grads, vars):
                    grad = grad + self.weight_decay * var
                    new_grads.append(grad)
                grads = new_grads
            if self.grad_clip is not None:
                if isinstance(self.grad_clip, tlx.ops.ClipByGlobalNorm):
                    new_grads, _ = self.grad_clip(grads)
                else:
                    new_grads = []
                    for g in grads:
                        new_grads.append(self.grad_clip(g))
                grads = new_grads
            grads_and_vars = zip(grads, vars)
        self.adagrad.apply_gradients(grads_and_vars)


class Adam(object):
    """Optimizer that implements the Adam algorithm. Equivalent to tf.optimizers.Adam.

    References
    ----------
    - https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers/Adam?hl=en

    Parameters
    ----------
    lr : A Tensor, floating point value
        The learning rate. Defaults to 0.001.
    beta_1 : float or constant float tensor
        The exponential decay rate for the 1st moment estimates. Defaults to 0.9.
    beta_2 : float or constant float tensor
        The exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
    eps : float
        A small constant for numerical stability.Defaults to 1e-7.
    weight_decay : float
        weight decay (L2 penalty) (default: 0.0)
    grad_clip : GradientClip or None
        Gradient cliping strategy.There are three cliping strategies
        ( `tlx.ops.ClipGradByValue` ,
        `tlx.ops.ClipGradByNorm`,
        `tlx.ops.ClipByGlobalNorm`  ).
        Default None, meaning there is no gradient clipping.

    Examples
    --------
    With TensorLayerx

    >>> import tensorlayerx as tlx
    >>> optimizer = tlx.optimizers.Adam(0.001)
    >>> optimizer.apply_gradients(zip(grad, train_weights))

    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, eps=1e-07, weight_decay=0.0, grad_clip=None):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        if weight_decay < 0.0:
            raise ValueError("weight_decay should not smaller than 0.0, but got {}".format(weight_decay))
        self.weight_decay = tf.convert_to_tensor(float(weight_decay))
        self.grad_clip = grad_clip
        self.adam = tf.optimizers.Adam(
            learning_rate=self.lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.eps
        )

    def apply_gradients(self, grads_and_vars):
        if grads_and_vars is None:
            raise ValueError('grads_and_vars is not set.')
        if self.weight_decay != 0.0 or self.grad_clip is not None:
            grads, vars = zip(*grads_and_vars)
            if self.weight_decay != 0.0:
                new_grads = []
                for grad, var in zip(grads, vars):
                    grad = grad + self.weight_decay * var
                    new_grads.append(grad)
                grads = new_grads
            if self.grad_clip is not None:
                if isinstance(self.grad_clip, tlx.ops.ClipByGlobalNorm):
                    new_grads, _ = self.grad_clip(grads)
                else:
                    new_grads = []
                    for g in grads:
                        new_grads.append(self.grad_clip(g))
                grads = new_grads
            grads_and_vars = zip(grads, vars)
        self.adam.apply_gradients(grads_and_vars)


class Adamax(object):
    """Optimizer that implements the Adamax algorithm. Equivalent to tf.optimizers.Adamax.

    References
    ----------
    - https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers/Adamax?hl=en

    Parameters
    ----------
    lr : A Tensor, floating point value
        The learning rate. Defaults to 0.001.
    beta_1 : float or constant float tensor
        The exponential decay rate for the 1st moment estimates. Defaults to 0.9.
    beta_2 : float or constant float tensor
        The exponential decay rate for the exponentially weighted infinity norm. Defaults to 0.999.
    eps : float
        A small constant for numerical stability.Defaults to 1e-7.
    weight_decay : float
        weight decay (L2 penalty) (default: 0.0)
    grad_clip : GradientClip or None
        Gradient cliping strategy.There are three cliping strategies
        ( `tlx.ops.ClipGradByValue` ,
        `tlx.ops.ClipGradByNorm`,
        `tlx.ops.ClipByGlobalNorm`  ).
        Default None, meaning there is no gradient clipping.

    Examples
    --------
    With TensorLayerx

    >>> import tensorlayerx as tlx
    >>> optimizer = tlx.optimizers.Adamax(0.001)
    >>> optimizer.apply_gradients(zip(grad, train_weights))

    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, eps=1e-07, weight_decay=0.0, grad_clip=None):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        if weight_decay < 0.0:
            raise ValueError("weight_decay should not smaller than 0.0, but got {}".format(weight_decay))
        self.weight_decay = tf.convert_to_tensor(float(weight_decay))
        self.grad_clip = grad_clip
        self.adamax = tf.optimizers.Adamax(
            learning_rate=self.lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.eps
        )

    def apply_gradients(self, grads_and_vars):
        if grads_and_vars is None:
            raise ValueError('grads_and_vars is not set.')
        if self.weight_decay != 0.0 or self.grad_clip is not None:
            grads, vars = zip(*grads_and_vars)
            if self.weight_decay != 0.0:
                new_grads = []
                for grad, var in zip(grads, vars):
                    grad = grad + self.weight_decay * var
                    new_grads.append(grad)
                grads = new_grads
            if self.grad_clip is not None:
                if isinstance(self.grad_clip, tlx.ops.ClipByGlobalNorm):
                    new_grads, _ = self.grad_clip(grads)
                else:
                    new_grads = []
                    for g in grads:
                        new_grads.append(self.grad_clip(g))
                grads = new_grads
            grads_and_vars = zip(grads, vars)
        self.adamax.apply_gradients(grads_and_vars)


class Ftrl(object):
    """Optimizer that implements the FTRL algorithm. Equivalent to tf.optimizers.Ftrl.

    References
    ----------
    - https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers/Ftrl?hl=en

    Parameters
    ----------
    lr : A Tensor, floating point value
        The learning rate. Defaults to 0.001.
    lr_power : float
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
    weight_decay : float
        weight decay (L2 penalty) (default: 0.0)
    grad_clip : GradientClip or None
        Gradient cliping strategy.There are three cliping strategies
        ( `tlx.ops.ClipGradByValue` ,
        `tlx.ops.ClipGradByNorm`,
        `tlx.ops.ClipByGlobalNorm`  ).
        Default None, meaning there is no gradient clipping.

    Examples
    --------
    With TensorLayerx

    >>> import tensorlayerx as tlx
    >>> optimizer = tlx.optimizers.Ftrl(0.001)
    >>> optimizer.apply_gradients(zip(grad, train_weights))

    """

    def __init__(
        self, lr=0.001, lr_power=-0.5, initial_accumulator_value=0.1,
        l1_regularization_strength=0.0, l2_regularization_strength=0.0, beta=0.0,
        l2_shrinkage_regularization_strength=0.0, weight_decay=0.0, grad_clip=None
    ):
        self.lr = lr
        self.lr_power = lr_power
        self.initial_accumulator_value = initial_accumulator_value
        self.l1_regularization_strength = l1_regularization_strength
        self.l2_regularization_strength = l2_regularization_strength
        self.beta = beta
        self.l2_shrinkage_regularization_strength = l2_shrinkage_regularization_strength
        if weight_decay < 0.0:
            raise ValueError("weight_decay should not smaller than 0.0, but got {}".format(weight_decay))
        self.weight_decay = tf.convert_to_tensor(float(weight_decay))
        self.grad_clip = grad_clip
        self.ftrl = tf.optimizers.Ftrl(
            learning_rate=self.lr, learning_rate_power=self.lr_power,
            initial_accumulator_value=self.initial_accumulator_value,
            l1_regularization_strength=self.l1_regularization_strength,
            l2_regularization_strength=self.l2_regularization_strength, beta=self.beta,
            l2_shrinkage_regularization_strength=self.l2_shrinkage_regularization_strength
        )

    def apply_gradients(self, grads_and_vars):
        if grads_and_vars is None:
            raise ValueError('grads_and_vars is not set.')
        if self.weight_decay != 0.0 or self.grad_clip is not None:
            grads, vars = zip(*grads_and_vars)
            if self.weight_decay != 0.0:
                new_grads = []
                for grad, var in zip(grads, vars):
                    grad = grad + self.weight_decay * var
                    new_grads.append(grad)
                grads = new_grads
            if self.grad_clip is not None:
                if isinstance(self.grad_clip, tlx.ops.ClipByGlobalNorm):
                    new_grads, _ = self.grad_clip(grads)
                else:
                    new_grads = []
                    for g in grads:
                        new_grads.append(self.grad_clip(g))
                grads = new_grads
            grads_and_vars = zip(grads, vars)
        self.ftrl.apply_gradients(grads_and_vars)


class Nadam(object):
    """Optimizer that implements the NAdam algorithm. Equivalent to tf.optimizers.Nadam.

    References
    ----------
    - https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers/Nadam?hl=en

    Parameters
    ----------
    lr : A Tensor, floating point value
        The learning rate. Defaults to 0.001.
    beta_1 : float or constant float tensor
        The exponential decay rate for the 1st moment estimates. Defaults to 0.9.
    beta_2 : float or constant float tensor
         The exponential decay rate for the exponentially weighted infinity norm. Defaults to 0.999.
    eps : float
        A small constant for numerical stability.Defaults to 1e-7.
    weight_decay : float
        weight decay (L2 penalty) (default: 0.0)
    grad_clip : GradientClip or None
        Gradient cliping strategy.There are three cliping strategies
        ( `tlx.ops.ClipGradByValue` ,
        `tlx.ops.ClipGradByNorm`,
        `tlx.ops.ClipByGlobalNorm`  ).
        Default None, meaning there is no gradient clipping.

    Examples
    --------
    With TensorLayerx

    >>> import tensorlayerx as tlx
    >>> optimizer = tlx.optimizers.Nadam(0.001)
    >>> optimizer.apply_gradients(zip(grad, train_weights))

    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, eps=1e-07, weight_decay=0.0, grad_clip=None):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        if weight_decay < 0.0:
            raise ValueError("weight_decay should not smaller than 0.0, but got {}".format(weight_decay))
        self.weight_decay = tf.convert_to_tensor(float(weight_decay))
        self.grad_clip = grad_clip
        self.nadam = tf.optimizers.Nadam(
            learning_rate=self.lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.eps
        )

    def apply_gradients(self, grads_and_vars):
        if grads_and_vars is None:
            raise ValueError('grads_and_vars is not set.')
        if self.weight_decay != 0.0 or self.grad_clip is not None:
            grads, vars = zip(*grads_and_vars)
            if self.weight_decay != 0.0:
                new_grads = []
                for grad, var in zip(grads, vars):
                    grad = grad + self.weight_decay * var
                    new_grads.append(grad)
                grads = new_grads
            if self.grad_clip is not None:
                if isinstance(self.grad_clip, tlx.ops.ClipByGlobalNorm):
                    new_grads, _ = self.grad_clip(grads)
                else:
                    new_grads = []
                    for g in grads:
                        new_grads.append(self.grad_clip(g))
                grads = new_grads
            grads_and_vars = zip(grads, vars)
        self.nadam.apply_gradients(grads_and_vars)


class RMSprop(object):
    """Optimizer that implements the RMSprop algorithm. Equivalent to tf.optimizers.RMSprop.

    References
    ----------
    - https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers/RMSprop?hl=en

    Parameters
    ----------
    lr : A Tensor, floating point value
        The learning rate. Defaults to 0.001.
    rho : float
        Discounting factor for the history/coming gradient. Defaults to 0.9.
    momentum : float
         A scalar or a scalar Tensor. Defaults to 0.0.
    eps : float
        A small constant for numerical stability.Defaults to 1e-7.
    centered : bool
        If True, gradients are normalized by the estimated variance of the gradient; if False, by the uncentered second moment.
        Setting this to True may help with training, but is slightly more expensive in terms of computation and memory.
        Defaults to False.
    weight_decay : float
        weight decay (L2 penalty) (default: 0.0)
    grad_clip : GradientClip or None
        Gradient cliping strategy.There are three cliping strategies
        ( `tlx.ops.ClipGradByValue` ,
        `tlx.ops.ClipGradByNorm`,
        `tlx.ops.ClipByGlobalNorm`  ).
        Default None, meaning there is no gradient clipping.

    Examples
    --------
    With TensorLayerx

    >>> import tensorlayerx as tlx
    >>> optimizer = tlx.optimizers.RMSprop(0.001)
    >>> optimizer.apply_gradients(zip(grad, train_weights))

    """

    def __init__(
        self, lr=0.001, rho=0.9, momentum=0.0, eps=1e-07, centered=False, weight_decay=0.0,
        grad_clip=None
    ):
        self.lr = lr
        self.rho = rho
        self.momentum = momentum
        self.eps = eps
        self.centered = centered
        if weight_decay < 0.0:
            raise ValueError("weight_decay should not smaller than 0.0, but got {}".format(weight_decay))
        self.weight_decay = tf.convert_to_tensor(float(weight_decay))
        self.grad_clip = grad_clip
        self.rmsprop = tf.optimizers.RMSprop(
            learning_rate=self.lr, rho=self.rho, momentum=self.momentum, epsilon=self.eps,
            centered=self.centered
        )

    def apply_gradients(self, grads_and_vars):
        if grads_and_vars is None:
            raise ValueError('grads_and_vars is not set.')
        if self.weight_decay != 0.0 or self.grad_clip is not None:
            grads, vars = zip(*grads_and_vars)
            if self.weight_decay != 0.0:
                new_grads = []
                for grad, var in zip(grads, vars):
                    grad = grad + self.weight_decay * var
                    new_grads.append(grad)
                grads = new_grads
            if self.grad_clip is not None:
                if isinstance(self.grad_clip, tlx.ops.ClipByGlobalNorm):
                    new_grads, _ = self.grad_clip(grads)
                else:
                    new_grads = []
                    for g in grads:
                        new_grads.append(self.grad_clip(g))
                grads = new_grads
            grads_and_vars = zip(grads, vars)
        self.rmsprop.apply_gradients(grads_and_vars)


class SGD(object):
    """Gradient descent (with momentum) optimizer. Equivalent to tf.optimizers.SGD.

    References
    ----------
    - https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers/SGD?hl=en

    Parameters
    ----------
    lr : A Tensor, floating point value
        The learning rate. Defaults to 0.001.
    momentum : float
        float hyperparameter >= 0 that accelerates gradient descent in the relevant direction and dampens oscillations.
        Defaults to 0, i.e., vanilla gradient descent.
    weight_decay : float
        weight decay (L2 penalty) (default: 0.0)
    grad_clip : GradientClip or None
        Gradient cliping strategy.There are three cliping strategies
        ( `tlx.ops.ClipGradByValue` ,
        `tlx.ops.ClipGradByNorm`,
        `tlx.ops.ClipByGlobalNorm`  ).
        Default None, meaning there is no gradient clipping.

    Examples
    --------
    With TensorLayerx

    >>> import tensorlayerx as tlx
    >>> optimizer = tlx.optimizers.SGD(0.01)
    >>> optimizer.apply_gradients(zip(grad, train_weights))

    """

    def __init__(self, lr=0.01, momentum=0.0, weight_decay=0.0, grad_clip=None):
        self.lr = lr
        self.momentum = momentum
        if weight_decay < 0.0:
            raise ValueError("weight_decay should not smaller than 0.0, but got {}".format(weight_decay))
        self.weight_decay = tf.convert_to_tensor(float(weight_decay))
        self.grad_clip = grad_clip
        self.sgd = tf.optimizers.SGD(learning_rate=self.lr, momentum=self.momentum, nesterov=False)

    def apply_gradients(self, grads_and_vars):
        if grads_and_vars is None:
            raise ValueError('grads_and_vars is not set.')
        if self.weight_decay != 0.0 or self.grad_clip is not None:
            grads, vars = zip(*grads_and_vars)
            if self.weight_decay != 0.0:
                new_grads = []
                for grad, var in zip(grads, vars):
                    grad = grad + self.weight_decay * var
                    new_grads.append(grad)
                grads = new_grads
            if self.grad_clip is not None:
                if isinstance(self.grad_clip, tlx.ops.ClipByGlobalNorm):
                    new_grads, _ = self.grad_clip(grads)
                else:
                    new_grads = []
                    for g in grads:
                        new_grads.append(self.grad_clip(g))
                grads = new_grads
            grads_and_vars = zip(grads, vars)
        self.sgd.apply_gradients(grads_and_vars)


class Momentum(object):
    """Optimizer that implements the Momentum algorithm. Equivalent to tf.compat.v1.train.MomentumOptimizer

    References
    ----------
    - https://tensorflow.google.cn/api_docs/python/tf/compat/v1/train/MomentumOptimizer?hl=en&version=nightly

    Parameters
    ----------
    lr : A Tensor, floating point value
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
    weight_decay : float
        weight decay (L2 penalty) (default: 0.0)
    grad_clip : GradientClip or None
        Gradient cliping strategy.There are three cliping strategies
        ( `tlx.ops.ClipGradByValue` ,
        `tlx.ops.ClipGradByNorm`,
        `tlx.ops.ClipByGlobalNorm`  ).
        Default None, meaning there is no gradient clipping.

    Examples
    --------
    With TensorLayerx

    >>> import tensorlayerx as tlx
    >>> optimizer = tlx.optimizers.Momentum(0.01, momentum=0.9)
    >>> optimizer.apply_gradients(zip(grad, train_weights))

    """

    def __init__(self, lr=0.01, momentum=0.0, nesterov=False, weight_decay=0.0, grad_clip=None):
        self.lr = lr
        self.momentum = momentum
        if weight_decay < 0.0:
            raise ValueError("weight_decay should not smaller than 0.0, but got {}".format(weight_decay))
        self.weight_decay = tf.convert_to_tensor(float(weight_decay))
        self.grad_clip = grad_clip
        self.nesterov = nesterov
        self.sgd = tf.optimizers.SGD(learning_rate=self.lr, momentum=self.momentum, nesterov=self.nesterov)

    def apply_gradients(self, grads_and_vars):
        if grads_and_vars is None:
            raise ValueError('grads_and_vars is not set.')
        if self.weight_decay != 0.0 or self.grad_clip is not None:
            grads, vars = zip(*grads_and_vars)
            if self.weight_decay != 0.0:
                new_grads = []
                for grad, var in zip(grads, vars):
                    grad = grad + self.weight_decay * var
                    new_grads.append(grad)
                grads = new_grads
            if self.grad_clip is not None:
                if isinstance(self.grad_clip, tlx.ops.ClipByGlobalNorm):
                    new_grads, _ = self.grad_clip(grads)
                else:
                    new_grads = []
                    for g in grads:
                        new_grads.append(self.grad_clip(g))
                grads = new_grads
            grads_and_vars = zip(grads, vars)
        self.sgd.apply_gradients(grads_and_vars)


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
