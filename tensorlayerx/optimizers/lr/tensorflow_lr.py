#! /usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import math
import numpy as np
# reference to PaddlePaddle paddle.optimizer.lr
__all__ = [
    'LRScheduler', 'NoamDecay', 'PiecewiseDecay', 'NaturalExpDecay', 'InverseTimeDecay', 'PolynomialDecay',
    'LinearWarmup', 'ExponentialDecay', 'MultiStepDecay', 'StepDecay', 'LambdaDecay', 'ReduceOnPlateau',
    'CosineAnnealingDecay'
]


class LRScheduler(object):
    """
    LRScheduler Base class. Define the common interface of a learning rate scheduler.

    User can import it by ``from tl.optimizer.lr import LRScheduler`` ,

    then overload it for your subclass and have a custom implementation of ``get_lr()`` .

    References
    ----------
    - https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/LRScheduler_cn.html

    Parameters
    ----------
    learning_rate : A floating point value
        The learning rate. Defaults to 0.1.
    last_epoch : int
        The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    verbose : bool
        If ``True``, prints a message to stdout for each update. Default: ``False`` .


    Examples
    --------
    With TensorLayer

    >>> #Here is an example of a simple ``StepDecay`` implementation.
    >>> import tensorlayerx as tlx
    >>> from tensorlayerx.optimizers.lr import LRScheduler
    >>> class StepDecay(LRScheduler):
    >>>     def __init__(self, learning_rate, step_size, gamma = 0.1, last_epoch = -1, verbose=False):
    >>>         if not isinstance(step_size, int):
    >>>             raise TypeError("The type of 'step_size' must be 'int', but received %s." %type(step_size))
    >>>         if gamma >= 1.0 :
    >>>             raise ValueError('gamma should be < 1.0.')
    >>>         self.step_size = step_size
    >>>         self.gamma = gamma
    >>>         super(StepDecay, self).__init__(learning_rate, last_epoch, verbose)
    >>>     def get_lr(self):
    >>>         i = self.last_epoch // self.step_size
    >>>         return self.base_lr * (self.gamma**i)

    """

    def __init__(self, learning_rate=0.1, last_epoch=-1, verbose=False):
        if not isinstance(learning_rate, (float, int)):
            raise TypeError("The type of learning rate must be float, but received {}".format(type(learning_rate)))
        self.base_lr = tf.Variable(initial_value=float(learning_rate))
        self.last_lr = tf.Variable(initial_value=float(learning_rate))
        self.last_epoch = last_epoch
        self.verbose = verbose

        self.step()

    def __call__(self):

        return self.last_lr

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
            new_lr = self.get_lr()
        else:
            self.last_epoch = epoch
            if hasattr(self, "_get_closed_form_lr"):
                new_lr = self._get_closed_form_lr()
            else:
                new_lr = self.get_lr()
        self.last_lr.assign(new_lr)
        if self.verbose:
            print(
                'Epoch {}: {} set learning rate to {}.'.format(self.last_epoch, self.__class__.__name__, self.last_lr)
            )

    def get_lr(self):

        raise NotImplementedError


class StepDecay(LRScheduler):
    """Update the learning rate of ``optimizer`` by ``gamma`` every ``step_size`` number of epoch.

    .. math::
        new\_learning\_rate = learning\_rate * gamma^{epoch // step_size}

    References
    ----------
    - https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/StepDecay_cn.html

    Parameters
    ----------
    learning_rate : float
        The learning rate.
    step_size : int
        the interval to update.
    gamma : float
        The Ratio that the learning rate will be reduced. ``new_lr = origin_lr * gamma`` .
        It should be less than 1.0. Default: 0.1.
    last_epoch : int
        The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    verbose : bool
        If ``True``, prints a message to stdout for each update. Default: ``False`` .


    Examples
    --------
    With TensorLayer

    >>> import tensorlayerx as tlx
    >>> scheduler = tlx.optimizers.lr.StepDecay(learning_rate = 0.1, step_size = 10,  gamma = 0.1, last_epoch = -1, verbose = False)
    >>> sgd = tlx.optimizers.SGD(learning_rate=scheduler,momentum=0.2)
    >>> for epoch in range(100):
    >>>     for batch in range(100):
    >>>        # train model
    >>>         scheduler.step() # If you update learning rate each batch
    >>>     #scheduler.step()    # If you update learning rate each epoch

    """

    def __init__(self, learning_rate, step_size, gamma=0.1, last_epoch=-1, verbose=False):
        if not isinstance(step_size, int):
            raise TypeError("The type of 'step_size' must be 'int', but received %s." % type(step_size))
        if gamma >= 1.0:
            raise ValueError('gamma should be < 1.0.')
        self.step_size = step_size
        self.gamma = gamma
        super(StepDecay, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        i = self.last_epoch // self.step_size
        return self.base_lr * (self.gamma**i)


class CosineAnnealingDecay(LRScheduler):
    """Set the learning rate using a cosine annealing schedule, where :math:`\eta_{max}` is set to
    the initial learning_rate. :math:`T_{cur}` is the number of epochs since the last restart in
    SGDR.

    .. math::
        \\begin{aligned}
            \eta_t & = \eta_{min} + \\frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\\frac{T_{cur}}{T_{max}}\pi\\right)\\right),
            & T_{cur} \\neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \\frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\\frac{1}{T_{max}}\pi\\right)\\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    References
    ----------
    - https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/CosineAnnealingDecay_cn.html

    Parameters
    ----------
    learning_rate : float or int
        The initial learning rate, that is :math:`\eta_{max}` . It can be set to python float or int number.
    T_max : int
        Maximum number of iterations. It is half of the decay cycle of learning rate.
    eta_min : float or int
        Minimum learning rate, that is :math:`\eta_{min}` . Default: 0.
    last_epoch : int
        The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    verbose : bool
        If ``True``, prints a message to stdout for each update. Default: ``False`` .


    Examples
    --------
    With TensorLayer

    >>> import tensorlayerx as tlx
    >>> scheduler = tlx.optimizers.lr.CosineAnnealingDecay(learning_rate = 0.1, step = 10,  gamma = 0.1, last_epoch = -1, verbose = False)
    >>> sgd = tlx.optimizers.SGD(learning_rate=scheduler,momentum=0.2)
    >>> for epoch in range(100):
    >>>     for step in range(100):
    >>>        # train model
    >>>         scheduler.step() # If you update learning rate each step
    >>>     #scheduler.step()    # If you update learning rate each epoch

    """

    def __init__(self, learning_rate, T_max, eta_min=0, last_epoch=-1, verbose=False):
        if not isinstance(T_max, int):
            raise TypeError(
                "The type of 'T_max' in 'CosineAnnealingDecay' must be 'int', but received %s." % type(T_max)
            )
        if not isinstance(eta_min, (float, int)):
            raise TypeError(
                "The type of 'eta_min' in 'CosineAnnealingDecay' must be 'float, int', but received %s." %
                type(eta_min)
            )
        self.T_max = T_max
        self.eta_min = float(eta_min)
        super(CosineAnnealingDecay, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lr
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return self.last_lr + (self.base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2

        return (1 + math.cos(math.pi * self.last_epoch / self.T_max)
               ) / (1 + math.cos(math.pi *
                                 (self.last_epoch - 1) / self.T_max)) * (self.last_lr - self.eta_min) + self.eta_min

    def _get_closed_form_lr(self):
        return self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2


class NoamDecay(LRScheduler):
    """Applies Noam Decay to the initial learning rate.

    .. math::
        new\_learning\_rate = learning\_rate * d_{model}^{-0.5} * min(epoch^{-0.5}, epoch * warmup\_steps^{-1.5})

    References
    ----------
    - https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/NoamDecay_cn.html
    - 'Attention is all you need'<https://arxiv.org/pdf/1706.03762.pdf>_

    Parameters
    ----------
    d_model : int
        The dimensionality of input and output feature vector of model. It is a python int number.
    warmup_steps : int
        The number of warmup steps. A super parameter. It is a python int number
    learning_rate : float
        The initial learning rate. It is a python float number. Default: 1.0.
    last_epoch : int
        The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    verbose : bool
        If ``True``, prints a message to stdout for each update. Default: ``False`` .


    Examples
    --------
    With TensorLayer

    >>> import tensorlayerx as tlx
    >>> scheduler = tlx.optimizers.lr.NoamDecay(d_model=0.01, warmup_steps=100, verbose=True)
    >>> sgd = tlx.optimizers.SGD(learning_rate=scheduler,momentum=0.2)
    >>> for epoch in range(100):
    >>>     for step in range(100):
    >>>        # train model
    >>>         scheduler.step() # If you update learning rate each step
    >>>     #scheduler.step()    # If you update learning rate each epoch

    """

    def __init__(self, d_model, warmup_steps, learning_rate=1.0, last_epoch=-1, verbose=False):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(NoamDecay, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0:
            a = 1
        else:
            a = self.last_epoch**-0.5
        b = self.warmup_steps**-1.5 * self.last_epoch
        return self.base_lr * (self.d_model**-0.5) * min(a, b)


class PiecewiseDecay(LRScheduler):
    """Piecewise learning rate scheduler.

    .. code-block:: text

        boundaries = [100, 200]
        values = [1.0, 0.5, 0.1]
        if epoch < 100:
            learning_rate = 1.0
        elif 100 <= global_step < 200:
            learning_rate = 0.5
        else:
            learning_rate = 0.1

    References
    ----------
    - https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/PiecewiseDecay_cn.html


    Parameters
    ----------
    boundaries : list
        A list of steps numbers.
    values : list
        A list of learning rate values that will be picked during different epoch boundaries.
    last_epoch : int
        The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    verbose : bool
        If ``True``, prints a message to stdout for each update. Default: ``False`` .


    Examples
    --------
    With TensorLayer

    >>> import tensorlayerx as tlx
    >>> scheduler = tlx.optimizers.lr.PiecewiseDecay(boundaries=[100, 200], values=[0.1, 0.5, 0.1], verbose=True)
    >>> sgd = tlx.optimizers.SGD(learning_rate=scheduler,momentum=0.2)
    >>> for epoch in range(100):
    >>>     for step in range(100):
    >>>        # train model
    >>>         scheduler.step() # If you update learning rate each step
    >>>     #scheduler.step()    # If you update learning rate each epoch

    """

    def __init__(self, boundaries, values, last_epoch=-1, verbose=False):
        self.boundaries = boundaries
        self.values = values
        super(PiecewiseDecay, self).__init__(last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        for i in range(len(self.boundaries)):
            if self.last_epoch < self.boundaries[i]:
                return self.values[i]
        return self.values[len(self.values) - 1]


class NaturalExpDecay(LRScheduler):
    """Applies natural exponential decay to the initial learning rate.

    .. math::

        new\_learning\_rate = learning\_rate * e^{- gamma * epoch}

    References
    ----------
    - https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/NaturalExpDecay_cn.html


    Parameters
    ----------
    learning_rate : float
        The initial learning rate.
    gamma : float
        A Ratio to update the learning rate. Default: 0.1.
    last_epoch : int
        The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    verbose : bool
        If ``True``, prints a message to stdout for each update. Default: ``False`` .


    Examples
    --------
    With TensorLayer

    >>> import tensorlayerx as tlx
    >>> scheduler = tlx.optimizers.lr.NaturalExpDecay(learning_rate=0.1, gamma=0.1, verbose=True)
    >>> sgd = tlx.optimizers.SGD(learning_rate=scheduler,momentum=0.2)
    >>> for epoch in range(100):
    >>>     for step in range(100):
    >>>        # train model
    >>>         scheduler.step() # If you update learning rate each step
    >>>     #scheduler.step()    # If you update learning rate each epoch

    """

    def __init__(self, learning_rate, gamma, last_epoch=-1, verbose=False):
        self.gamma = gamma
        super(NaturalExpDecay, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        return self.base_lr * math.exp(-1 * self.gamma * self.last_epoch)


class InverseTimeDecay(LRScheduler):
    """Applies inverse time decay to the initial learning rate.

    .. math::

        new\_learning\_rate = \\frac{learning\_rate}{1 + gamma * epoch}

    References
    ----------
    - https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/InverseTimeDecay_cn.html


    Parameters
    ----------
    learning_rate : float
        The initial learning rate.
    gamma : float
        A Ratio to update the learning rate. Default: 0.1.
    last_epoch : int
        The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    verbose : bool
        If ``True``, prints a message to stdout for each update. Default: ``False`` .


    Examples
    --------
    With TensorLayer

    >>> import tensorlayerx as tlx
    >>> scheduler = tlx.optimizers.lr.InverseTimeDecay(learning_rate=0.1, gamma=0.1, verbose=True)
    >>> sgd = tlx.optimizers.SGD(learning_rate=scheduler,momentum=0.2)
    >>> for epoch in range(100):
    >>>     for step in range(100):
    >>>        # train model
    >>>         scheduler.step() # If you update learning rate each step
    >>>     #scheduler.step()    # If you update learning rate each epoch

    """

    def __init__(self, learning_rate, gamma, last_epoch=-1, verbose=False):
        self.gamma = gamma
        super(InverseTimeDecay, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        return self.base_lr / (1 + self.gamma * self.last_epoch)


class PolynomialDecay(LRScheduler):
    """Applies polynomial decay to the initial learning rate.

    If cycle is set to True, then:

    .. math::

        decay\_steps & = decay\_steps * math.ceil(\\frac{epoch}{decay\_steps})

        new\_learning\_rate & = (learning\_rate-end\_lr)*(1-\\frac{epoch}{decay\_steps})^{power}+end\_lr

    If cycle is set to False, then:

    .. math::

        epoch & = min(epoch, decay\_steps)

        new\_learning\_rate & = (learning\_rate-end\_lr)*(1-\\frac{epoch}{decay\_steps})^{power}+end\_lr

    References
    ----------
    - https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/PolynomialDecay_cn.html


    Parameters
    ----------
    learning_rate : float
        The initial learning rate.
    decay_steps : int
        The decay step size. It determines the decay cycle.
    end_lr : float
        The minimum final learning rate. Default: 0.0001.
    power : float
        Power of polynomial. Default: 1.0.
    cycle : bool
        Whether the learning rate rises again. If True, then the learning rate will rise when it decrease to ``end_lr`` .  If False, the learning rate is monotone decreasing. Default: False.
    last_epoch : int
        The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    verbose : bool
        If ``True``, prints a message to stdout for each update. Default: ``False`` .


    Examples
    --------
    With TensorLayer

    >>> import tensorlayerx as tlx
    >>> scheduler = tlx.optimizers.lr.PolynomialDecay(learning_rate=0.1, decay_steps=50, verbose=True)
    >>> sgd = tlx.optimizers.SGD(learning_rate=scheduler,momentum=0.2)
    >>> for epoch in range(100):
    >>>     for step in range(100):
    >>>        # train model
    >>>         scheduler.step() # If you update learning rate each step
    >>>     #scheduler.step()    # If you update learning rate each epoch

    """

    def __init__(self, learning_rate, decay_steps, end_lr=0.0001, power=1.0, cycle=False, last_epoch=-1, verbose=False):
        self.decay_steps = decay_steps
        self.end_lr = end_lr
        self.power = power
        self.cycle = cycle
        super(PolynomialDecay, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        tmp_epoch_num = self.last_epoch
        tmp_decay_steps = self.decay_steps
        if self.cycle:
            div_res = math.ceil(float(self.last_epoch) / float(self.decay_steps))

            if self.last_epoch == 0:
                div_res = 1
            tmp_decay_steps = self.decay_steps * div_res
        else:
            tmp_epoch_num = min(self.last_epoch, self.decay_steps)

        return (self.base_lr -
                self.end_lr) * ((1 - float(tmp_epoch_num) / float(tmp_decay_steps))**self.power) + self.end_lr


class LinearWarmup(LRScheduler):
    """Linear learning rate warm up strategy. Update the learning rate preliminarily before the normal learning rate scheduler.

    When epoch < warmup_steps, learning rate is updated as:

    .. math::

            lr = start\_lr + (end\_lr - start\_lr) * \\frac{epoch}{warmup\_steps}

    where start_lr is the initial learning rate, and end_lr is the final learning rate;

    When epoch >= warmup_steps, learning rate is updated as:

    .. math::

            lr = learning_rate

    where ``learning_rate`` is float or any subclass of ``LRScheduler`` .

    References
    ----------
    - https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/LinearWarmup_cn.html
    - `Bag of Tricks for Image Classification with Convolutional Neural Networks <https://arxiv.org/abs/1812.01187>`_

    Parameters
    ----------
    learning_rate : float
        The initial learning rate.
    warmup_steps : int
        total steps of warm up.
    start_lr : float
        Initial learning rate of warm up.
    end_lr : float
        Final learning rate of warm up.
    last_epoch : int
        The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    verbose : bool
        If ``True``, prints a message to stdout for each update. Default: ``False`` .


    Examples
    --------
    With TensorLayer

    >>> import tensorlayerx as tlx
    >>> scheduler = tlx.optimizers.lr.LinearWarmup(learning_rate=0.1, warmup_steps=20, start_lr=0.0, end_lr=0.5, verbose=True)
    >>> sgd = tlx.optimizers.SGD(learning_rate=scheduler,momentum=0.2)
    >>> for epoch in range(100):
    >>>     for step in range(100):
    >>>        # train model
    >>>         scheduler.step() # If you update learning rate each step
    >>>     #scheduler.step()    # If you update learning rate each epoch

    """

    def __init__(self, learning_rate, warmup_steps, start_lr, end_lr, last_epoch=-1, verbose=False):
        type_check = isinstance(learning_rate, float) or isinstance(learning_rate,
                                                                    int) or isinstance(learning_rate, LRScheduler)
        if not type_check:
            raise TypeError(
                "the type of learning_rate should be [int, float or LRScheduler], the current type is {}".
                format(learning_rate)
            )
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.end_lr = end_lr
        assert end_lr > start_lr, "end_lr {} must be greater than start_lr {}".format(end_lr, start_lr)
        super(LinearWarmup, self).__init__(start_lr, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return (self.end_lr - self.start_lr) * float(self.last_epoch) / float(self.warmup_steps) + self.start_lr
        else:
            if isinstance(self.learning_rate, LRScheduler):
                lr_value = self.learning_rate()
                self.learning_rate.step()
                return lr_value

            return self.learning_rate


class ExponentialDecay(LRScheduler):
    """Update learning rate by `gamma` each epoch.

    When epoch < warmup_steps, learning rate is updated as:

    .. math::

        new\_learning\_rate = last\_learning\_rate * gamma

    References
    ----------
    - https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/ExponentialDecay_cn.html

    Parameters
    ----------
    learning_rate : float
        The initial learning rate.
    gamma : float
        The Ratio that the learning rate will be reduced.
        It should be less than 1.0. Default: 0.1.
    last_epoch : int
        The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    verbose : bool
        If ``True``, prints a message to stdout for each update. Default: ``False`` .


    Examples
    --------
    With TensorLayer

    >>> import tensorlayerx as tlx
    >>> scheduler = tlx.optimizers.lr.ExponentialDecay(learning_rate=0.1, gamma=0.9, verbose=True)
    >>> sgd = tlx.optimizers.SGD(learning_rate=scheduler,momentum=0.2)
    >>> for epoch in range(100):
    >>>     for step in range(100):
    >>>        # train model
    >>>         scheduler.step() # If you update learning rate each step
    >>>     #scheduler.step()    # If you update learning rate each epoch

    """

    def __init__(self, learning_rate, gamma, last_epoch=-1, verbose=False):
        self.gamma = gamma
        super(ExponentialDecay, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        return self.base_lr * (self.gamma**self.last_epoch)


class MultiStepDecay(LRScheduler):
    """Update the learning rate by ``gamma`` once ``epoch`` reaches one of the milestones.
    The algorithm can be described as the code below.

    .. code-block:: text

        learning_rate = 0.1
        milestones = [50, 100]
        gamma = 0.1
        if epoch < 50:
            learning_rate = 0.1
        elif epoch < 100:
            learning_rate = 0.01
        else:
            learning_rate = 0.001

    References
    ----------
    - https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/MultiStepDecay_cn.html

    Parameters
    ----------
    learning_rate : float
        The initial learning rate.
    milestones : list
        List or tuple of each boundaries. Must be increasing.
    gamma : float
        The Ratio that the learning rate will be reduced.
        It should be less than 1.0. Default: 0.1.
    last_epoch : int
        The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    verbose : bool
        If ``True``, prints a message to stdout for each update. Default: ``False`` .


    Examples
    --------
    With TensorLayer

    >>> import tensorlayerx as tlx
    >>> scheduler = tlx.optimizers.lr.MultiStepDecay(learning_rate=0.1, milestones=[50, 100], gamma=0.1, verbose=True)
    >>> sgd = tlx.optimizers.SGD(learning_rate=scheduler,momentum=0.2)
    >>> for epoch in range(100):
    >>>     for step in range(100):
    >>>        # train model
    >>>         scheduler.step() # If you update learning rate each step
    >>>     #scheduler.step()    # If you update learning rate each epoch

    """

    def __init__(self, learning_rate, milestones, gamma=0.1, last_epoch=-1, verbose=False):
        if not isinstance(milestones, (tuple, list)):
            raise TypeError(
                "The type of 'milestones' in 'MultiStepDecay' must be 'tuple, list', but received %s." %
                type(milestones)
            )

        if not all([milestones[i] < milestones[i + 1] for i in range(len(milestones) - 1)]):
            raise ValueError('The elements of milestones must be incremented')
        if gamma >= 1.0:
            raise ValueError('gamma should be < 1.0.')

        self.milestones = milestones
        self.gamma = gamma
        super(MultiStepDecay, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        for i in range(len(self.milestones)):
            if self.last_epoch < self.milestones[i]:
                return self.base_lr * (self.gamma**i)
        return self.base_lr * (self.gamma**len(self.milestones))


class LambdaDecay(LRScheduler):
    """Sets the learning rate of ``optimizer`` by function ``lr_lambda`` . ``lr_lambda`` is funciton which receives ``epoch`` .

    The algorithm can be described as the code below.

    .. code-block:: text

        learning_rate = 0.5        # init learning_rate
        lr_lambda = lambda epoch: 0.95 ** epoch

        learning_rate = 0.5        # epoch 0, 0.5*0.95**0
        learning_rate = 0.475      # epoch 1, 0.5*0.95**1
        learning_rate = 0.45125    # epoch 2, 0.5*0.95**2

    References
    ----------
    - https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/LambdaDecay_cn.html

    Parameters
    ----------
    learning_rate : float
        The initial learning rate.
    lr_lambda : function
        A function which computes a factor by ``epoch`` , and then multiply the initial learning rate by this factor.
    last_epoch : int
        The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    verbose : bool
        If ``True``, prints a message to stdout for each update. Default: ``False`` .


    Examples
    --------
    With TensorLayer

    >>> import tensorlayerx as tlx
    >>> scheduler = tlx.optimizers.lr.LambdaDecay(learning_rate=0.1, lr_lambda=lambda x:0.9**x, verbose=True)
    >>> sgd = tlx.optimizers.SGD(learning_rate=scheduler,momentum=0.2)
    >>> for epoch in range(100):
    >>>     for step in range(100):
    >>>        # train model
    >>>         scheduler.step() # If you update learning rate each step
    >>>     #scheduler.step()    # If you update learning rate each epoch

    """

    def __init__(self, learning_rate, lr_lambda, last_epoch=-1, verbose=False):
        if not callable(lr_lambda):
            raise TypeError(
                "The type of 'lr_lambda' in 'LambdaDecay' must be 'function', but received %s." % type(lr_lambda)
            )

        self.lr_lambda = lr_lambda
        super(LambdaDecay, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        return self.base_lr * self.lr_lambda(self.last_epoch)


class ReduceOnPlateau(LRScheduler):
    """Reduce learning rate when ``metrics`` has stopped descending. Models often benefit from reducing the learning rate
    by 2 to 10 times once model performance has no longer improvement.

    The ``metrics`` is the one which has been pass into ``step`` , it must be 1-D Tensor with shape [1]. When ``metrics``
    stop descending for a ``patience`` number of epochs, the learning rate will be reduced to ``learning_rate * factor`` .
    (Specially, ``mode`` can also be set to ``'max`` , in this case, when ``metrics`` stop ascending for a ``patience``
    number of epochs, the learning rate will be reduced.)

    In addition, After each reduction, it will wait a ``cooldown`` number of epochs before resuming above operation.

    References
    ----------
    - https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/LambdaDecay_cn.html

    Parameters
    ----------
    learning_rate : float
        The initial learning rate.
    mode : str
        ``'min'`` or ``'max'`` can be selected. Normally, it is ``'min'`` , which means that the learning rate will reduce when ``loss`` stops descending.
         Specially, if it's set to ``'max'`` ,  the learning rate will reduce when ``loss`` stops ascending. Default: ``'min'`` .
    factor : float
        The Ratio that the learning rate will be reduced.It should be less than 1.0. Default: 0.1.
    patience : int
        When ``loss`` doesn't improve for this number of epochs, learing rate will be reduced. Default: 10.
    threshold : float
        ``threshold`` and ``threshold_mode`` will determine the minimum change of ``loss`` . This make tiny changes of ``loss`` will be ignored. Default: 1e-4.
    threshold_mode : str
        ``'rel'`` or ``'abs'`` can be selected. In ``'rel'`` mode, the minimum change of ``loss`` is ``last_loss * threshold`` ,
        where ``last_loss`` is ``loss`` in last epoch. In ``'abs'`` mode, the minimum change of ``loss`` is ``threshold`` . Default: ``'rel'`` .
    cooldown : int
        The number of epochs to wait before resuming normal operation. Default: 0.
    min_lr : float
        The lower bound of the learning rate after reduction. Default: 0.
    epsilon : float
        Minimal decay applied to lr. If the difference between new and old lr is smaller than epsilon, the update is ignored. Default: 1e-8.
    verbose : bool
        If ``True``, prints a message to stdout for each update. Default: ``False`` .


    Examples
    --------
    With TensorLayer

    >>> import tensorlayerx as tlx
    >>> scheduler = tlx.optimizers.lr.ReduceOnPlateau(learning_rate=1.0, factor=0.5, patience=5, verbose=True)
    >>> sgd = tlx.optimizers.SGD(learning_rate=scheduler,momentum=0.2)
    >>> for epoch in range(100):
    >>>     for step in range(100):
    >>>        # train model
    >>>         scheduler.step() # If you update learning rate each step
    >>>     #scheduler.step()    # If you update learning rate each epoch

    """

    def __init__(
        self, learning_rate, mode='min', factor=0.1, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0,
        min_lr=0, epsilon=1e-8, verbose=False
    ):
        mode = mode.lower()
        if mode not in ['min', 'max']:
            raise ValueError('mode: ' + mode + ' is unknown!')
        self.mode = mode

        if factor >= 1.0:
            raise ValueError('new_lr = origin_lr * gamma and gamma should be < 1.0.')
        self.factor = factor

        threshold_mode = threshold_mode.lower()
        if threshold_mode not in ['rel', 'abs']:
            raise ValueError('threshold mode: ' + threshold_mode + ' is unknown!')
        self.threshold_mode = threshold_mode
        if not isinstance(learning_rate, (float, int)):
            raise TypeError(
                "The type of 'learning_rate' in 'ReduceOnPlateau' must be 'float', but received %s." %
                type(learning_rate)
            )

        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.epsilon = epsilon

        self.cooldown_counter = 0
        self.best = None
        self.num_bad_epochs = 0

        # Can not call Parent __init__, so implement here.
        self.base_lr = tf.Variable(initial_value=float(learning_rate))
        self.last_lr = tf.Variable(initial_value=float(learning_rate))
        self.last_epoch = 0
        self.verbose = verbose
        self._var_name = None

    # "cooldown_counter / best / num_bad_epochs / last_epoch / last_lr" will be stored.
    def step(self, metrics, epoch=None):

        if epoch is None:
            self.last_epoch = self.last_epoch + 1
        else:
            self.last_epoch = epoch

        # loss must be float, numpy.ndarray or 1-D Tensor with shape [1]
        if isinstance(metrics, (tf.Tensor, np.ndarray)):
            assert len(metrics.shape) == 1 and metrics.shape[0] == 1, "the metrics.shape " \
                                                                      "should be (1L,), but the current metrics.shape is {}. Maybe that " \
                                                                      "you should call paddle.mean to process it first.".format(
                metrics.shape)
        elif not isinstance(metrics, (int, float, np.float32, np.float64)):
            raise TypeError(
                "metrics must be 'int', 'float', 'np.float', 'numpy.ndarray' or 'paddle.Tensor', but receive {}".format(
                    type(metrics)
                )
            )

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        else:
            if self.best is None or self._is_better(metrics, self.best):
                self.best = metrics
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.num_bad_epochs > self.patience:
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0
                new_lr = max(self.last_lr * self.factor, self.min_lr)
                if self.last_lr - new_lr > self.epsilon:
                    self.last_lr.assign(new_lr)
                    if self.verbose:
                        print(
                            'Epoch {}: {} set learning rate to {}.'.format(
                                self.last_epoch, self.__class__.__name__, self.last_lr
                            )
                        )

    def _is_better(self, current, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            return current < best - best * self.threshold

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return current < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            return current > best + best * self.threshold

        else:
            return current > best + self.threshold
