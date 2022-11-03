#! /usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import torch
import math
import numpy as np

__all__ = [
    'LRScheduler', 'NoamDecay', 'PiecewiseDecay', 'NaturalExpDecay', 'InverseTimeDecay', 'PolynomialDecay',
    'LinearWarmup', 'ExponentialDecay', 'MultiStepDecay', 'StepDecay', 'LambdaDecay', 'ReduceOnPlateau',
    'CosineAnnealingDecay'
]


class LRScheduler(object):
    """
    LRScheduler Base class. Define the common interface of a learning rate scheduler.

    User can import it by ``from tlx.optimizer.lr import LRScheduler`` ,

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
    With TensorLayerX

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
        self.base_lr = torch.tensor(float(learning_rate))
        self.last_lr = torch.tensor(float(learning_rate))
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
        self.last_lr.fill_(new_lr)
        if self.verbose:
            print(
                'Epoch {}: {} set learning rate to {}.'.format(self.last_epoch, self.__class__.__name__, self.last_lr)
            )

    def get_lr(self):

        raise NotImplementedError


class StepDecay(LRScheduler):

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

    def __init__(self, learning_rate, gamma, last_epoch=-1, verbose=False):
        self.gamma = gamma
        super(NaturalExpDecay, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        return self.base_lr * math.exp(-1 * self.gamma * self.last_epoch)


class InverseTimeDecay(LRScheduler):

    def __init__(self, learning_rate, gamma, last_epoch=-1, verbose=False):
        self.gamma = gamma
        super(InverseTimeDecay, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        return self.base_lr / (1 + self.gamma * self.last_epoch)


class PolynomialDecay(LRScheduler):
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

    def __init__(self, learning_rate, gamma, last_epoch=-1, verbose=False):
        self.gamma = gamma
        super(ExponentialDecay, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        return self.base_lr * (self.gamma**self.last_epoch)


class MultiStepDecay(LRScheduler):

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
        self.base_lr = torch.tensor(float(learning_rate))
        self.last_lr = torch.tensor(float(learning_rate))
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
        if isinstance(metrics, (torch.Tensor, np.ndarray)):
            assert len(metrics.shape) == 1 and metrics.shape[0] == 1, "the metrics.shape " \
                                                                      "should be (1L,), but the current metrics.shape is {}. Maybe that " \
                                                                      "you should call tlx.reudce_mean to process it first.".format(
                metrics.shape)
        elif not isinstance(metrics, (int, float, np.float32, np.float64)):
            raise TypeError(
                "metrics must be 'int', 'float', 'np.float', 'numpy.ndarray', but receive {}".format(
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
                    self.last_lr.fill_(new_lr)
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