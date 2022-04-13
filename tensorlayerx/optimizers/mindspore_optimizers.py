#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
from mindspore.nn import optim as optimizer
import mindspore as ms
from mindspore.nn import Cell

__all__ = ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD', 'Momentum', 'Lamb', 'LARS']


class Adadelta(Cell):

    def __init__(self):
        pass

    def app_gradients(self):
        raise Exception('Adadelta optimizer function not implemented')


class Adagrad(Cell):

    def __init__(self, lr=0.001, initial_accumulator=0.1, eps=1e-07, weight_decay=0.0, grad_clip=None):
        super(Adagrad, self).__init__()
        self.lr = lr
        self.initial_accumulator = initial_accumulator
        self.eps = eps
        self.weight_decay = weight_decay
        self.adagrad = optimizer.Adagrad

    def apply_gradients(self, grads_and_vars):
        grads, vars = list(zip(*grads_and_vars))
        optimizer = self.adagrad(
            vars, learning_rate=self.lr, accum=self.initial_accumulator, weight_decay=self.weight_decay
        )
        optimizer(grads)


class Adam(Cell):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, eps=1e-8, weight_decay=0.0, grad_clip=None):
        super(Adam, self).__init__()
        self.adam = optimizer.Adam
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.weight_decay = weight_decay

    def apply_gradients(self, grads_and_vars):
        grads, vars = list(zip(*grads_and_vars))
        optimizer_adam = self.adam(
            vars, learning_rate=self.lr, beta1=self.beta_1, beta2=self.beta_2, eps=self.eps,
            weight_decay=self.weight_decay
        )
        optimizer_adam(grads)


class Adamax(Cell):

    def __init__(self):
        pass

    def apply_gradients(self):
        raise Exception('Adamax optimizer function not implemented')


class Ftrl(Cell):

    def __init__(
        self, lr=0.001, lr_power=-0.5, initial_accumulator_value=0.1,
        l1_regularization_strength=0.0, l2_regularization_strength=0.0, beta=0.0,
        l2_shrinkage_regularization_strength=0.0, weight_decay=0.0, grad_clip=None
    ):
        super(Ftrl, self).__init__()
        self.ftrl = optimizer.FTRL
        self.lr = lr
        self.lr_power = lr_power
        self.init_accum = initial_accumulator_value
        self.l1 = l1_regularization_strength
        self.l2 = l2_regularization_strength
        self.weight_decay = weight_decay

    def apply_gradients(self, grads_and_vars):
        grads, vars = list(zip(*grads_and_vars))
        optimizer_adam = self.ftrl(
            vars, learning_rate=self.lr, initial_accum=self.init_accum, lr_power=self.lr_power, l1=self.l1, l2=self.l2,
            weight_decay=self.weight_decay
        )
        optimizer_adam(grads)


class Nadam(Cell):

    def __init__(self):
        pass

    def apply_gradients(self):
        raise Exception('Nadam optimizer function not implemented')


class RMSprop(Cell):

    def __init__(
        self, lr=0.01, rho=0.9, eps=1.0e-10, momentum=0.0, centered=False, weight_decay=0.0,
        grad_clip=None
    ):
        super(RMSprop, self).__init__()
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.momentum = momentum
        self.centered = centered
        self.weight_decay = weight_decay
        self.rmsprop = optimizer.RMSProp

    def apply_gradients(self, grads_and_vars):
        grads, vars = list(zip(*grads_and_vars))
        optimizer_adam = self.rmsprop(
            vars, learning_rate=self.lr, decay=self.rho, momentum=self.momentum, epsilon=self.eps,
            centered=self.centered, weight_decay=self.weight_decay
        )
        optimizer_adam(grads)


class SGD(Cell):

    def __init__(self, lr=0.1, momentum=0.0, weight_decay=0.0, grad_clip=None):
        super(SGD, self).__init__()
        self.sgd = optimizer.SGD
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def apply_gradients(self, grads_and_vars):
        grads, vars = list(zip(*grads_and_vars))
        optimizer_sgd = self.sgd(
            vars, learning_rate=self.lr, momentum=self.momentum, weight_decay=self.weight_decay
        )
        optimizer_sgd(grads)


class Momentum(Cell):

    def __init__(self, lr, momentum, use_nesterov=False, weight_decay=0.0, grad_clip=None):
        super(Momentum, self).__init__()
        self.mom = optimizer.Momentum
        self.lr = lr
        self.momentum = momentum
        self.use_nesterov = use_nesterov
        self.weight_decay = weight_decay

    def apply_gradients(self, grads_and_vars):
        grads, vars = list(zip(*grads_and_vars))
        optimizer_mom = self.mom(
            vars, learning_rate=self.lr, momentum=self.momentum, use_nesterov=self.use_nesterov,
            weight_decay=self.weight_decay
        )
        optimizer_mom(grads)


class Lamb(Cell):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, eps=1.0e-6, weight_decay=0.0, grad_clip=None):
        super(Lamb, self).__init__()
        self.lamb = optimizer.Lamb
        self.lr = lr
        self.beta1 = beta_1
        self.beta2 = beta_2
        self.eps = eps
        self.weight_decay = weight_decay

    def apply_gradients(self, grads_and_vars):
        grads, vars = list(zip(*grads_and_vars))
        optimizer_lamb = self.lamb(
            vars, learning_rate=self.lr, beta1=self.beta1, beta2=self.beta2, eps=self.eps,
            weight_decay=self.weight_decay
        )
        optimizer_lamb(grads)


class LARS(Cell):

    def __init__(self, optimizer, **kwargs):
        super(LARS, self).__init__()
        self.lars = ms.nn.LARS(optimizer=optimizer, **kwargs)

    def apply_gradients(self, grads_and_vars):
        grads, _ = list(zip(*grads_and_vars))
        self.lars(grads)
