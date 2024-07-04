#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import jittor.optim as optimizer
import jittor as jt
import jittor.nn as nn
from tensorlayerx.optimizers.lr import LRScheduler

__all__ = ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD', 'Momentum', 'Lamb', 'LARS']


class Adadelta(object):

    def __init__(self):
        pass

    def app_gradients(self):
        raise Exception('Adadelta optimizer function not implemented')


class Adagrad(object):

    def __init__(self):
        pass

    def app_gradients(self):
        raise Exception('Adagrad optimizer function not implemented')


class Adam(object):
    def __init__(self, params, lr=0.001, beta_1=0.9, beta_2=0.999, eps=1e-8, weight_decay=0.0):
        self.optimizer = optimizer.Adam(params, lr=lr, eps=eps, betas=(beta_1, beta_2), weight_decay=weight_decay)

    def step(self, loss=None):
        self.optimizer.step(loss)

    def zero_grad(self):
        self.optimizer.zero_grad()

class AdamW(object):
    def __init__(self, params, lr=0.001, beta_1=0.9, beta_2=0.999, eps=1e-8, weight_decay=0.0):
        self.optimizer = optimizer.AdamW(params, lr=lr, eps=eps, betas=(beta_1, beta_2), weight_decay=weight_decay)

    def step(self, loss=None):
        self.optimizer.step(loss)

    def zero_grad(self):
        self.optimizer.zero_grad()


class Adan(object):
    def __init__(self, params, lr=0.001, beta_1=0.9, beta_2=0.999, eps=1e-8, weight_decay=0.0):
        self.optimizer = optimizer.Adan(params, lr=lr, eps=eps, betas=(beta_1, beta_2), weight_decay=weight_decay)

    def step(self, loss=None):
        self.optimizer.step(loss)

    def zero_grad(self):
        self.optimizer.zero_grad()


class Adamax(object):

    def __init__(self):
        pass

    def apply_gradients(self):
        raise Exception('Adamax optimizer function not implemented')



class Ftrl(object):

    def __init__(self):
        raise NotImplementedError("Ftrl optimizer is not implemented")

    def apply_gradients(self):
        pass

    def gradient(self, train_weights=None):
        pass


class Nadam(object):

    def __init__(self):
        raise NotImplementedError("Nadam optimizer is not implemented")

    def apply_gradients(self):
        pass

    def gradient(self, train_weights=None):
        pass


class RMSprop(object):

    def __init__(
        self,
        lr=0.001,
        rho=0.99,
        momentum=0.0,
        eps=1e-08,
        centered=False,
        weight_decay=0.0,
        grad_clip=None,
    ):
        self.lr = lr
        self.rho = rho
        self.momentum = momentum
        self.eps = eps
        self.centered = centered
        self.init_optim = False
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    @jt.no_grad()
    def apply_gradients(self, grads_and_vars=None, closure=None):
        if not self.init_optim:
            raise AttributeError("Can not apply gradients before zero_grad call.")

        loss = None
        if closure is not None:
            with jt.enable_grad():
                loss = closure()

        for group in self.optimizer_rmsprop.param_groups:
            params_with_grad = []
            grads = []
            square_avgs = []
            grad_avgs = []
            momentum_buffer_list = []

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)

                if p.grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                grads.append(p.grad)

                state = self.optimizer_rmsprop.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = jt.zeros_like(p)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = jt.zeros_like(p)
                    if group['centered']:
                        state['grad_avg'] = jt.zeros_like(p)

                square_avgs.append(state['square_avg'])

                if group['momentum'] > 0:
                    momentum_buffer_list.append(state['momentum_buffer'])
                if group['centered']:
                    grad_avgs.append(state['grad_avg'])

                state['step'] += 1

            optimizer.RMSprop(params_with_grad,
                      grads,
                      square_avgs,
                      grad_avgs,
                      momentum_buffer_list,
                      lr=get_lr(self.lr),
                      alpha=group['alpha'],
                      eps=group['eps'],
                      weight_decay=group['weight_decay'],
                      momentum=group['momentum'],
                      centered=group['centered'])

        return loss

    def gradient(self, loss, weights=None, return_grad=True):
        if weights is None:
            raise AttributeError("Parameter train_weights must be entered.")
        if not self.init_optim:
            self.optimizer_rmsprop = optimizer.RMSprop(
                params=weights, lr=get_lr(self.lr), alpha=self.rho, eps=self.eps, momentum=self.momentum,
                centered=self.centered, weight_decay=self.weight_decay
            )
            self.init_optim = True
        self.optimizer_rmsprop.zero_grad()
        loss.backward()

        if self.grad_clip is not None:
            self.grad_clip(weights)

        if return_grad ==True:
            return _grads(weights)
        else:
            return None


class SGD(object):

    def __init__(
        self,
        lr=0.001,
        momentum=0,
        weight_decay=0.0,
        grad_clip=None,
    ):
        self.lr = lr
        self.momentum = momentum
        self.init_optim = False
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    @jt.no_grad()
    def apply_gradients(self, grads_and_vars=None, closure=None):
        if not self.init_optim:
            raise AttributeError("Can not apply gradients before zero_grad call.")

        loss = None
        if closure is not None:
            with jt.enable_grad():
                loss = closure()

        for group in self.optimizer_sgd.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = get_lr(self.lr)

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.optimizer_sgd.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            optimizer.SGD(params_with_grad,
                  d_p_list,
                  momentum_buffer_list,
                  weight_decay=weight_decay,
                  momentum=momentum,
                  lr=lr,
                  dampening=dampening,
                  nesterov=nesterov)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.optimizer_sgd.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss

    def gradient(self, loss, weights=None, return_grad=True):
        if weights is None:
            raise AttributeError("Parameter train_weights must be entered.")
        if not self.init_optim:
            self.optimizer_sgd = optimizer.SGD(
                params=weights, lr=get_lr(self.lr), momentum=self.momentum, weight_decay=self.weight_decay
            )
            self.init_optim = True
        self.optimizer_sgd.zero_grad()
        loss.backward()

        if self.grad_clip is not None:
            self.grad_clip(weights)

        if return_grad ==True:
            return _grads(weights)
        else:
            return None



class Momentum(object):

    def __init__(
        self,
        lr=0.001,
        momentum=0,
        weight_decay=0.0,
        nesterov=False,
        grad_clip=None,
    ):
        self.lr = lr
        self.momentum = momentum
        self.init_optim = False
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.grad_clip = grad_clip

    @jt.no_grad()
    def apply_gradients(self, grads_and_vars=None, closure=None):
        if not self.init_optim:
            raise AttributeError("Can not apply gradients before zero_grad call.")

        loss = None
        if closure is not None:
            with jt.enable_grad():
                loss = closure()

        for group in self.optimizer_momentum.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = get_lr(self.lr)

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.optimizer_momentum.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            optimizer.SGD(params_with_grad,
                  d_p_list,
                  momentum_buffer_list,
                  weight_decay=weight_decay,
                  momentum=momentum,
                  lr=lr,
                  dampening=dampening,
                  nesterov=nesterov)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.optimizer_momentum.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss

    def gradient(self, loss, weights=None, return_grad=True):
        if weights is None:
            raise AttributeError("Parameter train_weights must be entered.")
        if not self.init_optim:
            self.optimizer_momentum = optimizer.SGD(
                params=weights, lr=get_lr(self.lr), momentum=self.momentum, weight_decay=self.weight_decay, nesterov=self.nesterov
            )
            self.init_optim = True
        self.optimizer_momentum.zero_grad()
        loss.backward()

        if self.grad_clip is not None:
            self.grad_clip(weights)

        if return_grad ==True:
            return _grads(weights)
        else:
            return None


def Lamb(**kwargs):
    raise Exception('Lamb optimizer function not implemented')


def LARS(**kwargs):
    raise Exception('LARS optimizer function not implemented')


def _grads(weights):
    grads = []
    for w in weights:
        grads.append(w.grad)
    return grads

def get_lr(lr):
    if isinstance(lr, LRScheduler):
        return lr()
    return lr
