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


            # params, 
        # self.optimizer = optimizer.Adam(
        #     params, 
        #     lr=lr, 
        #     eps=eps, 
        #     betas=(beta_1, beta_2), 
        #     weight_decay=weight_decay)
    # @jt.no_grad()
    # def apply_gradients(self, loss, weights):
    #     if weights is None:
    #         raise AttributeError("Parameter train_weights must be entered.")
        
    #     if not self.init_optim:
    #         self.optimizer_adam = optimizer.Adam(
    #             params=weights, lr=get_lr(self.lr), betas=self.betas, eps=self.eps,
    #             weight_decay=self.weight_decay
    #         )
    #         self.init_optim = True
        
    #     self.optimizer_adam.zero_grad()
        
    #     # Compute and apply gradients
    #     self.optimizer_adam.step(loss)


    # def gradient(self, loss, weights=None, return_grad=True):
    #     if weights is None:
    #         raise AttributeError("Parameter train_weights must be entered.")
        
    #     if not self.init_optim:
    #         self.optimizer_adam = optimizer.Adam(
    #             params=weights, lr=get_lr(self.lr), betas=self.betas, eps=self.eps,
    #             weight_decay=self.weight_decay
    #         )
    #         self.init_optim = True
        
    #     self.optimizer_adam.zero_grad()
        
    #     # Compute gradients
    #     self.optimizer_adam.step(loss)
        
    #     grads = [p.opt_grad(self.optimizer_adam) for p in weights]
        
    #     # Optionally clip gradients
    #     if self.grad_clip is not None:
    #         self.grad_clip(grads, self.optimizer_adam)
        
    #     if return_grad:
            # return grads
class Adam(object):
    def __init__(
            self,
            lr=0.001, 
            beta_1=0.9, 
            beta_2=0.999, 
            eps=1e-8, 
            weight_decay=0.0,
            momentum = 0.0,
            grad_clip=None                    
            ):
        
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.betas = (beta_1,beta_2)
        self.eps = eps
        self.init_optim = False
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    def set(self, weights):
        if not self.init_optim:
            self.optimizer_adam = optimizer.Adam(
                params=weights, lr=self.lr, betas=self.betas, eps=self.eps,
                weight_decay=self.weight_decay
            )
            self.init_optim = True

    def zero_grad(self):
        self.optimizer_adam.zero_grad()

    def step(self, loss=None):
        self.optimizer_adam.step(loss)

class AdamW(object):
    def __init__(
            self,
            lr=0.001, 
            beta_1=0.9, 
            beta_2=0.999, 
            eps=1e-8, 
            weight_decay=0.01,
            grad_clip=None                    
            ):
        
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.betas = (beta_1, beta_2)
        self.eps = eps
        self.init_optim = False
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    def set(self, weights):
        if not self.init_optim:
            self.optimizer_adamw = optimizer.AdamW(
                params=weights, lr=self.lr, betas=self.betas, eps=self.eps,
                weight_decay=self.weight_decay
            )
            self.init_optim = True

    def zero_grad(self):
        self.optimizer_adamw.zero_grad()

    def step(self, loss=None):
        self.optimizer_adamw.step(loss)



class Adan(object):
    def __init__(
            self,
            lr=0.001, 
            beta_1=0.9, 
            beta_2=0.999, 
            beta_3=0.99,
            eps=1e-8, 
            weight_decay=0.0,
            grad_clip=None                    
            ):
        
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        self.betas = (beta_1, beta_2, beta_3)
        self.eps = eps
        self.init_optim = False
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    def set(self, weights):
        if not self.init_optim:
            self.optimizer_adan = optimizer.Adan(
                params=weights, lr=self.lr, betas=self.betas, eps=self.eps,
                weight_decay=self.weight_decay
            )
            self.init_optim = True

    def zero_grad(self):
        self.optimizer_adan.zero_grad()

    def step(self, loss=None):
        self.optimizer_adan.step(loss)



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
            eps=1e-8, 
            alpha=0.99, 
            # weight_decay=0.0,
            grad_clip=None                    
            ):
        
        self.lr = lr
        self.eps = eps
        self.alpha = alpha
        self.init_optim = False
        # self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    def set(self, weights):
        if not self.init_optim:
            self.optimizer_rmsprop = optimizer.RMSprop(
                params=weights, lr=self.lr, eps=self.eps, alpha=self.alpha,
            )
            self.init_optim = True

    def zero_grad(self):
        self.optimizer_rmsprop.zero_grad()

    def step(self, loss=None):
        self.optimizer_rmsprop.step(loss)



class SGD(object):
    def __init__(
            self,
            lr=0.01,
            momentum=0.0,
            weight_decay=0.0,
            dampening=0.0,
            nesterov=False,
            grad_clip=None
            ):

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov
        self.init_optim = False
        self.grad_clip = grad_clip

    def set(self, weights):
        if not self.init_optim:
            self.optimizer_sgd = optimizer.SGD(
                params=weights, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay,
                dampening=self.dampening, nesterov=self.nesterov
            )
            self.init_optim = True

    def zero_grad(self):
        self.optimizer_sgd.zero_grad()

    def step(self, loss=None):
        self.optimizer_sgd.step(loss)


class Momentum(object):
    def __init__(
            self,
            lr=0.001, 
            momentum=0.9,
            weight_decay=0.0,
            nesterov=False,
            grad_clip=None                    
            ):
        
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.grad_clip = grad_clip
        self.init_optim = False

    def set(self, weights):
        if not self.init_optim:
            self.optimizer_momentum = optimizer.SGD(
                params=weights, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay, nesterov=self.nesterov
            )
            self.init_optim = True

    def zero_grad(self):
        self.optimizer_momentum.zero_grad()

    def step(self, loss=None):
        self.optimizer_momentum.step(loss)




def Lamb(**kwargs):
    raise Exception('Lamb optimizer function not implemented')


def LARS(**kwargs):
    raise Exception('LARS optimizer function not implemented')


def _grads(weights, optimizer):
    grads = []
    for w in weights:
        grads.append(w.opt_grad(optimizer))
    return grads


def get_lr(lr):
    if isinstance(lr, LRScheduler):
        return lr()
    return lr
