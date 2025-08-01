#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
from abc import ABC, abstractmethod
import paddle
from paddle.optimizer import Optimizer
import warnings
from .. import is_distributed

__all__ = ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD', 'Momentum', 'Lamb', 'LARS']

class TLX_Optimizer(Optimizer, ABC):
    def __init__(self):
        self._optimizer = None
        self._distribution = is_distributed()

    def gradient(self, loss, weights):
        if loss is None:
            raise ValueError('loss is not set.')
        if weights is None:
            raise ValueError('weights is not set.')
        if not self.init_optim:
            self._optimizer = self._create_optimizer(weights)
            if self._distribution:
                self._optimizer = paddle.distributed.fleet.distributed_optimizer(self._optimizer)
            self.init_optim = True

        loss.backward()
        grads_and_vars = self._optimizer.backward(loss=loss, parameters=weights)

        params, grads, self.grad_succeed = filter_grads(grads_and_vars, weights)
        self.grads_and_vars = grads_and_vars
        return grads

    @abstractmethod
    def _create_optimizer(self, weights):
        pass

    def apply_gradients(self, grads_and_vars):
        grads_and_vars = zip_grads_and_params(grads_and_vars, self.grad_succeed, self.grads_and_vars)
        if grads_and_vars is None:
            raise ValueError('grads_and_vars is not set.')
        self._optimizer._apply_optimize(loss=None, startup_program=None, params_grads=grads_and_vars)
        self._optimizer.clear_grad()


class Adadelta(TLX_Optimizer):

    def __init__(self, lr=0.001, eps=1.0e-6, rho=0.95, weight_decay=0.0, grad_clip=None):
        super().__init__()
        if lr is None:
            raise ValueError('learn_rate is not set.')
        if eps is None:
            raise ValueError('eps is not set.')
        if rho is None:
            raise ValueError('rho is not set')
        self.lr = lr
        self.eps = eps
        self.rho = rho
        self.grad_succeed = True
        self.init_optim = False
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self._grad_clip = grad_clip

    def _create_optimizer(self, weights):
        return paddle.optimizer.Adadelta(
            learning_rate=self.lr, epsilon=self.eps, rho=self.rho, parameters=weights,
            grad_clip=self.grad_clip, weight_decay=self.weight_decay
        )


class Adagrad(TLX_Optimizer):

    def __init__(self, lr=0.001, initial_accumulator_value=0.0, eps=1.0e-6, weight_decay=0.0, grad_clip=None):

        super().__init__()
        if lr is None:
            raise ValueError('lr is not set.')
        if initial_accumulator_value is None:
            raise ValueError('initial_accumulator_value is not set.')
        if eps is None:
            raise ValueError('eps is not set.')

        self.lr = lr
        self.initial_accumulator_value = initial_accumulator_value
        self.eps = eps
        self.grad_succeed = True
        self.init_optim = False
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self._grad_clip = grad_clip

    def _create_optimizer(self, weights):
        return paddle.optimizer.Adagrad(
            learning_rate=self.lr, epsilon=self.eps, initial_accumulator_value=self.initial_accumulator_value,
            parameters=weights, grad_clip=self.grad_clip, weight_decay=self.weight_decay
        )


class Adam(TLX_Optimizer):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, eps=1.0e-8, weight_decay=0.0, grad_clip=None):

        super().__init__()
        if lr is None:
            raise ValueError('lr is not set.')
        if beta_1 is None:
            raise ValueError('beta_1 is not set.')
        if beta_2 is None:
            raise ValueError('beta_2 is not set.')
        if eps is None:
            raise ValueError('eps is not set.')

        if not 0 <= beta_1 < 1:
            raise ValueError("Invaild value of beta1, expect beta1 in [0,1).")
        if not 0 <= beta_2 < 1:
            raise ValueError("Invaild value of beta2, expect beta2 in [0,1).")

        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.init_optim = False
        self.grad_succeed = True
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self._grad_clip = grad_clip
        self._parameter_list = None

    def _create_optimizer(self, weights):
        return paddle.optimizer.Adam(
            learning_rate=self.lr, beta1=self.beta_1, beta2=self.beta_2, epsilon=self.eps,
            parameters=weights, grad_clip=self.grad_clip, weight_decay=self.weight_decay
        )
    

class Adamax(TLX_Optimizer):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, eps=1.0e-8, weight_decay=0.0, grad_clip=None):

        super().__init__()
        if lr is None:
            raise ValueError('lr is not set.')
        if beta_1 is None:
            raise ValueError('beta_1 is not set.')
        if beta_2 is None:
            raise ValueError('beta_2 is not set.')
        if eps is None:
            raise ValueError('eps is not set.')

        if not 0 <= beta_1 < 1:
            raise ValueError("Invaild value of beta1, expect beta1 in [0,1).")
        if not 0 <= beta_2 < 1:
            raise ValueError("Invaild value of beta2, expect beta2 in [0,1).")

        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.grad_succeed = True
        self.init_optim = False
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self._grad_clip = grad_clip

    def _create_optimizer(self, weights):
        return paddle.optimizer.Adamax(
            learning_rate=self.lr, beta1=self.beta_1, beta2=self.beta_2, epsilon=self.eps,
            parameters=weights, grad_clip=self.grad_clip, weight_decay=self.weight_decay
        )


class Ftrl(TLX_Optimizer):

    def __init__(self):

        raise Exception('Ftrl optimizer function not implemented')


class Nadam(TLX_Optimizer):

    def __init__(self):

        raise Exception('Nadam optimizer function not implemented')


class RMSprop(TLX_Optimizer):

    def __init__(
        self, lr=0.001, rho=0.95, eps=1.0e-6, momentum=0.0, centered=False, weight_decay=0.0,
        grad_clip=None
    ):
        super().__init__()
        if lr is None:
            raise ValueError("lr is not set.")
        if rho is None:
            raise ValueError("rho is not set.")
        if eps is None:
            raise ValueError("eps is not set.")
        if momentum is None:
            raise ValueError("momentum is not set.")
        if not 0.0 <= eps:
            raise ValueError("Invalid value of eps, expect eps >= 0.")
        if not 0.0 <= momentum:
            raise ValueError("Invalid value of momentum, expect momentum >= 0.")
        if not 0.0 <= rho:
            raise ValueError("Invalid value of rho, expect rho >= 0.")

        self.lr = lr
        self.eps = eps
        self.rho = rho
        self.momentum = momentum
        self.centered = centered
        self.grad_succeed = True
        self.init_optim = False
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self._grad_clip = grad_clip

    def _create_optimizer(self, weights):
        return paddle.optimizer.RMSprop(
            learning_rate=self.lr, epsilon=self.eps, rho=self.rho, momentum=self.momentum,
            parameters=weights, grad_clip=self.grad_clip, weight_decay=self.weight_decay
        )


class SGD(TLX_Optimizer):

    def __init__(self, lr=0.1, momentum=0.0, weight_decay=0.0, grad_clip=None):
        super().__init__()
        if lr is None:
            raise ValueError("lr is not set.")

        self.lr = lr
        self.grad_succeed = True
        self.init_optim = False
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self._grad_clip = grad_clip

    def _create_optimizer(self, weights):
        return paddle.optimizer.SGD(
            learning_rate=self.lr, parameters=weights, grad_clip=self.grad_clip,
            weight_decay=self.weight_decay
        )


class Momentum(TLX_Optimizer):

    def __init__(self, lr=0.001, momentum=0.9,  weight_decay=0.0, nesterov=False, grad_clip=None):
        super().__init__()
        if lr is None:
            raise ValueError("lr is not set")
        if momentum is None:
            raise ValueError("momentum is not set")

        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.grad_succeed = True
        self.init_optim = False
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self._grad_clip = grad_clip

    def _create_optimizer(self, weights):
        return paddle.optimizer.Momentum(
            learning_rate=self.lr, momentum=self.momentum, parameters=weights,
            use_nesterov=self.nesterov, grad_clip=self.grad_clip, weight_decay=self.weight_decay
        )


class Lamb(TLX_Optimizer):

    def __init__(
        self, lr=0.001, beta_1=0.9, beta_2=0.999, eps=1.0e-6, weight_decay=0.01, grad_clip=None
    ):

        super().__init__()
        if lr is None:
            raise ValueError('lr is not set.')
        if beta_1 is None:
            raise ValueError('beta_1 is not set.')
        if beta_2 is None:
            raise ValueError('beta_2 is not set.')
        if eps is None:
            raise ValueError('eps is not set.')

        if not 0 <= beta_1 < 1:
            raise ValueError("Invaild value of beta1, expect beta1 in [0,1).")
        if not 0 <= beta_2 < 1:
            raise ValueError("Invaild value of beta2, expect beta2 in [0,1).")

        self.lr = lr
        self.lamb_weight_decay = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.grad_succeed = True
        self.init_optim = False
        self.grad_clip = grad_clip
        self._grad_clip = grad_clip

    def _create_optimizer(self, weights):
        return paddle.optimizer.Lamb(
            learning_rate=self.lr, lamb_weight_decay=self.lamb_weight_decay, beta1=self.beta_1,
            beta2=self.beta_2, epsilon=self.eps, parameters=weights, grad_clip=self.grad_clip
        )


class LARS(TLX_Optimizer):

    def __init__(self):

        pass

    def gradient(self):

        pass

    def _create_optimizer(self, weights):
        raise Exception('LARS optimizer function not implemented')
    
    def apply_gradients(self, grads_and_vars):

        raise Exception('LARS optimizer function not implemented')


# TODO There may be gradient incompleteness when calculating gradient paddle.optimizer.backward.
def filter_grads(grads_and_vars, weights):
    try:
        params, grads = list(zip(*grads_and_vars))
    except:
        params, grads = [], []

    if len(grads) - len(weights) == 0:
        grad_succeed = True
    else:
        grad_succeed = False
    return params, grads, grad_succeed


def zip_grads_and_params(grads_and_vars, grad_succeed, call_grads_and_vars):
    if grad_succeed == False:
        grads_and_vars = call_grads_and_vars
        warnings.warn("The number of gradients and training parameters are not equal", RuntimeWarning)
    else:
        grads, params = list(zip(*grads_and_vars))
        grads_and_vars = list(zip(params, grads))
    return grads_and_vars
