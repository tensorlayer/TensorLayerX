#TODO

from __future__ import absolute_import, division, print_function
import oneflow.optim as optimizer
import oneflow as flow
import oneflow.nn.functional as F

import math

from tensorlayerx.optimizers.lr import LRScheduler

__all__ = ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD', 'Momentum', 'Lamb', 'LARS']


class Adadelta(object):

    def __init__(
        self,
        lr=0.001,
        rho=0.95,
        eps=1e-10,
        weight_decay=0.0,
        grad_clip=None,
    ):
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.init_optim = False
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    @flow.no_grad()
    def apply_gradients(self, grads_and_vars=None, closure=None):
        if not self.init_optim:
            raise AttributeError("Can not apply gradients before zero_grad call.")
        loss = None
        with flow.enable_grad():
            if closure is not None:
                    loss = closure()
            for param_group in self.optimizer_adadelta.param_groups:
                kwargs = {
                    "learning_rate": param_group["lr"],
                    "l2": param_group["weight_decay"],
                    "rho": param_group["rho"],
                    "epsilon": param_group["eps"],
                    "maximize": param_group["maximize"],
                }
                for param in param_group.parameters:
                    if param.grad is None:
                        continue
                    square_avgs_tensor = self.optimizer_adadelta._state[param]["square_avgs"]
                    acc_deltas_tensor = self.optimizer_adadelta._state[param]["acc_deltas"]
                    flow._C.dispatch_adadelta_update(
                        self.optimizer_adadelta._op,
                        (param, param.grad, square_avgs_tensor, acc_deltas_tensor),
                        **kwargs,
                    )

            self.optimizer_adadelta._state["step"] = self.optimizer_adadelta._state["step"] + 1
            return loss


    def gradient(self, loss, weights=None, return_grad=True):
        if weights is None:
            raise AttributeError("Parameter train_weights must be entered.")
        if not self.init_optim:
            self.optimizer_adadelta = optimizer.Adadelta(
                params=weights, lr=get_lr(self.lr), rho=self.rho, eps=self.eps, weight_decay=self.weight_decay
            )
            self.init_optim = True
        self.optimizer_adadelta.zero_grad()
        loss.backward()

        if self.grad_clip is not None:
            self.grad_clip(weights)

        if return_grad ==True:
            return _grads(weights)
        else:
            return None

class Adagrad(object):
    
    def __init__(
        self,
        lr=0.001,
        lr_decay=0.0,
        weight_decay=0.0,
        initial_accumulator_value=0.0,
        grad_clip=None,
    ):
        self.lr = lr
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.initial_accumulator_value = initial_accumulator_value
        self.init_optim = False
        self.grad_clip = grad_clip

    @flow.no_grad()
    def apply_gradients(self, grads_and_vars=None, closure=None):
        if not self.init_optim:
            raise AttributeError("Can not apply gradients before zero_grad call.")
        loss = None
        with flow.enable_grad():
            if closure is not None:
                    loss = closure()
            for param_group in self.optimizer_adagrad.param_groups:
                kwargs = {
                "learning_rate": param_group["lr"],
                "l2": param_group["weight_decay"],
                "epsilon": param_group["eps"],
                "lr_decay": param_group["lr_decay"],
                "train_step_val": self.optimizer_adagrad._state["step"] + 1,
                }
                for param in param_group.parameters:
                    if param.grad is None:
                        continue
                    sum_tensor = self.optimizer_adagrad._state[param]["sum"]
                    flow._C.dispatch_adagrad_update(
                        self.optimizer_adagrad._op, (param, param.grad, sum_tensor), **kwargs
                    )

            self.optimizer_adagrad._state["step"] = self.optimizer_adagrad._state["step"] + 1
            return loss

    def gradient(self, loss, weights=None, return_grad=True):
        if weights is None:
            raise AttributeError("Parameter train_weights must be entered.")
        if not self.init_optim:
            self.optimizer_adagrad = optimizer.Adagrad(
                params=weights,
                lr=get_lr(self.lr),
                lr_decay=self.lr_decay,
                weight_decay=self.weight_decay,
                initial_accumulator_value=self.initial_accumulator_value,
            )
            self.init_optim = True

        self.optimizer_adagrad.zero_grad()
        loss.backward()

        if self.grad_clip is not None:
            self.grad_clip(weights)

        if return_grad ==True:
            return _grads(weights)
        else:
            return None

class Adam(object):
        
    def __init__(
        self,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
        grad_clip=None,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.init_optim = False
        self.grad_clip = grad_clip

    @flow.no_grad()
    def apply_gradients(self, grads_and_vars=None, closure=None):
        if not self.init_optim:
            raise AttributeError("Can not apply gradients before zero_grad call.")
        loss = None
        with flow.enable_grad():
            if closure is not None:
                loss = closure()
            for param_group in self.optimizer_adam.param_groups:
                kwargs = {
                    "learning_rate": param_group["lr"],
                    "bias_correction1": param_group["bias_correction1"],
                    "bias_correction2": param_group["bias_correction2"],
                    "l2": param_group["weight_decay"],
                    "beta1": param_group["betas"][0],
                    "beta2": param_group["betas"][1],
                    "epsilon": param_group["eps"],
                    "do_bias_correction": param_group["do_bias_correction"],
                    "amsgrad": param_group["amsgrad"],
                }
                for param in param_group.parameters:
                    if param.grad is None:
                        continue
                    if "exp_avg" not in self.optimizer_adam._state[param]:
                        self.optimizer_adam._state[param]["exp_avg"] = flow.zeros_like(param)
                    if "exp_avg_sq" not in self.optimizer_adam._state[param]:
                        self.optimizer_adam._state[param]["exp_avg_sq"] = flow.zeros_like(param)
                    if param_group["amsgrad"]:
                        if "max_exp_avg_sq" not in self.optimizer_adam._state[param]:
                            self.optimizer_adam._state[param]["max_exp_avg_sq"] = flow.zeros_like(
                                param
                            )

                    m_tensor = self.optimizer_adam._state[param]["exp_avg"]
                    v_tensor = self.optimizer_adam._state[param]["exp_avg_sq"]

                    if param_group["amsgrad"]:
                        max_v_tensor = self.optimizer_adam._state[param]["max_exp_avg_sq"]
                        flow._C.dispatch_adam_update(
                            self.optimizer_adam._op_with_amsgrad,
                            (param, param.grad, m_tensor, v_tensor, max_v_tensor),
                            **kwargs,
                        )
                    else:
                        flow._C.dispatch_adam_update(
                            self.optimizer_adam._op_without_amsgrad,
                            (param, param.grad, m_tensor, v_tensor),
                            **kwargs,
                        )

            self.optimizer_adam._state["step"] += 1

            return loss

    def gradient(self, loss, weights=None, return_grad=True):
        if weights is None:
            raise AttributeError("Parameter train_weights must be entered.")
        if not self.init_optim:
            self.optimizer_adam = optimizer.Adam(
                params=weights,
                lr=get_lr(self.lr),
                betas=(self.beta1, self.beta2),
                eps=self.eps,
                weight_decay=self.weight_decay,
                amsgrad=self.amsgrad,
            )
            self.init_optim = True
        self.optimizer_adam.zero_grad()
        loss.backward()

        if self.grad_clip is not None:
            self.grad_clip(weights)

        if return_grad ==True:
            return _grads(weights)
        else:
            return None


    

class Adamax(object):

    def __init__(
        self,
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        eps=1e-8,
        weight_decay=0.0,
        grad_clip=None,
    ):
        
        raise NotImplementedError("Adamax is not implemented for oneflow backend yet.")

    def apply_gradients(self):
        pass

    def gradient(self, train_weights=None):
        pass

class Ftrl(object):
    def __init__(self):
        NotImplementedError("Ftrl is not implemented for oneflow backend yet.")

    def apply_gradients(self):
        pass

    def gradient(self, train_weights=None):
        pass

class Nadam(object):

    def __init__(self):
        NotImplementedError("Nadam is not implemented for oneflow backend yet.")

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

    @flow.no_grad()
    def apply_gradients(self, grads_and_vars=None, closure=None):
        if not self.init_optim:
            raise AttributeError("Can not apply gradients before zero_grad call.")
        loss = None
        with flow.enable_grad():
            if closure is not None:
                loss = closure()
            for param_group in self.optimizer_rmsprop.param_groups:
                kwargs = {
                    "learning_rate": param_group["lr"],
                    "epsilon": param_group["eps"],
                    "decay_rate": param_group["alpha"],
                    "l2": param_group["weight_decay"],
                }
                for param in param_group.parameters:
                    if param.grad is None:
                        continue
                    if "square_avg" not in self.optimizer_rmsprop._state[param]:
                        self.optimizer_rmsprop._state[param]["square_avg"] = flow.zeros_like(param)
                    ms_tensor = self.optimizer_rmsprop._state[param]["square_avg"]

                    if param_group["centered"]:
                        if "grad_avg" not in self.optimizer_rmsprop._state[param]:
                            self.optimizer_rmsprop._state[param]["grad_avg"] = flow.zeros_like(param)
                        mg_tensor = self.optimizer_rmsprop._state[param]["grad_avg"]
                        flow._C.dispatch_rmsprop_update(
                            self.optimizer_rmsprop._centered_rmsprop,
                            (param, param.grad, ms_tensor, mg_tensor),
                            centered=True,
                            **kwargs,
                        )
                    else:
                        flow._C.dispatch_rmsprop_update(
                            self.optimizer_rmsprop._rmsprop, (param, param.grad, ms_tensor), **kwargs
                        )
            self.optimizer_rmsprop._state["step"] = self.optimizer_rmsprop._state["step"] + 1
            return loss

    def gradient(self, loss, weights=None, return_grad=True):
        if weights is None:
            raise AttributeError("Parameter train_weights must be entered.")
        if not self.init_optim:
            self.optimizer_rmsprop = optimizer.RMSprop(
                params=weights,
                lr=get_lr(self.lr),
                alpha=self.rho,
                eps=self.eps,
                centered=self.centered,
                weight_decay=self.weight_decay,
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
        lr=0.01,
        momentum=0.0,
        weight_decay=0.0,
        grad_clip=None,
    ):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.init_optim = False
        self.grad_clip = grad_clip

    @flow.no_grad()
    def apply_gradients(self, grads_and_vars=None, closure=None):
        if not self.init_optim:
            raise AttributeError("Can not apply gradients before zero_grad call.")
        loss = None
        with flow.enable_grad():
            if closure is not None:
                loss = closure()
                lr = param_group["lr"]
                l2 = param_group["weight_decay"]
                for param in param_group.parameters:
                    if param.grad is None:
                        continue
                    if param_group["momentum"] == 0.0:
                        # TODO: Support param `maximize` in Naive SGD Optimizer. (zhengzekang)
                        flow._C.dispatch_sgd_update(
                            self._sgd, (param, param.grad), learning_rate=lr, l2=l2
                        )
                    else:
                        if "momentum_buf" not in self._state[param]:
                            self._state[param]["momentum_buf"] = flow.zeros_like(param)
                        momentum_buf = self._state[param]["momentum_buf"]
                        beta = param_group["momentum"]
                        dampening = param_group["dampening"]
                        nesterov = param_group["nesterov"]
                        maximize = param_group["maximize"]
                        flow._C.dispatch_momentum_update(
                            self._momentum_sgd,
                            (param, param.grad, momentum_buf),
                            learning_rate=lr,
                            l2=l2,
                            beta=beta,
                            dampening=dampening,
                            nesterov=nesterov,
                            maximize=maximize,
                        )
            self._state["step"] = self._state["step"] + 1
            return loss

    def gradient(self, loss, weights=None, return_grad=True):
        if weights is None:
            raise AttributeError("Parameter train_weights must be entered.")
        if not self.init_optim:
            self.optimizer_sgd = optimizer.SGD(
                params=weights,
                lr=get_lr(self.lr),
                momentum=self.momentum,
                weight_decay=self.weight_decay,
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
    pass

class Lamb(object):
    pass

class LARS(object):
    pass


def _grads(weights):
    grads = []
    for w in weights:
        grads.append(w.grad)
    return grads

def get_lr(lr):
    if isinstance(lr, LRScheduler):
        return lr()
    return lr
