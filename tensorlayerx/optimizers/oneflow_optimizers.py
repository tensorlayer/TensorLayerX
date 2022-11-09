#TODO

from __future__ import absolute_import, division, print_function
import oneflow.optim as optimizer
import oneflow as flow
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
        if closure is not None:
            with flow.enable_grad():
                loss = closure()

        state=self.optimizer_adadelta._state

        for param_group in self.param_groups:
            if param_group["do_bias_correction"]:
                param_group["bias_correction1"] = 1.0 - math.pow(
                    param_group["betas"][0], state["step"] + 1
                )
                param_group["bias_correction2"] = 1.0 - math.pow(
                    param_group["betas"][1], state["step"] + 1
                )

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
                if "exp_avg" not in state[param]:
                    state[param]["exp_avg"] = flow.zeros_like(param)
                if "exp_avg_sq" not in state[param]:
                    state[param]["exp_avg_sq"] = flow.zeros_like(param)
                if param_group["amsgrad"]:
                    if "max_exp_avg_sq" not in state[param]:
                        state[param]["max_exp_avg_sq"] = flow.zeros_like(
                            param
                        )

                m_tensor = state[param]["exp_avg"]
                v_tensor = state[param]["exp_avg_sq"]

                if param_group["amsgrad"]:
                    max_v_tensor = state[param]["max_exp_avg_sq"]
                    flow._C.dispatch_adam_update(
                        self._op_with_amsgrad,
                        (param, param.grad, m_tensor, v_tensor, max_v_tensor),
                        **kwargs,
                    )
                else:
                    flow._C.dispatch_adam_update(
                        self._op_without_amsgrad,
                        (param, param.grad, m_tensor, v_tensor),
                        **kwargs,
                    )

        state["step"] += 1
        
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
    pass

class Adam(object):
    pass

class Adamax(object):
    pass

class Ftrl(object):
    pass

class Nadam(object):
    pass

class RMSprop(object):
    pass

class SGD(object):
    pass

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
