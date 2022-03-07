#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorlayerx as tlx
from tensorlayerx.nn import Module

if tlx.BACKEND == 'tensorflow':
    import tensorflow as tf
if tlx.BACKEND == 'mindspore':
    from mindspore.ops import composite
    from mindspore.common import ParameterTuple
if tlx.BACKEND == 'paddle':
    import paddle as pd
if tlx.BACKEND == 'torch':
    import torch

class WithLoss(Module):
    """
    High-Level API for Training or Testing.

    Wraps the network with loss function. This Module accepts data and label as inputs and
    the computed loss will be returned.

    Parameters
    ----------
    backbone : tensorlayer model
        The tensorlayer network.
    loss_fn : function
        Objective function

    Methods
    ---------
    forward()
        Model inference.

    Examples
    --------
    >>> import tensorlayerx as tlx
    >>> net = vgg16()
    >>> loss_fn = tlx.losses.softmax_cross_entropy_with_logits
    >>> net_with_loss = tlx.model.WithLoss(net, loss_fn)

    """

    def __init__(self, backbone, loss_fn):
        super(WithLoss, self).__init__()
        self._backbone = backbone
        self._loss_fn = loss_fn

    def forward(self, data, label):
        out = self._backbone(data)
        return self._loss_fn(out, label)

    @property
    def backbone_network(self):
        return self._backbone

class GradWrap(Module):
    """ GradWrap definition """

    def __init__(self, network, trainable_weights):
        super(GradWrap, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(trainable_weights)

    def forward(self, x, label):
        return composite.GradOperation(get_by_list=True)(self.network, self.weights)(x, label)


class WithGradMS(Module):
    "Module that returns the gradients."

    def __init__(self, network, loss_fn=None, sens=None, optimizer=None):
        super(WithGradMS, self).__init__()
        self.network = network
        self.loss_fn = loss_fn
        self.weights = ParameterTuple(network.trainable_weights)
        self.grad = composite.GradOperation(get_by_list=True, sens_param=(sens is not None))
        self.sens = sens
        self.optimizer = optimizer
        if self.loss_fn is None:
            self.network_with_loss = network
        else:
            self.network_with_loss = WithLoss(self.network, self.loss_fn)
        self.network.set_train()

    def forward(self, inputs, label):
        grads = self.grad(self.network_with_loss, self.weights)(inputs, label)
        return grads


class WithGradTF(object):

    def __init__(self, network, loss_fn=None, optimizer=None):
        self.network = network
        self.loss_fn = loss_fn
        self.train_weights = self.network.trainable_weights
        self.optimizer = optimizer
        if loss_fn is None:
            self.network_with_loss = network
        else:
            self.network_with_loss = WithLoss(self.network, self.loss_fn)
        self.network.set_train()

    def __call__(self, inputs, label):
        with tf.GradientTape() as tape:
            loss = self.network_with_loss(inputs, label)
        grads = tape.gradient(loss, self.train_weights)
        return grads


class WithGradPD(object):

    def __init__(self, network, loss_fn=None, optimizer=None):
        self.network = network
        self.loss_fn = loss_fn
        self.train_weights = self.network.trainable_weights
        self.optimizer = optimizer
        if loss_fn is None:
            self.network_with_loss = network
        else:
            self.network_with_loss = WithLoss(self.network, self.loss_fn)
        self.network.set_train()

    def __call__(self, inputs, label):
        loss = self.network_with_loss(inputs, label)
        grads = self.optimizer.gradient(loss, self.train_weights)
        return grads


class TrainOneStepWithTF(object):

    def __init__(self, net_with_loss, optimizer, train_weights):
        self.net_with_loss = net_with_loss
        self.optimzer = optimizer
        self.train_weights = train_weights

    def __call__(self, data, label):
        with tf.GradientTape() as tape:
            loss = self.net_with_loss(data, label)
        grad = tape.gradient(loss, self.train_weights)
        self.optimzer.apply_gradients(zip(grad, self.train_weights))
        return loss


class TrainOneStepWithMS(object):

    def __init__(self, net_with_loss, optimizer, train_weights):
        self.net_with_loss = net_with_loss
        self.optimizer = optimizer
        self.train_weights = train_weights
        self.net_with_loss = net_with_loss
        self.train_network = GradWrap(net_with_loss, train_weights)

    def __call__(self, data, label):
        loss = self.net_with_loss(data, label)
        grads = self.train_network(data, label)
        self.optimizer.apply_gradients(zip(grads, self.train_weights))
        loss = loss.asnumpy()
        return loss


class TrainOneStepWithPD(object):

    def __init__(self, net_with_loss, optimizer, train_weights):
        self.net_with_loss = net_with_loss
        self.optimizer = optimizer
        self.train_weights = train_weights

    def __call__(self, data, label):
        loss = self.net_with_loss(data, label)
        grads = self.optimizer.gradient(loss, self.train_weights)
        self.optimizer.apply_gradients(zip(grads, self.train_weights))
        return loss.numpy()


class TrainOneStepWithTH(object):

    def __init__(self, net_with_loss, optimizer, train_weights):
        self.net_with_loss = net_with_loss
        self.optimizer = optimizer
        self.train_weights = train_weights

    def __call__(self, data, label):
        loss = self.net_with_loss(data, label)
        grads = self.optimizer.gradient(loss, self.train_weights)
        self.optimizer.apply_gradients(zip(grads, self.train_weights))
        return loss