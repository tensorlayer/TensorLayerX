#! /usr/bin/python
# -*- coding: utf-8 -*-

import numbers
import tensorlayerx as tlx
from tensorlayerx import logging
from tensorlayerx.nn.core import Module

__all__ = [
    'DropconnectLinear',
]


class DropconnectLinear(Module):
    """
    The :class:`DropconnectLinear` class is :class:`Dense` with DropConnect
    behaviour which randomly removes connections between this layer and the previous
    layer according to a keeping probability.

    Parameters
    ----------
    keep : float
        The keeping probability.
        The lower the probability it is, the more activations are set to zero.
    out_features : int
        The number of units of this layer.
    act : activation function
        The activation function of this layer.
    W_init : weights initializer or str
        The initializer for the weight matrix.
    b_init : biases initializer or str
        The initializer for the bias vector.
    in_features: int
        The number of channels of the previous layer.
        If None, it will be automatically detected when the layer is forwarded for the first time.
    name : str
        A unique layer name.

    Examples
    --------
    >>> net = tlx.nn.Input([10, 784], name='input')
    >>> net = tlx.nn.DropconnectLinear(keep=0.8, out_features=800, act=tlx.ReLU, name='DropconnectLinear1')(net)
    >>> output shape :(10, 800)
    >>> net = tlx.nn.DropconnectLinear(keep=0.5, out_features=800, act=tlx.ReLU, name='DropconnectLinear2')(net)
    >>> output shape :(10, 800)
    >>> net = tlx.nn.DropconnectLinear(keep=0.5, out_features=10, name='DropconnectLinear3')(net)
    >>> output shape :(10, 10)

    References
    ----------
    - `Wan, L. (2013). Regularization of neural networks using dropconnect <http://machinelearning.wustl.edu/mlpapers/papers/icml2013_wan13>`__

    """

    def __init__(
        self,
        keep=0.5,
        out_features=100,
        act=None,
        W_init='truncated_normal',
        b_init='constant',
        in_features=None,
        name=None,  # 'dropconnect',
    ):
        super().__init__(name, act=act)

        if isinstance(keep, numbers.Real) and not (keep > 0 and keep <= 1):
            raise ValueError("keep must be a scalar tensor or a float in the " "range (0, 1], got %g" % keep)

        self.keep = keep
        self.out_features = out_features
        self.W_init = self.str_to_init(W_init)
        self.b_init = self.str_to_init(b_init)
        self.in_features = in_features

        if self.in_features is not None:
            self.build((None, self.in_features))
            self._built = True

        logging.info(
            "DropconnectLinear %s: %d %s" %
            (self.name, out_features, self.act.__class__.__name__ if self.act is not None else 'No Activation')
        )

    def __repr__(self):
        actstr = self.act.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(out_features={out_features}, ' + actstr)
        s += ', keep={keep}'
        if self.in_features is not None:
            s += ', in_features=\'{in_features}\''
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if len(inputs_shape) != 2:
            raise Exception("The input dimension must be rank 2")

        if self.in_features is None:
            self.in_features = inputs_shape[1]

        n_in = inputs_shape[-1]
        self.W = self._get_weights("weights", shape=(n_in, self.out_features), init=self.W_init)
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.out_features), init=self.b_init)

        self.dropout = tlx.ops.Dropout(p=self.keep)
        self.matmul = tlx.ops.MatMul()
        self.bias_add = tlx.ops.BiasAdd()

    def forward(self, inputs):
        if self._forward_state == False:
            if self._built == False:
                self.build(tlx.get_tensor_shape(inputs))
                self._built = True
            self._forward_state = True

        W_dropcon = self.dropout(self.W)
        outputs = self.matmul(inputs, W_dropcon)
        if self.b_init:
            outputs = self.bias_add(outputs, self.b)
        if self.act:
            outputs = self.act(outputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs
