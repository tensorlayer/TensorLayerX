#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayerx as tlx
from tensorlayerx import logging
from tensorlayerx.nn.core import Module

__all__ = [
    'Linear',
]


class Linear(Module):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Parameters
    ----------
    out_features : int
        The number of units of this layer.
    act : activation function
        The activation function of this layer.
    W_init : initializer or str
        The initializer for the weight matrix.
    b_init : initializer or None or str
        The initializer for the bias vector. If None, skip biases.
    in_features: int
        The number of channels of the previous layer.
        If None, it will be automatically detected when the layer is forwarded for the first time.
    name : None or str
        A unique layer name. If None, a unique name will be automatically generated.

    Examples
    --------
    With TensorLayerx

    >>> net = tlx.nn.Input([100, 50], name='input')
    >>> linear = tlx.nn.Linear(out_features=800, act=tlx.ReLU, in_features=50, name='linear_1')
    >>> tensor = tlx.nn.Linear(out_features=800, act=tlx.ReLU, name='linear_2')(net)

    Notes
    -----
    If the layer input has more than two axes, it needs to be flatten by using :class:`Flatten`.

    """

    def __init__(
        self,
        out_features,
        act=None,
        W_init='truncated_normal',
        b_init='constant',
        in_features=None,
        name=None,  # 'linear',
    ):

        super(Linear, self).__init__(name, act=act)

        self.out_features = out_features
        self.W_init = self.str_to_init(W_init)
        self.b_init = self.str_to_init(b_init)
        self.in_features = in_features

        if self.in_features is not None:
            self.build(self.in_features)
            self._built = True

        logging.info(
            "Linear  %s: %d %s" %
            (self.name, self.out_features, self.act.__class__.__name__ if self.act is not None else 'No Activation')
        )

    def __repr__(self):
        actstr = self.act.__class__.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(out_features={out_features}, ' + actstr)
        if self.in_features is not None:
            s += ', in_features=\'{in_features}\''
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if self.in_features is None and len(inputs_shape) < 2:
            raise AssertionError("The dimension of input should not be less than 2")
        if self.in_features:
            shape = [self.in_features, self.out_features]
        else:
            self.in_features = inputs_shape[-1]
            shape = [self.in_features, self.out_features]

        self.W = self._get_weights("weights", shape=tuple(shape), init=self.W_init)

        self.b_init_flag = False
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.out_features, ), init=self.b_init)
            self.b_init_flag = True
            self.bias_add = tlx.ops.BiasAdd(data_format='NHWC')

        self.act_init_flag = False
        if self.act:
            self.act_init_flag = True

        self.matmul = tlx.ops.MatMul()

    def forward(self, inputs):
        if self._forward_state == False:
            if self._built == False:
                self.build(tlx.get_tensor_shape(inputs))
                self._built = True
            self._forward_state = True

        z = self.matmul(inputs, self.W)
        if self.b_init_flag:
            z = self.bias_add(z, self.b)
        if self.act_init_flag:
            z = self.act(z)
        return z

