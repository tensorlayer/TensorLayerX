#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayerx as tlx
from tensorlayerx import logging
from tensorlayerx.nn.core import Module

__all__ = [
    'DorefaDense',
]


class DorefaDense(Module):
    """The :class:`DorefaDense` class is a binary fully connected layer, which weights are 'bitW' bits and the output of the previous layer
    are 'bitA' bits while inferencing.

    Note that, the bias vector would not be binarized.

    Parameters
    ----------
    bitW : int
        The bits of this layer's parameter
    bitA : int
        The bits of the output of previous layer
    n_units : int
        The number of units of this layer.
    act : activation function
        The activation function of this layer, usually set to ``tf.act.sign`` or apply :class:`Sign` after :class:`BatchNorm`.
    use_gemm : boolean
        If True, use gemm instead of ``tf.matmul`` for inferencing. (TODO).
    W_init : initializer or str
        The initializer for the weight matrix.
    b_init : initializer or None or str
        The initializer for the bias vector. If None, skip biases.
    in_channels: int
        The number of channels of the previous layer.
        If None, it will be automatically detected when the layer is forwarded for the first time.
    name : a str
        A unique layer name.

    Examples
    --------
    >>> net = tlx.nn.Input([10, 784], name='input')
    >>> net = tlx.nn.DorefaDense(n_units=800, act=tlx.ReLU, name='relu1')(net)
    >>> output shape :(10, 800)
    >>> net = tlx.nn.DorefaDense(n_units=10, name='output')(net)
    >>> output shape :(10, 10)

    """

    def __init__(
        self,
        bitW=1,
        bitA=3,
        n_units=100,
        act=None,
        use_gemm=False,
        W_init='truncated_normal',
        b_init='constant',
        in_channels=None,
        name=None,  #'dorefa_dense',
    ):
        super().__init__(name, act=act)
        self.bitW = bitW
        self.bitA = bitA
        self.n_units = n_units
        self.use_gemm = use_gemm
        self.W_init = self.str_to_init(W_init)
        self.b_init = self.str_to_init(b_init)
        self.in_channels = in_channels

        if self.in_channels is not None:
            self.build((None, self.in_channels))
            self._built = True

        logging.info(
            "DorefaDense  %s: %d %s" %
            (self.name, n_units, self.act.__class__.__name__ if self.act is not None else 'No Activation')
        )

    def __repr__(self):
        actstr = self.act.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(n_units={n_units}, ' + actstr)
        s += ', bitW={bitW}, bitA={bitA}'
        if self.in_channels is not None:
            s += ', in_channels=\'{in_channels}\''
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if len(inputs_shape) != 2:
            raise Exception("The input dimension must be rank 2, please reshape or flatten it")

        if self.in_channels is None:
            self.in_channels = inputs_shape[1]

        if self.use_gemm:
            raise Exception("TODO. The current version use tf.matmul for inferencing.")

        n_in = inputs_shape[-1]
        self.b = None
        self.W = self._get_weights("weights", shape=(n_in, self.n_units), init=self.W_init)
        if self.b_init is not None:
            self.b = self._get_weights("biases", shape=(self.n_units), init=self.b_init)
            self.bias_add = tlx.ops.BiasAdd()

        self.dorefa_dense = tlx.ops.DorefaDense(self.W, self.b, self.bitW, self.bitA)

    def forward(self, inputs):
        if self._forward_state == False:
            if self._built == False:
                self.build(tlx.get_tensor_shape(inputs))
                self._built = True
            self._forward_state = True

        outputs = self.dorefa_dense(inputs)

        if self.act:
            outputs = self.act(outputs)
        return outputs