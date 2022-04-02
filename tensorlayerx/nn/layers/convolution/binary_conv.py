#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayerx as tlx
from tensorlayerx import logging
from tensorlayerx.nn.core import Module

__all__ = [
    'BinaryConv2d',
]


class BinaryConv2d(Module):
    """
    The :class:`BinaryConv2d` class is a 2D binary CNN layer, which weights are either -1 or 1 while inference.

    Note that, the bias vector would not be binarized.

    Parameters
    ----------
    out_channels : int
        The number of filters.
    kernel_size : tuple of int
        The filter size (height, width).
    strides : tuple of int
        The sliding window strides of corresponding input dimensions.
        It must be in the same order as the ``shape`` parameter.
    act : activation function
        The activation function of this layer.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    data_format : str
        "channels_last" (NHWC, default) or "channels_first" (NCHW).
    dilation_rate : tuple of int
        Specifying the dilation rate to use for dilated convolution.
    W_init : initializer or str
        The initializer for the the weight matrix.
    b_init : initializer or None or str
        The initializer for the the bias vector. If None, skip biases.
    in_channels : int
        The number of in channels.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayer

    >>> net = tlx.nn.Input([8, 100, 100, 32], name='input')
    >>> binaryconv2d = tlx.nn.BinaryConv2d(
        ... out_channels=64, kernel_size=(3, 3), strides=(2, 2), act=tlx.ReLU, in_channels=32, name='binaryconv2d')(net)
    >>> print(binaryconv2d)
    >>> output shape : (8, 50, 50, 64)

    """

    def __init__(
        self, out_channels=32, kernel_size=(3, 3), strides=(1, 1), act=None, padding='VALID', data_format="channels_last",
        dilation_rate=(1, 1), W_init='truncated_normal', b_init='constant', in_channels=None, name=None
    ):
        super(BinaryConv2d, self).__init__(name, act=act)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self._strides = self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self._dilation_rate = self.dilation_rate = dilation_rate
        self.W_init = self.str_to_init(W_init)
        self.b_init = self.str_to_init(b_init)
        self.in_channels = in_channels

        if self.in_channels:
            self.build(None)
            self._built = True

        logging.info(
            "BinaryConv2d %s: out_channels: %d kernel_size: %s strides: %s pad: %s act: %s" % (
                self.name, out_channels, str(kernel_size), str(strides), padding,
                self.act.__class__.__name__ if self.act is not None else 'No Activation'
            )
        )

    def __repr__(self):
        actstr = self.act.__class__.__name__ if self.act is not None else 'No Activation'
        s = (
            '{classname}(in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}'
            ', strides={strides}, padding={padding}'
        )
        if self.dilation_rate != (1, ) * len(self.dilation_rate):
            s += ', dilation={dilation_rate}'
        if self.b_init is None:
            s += ', bias=False'
        s += (', ' + actstr)
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if self.data_format == 'channels_last':
            self.data_format = 'NHWC'
            if self.in_channels is None:
                self.in_channels = inputs_shape[-1]
            self._strides = [1, self._strides[0], self._strides[1], 1]
            self._dilation_rate = [1, self._dilation_rate[0], self._dilation_rate[1], 1]
        elif self.data_format == 'channels_first':
            self.data_format = 'NCHW'
            if self.in_channels is None:
                self.in_channels = inputs_shape[1]
            self._strides = [1, 1, self._strides[0], self._strides[1]]
            self._dilation_rate = [1, 1, self._dilation_rate[0], self._dilation_rate[1]]
        else:
            raise Exception("data_format should be either channels_last or channels_first")

        self.filter_shape = (self.kernel_size[0], self.kernel_size[1], self.in_channels, self.out_channels)

        self.W = self._get_weights("filters", shape=self.filter_shape, init=self.W_init)

        self.b_init_flag = False
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.out_channels, ), init=self.b_init)
            self.bias_add = tlx.ops.BiasAdd(self.data_format)
            self.b_init_flag = True

        self.act_init_flag = False
        if self.act:
            self.act_init_flag = True

        self.binaryconv2d = tlx.ops.BinaryConv2D(
            strides=self._strides,
            padding=self.padding,
            data_format=self.data_format,
            dilations=self._dilation_rate,
            out_channel=self.out_channels,
            k_size=self.kernel_size,
            in_channel=self.in_channels,
        )

    def forward(self, inputs):
        if self._forward_state == False:
            if self._built == False:
                self.build(tlx.get_tensor_shape(inputs))
                self._built = True
            self._forward_state = True

        outputs = self.binaryconv2d(inputs, self.W)

        if self.b_init_flag:
            outputs = self.bias_add(outputs, self.b)
        if self.act_init_flag:
            outputs = self.act(outputs)
        return outputs
