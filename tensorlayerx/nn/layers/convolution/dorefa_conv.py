#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayerx as tlx
from tensorlayerx import logging
from tensorlayerx.nn.core import Module

__all__ = [
    'DorefaConv2d',
]


class DorefaConv2d(Module):
    """The :class:`DorefaConv2d` class is a 2D quantized convolutional layer, which weights are 'bitW' bits and the output of the previous layer
    are 'bitA' bits while inferencing.

    Note that, the bias vector would not be binarized.

    Parameters
    ----------
    bitW : int
        The bits of this layer's parameter
    bitA : int
        The bits of the output of previous layer
    out_channels : int
        The number of filters.
    kernel_size : tuple or int
        The filter size (height, width).
    stride : tuple or int
        The sliding window strides of corresponding input dimensions.
        It must be in the same order as the ``shape`` parameter.
    act : activation function
        The activation function of this layer.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    data_format : str
        "channels_last" (NHWC, default) or "channels_first" (NCHW).
    dilation : tuple or int
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

    >>> net = tlx.nn.Input([8, 12, 12, 32], name='input')
    >>> dorefaconv2d = tlx.nn.DorefaConv2d(
    ...     out_channels=32, kernel_size=(5, 5), stride=(1, 1), act=tlx.ReLU, padding='SAME', name='dorefaconv2d'
    ... )(net)
    >>> print(dorefaconv2d)
    >>> output shape : (8, 12, 12, 32)

    """

    def __init__(
        self,
        bitW=1,
        bitA=3,
        out_channels=32,
        kernel_size=(3, 3),
        stride=(1, 1),
        act=None,
        padding='SAME',
        data_format="channels_last",
        dilation=(1, 1),
        W_init='truncated_normal',
        b_init='constant',
        in_channels=None,
        name=None  # 'dorefa_cnn2d',
    ):
        super().__init__(name, act=act)
        self.bitW = bitW
        self.bitA = bitA
        self.out_channels = out_channels
        self.kernel_size = self.check_param(kernel_size)
        self.stride = self._stride = self.check_param(stride)
        self.padding = padding
        self.data_format = data_format
        self.dilation = self._dilation_rate = self.check_param(dilation)
        self.W_init = self.str_to_init(W_init)
        self.b_init = self.str_to_init(b_init)
        self.in_channels = in_channels

        if self.in_channels:
            self.build(None)
            self._built = True

        logging.info(
            "DorefaConv2d %s: out_channels: %d kernel_size: %s strides: %s pad: %s act: %s" % (
                self.name, out_channels, str(kernel_size), str(stride), padding,
                self.act.__class__.__name__ if self.act is not None else 'No Activation'
            )
        )

    def __repr__(self):
        actstr = self.act.__class__.__name__ if self.act is not None else 'No Activation'
        s = (
            '{classname}(in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}'
            ', strides={strides}, padding={padding}'
        )
        if self.dilation != (1, ) * len(self.dilation):
            s += ', dilation={dilation}'
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
            self._stride = [1, self._stride[0], self._stride[1], 1]
            self._dilation_rate = [1, self._dilation_rate[0], self._dilation_rate[1], 1]
        elif self.data_format == 'channels_first':
            self.data_format = 'NCHW'
            if self.in_channels is None:
                self.in_channels = inputs_shape[1]
            self._stride = [1, 1, self._stride[0], self._stride[1]]
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

        self.dorefaconv2d = tlx.ops.DorefaConv2D(
            bitW=self.bitW, bitA=self.bitA, strides=self._stride, padding=self.padding, data_format=self.data_format,
            dilations=self._dilation_rate, out_channel=self.out_channels, k_size=self.kernel_size,
            in_channel=self.in_channels
        )

    def forward(self, inputs):

        if self._forward_state == False:
            if self._built == False:
                self.build(tlx.get_tensor_shape(inputs))
                self._built = True
            self._forward_state = True

        outputs = self.dorefaconv2d(inputs, self.W)

        if self.b_init_flag:
            outputs = self.bias_add(outputs, self.b)
        if self.act_init_flag:
            outputs = self.act(outputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs
