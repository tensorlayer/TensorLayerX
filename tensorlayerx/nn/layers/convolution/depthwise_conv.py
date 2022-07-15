#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayerx as tlx
from tensorlayerx import logging
from tensorlayerx.nn.core import Module

__all__ = [
    'DepthwiseConv2d',
]


class DepthwiseConv2d(Module):
    """Separable/Depthwise Convolutional 2D layer, see `tf.nn.depthwise_conv2d <https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/nn/depthwise_conv2d>`__.

    Input:
        4-D Tensor (batch, height, width, in_channels).
    Output:
        4-D Tensor (batch, new height, new width, in_channels * depth_multiplier).

    Parameters
    ------------
    kernel_size : tuple or int
        The filter size (height, width).
    stride : tuple or int
        The stride step (height, width).
    act : activation function
        The activation function of this layer.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    data_format : str
        "channels_last" (NHWC, default) or "channels_first" (NCHW).
    dilation: tuple or int
        The dilation rate in which we sample input values across the height and width dimensions in atrous convolution. If it is greater than 1, then all values of strides must be 1.
    depth_multiplier : int
        The number of channels to expand to.
    W_init : initializer or str
        The initializer for the weight matrix.
    b_init : initializer or None or str
        The initializer for the bias vector. If None, skip bias.
    in_channels : int
        The number of in channels.
    name : str
        A unique layer name.

    Examples
    ---------
    With TensorLayer

    >>> net = tlx.nn.Input([8, 200, 200, 32], name='input')
    >>> depthwiseconv2d = tlx.nn.DepthwiseConv2d(
    ...     kernel_size=(3, 3), stride=(1, 1), dilation=(2, 2), act=tlx.ReLU, depth_multiplier=2, name='depthwise'
    ... )(net)
    >>> print(depthwiseconv2d)
    >>> output shape : (8, 200, 200, 64)


    References
    -----------
    - tflearn's `grouped_conv_2d <https://github.com/tflearn/tflearn/blob/3e0c3298ff508394f3ef191bcd7d732eb8860b2e/tflearn/layers/conv.py>`__
    - keras's `separableconv2d <https://keras.io/layers/convolutional/#separableconv2d>`__

    """

    # https://zhuanlan.zhihu.com/p/31551004  https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/CNNs/MobileNet.py
    def __init__(
        self,
        kernel_size=(3, 3),
        stride=(1, 1),
        act=None,
        padding='SAME',
        data_format='channels_last',
        dilation=(1, 1),
        depth_multiplier=1,
        W_init='truncated_normal',
        b_init='constant',
        in_channels=None,
        name=None  # 'depthwise_conv2d'
    ):
        super().__init__(name, act=act)
        self.kernel_size = self.check_param(kernel_size)
        self.stride = self.check_param(stride)
        self.padding = padding
        self.dilation = self.check_param(dilation)
        self.data_format = data_format
        self.depth_multiplier = depth_multiplier
        self.W_init = self.str_to_init(W_init)
        self.b_init = self.str_to_init(b_init)
        self.in_channels = in_channels

        if self.in_channels:
            self.build(None)
            self._built = True

        logging.info(
            "DepthwiseConv2d %s: kernel_size: %s strides: %s pad: %s act: %s" % (
                self.name, str(kernel_size), str(stride), padding,
                self.act.__class__.__name__ if self.act is not None else 'No Activation'
            )
        )

    def __repr__(self):
        actstr = self.act.__class__.__name__ if self.act is not None else 'No Activation'
        s = (
            '{classname}(in_channels={in_channels}, out_channels={n_filter}, kernel_size={kernel_size}'
            ', stride={stride}, padding={padding}'
        )
        if self.dilation != (1, ) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.b_init is None:
            s += ', bias=False'
        s += (', ' + actstr)
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(
            classname=self.__class__.__name__, n_filter=self.in_channels * self.depth_multiplier, **self.__dict__
        )

    def build(self, inputs_shape):
        if self.data_format == 'channels_last':
            if self.in_channels is None:
                self.in_channels = inputs_shape[-1]
        elif self.data_format == 'channels_first':
            if self.in_channels is None:
                self.in_channels = inputs_shape[1]
        else:
            raise Exception("data_format should be either channels_last or channels_first")

        self.filter_depthwise = (self.kernel_size[0], self.kernel_size[1], 1, self.in_channels)
        self.filter_pointwise = (1, 1, self.in_channels, self.in_channels * self.depth_multiplier)

        self.filters = self._get_weights("filters", shape=self.filter_depthwise, init=self.W_init)
        self.point_filter = self._get_weights("point_filter", shape=self.filter_pointwise, init=self.W_init)

        self.depthwise_conv2d = tlx.ops.DepthwiseConv2d(
            strides=self.stride, padding=self.padding, data_format=self.data_format, dilations=self.dilation,
            ksize=self.kernel_size, channel_multiplier=self.depth_multiplier, in_channels=self.in_channels
        )

        self.b_init_flag = False
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.in_channels * self.depth_multiplier, ), init=self.b_init)
            self.bias_add = tlx.ops.BiasAdd(self.data_format)
            self.b_init_flag = True

        self.act_init_flag = False
        if self.act:
            self.act_init_flag = True

    def forward(self, inputs):
        if self._forward_state == False:
            if self._built == False:
                self.build(tlx.get_tensor_shape(inputs))
                self._built = True
            self._forward_state = True

        outputs = self.depthwise_conv2d(input=inputs, filter=self.filters, point_filter=self.point_filter)
        if self.b_init_flag:
            outputs = self.bias_add(outputs, self.b)
        if self.act_init_flag:
            outputs = self.act(outputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs
