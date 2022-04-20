#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayerx as tlx
from tensorlayerx import logging
from tensorlayerx.nn.core import Module
from tensorlayerx.backend import BACKEND

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
    kernel_size : tuple of 2 int
        The filter size (height, width).
    stride : tuple of 2 int
        The stride step (height, width).
    act : activation function
        The activation function of this layer.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    data_format : str
        "channels_last" (NHWC, default) or "channels_first" (NCHW).
    dilation: tuple of 2 int
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
        self.kernel_size = kernel_size
        self.stride = self._strides = stride
        self.padding = padding
        self.dilation = self._dilation = dilation
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
        return s.format(
            classname=self.__class__.__name__, n_filter=self.in_channels * self.depth_multiplier, **self.__dict__
        )

    def build(self, inputs_shape):
        if self.data_format == 'channels_last':
            self.data_format = 'NHWC'
            if self.in_channels is None:
                self.in_channels = inputs_shape[-1]
            self._strides = [1, self._strides[0], self._strides[1], 1]
        elif self.data_format == 'channels_first':
            self.data_format = 'NCHW'
            if self.in_channels is None:
                self.in_channels = inputs_shape[1]
            self._strides = [1, 1, self._strides[0], self._strides[1]]
        else:
            raise Exception("data_format should be either channels_last or channels_first")

        self.filter_shape = (self.kernel_size[0], self.kernel_size[1], self.in_channels, self.depth_multiplier)

        # Set the size of kernel as (K1,K2), then the shape is (K,Cin,K1,K2), K must be 1.
        if BACKEND == 'mindspore':
            self.filter_shape = (self.kernel_size[0], self.kernel_size[1], self.in_channels, 1)

        if BACKEND in ['tensorflow', 'mindspore']:
            self.W = self._get_weights("filters", shape=self.filter_shape, init=self.W_init, transposed=True)
            self.point_W = None
        # TODO The number of parameters on multiple backends is not equal.
        # TODO It might be better to use deepwise convolution and pointwise convolution for other backends as well.
        if BACKEND in ['paddle', 'torch']:
            self.filter_depthwise = (self.in_channels, 1, self.kernel_size[0], self.kernel_size[1])
            self.filter_pointwise = (self.in_channels * self.depth_multiplier, self.in_channels, 1, 1)
            self.W = self._get_weights("filters", shape=self.filter_depthwise, init=self.W_init, order=True)
            self.point_W = self._get_weights("point_filter", shape=self.filter_pointwise, init=self.W_init, order=True)

        self.depthwise_conv2d = tlx.ops.DepthwiseConv2d(
            strides=self._strides, padding=self.padding, data_format=self.data_format, dilations=self._dilation,
            ksize=self.kernel_size, channel_multiplier=self.depth_multiplier
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

        outputs = self.depthwise_conv2d(input=inputs, filter=self.W, point_filter=self.point_W)
        if self.b_init_flag:
            outputs = self.bias_add(outputs, self.b)
        if self.act_init_flag:
            outputs = self.act(outputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs
