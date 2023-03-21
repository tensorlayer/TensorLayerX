#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayerx as tlx
from tensorlayerx import logging
from tensorlayerx.nn.core import Module

__all__ = [
    'GroupConv2d',
]


class GroupConv2d(Module):
    """The :class:`GroupConv2d` class is 2D grouped convolution, see `here <https://blog.yani.io/filter-group-tutorial/>`__.

      Parameters
      --------------
      out_channels : int
          The number of filters.
      kernel_size : tuple or int
          The filter size.
      stride : tuple or int
          The stride step.
      n_group : int
          The number of groups.
      act : activation function
          The activation function of this layer.
      padding : str
          The padding algorithm type: "SAME" or "VALID".
      data_format : str
          "channels_last" (NHWC, default) or "channels_first" (NCHW).
      dilation : tuple or int
          Specifying the dilation rate to use for dilated convolution.
      W_init : initializer or str
          The initializer for the weight matrix.
      b_init : initializer or None or str
          The initializer for the bias vector. If None, skip biases.
      in_channels : int
          The number of in channels.
      name : None or str
          A unique layer name.

      Examples
      ---------
      With TensorLayer

      >>> net = tlx.nn.Input([8, 24, 24, 32], name='input')
      >>> groupconv2d = tlx.nn.GroupConv2d(
      ...     out_channels=64, kernel_size=(3, 3), stride=(2, 2), n_group=2, name='group'
      ... )(net)
      >>> print(groupconv2d)
      >>> output shape : (8, 12, 12, 64)

      """

    def __init__(
        self,
        out_channels=32,
        kernel_size=(1, 1),
        stride=(1, 1),
        n_group=1,
        act=None,
        padding='SAME',
        data_format="channels_last",
        dilation=(1, 1),
        W_init='truncated_normal',
        b_init='constant',
        in_channels=None,
        name=None
    ):
        super().__init__(name, act=act)
        self.out_channels = out_channels
        self.kernel_size = self.check_param(kernel_size)
        self._stride = self.stride = self.check_param(stride)
        self.n_group = n_group
        self.padding = padding
        self.data_format = data_format
        self._dilation_rate = self.dilation = self.check_param(dilation)
        self.W_init = self.str_to_init(W_init)
        self.b_init = self.str_to_init(b_init)
        self.in_channels = in_channels

        if self.in_channels:
            self.build(None)
            self._built = True

        logging.info(
            "Conv2d %s: out_channels: %d kernel_size: %s stride: %s n_group: %d pad: %s  act: %s" % (
                self.name, out_channels, str(kernel_size), str(stride), n_group, padding,
                self.act.__class__.__name__ if self.act is not None else 'No Activation'
            )
        )

    def __repr__(self):
        actstr = self.act.__class__.__name__ if self.act is not None else "No Activation"
        s = (
            '{classname}(in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}'
            ', stride={stride}, n_group = {n_group}, padding={padding}'
        )
        if self.dilation != (1, ) * len(self.dilation):
            s += ', dilation = {dilation}'
        if self.b_init is None:
            s += ', bias=False'
        s += (', ' + actstr)
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if self.data_format == 'channels_last':
            if self.in_channels is None:
                self.in_channels = inputs_shape[-1]
            self._stride = [1, self._stride[0], self._stride[1], 1]
            self._dilation_rate = [1, self._dilation_rate[0], self._dilation_rate[1], 1]
        elif self.data_format == 'channels_first':
            if self.in_channels is None:
                self.in_channels = inputs_shape[1]
            self._stride = [1, 1, self._stride[0], self._stride[1]]
            self._dilation_rate = [1, 1, self._dilation_rate[0], self._dilation_rate[1]]
        else:
            raise Exception("data_format should be either channels_last or channels_first")

        if self.n_group < 1:
            raise ValueError(
                "The n_group must be a integer greater than or equal to 1, but we got :{}".format(self.n_group)
            )

        if self.in_channels % self.n_group != 0:
            raise ValueError(
                "The channels of input must be divisible by n_group, but we got: the channels of input"
                "is {}, the n_group is {}.".format(self.in_channels, self.n_group)
            )

        if self.out_channels % self.n_group != 0:
            raise ValueError(
                "The number of filters must be divisible by n_group, but we got: the number of filters "
                "is {}, the n_group is {}. ".format(self.out_channels, self.n_group)
            )

        # TODO channels first filter shape [out_channel, in_channel/n_group, filter_h, filter_w]
        self.filter_shape = (
            self.kernel_size[0], self.kernel_size[1], int(self.in_channels / self.n_group), self.out_channels
        )

        self.filters = self._get_weights("filters", shape=self.filter_shape, init=self.W_init)

        self.b_init_flag = False
        if self.b_init:
            self.biases = self._get_weights("biases", shape=(self.out_channels, ), init=self.b_init)
            self.bias_add = tlx.ops.BiasAdd(self.data_format)
            self.b_init_flag = True

        self.group_conv2d = tlx.ops.GroupConv2D(
            strides=self._stride, padding=self.padding, data_format=self.data_format, dilations=self._dilation_rate,
            out_channel=self.out_channels, k_size=(self.kernel_size[0], self.kernel_size[1]), groups=self.n_group
        )

        self.act_init_flag = False
        if self.act:
            self.act_init_flag = True

    def forward(self, inputs):
        if self._forward_state == False:
            if self._built == False:
                self.build(tlx.get_tensor_shape(inputs))
                self._built = True
            self._forward_state = True

        outputs = self.group_conv2d(inputs, self.filters)
        if self.b_init_flag:
            outputs = self.bias_add(outputs, self.biases)
        if self.act_init_flag:
            outputs = self.act(outputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs
