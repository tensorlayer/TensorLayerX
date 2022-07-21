#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayerx as tlx
import numpy as np
from tensorlayerx import logging
from tensorlayerx.nn.core import Module

__all__ = ['MaskedConv3d']


class MaskedConv3d(Module):
    """ MaskedConv3D.
        Reference:
        [1] Nguyen D T ,  Quach M ,  Valenzise G , et al.
        Lossless Coding of Point Cloud Geometry using a Deep Generative Model[J].
        IEEE Transactions on Circuits and Systems for Video Technology, 2021, PP(99):1-1.

        Parameters
        ----------
        mask_type : str
            The mask type('A', 'B')
        out_channels : int
            The number of filters.
        kernel_size : tuple or int
            The filter size (height, width).
        stride : tuple or int
            The sliding window stride of corresponding input dimensions.
            It must be in the same order as the ``shape`` parameter.
        dilation : tuple or int
            Specifying the dilation rate to use for dilated convolution.
        act : activation function
            The activation function of this layer.
        padding : str
            The padding algorithm type: "SAME" or "VALID".
        data_format : str
            "channels_last" (NDHWC, default) or "channels_first" (NCDHW).
        kernel_initializer : initializer or str
            The initializer for the the weight matrix.
        bias_initializer : initializer or None or str
            The initializer for the the bias vector. If None, skip biases.
        in_channels : int
            The number of in channels.
        name : None or str
            A unique layer name.

        Examples
        --------
        With TensorLayer

        >>> net = tlx.nn.Input([8, 20, 20, 20, 3], name='input')
        >>> conv3d = tlx.nn.MaskedConv3d(mask_type='A', out_channels=32, kernel_size=(3, 3, 3), stride=(2, 2, 2), bias_initializer=None, in_channels=3, name='conv3d_1')
        >>> print(conv3d)
        >>> tensor = tlx.nn.MaskedConv3d(mask_type='B', out_channels=32, kernel_size=(3, 3, 3), stride=(2, 2, 2), act=tlx.ReLU, name='conv3d_2')(net)
        >>> print(tensor)

        """

    def __init__(
        self,
        mask_type,
        out_channels,
        kernel_size=(3, 3, 3),
        stride=(1, 1, 1),
        dilation=(1, 1, 1),
        padding='SAME',
        act=None,
        in_channels=None,
        data_format='channels_last',
        kernel_initializer='he_normal',
        bias_initializer='zeros',
        name=None
    ):
        super(MaskedConv3d, self).__init__(name, act)

        assert mask_type in {'A', 'B'}
        self.mask_type = mask_type

        self.out_channels = out_channels
        self.kernel_size = self.check_param(kernel_size, '3d')
        self.stride = self.check_param(stride, '3d')
        self.dilation = self.check_param(dilation, '3d')
        self.padding = padding
        self.kernel_initializer = self.str_to_init(kernel_initializer)
        self.bias_initializer = self.str_to_init(bias_initializer)
        self.in_channels = in_channels
        self.data_format = data_format

        if self.in_channels:
            self.build(None)
            self._built = True

        logging.info(
            "MaskedConv3D  %s: out_channels: %d kernel_size: %s stride: %s mask_type: %s act: %s" % (
                self.name, out_channels, str(kernel_size), str(stride), mask_type,
                self.act.__class__.__name__ if self.act is not None else 'No Activation'
            )
        )

    def __repr__(self):
        actstr = self.act.__class__.__name__ if self.act is not None else 'No Activation'
        s = (
            '{classname}(in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}'
            ', stride={stride}, padding={padding}'
        )
        if self.bias_initializer is None:
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
            self._strides = [1, self.stride[0], self.stride[1], self.stride[2], 1]
            self._dilation_rate = [1, self.dilation[0], self.dilation[1], self.dilation[2], 1]
        elif self.data_format == 'channels_first':
            if self.in_channels is None:
                self.in_channels = inputs_shape[1]
            self._strides = [1, 1, self.stride[0], self.stride[1], self.stride[2]]
            self._dilation_rate = [1, 1, self.dilation[0], self.dilation[1], self.dilation[2]]
        else:
            raise Exception("data_format should be either channels_last or channels_first")

        self.filter_shape = (
            self.kernel_size[0], self.kernel_size[1], self.kernel_size[2], self.in_channels, self.out_channels
        )

        self.kernel = self._get_weights('kernel', shape=self.filter_shape, init=self.kernel_initializer)

        self.b_init = False
        if self.bias_initializer:
            self.bias = self._get_weights('bias', shape=(self.out_channels, ), init=self.bias_initializer)
            self.bias_add = tlx.ops.BiasAdd(data_format=self.data_format)
            self.b_init = True

        center = self.kernel_size[0] // 2

        mask = np.ones(self.kernel.shape, dtype=np.float32)
        if tlx.BACKEND == 'tensorflow':
            mask[center, center, center + (self.mask_type == 'B'):, :, :] = 0.  # centre depth layer, center row
            mask[center, center + 1:, :, :, :] = 0.  # center depth layer, lower row
            mask[center + 1:, :, :, :, :] = 0.  # behind layers,all row, columns
        else:
            mask[:, :, center + (self.mask_type == 'B'):, center, center] = 0.
            mask[:, :, :, center + 1:, center] = 0.
            mask[:, :, :, :, center + 1:] = 0

        self.mask = tlx.ops.convert_to_tensor(mask, tlx.float32)

        self.conv3d = tlx.ops.Conv3D(
            strides=self._strides, padding=self.padding, data_format=self.data_format, dilations=self._dilation_rate,
            out_channel=self.out_channels, k_size=self.kernel_size
        )
        self.act_init_flag = False
        if self.act:
            self.act_init_flag = True

    def forward(self, inputs):  #input：[batch, in_depth, in_height, in_width, in_channels]
        if self._forward_state == False:
            if self._built == False:
                self.build(tlx.get_tensor_shape(inputs))
                self._built = True
            self._forward_state = True

        self.masked_kernel = tlx.ops.multiply(self.mask, self.kernel)
        x = self.conv3d(inputs, self.masked_kernel)
        if self.b_init:
            x = self.bias_add(x, self.bias)
        if self.act_init_flag:
            x = self.act(x)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, x)
            self._nodes_fixed = True
        return x
