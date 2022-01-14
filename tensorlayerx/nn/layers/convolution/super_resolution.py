#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayerx as tlx
from tensorlayerx import logging
from tensorlayerx.nn.core import Module

__all__ = [
    'SubpixelConv1d',
    'SubpixelConv2d',
]


class SubpixelConv1d(Module):
    """It is a 1D sub-pixel up-sampling layer.

    Calls a TensorFlow function that directly implements this functionality.
    We assume input has dim (batch, width, r)

    Parameters
    ------------
    scale : int
        The up-scaling ratio, a wrong setting will lead to Dimension size error.
    act : activation function
        The activation function of this layer.
    in_channels : int
        The number of in channels.
    name : str
        A unique layer name.

    Examples
    ----------
    With TensorLayer

    >>> net = tlx.nn.Input([8, 25, 32], name='input')
    >>> subpixelconv1d = tlx.nn.SubpixelConv1d(scale=2, name='subpixelconv1d')(net)
    >>> print(subpixelconv1d)
    >>> output shape : (8, 50, 16)

    References
    -----------
    `Audio Super Resolution Implementation <https://github.com/kuleshov/audio-super-res/blob/master/src/models/layers/subpixel.py>`__.

    """

    def __init__(
        self,
        scale=2,
        act=None,
        in_channels=None,
        name=None  # 'subpixel_conv1d'
    ):
        super().__init__(name, act=act)
        self.scale = scale
        self.in_channels = in_channels
        # self.out_channels = int(self.in_channels / self.scale)

        if self.in_channels is not None:
            self.build(None)
            self._built = True

        logging.info(
            "SubpixelConv1d  %s: scale: %d act: %s" %
            (self.name, scale, self.act.__class__.__name__ if self.act is not None else 'No Activation')
        )

    def __repr__(self):
        actstr = self.act.__class__.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(in_channels={in_channels}, out_channels={out_channels}')
        s += (', ' + actstr)
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if inputs_shape is not None:
            self.in_channels = inputs_shape[-1]
        self.out_channels = int(self.in_channels / self.scale)
        self.transpose = tlx.ops.Transpose(perm=[2, 1, 0])
        self.batch_to_space = tlx.ops.BatchToSpace(block_size=[self.scale], crops=[[0, 0]])

    def forward(self, inputs):
        if self._forward_state == False:
            if self._built == False:
                self.build(tlx.get_tensor_shape(inputs))
                self._built = True
            self._forward_state = True

        outputs = self._PS(inputs)
        if self.act is not None:
            outputs = self.act(outputs)
        return outputs

    def _PS(self, I):
        X = self.transpose(I)  # (r, w, b)
        X = self.batch_to_space(X)  # (1, r*w, b)
        X = self.transpose(X)
        return X


class SubpixelConv2d(Module):
    """It is a 2D sub-pixel up-sampling layer, usually be used
    for Super-Resolution applications, see `SRGAN <https://github.com/tensorlayer/srgan/>`__ for example.

    Parameters
    ------------
    scale : int
        factor to increase spatial resolution.
    data_format : str
        "channels_last" (NHWC, default) or "channels_first" (NCHW).
    act : activation function
        The activation function of this layer.
    name : str
        A unique layer name.

    Examples
    ---------
    With TensorLayer

    >>> net = tlx.nn.Input([2, 16, 16, 4], name='input1')
    >>> subpixelconv2d = tlx.nn.SubpixelConv2d(scale=2, data_format='channels_last', name='subpixel_conv2d1')(net)
    >>> print(subpixelconv2d)
    >>> output shape : (2, 32, 32, 1)

    >>> net = tlx.nn.Input([2, 16, 16, 40], name='input2')
    >>> subpixelconv2d = tlx.nn.SubpixelConv2d(scale=2, data_format='channels_last', name='subpixel_conv2d2')(net)
    >>> print(subpixelconv2d)
    >>> output shape : (2, 32, 32, 10)

    >>> net = tlx.nn.Input([2, 16, 16, 250], name='input3')
    >>> subpixelconv2d = tlx.nn.SubpixelConv2d(scale=5, data_format='channels_last', name='subpixel_conv2d3')(net)
    >>> print(subpixelconv2d)
    >>> output shape : (2, 80, 80, 10)

    References
    ------------
    - `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network <https://arxiv.org/pdf/1609.05158.pdf>`__

    """

    # github/Tetrachrome/subpixel  https://github.com/Tetrachrome/subpixel/blob/master/subpixel.py
    def __init__(
        self,
        scale=2,
        data_format='channels_last',
        act=None,
        name=None  # 'subpixel_conv2d'
    ):
        super().__init__(name, act=act)
        self.scale = scale
        self.data_format = data_format
        self.build(None)
        self._built = True
        logging.info(
            "SubpixelConv2d  %s: scale: %d act: %s" %
            (self.name, scale, self.act.__class__.__name__ if self.act is not None else 'No Activation')
        )

    def __repr__(self):
        actstr = self.act.__class__.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(scale={scale})')
        s += (', ' + actstr)
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):

        self.depth_to_space = tlx.ops.DepthToSpace(block_size=self.scale, data_format=self.data_format)

    def forward(self, inputs):
        if self._forward_state == False:
            if self._built == False:
                self.build(tlx.get_tensor_shape(inputs))
                self._built = True
            self._forward_state = True

        outputs = self.depth_to_space(inputs)
        if self.act is not None:
            outputs = self.act(outputs)
        return outputs
