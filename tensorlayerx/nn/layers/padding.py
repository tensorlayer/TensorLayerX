#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayerx as tlx
from tensorlayerx import logging
from tensorlayerx.nn.core import Module

__all__ = [
    'PadLayer',
    'ZeroPad1d',
    'ZeroPad2d',
    'ZeroPad3d',
]


class PadLayer(Module):
    """The :class:`PadLayer` class is a padding layer for any mode and dimension.
    Please see `tf.pad <https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/pad>`__ for usage.

    Parameters
    ----------
    padding : list of lists of 2 ints, or a Tensor of type int32.
        The int32 values to pad.
    mode : str
        "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive).
    name : None or str
        A unique layer name.

    Examples
    --------
    With TensorLayer

    >>> net = tlx.nn.Input([10, 224, 224, 3], name='input')
    >>> padlayer = tlx.nn.PadLayer([[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT", name='inpad')(net)
    >>> print(padlayer)
    >>> output shape : (10, 230, 230, 3)

    """

    def __init__(
        self,
        padding=None,
        mode='CONSTANT',
        constant_values=0,
        name=None,  # 'pad_layer',
    ):
        super().__init__(name)
        self.padding = padding
        self.mode = mode
        self.constant_values = constant_values

        logging.info("PadLayer   %s: padding: %s mode: %s" % (self.name, self.padding, self.mode))

        if self.padding is None:
            raise Exception(
                "padding should be a Tensor of type int32. see https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/pad"
            )

        self.build()
        self._built = True

    def __repr__(self):
        s = '{classname}(padding={padding}, mode={mode}'
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        self.pad = tlx.ops.Pad(paddings=self.padding, mode=self.mode, constant_values=self.constant_values)

    def forward(self, inputs):
        outputs = self.pad(inputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs


class ZeroPad1d(Module):
    """
    The :class:`ZeroPad1d` class is a 1D padding layer for signal [batch, length, channel].

    Parameters
    ----------
    padding : tuple of 2 ints
            - If tuple of 2 ints, zeros to add at the beginning and at the end of the padding dimension.
    name : None or str
        A unique layer name.

    Examples
    --------
    With TensorLayer

    >>> net = tlx.nn.Input([10, 100, 1], name='input')
    >>> pad1d = tlx.nn.ZeroPad1d(padding=(3, 3))(net)
    >>> print(pad1d)
    >>> output shape : (10, 106, 1)

    """

    def __init__(
        self,
        padding,
        name=None,
        data_format='channels_last',
    ):
        super().__init__(name)
        self.padding = padding
        self.data_format = data_format
        logging.info("ZeroPad1d   %s: padding: %s" % (self.name, str(padding)))

        if not isinstance(self.padding, (int, tuple, dict)):
            raise AssertionError()

        self.build()
        self._built = True

    def __repr__(self):
        s = '{classname}(padding={padding}'
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        self.layer = tlx.ops.ZeroPadding1D(padding=self.padding, data_format=self.data_format)

    def forward(self, inputs):
        outputs = self.layer(inputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs


class ZeroPad2d(Module):
    """
    The :class:`ZeroPad2d` class is a 2D padding layer for image [batch, height, width, channel].

    Parameters
    ----------
    padding : tuple of 2 tuples of 2 ints.
            - If tuple of 2 tuples of 2 ints, interpreted as ``((top_pad, bottom_pad), (left_pad, right_pad))``.
    name : None or str
        A unique layer name.

    Examples
    --------
    With TensorLayer

    >>> net = tlx.nn.Input([10, 100, 100, 3], name='input')
    >>> pad2d = tlx.nn.ZeroPad2d(padding=((3, 3), (4, 4)))(net)
    >>> print(pad2d)
    >>> output shape : (10, 106, 108, 3)

    """

    def __init__(
        self,
        padding,
        name=None,
        data_format='channels_last',
    ):
        super().__init__(name)
        self.padding = padding
        self.data_format = data_format
        logging.info("ZeroPad2d   %s: padding: %s" % (self.name, str(self.padding)))

        if not isinstance(self.padding, (int, tuple)):
            raise AssertionError("Padding should be of type `int` or `tuple`")

        self.build()
        self._built = True

    def __repr__(self):
        s = '{classname}(padding={padding}'
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        self.layer = tlx.ops.ZeroPadding2D(padding=self.padding, data_format=self.data_format)

    def forward(self, inputs):
        outputs = self.layer(inputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs


class ZeroPad3d(Module):
    """
    The :class:`ZeroPad3d` class is a 3D padding layer for volume [batch, depth, height, width, channel].

    Parameters
    ----------
    padding : tuple of 2 tuples of 2 ints.
            - If tuple of 2 tuples of 2 ints, interpreted as ``((left_dim1_pad, right_dim1_pad), (left_dim2_pad, right_dim2_pad), (left_dim3_pad, right_dim3_pad))``.
    name : None or str
        A unique layer name.

    Examples
    --------
    With TensorLayer

    >>> net = tlx.nn.Input([10, 100, 100, 100, 3], name='input')
    >>> pad3d = tlx.nn.ZeroPad3d(padding=((3, 3), (4, 4), (5, 5)))(net)
    >>> print(pad3d)
    >>> output shape : (10, 106, 108, 110, 3)

    """

    def __init__(
        self,
        padding,
        name=None,
        data_format='channels_last',
    ):
        super().__init__(name)
        self.padding = padding
        self.data_format = data_format
        logging.info("ZeroPad3d   %s: padding: %s" % (self.name, str(self.padding)))

        if not isinstance(self.padding, (int, tuple)):
            raise AssertionError()

        self.build()
        self._built = True

    def __repr__(self):
        s = '{classname}(padding={padding}'
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        self.layer = tlx.ops.ZeroPadding3D(padding=self.padding, data_format=self.data_format)

    def forward(self, inputs):
        outputs = self.layer(inputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs
