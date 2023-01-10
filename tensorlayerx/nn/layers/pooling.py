#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayerx as tlx
from tensorlayerx import logging
from tensorlayerx.nn.core import Module

__all__ = [
    'MaxPool1d',
    'AvgPool1d',
    'MaxPool2d',
    'AvgPool2d',
    'MaxPool3d',
    'AvgPool3d',
    'GlobalMaxPool1d',
    'GlobalAvgPool1d',
    'GlobalMaxPool2d',
    'GlobalAvgPool2d',
    'GlobalMaxPool3d',
    'GlobalAvgPool3d',
    'AdaptiveAvgPool1d',
    'AdaptiveAvgPool2d',
    'AdaptiveAvgPool3d',
    'AdaptiveMaxPool1d',
    'AdaptiveMaxPool2d',
    'AdaptiveMaxPool3d',
    'CornerPool2d',
]

class MaxPool1d(Module):
    """Max pooling for 1D signal.

    Parameters
    ----------
    kernel_size : int
        Pooling window size.
    stride : int
        Stride of the pooling operation.
    padding : str or int
        The padding method: 'VALID' or 'SAME'.
    return_mask : bool
        Whether to return the max indices along with the outputs.
    data_format : str
        One of channels_last (default, [batch, length, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayerX

    >>> net = tlx.nn.Input([10, 50, 32], name='input')
    >>> net = tlx.nn.MaxPool1d(kernel_size=3, stride=2, padding='SAME', name='maxpool1d')(net)
    >>> output shape : [10, 25, 32]

    """

    def __init__(
        self,
        kernel_size=3,
        stride=2,
        padding='SAME',
        return_mask = False,
        data_format='channels_last',
        name=None  # 'maxpool1d'
    ):
        super().__init__(name)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.return_mask = return_mask
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info(
            "MaxPool1d %s: kernel_size: %s stride: %s padding: %s return_mask: %s" %
            (self.name, str(kernel_size), str(stride), str(padding), str(return_mask))
        )

    def __repr__(self):
        s = ('{classname}(kernel_size={kernel_size}' ', stride={stride}, padding={padding}, return_mask={return_mask}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        # https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/nn/pool
        self.max_pool = tlx.ops.MaxPool1d(
            ksize=self.kernel_size, strides=self.stride, padding=self.padding,
            return_mask=self.return_mask, data_format=self.data_format
        )

    def forward(self, inputs):
        outputs = self.max_pool(inputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs


class AvgPool1d(Module):
    """Avg pooling for 1D signal.

    Parameters
    ------------
    kernel_size : int
        Pooling window size.
    stride : int
        Strides of the pooling operation.
    padding : int、tuple or str
        The padding method: 'VALID' or 'SAME'.
    data_format : str
        One of channels_last (default, [batch, length, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayerX

    >>> net = tlx.nn.Input([10, 50, 32], name='input')
    >>> net = tlx.nn.AvgPool1d(kernel_size=3, stride=2, padding='SAME')(net)
    >>> output shape : [10, 25, 32]

    """

    def __init__(
        self,
        kernel_size=3,
        stride=2,
        padding='SAME',
        data_format='channels_last',
        dilation_rate=1,
        name=None  # 'Avgpool1d'
    ):
        super().__init__(name)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info(
            "AvgPool1d %s: kernel_size: %s stride: %s padding: %s" %
            (self.name, str(kernel_size), str(stride), str(padding))
        )

    def __repr__(self):
        s = ('{classname}(kernel_size={kernel_size}' ', stride={stride}, padding={padding}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        # https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/nn/pool
        self.avg_pool = tlx.ops.AvgPool1d(
            ksize=self.kernel_size, strides=self.stride, padding=self.padding, data_format=self.data_format
        )

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs


class MaxPool2d(Module):
    """Max pooling for 2D image.

    Parameters
    -----------
    kernel_size : tuple or int
        (height, width) for filter size.
    stride : tuple or int
        (height, width) for stride.
    padding : int、tuple or str
        The padding method: 'VALID' or 'SAME'.
    return_mask : bool
        Whether to return the max indices along with the outputs.
    data_format : str
        One of channels_last (default, [batch, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayerX

    >>> net = tlx.nn.Input([10, 50, 50, 32], name='input')
    >>> net = tlx.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding='SAME')(net)
    >>> output shape : [10, 25, 25, 32]

    """

    def __init__(
        self,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding='SAME',
        return_mask=False,
        data_format='channels_last',
        name=None  # 'maxpool2d'
    ):
        super().__init__(name)
        self.kernel_size = self.check_param(kernel_size)
        self.stride = self.check_param(stride)
        self.padding = padding
        self.return_mask = return_mask
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info(
            "MaxPool2d %s: kernel_size: %s stride: %s padding: %s return_mask: %s" %
            (self.name, str(kernel_size), str(stride), str(padding), str(return_mask))
        )

    def __repr__(self):
        s = ('{classname}(kernel_size={kernel_size}' ', stride={stride}, padding={padding}, return_mask={return_mask}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        self.max_pool = tlx.ops.MaxPool(
            ksize=self.kernel_size, strides=self.stride, padding=self.padding,
            return_mask=self.return_mask, data_format=self.data_format
        )

    def forward(self, inputs):
        outputs = self.max_pool(inputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs


class AvgPool2d(Module):
    """Avg pooling for 2D image [batch, height, width, channel].

    Parameters
    -----------
    kernel_size : tuple or int
        (height, width) for filter size.
    stride : tuple or int
        (height, width) for stride.
    padding : int、tuple or str
        The padding method: 'VALID' or 'SAME'.
    data_format : str
        One of channels_last (default, [batch, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayerX

    >>> net = tlx.nn.Input([10, 50, 50, 32], name='input')
    >>> net = tlx.nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2), padding='SAME')(net)
    >>> output shape : [10, 25, 25, 32]

    """

    def __init__(
        self,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding='SAME',
        data_format='channels_last',
        name=None
    ):
        super().__init__(name)
        self.kernel_size = self.check_param(kernel_size)
        self.stride = self.check_param(stride)
        self.padding = padding
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info(
            "AvgPool2d %s: kernel_size: %s stride: %s padding: %s" %
            (self.name, str(kernel_size), str(stride), str(padding))
        )

    def __repr__(self):
        s = ('{classname}(kernel_size={kernel_size}' ', stride={stride}, padding={padding}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        self.avg_pool = tlx.ops.AvgPool(
            ksize=self.kernel_size, strides=self.stride, padding=self.padding, data_format=self.data_format
        )

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs


class MaxPool3d(Module):
    """Max pooling for 3D volume.

    Parameters
    ------------
    kernel_size : tuple or int
        Pooling window size.
    stride : tuple or int
        Strides of the pooling operation.
    padding : int、tuple or str
        The padding method: 'VALID' or 'SAME'.
    return_mask : bool
        Whether to return the max indices along with the outputs.
    data_format : str
        One of channels_last (default, [batch, depth, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    Returns
    -------
    :class:`tf.Tensor`
        A max pooling 3-D layer with a output rank as 5.

    Examples
    ---------
    With TensorLayerX

    >>> net = tlx.nn.Input([10, 50, 50, 50, 32], name='input')
    >>> net = tlx.nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME')(net)
    >>> output shape : [10, 25, 25, 25, 32]

    """

    def __init__(
        self,
        kernel_size=(3, 3, 3),
        stride=(2, 2, 2),
        padding='VALID',
        return_mask = False,
        data_format='channels_last',
        name=None  # 'maxpool3d'
    ):
        super().__init__(name)
        self.kernel_size = self.check_param(kernel_size, '3d')
        self.stride = self.check_param(stride, '3d')
        self.padding = padding
        self.return_mask = return_mask
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info(
            "MaxPool3d %s: kernel_size: %s stride: %s padding: %s return_mask: %s" %
            (self.name, str(kernel_size), str(stride), str(padding), str(return_mask))
        )

    def __repr__(self):
        s = ('{classname}(kernel_size={kernel_size}' ', stride={stride}, padding={padding}, return_mask={return_mask}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        self.max_pool3d = tlx.ops.MaxPool3d(
            ksize=self.kernel_size, strides=self.stride, padding=self.padding,
            return_mask=self.return_mask, data_format=self.data_format
        )

    def forward(self, inputs):
        outputs = self.max_pool3d(inputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs


class AvgPool3d(Module):
    """Avg pooling for 3D volume.

    Parameters
    ------------
    kernel_size : tuple or int
        Pooling window size.
    stride : tuple or int
        Strides of the pooling operation.
    padding : int、tuple or str
        The padding method: 'VALID' or 'SAME'.
    data_format : str
        One of channels_last (default, [batch, depth, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    Returns
    -------
    :class:`tf.Tensor`
        A Avg pooling 3-D layer with a output rank as 5.

    Examples
    ---------
    With TensorLayerX

    >>> net = tlx.nn.Input([10, 50, 50, 50, 32], name='input')
    >>> net = tlx.nn.AvgPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME')(net)
    >>> output shape : [10, 25, 25, 25, 32]

    """

    def __init__(
        self,
        kernel_size=(3, 3, 3),
        stride=(2, 2, 2),
        padding='VALID',
        data_format='channels_last',
        name=None  # 'Avgpool3d'
    ):
        super().__init__(name)
        self.kernel_size = self.check_param(kernel_size, '3d')
        self.stride = self.check_param(stride, '3d')
        self.padding = padding
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info(
            "AvgPool3d %s: kernel_size: %s stride: %s padding: %s" %
            (self.name, str(kernel_size), str(stride), str(padding))
        )

    def __repr__(self):
        s = ('{classname}(kernel_size={kernel_size}' ', stride={stride}, padding={padding}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        self.avg_pool3d = tlx.ops.AvgPool3d(
            ksize=self.kernel_size, strides=self.stride, padding=self.padding, data_format=self.data_format
        )

    def forward(self, inputs):
        outputs = self.avg_pool3d(inputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs


class GlobalMaxPool1d(Module):
    """The :class:`GlobalMaxPool1d` class is a 1D Global Max Pooling layer.

    Parameters
    ------------
    data_format : str
        One of channels_last (default, [batch, length, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayerX

    >>> net = tlx.nn.Input([10, 100, 30], name='input')
    >>> net = tlx.nn.GlobalMaxPool1d()(net)
    >>> output shape : [10, 30]

    """

    def __init__(
        self,
        data_format="channels_last",
        name=None  # 'globalmaxpool1d'
    ):
        super().__init__(name)

        self.data_format = data_format

        self.build()
        self._built = True

        logging.info("GlobalMaxPool1d %s" % self.name)

    def __repr__(self):
        s = '{classname}('
        if self.name is not None:
            s += 'name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        if self.data_format == 'channels_last':
            self.reduce_max = tlx.ReduceMax(axis=1)
        elif self.data_format == 'channels_first':
            self.reduce_max = tlx.ReduceMax(axis=2)
        else:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )

    def forward(self, inputs):
        outputs = self.reduce_max(inputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs


class GlobalAvgPool1d(Module):
    """The :class:`GlobalAvgPool1d` class is a 1D Global Avg Pooling layer.

    Parameters
    ------------
    data_format : str
        One of channels_last (default, [batch, length, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayerX

    >>> net = tlx.nn.Input([10, 100, 30], name='input')
    >>> net = tlx.nn.GlobalAvgPool1d()(net)
    >>> output shape : [10, 30]

    """

    def __init__(
        self,
        data_format='channels_last',
        name=None  # 'globalAvgpool1d'
    ):
        super().__init__(name)
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info("GlobalAvgPool1d %s" % self.name)

    def __repr__(self):
        s = '{classname}('
        if self.name is not None:
            s += 'name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        if self.data_format == 'channels_last':
            self.reduce_mean = tlx.ReduceMean(axis=1)
        elif self.data_format == 'channels_first':
            self.reduce_mean = tlx.ReduceMean(axis=2)
        else:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )

    def forward(self, inputs):
        outputs = self.reduce_mean(inputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs


class GlobalMaxPool2d(Module):
    """The :class:`GlobalMaxPool2d` class is a 2D Global Max Pooling layer.

    Parameters
    ------------
    data_format : str
        One of channels_last (default, [batch, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayerX

    >>> net = tlx.nn.Input([10, 100, 100, 30], name='input')
    >>> net = tlx.nn.GlobalMaxPool2d()(net)
    >>> output shape : [10, 30]

    """

    def __init__(
        self,
        data_format='channels_last',
        name=None  # 'globalmaxpool2d'
    ):
        super().__init__(name)
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info("GlobalMaxPool2d %s" % self.name)

    def __repr__(self):
        s = '{classname}('
        if self.name is not None:
            s += 'name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        if self.data_format == 'channels_last':
            self.reduce_max = tlx.ReduceMax(axis=[1, 2])
        elif self.data_format == 'channels_first':
            self.reduce_max = tlx.ReduceMax(axis=[2, 3])
        else:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )

    def forward(self, inputs):
        outputs = self.reduce_max(inputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs


class GlobalAvgPool2d(Module):
    """The :class:`GlobalAvgPool2d` class is a 2D Global Avg Pooling layer.

    Parameters
    ------------
    data_format : str
        One of channels_last (default, [batch, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayerX

    >>> net = tlx.nn.Input([10, 100, 100, 30], name='input')
    >>> net = tlx.nn.GlobalAvgPool2d()(net)
    >>> output shape : [10, 30]

    """

    def __init__(
        self,
        data_format='channels_last',
        name=None  # 'globalAvgpool2d'
    ):
        super().__init__(name)

        self.data_format = data_format

        self.build()
        self._built = True

        logging.info("GlobalAvgPool2d %s" % self.name)

    def __repr__(self):
        s = '{classname}('
        if self.name is not None:
            s += 'name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        if self.data_format == 'channels_last':
            self.reduce_mean = tlx.ReduceMean(axis=[1, 2])
        elif self.data_format == 'channels_first':
            self.reduce_mean = tlx.ReduceMean(axis=[2, 3])
        else:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )

    def forward(self, inputs):
        outputs = self.reduce_mean(inputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs


class GlobalMaxPool3d(Module):
    """The :class:`GlobalMaxPool3d` class is a 3D Global Max Pooling layer.

    Parameters
    ------------
    data_format : str
        One of channels_last (default, [batch, depth, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayerX

    >>> net = tlx.nn.Input([10, 100, 100, 100, 30], name='input')
    >>> net = tlx.nn.GlobalMaxPool3d()(net)
    >>> output shape : [10, 30]

    """

    def __init__(
        self,
        data_format='channels_last',
        name=None  # 'globalmaxpool3d'
    ):
        super().__init__(name)

        self.data_format = data_format

        self.build()
        self._built = True

        logging.info("GlobalMaxPool3d %s" % self.name)

    def __repr__(self):
        s = '{classname}('
        if self.name is not None:
            s += 'name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        if self.data_format == 'channels_last':
            self.reduce_max = tlx.ReduceMax(axis=[1, 2, 3])
        elif self.data_format == 'channels_first':
            self.reduce_max = tlx.ReduceMax(axis=[2, 3, 4])
        else:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )

    def forward(self, inputs):
        outputs = self.reduce_max(inputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs


class GlobalAvgPool3d(Module):
    """The :class:`GlobalAvgPool3d` class is a 3D Global Avg Pooling layer.

    Parameters
    ------------
    data_format : str
        One of channels_last (default, [batch, depth, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayerX

    >>> net = tlx.nn.Input([10, 100, 100, 100, 30], name='input')
    >>> net = tlx.nn.GlobalAvgPool3d()(net)
    >>> output shape : [10, 30]

    """

    def __init__(
        self,
        data_format='channels_last',
        name=None  # 'globalAvgpool3d'
    ):
        super().__init__(name)
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info("GlobalAvgPool3d %s" % self.name)

    def __repr__(self):
        s = '{classname}('
        if self.name is not None:
            s += 'name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        if self.data_format == 'channels_last':
            self.reduce_mean = tlx.ReduceMean(axis=[1, 2, 3])
        elif self.data_format == 'channels_first':
            self.reduce_mean = tlx.ReduceMean(axis=[2, 3, 4])
        else:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )

    def forward(self, inputs):
        outputs = self.reduce_mean(inputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs


class CornerPool2d(Module):
    """Corner pooling for 2D image [batch, height, width, channel], see `here <https://arxiv.org/abs/1808.01244>`__.

    Parameters
    ----------
    mode : str
        TopLeft for the top left corner,
        Bottomright for the bottom right corner.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayerX

    >>> net = tlx.nn.Input([10, 32, 32, 8], name='input')
    >>> net = tlx.nn.CornerPool2d(mode='TopLeft',name='cornerpool2d')(net)
    >>> output shape : [10, 32, 32, 8]

    """

    def __init__(
        self,
        mode='TopLeft',
        name=None  # 'cornerpool2d'
    ):
        super().__init__(name)
        self.mode = mode
        self.build()
        self._built = True

        logging.info("CornerPool2d %s : mode: %s" % (self.name, str(mode)))

    def __repr__(self):
        s = ('{classname}(mode={mode}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        pass

    def forward(self, inputs):
        _, input_width, input_height, _ = tlx.get_tensor_shape(inputs)
        # input_width = inputs.shape[2]
        # input_height = inputs.shape[1]
        batch_min = tlx.reduce_min(inputs)
        if self.mode == 'TopLeft':
            temp_bottom = tlx.pad(
                inputs, tlx.constant([[0, 0], [0, input_height - 1], [0, 0], [0, 0]]), constant_values=batch_min
            )
            temp_right = tlx.pad(
                inputs, tlx.constant([[0, 0], [0, 0], [0, input_width - 1], [0, 0]]), constant_values=batch_min
            )
            temp_bottom = tlx.ops.max_pool(temp_bottom, ksize=(input_height, 1), strides=(1, 1), padding='VALID')
            temp_right = tlx.ops.max_pool(temp_right, ksize=(1, input_width), strides=(1, 1), padding='VALID')
            outputs = tlx.add(temp_bottom, temp_right)  #, name=self.name)
        elif self.mode == 'BottomRight':
            temp_top = tlx.pad(
                inputs, tlx.constant([[0, 0], [input_height - 1, 0], [0, 0], [0, 0]]), constant_values=batch_min
            )
            temp_left = tlx.pad(
                inputs, tlx.constant([[0, 0], [0, 0], [input_width - 1, 0], [0, 0]]), constant_values=batch_min
            )
            temp_top = tlx.ops.max_pool(temp_top, ksize=(input_height, 1), strides=(1, 1), padding='VALID')
            temp_left = tlx.ops.max_pool(temp_left, ksize=(1, input_width), strides=(1, 1), padding='VALID')
            outputs = tlx.add(temp_top, temp_left)
        else:
            outputs = tlx.identity(inputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs


class AdaptiveAvgPool1d(Module):
    """The :class:`AdaptiveAvgPool1d` class is a 1D Adaptive Avg Pooling layer.

    Parameters
    ------------
    output_size : int
        The target output size. It must be an integer.
    data_format : str
        One of channels_last (default, [batch,  width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayerX

    >>> net = tlx.nn.Input([10, 32, 3], name='input')
    >>> net = tlx.nn.AdaptiveAvgPool1d(output_size=16)(net)
    >>> output shape : [10, 16, 3]

    """

    def __init__(self, output_size, data_format='channels_last', name=None):
        super(AdaptiveAvgPool1d, self).__init__(name)
        self.output_size = output_size
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info("AdaptiveAvgPool1d %s: output_size: %s " % (self.name, str(output_size)))

    def __repr__(self):
        s = ('{classname}(output_size={output_size}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):

        self.adaptivemeanpool1d = tlx.ops.AdaptiveMeanPool1D(output_size=self.output_size, data_format=self.data_format)

    def forward(self, inputs):
        outputs = self.adaptivemeanpool1d(inputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs


class AdaptiveAvgPool2d(Module):
    """The :class:`AdaptiveAvgPool2d` class is a 2D Adaptive Avg Pooling layer.

    Parameters
    ------------
    output_size : int or list or  tuple
        The target output size. It cloud be an int \[int,int]\(int, int).
    data_format : str
        One of channels_last (default, [batch,  height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayerX

    >>> net = tlx.nn.Input([10,32, 32, 3], name='input')
    >>> net = tlx.nn.AdaptiveAvgPool2d(output_size=16)(net)
    >>> output shape : [10,16, 16, 3]

    """

    def __init__(self, output_size, data_format='channels_last', name=None):
        super(AdaptiveAvgPool2d, self).__init__(name)
        self.output_size = output_size
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info("AdaptiveAvgPool2d %s: output_size: %s " % (self.name, str(output_size)))

    def __repr__(self):
        s = ('{classname}(output_size={output_size}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):

        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, ) * 2

        self.adaptivemeanpool2d = tlx.ops.AdaptiveMeanPool2D(output_size=self.output_size, data_format=self.data_format)

    def forward(self, inputs):
        outputs = self.adaptivemeanpool2d(inputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs


class AdaptiveAvgPool3d(Module):
    """The :class:`AdaptiveAvgPool3d` class is a 3D Adaptive Avg Pooling layer.

        Parameters
        ------------
        output_size : int or list or  tuple
            The target output size. It cloud be an int \[int,int,int]\(int, int, int).
        data_format : str
            One of channels_last (default, [batch,  depth, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
        name : None or str
            A unique layer name.

        Examples
        ---------
        With TensorLayerX

        >>> net = tlx.nn.Input([10,32, 32, 32, 3], name='input')
        >>> net = tlx.nn.AdaptiveAvgPool3d(output_size=16)(net)
        >>> output shape : [10, 16, 16, 16, 3]

        """

    def __init__(self, output_size, data_format='channels_last', name=None):
        super(AdaptiveAvgPool3d, self).__init__(name)
        self.output_size = output_size
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info("AdaptiveAvgPool3d %s: output_size: %s " % (self.name, str(output_size)))

    def __repr__(self):
        s = ('{classname}(output_size={output_size}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):

        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, ) * 3

        self.adaptivemeanpool3d = tlx.ops.AdaptiveMeanPool3D(output_size=self.output_size, data_format=self.data_format)

    def forward(self, inputs):
        outputs = self.adaptivemeanpool3d(inputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs


class AdaptiveMaxPool1d(Module):
    """The :class:`AdaptiveMaxPool1d` class is a 1D Adaptive Max Pooling layer.

        Parameters
        ------------
        output_size : int
            The target output size. It must be an integer.
        data_format : str
            One of channels_last (default, [batch,  width, channel]) or channels_first. The ordering of the dimensions in the inputs.
        name : None or str
            A unique layer name.

        Examples
        ---------
        With TensorLayerX

        >>> net = tlx.nn.Input([10, 32, 3], name='input')
        >>> net = tlx.nn.AdaptiveMaxPool1d(output_size=16)(net)
        >>> output shape : [10, 16, 3]

        """

    def __init__(self, output_size, data_format='channels_last', name=None):
        super(AdaptiveMaxPool1d, self).__init__(name)
        self.output_size = output_size
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info("AdaptiveMaxPool1d %s: output_size: %s " % (self.name, str(output_size)))

    def __repr__(self):
        s = ('{classname}(output_size={output_size}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):

        self.adaptivemaxpool1d = tlx.ops.AdaptiveMaxPool1D(output_size=self.output_size, data_format=self.data_format)

    def forward(self, inputs):
        outputs = self.adaptivemaxpool1d(inputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs


class AdaptiveMaxPool2d(Module):
    """The :class:`AdaptiveMaxPool2d` class is a 2D Adaptive Max Pooling layer.

        Parameters
        ------------
        output_size : int or list or  tuple
            The target output size. It cloud be an int \[int,int]\(int, int).
        data_format : str
            One of channels_last (default, [batch, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
        name : None or str
            A unique layer name.

        Examples
        ---------
        With TensorLayerX

        >>> net = tlx.nn.Input([10, 32, 32, 3], name='input')
        >>> net = tlx.nn.AdaptiveMaxPool2d(output_size=16)(net)
        >>> output shape : [10, 16, 16, 3]

    """

    def __init__(self, output_size, data_format='channels_last', name=None):
        super(AdaptiveMaxPool2d, self).__init__(name)
        self.output_size = output_size
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info("AdaptiveMaxPool1d %s: output_size: %s " % (self.name, str(output_size)))

    def __repr__(self):
        s = ('{classname}(output_size={output_size}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):

        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, ) * 2

        self.adaptivemaxpool2d = tlx.ops.AdaptiveMaxPool2D(output_size=self.output_size, data_format=self.data_format)

    def forward(self, inputs):
        outputs = self.adaptivemaxpool2d(inputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs


class AdaptiveMaxPool3d(Module):
    """The :class:`AdaptiveMaxPool3d` class is a 3D Adaptive Max Pooling layer.

        Parameters
        ------------
        output_size : int or list or  tuple
            The target output size. It cloud be an int \[int,int,int]\(int, int, int).
        data_format : str
            One of channels_last (default, [batch,  depth, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
        name : None or str
            A unique layer name.

        Examples
        ---------
        With TensorLayerX

        >>> net = tlx.nn.Input([10,32, 32, 32, 3], name='input')
        >>> net = tlx.nn.AdaptiveMaxPool3d(output_size=16)(net)
        >>> output shape : [10, 16, 16, 16, 3]

        """

    def __init__(self, output_size, data_format='channels_last', name=None):
        super(AdaptiveMaxPool3d, self).__init__(name)
        self.output_size = output_size
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info("AdaptiveMaxPool3d %s: output_size: %s " % (self.name, str(output_size)))

    def __repr__(self):
        s = ('{classname}(output_size={output_size}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):

        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, ) * 3

        self.adaptivemaxpool3d = tlx.ops.AdaptiveMaxPool3D(output_size=self.output_size, data_format=self.data_format)

    def forward(self, inputs):
        outputs = self.adaptivemaxpool3d(inputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs