#! /usr/bin/python
# -*- coding: utf-8 -*-

from tensorlayerx.nn.core import Module
import tensorlayerx as tlx
from tensorlayerx import logging

__all__ = [
    'Conv1d',
    'Conv2d',
    'Conv3d',
    'ConvTranspose1d',
    'ConvTranspose2d',
    'ConvTranspose3d',
]


class Conv1d(Module):
    """Applies a 1D convolution over an input signal composed of several input planes.

    Parameters
    ----------
    out_channels  : int
        Number of channels produced by the convolution
    kernel_size : int
        The kernel size
    stride : int
        The stride step
    dilation : int
        Specifying the dilation rate to use for dilated convolution.
    act : activation function
        The function that is applied to the layer activations
    padding : str or int
        The padding algorithm type: "SAME" or "VALID".
    data_format : str
        "channel_last" (NWC, default) or "channels_first" (NCW).
    W_init : initializer or str
        The initializer for the kernel weight matrix.
    b_init : initializer or None or str
        The initializer for the bias vector. If None, skip biases.
    in_channels : int
        The number of in channels.
    name : None or str
        A unique layer name

    Examples
    --------
    With TensorLayerx

    >>> net = tlx.nn.Input([8, 100, 1], name='input')
    >>> conv1d = tlx.nn.Conv1d(out_channels =32, kernel_size=5, stride=2, b_init=None, in_channels=1, name='conv1d_1')
    >>> print(conv1d)
    >>> tensor = tlx.nn.Conv1d(out_channels =32, kernel_size=5, stride=2, act=tlx.ReLU, name='conv1d_2')(net)
    >>> print(tensor)

    """

    def __init__(
        self,
        out_channels =32,
        kernel_size=5,
        stride=1,
        act=None,
        padding='SAME',
        data_format="channels_last",
        dilation=1,
        W_init='truncated_normal',
        b_init='constant',
        in_channels=None,
        name=None  # 'conv1d'
    ):
        super().__init__(name, act=act)
        self.out_channels  = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.data_format = data_format
        self.dilation = dilation
        self.W_init = self.str_to_init(W_init)
        self.b_init = self.str_to_init(b_init)
        self.in_channels = in_channels

        if self.in_channels:
            self.build(None)
            self._built = True

        logging.info(
            "Conv1d %s: out_channels : %d kernel_size: %s stride: %d pad: %s act: %s" % (
                self.name, out_channels , kernel_size, stride, padding,
                self.act.__class__.__name__ if self.act is not None else 'No Activation'
            )
        )

    def __repr__(self):
        actstr = self.act.__class__.__name__ if self.act is not None else 'No Activation'
        s = (
            '{classname}(in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}'
            ', stride={stride}, padding={padding}'
        )
        if self.dilation != 1:
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
            if self.in_channels is None:
                self.in_channels = inputs_shape[-1]
        elif self.data_format == 'channels_first':
            if self.in_channels is None:
                self.in_channels = inputs_shape[1]
        else:
            raise Exception("data_format should be either channels_last or channels_first")

        self.filter_shape = (self.kernel_size, self.in_channels, self.out_channels )

        # TODO : check
        self.W = self._get_weights("filters", shape=self.filter_shape, init=self.W_init)

        self.b_init_flag = False
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.out_channels , ), init=self.b_init)
            self.bias_add = tlx.ops.BiasAdd(self.data_format)
            self.b_init_flag = True

        self.conv1d = tlx.ops.Conv1D(
            stride=self.stride, padding=self.padding, data_format=self.data_format, dilations=self.dilation,
            out_channel=self.out_channels , k_size=self.kernel_size
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

        outputs = self.conv1d(inputs, self.W)
        if self.b_init_flag:
            outputs = self.bias_add(outputs, self.b)
        if self.act_init_flag:
            outputs = self.act(outputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs


class Conv2d(Module):
    """Applies a 2D convolution over an input signal composed of several input planes.

    Parameters
    ----------
    out_channels  : int
        Number of channels produced by the convolution
    kernel_size : tuple or int
        The kernel size (height, width).
    stride : tuple or int
        The sliding window stride of corresponding input dimensions.
        It must be in the same order as the ``shape`` parameter.
    dilation : tuple or int
        Specifying the dilation rate to use for dilated convolution.
    act : activation function
        The activation function of this layer.
    padding : int, tuple or str
        The padding algorithm type: "SAME" or "VALID".
        If padding is int or tuple, padding added to all four sides of the input. Default: ‘SAME’
    data_format : str
        "channels_last" (NHWC, default) or "channels_first" (NCHW).
    W_init : initializer or str
        The initializer for the the kernel weight matrix.
    b_init : initializer or None or str
        The initializer for the the bias vector. If None, skip biases.
    in_channels : int
        The number of in channels.
    name : None or str
        A unique layer name.

    Examples
    --------
    With TensorLayerx

    >>> net = tlx.nn.Input([8, 400, 400, 3], name='input')
    >>> conv2d = tlx.nn.Conv2d(out_channels =32, kernel_size=(3, 3), stride=(2, 2), b_init=None, in_channels=3, name='conv2d_1')
    >>> print(conv2d)
    >>> tensor = tlx.nn.Conv2d(out_channels =32, kernel_size=(3, 3), stride=(2, 2), act=tlx.ReLU, name='conv2d_2')(net)
    >>> print(tensor)

    """

    def __init__(
        self,
        out_channels =32,
        kernel_size=(3, 3),
        stride=(1, 1),
        act=None,
        padding='SAME',
        data_format='channels_last',
        dilation=(1, 1),
        W_init='truncated_normal',
        b_init='constant',
        in_channels=None,
        name=None,  # 'conv2d',
    ):
        super(Conv2d, self).__init__(name, act=act)
        self.out_channels = out_channels
        self.kernel_size = self.check_param(kernel_size)
        self._strides = self.stride = self.check_param(stride)
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
            "Conv2d %s: out_channels : %d kernel_size: %s stride: %s pad: %s act: %s" % (
                self.name, out_channels , str(kernel_size), str(stride), padding,
                self.act.__class__.__name__ if self.act is not None else 'No Activation'
            )
        )

    def __repr__(self):
        actstr = self.act.__class__.__name__ if self.act is not None else 'No Activation'
        s = (
            '{classname}(in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}'
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
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if self.data_format == 'channels_last':
            if self.in_channels is None:
                self.in_channels = inputs_shape[-1]
            self._strides = [1, self._strides[0], self._strides[1], 1]
            self._dilation_rate = [1, self._dilation_rate[0], self._dilation_rate[1], 1]
        elif self.data_format == 'channels_first':
            if self.in_channels is None:
                self.in_channels = inputs_shape[1]
            self._strides = [1, 1, self._strides[0], self._strides[1]]
            self._dilation_rate = [1, 1, self._dilation_rate[0], self._dilation_rate[1]]
        else:
            raise Exception("data_format should be either channels_last or channels_first")

        #TODO channels first filter shape [out_channel, in_channel, filter_h, filter_w]
        self.filter_shape = (self.kernel_size[0], self.kernel_size[1], self.in_channels, self.out_channels)
        self.W = self._get_weights("filters", shape=self.filter_shape, init=self.W_init)

        self.b_init_flag = False
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.out_channels , ), init=self.b_init)
            self.bias_add = tlx.ops.BiasAdd(self.data_format)
            self.b_init_flag = True

        self.conv2d = tlx.ops.Conv2D(
            strides=self._strides, padding=self.padding, data_format=self.data_format, dilations=self._dilation_rate,
            out_channel=self.out_channels , k_size=(self.kernel_size[0], self.kernel_size[1])
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

        outputs = self.conv2d(inputs, self.W)
        if self.b_init_flag:
            outputs = self.bias_add(outputs, self.b)
        if self.act_init_flag:
            outputs = self.act(outputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs


class Conv3d(Module):
    """Applies a 3D convolution over an input signal composed of several input planes.

    Parameters
    ----------
    out_channels  : int
        Number of channels produced by the convolution
    kernel_size : tuple or int
        The kernel size (depth, height, width).
    stride : tuple or int
        The sliding window stride of corresponding input dimensions.
        It must be in the same order as the ``shape`` parameter.
    dilation : tuple or int
        Specifying the dilation rate to use for dilated convolution.
    act : activation function
        The activation function of this layer.
    padding : int, tuple or str
        The padding algorithm type: "SAME" or "VALID".
    data_format : str
        "channels_last" (NDHWC, default) or "channels_first" (NCDHW).
    W_init : initializer or str
        The initializer for the the kernel weight matrix.
    b_init : initializer or None or str
        The initializer for the the bias vector. If None, skip biases.
    in_channels : int
        The number of in channels.
    name : None or str
        A unique layer name.

    Examples
    --------
    With TensorLayerx

    >>> net = tlx.nn.Input([8, 20, 20, 20, 3], name='input')
    >>> conv3d = tlx.nn.Conv3d(out_channels =32, kernel_size=(3, 3, 3), stride=(2, 2, 2), b_init=None, in_channels=3, name='conv3d_1')
    >>> print(conv3d)
    >>> tensor = tlx.nn.Conv3d(out_channels =32, kernel_size=(3, 3, 3), stride=(2, 2, 2), act=tlx.ReLU, name='conv3d_2')(net)
    >>> print(tensor)

    """

    def __init__(
        self,
        out_channels =32,
        kernel_size=(3, 3, 3),
        stride=(1, 1, 1),
        act=None,
        padding='SAME',
        data_format='channels_last',
        dilation=(1, 1, 1),
        W_init='truncated_normal',
        b_init='constant',
        in_channels=None,
        name=None  # 'conv3d',
    ):
        super().__init__(name, act=act)
        self.out_channels  = out_channels
        self.kernel_size = self.check_param(kernel_size, dim='3d')
        self._strides = self.stride = self.check_param(stride, dim='3d')
        self.padding = padding
        self.data_format = data_format
        self._dilation_rate = self.dilation = self.check_param(dilation, dim='3d')
        self.W_init = self.str_to_init(W_init)
        self.b_init = self.str_to_init(b_init)
        self.in_channels = in_channels

        if self.in_channels:
            self.build(None)
            self._built = True

        logging.info(
            "Conv3d %s: out_channels : %d kernel_size: %s stride: %s pad: %s act: %s" % (
                self.name, out_channels , str(kernel_size), str(stride), padding,
                self.act.__class__.__name__ if self.act is not None else 'No Activation'
            )
        )

    def __repr__(self):
        actstr = self.act.__class__.__name__ if self.act is not None else 'No Activation'
        s = (
            '{classname}(in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}'
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
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if self.data_format == 'channels_last':
            if self.in_channels is None:
                self.in_channels = inputs_shape[-1]
            self._strides = [1, self._strides[0], self._strides[1], self._strides[2], 1]
            self._dilation_rate = [1, self.dilation[0], self.dilation[1], self.dilation[2], 1]
        elif self.data_format == 'channels_first':
            if self.in_channels is None:
                self.in_channels = inputs_shape[1]
            self._strides = [1, 1, self._strides[0], self._strides[1], self._strides[2]]
            self._dilation_rate = [1, 1, self._dilation_rate[0], self._dilation_rate[1], self._dilation_rate[2]]
        else:
            raise Exception("data_format should be either channels_last or channels_first")

        self.filter_shape = (
            self.kernel_size[0], self.kernel_size[1], self.kernel_size[2], self.in_channels, self.out_channels
        )

        self.W = self._get_weights("filters", shape=self.filter_shape, init=self.W_init)

        self.b_init_flag = False
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.out_channels , ), init=self.b_init)
            self.bias_add = tlx.ops.BiasAdd(self.data_format)
            self.b_init_flag = True

        self.conv3d = tlx.ops.Conv3D(
            strides=self._strides, padding=self.padding, data_format=self.data_format, dilations=self._dilation_rate,
            out_channel=self.out_channels , k_size=(self.kernel_size[0], self.kernel_size[1], self.kernel_size[2])
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

        outputs = self.conv3d(inputs, self.W)
        if self.b_init_flag:
            outputs = self.bias_add(outputs, self.b)
        if self.act_init_flag:
            outputs = self.act(outputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs


class ConvTranspose1d(Module):
    """Applies a 1D transposed convolution operator over an input image composed of several input planes.

    Parameters
    ----------
    out_channels  : int
        Number of channels produced by the convolution
    kernel_size : int
        The kernel size
    stride : int or list
        An int or list of `ints` that has length `1` or `3`.  The number of entries by which the filter is moved right at each step.
    dilation : int or list
        Specifying the dilation rate to use for dilated convolution.
    act : activation function
        The function that is applied to the layer activations
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    data_format : str
        "channel_last" (NWC, default) or "channels_first" (NCW).
    W_init : initializer or str
        The initializer for the kernel weight matrix.
    b_init : initializer or None or str
        The initializer for the bias vector. If None, skip biases.
    in_channels : int
        The number of in channels.
    name : None or str
        A unique layer name

    Examples
    --------
    With TensorLayerx

    >>> net = tlx.nn.Input([8, 100, 1], name='input')
    >>> conv1d = tlx.nn.ConvTranspose1d(out_channels=32, kernel_size=5, stride=2, b_init=None, in_channels=1, name='Deonv1d_1')
    >>> print(conv1d)
    >>> tensor = tlx.nn.ConvTranspose1d(out_channels=32, kernel_size=5, stride=2, act=tlx.ReLU, name='ConvTranspose1d_2')(net)
    >>> print(tensor)

    """

    def __init__(
        self,
        out_channels=32,
        kernel_size=15,
        stride=1,
        act=None,
        padding='SAME',
        data_format="channels_last",
        dilation=1,
        W_init='truncated_normal',
        b_init='constant',
        in_channels=None,
        name=None  # 'conv1d_transpose'
    ):
        super(ConvTranspose1d, self).__init__(name, act=act)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.data_format = data_format
        self.dilation = dilation
        self.W_init = self.str_to_init(W_init)
        self.b_init = self.str_to_init(b_init)
        self.in_channels = in_channels

        if self.in_channels:
            self.build(None)
            self._built = True

        logging.info(
            "ConvTranspose1d %s: out_channels: %d kernel_size: %s stride: %d pad: %s act: %s" % (
                self.name, out_channels, kernel_size, stride, padding,
                self.act.__class__.__name__ if self.act is not None else 'No Activation'
            )
        )

    def __repr__(self):
        actstr = self.act.__class__.__name__ if self.act is not None else 'No Activation'
        s = (
            '{classname}(in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}'
            ', stride={stride}, padding={padding}'
        )
        if self.dilation != 1:
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
            if self.in_channels is None:
                self.in_channels = inputs_shape[-1]
        elif self.data_format == 'channels_first':
            if self.in_channels is None:
                self.in_channels = inputs_shape[1]
        else:
            raise Exception("data_format should be either channels_last or channels_first")

        self.filter_shape = (self.kernel_size, self.out_channels, self.in_channels)

        # TODO : check
        self.W = self._get_weights("filters", shape=self.filter_shape, init=self.W_init)

        self.b_init_flag = False
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.out_channels, ), init=self.b_init)
            self.bias_add = tlx.ops.BiasAdd(self.data_format)
            self.b_init_flag = True

        self.conv1d_transpose = tlx.ops.Conv1d_transpose(
            stride=self.stride,
            padding=self.padding,
            data_format=self.data_format,
            dilations=self.dilation,
            out_channel=self.out_channels,
            k_size=self.kernel_size,
            in_channels=self.in_channels,
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

        outputs = self.conv1d_transpose(inputs, self.W)
        if self.b_init_flag:
            outputs = self.bias_add(outputs, self.b)
        if self.act_init_flag:
            outputs = self.act(outputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs


class ConvTranspose2d(Module):
    """Applies a 2D transposed convolution operator over an input image composed of several input planes.

    Parameters
    ----------
    out_channels : int
        Number of channels produced by the convolution
    kernel_size : tuple or int
        The kernel size (height, width).
    stride : tuple or int
        The sliding window stride of corresponding input dimensions.
        It must be in the same order as the ``shape`` parameter.
    dilation : tuple or int
        Specifying the dilation rate to use for dilated convolution.
    act : activation function
        The activation function of this layer.
    padding : int, tuple or str
        The padding algorithm type: "SAME" or "VALID".
    data_format : str
        "channels_last" (NHWC, default) or "channels_first" (NCHW).
    W_init : initializer or str
        The initializer for the the kernel weight matrix.
    b_init : initializer or None or str
        The initializer for the the bias vector. If None, skip biases.
    in_channels : int
        The number of in channels.
    name : None or str
        A unique layer name.

    Examples
    --------
    With TensorLayerx

    >>> net = tlx.nn.Input([8, 400, 400, 3], name='input')
    >>> conv2d_transpose = tlx.nn.ConvTranspose2d(out_channels=32, kernel_size=(3, 3), stride=(2, 2), b_init=None, in_channels=3, name='conv2d_transpose_1')
    >>> print(conv2d_transpose)
    >>> tensor = tlx.nn.ConvTranspose2d(out_channels=32, kernel_size=(3, 3), stride=(2, 2), act=tlx.ReLU, name='conv2d_transpose_2')(net)
    >>> print(tensor)

    """

    def __init__(
        self,
        out_channels=32,
        kernel_size=(3, 3),
        stride=(1, 1),
        act=None,
        padding='SAME',
        data_format='channels_last',
        dilation=(1, 1),
        W_init='truncated_normal',
        b_init='constant',
        in_channels=None,
        name=None,  # 'conv2d_transpose',
    ):
        super(ConvTranspose2d, self).__init__(name, act=act)
        self.out_channels = out_channels
        self.kernel_size = self.check_param(kernel_size)
        self.stride = self.check_param(stride)
        self.padding = padding
        self.data_format = data_format
        self.dilation = self.check_param(dilation)
        self.W_init = self.str_to_init(W_init)
        self.b_init = self.str_to_init(b_init)
        self.in_channels = in_channels

        if self.in_channels:
            self.build(None)
            self._built = True

        logging.info(
            "ConvTranspose2d %s: out_channels: %d kernel_size: %s stride: %s pad: %s act: %s" % (
                self.name, out_channels, str(kernel_size), str(stride), padding,
                self.act.__class__.__name__ if self.act is not None else 'No Activation'
            )
        )

    def __repr__(self):
        actstr = self.act.__class__.__name__ if self.act is not None else 'No Activation'
        s = (
            '{classname}(in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}'
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
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if self.data_format == 'channels_last':
            if self.in_channels is None:
                self.in_channels = inputs_shape[-1]
        elif self.data_format == 'channels_first':
            if self.in_channels is None:
                self.in_channels = inputs_shape[1]
        else:
            raise Exception("data_format should be either channels_last or channels_first")

        #TODO channels first filter shape [out_channel, in_channel, filter_h, filter_w]
        self.filter_shape = (self.kernel_size[0], self.kernel_size[1], self.out_channels, self.in_channels)
        self.W = self._get_weights("filters", shape=self.filter_shape, init=self.W_init)#, transposed=True)

        self.b_init_flag = False
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.out_channels, ), init=self.b_init)
            self.bias_add = tlx.ops.BiasAdd(self.data_format)
            self.b_init_flag = True

        self.conv2d_transpose = tlx.ops.Conv2d_transpose(
            strides=self.stride, padding=self.padding, data_format=self.data_format, dilations=self.dilation,
            out_channel=self.out_channels, k_size=(self.kernel_size[0], self.kernel_size[1]), in_channels=self.in_channels
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

        outputs = self.conv2d_transpose(inputs, self.W)
        if self.b_init_flag:
            outputs = self.bias_add(outputs, self.b)
        if self.act_init_flag:
            outputs = self.act(outputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs


class ConvTranspose3d(Module):
    """Applies a 3D transposed convolution operator over an input image composed of several input planes.

    Parameters
    ----------
    out_channels : int
        Number of channels produced by the convolution
    kernel_size : tuple or int
        The kernel size (depth, height, width).
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
    W_init : initializer or str
        The initializer for the the kernel weight matrix.
    b_init : initializer or None or str
        The initializer for the the bias vector. If None, skip biases.
    in_channels : int
        The number of in channels.
    name : None or str
        A unique layer name.

    Examples
    --------
    With TensorLayerx

    >>> net = tlx.nn.Input([8, 20, 20, 20, 3], name='input')
    >>> ConvTranspose3d = tlx.nn.ConvTranspose3d(out_channels=32, kernel_size=(3, 3, 3), stride=(2, 2, 2), b_init=None, in_channels=3, name='deconv3d_1')
    >>> print(deconv3d)
    >>> tensor = tlx.nn.ConvTranspose3d(out_channels=32, kernel_size=(3, 3, 3), stride=(2, 2, 2), act=tlx.ReLU, name='ConvTranspose3d_2')(net)
    >>> print(tensor)

    """

    def __init__(
        self,
        out_channels=32,
        kernel_size=(3, 3, 3),
        stride=(1, 1, 1),
        act=None,
        padding='SAME',
        data_format='channels_last',
        dilation=(1, 1, 1),
        W_init='truncated_normal',
        b_init='constant',
        in_channels=None,
        name=None  # 'deconv3d',
    ):
        super(ConvTranspose3d, self).__init__(name, act=act)
        self.out_channels = out_channels
        self.kernel_size = self.check_param(kernel_size, dim='3d')
        self.stride = self.check_param(stride, dim='3d')
        self.padding = padding
        self.data_format = data_format
        self.dilation = self.check_param(dilation, dim='3d')
        self.W_init = self.str_to_init(W_init)
        self.b_init = self.str_to_init(b_init)
        self.in_channels = in_channels

        if self.in_channels:
            self.build(None)
            self._built = True

        logging.info(
            "ConvTranspose3d %s: out_channels: %d kernel_size: %s stride: %s pad: %s act: %s" % (
                self.name, out_channels, str(kernel_size), str(stride), padding,
                self.act.__class__.__name__ if self.act is not None else 'No Activation'
            )
        )

    def __repr__(self):
        actstr = self.act.__class__.__name__ if self.act is not None else 'No Activation'
        s = (
            '{classname}(in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}'
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
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if self.data_format == 'channels_last':
            if self.in_channels is None:
                self.in_channels = inputs_shape[-1]
        elif self.data_format == 'channels_first':
            if self.in_channels is None:
                self.in_channels = inputs_shape[1]
        else:
            raise Exception("data_format should be either channels_last or channels_first")

        self.filter_shape = (
            self.kernel_size[0], self.kernel_size[1], self.kernel_size[2], self.out_channels, self.in_channels
        )

        self.W = self._get_weights("filters", shape=self.filter_shape, init=self.W_init)#, transposed=True)

        self.b_init_flag = False
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.out_channels, ), init=self.b_init)
            self.bias_add = tlx.ops.BiasAdd(self.data_format)
            self.b_init_flag = True

        self.conv3d_transpose = tlx.ops.Conv3d_transpose(
            strides=self.stride, padding=self.padding, data_format=self.data_format, dilations=self.dilation,
            out_channel=self.out_channels, k_size=(self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]),
            in_channels=self.in_channels
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

        outputs = self.conv3d_transpose(inputs, self.W)
        if self.b_init_flag:
            outputs = self.bias_add(outputs, self.b)
        if self.act_init_flag:
            outputs = self.act(outputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs
