#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, array_ops
from tensorflow.python.training import moving_averages
from math import floor, ceil
import warnings
import math
import numpy as np
# loss function
sparse_softmax_cross_entropy_with_logits = tf.nn.sparse_softmax_cross_entropy_with_logits
sigmoid_cross_entropy_with_logits = tf.nn.sigmoid_cross_entropy_with_logits


def padding_format(padding):
    """
    Checks that the padding format correspond format.

    Parameters
    ----------
    padding : str
        Must be one of the following:"same", "SAME", "VALID", "valid"

    Returns
    -------
        str "SAME" or "VALID"
    """

    if padding in ["SAME", "same"]:
        padding = "SAME"
    elif padding in ["VALID", "valid"]:
        padding = "VALID"
    elif padding == None:
        padding = None
    elif isinstance(padding, tuple) or isinstance(padding, int):
        return padding
    else:
        raise Exception("Unsupported padding: " + str(padding))
    return padding

def channel_format(data_format, dim='2d'):
    if dim == '1d':
        if data_format in ["channels_last", "NWC", 'NLC']:
            data_format = "NWC"
        elif data_format in ["channels_first", "NCW", 'NCL']:
            data_format = "NCW"
        elif data_format == None:
            data_format = None
        else:
            raise Exception("Unsupported data format: " + str(data_format))
    elif dim == '2d':
        if data_format in ["channels_last", "NHWC"]:
            data_format = "NHWC"
        elif data_format in ["channels_first", "NCHW"]:
            data_format = "NCHW"
        elif data_format == None:
            data_format = None
        else:
            raise Exception("Unsupported data format: " + str(data_format))
    elif dim == '3d':
        if data_format in ['channels_last', 'NDHWC']:
            data_format = 'NDHWC'
        elif data_format in ['channels_first', 'NCDHW']:
            data_format = 'NCDHW'
        elif data_format == None:
            data_format = None
        else:
            raise Exception("Unsupported data format: " + str(data_format))
    else:
        raise Exception("dim must be '1d', '2d', '3d'.")
    return data_format

def preprocess_padding(padding, dim='2d', data_format='NHWC'):
    # When explicit padding is used and data_format is "NHWC",
    # this should be in the form [[0, 0], [pad_top, pad_bottom],[pad_left, pad_right], [0, 0]].
    # When explicit padding used and data_format is "NCHW",
    # this should be in the form [[0, 0], [0, 0],[pad_top, pad_bottom], [pad_left, pad_right]].
    check_padding(padding, dim)
    if dim == '1d':
        if data_format == 'NWC':
            out_padding = [[0, 0], [padding, padding], [0, 0]]
        else:
            out_padding = [[0, 0], [0, 0], [padding, padding]]
    elif dim == '2d':
        if isinstance(padding, int):
            if data_format == 'NHWC':
                out_padding = [[0, 0], [padding, padding], [padding, padding], [0, 0]]
            else:
                out_padding = [[0, 0], [0, 0],[padding, padding], [padding, padding]]
        elif isinstance(padding, tuple):
            if data_format == 'NHWC':
                out_padding = [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]]
            else:
                out_padding = [[0, 0], [0, 0],[padding[0], padding[0]], [padding[1], padding[1]]]
    elif dim == '3d':
        if isinstance(padding, int):
            if data_format == 'NDHWC':
                out_padding = [[0, 0], [padding, padding], [padding, padding], [padding, padding], [0, 0]]
            else:
                out_padding = [[0, 0], [0, 0], [padding, padding], [padding, padding], [padding, padding]]
        elif isinstance(padding, tuple):
            if data_format == 'NDHWC':
                out_padding = [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [padding[2], padding[2]], [0, 0]]
            else:
                out_padding = [[0, 0], [0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [padding[2], padding[2]]]
    else:
        raise RuntimeError("Unsupported input dimensions.")
    return out_padding


def check_padding(padding, dim='2d'):
    if dim == '1d' and isinstance(object, tuple):
        raise RuntimeError("expected padding to be a single integer value or a list of 1 values to match the convolution dimensions.")
    if dim == '2d' and isinstance(object, tuple) and len(padding) > 2:
        raise RuntimeError("expected padding to be a single integer value or a list of 2 values to match the convolution dimensions.")
    if dim == '3d' and isinstance(object, tuple) and len(padding) > 3:
        raise RuntimeError("expected padding to be a single integer value or a list of 3 values to match the convolution dimensions.")


def preprocess_1d_format(data_format, padding):
    """
    Checks that the 1-D dataformat format correspond format.

    Parameters
    ----------
    data_format : str
        Must be one of the following:"channels_last","NWC","NCW","channels_first"
    padding : str
        Must be one of the following:"same","valid","SAME","VALID"

    Returns
    -------
        str "NWC" or "NCW" and "SAME" or "VALID"
    """
    data_format = channel_format(data_format, dim='1d')
    padding = padding_format(padding)
    return data_format, padding


def preprocess_2d_format(data_format, padding):
    """
    Checks that the 2-D dataformat format correspond format.

    Parameters
    ----------
    data_format : str
        Must be one of the following:"channels_last","NHWC","NCHW","channels_first"
    padding : str
        Must be one of the following:"same","valid","SAME","VALID"

    Returns
    -------
        str "NHWC" or "NCHW" and "SAME" or "VALID"
    """

    data_format = channel_format(data_format, dim='2d')
    padding = padding_format(padding)
    return data_format, padding


def preprocess_3d_format(data_format, padding):
    """
    Checks that the 3-D dataformat format correspond format.

    Parameters
    ----------
    data_format : str
        Must be one of the following:"channels_last","NDHWC","NCDHW","channels_first"
    padding : str
        Must be one of the following:"same","valid","SAME","VALID"

    Returns
    -------
        str "NDHWC" or "NCDHW" and "SAME" or "VALID"
    """

    data_format = channel_format(data_format, dim='3d')
    padding = padding_format(padding)
    return data_format, padding


def nchw_to_nhwc(x):
    """
    Channels first to channels last

    Parameters
    ----------
    x : tensor
        channels first tensor data

    Returns
    -------
        channels last tensor data
    """

    if len(x.shape) == 3:
        x = tf.transpose(x, (0, 2, 1))
    elif len(x.shape) == 4:
        x = tf.transpose(x, (0, 2, 3, 1))
    elif len(x.shape) == 5:
        x = tf.transpose(x, (0, 2, 3, 4, 1))
    else:
        raise Exception("Unsupported dimensions")
    return x


def nhwc_to_nchw(x):
    """
    Channles last to channels first

    Parameters
    ----------
    x : tensor
        channels last tensor data

    Returns
    -------
        channels first tensor data
    """

    if len(x.shape) == 3:
        x = tf.transpose(x, (0, 2, 1))
    elif len(x.shape) == 4:
        x = tf.transpose(x, (0, 3, 1, 2))
    elif len(x.shape) == 5:
        x = tf.transpose(x, (0, 4, 1, 2, 3))
    else:
        raise Exception("Unsupported dimensions")
    return x


class ReLU(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return tf.nn.relu(x)


def relu(x):
    """
    Computes rectified linear: max(features, 0).

    Parameters
    ----------
    x : tensor
        Must be one of the following types: float32, float64, int32, uint8, int16,
        int8, int64, bfloat16, uint16, half, uint32, uint64, qint8.

    Returns
    -------
        A Tensor. Has the same type as features.
    """

    return tf.nn.relu(x)


class ELU(object):

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, x):
        res = tf.nn.elu(x)
        if self.alpha == 1:
            return res
        else:
            return tf.where(x > 0, res, self.alpha * res)


def elu(x, alpha=1.0):
    """
    Computes exponential linear: `exp(features) - 1` if < 0, `features` otherwise.

    See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
    ](http://arxiv.org/abs/1511.07289)

    Parameters
    ----------
    x : tensor
        Must be one of the following types: half, bfloat16, float32, float64.

    Returns
    -------
        A Tensor with the same type as features.
  """

    return tf.nn.elu(x * alpha)


class ReLU6(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return tf.nn.relu6(x)


def relu6(x):
    """
    Computes Rectified Linear 6: min(max(features, 0), 6).

    Parameters
    ----------
    x : tensor
        Must be one of the following types: float32, float64, int32, uint8, int16,
        int8, int64, bfloat16, uint16, half, uint32, uint64, qint8.

    Returns
    -------
        A Tensor with the same type as features.
    """

    return tf.nn.relu6(x)


class LeakyReLU(object):

    def __init__(self, negative_slope=0.2):
        self.negative_slope = negative_slope

    def __call__(self, x):
        return tf.nn.leaky_relu(x, alpha=self.negative_slope)


def leaky_relu(x, negative_slope=0.2):
    """
    Compute the Leaky ReLU activation function.

    Parameters
    ----------
    x : tensor
        representing preactivation values. Must be one of the following types:
        float16, float32, float64, int32, int64.

    Returns
    -------
        The activation value.
    """

    return tf.nn.leaky_relu(x, alpha=negative_slope)


class Softplus(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return tf.nn.softplus(x)


class Tanh(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return tf.nn.tanh(x)


class Sigmoid(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return tf.nn.sigmoid(x)


def sigmoid(x):
    """
    Computes sigmoid of x element-wise.

    Parameters
    ----------
    x : tensor
        A Tensor with type float16, float32, float64, complex64, or complex128.

    Returns
    -------
        A Tensor with the same type as x.
    """

    return tf.nn.sigmoid(x)


class Softmax(object):

    def __init__(self, axis = -1):
        self.axis = axis

    def __call__(self, x):
        return tf.nn.softmax(x, axis = self.axis)


def softmax(logits, axis=-1):
    """
    Computes softmax activations.

    Parameters
    ----------
    logits : tensor
        Must be one of the following types: half, float32, float64.
    axis : int
        The dimension softmax would be performed on. The default is -1 which indicates the last dimension.

    Returns
    -------
        A Tensor. Has the same type and shape as logits.
    """

    return tf.nn.softmax(logits, axis)


class GeLU(object):

    def __init__(self, approximate=False):
        self.approximate = approximate

    def __call__(self, x):
        return tf.nn.gelu(x, approximate=self.approximate)


def gelu(x, approximate=False):

    return tf.nn.gelu(x, approximate=approximate)


class Dropout(object):

    def __init__(self, p, seed=0):
        self.p = p
        self.seed = seed

    def __call__(self, inputs, *args, **kwargs):
        outputs = tf.nn.dropout(inputs, rate=self.p, seed=self.seed)
        return outputs


class BiasAdd(object):
    """
    Adds bias to value.

    Parameters
    ----------
    x : tensor
        A Tensor with type float, double, int64, int32, uint8, int16, int8, complex64, or complex128.
    bias : tensor
        Must be the same type as value unless value is a quantized type,
        in which case a different quantized type may be used.
    Returns
    -------
        A Tensor with the same type as value.
    """

    def __init__(self, data_format='channels_last'):
        if data_format in ['channels_first', 'NCL', 'NCW', 'NCHW', 'NCDHW']:
            self.data_format = "NCHW"
        elif data_format in ['channels_last', 'NLC', 'NWC', 'NHWC', 'NDHWC']:
            self.data_format = "NHWC"
    def __call__(self, x, bias):
        return tf.nn.bias_add(x, bias, data_format=self.data_format)


def bias_add(x, bias, data_format=None, name=None):
    """
    Adds bias to value.

    Parameters
    ----------
    x : tensor
        A Tensor with type float, double, int64, int32, uint8, int16, int8, complex64, or complex128.
    bias : tensor
        Must be the same type as value unless value is a quantized type,
        in which case a different quantized type may be used.
    data_format : A string.
        'N...C' and 'NC...' are supported.
    name : str
        A name for the operation (optional).
    Returns
    -------
        A Tensor with the same type as value.
    """
    data_format , _ = preprocess_2d_format(data_format, None)
    x = tf.nn.bias_add(x, bias, data_format=data_format, name=name)
    return x


class Conv1D(object):

    def __init__(self, stride, padding, data_format='NWC', dilations=None, out_channel=None, k_size=None):
        self.stride = stride
        self.dilations = dilations
        self.data_format, self.padding = preprocess_1d_format(data_format, padding)
        self.pad_value = None

        if isinstance(padding, int):
            self.pad_value = preprocess_padding(self.padding, '1d', self.data_format)
            self.padding = 'VALID'


    def __call__(self, input, filters):
        if self.pad_value is not None:
            input = tf.pad(input, paddings=self.pad_value)
        outputs = tf.nn.conv1d(
            input=input,
            filters=filters,
            stride=self.stride,
            padding=self.padding,
            data_format=self.data_format,
            dilations=self.dilations,
            # name=name
        )
        return outputs


def conv1d(input, filters, stride, padding, data_format='NWC', dilations=None):
    """
    Computes a 1-D convolution given 3-D input and filter tensors.

    Parameters
    ----------
    input : tensor
        A 3D Tensor. Must be of type float16, float32, or float64
    filters : tensor
        A 3D Tensor. Must have the same type as input.
    stride : int of list
         An int or list of ints that has length 1 or 3. The number of entries by which the filter is moved right at each step.
    padding : string
         'SAME' or 'VALID'
    data_format : string
        An optional string from "NWC", "NCW". Defaults to "NWC", the data is stored in the order of
        [batch, in_width, in_channels]. The "NCW" format stores data as [batch, in_channels, in_width].
    dilations : int or list
        An int or list of ints that has length 1 or 3 which defaults to 1.
        The dilation factor for each dimension of input. If set to k > 1,
        there will be k-1 skipped cells between each filter element on that dimension.
        Dilations in the batch and depth dimensions must be 1.
    name : string
        A name for the operation (optional).
    Returns
    -------
        A Tensor. Has the same type as input.
    """

    data_format, padding = preprocess_1d_format(data_format, padding)
    outputs = tf.nn.conv1d(
        input=input,
        filters=filters,
        stride=stride,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        # name=name
    )
    return outputs


class Conv2D(object):

    def __init__(self, strides, padding, data_format='NHWC', dilations=None, out_channel=None, k_size=None):
        self.strides = strides
        self.dilations = dilations
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)

        if isinstance(padding, int) or isinstance(padding, tuple):
            self.padding = preprocess_padding(self.padding, '2d', self.data_format)

    def __call__(self, input, filters):
        outputs = tf.nn.conv2d(
            input=input,
            filters=filters,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilations=self.dilations,
        )
        return outputs


def conv2d(input, filters, strides, padding, data_format='NHWC', dilations=None):
    """
    Computes a 2-D convolution given 4-D input and filters tensors.

    Parameters
    ----------
    input : tensor
        Must be one of the following types: half, bfloat16, float32, float64. A 4-D tensor.
        The dimension order is interpreted according to the value of data_format, see below for details.
    filters : tensor
         Must have the same type as input. A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
    strides : int of list
        The stride of the sliding window for each dimension of input. If a single value is given it is replicated in the H and W dimension.
        By default the N and C dimensions are set to 1. The dimension order is determined by the value of data_format, see below for details.
    padding : string
        "SAME" or "VALID"
    data_format : string
        "NHWC", "NCHW". Defaults to "NHWC".
    dilations : list or ints
        list of ints that has length 1, 2 or 4, defaults to 1. The dilation factor for each dimension ofinput.
    name : string
         A name for the operation (optional).

    Returns
    -------
        A Tensor. Has the same type as input.
    """

    data_format, padding = preprocess_2d_format(data_format, padding)
    outputs = tf.nn.conv2d(
        input=input,
        filters=filters,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
    )
    return outputs


class Conv3D(object):

    def __init__(self, strides, padding, data_format='NDHWC', dilations=None, out_channel=None, k_size=None):
        self.strides = strides
        self.dilations = dilations
        self.data_format, self.padding = preprocess_3d_format(data_format, padding)
        self.pad_value = None

        if isinstance(padding, int) or isinstance(padding, tuple):
            self.pad_value = preprocess_padding(self.padding, '3d', self.data_format)
            self.padding = 'VALID'

    def __call__(self, input, filters):
        if self.pad_value is not None:
            input = tf.pad(input, paddings=self.pad_value)
        outputs = tf.nn.conv3d(
            input=input,
            filters=filters,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilations=self.dilations,
        )
        return outputs


def conv3d(input, filters, strides, padding, data_format='NDHWC', dilations=None):
    """
    Computes a 3-D convolution given 5-D input and filters tensors.

    Parameters
    ----------
    input : tensor
        Must be one of the following types: half, bfloat16, float32, float64.
        Shape [batch, in_depth, in_height, in_width, in_channels].
    filters : tensor
        Must have the same type as input. Shape [filter_depth, filter_height, filter_width, in_channels, out_channels].
        in_channels must match between input and filters.
    strides : list of ints
        A list of ints that has length >= 5. 1-D tensor of length 5.
        The stride of the sliding window for each dimension of input.
        Must have strides[0] = strides[4] = 1.
    padding : string
        A string from: "SAME", "VALID". The type of padding algorithm to use.
    data_format : string
        An optional string from: "NDHWC", "NCDHW". Defaults to "NDHWC". The data format of the input and output data.
        With the default format "NDHWC", the data is stored in the order of: [batch, in_depth, in_height, in_width, in_channels].
        Alternatively, the format could be "NCDHW", the data storage order is: [batch, in_channels, in_depth, in_height, in_width].
    dilations : list of ints
        Defaults to [1, 1, 1, 1, 1]. 1-D tensor of length 5. The dilation factor for each dimension of input.
        If set to k > 1, there will be k-1 skipped cells between each filter element on that dimension.
        The dimension order is determined by the value of data_format, see above for details.
        Dilations in the batch and depth dimensions must be 1.
    name : string
        A name for the operation (optional).

    Returns
    -------
        A Tensor. Has the same type as input.
    """

    data_format, padding = preprocess_3d_format(data_format, padding)
    outputs = tf.nn.conv3d(
        input=input,
        filters=filters,
        strides=strides,
        padding=padding,
        data_format=data_format,  # 'NDHWC',
        dilations=dilations,  # [1, 1, 1, 1, 1],
        # name=name,
    )
    return outputs


def lrn(inputs, depth_radius, bias, alpha, beta):
    """
    Local Response Normalization.

    Parameters
    ----------
    inputs : tensor
        Must be one of the following types: half, bfloat16, float32. 4-D.
    depth_radius : int
        Defaults to 5. 0-D. Half-width of the 1-D normalization window.
    bias : float
        Defaults to 1. An offset (usually positive to avoid dividing by 0).
    alpha : float
        Defaults to 1. A scale factor, usually positive.
    beta : float
         Defaults to 0.5. An exponent.

    Returns
    -------
        A Tensor. Has the same type as input.
    """

    outputs = tf.nn.lrn(inputs, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta)
    return outputs


def moments(x, axes, shift=None, keepdims=False):
    """
    Calculates the mean and variance of x.

    Parameters
    ----------
    x : tensor
        A Tensor
    axes : list or ints
        Axes along which to compute mean and variance.
    shift : int
        Not used in the current implementation.
    keepdims : bool
        produce moments with the same dimensionality as the input.

    Returns
    -------
        Two Tensor objects: mean and variance.
    """

    outputs = tf.nn.moments(x, axes, shift, keepdims)
    return outputs


class MaxPool1d(object):

    def __init__(self, ksize, strides, padding, data_format=None):
        self.data_format, self.padding = preprocess_1d_format(data_format=data_format, padding=padding)
        self.ksize = ksize
        self.strides = strides
        self.padding_value = None
        if not isinstance(self.padding, str):
            self.padding_value = preprocess_padding(self.padding, '1d', self.data_format)
            self.padding = "VALID"

    def __call__(self, inputs):
        if self.padding_value is not None:
            inputs = tf.pad(inputs, self.padding_value)
        outputs = tf.nn.max_pool(
            input=inputs, ksize=self.ksize, strides=self.strides, padding=self.padding, data_format=self.data_format
        )
        return outputs


class MaxPool(object):

    def __init__(self, ksize, strides, padding, data_format=None):
        self.ksize = ksize
        self.strides = strides
        self.data_format = data_format
        self.padding = padding

    def __call__(self, inputs):
        if len(inputs.shape) == 3:
            self.data_format, self.padding = preprocess_1d_format(data_format=self.data_format, padding=self.padding)
            if not isinstance(self.padding, str):
                self.padding_value = preprocess_padding(self.padding, '1d', self.data_format)
                self.padding = "VALID"
                inputs = tf.pad(inputs, self.padding_value)
        elif len(inputs.shape) == 4:
            self.data_format, self.padding = preprocess_2d_format(data_format=self.data_format, padding=self.padding)
            if not isinstance(self.padding, str):
                self.padding_value = preprocess_padding(self.padding, '2d', self.data_format)
                self.padding = "VALID"
                inputs = tf.pad(inputs, self.padding_value)
        elif len(inputs.shape) == 5:
            self.data_format, self.padding = preprocess_3d_format(data_format=self.data_format, padding=self.padding)
            if not isinstance(self.padding, str):
                self.padding_value = preprocess_padding(self.padding, '3d', self.data_format)
                self.padding = "VALID"
                inputs = tf.pad(inputs, self.padding_value)

        outputs = tf.nn.max_pool(
            input=inputs, ksize=self.ksize, strides=self.strides, padding=self.padding, data_format=self.data_format
        )
        return outputs


def max_pool(input, ksize, strides, padding, data_format=None):
    """
    Performs the max pooling on the input.

    Parameters
    ----------
    input : tensor
        Tensor of rank N+2, of shape [batch_size] + input_spatial_shape + [num_channels] if data_format does not start
        with "NC" (default), or [batch_size, num_channels] + input_spatial_shape if data_format starts with "NC".
        Pooling happens over the spatial dimensions only.
    ksize : int or list of ints
        An int or list of ints that has length 1, N or N+2.
        The size of the window for each dimension of the input tensor.
    strides : int or list of ints
        An int or list of ints that has length 1, N or N+2.
        The stride of the sliding window for each dimension of the input tensor.
    padding : string
        'VALID' or 'SAME'. The padding algorithm. See the "returns" section of tf.ops.convolution for details.
    name : string
        A name for the operation (optional).

    Returns
    -------
        A Tensor of format specified by data_format. The max pooled output tensor.
    """

    if len(input.shape) == 3:
        data_format, padding = preprocess_1d_format(data_format=data_format, padding=padding)
    elif len(input.shape) == 4:
        data_format, padding = preprocess_2d_format(data_format=data_format, padding=padding)
    elif len(input.shape) == 5:
        data_format, padding = preprocess_3d_format(data_format=data_format, padding=padding)

    outputs = tf.nn.max_pool(input=input, ksize=ksize, strides=strides, padding=padding, data_format=data_format)
    return outputs


class AvgPool1d(object):

    def __init__(self, ksize, strides, padding, data_format=None):
        self.data_format, self.padding = preprocess_1d_format(data_format=data_format, padding=padding)
        self.ksize = [ksize, ]
        self.strides = [strides, ]
        self.padding_value = None
        if not isinstance(self.padding, str):
            self.padding_value = preprocess_padding(self.padding, '1d', self.data_format)
            self.padding = "VALID"

    def __call__(self, inputs):
        if self.padding_value is not None:
            inputs = tf.pad(inputs, self.padding_value)
        outputs = tf.nn.pool(
            input=inputs,
            window_shape=self.ksize,
            pooling_type="AVG",
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
        )
        return outputs


class AvgPool(object):

    def __init__(self, ksize, strides, padding, data_format=None):
        self.ksize = ksize
        self.strides = strides
        self.data_format, self.padding = preprocess_2d_format(data_format=data_format, padding=padding)
        self.padding_value = None
        if not isinstance(self.padding, str):
            self.padding_value = preprocess_padding(self.padding, '2d', self.data_format)
            self.padding = "VALID"

    def __call__(self, inputs):
        data_format = channel_format(self.data_format, str(len(inputs.shape) - 2) + 'd')
        if self.padding_value is not None:
            inputs = tf.pad(inputs, self.padding_value)
        outputs = tf.nn.avg_pool(
            input=inputs, ksize=self.ksize, strides=self.strides, padding=self.padding, data_format=data_format
        )
        return outputs


def avg_pool(input, ksize, strides, padding):
    """
    Performs the avg pooling on the input.

    Parameters
    ----------
    input : tensor
        Tensor of rank N+2, of shape [batch_size] + input_spatial_shape + [num_channels]
        if data_format does not start with "NC" (default), or [batch_size, num_channels] + input_spatial_shape
        if data_format starts with "NC". Pooling happens over the spatial dimensions only.
    ksize : int or list of ints
        An int or list of ints that has length 1, N or N+2.
        The size of the window for each dimension of the input tensor.
    strides : int or list of ints
        An int or list of ints that has length 1, N or N+2.
        The stride of the sliding window for each dimension of the input tensor.
    padding : string
        'VALID' or 'SAME'. The padding algorithm. See the "returns" section of tf.ops.convolution for details.
    name : string
        Optional name for the operation.

    Returns
    -------
        A Tensor of format specified by data_format. The average pooled output tensor.
    """

    padding = padding_format(padding)
    outputs = tf.nn.avg_pool(
        input=input,
        ksize=ksize,
        strides=strides,
        padding=padding,
    )
    return outputs


class MaxPool3d(object):

    def __init__(self, ksize, strides, padding, data_format=None):
        self.data_format, self.padding = preprocess_3d_format(data_format, padding)
        self.ksize = ksize
        self.strides = strides
        self.padding_value = None
        if not isinstance(self.padding, str):
            self.padding_value = preprocess_padding(self.padding, '3d', self.data_format)
            self.padding = "VALID"

    def __call__(self, inputs):
        if self.padding_value is not None:
            inputs = tf.pad(inputs, self.padding_value)
        outputs = tf.nn.max_pool3d(
            input=inputs,
            ksize=self.ksize,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
        )
        return outputs


def max_pool3d(input, ksize, strides, padding, data_format=None):
    """
    Performs the max pooling on the input.

    Parameters
    ----------
    input : tensor
         A 5-D Tensor of the format specified by data_format.
    ksize : int or list of ints
        An int or list of ints that has length 1, 3 or 5.
        The size of the window for each dimension of the input tensor.
    strides : int or list of ints
        An int or list of ints that has length 1, 3 or 5.
        The stride of the sliding window for each dimension of the input tensor.
    padding : string
        'VALID' or 'SAME'. The padding algorithm. See the "returns" section of tf.ops.convolution for details.
    data_format : string
         "NDHWC", "NCDHW". Defaults to "NDHWC". The data format of the input and output data.
         With the default format "NDHWC", the data is stored in the order of: [batch, in_depth, in_height, in_width, in_channels].
         Alternatively, the format could be "NCDHW", the data storage order is: [batch, in_channels, in_depth, in_height, in_width].
    name : string
         A name for the operation (optional).

    Returns
    -------
        A Tensor of format specified by data_format. The max pooled output tensor.
    """

    data_format, padding = preprocess_3d_format(data_format, padding)
    outputs = tf.nn.max_pool3d(
        input=input,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
    )
    return outputs


class AvgPool3d(object):

    def __init__(self, ksize, strides, padding, data_format=None):
        self.data_format, self.padding = preprocess_3d_format(data_format, padding)
        self.ksize = ksize
        self.strides = strides
        self.padding_value = None
        if not isinstance(self.padding, str):
            self.padding_value = preprocess_padding(self.padding, '3d', self.data_format)
            self.padding = "VALID"

    def __call__(self, inputs):
        if self.padding_value is not None:
            inputs = tf.pad(inputs, self.padding_value)
        outputs = tf.nn.avg_pool3d(
            input=inputs,
            ksize=self.ksize,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
        )
        return outputs


def avg_pool3d(input, ksize, strides, padding, data_format=None):
    """
    Performs the average pooling on the input.

    Parameters
    ----------
    input : tensor
        A 5-D Tensor of shape [batch, height, width, channels] and type float32, float64, qint8, quint8, or qint32.
    ksize : int or list of ints
        An int or list of ints that has length 1, 3 or 5. The size of the window for each dimension of the input tensor.
    strides : int or list of ints
        An int or list of ints that has length 1, 3 or 5.
        The stride of the sliding window for each dimension of the input tensor.
    padding : string
        'VALID' or 'SAME'. The padding algorithm. See the "returns" section of tf.ops.convolution for details.
    data_format : string
        'NDHWC' and 'NCDHW' are supported.
    name : string
        Optional name for the operation.

    Returns
    -------
        A Tensor with the same type as value. The average pooled output tensor.
    """

    data_format, padding = preprocess_3d_format(data_format, padding)
    outputs = tf.nn.avg_pool3d(
        input=input,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
    )
    return outputs


def pool(input, window_shape, pooling_type, strides=None, padding='VALID', data_format=None, dilations=None, name=None):
    """
    Performs an N-D pooling operation.

    Parameters
    ----------
    input : tensor
        Tensor of rank N+2, of shape [batch_size] + input_spatial_shape + [num_channels]
        if data_format does not start with "NC" (default), or [batch_size, num_channels] + input_spatial_shape
        if data_format starts with "NC". Pooling happens over the spatial dimensions only.
    window_shape : int
        Sequence of N ints >= 1.
    pooling_type : string
        Specifies pooling operation, must be "AVG" or "MAX".
    strides : ints
        Sequence of N ints >= 1. Defaults to [1]*N. If any value of strides is > 1, then all values of dilation_rate must be 1.
    padding : string
        The padding algorithm, must be "SAME" or "VALID". Defaults to "SAME".
        See the "returns" section of tf.ops.convolution for details.
    data_format : string
        Specifies whether the channel dimension of the input and output is the last dimension (default, or if data_format does not start with "NC"),
        or the second dimension (if data_format starts with "NC").
        For N=1, the valid values are "NWC" (default) and "NCW". For N=2, the valid values are "NHWC" (default) and "NCHW".
        For N=3, the valid values are "NDHWC" (default) and "NCDHW".
    dilations : list of ints
        Dilation rate. List of N ints >= 1. Defaults to [1]*N. If any value of dilation_rate is > 1, then all values of strides must be 1.
    name : string
        Optional. Name of the op.

    Returns
    -------
        Tensor of rank N+2, of shape [batch_size] + output_spatial_shape + [num_channels]
    """
    if pooling_type in ["MAX", "max"]:
        pooling_type = "MAX"
    elif pooling_type in ["AVG", "avg"]:
        pooling_type = "AVG"
    else:
        raise ValueError('Unsupported pool_mode: ' + str(pooling_type))
    padding = padding_format(padding)
    outputs = tf.nn.pool(
        input=input,
        window_shape=window_shape,
        pooling_type=pooling_type,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name,
    )
    return outputs


class DepthwiseConv2d(object):

    def __init__(self, strides, padding, data_format=None, dilations=None, ksize=None, channel_multiplier=1):
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.strides = strides
        self.dilations = dilations

    def __call__(self, input, filter, point_filter=None):
        outputs = tf.nn.depthwise_conv2d(
            input=input,
            filter=filter,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilations=self.dilations,
        )
        return outputs


def depthwise_conv2d(input, filter, strides, padding, data_format=None, dilations=None, name=None):
    """
    Depthwise 2-D convolution.

    Parameters
    ----------
    input : tensor
        4-D with shape according to data_format.
    filter : tensor
        4-D with shape [filter_height, filter_width, in_channels, channel_multiplier].
    strides : list
        1-D of size 4. The stride of the sliding window for each dimension of input.
    padding : string
        'VALID' or 'SAME'. The padding algorithm. See the "returns" section of tf.ops.convolution for details.
    data_format : string
        The data format for input. Either "NHWC" (default) or "NCHW".
    dilations : list
        1-D of size 2. The dilation rate in which we sample input values across the height and width dimensions in atrous convolution.
        If it is greater than 1, then all values of strides must be 1.
    name : string
        A name for this operation (optional).

    Returns
    -------
        A 4-D Tensor with shape according to data_format.
        E.g., for "NHWC" format, shape is [batch, out_height, out_width, in_channels * channel_multiplier].
    """

    data_format, padding = preprocess_2d_format(data_format, padding)
    outputs = tf.nn.depthwise_conv2d(
        input=input,
        filter=filter,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name,
    )
    return outputs


class Conv1d_transpose(object):

    def __init__(
        self, stride, padding, data_format='NWC', dilations=None, out_channel=None, k_size=None, in_channels=None
    ):
        self.stride = stride
        self.dilations = dilations
        self.data_format, self.padding = preprocess_1d_format(data_format, padding)

    def __call__(self, input, filters):
        batch_size = input.shape[0]
        if self.data_format == 'NWC':
            w_axis, c_axis = 1, 2
        else:
            w_axis, c_axis = 2, 1

        input_shape = input.shape.as_list()
        filters_shape = filters.shape.as_list()
        input_w = input_shape[w_axis]
        filters_w = filters_shape[0]
        output_channels = filters_shape[1]
        dilations_w = 1

        if isinstance(self.stride, int):
            strides_w = self.stride
        else:
            strides_list = list(self.stride)
            strides_w = strides_list[w_axis]

        if self.dilations is not None:
            if isinstance(self.dilations, int):
                dilations_w = self.dilations
            else:
                dilations_list = list(self.dilations)
                dilations_w = dilations_list[w_axis]

        filters_w = filters_w + (filters_w - 1) * (dilations_w - 1)
        assert self.padding in {'SAME', 'VALID'}
        if self.padding == 'VALID':
            output_w = input_w * strides_w + max(filters_w - strides_w, 0)
        elif self.padding == 'SAME':
            output_w = input_w * strides_w

        if self.data_format == 'NCW':
            output_shape = (batch_size, output_channels, output_w)
        else:
            output_shape = (batch_size, output_w, output_channels)
        output_shape = tf.stack(output_shape)
        outputs = tf.nn.conv1d_transpose(
            input=input,
            filters=filters,
            output_shape=output_shape,
            strides=self.stride,
            padding=self.padding,
            data_format=self.data_format,
            dilations=self.dilations,
        )
        return outputs


def conv1d_transpose(
    input, filters, output_shape, strides, padding='SAME', data_format='NWC', dilations=None, name=None
):
    """
    The transpose of conv1d.

    Parameters
    ----------
    input : tensor
        A 3-D Tensor of type float and shape [batch, in_width, in_channels]
        for NWC data format or [batch, in_channels, in_width] for NCW data format.
    filters : tensor
        A 3-D Tensor with the same type as value and shape [filter_width, output_channels, in_channels].
        filter's in_channels dimension must match that of value.
    output_shape : tensor
        A 1-D Tensor, containing three elements, representing the output shape of the deconvolution op.
    strides : list
        An int or list of ints that has length 1 or 3. The number of entries by which the filter is moved right at each step.
    padding : string
        'VALID' or 'SAME'. The padding algorithm. See the "returns" section of tf.ops.convolution for details.
    data_format : string
        'NWC' and 'NCW' are supported.
    dilations : list
         An int or list of ints that has length 1 or 3 which defaults to 1.
         The dilation factor for each dimension of input. If set to k > 1,
         there will be k-1 skipped cells between each filter element on that dimension.
         Dilations in the batch and depth dimensions must be 1.
    name : string
        Optional name for the returned tensor.

    Returns
    -------
        A Tensor with the same type as value.
    """

    data_format, padding = preprocess_1d_format(data_format, padding)
    outputs = tf.nn.conv1d_transpose(
        input=input,
        filters=filters,
        output_shape=output_shape,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name,
    )
    return outputs


class Conv2d_transpose(object):

    def __init__(
        self, strides, padding, data_format='NHWC', dilations=None, name=None, out_channel=None, k_size=None,
        in_channels=None
    ):
        self.strides = strides
        self.dilations = dilations
        self.name = name
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self._padding = padding

    def __call__(self, input, filters):
        if self.data_format == 'NHWC':
            h_axis, w_axis = 1, 2
        else:
            h_axis, w_axis = 2, 3

        input_shape = input.shape.as_list()
        filters_shape = filters.shape.as_list()
        batch_size = input.shape[0]
        input_h, input_w = input_shape[h_axis], input_shape[w_axis]
        kernel_h, kernel_w = filters_shape[0], filters_shape[1]
        output_channels = filters_shape[2]
        dilations_h, dilations_w = 1, 1

        if isinstance(self.strides, int):
            strides_h = self.strides
            strides_w = self.strides
        else:
            strides_list = list(self.strides)
            if len(strides_list) == 2:
                strides_h = strides_list[0]
                strides_w = strides_list[1]
            elif len(strides_list) == 4:
                strides_h = strides_list[h_axis]
                strides_w = strides_list[w_axis]

        if self.dilations is not None:
            if isinstance(self.dilations, int):
                dilations_h = self.dilations
                dilations_w = self.dilations
            else:
                dilations_list = list(self.dilations)
                if len(dilations_list) == 2:
                    dilations_h = dilations_list[0]
                    dilations_w = dilations_list[1]
                elif len(dilations_list) == 4:
                    dilations_h = dilations_list[h_axis]
                    dilations_w = dilations_list[w_axis]

        kernel_h = kernel_h + (kernel_h - 1) * (dilations_h - 1)
        kernel_w = kernel_w + (kernel_w - 1) * (dilations_w - 1)

        if tf.__version__ < '2.4.0' and not isinstance(self.padding, str):
            assert self.padding in {'SAME', 'VALID'}
        if self.padding == 'VALID':
            output_h = input_h * strides_h + max(kernel_h - strides_h, 0)
            output_w = input_w * strides_w + max(kernel_w - strides_w, 0)
        elif self.padding == 'SAME':
            output_h = input_h * strides_h
            output_w = input_w * strides_w
        else:
            if isinstance(self.padding, int):
                output_h = input_h * strides_h + max(kernel_h - strides_h, 0) - 2 * self._padding
                output_w = input_w * strides_w + max(kernel_w - strides_w, 0) - 2 * self._padding
                self.padding = [[0, 0], [self._padding, self._padding],[self._padding, self._padding], [0, 0]]
            else:
                output_h = input_h * strides_h + max(kernel_h - strides_h, 0) - 2 * self._padding[0]
                output_w = input_w * strides_w + max(kernel_w - strides_w, 0) - 2* self._padding[1]
                self.padding = [[0, 0], [self._padding[0], self._padding[0]],[self._padding[1], self._padding[1]], [0, 0]]

        if self.data_format == 'NCHW':
            out_shape = (batch_size, output_channels, output_h, output_w)
        else:
            out_shape = (batch_size, output_h, output_w, output_channels)

        output_shape = tf.stack(out_shape)

        outputs = tf.nn.conv2d_transpose(
            input=input, filters=filters, output_shape=output_shape, strides=self.strides, padding=self.padding,
            data_format=self.data_format, dilations=self.dilations, name=self.name
        )
        return outputs


def conv2d_transpose(
    input, filters, output_shape, strides, padding='SAME', data_format='NHWC', dilations=None, name=None
):
    """
    The transpose of conv2d.

    Parameters
    ----------
    input : tensor
        A 4-D Tensor of type float and shape [batch, height, width, in_channels]
        for NHWC data format or [batch, in_channels, height, width] for NCHW data format.
    filters : tensor
        A 4-D Tensor with the same type as input and shape [height, width,
        output_channels, in_channels]. filter's in_channels dimension must match that of input.
    output_shape : tensor
        A 1-D Tensor representing the output shape of the deconvolution op.
    strides : list
        An int or list of ints that has length 1, 2 or 4. The stride of the sliding window for each dimension of input.
        If a single value is given it is replicated in the H and W dimension.
        By default the N and C dimensions are set to 0.
        The dimension order is determined by the value of data_format, see below for details.
    padding : string
        'VALID' or 'SAME'. The padding algorithm. See the "returns" section of tf.ops.convolution for details.
    data_format : string
         'NHWC' and 'NCHW' are supported.
    dilations : list
        An int or list of ints that has length 1, 2 or 4, defaults to 1.
    name : string
        Optional name for the returned tensor.

    Returns
    -------
        A Tensor with the same type as input.
    """

    data_format, padding = preprocess_2d_format(data_format, padding)
    outputs = tf.nn.conv2d_transpose(
        input=input,
        filters=filters,
        output_shape=output_shape,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name,
    )
    return outputs


class Conv3d_transpose(object):

    def __init__(
        self, strides, padding, data_format='NDHWC', dilations=None, name=None, out_channel=None, k_size=None,
        in_channels=None
    ):
        self.strides = strides
        self.dilations = dilations
        self.name = name
        self.out_channel = out_channel
        self.data_format, self.padding = preprocess_3d_format(data_format, padding)

    def __call__(self, input, filters):
        if self.data_format == 'NDHWC':
            d_axis, h_axis, w_axis = 1, 2, 3
        else:
            d_axis, h_axis, w_axis = 2, 3, 4

        input_shape = input.shape.as_list()
        filters_shape = filters.shape.as_list()
        batch_size = input_shape[0]
        input_d, input_h, input_w = input_shape[d_axis], input_shape[h_axis], input_shape[w_axis]
        kernel_d, kernel_h, kernel_w = filters_shape[0], filters_shape[1], filters_shape[2]
        dilations_d, dilations_h, dilations_w = 1, 1, 1

        if isinstance(self.strides, int):
            strides_d, strides_h, strides_w = self.strides
        else:
            strides_list = list(self.strides)
            if len(strides_list) == 3:
                strides_d, strides_h, strides_w = \
                    strides_list[0], \
                    strides_list[1], \
                    strides_list[2]
            elif len(strides_list) == 5:
                strides_d, strides_h, strides_w = \
                    strides_list[d_axis], \
                    strides_list[h_axis], \
                    strides_list[w_axis]

        if self.dilations is not None:
            if isinstance(self.dilations, int):
                dilations_d, dilations_h, dilations_w = self.dilations
            else:
                dilations_list = list(self.dilations)
                if len(dilations_list) == 3:
                    dilations_d, dilations_h, dilations_w = \
                        dilations_list[0], \
                        dilations_list[1], \
                        dilations_list[2]
                elif len(dilations_list) == 5:
                    dilations_d, dilations_h, dilations_w = \
                        dilations_list[d_axis],\
                        dilations_list[h_axis], \
                        dilations_list[w_axis]

        assert self.padding in {'VALID', 'SAME'}

        kernel_d = kernel_d + (kernel_d - 1) * (dilations_d - 1)
        kernel_h = kernel_h + (kernel_h - 1) * (dilations_h - 1)
        kernel_w = kernel_w + (kernel_w - 1) * (dilations_w - 1)

        if self.padding == 'VALID':
            output_d = input_d * strides_d + max(kernel_d - strides_d, 0)
            output_h = input_h * strides_h + max(kernel_h - strides_h, 0)
            output_w = input_w * strides_w + max(kernel_w - strides_w, 0)
        elif self.padding == 'SAME':
            output_d = input_d * strides_d
            output_h = input_h * strides_h
            output_w = input_w * strides_w

        if self.data_format == 'NDHWC':
            output_shape = (batch_size, output_d, output_h, output_w, self.out_channel)
        else:
            output_shape = (batch_size, self.out_channel, output_d, output_h, output_w)

        output_shape = tf.stack(output_shape)
        outputs = tf.nn.conv3d_transpose(
            input=input, filters=filters, output_shape=output_shape, strides=self.strides, padding=self.padding,
            data_format=self.data_format, dilations=self.dilations, name=self.name
        )

        return outputs


def conv3d_transpose(
    input, filters, output_shape, strides, padding='SAME', data_format='NDHWC', dilations=None, name=None
):
    """
    The transpose of conv3d.

    Parameters
    ----------
    input : tensor
         A 5-D Tensor of type float and shape [batch, height, width, in_channels] for
         NHWC data format or [batch, in_channels, height, width] for NCHW data format.
    filters : tensor
        A 5-D Tensor with the same type as value and shape [height, width, output_channels, in_channels].
        filter's in_channels dimension must match that of value.
    output_shape : tensor
        A 1-D Tensor representing the output shape of the deconvolution op.
    strides : list
        An int or list of ints that has length 1, 3 or 5.
    padding : string
        'VALID' or 'SAME'. The padding algorithm. See the "returns" section of tf.ops.convolution for details.
    data_format : string
        'NDHWC' and 'NCDHW' are supported.
    dilations : list of ints
        An int or list of ints that has length 1, 3 or 5, defaults to 1.
    name : string
        Optional name for the returned tensor.

    Returns
    -------
        A Tensor with the same type as value.
    """

    data_format, padding = preprocess_3d_format(data_format, padding)
    outputs = tf.nn.conv3d_transpose(
        input=input, filters=filters, output_shape=output_shape, strides=strides, padding=padding,
        data_format=data_format, dilations=dilations, name=name
    )
    return outputs


def depthwise_conv2d(input, filters, strides, padding='SAME', data_format='NHWC', dilations=None, name=None):
    """
    Depthwise 2-D convolution.

    Parameters
    ----------
    input : tensor
        4-D with shape according to data_format.
    filters : tensor
        4-D with shape [filter_height, filter_width, in_channels, channel_multiplier].
    strides : tuple
        1-D of size 4. The stride of the sliding window for each dimension of input.
    padding : string
        'VALID' or 'SAME'
    data_format : string
        "NHWC" (default) or "NCHW".
    dilations : tuple
        The dilation rate in which we sample input values across the height and width dimensions in atrous convolution.
        If it is greater than 1, then all values of strides must be 1.
    name : string
        A name for this operation (optional).

    Returns
    -------
        A 4-D Tensor with shape according to data_format.
    """

    data_format, padding = preprocess_2d_format(data_format, padding)
    outputs = tf.nn.depthwise_conv2d(
        input=input,
        filter=filters,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name,
    )
    return outputs


def _to_channel_first_bias(b):
    """Reshape [c] to [c, 1, 1]."""
    channel_size = int(b.shape[0])
    new_shape = (channel_size, 1, 1)
    return tf.reshape(b, new_shape)


def _bias_scale(x, b, data_format):
    """The multiplication counter part of tf.nn.bias_add."""
    if data_format == 'NHWC':
        return x * b
    elif data_format == 'NCHW':
        return x * _to_channel_first_bias(b)
    else:
        raise ValueError('invalid data_format: %s' % data_format)


def _bias_add(x, b, data_format):
    """Alternative implementation of tf.nn.bias_add which is compatiable with tensorRT."""
    if data_format == 'NHWC':
        return tf.add(x, b)
    elif data_format == 'NCHW':
        return tf.add(x, _to_channel_first_bias(b))
    else:
        raise ValueError('invalid data_format: %s' % data_format)


def batch_normalization(x, mean, variance, offset, scale, variance_epsilon, data_format, name=None):
    """Data Format aware version of tf.nn.batch_normalization."""
    if data_format == 'channels_last':
        mean = tf.reshape(mean, [1] * (len(x.shape) - 1) + [-1])
        variance = tf.reshape(variance, [1] * (len(x.shape) - 1) + [-1])
        offset = tf.reshape(offset, [1] * (len(x.shape) - 1) + [-1])
        scale = tf.reshape(scale, [1] * (len(x.shape) - 1) + [-1])
    elif data_format == 'channels_first':
        mean = tf.reshape(mean, [1] + [-1] + [1] * (len(x.shape) - 2))
        variance = tf.reshape(variance, [1] + [-1] + [1] * (len(x.shape) - 2))
        offset = tf.reshape(offset, [1] + [-1] + [1] * (len(x.shape) - 2))
        scale = tf.reshape(scale, [1] + [-1] + [1] * (len(x.shape) - 2))
    else:
        raise ValueError('invalid data_format: %s' % data_format)

    with ops.name_scope(name, 'batchnorm', [x, mean, variance, scale, offset]):
        inv = math_ops.rsqrt(variance + variance_epsilon)
        if scale is not None:
            inv *= scale

        a = math_ops.cast(inv, x.dtype)
        b = math_ops.cast(offset - mean * inv if offset is not None else -mean * inv, x.dtype)
        # Return a * x + b with customized data_format.
        # Currently TF doesn't have bias_scale, and tensorRT has bug in converting tf.nn.bias_add
        # So we reimplemted them to allow make the model work with tensorRT.
        # See https://github.com/tensorlayer/openpose-plus/issues/75 for more details.
        # df = {'channels_first': 'NCHW', 'channels_last': 'NHWC'}
        # return _bias_add(_bias_scale(x, a, df[data_format]), b, df[data_format])
        return a * x + b


class BatchNorm(object):
    """
    The :class:`BatchNorm` is a batch normalization layer for both fully-connected and convolution outputs.
    See ``tf.nn.batch_normalization`` and ``tf.nn.moments``.

    Parameters
    ----------
    decay : float
        A decay factor for `ExponentialMovingAverage`.
        Suggest to use a large value for large dataset.
    epsilon : float
        Eplison.
    act : activation function
        The activation function of this layer.
    is_train : boolean
        Is being used for training or inference.
    beta_init : initializer or None
        The initializer for initializing beta, if None, skip beta.
        Usually you should not skip beta unless you know what happened.
    gamma_init : initializer or None
        The initializer for initializing gamma, if None, skip gamma.
        When the batch normalization layer is use instead of 'biases', or the next layer is linear, this can be
        disabled since the scaling can be done by the next layer. see `Inception-ResNet-v2 <https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py>`__
    moving_mean_init : initializer or None
        The initializer for initializing moving mean, if None, skip moving mean.
    moving_var_init : initializer or None
        The initializer for initializing moving var, if None, skip moving var.
    num_features: int
        Number of features for input tensor. Useful to build layer if using BatchNorm1d, BatchNorm2d or BatchNorm3d,
        but should be left as None if using BatchNorm. Default None.
    data_format : str
        channels_last 'channel_last' (default) or channels_first.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayer

    >>> net = tl.layers.Input([None, 50, 50, 32], name='input')
    >>> net = tl.layers.BatchNorm()(net)

    Notes
    -----
    The :class:`BatchNorm` is universally suitable for 3D/4D/5D input in static model, but should not be used
    in dynamic model where layer is built upon class initialization. So the argument 'num_features' should only be used
    for subclasses :class:`BatchNorm1d`, :class:`BatchNorm2d` and :class:`BatchNorm3d`. All the three subclasses are
    suitable under all kinds of conditions.

    References
    ----------
    - `Source <https://github.com/ry/tensorflow-resnet/blob/master/resnet.py>`__
    - `stackoverflow <http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow>`__

    """

    def __init__(
        self, decay=0.9, epsilon=0.00001, beta=None, gamma=None, moving_mean=None, moving_var=None, num_features=None,
        data_format='channels_last', is_train=False
    ):
        self.decay = decay
        self.epsilon = epsilon
        self.data_format = data_format
        self.beta = beta
        self.gamma = gamma
        self.moving_mean = moving_mean
        self.moving_var = moving_var
        self.num_features = num_features
        self.is_train = is_train
        self.axes = None

        if self.decay < 0.0 or 1.0 < self.decay:
            raise ValueError("decay should be between 0 to 1")

    def _get_param_shape(self, inputs_shape):
        if self.data_format == 'channels_last':
            axis = -1
        elif self.data_format == 'channels_first':
            axis = 1
        else:
            raise ValueError('data_format should be either %s or %s' % ('channels_last', 'channels_first'))

        channels = inputs_shape[axis]
        params_shape = [channels]

        return params_shape

    def _check_input_shape(self, inputs):
        if len(inputs.shape) <= 1:
            raise ValueError('expected input at least 2D, but got {}D input'.format(len(inputs.shape)))

    def __call__(self, inputs):
        self._check_input_shape(inputs)
        self.channel_axis = len(inputs.shape) - 1 if self.data_format == 'channels_last' else 1
        if self.axes is None:
            self.axes = [i for i in range(len(inputs.shape)) if i != self.channel_axis]

        mean, var = tf.nn.moments(inputs, self.axes, keepdims=False)
        if self.is_train:
            # update moving_mean and moving_var
            self.moving_mean = moving_averages.assign_moving_average(
                self.moving_mean, mean, self.decay, zero_debias=False
            )
            self.moving_var = moving_averages.assign_moving_average(self.moving_var, var, self.decay, zero_debias=False)
            outputs = batch_normalization(inputs, mean, var, self.beta, self.gamma, self.epsilon, self.data_format)
        else:
            outputs = batch_normalization(
                inputs, self.moving_mean, self.moving_var, self.beta, self.gamma, self.epsilon, self.data_format
            )

        return outputs


class GroupConv2D(object):

    def __init__(self, strides, padding, data_format, dilations, out_channel, k_size, groups):
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.strides = strides
        self.dilations = dilations
        self.groups = groups
        if self.data_format == 'NHWC':
            self.channels_axis = 3
        else:
            self.channels_axis = 1

    def __call__(self, input, filters):

        if self.groups == 1:
            outputs = tf.nn.conv2d(
                input=input,
                filters=filters,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilations=self.dilations,
            )
        else:
            inputgroups = tf.split(input, num_or_size_splits=self.groups, axis=self.channels_axis)
            weightsgroups = tf.split(filters, num_or_size_splits=self.groups, axis=self.channels_axis)
            convgroups = []
            for i, k in zip(inputgroups, weightsgroups):
                convgroups.append(
                    tf.nn.conv2d(
                        input=i,
                        filters=k,
                        strides=self.strides,
                        padding=self.padding,
                        data_format=self.data_format,
                        dilations=self.dilations,
                    )
                )
            outputs = tf.concat(axis=self.channels_axis, values=convgroups)

        return outputs


class SeparableConv1D(object):

    def __init__(self, stride, padding, data_format, dilations, out_channel, k_size, in_channel, depth_multiplier):
        self.data_format, self.padding = preprocess_1d_format(data_format, padding)

        if self.data_format == 'NWC':
            self.spatial_start_dim = 1
            self.strides = (1, stride, stride, 1)
            self.data_format = 'NHWC'
        else:
            self.spatial_start_dim = 2
            self.strides = (1, 1, stride, stride)
            self.data_format = 'NCHW'
        self.dilation_rate = (1, dilations)

    def __call__(self, inputs, depthwise_filters, pointwise_filters):
        inputs = tf.expand_dims(inputs, axis=self.spatial_start_dim)
        depthwise_filters = tf.expand_dims(depthwise_filters, 0)
        pointwise_filters = tf.expand_dims(pointwise_filters, 0)

        outputs = tf.nn.separable_conv2d(
            inputs, depthwise_filters, pointwise_filters, strides=self.strides, padding=self.padding,
            dilations=self.dilation_rate, data_format=self.data_format
        )

        outputs = tf.squeeze(outputs, axis=self.spatial_start_dim)

        return outputs


class SeparableConv2D(object):

    def __init__(self, strides, padding, data_format, dilations, out_channel, k_size, in_channel, depth_multiplier):
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.strides = strides
        self.dilations = (dilations[2], dilations[2])

    def __call__(self, inputs, depthwise_filters, pointwise_filters):

        outputs = tf.nn.separable_conv2d(
            inputs, depthwise_filters, pointwise_filters, strides=self.strides, padding=self.padding,
            dilations=self.dilations, data_format=self.data_format
        )

        return outputs


class AdaptiveMeanPool1D(object):

    def __init__(self, output_size, data_format):
        self.data_format, _ = preprocess_1d_format(data_format, None)
        self.output_size = output_size

    def __call__(self, input):

        if self.data_format == 'NWC':
            n, w, c = input.shape
        else:
            n, c, w = input.shape

        stride = floor(w / self.output_size)
        kernel = w - (self.output_size - 1) * stride
        output = tf.nn.avg_pool1d(input, ksize=kernel, strides=stride, data_format=self.data_format, padding='VALID')

        return output


class AdaptiveMeanPool2D(object):

    def __init__(self, output_size, data_format):
        self.data_format, _ = preprocess_2d_format(data_format, None)
        self.output_size = output_size

    def __call__(self, inputs):

        if self.data_format == 'NHWC':
            n, h, w, c = inputs.shape
        else:
            n, c, h, w = inputs.shape

        out_h, out_w = self.output_size
        stride_h = floor(h / out_h)
        kernel_h = h - (out_h - 1) * stride_h
        stride_w = floor(w / out_w)
        kernel_w = w - (out_w - 1) * stride_w

        outputs = tf.nn.avg_pool2d(
            inputs, ksize=(kernel_h, kernel_w), strides=(stride_h, stride_w), data_format=self.data_format,
            padding='VALID'
        )

        return outputs


class AdaptiveMeanPool3D(object):

    def __init__(self, output_size, data_format):
        self.data_format, _ = preprocess_3d_format(data_format, None)
        self.output_size = output_size

    def __call__(self, inputs):

        if self.data_format == 'NDHWC':
            n, d, h, w, c = inputs.shape
        else:
            n, c, d, h, w = inputs.shape

        out_d, out_h, out_w = self.output_size
        stride_d = floor(d / out_d)
        kernel_d = d - (out_d - 1) * stride_d
        stride_h = floor(h / out_h)
        kernel_h = h - (out_h - 1) * stride_h
        stride_w = floor(w / out_w)
        kernel_w = w - (out_w - 1) * stride_w

        outputs = tf.nn.avg_pool3d(
            inputs, ksize=(kernel_d, kernel_h, kernel_w), strides=(stride_d, stride_h, stride_w),
            data_format=self.data_format, padding='VALID'
        )

        return outputs


class AdaptiveMaxPool1D(object):

    def __init__(self, output_size, data_format):
        self.data_format, _ = preprocess_1d_format(data_format, None)
        self.output_size = output_size

    def __call__(self, input):

        if self.data_format == 'NWC':
            n, w, c = input.shape
        else:
            n, c, w = input.shape

        stride = floor(w / self.output_size)
        kernel = w - (self.output_size - 1) * stride
        output = tf.nn.max_pool1d(input, ksize=kernel, strides=stride, data_format=self.data_format, padding='VALID')

        return output


class AdaptiveMaxPool2D(object):

    def __init__(self, output_size, data_format):
        self.data_format, _ = preprocess_2d_format(data_format, None)
        self.output_size = output_size

    def __call__(self, inputs):

        if self.data_format == 'NHWC':
            n, h, w, c = inputs.shape
        else:
            n, c, h, w = inputs.shape

        out_h, out_w = self.output_size
        stride_h = floor(h / out_h)
        kernel_h = h - (out_h - 1) * stride_h
        stride_w = floor(w / out_w)
        kernel_w = w - (out_w - 1) * stride_w

        outputs = tf.nn.max_pool2d(
            inputs, ksize=(kernel_h, kernel_w), strides=(stride_h, stride_w), data_format=self.data_format,
            padding='VALID'
        )

        return outputs


class AdaptiveMaxPool3D(object):

    def __init__(self, output_size, data_format):
        self.data_format, _ = preprocess_3d_format(data_format, None)
        self.output_size = output_size

    def __call__(self, inputs):

        if self.data_format == 'NDHWC':
            n, d, h, w, c = inputs.shape
        else:
            n, c, d, h, w = inputs.shape

        out_d, out_h, out_w = self.output_size
        stride_d = floor(d / out_d)
        kernel_d = d - (out_d - 1) * stride_d
        stride_h = floor(h / out_h)
        kernel_h = h - (out_h - 1) * stride_h
        stride_w = floor(w / out_w)
        kernel_w = w - (out_w - 1) * stride_w

        outputs = tf.nn.max_pool3d(
            inputs, ksize=(kernel_d, kernel_h, kernel_w), strides=(stride_d, stride_h, stride_w),
            data_format=self.data_format, padding='VALID'
        )

        return outputs


class BinaryConv2D(object):

    def __init__(self, strides, padding, data_format, dilations, out_channel, k_size, in_channel):
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.strides = strides
        self.dilations = dilations

    # @tf.RegisterGradient("TL_Sign_QuantizeGrad")
    # def _quantize_grad(op, grad):
    #     """Clip and binarize tensor using the straight through estimator (STE) for the gradient."""
    #     return tf.clip_by_value(grad, -1, 1)

    def quantize(self, x):
        # ref: https://github.com/AngusG/tensorflow-xnor-bnn/blob/master/models/binary_net.py#L70
        #  https://github.com/itayhubara/BinaryNet.tf/blob/master/nnUtils.py
        with tf.compat.v1.get_default_graph().gradient_override_map({"Sign": "TL_Sign_QuantizeGrad"}):
            return tf.sign(x)

    def __call__(self, inputs, filters):

        filters = self.quantize(filters)

        outputs = tf.nn.conv2d(
            input=inputs, filters=filters, strides=self.strides, padding=self.padding, data_format=self.data_format,
            dilations=self.dilations
        )

        return outputs


class DorefaConv2D(object):

    def __init__(self, bitW, bitA, strides, padding, data_format, dilations, out_channel, k_size, in_channel):
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.strides = strides
        self.dilations = dilations
        self.bitW = bitW
        self.bitA = bitA

    def _quantize_dorefa(self, x, k):
        G = tf.compat.v1.get_default_graph()
        n = float(2**k - 1)
        with G.gradient_override_map({"Round": "Identity"}):
            return tf.round(x * n) / n

    def cabs(self, x):
        return tf.minimum(1.0, tf.abs(x), name='cabs')

    def quantize_active(self, x, bitA):
        if bitA == 32:
            return x
        return self._quantize_dorefa(x, bitA)

    def quantize_weight(self, x, bitW, force_quantization=False):

        G = tf.compat.v1.get_default_graph()
        if bitW == 32 and not force_quantization:
            return x
        if bitW == 1:  # BWN
            with G.gradient_override_map({"Sign": "Identity"}):
                E = tf.stop_gradient(tf.reduce_mean(input_tensor=tf.abs(x)))
                return tf.sign(x / E) * E
        x = tf.clip_by_value(
            x * 0.5 + 0.5, 0.0, 1.0
        )  # it seems as though most weights are within -1 to 1 region anyways
        return 2 * self._quantize_dorefa(x, bitW) - 1

    def __call__(self, inputs, filters):

        inputs = self.quantize_active(self.cabs(inputs), self.bitA)

        filters = self.quantize_weight(filters, self.bitW)

        outputs = tf.nn.conv2d(
            input=inputs,
            filters=filters,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilations=self.dilations,
        )

        return outputs


class rnncell(object):

    def __init__(self, weight_ih, weight_hh, bias_ih, bias_hh, act):
        self.weight_ih = weight_ih
        self.weight_hh = weight_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self.act_fn = tf.nn.relu if act == 'relu' else tf.nn.tanh

    def __call__(self, input, h):

        i2h = tf.matmul(input, self.weight_ih, transpose_b=True)
        if self.bias_ih is not None:
            i2h += self.bias_ih
        h2h = tf.matmul(h, self.weight_hh, transpose_b=True)
        if self.bias_hh is not None:
            h2h += self.bias_hh
        h = self.act_fn(i2h + h2h)
        return h, h


class lstmcell(object):

    def __init__(self, weight_ih, weight_hh, bias_ih, bias_hh, act=None):
        self.weight_ih = weight_ih
        self.weight_hh = weight_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self.gate_act_fn = tf.sigmoid
        self.act_fn = tf.tanh

    def __call__(self, input, h, c):
        gates = tf.matmul(input, self.weight_ih, transpose_b=True)
        if self.bias_ih is not None:
            gates = gates + self.bias_ih
        gates += tf.matmul(h, self.weight_hh, transpose_b=True)
        if self.bias_hh is not None:
            gates += self.bias_hh

        gate_slices = tf.split(gates, num_or_size_splits=4, axis=-1)
        i = self.gate_act_fn(gate_slices[0])
        f = self.gate_act_fn(gate_slices[1])
        o = self.gate_act_fn(gate_slices[3])
        c = f * c + i * self.act_fn(gate_slices[2])
        h = o * self.act_fn(c)

        return h, h, c


class grucell(object):

    def __init__(self, weight_ih, weight_hh, bias_ih, bias_hh, act=None):
        self.weight_ih = weight_ih
        self.weight_hh = weight_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self.gate_act_fn = tf.sigmoid
        self.act_fn = tf.tanh

    def __call__(self, input, h):

        x_gates = tf.matmul(input, self.weight_ih, transpose_b=True)
        if self.bias_ih is not None:
            x_gates = x_gates + self.bias_ih
        h_gates = tf.matmul(h, self.weight_hh, transpose_b=True)
        if self.bias_hh is not None:
            h_gates = h_gates + self.bias_hh

        x_r, x_z, x_c = tf.split(x_gates, num_or_size_splits=3, axis=-1)
        h_r, h_z, h_c = tf.split(h_gates, num_or_size_splits=3, axis=-1)

        r = self.gate_act_fn(x_r + h_r)
        z = self.gate_act_fn(x_r + h_z)
        c = self.act_fn(x_c + r * h_c)
        h = (h - c) * z + c

        return h, h


class rnnbase(object):

    def __init__(
        self,
        mode,
        input_size,
        hidden_size,
        num_layers,
        bias,
        batch_first,
        dropout,
        bidirectional,
        is_train,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
    ):
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.train = is_train
        if not 0 <= dropout < 1:
            raise ValueError("dropout should be a number in range [0, 1).")
        if dropout > 0 and num_layers == 1:
            raise ValueError(
                "dropout option adds dropout after all but last "
                "recurrent layer, so non-zero dropout expects "
                "num_layers greater than 1, but got dropout={} and "
                "num_layers={}".format(dropout, num_layers)
            )
        self.bidirect = 2 if bidirectional else 1

        self.w_ih = w_ih
        self.w_hh = w_hh
        self.b_ih = b_ih
        self.b_hh = b_hh
        self.act_fn = None
        if mode == 'LSTM':
            # gate_size = 4 * hidden_size
            self.rnn_cell = lstmcell
        elif mode == 'GRU':
            # gate_size = 3 * hidden_size
            self.rnn_cell = grucell
        elif mode == 'RNN_TANH':
            # gate_size = hidden_size
            self.rnn_cell = rnncell
            self.act_fn = 'tanh'
        elif mode == 'RNN_RELU':
            # gate_size = hidden_size
            self.rnn_cell = rnncell
            self.act_fn = 'relu'

    def _bi_rnn_forward(self, x, h, c=None):
        time_step, batch_size, input_size = x.shape
        h_out = []
        c_out = []
        y = []
        pre_layer = x
        for i in range(self.num_layers):
            weight_ih_fw = self.w_ih[2 * i]
            weight_hh_fw = self.w_hh[2 * i]
            weight_ih_bw = self.w_ih[2 * i + 1]
            weight_hh_bw = self.w_hh[2 * i + 1]
            if self.bias:
                bias_ih_fw = self.b_ih[2 * i]
                bias_hh_fw = self.b_hh[2 * i]
                bias_ih_bw = self.b_ih[2 * i + 1]
                bias_hh_bw = self.b_hh[2 * i + 1]
            else:
                bias_ih_fw = None
                bias_hh_fw = None
                bias_ih_bw = None
                bias_hh_bw = None
            h_i_fw = h[i, :, :]
            h_i_bw = h[i + 1, :, :]
            if i != 0 and self.train:
                pre_layer = tf.nn.dropout(pre_layer, rate=self.dropout)
            if c is not None:
                c_i_fw = c[i, :, :]
                c_i_bw = c[i + 1, :, :]
                for j in range(time_step):
                    input = pre_layer[j, :, :]
                    cell_fw = self.rnn_cell(weight_ih_fw, weight_hh_fw, bias_ih_fw, bias_hh_fw, self.act_fn)
                    cell_bw = self.rnn_cell(weight_ih_bw, weight_hh_bw, bias_ih_bw, bias_hh_bw, self.act_fn)
                    bw_input = tf.reverse(input, axis=[0])
                    step_out_fw, h_i_fw, c_i_fw = cell_fw(input, h_i_fw, c_i_fw)
                    step_out_bw, h_i_bw, c_i_bw = cell_bw(bw_input, h_i_bw, c_i_bw)
                    step_out_bw = tf.reverse(step_out_bw, axis=[0])
                    step_out = tf.concat([step_out_fw, step_out_bw], axis=-1)
                    y.append(step_out)
                h_out.append(h_i_fw)
                h_out.append(h_i_bw)
                c_out.append(c_i_fw)
                c_out.append(c_i_bw)
                pre_layer = tf.stack(y)
                y = []
            else:
                for j in range(time_step):
                    input = pre_layer[j, :, :]
                    cell_fw = self.rnn_cell(weight_ih_fw, weight_hh_fw, bias_ih_fw, bias_hh_fw, self.act_fn)
                    cell_bw = self.rnn_cell(weight_ih_bw, weight_hh_bw, bias_ih_bw, bias_hh_bw, self.act_fn)
                    bw_input = tf.reverse(input, axis=[0])
                    step_out_fw, h_i_fw = cell_fw(input, h_i_fw)
                    step_out_bw, h_i_bw = cell_bw(bw_input, h_i_bw)
                    step_out_bw = tf.reverse(step_out_bw, axis=[0])
                    step_out = tf.concat([step_out_fw, step_out_bw], axis=-1)
                    y.append(step_out)
                h_out.append(h_i_fw)
                h_out.append(h_i_bw)
                pre_layer = tf.stack(y)
                y = []
        h_out = tf.stack(h_out)
        c_out = tf.stack(c_out) if c is not None else None

        return pre_layer, h_out, c_out

    def _rnn_forward(self, x, h, c=None):
        pre_layer = x
        h_out = []
        c_out = []
        y = []
        time_step, batch_size, input_size = x.shape
        for i in range(self.num_layers):
            weight_ih = self.w_ih[i]
            weight_hh = self.w_hh[i]
            if self.bias:
                bias_ih = self.b_ih[i]
                bias_hh = self.b_hh[i]
            else:
                bias_ih = None
                bias_hh = None
            h_i = h[i, :, :]
            if i != 0 and self.train:
                pre_layer = tf.nn.dropout(pre_layer, rate=self.dropout)
            if c is not None:
                c_i = c[i, :, :]
                for j in range(time_step):
                    input = pre_layer[j, :, :]
                    cell = self.rnn_cell(weight_ih, weight_hh, bias_ih, bias_hh, self.act_fn)
                    step_out, h_i, c_i = cell(input, h_i, c_i)
                    y.append(step_out)
                h_out.append(h_i)
                c_out.append(c_i)
                pre_layer = tf.stack(y)
                y = []
            else:
                for j in range(time_step):
                    input = pre_layer[j, :, :]
                    cell = self.rnn_cell(weight_hh, weight_ih, bias_ih, bias_hh, self.act_fn)
                    step_out, h_i = cell(input, h_i)
                    y.append(step_out)
                h_out.append(h_i)
                pre_layer = tf.stack(y)
                y = []
        h_out = tf.stack(h_out)
        c_out = tf.stack(c_out) if c is not None else None

        return pre_layer, h_out, c_out

    def check_input(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("input must have 3 dimensions. But got {}.".format(len(input_shape)))
        if self.input_size != input_shape[-1]:
            raise ValueError(
                "The last dimension of input should be equal to input_size {}.But got {}".format(
                    self.input_size, input_shape[-1]
                )
            )

    def check_hidden(self, h, batch_size):
        expected_hidden_size = (self.num_layers * self.bidirect, batch_size, self.hidden_size)
        if h.shape != expected_hidden_size:
            raise ValueError('Expected hidden size {}, got {}.'.format(expected_hidden_size, h.shape))

    def __call__(self, input, initial_states=None):
        if self.batch_first:
            input = tf.transpose(input, perm=(1, 0, 2))
        input_dtype = input.dtype
        input_shape = input.shape
        time_step, batch_size, input_size = input_shape
        self.check_input(input_shape)
        if self.mode == "LSTM":
            if initial_states is not None:
                h, c = initial_states
                self.check_hidden(h, batch_size)
                self.check_hidden(c, batch_size)
            else:
                h = tf.zeros(shape=(self.num_layers * self.bidirect, batch_size, self.hidden_size), dtype=input_dtype)
                c = tf.zeros(shape=(self.num_layers * self.bidirect, batch_size, self.hidden_size), dtype=input_dtype)
            if self.bidirect == 1:
                y, new_h, new_c = self._rnn_forward(input, h, c)
            else:
                y, new_h, new_c = self._bi_rnn_forward(input, h, c)
            new_states = (new_h, new_c)
        else:
            if initial_states is not None:
                h = initial_states
                self.check_hidden(h, batch_size)
            else:
                h = tf.zeros(shape=(self.num_layers * self.bidirect, batch_size, self.hidden_size), dtype=input_dtype)
            if self.bidirect == 1:
                y, new_h, _ = self._rnn_forward(input, h)
            else:
                y, new_h, _ = self._bi_rnn_forward(input, h)
            new_states = new_h
        if self.batch_first:
            y = tf.transpose(y, perm=(1, 0, 2))
        return y, new_states


class layernorm(object):

    def __init__(self, normalized_shape, gamma, beta, eps, input_shape):
        self.normalized_shape = normalized_shape
        self.gamma = gamma
        self.beta = beta
        self.eps = eps
        self.input_shape = input_shape
        self.axis = list(range((len(input_shape) - len(normalized_shape)), len(input_shape)))
        self.ndims = len(input_shape)
        self.broadcast_shape = [1] * self.ndims
        for dim in self.axis:
            self.broadcast_shape[dim] = input_shape[dim]

    def _broadcast(self, v):
        if (v is not None and len(v.shape) != self.ndims and self.axis != [self.ndims - 1]):
            return array_ops.reshape(v, self.broadcast_shape)
        return v

    def __call__(self, input):
        input_dtype = input.dtype
        if input_dtype in ('float16', 'bfloat16'):
            # If mixed precision is used, cast inputs to float32 so that this is at
            # least as numerically stable as the fused version.
            inputs = math_ops.cast(input, 'float32')
        mean, var = tf.nn.moments(input, self.axis, keepdims=True)
        scale, offset = self._broadcast(self.gamma), self._broadcast(self.beta)
        with ops.name_scope(None, 'layernorm', [input, mean, var, scale, offset]):
            inv = math_ops.rsqrt(var + self.eps)
            if scale is not None:
                inv *= scale

            a = math_ops.cast(inv, input.dtype)
            b = math_ops.cast(offset - mean * inv if offset is not None else -mean * inv, input.dtype)

            output = a * input + b
        output = math_ops.cast(output, input_dtype)
        return output


class multiheadattention(object):

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout,
        batch_first,
        need_weights,
        q_weight,
        k_weight,
        v_weight,
        out_weight,
        q_bias,
        k_bias,
        v_bias,
        out_bias,
        train,
    ):

        self.embed_dim_check = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.need_weights = need_weights
        self.q_weight = q_weight
        self.k_weight = k_weight
        self.v_weight = v_weight
        self.out_weight = out_weight
        self.q_bias = q_bias
        self.k_bias = k_bias
        self.v_bias = v_bias
        self.out_bias = out_bias
        self.train = train

    def __call__(self, q, k, v, attn_mask, key_padding_mask):
        k = q if k is None else k
        v = q if v is None else v
        if self.batch_first:
            q = tf.transpose(q, perm=(1, 0, 2))
            k = tf.transpose(k, perm=(1, 0, 2))
            v = tf.transpose(v, perm=(1, 0, 2))

        # check tensor shape
        tgt_len, batch_size, embed_dim = q.shape
        src_len, _, _ = k.shape

        if embed_dim != self.embed_dim_check:
            raise ValueError("Expecting embedding dimension is {}, but got {}".format(self.embed_dim_check, embed_dim))

        head_dim = embed_dim // self.num_heads
        if head_dim * self.num_heads != embed_dim:
            raise ValueError("embedding dimension {} not divisible by num_heads {}".format(embed_dim, self.num_heads))
        if k.shape[:2] != v.shape[:2]:
            raise ValueError(
                "key's sequence length and batch size {} do not match value's {}".format(k.shape[:2], v.shape[:2])
            )

        # compute q k v linear projection
        q = tf.matmul(q, self.q_weight)
        if self.q_bias is not None:
            q = tf.nn.bias_add(q, self.q_bias)
        k = tf.matmul(k, self.k_weight)
        if self.k_bias is not None:
            k = tf.nn.bias_add(k, self.k_bias)
        v = tf.matmul(v, self.v_weight)
        if self.v_bias is not None:
            v = tf.nn.bias_add(v, self.v_bias)

        # check and prep attention mask
        if attn_mask is not None:
            if attn_mask.dtype == tf.uint8:
                warnings.warn("attn_mask tensor dtype should better be bool.")
                attn_mask = tf.cast(attn_mask, dtype=tf.bool)
            elif attn_mask.dtype not in (tf.float32, tf.float64, tf.bool):
                raise TypeError(
                    "attn_mask tensor dtype should be in ('float32', 'float64', 'bool', 'uint8'),"
                    "but got {}".format(attn_mask.dtype)
                )
            if attn_mask._rank() == 2:
                if attn_mask.shape != (tgt_len, src_len):
                    raise ValueError(
                        "The shape of 2D attn_mask should be {}, but got {}.".format(
                            (tgt_len, src_len), attn_mask.shape
                        )
                    )
                attn_mask = tf.expand_dims(attn_mask, axis=0)
                attn_mask = tf.broadcast_to(attn_mask, (batch_size * self.num_heads, tgt_len, src_len))
            elif attn_mask._rank() == 3:
                size_3d = (batch_size * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != size_3d:
                    raise ValueError(
                        "The shape of 3D attn_mask should be {}, but got {}.".format(size_3d, attn_mask.shape)
                    )
            else:
                raise ValueError("attn_mask's dimension {} is not supported.".format(attn_mask.dim()))

        # prep mulithead q k v

        q = tf.transpose(tf.reshape(q, shape=(tgt_len, batch_size * self.num_heads, head_dim)), perm=(1, 0, 2))
        k = tf.transpose(tf.reshape(k, shape=(src_len, batch_size * self.num_heads, head_dim)), perm=(1, 0, 2))
        v = tf.transpose(tf.reshape(v, shape=(src_len, batch_size * self.num_heads, head_dim)), perm=(1, 0, 2))

        #check and prep key padding mask
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch_size, src_len):
                raise ValueError(
                    "Expecting key_padding_mask shape is {}, but got {}.".format(
                        (batch_size, src_len), key_padding_mask.shape
                    )
                )

            if key_padding_mask.dtype == tf.uint8:
                warnings.warn("key_padding_mask tensor dtype should better be bool.")
                key_padding_mask = tf.cast(key_padding_mask, dtype=tf.bool)
            elif key_padding_mask.dtype != tf.bool:
                raise TypeError(
                    "key_padding_mask tensor dtype should be 'bool' or 'uint8', but got {}.".format(
                        key_padding_mask.dtype
                    )
                )

            key_padding_mask = tf.reshape(key_padding_mask, (batch_size, 1, 1, src_len))
            key_padding_mask = tf.broadcast_to(key_padding_mask, (1, self.num_heads, 1, 1))
            key_padding_mask = tf.reshape(key_padding_mask, (batch_size * self.num_heads, 1, src_len))

            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == tf.bool:
                attn_mask = tf.logical_or(attn_mask, key_padding_mask)
            else:
                key_padding_mask_inf = tf.fill(key_padding_mask.shape, float('-inf'))
                attn_mask = tf.where(key_padding_mask, key_padding_mask_inf, attn_mask)

        # convert bool mask to float
        if attn_mask is not None and attn_mask.dtype == tf.bool:
            new_attn_mask_zero = tf.zeros_like(attn_mask, dtype=tf.float32)
            new_attn_mask_inf = tf.fill(attn_mask.shape, float('-inf'))
            attn_mask = tf.where(attn_mask, new_attn_mask_inf, new_attn_mask_zero)

        q = q / math.sqrt(embed_dim)
        k = tf.transpose(k, perm=(0, 2, 1))
        attn = tf.matmul(q, k)
        if attn_mask is not None:
            attn += attn_mask
        attn = tf.nn.softmax(attn)
        if self.train:
            attn = tf.nn.dropout(attn, self.dropout)
        output = tf.matmul(attn, v)

        output = tf.reshape(tf.transpose(output, perm=(1, 0, 2)), shape=(tgt_len, batch_size, embed_dim))
        output = tf.matmul(output, self.out_weight)
        if self.out_bias is not None:
            output = tf.nn.bias_add(output, self.out_bias)

        if self.batch_first:
            output = tf.transpose(output, perm=(1, 0, 2))

        if self.need_weights:
            attn = tf.reshape(attn, shape=(batch_size, self.num_heads, tgt_len, src_len))
            attn = tf.reduce_sum(attn, axis=1) / self.num_heads
            return output, attn
        else:
            return output, None


class BinaryDense(object):

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def __call__(self, inputs):
        self.weights = quantize(self.weights)
        outputs = tf.matmul(inputs, self.weights)

        if self.bias is not None:
            outputs = tf.nn.bias_add(outputs, self.bias)

        return outputs


def quantize(x):
    # ref: https://github.com/AngusG/tensorflow-xnor-bnn/blob/master/models/binary_net.py#L70
    #  https://github.com/itayhubara/BinaryNet.tf/blob/master/nnUtils.py
    with tf.compat.v1.get_default_graph().gradient_override_map({"Sign": "TL_Sign_QuantizeGrad"}):
        return tf.sign(x)


def _quantize_dorefa(x, k):
    G = tf.compat.v1.get_default_graph()
    n = float(2**k - 1)
    with G.gradient_override_map({"Round": "Identity"}):
        return tf.round(x * n) / n


def quantize_active(x, bitA):
    if bitA == 32:
        return x
    return _quantize_dorefa(x, bitA)


def quantize_weight(x, bitW, force_quantization=False):
    G = tf.compat.v1.get_default_graph()
    if bitW == 32 and not force_quantization:
        return x
    if bitW == 1:  # BWN
        with G.gradient_override_map({"Sign": "Identity"}):
            E = tf.stop_gradient(tf.reduce_mean(input_tensor=tf.abs(x)))
            return tf.sign(x / E) * E
    x = tf.clip_by_value(x * 0.5 + 0.5, 0.0, 1.0)  # it seems as though most weights are within -1 to 1 region anyways
    return 2 * _quantize_dorefa(x, bitW) - 1


def cabs(x):
    return tf.minimum(1.0, tf.abs(x), name='cabs')


def _compute_threshold(x):
    """
    ref: https://github.com/XJTUWYD/TWN
    Computing the threshold.
    """
    x_sum = tf.reduce_sum(input_tensor=tf.abs(x), axis=None, keepdims=False, name=None)
    # threshold = tf.compat.v1.div(x_sum, tf.cast(tf.size(input=x), tf.float32), name=None)
    threshold = tf.math.divide(x_sum, tf.cast(tf.size(input=x), tf.float32), name=None)
    threshold = tf.multiply(0.7, threshold, name=None)
    return threshold


def compute_alpha(x):
    """Computing the scale parameter."""
    threshold = _compute_threshold(x)
    alpha1_temp1 = tf.where(tf.greater(x, threshold), x, tf.zeros_like(x, tf.float32))
    alpha1_temp2 = tf.where(tf.less(x, -threshold), x, tf.zeros_like(x, tf.float32))
    alpha_array = tf.add(alpha1_temp1, alpha1_temp2, name=None)
    alpha_array_abs = tf.abs(alpha_array)
    alpha_array_abs1 = tf.where(
        tf.greater(alpha_array_abs, 0), tf.ones_like(alpha_array_abs, tf.float32),
        tf.zeros_like(alpha_array_abs, tf.float32)
    )
    alpha_sum = tf.reduce_sum(input_tensor=alpha_array_abs)
    n = tf.reduce_sum(input_tensor=alpha_array_abs1)
    # alpha = tf.compat.v1.div(alpha_sum, n)
    alpha = tf.math.divide(alpha_sum, n)
    return alpha


def _quantize_overflow(x, k):
    G = tf.compat.v1.get_default_graph()
    n = float(2**k - 1)
    max_value = tf.reduce_max(input_tensor=x)
    min_value = tf.reduce_min(input_tensor=x)
    with G.gradient_override_map({"Round": "Identity"}):
        step = tf.stop_gradient((max_value - min_value) / n)
        return tf.round((tf.maximum(tf.minimum(x, max_value), min_value) - min_value) / step) * step + min_value


def quantize_active_overflow(x, bitA):
    if bitA == 32:
        return x
    return _quantize_overflow(x, bitA)


def quantize_weight_overflow(x, bitW):
    if bitW == 32:
        return x
    return _quantize_overflow(x, bitW)


def ternary_operation(x):
    """Ternary operation use threshold computed with weights."""
    g = tf.compat.v1.get_default_graph()
    with g.gradient_override_map({"Sign": "Identity"}):
        threshold = _compute_threshold(x)
        x = tf.sign(tf.add(tf.sign(tf.add(x, threshold)), tf.sign(tf.add(x, -threshold))))
        return x


def mean_var_with_update(update_moving_mean, update_moving_variance, mean, variance):
    with tf.control_dependencies([update_moving_mean, update_moving_variance]):
        return tf.identity(mean), tf.identity(variance)


def w_fold(w, gama, var, epsilon):
    return tf.compat.v1.div(tf.multiply(gama, w), tf.sqrt(var + epsilon))


def bias_fold(beta, gama, mean, var, epsilon):
    return tf.subtract(beta, tf.compat.v1.div(tf.multiply(gama, mean), tf.sqrt(var + epsilon)))


class DorefaDense(object):

    def __init__(self, weights, bias, bitW, bitA):
        self.weights = weights
        self.bias = bias
        self.bitW = bitW
        self.bitA = bitA

    def __call__(self, inputs):

        inputs = quantize_active(cabs(inputs), self.bitA)
        self.W = quantize_weight(self.weights, self.bitW)
        outputs = tf.matmul(inputs, self.W)
        if self.bias is not None:
            outputs = tf.nn.bias_add(outputs, self.bias)

        return outputs


class TernaryDense(object):

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def __call__(self, inputs):
        alpha = compute_alpha(self.weights)
        W_ = ternary_operation(self.weights)
        W_ = tf.math.multiply(alpha, W_)
        outputs = tf.matmul(inputs, W_)

        if self.bias is not None:
            outputs = tf.nn.bias_add(outputs, self.bias)

        return outputs


class QuanDense(object):

    def __init__(self, weights, bias, bitW, bitA):
        self.weights = weights
        self.bias = bias
        self.bitW = bitW
        self.bitA = bitA

    def __call__(self, inputs):
        inputs = quantize_active_overflow(inputs, self.bitA)
        W_ = quantize_weight_overflow(self.weights, self.bitW)
        outputs = tf.matmul(inputs, W_)
        if self.bias is not None:
            outputs = tf.nn.bias_add(outputs, self.bias)
        return outputs


class QuanDenseBn(object):

    def __init__(
        self, weights, scale_para, offset_para, moving_mean, moving_variance, decay, bitW, bitA, epsilon, is_train
    ):
        self.weights = weights
        self.scale_para = scale_para
        self.offset_para = offset_para
        self.moving_mean = moving_mean
        self.moving_variance = moving_variance
        self.decay = decay
        self.bitW = bitW
        self.bitA = bitA
        self.epsilon = epsilon
        self.is_train = is_train

    def __call__(self, inputs):
        x = inputs
        inputs = quantize_active_overflow(inputs, self.bitA)
        mid_out = tf.matmul(x, self.weights)
        mean, variance = moments(x=mid_out, axes=list(range(len(mid_out.get_shape()) - 1)))
        update_moving_mean = moving_averages.assign_moving_average(
            self.moving_mean, mean, self.decay, zero_debias=False
        )
        update_moving_variance = moving_averages.assign_moving_average(
            self.moving_variance, variance, self.decay, zero_debias=False
        )

        if self.is_train:
            mean, var = mean_var_with_update(update_moving_mean, update_moving_variance, mean, variance)
        else:
            mean, var = self.moving_mean, self.moving_variance

        _w_fold = w_fold(self.weights, self.scale_para, var, self.epsilon)
        W = quantize_weight_overflow(_w_fold, self.bitW)
        outputs = tf.matmul(inputs, W)

        if self.offset_para is not None:
            _bias_fold = bias_fold(self.offset_para, self.scale_para, mean, var, self.epsilon)
            outputs = tf.nn.bias_add(outputs, _bias_fold)

        return outputs


class TernaryConv(object):

    def __init__(self, weights, strides, padding, data_format, dilations):
        self.weights = weights
        self.strides = strides
        self.dilations = dilations
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)

    def __call__(self, inputs):
        alpha = compute_alpha(self.weights)
        W_ = ternary_operation(self.weights)
        W_ = tf.multiply(alpha, W_)
        outputs = tf.nn.conv2d(
            input=inputs,
            filters=W_,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilations=self.dilations,
        )
        return outputs


class QuanConv(object):

    def __init__(self, weights, strides, padding, data_format, dilations, bitW, bitA):
        self.weights = weights
        self.strides = strides
        self.dilations = dilations
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.bitW = bitW
        self.bitA = bitA

    def __call__(self, inputs):
        inputs = quantize_active_overflow(inputs, self.bitA)
        W_ = quantize_weight_overflow(self.weights, self.bitW)
        outputs = tf.nn.conv2d(
            input=inputs,
            filters=W_,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilations=self.dilations,
        )

        return outputs


class QuanConvBn(object):

    def __init__(
        self, weights, scale_para, offset_para, moving_mean, moving_variance, strides, padding, data_format, dilations,
        bitW, bitA, decay, epsilon, is_train
    ):
        self.weights = weights
        self.strides = strides
        self.dilations = dilations
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.bitW = bitW
        self.bitA = bitA
        self.scale_para = scale_para
        self.offset_para = offset_para
        self.moving_mean = moving_mean
        self.moving_variance = moving_variance
        self.decay = decay
        self.epsilon = epsilon
        self.is_train = is_train

    def __call__(self, inputs):
        x = inputs
        inputs = quantize_active_overflow(inputs, self.bitA)
        outputs = tf.nn.conv2d(
            input=x, filters=self.weights, strides=self.strides, padding=self.padding, data_format=self.data_format,
            dilations=self.dilations
        )
        mean, variance = tf.nn.moments(outputs, axes=list(range(len(outputs.get_shape()) - 1)))
        update_moving_mean = moving_averages.assign_moving_average(
            self.moving_mean, mean, self.decay, zero_debias=False
        )
        update_moving_variance = moving_averages.assign_moving_average(
            self.moving_variance, variance, self.decay, zero_debias=False
        )

        if self.is_train:
            mean, var = mean_var_with_update(update_moving_mean, update_moving_variance, mean, variance)
        else:
            mean, var = self.moving_mean, self.moving_variance

        _w_fold = w_fold(self.weights, self.scale_para, var, self.epsilon)

        W_ = quantize_weight_overflow(_w_fold, self.bitW)

        conv_fold = tf.nn.conv2d(inputs, W_, strides=self.strides, padding=self.padding, data_format=self.data_format)

        if self.offset_para is not None:
            _bias_fold = bias_fold(self.offset_para, self.scale_para, mean, var, self.epsilon)
            conv_fold = tf.nn.bias_add(conv_fold, _bias_fold, name='bn_bias_add')

        return conv_fold

class PReLU(object):

    def __init__(self, data_format):

        self.data_format = data_format

    def __call__(self, input, weight):

        pos = tf.nn.relu(input)
        neg = -tf.nn.sigmoid(weight) * tf.nn.relu(-input)
        return pos + neg


def prelu(input, weight, data_format):

    pos = tf.nn.relu(input)
    neg = -tf.nn.sigmoid(weight) * tf.nn.relu(-input)
    return pos + neg