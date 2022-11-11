#! /usr/bin/python
# -*- coding: utf-8 -*-

import paddle as pd
import paddle.nn
from paddle import framework
import paddle.nn.functional as F
import numpy as np
import paddle.fluid as fluid
from paddle.nn import initializer as I
from paddle.fluid.layers.utils import map_structure, flatten, pack_sequence_as
from paddle.fluid.data_feeder import convert_dtype
from paddle.fluid.dygraph import Layer, LayerList
from paddle.nn.layer.rnn import RNNCellBase
import warnings
import math
from paddle import _C_ops
from paddle.framework import core
from paddle import in_dynamic_mode

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

    if data_format in ["channels_last", "NWC", "NLC"]:
        data_format = "NLC"
    elif data_format in ["channels_first", "NCW", "NCL"]:
        data_format = "NCL"
    elif data_format == None:
        data_format = None
    else:
        raise Exception("Unsupported data format: " + str(data_format))
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

    if data_format in ["channels_last", "NHWC", "nhwc"]:
        data_format = "NHWC"
    elif data_format in ["channels_first", "NCHW", "nchw"]:
        data_format = "NCHW"
    elif data_format == None:
        data_format = None
    else:
        raise Exception("Unsupported data format: " + str(data_format))
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

    if data_format in ['channels_last', 'NDHWC']:
        data_format = 'NDHWC'
    elif data_format in ['channels_first', 'NCDHW']:
        data_format = 'NCDHW'
    elif data_format == None:
        data_format = None
    else:
        raise Exception("Unsupported data format: " + str(data_format))
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
        x = pd.transpose(x, (0, 2, 1))
    elif len(x.shape) == 4:
        x = pd.transpose(x, (0, 2, 3, 1))
    elif len(x.shape) == 5:
        x = pd.transpose(x, (0, 2, 3, 4, 1))
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
        x = pd.transpose(x, (0, 2, 1))
    elif len(x.shape) == 4:
        x = pd.transpose(x, (0, 3, 1, 2))
    elif len(x.shape) == 5:
        x = pd.transpose(x, (0, 4, 1, 2, 3))
    else:
        raise Exception("Unsupported dimensions")
    return x


class ReLU(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return F.relu(x)


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
    return F.relu(x)


class ELU(object):

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, x):
        return F.elu(x, alpha=self.alpha)


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

    return F.elu(x, alpha=alpha)


class ReLU6(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return F.relu6(x)


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
    return F.relu6(x)


class LeakyReLU(object):

    def __init__(self, negative_slope=0.2):
        self.negative_slope = negative_slope

    def __call__(self, x):
        return F.leaky_relu(x, negative_slope=self.negative_slope)


def leaky_relu(x, negative_slope=0.01):
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

    return F.leaky_relu(x, negative_slope)


class Softplus(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return F.softplus(x)


class Tanh(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return F.tanh(x)


class Sigmoid(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return F.sigmoid(x)


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
    return F.sigmoid(x)


class Softmax(object):

    def __init__(self, axis = -1):
        self.axis = axis

    def __call__(self, x):
        return F.softmax(x, axis=self.axis)


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
    return F.softmax(logits, axis=axis)


class GeLU(object):

    def __init__(self, approximate=False):
        self.approximate = approximate

    def __call__(self, x):
        return F.gelu(x, approximate=self.approximate)


def gelu(x, approximate=False):

    return F.gelu(x, approximate=approximate)


class Dropout(object):

    def __init__(self, p, seed=1):
        self.p = p
        self.seed = seed

    def __call__(self, inputs):
        output = F.dropout(inputs, p=self.p, mode='upscale_in_train')
        return output


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
        super(BiasAdd, self).__init__()
        if data_format in ['channels_first', 'NCL', 'NCW', 'NCHW', 'NCDHW']:
            self.data_format = 'channels_first'
        elif data_format in ['channels_last', 'NLC', 'NWC', 'NHWC', 'NDHWC']:
            self.data_format = 'channels_last'
        else:
            raise ("Unsupported data format: " + str(data_format))

    def __call__(self, x, bias):
        if len(x.shape) > 2 and self.data_format == 'channels_first':
            x = nchw_to_nhwc(x)
        outputs = pd.add(x, bias)
        if len(x.shape) > 2 and self.data_format == 'channels_first':
            outputs = nhwc_to_nchw(outputs)
        return outputs


def bias_add(x, bias):
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

    #TODO the bias_add only supports channels_last
    outputs = pd.add(x, bias)
    return outputs


class Conv1D(object):

    def __init__(self, stride, padding, data_format='NWC', dilations=None, out_channel=None, k_size=None):
        super(Conv1D, self).__init__()
        self.data_format, self.padding = preprocess_1d_format(padding=padding, data_format=data_format)
        self.stride = stride
        self.dilations = dilations

    def __call__(self, input, filters):
        output = F.conv1d(
            x=input, weight=filters, stride=self.stride, dilation=self.dilations, data_format=self.data_format,
            padding=self.padding
        )
        return output


def conv1d(input, filters, stride, padding, data_format='NWC', dilations=None, name=None):
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

    outputs = F.conv1d(
        x=input, weight=filters, stride=stride, padding=padding, data_format=data_format, dilation=dilations, name=name
    )
    return outputs


class Conv2D(object):

    def __init__(self, strides, padding, data_format='NHWC', dilations=None, out_channel=None, k_size=None):
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        if self.data_format is 'NHWC':
            self._stride = (strides[1], strides[2])
            self._dilation = (dilations[1], dilations[2])
        elif self.data_format is 'NCHW':
            self._stride = (strides[2], strides[3])
            self._dilation = (dilations[2], dilations[3])

    def __call__(self, inputs, filters):
        outputs = F.conv2d(
            x=inputs, weight=filters, stride=self._stride, dilation=self._dilation, padding=self.padding,
            data_format=self.data_format
        )
        return outputs


def conv2d(input, filters, strides, padding, data_format='NCHW', dilations=None):
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
        "NHWC", "NCHW". Defaults to "NCHW".
    dilations : list or ints
        list of ints that has length 1, 2 or 4, defaults to 1. The dilation factor for each dimension ofinput.

    Returns
    -------
        A Tensor. Has the same type as input.
    """
    data_format, padding = preprocess_2d_format(data_format, padding)
    if data_format is 'NHWC':
        _stride = (strides[1], strides[2])
        _dilation = (dilations[1], dilations[2])
    elif data_format is 'NCHW':
        _stride = (strides[2], strides[3])
        _dilation = (dilations[2], dilations[3])
    outputs = F.conv2d(
        x=input, weight=filters, stride=_stride, dilation=_dilation, padding=padding, data_format=data_format
    )
    return outputs


class Conv3D(object):

    def __init__(self, strides, padding, data_format='NDHWC', dilations=None, out_channel=None, k_size=None):
        self.data_format, self.padding = preprocess_3d_format(data_format, padding)
        if self.data_format is 'NDHWC':
            self._strides = (strides[1], strides[2], strides[3])
            self._dilations = (dilations[1], dilations[2], dilations[3])
        elif self.data_format is 'NCDHW':
            self._strides = (strides[2], strides[3], strides[4])
            self._dilations = (dilations[2], dilations[3], dilations[4])

    def __call__(self, input, filters):
        outputs = F.conv3d(
            x=input, weight=filters, stride=self._strides, dilation=self._dilations, data_format=self.data_format,
            padding=self.padding
        )
        return outputs


def conv3d(input, filters, strides, padding, data_format='NDHWC', dilations=None, name=None):
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
    strides : tuple of ints
        A list of ints that has length >= 5. 1-D tensor of length 5.
        The stride of the sliding window for each dimension of input.
        Must have strides[0] = strides[4] = 1.
    padding : string
        A string from: "SAME", "VALID". The type of padding algorithm to use.
    data_format : string
        An optional string from: "NDHWC", "NCDHW". Defaults to "NDHWC". The data format of the input and output data.
        With the default format "NDHWC", the data is stored in the order of: [batch, in_depth, in_height, in_width, in_channels].
        Alternatively, the format could be "NCDHW", the data storage order is: [batch, in_channels, in_depth, in_height, in_width].
    dilations : touple of ints
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
    if data_format is 'NDHWC':
        _strides = (strides[1], strides[2], strides[3])
        _dilations = (dilations[1], dilations[2], dilations[3])
    elif data_format is 'NCDHW':
        _strides = (strides[2], strides[3], strides[4])
        _dilations = (dilations[2], dilations[3], dilations[4])
    outputs = F.conv3d(
        x=input, weight=filters, stride=_strides, dilation=_dilations, data_format=data_format, padding=padding,
        name=name
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
    pass


def moments(x, axes, shift=None, keepdims=False):
    """
    Calculates the mean and variance of x.

    Parameters
    ----------
    x : tensor
        A Tensor
    axes : ints
        Axes along which to compute mean and variance.
    shift : int
        Not used in the current implementation.
    keepdims : bool
        produce moments with the same dimensionality as the input.

    Returns
    -------
        Two Tensor objects: mean and variance.
    """

    pass


class MaxPool1d(object):

    def __init__(self, ksize, strides, padding, data_format=None):
        self.data_format, self.padding = preprocess_1d_format(data_format=data_format, padding=padding)
        self.ksize = ksize
        self.strides = strides

    def __call__(self, inputs):
        if self.data_format == 'NLC':
            inputs = nhwc_to_nchw(inputs)
        outputs = F.max_pool1d(inputs, self.ksize, self.strides, self.padding)
        if self.data_format == 'NLC':
            outputs = nchw_to_nhwc(outputs)
        return outputs


class MaxPool(object):

    def __init__(self, ksize, strides, padding, data_format=None):
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.ksize = ksize
        self.strides = strides

    def __call__(self, inputs):
        outputs = F.max_pool2d(
            x=inputs, kernel_size=self.ksize, stride=self.strides, padding=self.padding, data_format=self.data_format
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

    Returns
    -------
        A Tensor of format specified by data_format. The max pooled output tensor.
    """
    pass


class AvgPool1d(object):

    def __init__(self, ksize, strides, padding, data_format=None):
        self.data_format, self.padding = preprocess_1d_format(data_format=data_format, padding=padding)
        self.ksize = ksize
        self.strides = strides

    def __call__(self, inputs):
        if self.data_format == 'NLC':
            inputs = nhwc_to_nchw(inputs)
        outputs = F.avg_pool1d(inputs, self.ksize, self.strides, self.padding)
        if self.data_format == 'NLC':
            outputs = nchw_to_nhwc(outputs)
        return outputs


class AvgPool(object):

    def __init__(self, ksize, strides, padding, data_format=None):
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.filter_size = ksize
        self._stride = strides

    def __call__(self, inputs):
        outputs = F.avg_pool2d(
            inputs, kernel_size=self.filter_size, stride=self._stride, padding=self.padding,
            data_format=self.data_format
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

    Returns
    -------
        A Tensor of format specified by data_format. The average pooled output tensor.
    """
    pass


class MaxPool3d(object):

    def __init__(self, ksize, strides, padding, data_format=None):
        self.data_format, self.padding = preprocess_3d_format(data_format, padding)
        self.ksize = ksize
        self.strides = strides

    def __call__(self, inputs):
        outputs = F.max_pool3d(
            inputs, kernel_size=self.ksize, stride=self.strides, padding=self.padding, data_format=self.data_format
        )
        return outputs


def max_pool3d(input, ksize, strides, padding, data_format=None, name=None):
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
    pass


class AvgPool3d(object):

    def __init__(self, ksize, strides, padding, data_format=None):
        self.data_format, self.padding = preprocess_3d_format(data_format, padding)
        self.ksize = ksize
        self.strides = strides

    def __call__(self, inputs):
        outputs = F.avg_pool3d(
            inputs, kernel_size=self.ksize, stride=self.strides, padding=self.padding, data_format=self.data_format
        )
        return outputs


def avg_pool3d(input, ksize, strides, padding, data_format=None, name=None):
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
    pass


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
    pass


class DepthwiseConv2d(object):

    def __init__(self, strides, padding, data_format=None, dilations=None, ksize=None, channel_multiplier=1, in_channels=None):
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self._stride = strides
        self.dilations = dilations
        self.in_channel = in_channels

    def __call__(self, input, filter, point_filter=None):
        depthwise_conv = F.conv2d(
            input, filter, data_format=self.data_format, groups=self.in_channel, dilation=self.dilations, stride=self._stride,
            padding=self.padding
        )
        pointwise_conv = F.conv2d(depthwise_conv, point_filter, data_format=self.data_format, padding=self.padding)
        return pointwise_conv


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

    pass


class Conv1d_transpose(object):

    def __init__(
        self, stride, padding, data_format='NWC', dilations=None, out_channel=None, k_size=None, in_channels=None
    ):
        self.stride = stride
        self.dilations = dilations
        self.data_format, self.padding = preprocess_1d_format(data_format, padding)

    def __call__(self, input, filters):
        out = F.conv1d_transpose(
            x=input,
            weight=filters,
            padding=self.padding,
            stride=self.stride,
            dilation=self.dilations,
            data_format=self.data_format,
        )
        return out


def conv1d_transpose(
    input, filters, output_shape, stride, padding='SAME', data_format='NWC', dilations=None, name=None
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
    output = F.conv1d_transpose(
        x=input,
        weight=filters,
        stride=stride,
        padding=padding,
        dilation=dilations,
        data_format=data_format,
        output_size=output_shape,
    )
    return output


class Conv2d_transpose(object):

    def __init__(
        self, strides, padding, data_format='NHWC', dilations=None, name=None, out_channel=None, k_size=None,
        in_channels=None
    ):
        self.strides = strides
        self.dilations = dilations
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)

    def __call__(self, input, filters):
        output = F.conv2d_transpose(
            x=input, weight=filters, stride=self.strides, padding=self.padding, dilation=self.dilations,
            data_format=self.data_format
        )
        return output


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
    output = F.conv2d_transpose(
        x=input,
        weight=filters,
        output_size=output_shape,
        stride=strides,
        padding=padding,
        dilation=dilations,
        data_format=data_format,
    )
    return output


class Conv3d_transpose(object):

    def __init__(
        self, strides, padding, data_format='NDHWC', dilations=None, name=None, out_channel=None, k_size=None,
        in_channels=None
    ):
        self.strides = strides
        self.dilations = dilations
        self.data_format, self.padding = preprocess_3d_format(data_format, padding)

    def __call__(self, input, filters):
        output = F.conv3d_transpose(
            x=input, weight=filters, stride=self.strides, padding=self.padding, dilation=self.dilations,
            data_format=self.data_format
        )
        return output


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
    output = F.conv3d_transpose(
        x=input,
        weight=filters,
        output_size=output_shape,
        stride=strides,
        padding=padding,
        dilation=dilations,
        data_format=data_format,
    )
    return output


class BatchNorm(object):

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

    def __call__(self, inputs):
        data_format = self.channel_format(inputs)
        outputs = pd.nn.functional.batch_norm(
            inputs, self.moving_mean, self.moving_var, weight=self.gamma, bias=self.beta, training=self.is_train,
            momentum=self.decay, epsilon=self.epsilon, data_format=data_format
        )
        return outputs

    def channel_format(self, inputs):
        """ return "NC", "NCL", "NCHW", "NCDHW", "NLC", "NHWC" or "NDHWC". """
        len_in_shape = len(inputs.shape)
        if len_in_shape == 2:
            return 'NC'
        if self.data_format == 'channels_last':
            if len_in_shape == 3:
                return 'NLC'
            if len_in_shape == 4:
                return 'NHWC'
            if len_in_shape == 5:
                return 'NDHWC'
        if self.data_format == 'channels_first':
            if len_in_shape == 3:
                return 'NCL'
            if len_in_shape == 4:
                return 'NCHW'
            if len_in_shape == 5:
                return 'NCDHW'


class GroupConv2D(object):

    def __init__(self, strides, padding, data_format, dilations, out_channel, k_size, groups):
        self.out_channel = out_channel
        self.k_size = k_size
        self.groups = groups
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        if self.data_format is 'NHWC':
            self.strides = (strides[1], strides[2])
            self.dilations = (dilations[1], dilations[2])
        elif self.data_format is 'NCHW':
            self.strides = (strides[2], strides[3])
            self.dilations = (dilations[2], dilations[3])

    def __call__(self, inputs, filters):
        outputs = F.conv2d(
            inputs, weight=filters, stride=self.strides, padding=self.padding, dilation=self.dilations,
            groups=self.groups, data_format=self.data_format
        )
        return outputs


class SeparableConv1D(object):

    def __init__(self, stride, padding, data_format, dilations, out_channel, k_size, in_channel, depth_multiplier):
        self.stride = stride
        self.dilations = dilations
        self.out_channel = out_channel
        self.k_size = k_size
        self.in_channel = int(in_channel)
        self.depth_multiplier = depth_multiplier
        self.data_format, self.padding = preprocess_1d_format(data_format, padding)

    def __call__(self, inputs, depthwise_filters, pointwise_filters):
        outputs = F.conv1d(
            inputs, weight=depthwise_filters, stride=self.stride, padding=self.padding, dilation=self.dilations,
            groups=self.in_channel, data_format=self.data_format
        )
        outputs = F.conv1d(
            outputs, weight=pointwise_filters, stride=1, padding=self.padding, dilation=1, groups=1,
            data_format=self.data_format
        )
        return outputs


class SeparableConv2D(object):

    def __init__(self, strides, padding, data_format, dilations, out_channel, k_size, in_channel, depth_multiplier):
        self.out_channel = out_channel
        self.k_size = k_size
        self.in_channel = int(in_channel)
        self.depth_multiplier = depth_multiplier
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        if self.data_format is 'NHWC':
            self.strides = (strides[1], strides[2])
            self.dilations = (dilations[1], dilations[2])
        elif self.data_format is 'NCHW':
            self.strides = (strides[2], strides[3])
            self.dilations = (dilations[2], dilations[3])

    def __call__(self, inputs, depthwise_filters, pointwise_filters):
        outputs = F.conv2d(
            inputs, weight=depthwise_filters, stride=self.strides, padding=self.padding, dilation=self.dilations,
            groups=self.in_channel, data_format=self.data_format
        )
        outputs = F.conv2d(
            outputs, weight=pointwise_filters, stride=1, padding=self.padding, dilation=1, groups=1,
            data_format=self.data_format
        )
        return outputs


class AdaptiveMeanPool1D(object):

    def __init__(self, output_size, data_format):
        self.data_format, _ = preprocess_1d_format(data_format, None)
        self.output_size = output_size

    def __call__(self, input):

        if self.data_format == 'NLC':
            input = nhwc_to_nchw(input)

        output = F.adaptive_avg_pool1d(input, self.output_size)

        if self.data_format == 'NLC':
            output = nchw_to_nhwc(output)

        return output


class AdaptiveMeanPool2D(object):

    def __init__(self, output_size, data_format):
        self.data_format, _ = preprocess_2d_format(data_format, None)
        self.output_size = output_size

    def __call__(self, inputs):

        return F.adaptive_avg_pool2d(inputs, output_size=self.output_size, data_format=self.data_format)


class AdaptiveMeanPool3D(object):

    def __init__(self, output_size, data_format):
        self.data_format, _ = preprocess_3d_format(data_format, None)
        self.output_size = output_size

    def __call__(self, inputs):

        return F.adaptive_avg_pool3d(inputs, output_size=self.output_size, data_format=self.data_format)


class AdaptiveMaxPool1D(object):

    def __init__(self, output_size, data_format):

        self.data_format, _ = preprocess_1d_format(data_format, None)
        self.output_size = output_size

    def __call__(self, input):

        if self.data_format == 'NLC':
            input = nhwc_to_nchw(input)

        output = F.adaptive_max_pool1d(input, self.output_size)

        if self.data_format == 'NLC':
            output = nchw_to_nhwc(output)

        return output


class AdaptiveMaxPool2D(object):

    def __init__(self, output_size, data_format):
        self.data_format, _ = preprocess_2d_format(data_format, None)
        self.output_size = output_size

    def __call__(self, inputs):
        if self.data_format == 'NHWC':
            inputs = nhwc_to_nchw(inputs)

        output = F.adaptive_max_pool2d(inputs, self.output_size)

        if self.data_format == 'NHWC':
            output = nchw_to_nhwc(output)

        return output


class AdaptiveMaxPool3D(object):

    def __init__(self, output_size, data_format):
        self.data_format, _ = preprocess_3d_format(data_format, None)
        self.output_size = output_size

    def __call__(self, inputs):
        if self.data_format == 'NDHWC':
            inputs = nhwc_to_nchw(inputs)

        output = F.adaptive_max_pool3d(inputs, self.output_size)

        if self.data_format == 'NDHWC':
            output = nchw_to_nhwc(output)

        return output


class BinaryConv2D(object):

    def __init__(self, strides, padding, data_format, dilations, out_channel, k_size, in_channel):
        pass

    def __call__(self, inputs, filters):
        raise NotImplementedError


class DorefaConv2D(object):

    def __init__(self, bitW, bitA, strides, padding, data_format, dilations, out_channel, k_size, in_channel):
        pass

    def __call__(self, inputs, filters):
        raise NotImplementedError


class rnncell(RNNCellBase):

    def __init__(self, weight_ih, weight_hh, bias_ih, bias_hh, act):
        super(rnncell, self).__init__()
        self.weight_ih = weight_ih
        self.weight_hh = weight_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self.act_fn = F.relu if act == 'relu' else F.tanh
        self.input_size = weight_ih.shape[1]

    def forward(self, input, h):

        i2h = pd.matmul(input, self.weight_ih, transpose_y=True)
        if self.bias_ih is not None:
            i2h += self.bias_ih
        h2h = pd.matmul(h, self.weight_hh, transpose_y=True)
        if self.bias_hh is not None:
            h2h += self.bias_hh
        h = self.act_fn(i2h + h2h)
        return h, h


class lstmcell(RNNCellBase):

    def __init__(self, weight_ih, weight_hh, bias_ih, bias_hh, act=None):
        super(lstmcell, self).__init__()
        self.weight_ih = weight_ih
        self.weight_hh = weight_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self.gate_act_fn = F.sigmoid
        self.act_fn = F.tanh
        self.input_size = weight_ih.shape[1]

    def forward(self, inputs, h, c):
        gates = pd.matmul(inputs, self.weight_ih, transpose_y=True)
        if self.bias_ih is not None:
            gates += self.bias_ih
        gates += pd.matmul(h, self.weight_hh, transpose_y=True)
        if self.bias_hh is not None:
            gates += self.bias_hh

        gates_slices = pd.split(gates, num_or_sections=4, axis=-1)

        i = self.gate_act_fn(gates_slices[0])
        f = self.gate_act_fn(gates_slices[1])
        o = self.gate_act_fn(gates_slices[3])
        c = f * c + i * self.act_fn(gates_slices[2])
        h = o * self.act_fn(c)

        return h, h, c


class grucell(RNNCellBase):

    def __init__(self, weight_ih, weight_hh, bias_ih, bias_hh, act=None):
        super(grucell, self).__init__()
        self.weight_ih = weight_ih
        self.weight_hh = weight_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self.gate_act_fn = F.sigmoid
        self.act_fn = F.tanh
        self.input_size = weight_ih.shape[1]

    def forward(self, input, h):

        x_gates = pd.matmul(input, self.weight_ih, transpose_y=True)
        if self.bias_ih is not None:
            x_gates = x_gates + self.bias_ih
        h_gates = pd.matmul(h, self.weight_hh, transpose_y=True)
        if self.bias_hh is not None:
            h_gates = h_gates + self.bias_hh

        x_r, x_z, x_c = pd.split(x_gates, num_or_sections=3, axis=-1)
        h_r, h_z, h_c = pd.split(h_gates, num_or_sections=3, axis=-1)

        r = self.gate_act_fn(x_r + h_r)
        z = self.gate_act_fn(x_z + h_z)
        c = self.act_fn(x_c + r * h_c)  # apply reset gate after mm
        h = (h - c) * z + c

        return h, h


def split_states(states, bidirectional=False, state_components=1):
    if state_components == 1:
        states = pd.unstack(states)
        if not bidirectional:
            return states
        else:
            return list(zip(states[::2], states[1::2]))
    else:
        assert len(states) == state_components
        states = tuple([pd.unstack(item) for item in states])
        if not bidirectional:
            return list(zip(*states))
        else:
            states = list(zip(*states))
            return list(zip(states[::2], states[1::2]))


def concat_states(states, bidirectional=False, state_components=1):
    if state_components == 1:
        return pd.stack(flatten(states))
    else:
        states = flatten(states)
        componnets = []
        for i in range(state_components):
            componnets.append(states[i::state_components])
        return tuple([pd.stack(item) for item in componnets])


class rnnbase(LayerList):

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
        super(rnnbase, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.time_major = False if batch_first else True
        self.dropout = dropout
        self.bidirect = 2 if bidirectional else 1
        self.state_components = 2 if mode == 'LSTM' else 1
        self.training = is_train
        self.w_ih = w_ih
        self.w_hh = w_hh
        self.b_ih = b_ih
        self.b_hh = b_hh
        self.bias = bias
        RNN = pd.nn.RNN
        BiRNN = pd.nn.BiRNN
        kwargs = {"weight_ih_attr": None, "weight_hh_attr": None, "bias_ih_attr": self.bias, "bias_hh_attr": self.bias}
        act = None
        rnn_cls = None
        if mode == "LSTM":
            rnn_cls = pd.nn.LSTMCell
        elif mode == "GRU":
            rnn_cls = pd.nn.GRUCell
        elif mode == 'RNN_TANH':
            rnn_cls = pd.nn.SimpleRNNCell
            kwargs["activation"] = 'tanh'
        elif mode == 'RNN_RELU':
            rnn_cls = pd.nn.SimpleRNNCell
            kwargs["activation"] = 'relu'

        if not bidirectional:
            is_reverse = False
            for i in range(self.num_layers):
                weight_ih = self.w_ih[i]
                weight_hh = self.w_hh[i]
                if self.bias:
                    bias_ih = self.b_ih[i]
                    bias_hh = self.b_hh[i]
                else:
                    bias_ih = None
                    bias_hh = None
                cell = rnn_cls(input_size=self.input_size, hidden_size=self.hidden_size, **kwargs)
                cell.weight_ih = weight_ih
                cell.weight_hh = weight_hh
                cell.bias_ih = bias_ih
                cell.bias_hh = bias_hh
                # cell = rnn_cls(weight_ih, weight_hh, bias_ih, bias_hh, act)
                self.append(RNN(cell, is_reverse, self.time_major))
        else:
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
                layer_input_size = self.input_size if i == 0 else self.hidden_size * self.bidirect
                cell_fw = rnn_cls(input_size=layer_input_size, hidden_size=self.hidden_size, **kwargs)
                cell_fw.weight_ih = weight_ih_fw
                cell_fw.weight_hh = weight_hh_fw
                cell_fw.bias_ih = bias_ih_fw
                cell_fw.bias_hh = bias_hh_fw
                cell_bw = rnn_cls(input_size=layer_input_size, hidden_size=self.hidden_size, **kwargs)
                cell_bw.weight_ih = weight_ih_bw
                cell_bw.weight_hh = weight_hh_bw
                cell_bw.bias_ih = bias_ih_bw
                cell_bw.bias_hh = bias_hh_bw
                self.append(BiRNN(cell_fw, cell_bw, self.time_major))
        self.could_use_cudnn = True
        self.could_use_cudnn &= len(self.parameters()) == num_layers * 4 * self.bidirect

        param_names = []
        for layer in range(self.num_layers):
            for direction in range(self.bidirect):
                suffix = '_reverse' if direction == 1 else ''
                param_names.extend(['weight_ih_l{}{}', 'weight_hh_l{}{}'])
                if bias != False: param_names.append('bias_ih_l{}{}')
                if bias != False: param_names.append('bias_hh_l{}{}')
                param_names = [x.format(layer, suffix) for x in param_names]
        for name, param in zip(param_names, self.parameters()):
            setattr(self, name, param)

        self.flatten_parameters()

    def flatten_parameters(self):
        """
        Resets parameter data pointer to address in continuous memory block for
        cudnn usage.
        """
        if self.could_use_cudnn:
            # layer.parameters() is depth first and ordered
            # for i in layer: for j in direct: w_ih, w_hh, b_ih, b_hh
            # need to reorganize to cudnn param layout:
            # all bias following all weights
            params = self.parameters(include_sublayers=False)
            shape = [np.prod(param.shape) for param in params]
            self._all_weights = [None] * len(params)
            for i, param in enumerate(params):
                base = self.num_layers * self.bidirect
                num = i // base
                odd = num % 2
                offset = (2 * base) * (num // 2)
                new_id = (i - num * base) * 2 + odd + offset
                self._all_weights[new_id] = param
            # Wrap using a list to avoid registed into params and saving, maybe
            # need a better way to handle this later. Use `create_parameter` to
            # add both to main_program and startup_program for static-graph.
            # Use Constant initializer to avoid make effect on random generator.
            self._flat_weight = [
                self.create_parameter(
                    shape=[np.sum(shape)],
                    dtype=params[0].dtype,
                    default_initializer=I.Constant(0.0))
            ]
            # dropout state may also can be hided and avoid saving
            # should dropout state be persistable for static-graph
            self._dropout_state = self.create_variable(
                dtype=core.VarDesc.VarType.UINT8)
            with fluid.program_guard(fluid.default_startup_program(),
                                     fluid.default_startup_program()):
                with paddle.no_grad():
                    self._helper.append_op(
                        type="coalesce_tensor",
                        inputs={"Input": self._all_weights},
                        outputs={
                            "Output": self._all_weights,
                            "FusedOutput": self._flat_weight
                        },
                        attrs={
                            "copy_data": True,
                            "use_align": False,
                            "dtype": params[0].dtype
                        })

    def _cudnn_impl(self, inputs, initial_states, sequence_length):
        if not self.time_major:
            inputs = paddle.tensor.transpose(inputs, [1, 0, 2])

        if in_dynamic_mode():
            _, _, out, state = _C_ops.rnn(
                inputs, initial_states, self._all_weights, sequence_length,
                self._dropout_state, self.state_components, 'dropout_prob',
                self.dropout, 'is_bidirec', self.bidirect == 2,
                'input_size', self.input_size, 'hidden_size', self.hidden_size,
                'num_layers', self.num_layers, 'mode', self.mode, 'is_test',
                not self.training)
        else:
            out = self._helper.create_variable_for_type_inference(inputs.dtype)
            state = [
                self._helper.create_variable_for_type_inference(inputs.dtype)
                for i in range(self.state_components)
            ]
            reserve = self._helper.create_variable_for_type_inference(
                dtype=core.VarDesc.VarType.UINT8, stop_gradient=True)

            inputs = {
                'Input': inputs,
                'WeightList': self._all_weights,
                'PreState': initial_states,
                'SequenceLength': sequence_length
            }
            attrs = {
                'dropout_prob': self.dropout,
                'is_bidirec': self.bidirect == 2,
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'mode': self.mode,
                'is_test': not self.training
            }

            outputs = {
                'Out': out,
                'State': state,
                'Reserve': reserve,
                'DropoutState': self._dropout_state,
            }

            self._helper.append_op(
                type="rnn", inputs=inputs, outputs=outputs, attrs=attrs)

        out = paddle.tensor.transpose(out,
                                      [1, 0, 2]) if not self.time_major else out
        return out, tuple(state) if len(state) > 1 else state[0]

    def check_hidden(self, h, batch_size):
        expected_hidden_size = [self.num_layers * self.bidirect, batch_size, self.hidden_size]
        if h.shape != expected_hidden_size:
            raise ValueError('Expected hidden size {}, got {}.'.format(expected_hidden_size, h.shape))

    def forward(self, inputs, initial_states=None):
        batch_index = 1 if self.time_major else 0
        dtype = inputs.dtype
        sequence_length = None
        batch_size = inputs.shape[batch_index]
        if initial_states is None:
            state_shape = (self.num_layers * self.bidirect, -1, self.hidden_size)
            if self.state_components == 1:
                initial_states = pd.fluid.layers.fill_constant_batch_size_like(
                    inputs, state_shape, dtype, 0, batch_index, 1
                )
            else:
                initial_states = tuple(
                    [
                        pd.fluid.layers.fill_constant_batch_size_like(inputs, state_shape, dtype, 0, batch_index, 1)
                        for _ in range(self.state_components)
                    ]
                )
        else:
            if self.mode == 'LSTM':
                h, c = initial_states
                self.check_hidden(h, batch_size)
                self.check_hidden(c, batch_size)
            else:
                self.check_hidden(initial_states, batch_size)

        if not isinstance(initial_states, (tuple, list)):
            initial_states = [initial_states, ]

        if self.could_use_cudnn and (
                not paddle.device.is_compiled_with_rocm() or
                sequence_length is None):
            # Add CPU kernel and dispatch in backend later
            return self._cudnn_impl(inputs, initial_states, sequence_length)

        states = split_states(initial_states, self.bidirect == 2, self.state_components)
        final_states = []

        for i, rnn_layer in enumerate(self):
            if i > 0:
                inputs = F.dropout(
                    inputs,
                    self.dropout,
                    training=self.training,
                    mode="upscale_in_train")
            outputs, final_state = rnn_layer(inputs, states[i], sequence_length)
            final_states.append(final_state)
            inputs = outputs

        final_states = concat_states(final_states, self.bidirect == 2, self.state_components)
        return outputs, final_states

class layernorm(object):

    def __init__(self, normalized_shape, gamma, beta, eps, input_shape):
        self.normalized_shape = normalized_shape
        self.gamma = gamma
        self.beta = beta
        self.eps = eps
        self.input_shape = input_shape

    def __call__(self, input):
        scale = pd.flatten(self.gamma)
        offset = pd.flatten(self.beta)
        output = pd.nn.functional.layer_norm(
            input, normalized_shape=self.normalized_shape, weight=scale, bias=offset, epsilon=self.eps
        )
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
        #transpose tensor shape
        k = q if k is None else k
        v = q if v is None else v
        if self.batch_first:
            q = pd.transpose(q, perm=(1, 0, 2))
            k = pd.transpose(k, perm=(1, 0, 2))
            v = pd.transpose(v, perm=(1, 0, 2))

        #check tensor shape
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

        #compute q k v linear projection

        q = F.linear(q, self.q_weight, self.q_bias)
        k = F.linear(k, self.k_weight, self.k_bias)
        v = F.linear(v, self.v_weight, self.v_bias)
        # check and prep attention mask
        if attn_mask is not None:
            if convert_dtype(attn_mask.dtype) == 'uint8':
                warnings.warn("attn_mask tensor dtype should better be bool.")
                attn_mask = pd.cast(attn_mask, dtype='bool')
            elif convert_dtype(attn_mask.dtype) not in ('float32', 'float64', 'bool'):
                raise TypeError(
                    "attn_mask tensor dtype should be in ('float32', 'float64', 'bool','uint8'),"
                    "but got {}".format(attn_mask.dtype)
                )
            if attn_mask.dim() == 2:
                if attn_mask.shape != [tgt_len, src_len]:
                    raise ValueError(
                        "The shape of 2D attn_mask should be {}, but got {}.".format(
                            [tgt_len, src_len], attn_mask.shape
                        )
                    )
                attn_mask = pd.unsqueeze(attn_mask, axis=0)
            elif attn_mask.dim() == 3:
                size_3d = [batch_size * self.num_heads, tgt_len, src_len]
                if attn_mask.shape != size_3d:
                    raise ValueError(
                        "The shape of 3D attn_mask should be {}, but got {}.".format(size_3d, attn_mask.shape)
                    )
            else:
                raise ValueError("attn_mask's dimension {} is not supported.".format(attn_mask.dim()))

        # prep mulithead q k v

        q = pd.transpose(pd.reshape(q, shape=(tgt_len, batch_size * self.num_heads, head_dim)), perm=(1, 0, 2))
        k = pd.transpose(pd.reshape(k, shape=(src_len, batch_size * self.num_heads, head_dim)), perm=(1, 0, 2))
        v = pd.transpose(pd.reshape(v, shape=(src_len, batch_size * self.num_heads, head_dim)), perm=(1, 0, 2))

        #check and prep key padding mask
        if key_padding_mask is not None:
            if key_padding_mask.shape != [batch_size, src_len]:
                raise ValueError(
                    "Expecting key_padding_mask shape is {}, but got {}.".format(
                        [batch_size, src_len], key_padding_mask.shape
                    )
                )

            if convert_dtype(key_padding_mask.dtype) == 'uint8':
                warnings.warn("key_padding_mask tensor dtype should better be bool.")
                key_padding_mask = pd.cast(key_padding_mask, dtype='bool')
            elif convert_dtype(key_padding_mask.dtype) != 'bool':
                raise TypeError(
                    "key_padding_mask tensor dtype should be 'bool' or 'uint8', but got {}.".format(
                        key_padding_mask.dtype
                    )
                )

            key_padding_mask = key_padding_mask.reshape((batch_size, 1, 1, src_len)).expand(
                (-1, self.num_heads, -1, -1)
            ).reshape((batch_size * self.num_heads, 1, src_len))

            if attn_mask is None:
                attn_mask = key_padding_mask
            elif convert_dtype(attn_mask.dtype) == 'bool':
                attn_mask = pd.logical_or(attn_mask, key_padding_mask)
            else:
                # attn_mask = attn_mask.expand((self.num_heads * batch_size, -1, -1))
                # key_padding_mask = key_padding_mask.expand((-1,tgt_len, -1))
                # attn_mask = attn_mask.numpy()
                # key_padding_mask = key_padding_mask.numpy()
                # attn_mask[key_padding_mask] = float('-inf')
                # attn_mask = pd.to_tensor(attn_mask, dtype='float32')
                key_padding_mask_inf = pd.full_like(key_padding_mask, fill_value=float('-inf'), dtype='float32')
                attn_mask = pd.where(key_padding_mask, key_padding_mask_inf, attn_mask)

        #convert bool mask to float
        if attn_mask is not None and convert_dtype(attn_mask.dtype) == 'bool':
            # new_attn_mask = pd.zeros_like(attn_mask, dtype='float32')
            # np_new_attn_mask = new_attn_mask.numpy()
            # np_attn_mask = attn_mask.numpy()
            # np_new_attn_mask[np_attn_mask] = float('-inf')
            # attn_mask = pd.to_tensor(np_new_attn_mask, dtype='float32')
            new_attn_mask_zero = pd.zeros_like(attn_mask, dtype='float32')
            new_attn_mask_inf = pd.ones_like(attn_mask, dtype='float32') * -np.inf
            attn_mask = pd.where(attn_mask, new_attn_mask_inf, new_attn_mask_zero)

        # attention and out projection
        q = q / math.sqrt(embed_dim)
        k = pd.transpose(k, perm=(0, 2, 1))
        attn = pd.bmm(q, k)
        if attn_mask is not None:
            attn += attn_mask
        attn = pd.nn.functional.softmax(attn)
        if self.dropout:
            attn = F.dropout(attn, self.dropout, training=self.train, mode="upscale_in_train")

        output = pd.bmm(attn, v)
        output = pd.reshape(pd.transpose(output, perm=(1, 0, 2)), shape=(tgt_len, batch_size, embed_dim))
        output = F.linear(output, weight=self.out_weight, bias=self.out_bias)

        if self.batch_first:
            output = pd.transpose(output, perm=(1, 0, 2))

        if self.need_weights:
            attn = pd.reshape(attn, shape=(batch_size, self.num_heads, tgt_len, src_len))
            attn = pd.sum(attn, axis=1) / self.num_heads
            return output, attn
        else:
            return output, None


class BinaryDense(object):

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def __call__(self, inputs):
        raise NotImplementedError


class DorefaDense(object):

    def __init__(self, weights, bias, bitW, bitA):
        self.weights = weights
        self.bias = bias
        self.bitW = bitW
        self.bitA = bitA

    def __call__(self, inputs):
        raise NotImplementedError


class TernaryDense(object):

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def __call__(self, inputs):
        raise NotImplementedError


class QuanDense(object):

    def __init__(self, weights, bias, bitW, bitA):
        self.weights = weights
        self.bias = bias
        self.bitW = bitW
        self.bitA = bitA

    def __call__(self, inputs):
        raise NotImplementedError


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
        raise NotImplementedError


class TernaryConv(object):

    def __init__(self, weights, strides, padding, data_format, dilations):
        self.weights = weights
        self.strides = strides
        self.dilations = dilations
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)

    def __call__(self, inputs):
        raise NotImplementedError


class QuanConv(object):

    def __init__(self, weights, strides, padding, data_format, dilations, bitW, bitA):
        self.weights = weights
        self.strides = strides
        self.dilations = dilations
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.bitW = bitW
        self.bitA = bitA

    def __call__(self, inputs):
        raise NotImplementedError


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
        raise NotImplementedError


class PReLU(object):

    def __init__(self, data_format):
        # self.data_format, _ = preprocess_2d_format(data_format, None)
        self.data_format = data_format

    def __call__(self, input, weight):
        dim = input.ndim
        if dim == 3:
            self.data_format, _ = preprocess_1d_format(self.data_format, None)
        elif dim == 4:
            self.data_format, _ = preprocess_2d_format(self.data_format, None)
        elif dim == 5:
            self.data_format, _ = preprocess_3d_format(self.data_format, None)

        return F.prelu(input, weight, data_format=self.data_format)


def prelu(input, weight, data_format):
    dim = input.ndim
    if dim == 3:
        data_format, _ = preprocess_1d_format(data_format, None)
    elif dim == 4:
        data_format, _ = preprocess_2d_format(data_format, None)
    elif dim == 5:
        data_format, _ = preprocess_3d_format(data_format, None)
    return F.prelu(input, weight, data_format=data_format)

def hardsigmoid(input):

    return F.hardsigmoid(input)

def hardswish(input):

    return F.hardswish(input)

def swish(input):

    return F.swish(input)