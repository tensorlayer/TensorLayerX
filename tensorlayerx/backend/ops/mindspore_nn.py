#! /usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import itertools
import mindspore as ms
import mindspore.ops as P
from mindspore import context
from mindspore.ops.primitive import constexpr
from mindspore.nn.cell import Cell
from mindspore._checkparam import Rel
from mindspore.ops import functional as F
from mindspore.communication import management
from mindspore.ops.operations import _inner_ops as inner
from mindspore._extends import cell_attr_register
from mindspore.ops._grad.grad_base import bprop_getters
from mindspore._checkparam import Validator as validator
from mindspore.communication.management import get_group_size, get_rank
from mindspore.ops.operations import LayerNorm
import mindspore.numpy as np
from mindspore.common.parameter import ParameterTuple
from mindspore.nn.layer.rnns import _DynamicRNN
import warnings
import math


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
        padding = "same"
    elif padding in ["VALID", "valid"]:
        padding = "valid"
    elif padding == None:
        padding = None
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

    if data_format in ["channels_last", "NWC"]:
        data_format = "NWC"
    elif data_format in ["channels_first", "NCW"]:
        data_format = "NCW"
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

    if len(P.Shape()(x)) == 3:
        x = P.Transpose()(x, (0, 2, 1))
    elif len(P.Shape()(x)) == 4:
        x = P.Transpose()(x, (0, 2, 3, 1))
    elif len(P.Shape()(x)) == 5:
        x = P.Transpose()(x, (0, 2, 3, 4, 1))
    # else:
    #     raise Exception("Unsupported dimensions")
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

    if len(P.Shape()(x)) == 3:
        x = P.Transpose()(x, (0, 2, 1))
    elif len(P.Shape()(x)) == 4:
        x = P.Transpose()(x, (0, 3, 1, 2))
    elif len(P.Shape()(x)) == 5:
        x = P.Transpose()(x, (0, 4, 1, 2, 3))
    # else:
    #     raise Exception("Unsupported dimensions")
    return x


class ReLU(Cell):

    def __init__(self):
        super(ReLU, self).__init__()
        self.relu = P.ReLU()

    def construct(self, x):
        return self.relu(x)


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
    outputs = P.ReLU()
    return outputs(x)


class ELU(Cell):

    def __init__(self, alpha=1.0):
        super(ELU, self).__init__()
        self.elu = P.Elu(alpha)

    def construct(self, x):
        return self.elu(x)


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
    outputs = P.Elu(alpha)
    return outputs(x)


class ReLU6(Cell):

    def __init__(self):
        super(ReLU6, self).__init__()
        self.relu6 = P.ReLU6()

    def construct(self, x):
        return self.relu6(x)


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
    outputs = P.ReLU6()
    return outputs(x)


class LeakyReLU(Cell):

    def __init__(self, alpha=0.2):
        super(LeakyReLU, self).__init__()
        self.leakyrelu = ms.nn.LeakyReLU(alpha=alpha)

    def construct(self, x):
        return self.leakyrelu(x)


def leaky_relu(x, alpha=0.2):
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

    leaky_relu = LeakyReLU(alpha=alpha)
    output = leaky_relu(x)
    return leaky_relu


class Softplus(Cell):

    def __init__(self):
        super(Softplus, self).__init__()
        self.softplus = P.Softplus()

    def construct(self, x):
        return self.softplus(x)


class Tanh(Cell):

    def __init__(self):
        super(Tanh, self).__init__()
        self.tanh = P.Tanh()

    def construct(self, x):
        return self.tanh(x)


class Sigmoid(Cell):

    def __init__(self):
        super(Sigmoid, self).__init__()
        self.sigmoid = P.Sigmoid()

    def construct(self, x):
        return self.sigmoid(x)


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
    outputs = P.Sigmoid()
    return outputs(x)


class Softmax(Cell):

    def __init__(self):
        super(Softmax, self).__init__()
        self.softmax = P.Softmax()

    def construct(self, x):
        return self.softmax(x)


def softmax(logits, axis=None):
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
    outputs = P.Softmax(axis)
    return outputs(logits)


class GeLU(Cell):

    def __init__(self):
        super(GeLU, self).__init__()
        self.gelu = P.GeLU()

    def construct(self, x):
        return self.gelu(x)


def gelu(x):

    outputs = P.GeLU()
    return outputs(x)


class Dropout(Cell):

    def __init__(self, keep, seed=0):
        super(Dropout, self).__init__()
        self.dropout = P.Dropout(keep_prob=keep)
        self.keep_prob = keep

    def construct(self, inputs):
        outputs, _ = self.dropout(inputs)
        return outputs


class BiasAdd(Cell):
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

    def __init__(self, data_format='channels_first'):
        super(BiasAdd, self).__init__()
        self.bias_add = P.BiasAdd()
        if data_format in ['channels_first', 'NCW', 'NCHW', 'NCDHW']:
            self.data_format = 'channels_first'
        elif data_format in ['channels_last', 'NWC', 'NHWC', 'NDHWC']:
            self.data_format = 'channels_last'
        else:
            raise ("Unsupported data format: " + str(data_format))

    def construct(self, x, bias):
        if self.data_format == 'channels_last':
            x = nhwc_to_nchw(x)
        outputs = self.bias_add(x, bias)
        if self.data_format == 'channels_last':
            outputs = nchw_to_nhwc(outputs)
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
    raise NotImplementedError


class Conv1D(Cell):

    def __init__(self, stride, padding, data_format='NWC', dilations=None, out_channel=None, k_size=None):
        super(Conv1D, self).__init__()
        self.data_format, self.padding = preprocess_1d_format(data_format, padding)
        self.stride = (1, stride)
        self.dilations = (1, dilations)
        self.k_size = (1, k_size)
        self.out_channel = out_channel

        self.conv2d = P.Conv2D(
            out_channel=self.out_channel, kernel_size=self.k_size, pad_mode=self.padding, stride=self.stride,
            dilation=self.dilations, mode=1, group=1
        )

        self.expand_dims = P.ExpandDims()
        self.squeeze = P.Squeeze(2)

    def construct(self, x, filters):
        if self.data_format == 'NWC':
            x = nhwc_to_nchw(x)

        x = self.expand_dims(x, 2)
        filters = self.expand_dims(filters, 2)

        output = self.conv2d(x, filters)
        output = self.squeeze(output)

        if self.data_format == 'NWC':
            output = nchw_to_nhwc(output)
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

    pass


class Conv2D(Cell):

    def __init__(self, strides, padding, data_format='NHWC', dilations=None, out_channel=None, k_size=None):
        super(Conv2D, self).__init__()
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)

        if self.data_format is 'NHWC':
            self.ms_stride = strides[1]
            self.ms_dilation = dilations[1]
        elif self.data_format is 'NCHW':
            self.ms_stride = strides[2]
            self.ms_dilation = dilations[2]

        self.conv2d = P.Conv2D(
            out_channel=out_channel, kernel_size=k_size, pad_mode=self.padding, stride=self.ms_stride,
            dilation=self.ms_dilation, mode=1, group=1, data_format=self.data_format
        )

    def construct(self, inputs, filters):
        outputs = self.conv2d(inputs, filters)
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
    raise NotImplementedError


class Conv3D(Cell):

    def __init__(self, strides, padding, data_format='NDHWC', dilations=None, out_channel=None, k_size=None):
        super(Conv3D, self).__init__()
        self.data_format, self.padding = preprocess_3d_format(data_format, padding)

        if self.data_format is 'NDHWC':
            self.ms_stride = strides[1]
            self.ms_dilation = dilations[1]
            raise NotImplementedError("The optional value for data format. Currently only support “NCDHW”.")
        elif self.data_format is 'NCDHW':
            self.ms_stride = strides[2]
            self.ms_dilation = dilations[2]

        self.conv3d = P.Conv3D(
            out_channel=out_channel, kernel_size=k_size, pad_mode=self.padding, stride=self.ms_stride,
            dilation=self.ms_dilation, data_format=data_format
        )

    def construct(self, input, filters):
        outputs = self.conv3d(input, filters)
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

    raise NotImplementedError


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


class MaxPool1d(Cell):

    def __init__(self, ksize, strides, padding, data_format=None):
        super(MaxPool1d, self).__init__()
        self.data_format, padding = preprocess_1d_format(data_format=data_format, padding=padding)
        self.expand = P.ExpandDims()
        _strides = (1, strides[0])
        _ksize = (1, ksize[0])
        if self.data_format == 'NWC':
            self.squeeze = P.Squeeze(1)
            _data_format = 'NHWC'
        if self.data_format == 'NCW':
            self.squeeze = P.Squeeze(2)
            _data_format = 'NCHW'

        self.max_pool = P.MaxPool(kernel_size=_ksize, strides=_strides, pad_mode=padding, data_format=_data_format)

    def construct(self, inputs):
        if self.data_format == 'NWC':
            x = self.expand(inputs, 1)
        if self.data_format == 'NCW':
            x = self.expand(inputs, 2)
        output = self.max_pool(x)
        output = self.squeeze(output)
        return output


class MaxPool(Cell):

    def __init__(self, ksize, strides, padding, data_format='NHWC'):
        super(MaxPool, self).__init__()
        data_format, padding = preprocess_2d_format(data_format=data_format, padding=padding)

        if data_format == 'NHWC':
            strides = (strides[1], strides[2])
            if len(ksize) == 4:
                ksize = (ksize[1], ksize[2])
        if data_format == 'NCHW':
            strides = (strides[2], strides[3])
            if len(ksize) == 4:
                ksize = (ksize[2], ksize[3])

        self.maxpool = P.MaxPool(kernel_size=ksize, strides=strides, pad_mode=padding, data_format=data_format)

    def construct(self, inputs):
        outputs = self.maxpool(inputs)
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
    strides : list or list of ints
        An int or list of ints that has length 1, N or N+2.
        The stride of the sliding window for each dimension of the input tensor.
    padding : string
        'VALID' or 'SAME'. The padding algorithm. See the "returns" section of tf.ops.convolution for details.

    Returns
    -------
        A Tensor of format specified by data_format. The max pooled output tensor.
    """
    data_format, padding = preprocess_2d_format(data_format=data_format, padding=padding)
    if data_format == 'NHWC':
        _strides = (strides[1], strides[2])
    if data_format == 'NCHW':
        _strides = (strides[2], strides[3])
    outputs = P.MaxPool(kernel_size=ksize, strides=_strides, pad_mode=padding, data_format=data_format)(input)
    return outputs


class AvgPool1d(Cell):

    def __init__(self, ksize, strides, padding, data_format=None):
        super(AvgPool1d, self).__init__()
        self.data_format, self.padding = preprocess_1d_format(data_format=data_format, padding=padding)
        self.kernel_size = (1, ksize[0])
        self.stride = (1, strides[0])

        if self.data_format == 'NWC':
            _data_format = 'NHWC'
            self.squeeze = P.Squeeze(1)
        if self.data_format == 'NCW':
            _data_format = 'NCHW'
            self.squeeze = P.Squeeze(2)

        self.avg_pool = P.AvgPool(
            kernel_size=self.kernel_size, strides=self.stride, pad_mode=self.padding, data_format=_data_format
        )
        self.reduce_mean = P.ReduceMean(keep_dims=True)
        self.slice = P.Slice()
        self.expand = P.ExpandDims()
        self.shape = P.Shape()

    def construct(self, inputs):
        x = inputs
        batch, channel, width = self.shape(inputs)
        if width == self.kernel_size[1]:
            x = self.reduce_mean(x, 2)
        elif width - self.kernel_size[1] < self.stride[1]:
            x = self.slice(x, (0, 0, 0), (batch, channel, self.kernel_size[1]))
            x = self.reduce_mean(x, 2)
        else:
            if self.data_format == 'NCW':
                x = self.expand(x, 2)
            if self.data_format == 'NWC':
                x = self.expand(x, 1)
            x = self.avg_pool(x)
            x = self.squeeze(x)
        return x


class AvgPool(Cell):

    def __init__(self, ksize, strides, padding, data_format=None):
        super(AvgPool, self).__init__()
        self.data_format, self.padding = preprocess_2d_format(data_format=data_format, padding=padding)
        ms_ksize = ksize[1]
        ms_strides = strides[1]
        self.avgpool = P.AvgPool(
            kernel_size=ms_ksize, strides=ms_strides, pad_mode=padding, data_format=self.data_format
        )

    def construct(self, inputs):
        outputs = self.avgpool(inputs)
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
    padding = padding_format(padding)
    ms_ksize = ksize[0]
    ms_strides = strides[1]
    outputs = P.AvgPool(ksize=ms_ksize, strides=ms_strides, padding=padding)
    return outputs(input)


class MaxPool3d(Cell):

    def __init__(self, ksize, strides, padding, data_format=None):
        super(MaxPool3d, self).__init__()
        self.data_format, self.padding = preprocess_3d_format(data_format, padding)
        if data_format == 'NDHWC':
            strides = (strides[1], strides[2], strides[3])
            raise NotImplementedError("The optional value for data format. Currently only support ‘NCDHW’.")
        if data_format == 'NCDHW':
            strides = (strides[2], strides[3], strides[4])
        self.max_pool3d = P.MaxPool3D(
            kernel_size=ksize, strides=strides, pad_mode=padding, data_format=self.data_format
        )

    def __call__(self, inputs):
        outputs = self.max_pool3d(inputs)
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


class AvgPool3d(Cell):

    def __init__(self, ksize, strides, padding, data_format='NCDHW'):
        super(AvgPool3d, self).__init__()
        self.data_format, self.padding = preprocess_3d_format(data_format, padding)
        if data_format == 'NDHWC':
            strides = (strides[1], strides[2], strides[3])
            raise NotImplementedError('The optional value for data format. Currently only support ‘NCDHW’.')
        if data_format == 'NCDHW':
            strides = (strides[2], strides[3], strides[4])
        print(ksize, strides, padding)
        self.avg_pool = P.AvgPool3D(kernel_size=ksize, strides=strides, pad_mode=padding, data_format=data_format)

    def __call__(self, inputs):
        return self.avg_pool(inputs)


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


class DepthwiseConv2d(Cell):

    def __init__(self, strides, padding, data_format=None, dilations=None, ksize=None, channel_multiplier=1):
        super(DepthwiseConv2d, self).__init__()
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.ms_stride = strides[1]
        self.ms_dilation = dilations[1]
        self.depthwise_conv2d = P.DepthwiseConv2dNative(
            channel_multiplier=channel_multiplier, kernel_size=ksize, stride=self.ms_stride, dilation=self.ms_dilation
        )

    def construct(self, input, filter, point_filter=None):
        if self.data_format == 'NHWC':
            input = nhwc_to_nchw(input)
        outputs = self.depthwise_conv2d(input, filter)
        if self.data_format == 'NHWC':
            outputs = nchw_to_nhwc(outputs)
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

    pass


class Conv1d_transpose(Cell):

    def __init__(self, stride, padding, data_format, dilations=None, out_channel=None, k_size=None, in_channels=None):
        super(Conv1d_transpose, self).__init__()
        self.data_format, self.padding = preprocess_1d_format(data_format, padding)
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.stride = (1, stride)
        self.dilations = (1, dilations)
        self.k_size = (1, k_size)
        if self.data_format == 'NWC':
            self.data_format = 'NHWC'
            self.h_axis = 1
            raise NotImplementedError("The optional value for data format. Currently only support “NCW”.")
        else:
            self.data_format = 'NCHW'
            self.h_axis = 2
        self.conv2d_transpose = P.Conv2DBackpropInput(
            out_channel=self.in_channels, kernel_size=self.k_size, pad_mode=self.padding, stride=self.stride,
            dilation=self.dilations, mode=1, group=1, data_format=self.data_format
        )
        self.shape = P.Shape()
        self.expand_dims = P.ExpandDims()
        self.squeeze = P.Squeeze(2)

    def _deconv_output_length(self, input_length, filter_size, stride_size, dilation_size):
        length = 0
        filter_size = filter_size + (filter_size - 1) * (dilation_size - 1)

        if self.padding == 'same':
            length = input_length * stride_size
        elif self.padding == 'valid':
            if filter_size - stride_size > 0:
                length = input_length * stride_size + filter_size - stride_size
            else:
                length = input_length * stride_size

        return length

    def construct(self, x, filters):
        if self.data_format == 'NHWC':
            x = nhwc_to_nchw(x)

        x = self.expand_dims(x, 2)
        filters = self.expand_dims(filters, 2)
        n, _, h, w = self.shape(x)
        h_out = self._deconv_output_length(h, self.k_size[0], self.stride[0], self.dilations[0])
        w_out = self._deconv_output_length(w, self.k_size[1], self.stride[1], self.dilations[1])

        output_size = (n, self.out_channel, h_out, w_out)
        output = self.conv2d_transpose(x, filters, output_size)
        output = self.squeeze(output)
        # TODO Conv2DBackpropInput is deprecated from version 1.5
        if self.data_format == 'NHWC':
            output = nchw_to_nhwc(output)
        return output


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
    pass


class Conv2d_transpose(Cell):

    def __init__(self, strides, padding, data_format, dilations=None, out_channel=None, k_size=None, in_channels=None):
        super(Conv2d_transpose, self).__init__()
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.k_size = k_size
        self.strides = strides
        self.dilations = dilations

        if self.data_format == 'NHWC':
            raise NotImplementedError("The optional value for data format. Currently only support “NCWH”.")

        self.conv2d_transpose = P.Conv2DBackpropInput(
            out_channel=self.in_channels, kernel_size=self.k_size, pad_mode=self.padding, stride=self.strides,
            dilation=self.dilations, mode=1, group=1, data_format=self.data_format
        )
        self.shape = P.Shape()

    def _deconv_output_length(self, input_length, filter_size, stride_size, dilation_size):
        length = 0
        filter_size = filter_size + (filter_size - 1) * (dilation_size - 1)

        if self.padding == 'same':
            length = input_length * stride_size
        elif self.padding == 'valid':
            length = input_length * stride_size + max(filter_size - stride_size, 0)

        return length

    def construct(self, x, filters):
        if self.data_format == 'NHWC':
            h_axis, w_axis = 1, 2
            n, h, w, _ = self.shape(x)
        else:
            h_axis, w_axis = 2, 3
            n, _, h, w = self.shape(x)

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

        h_out = self._deconv_output_length(h, self.k_size[0], strides_h, dilations_h)
        w_out = self._deconv_output_length(w, self.k_size[1], strides_w, dilations_w)

        if self.data_format == 'NCHW':
            output_size = (n, self.out_channel, h_out, w_out)
        else:
            output_size = (n, h_out, w_out, self.out_channel)
        output = self.conv2d_transpose(x, filters, output_size)

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
    pass


class Conv3d_transpose(Cell):

    def __init__(
        self, strides, padding, data_format='NDHWC', dilations=None, name=None, out_channel=None, k_size=None,
        in_channels=None
    ):
        super(Conv3d_transpose, self).__init__()
        self.data_format, self.padding = preprocess_3d_format(data_format, padding)

        self.conv3d_transpose = P.Conv3DTranspose(
            in_channel=in_channels, out_channel=out_channel, kernel_size=k_size, mode=1, pad_mode=self.padding,
            stride=strides, dilation=dilations, data_format=self.data_format
        )

    def construct(self, input, filters):
        output = self.conv3d_transpose(input, filters)
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

    pass


class BatchNorm(Cell):
    """Batch Normalization base class."""

    @cell_attr_register
    def __init__(
        self, num_features, epsilon=1e-5, decay=0.9, gamma=None, beta=None, moving_mean=None, moving_var=None,
        is_train=None, device_num_each_group=1, process_groups=0, data_format='NCHW'
    ):
        super(BatchNorm, self).__init__()
        if data_format in ["channels_last", "NHWC", "nhwc"]:
            data_format = "NHWC"
        elif data_format in ["channels_first", "NCHW", "nchw"]:
            data_format = "NCHW"
        validator.check_value_type('num_features', num_features, [int], self.cls_name)
        if num_features < 1:
            raise ValueError("num_features must be at least 1")

        if decay < 0 or decay > 1:
            raise ValueError("momentum should be a number in range [0, 1], but got {}".format(decay))
        self.format = validator.check_string(data_format, ['NCHW', 'NHWC'], 'format', self.cls_name)
        if context.get_context("device_target") != "GPU" and self.format == "NHWC":
            raise ValueError("NHWC format only support in GPU target.")
        self.use_batch_statistics = is_train
        self.num_features = num_features
        self.eps = epsilon
        self.moving_mean = moving_mean
        self.moving_variance = moving_var
        self.gamma = gamma
        self.beta = beta
        self.group_device_num = validator.check_positive_int(device_num_each_group)
        self.process_groups = process_groups
        self.is_global = False
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        global SYNC_BN_GROUP_NAME
        # for GlobalBatchNorm
        if self.group_device_num != 1:
            self.rank_id = get_rank()
            self.rank_size = get_group_size()
            self.device_list = [i for i in range(0, self.rank_size)]
            self.rank_list = self.list_group(self.device_list, self.group_device_num)
            self.rank_list_idx = len(self.rank_list)
            for i in range(self.rank_list_idx):
                if self.rank_id in self.rank_list[i]:
                    self.is_global = True
                    if SYNC_BN_GROUP_NAME == "":
                        SYNC_BN_GROUP_NAME = "sync_bn_group" + str(i)
                        management.create_group(SYNC_BN_GROUP_NAME, self.rank_list[i])
        # for SyncBatchNorm
        if self.process_groups != 0:
            self.rank_id = get_rank()
            self.rank_size = get_group_size()
            if self.process_groups is not None:
                validator.check_isinstance("process_groups", self.process_groups, list)
                self._check_rank_ids(self.process_groups, self.rank_size)
                for i in range(len(self.process_groups)):
                    validator.check_isinstance("process_groups[" + str(i) + "]", self.process_groups[i], list)
                    self.group_device_num = len(self.process_groups[i])
                    if self.rank_id in self.process_groups[i] and self.group_device_num > 1:
                        self.is_global = True
                        if SYNC_BN_GROUP_NAME == "":
                            SYNC_BN_GROUP_NAME = "sync_bn_group" + str(i)
                            management.create_group(SYNC_BN_GROUP_NAME, self.process_groups[i])
            elif self.rank_size > 1:
                self.is_global = True
                self.group_device_num = self.rank_size
                self.device_list = [i for i in range(0, self.rank_size)]
                if SYNC_BN_GROUP_NAME == "":
                    SYNC_BN_GROUP_NAME = "sync_bn_group0"
                    management.create_group(SYNC_BN_GROUP_NAME, self.device_list)

        self.shape = P.Shape()
        self.reduce_mean = P.ReduceMean(keep_dims=True)
        self.square = P.Square()
        self.sqrt = P.Sqrt()
        self.cast = P.Cast()
        self.dtype = P.DType()
        self.reshape = P.Reshape()
        self._target = context.get_context("device_target")
        self.is_graph_mode = context.get_context("mode") == context.GRAPH_MODE
        self.momentum = 1.0 - decay
        if context.get_context("enable_ge"):
            self.is_ge_backend = True
        else:
            self.is_ge_backend = False

        self.bn_train = P.BatchNorm(is_training=True, epsilon=self.eps, momentum=self.momentum, data_format=self.format)
        if self.is_global:
            self.bn_train = inner.SyncBatchNorm(
                epsilon=self.eps, momentum=self.momentum, group=SYNC_BN_GROUP_NAME, device_num=self.group_device_num
            )

        self.bn_infer = P.BatchNorm(is_training=False, epsilon=self.eps, data_format=self.format)

        data_parallel_strategy = ((1, ), (1, ))
        data_parallel_strategy_one = ((1, ), ())
        self.sub_mean = P.Sub().shard(data_parallel_strategy)
        self.sub_var = P.Sub().shard(data_parallel_strategy)
        self.mul_mean = P.Mul().shard(data_parallel_strategy_one)
        self.mul_var = P.Mul().shard(data_parallel_strategy_one)
        self.assign_sub_mean = P.AssignSub().shard(data_parallel_strategy)
        self.assign_sub_var = P.AssignSub().shard(data_parallel_strategy)

    def list_group(self, world_rank, group_size):
        if group_size > get_group_size():
            raise ValueError(
                "group size can not be greater than local rank size, group size is {}, "
                "local_rank_size is {}".format(group_size, get_group_size())
            )
        if len(world_rank) % group_size != 0:
            raise ValueError("please make your group size correct.")
        world_rank_list = zip(*(iter(world_rank), ) * group_size)
        group_list = [list(i) for i in world_rank_list]
        return group_list

    def _check_rank_ids(self, process_groups, rank_size):
        seen = set()
        for rid in itertools.chain(*process_groups):
            validator.check_int_range(rid, 0, rank_size, Rel.INC_LEFT, "rank id in process_groups")
            if rid in seen:
                raise ValueError("rank id in process_groups should not be duplicated.")
            seen.add(rid)

    def construct(self, inputs):
        x_shape = F.shape(inputs)
        if len(x_shape) == 5:
            inputs = self.reshape(inputs, (x_shape[0], x_shape[1], x_shape[2] * x_shape[3], x_shape[4]))

        flag = self.use_batch_statistics

        if flag:
            output = self.bn_train(inputs, self.gamma, self.beta, self.moving_mean, self.moving_variance)[0]

            if len(x_shape) == 5:
                output = self.reshape(output, x_shape)
            return output

        output = self.bn_infer(inputs, self.gamma, self.beta, self.moving_mean, self.moving_variance)[0]
        if len(x_shape) == 5:
            output = self.reshape(output, x_shape)
        return output

    def extend_repr(self):
        return 'num_features={}, eps={}, momentum={}, gamma={}, beta={}, moving_mean={}, moving_variance={}'.format(
            self.num_features, self.eps, self.momentum, self.gamma, self.beta, self.moving_mean, self.moving_variance
        )


class GroupConv2D(Cell):

    def __init__(self, strides, padding, data_format, dilations, out_channel, k_size, groups):
        super(GroupConv2D, self).__init__()
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)

        if self.data_format is 'NHWC':
            self.ms_stride = strides[1]
            self.ms_dilation = dilations[1]

        elif self.data_format is 'NCHW':
            self.ms_stride = strides[2]
            self.ms_dilation = dilations[2]

        self.conv2d = P.Conv2D(
            out_channel=out_channel, kernel_size=k_size, pad_mode=self.padding, stride=self.ms_stride,
            dilation=self.ms_dilation, mode=1, group=groups, data_format=self.data_format
        )

    def construct(self, inputs, filters):
        outputs = self.conv2d(inputs, filters)
        return outputs


class SeparableConv1D(Cell):

    def __init__(self, stride, padding, data_format, dilations, out_channel, k_size, in_channel, depth_multiplier):
        super(SeparableConv1D, self).__init__()
        self.data_format, self.padding = preprocess_1d_format(data_format, padding)
        self.stride = (1, stride)
        self.dilations = (1, dilations)
        self.k_size = (1, k_size)
        self.out_channel = out_channel
        self.in_channel = in_channel
        self.depth_multiplier = depth_multiplier
        self.depthwise_conv = P.Conv2D(
            out_channel=self.in_channel * self.depth_multiplier, kernel_size=self.k_size, pad_mode=self.padding,
            stride=self.stride, dilation=self.dilations, mode=1, group=self.in_channel
        )

        self.pointwise_conv = P.Conv2D(
            out_channel=self.out_channel, kernel_size=(1, 1), pad_mode=self.padding, stride=(1, 1), dilation=(1, 1),
            mode=1, group=1
        )

        self.expand_dims = P.ExpandDims()
        self.squeeze = P.Squeeze(2)

    def construct(self, x, depthwise_filters, pointwise_filters):

        if self.data_format == 'NWC':
            x = nhwc_to_nchw(x)

        x = self.expand_dims(x, 2)
        depthwise_filters = self.expand_dims(depthwise_filters, 2)
        pointwise_filters = self.expand_dims(pointwise_filters, 2)

        outputs = self.depthwise_conv(x, depthwise_filters)
        outputs = self.pointwise_conv(outputs, pointwise_filters)

        outputs = self.squeeze(outputs)

        if self.data_format == 'NWC':
            outputs = nchw_to_nhwc(outputs)
        return outputs


class SeparableConv2D(Cell):

    def __init__(self, strides, padding, data_format, dilations, out_channel, k_size, in_channel, depth_multiplier):
        super(SeparableConv2D, self).__init__()
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.k_size = k_size
        self.out_channel = out_channel
        self.in_channel = in_channel
        self.depth_multiplier = depth_multiplier

        if self.data_format is 'NHWC':
            self.ms_stride = strides[1]
            self.ms_dilation = dilations[1]
        elif self.data_format is 'NCHW':
            self.ms_stride = strides[2]
            self.ms_dilation = dilations[2]

        self.depthwise_conv = P.Conv2D(
            out_channel=self.in_channel * self.depth_multiplier, kernel_size=self.k_size, pad_mode=self.padding,
            stride=self.ms_stride, dilation=self.ms_dilation, mode=1, group=self.in_channel,
            data_format=self.data_format
        )

        self.pointwise_conv = P.Conv2D(
            out_channel=self.out_channel, kernel_size=(1, 1), pad_mode=self.padding, stride=(1, 1), dilation=(1, 1),
            mode=1, group=1, data_format=self.data_format
        )

    def construct(self, x, depthwise_filters, pointwise_filters):
        outputs = self.depthwise_conv(x, depthwise_filters)
        outputs = self.pointwise_conv(outputs, pointwise_filters)
        return outputs


class AdaptiveMeanPool1D(Cell):

    def __init__(self, output_size, data_format):
        super(AdaptiveMeanPool1D, self).__init__()
        self.data_format, _ = preprocess_1d_format(data_format, None)
        self.output_size = output_size
        if self.data_format == 'NWC':
            self.data_format = 'NHWC'
            self.h_axis = 1
        else:
            self.data_format = 'NCHW'
            self.h_axis = 2
        self.expand_dims = P.ExpandDims()
        self.squeeze = P.Squeeze(self.h_axis)
        self.shape = P.Shape()

    def construct(self, inputs):
        if self.data_format == 'NHWC':
            n, w, c = self.shape(inputs)
        else:
            n, c, w = self.shape(inputs)
        inputs = self.expand_dims(inputs, self.h_axis)
        stride = (1, w // self.output_size)
        kernel = (1, w - (self.output_size - 1) * stride[1])
        outputs = P.AvgPool(kernel_size=kernel, strides=stride, pad_mode='VALID', data_format=self.data_format)(inputs)
        outputs = self.squeeze(outputs)

        return outputs


class AdaptiveMeanPool2D(Cell):

    def __init__(self, output_size, data_format):
        super(AdaptiveMeanPool2D, self).__init__()
        self.data_format, _ = preprocess_2d_format(data_format, None)
        self.output_size = output_size
        if self.data_format == 'NHWC':
            self.h_axis = 1
        else:
            self.h_axis = 2
        self.shape = P.Shape()

    def construct(self, inputs):
        if self.data_format == 'NHWC':
            n, h, w, c = self.shape(inputs)
        else:
            n, c, h, w = self.shape(inputs)

        out_h, out_w = self.output_size
        stride_h = h // out_h
        kernel_h = h - (out_h - 1) * stride_h
        stride_w = w // out_w
        kernel_w = w - (out_w - 1) * stride_w
        outputs = P.AvgPool(
            kernel_size=(kernel_h, kernel_w), strides=(stride_h, stride_w), pad_mode='VALID',
            data_format=self.data_format
        )(inputs)

        return outputs


class AdaptiveMeanPool3D(Cell):

    def __init__(self, output_size, data_format):
        pass

    def __call__(self, inputs):
        raise NotImplementedError


class AdaptiveMaxPool1D(Cell):

    def __init__(self, output_size, data_format):
        super(AdaptiveMaxPool1D, self).__init__()
        self.data_format, _ = preprocess_1d_format(data_format, None)
        self.output_size = output_size
        if self.data_format == 'NWC':
            self.data_format = 'NHWC'
            self.h_axis = 1
        else:
            self.data_format = 'NCHW'
            self.h_axis = 2
        self.expand_dims = P.ExpandDims()
        self.squeeze = P.Squeeze(self.h_axis)
        self.shape = P.Shape()

    def construct(self, inputs):

        if self.data_format == 'NHWC':
            n, w, c = self.shape(inputs)
        else:
            n, c, w = self.shape(inputs)
        inputs = self.expand_dims(inputs, self.h_axis)
        stride = (1, w // self.output_size)
        kernel = (1, w - (self.output_size - 1) * stride[1])
        outputs = P.MaxPool(kernel_size=kernel, strides=stride, pad_mode='VALID', data_format=self.data_format)(inputs)
        outputs = self.squeeze(outputs)

        return outputs


class AdaptiveMaxPool2D(Cell):

    def __init__(self, output_size, data_format):
        super(AdaptiveMaxPool2D, self).__init__()
        self.data_format, _ = preprocess_2d_format(data_format, None)
        self.output_size = output_size
        if self.data_format == 'NHWC':
            self.h_axis = 1
        else:
            self.h_axis = 2
        self.shape = P.Shape()

    def construct(self, inputs):
        if self.data_format == 'NHWC':
            n, h, w, c = self.shape(inputs)
        else:
            n, c, h, w = self.shape(inputs)
        out_h, out_w = self.output_size
        stride_h = h // out_h
        kernel_h = h - (out_h - 1) * stride_h
        stride_w = w // out_w
        kernel_w = w - (out_w - 1) * stride_w
        outputs = P.MaxPool(
            kernel_size=(kernel_h, kernel_w), strides=(stride_h, stride_w), pad_mode='VALID',
            data_format=self.data_format
        )(inputs)

        return outputs


class AdaptiveMaxPool3D(Cell):

    def __init__(self, output_size, data_format):
        pass

    def __call__(self, inputs):
        raise NotImplementedError


class BinaryConv2D(Cell):

    def __init__(self, strides, padding, data_format, dilations, out_channel, k_size, in_channel):
        super(BinaryConv2D, self).__init__()
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        if self.data_format is 'NHWC':
            self.ms_stride = strides[1]
            self.ms_dilation = dilations[1]
        elif self.data_format is 'NCHW':
            self.ms_stride = strides[2]
            self.ms_dilation = dilations[2]

        self.conv2d = P.Conv2D(
            out_channel=out_channel, kernel_size=k_size, pad_mode=self.padding, stride=self.ms_stride,
            dilation=self.ms_dilation, mode=1, group=1, data_format=self.data_format
        )

        @bprop_getters.register(P.Sign)
        def get_bprop_Sign(self):

            def bprop(x, out, dout):

                grad = P.clip_by_value(dout, -1, 1)
                return (grad, )

            return bprop

        self.sign = P.Sign()

    def construct(self, inputs, filters):

        filters = self.sign(filters)
        outputs = self.conv2d(inputs, filters)

        return outputs


class DorefaConv2D(Cell):

    def __init__(self, bitW, bitA, strides, padding, data_format, dilations, out_channel, k_size, in_channel):
        super(DorefaConv2D, self).__init__()
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.bitW = ms.Tensor(bitW)
        self.bitA = ms.Tensor(bitA)
        if self.data_format is 'NHWC':
            self.ms_stride = strides[1]
            self.ms_dilation = dilations[1]
            # self.transpose = P.Transpose()
        elif self.data_format is 'NCHW':
            self.ms_stride = strides[2]
            self.ms_dilation = dilations[2]

        self.conv2d = P.Conv2D(
            out_channel=out_channel, kernel_size=k_size, pad_mode=self.padding, stride=self.ms_stride,
            dilation=self.ms_dilation, mode=1, group=1
        )

        @bprop_getters.register(P.Round)
        def get_bprop_Round(self):

            def bprop(x, out, dout):

                return (dout, )

            return bprop

        @bprop_getters.register(P.Sign)
        def get_bprop_Sign(self):

            def bprop(x, out, dout):

                return (dout, )

            return bprop

        self.mimimum = P.Minimum()
        self.abs = P.Abs()
        self.round = P.Round()
        self.reducemean = P.ReduceMean()
        self.sign = P.Sign()
        self.pow = P.Pow()
        self.sub = P.Sub()
        self.oneslike = P.OnesLike()

    def cabs(self, inputs):

        a = P.stop_gradient(self.oneslike(inputs))
        return self.mimimum(self.abs(inputs), a)

    def _quantize_dorefa(self, x, k):

        n = self.sub(self.pow(2.0, k), 1)
        return self.round(x * n) / n

    def quantize_active(self, x, bitA):
        if bitA == 32:
            return x
        return self._quantize_dorefa(x, bitA)

    def quantize_weight(self, x, bitW, force_quantization=False):

        if bitW == 32 and not force_quantization:
            return x

        if bitW == 1:
            E = P.stop_gradient(self.reducemean(self.abs(x)))
            return self.sign(x / E) * E

        x = P.clip_by_value(x * 0.5 + 0.5, 0.0, 1.0)

        return 2 * self._quantize_dorefa(x, bitW) - 1

    def construct(self, inputs, filters):

        if self.data_format == 'NHWC':
            inputs = nhwc_to_nchw(inputs)

        inputs = self.quantize_active(self.cabs(inputs), self.bitA)

        filters = self.quantize_weight(filters, self.bitW)

        outputs = self.conv2d(inputs, filters)

        if self.data_format == 'NHWC':
            outputs = nchw_to_nhwc(outputs)

        return outputs


class rnncell(Cell):

    def __init__(self, weight_ih, weight_hh, bias_ih, bias_hh, act):
        super(rnncell, self).__init__()
        self.weight_ih = weight_ih
        self.weight_hh = weight_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self.act_fn = P.ReLU() if act == 'relu' else P.Tanh()

    def construct(self, input, h):
        i2h = P.MatMul(False, True)(input, self.weight_ih)
        if self.bias_ih is not None:
            i2h += self.bias_ih
        h2h = P.MatMul(False, True)(h, self.weight_hh)
        if self.bias_hh is not None:
            h2h += self.bias_hh
        h = self.act_fn(i2h + h2h)
        return h, h


class lstmcell(Cell):

    def __init__(self, weight_ih, weight_hh, bias_ih, bias_hh):
        super(lstmcell, self).__init__()
        self.weight_ih = weight_ih
        self.weight_hh = weight_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self.gate_act_fn = P.Sigmoid()
        self.act_fn = P.Tanh()
        self.split = P.Split(axis=-1, output_num=4)

    def construct(self, input, h, c):

        gates = P.MatMul(False, True)(input, self.weight_ih)
        if self.bias_ih is not None:
            gates += self.bias_ih
        gates += P.MatMul(False, True)(h, self.weight_hh)
        if self.bias_hh is not None:
            gates += self.bias_hh

        gate_slices = self.split(gates)
        i = self.gate_act_fn(gate_slices[0])
        f = self.gate_act_fn(gate_slices[1])
        o = self.gate_act_fn(gate_slices[3])
        c = f * c + i * self.act_fn(gate_slices[2])
        h = o * self.act_fn(c)

        return h, h, c


class grucell(Cell):

    def __init__(self, weight_ih, weight_hh, bias_ih, bias_hh):
        super(grucell, self).__init__()
        self.weight_ih = weight_ih
        self.weight_hh = weight_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self.gate_act_fn = P.Sigmoid()
        self.act_fn = P.Tanh()
        self.transpose = P.Transpose()
        self.split = P.Split(axis=-1, output_num=3)

    def construct(self, input, h):

        x_gates = P.MatMul(False, True)(input, self.weight_ih)
        if self.bias_ih is not None:
            x_gates += self.bias_ih
        h_gates = P.MatMul(False, True)(h, self.weight_hh)
        if self.bias_hh is not None:
            h_gates += self.bias_hh

        x_r, x_z, x_c = self.split(x_gates)
        h_r, h_z, h_c = self.split(h_gates)

        r = self.gate_act_fn(x_r + h_r)
        z = self.gate_act_fn(x_r + h_z)
        c = self.act_fn(x_c + r * h_c)
        h = (h - c) * z + c

        return h, h


@constexpr
def _init_state(shape, dtype, is_lstm):
    hx = ms.Tensor(np.zeros(shape), dtype)
    cx = ms.Tensor(np.zeros(shape), dtype)
    if is_lstm:
        return (hx, cx)
    return hx

@constexpr
def _check_input_dtype_same_and_valid(args_name, args_value, valid_values, cls_name):
    args = {args_name[i]: args_value[i] for i in range(len(args_value))}
    validator.check_types_same_and_valid(args, valid_values, cls_name)

class rnnbase(Cell):

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
        if not 0 <= dropout < 1:
            raise ValueError("dropout should be a number in range [0, 1).")
        if dropout > 0 and num_layers == 1:
            raise ValueError(
                "dropout option adds dropout after all but last "
                "recurrent layer, so non-zero dropout expects "
                "num_layers greater than 1, but got dropout={} and "
                "num_layers={}".format(dropout, num_layers)
            )
        self.mode = mode
        self.reverse = P.ReverseV2([0])
        self.reverse_sequence = P.ReverseSequence(0, 1)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.dropout_op = ms.nn.Dropout(float(1 - dropout))
        self.has_bias = bias
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.train = is_train
        self.w_ih_list = ParameterTuple(w_ih)
        self.w_hh_list = ParameterTuple(w_hh)
        self.b_ih_list = ParameterTuple(b_ih)
        self.b_hh_list = ParameterTuple(b_hh)
        self.rnn = _DynamicRNN(mode)
        self.is_lstm = mode == "LSTM"

        self.zeros = P.Zeros()

    def _stacked_bi_dynamic_rnn(self, x, h, seq_length):
        """stacked bidirectional dynamic_rnn"""
        pre_layer = x
        h_n = ()
        c_n = ()
        output = 0
        for i in range(self.num_layers):
            offset = i * 2
            if self.has_bias:
                w_f_ih, w_f_hh, b_f_ih, b_f_hh = \
                    self.w_ih_list[offset], self.w_hh_list[offset], \
                    self.b_ih_list[offset], self.b_hh_list[offset]
                w_b_ih, w_b_hh, b_b_ih, b_b_hh = \
                    self.w_ih_list[offset + 1], self.w_hh_list[offset + 1], \
                    self.b_ih_list[offset + 1], self.b_hh_list[offset + 1]
            else:
                w_f_ih, w_f_hh = self.w_ih_list[offset], self.w_hh_list[offset]
                w_b_ih, w_b_hh = self.w_ih_list[offset + 1], self.w_hh_list[offset + 1]
                b_f_ih, b_f_hh, b_b_ih, b_b_hh = None, None, None, None
            if self.is_lstm:
                h_f_i = (h[0][offset], h[1][offset])
                h_b_i = (h[0][offset + 1], h[1][offset + 1])
            else:
                h_f_i = h[offset]
                h_b_i = h[offset + 1]
            if seq_length is None:
                x_b = self.reverse(pre_layer)
            else:
                x_b = self.reverse_sequence(pre_layer, seq_length)
            output_f, h_t_f = self.rnn(pre_layer, h_f_i, seq_length, w_f_ih, w_f_hh, b_f_ih, b_f_hh)
            output_b, h_t_b = self.rnn(x_b, h_b_i, seq_length, w_b_ih, w_b_hh, b_b_ih, b_b_hh)
            if seq_length is None:
                output_b = self.reverse(output_b)
            else:
                output_b = self.reverse_sequence(output_b, seq_length)
            output = P.Concat(2)((output_f, output_b))
            pre_layer = self.dropout_op(output) if (self.dropout != 0 and i < self.num_layers - 1) else output
            if self.is_lstm:
                h_n += (
                    h_t_f[0],
                    h_t_b[0],
                )
                c_n += (
                    h_t_f[1],
                    h_t_b[1],
                )
            else:
                h_n += (
                    h_t_f,
                    h_t_b,
                )
        if self.is_lstm:
            h_n = P.Concat(0)(h_n)
            c_n = P.Concat(0)(c_n)
            h_n = h_n.view(h[0].shape)
            c_n = c_n.view(h[1].shape)
            return output, (h_n.view(h[0].shape), c_n.view(h[1].shape))
        h_n = P.Concat(0)(h_n)
        return output, h_n.view(h.shape)

    def _stacked_dynamic_rnn(self, x, h, seq_length):
        """stacked mutil_layer dynamic_rnn"""
        pre_layer = x
        h_n = ()
        c_n = ()
        output = 0
        for i in range(self.num_layers):
            if self.has_bias:
                w_ih, w_hh, b_ih, b_hh = self.w_ih_list[i], self.w_hh_list[i], self.b_ih_list[i], self.b_hh_list[i]
            else:
                w_ih, w_hh = self.w_ih_list[i], self.w_hh_list[i]
                b_ih, b_hh = None, None
            if self.is_lstm:
                h_i = (h[0][i], h[1][i])
            else:
                h_i = h[i]
            output, h_t = self.rnn(pre_layer, h_i, seq_length, w_ih, w_hh, b_ih, b_hh)
            pre_layer = self.dropout_op(output) if (self.dropout != 0 and i < self.num_layers - 1) else output
            if self.is_lstm:
                h_n += (h_t[0], )
                c_n += (h_t[1], )
            else:
                h_n += (h_t, )
        if self.is_lstm:
            h_n = P.Concat(0)(h_n)
            c_n = P.Concat(0)(c_n)
            h_n = h_n.view(h[0].shape)
            c_n = c_n.view(h[1].shape)
            return output, (h_n.view(h[0].shape), c_n.view(h[1].shape))
        h_n = P.Concat(0)(h_n)
        return output, h_n.view(h.shape)

    def construct(self, x, hx=None, seq_length=None):
        '''Defines the RNN like operators performed'''
        max_batch_size = x.shape[0] if self.batch_first else x.shape[1]
        num_directions = 2 if self.bidirectional else 1
        x_dtype = x.dtype
        if hx is not None:
            if not self.is_lstm:
                _check_input_dtype_same_and_valid(['x', 'hx'], [x_dtype, hx.dtype], \
                                                  [ms.float32, ms.float16], self.cls_name)
            else:
                _check_input_dtype_same_and_valid(['x', 'hx[0]', 'hx[1]'], [x_dtype, hx[0].dtype, hx[1].dtype], \
                                                 [ms.float32, ms.float16], self.cls_name)
        else:
            hx = _init_state((self.num_layers * num_directions, max_batch_size, self.hidden_size), x_dtype, self.is_lstm)
        if self.batch_first:
            x = P.Transpose()(x, (1, 0, 2))
        if self.bidirectional:
            x_n, hx_n = self._stacked_bi_dynamic_rnn(x, hx, seq_length)
        else:
            x_n, hx_n = self._stacked_dynamic_rnn(x, hx, seq_length)
        if self.batch_first:
            x_n = P.Transpose()(x_n, (1, 0, 2))
        if not self.is_lstm:
            return x_n.astype(x_dtype), hx_n.astype(x_dtype)
        return x_n.astype(x_dtype), (hx_n[0].astype(x_dtype), hx_n[1].astype(x_dtype))


class layernorm(Cell):

    def __init__(self, normalized_shape, gamma, beta, eps, input_shape):
        super(layernorm, self).__init__()
        begin_norm_axis = len(input_shape) - len(normalized_shape)
        begin_params_axis = len(input_shape) - len(normalized_shape)
        self.gamma = gamma
        self.beta = beta
        self.layernorm = LayerNorm(begin_norm_axis=begin_norm_axis, begin_params_axis=begin_params_axis, epsilon=eps)

    def construct(self, inputs):

        output, _, _ = self.layernorm(inputs, self.gamma, self.beta)

        return output


class multiheadattention(Cell):

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
        super(multiheadattention, self).__init__()
        self.embed_dim_check = embed_dim
        self.num_heads = num_heads
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
        self.bmm = P.BatchMatMul()
        self.bias_add = P.BiasAdd()
        self.cast = P.Cast()
        self.expand_dims = P.ExpandDims()
        self.logical_or = P.LogicalOr()
        self.fill = P.Fill()
        self.softmax = P.Softmax()
        self.dropout = P.Dropout(float(1 - dropout))
        self.reduce_sum = P.ReduceSum()

    def construct(self, q, k, v, attn_mask, key_padding_mask):
        #transpose tensor shape
        k = q if k is None else k
        v = q if v is None else v
        if self.batch_first:
            q = q.transpose((1, 0, 2))
            k = k.transpose((1, 0, 2))
            v = v.transpose((1, 0, 2))

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
        q = P.matmul(q, self.q_weight)
        if self.q_bias is not None:
            q = q.transpose((0, 2, 1))
            q = self.bias_add(q, self.q_bias)
            q = q.transpose((0, 2, 1))
        k = P.matmul(k, self.k_weight)
        if self.k_bias is not None:
            k = k.transpose((0, 2, 1))
            k = self.bias_add(k, self.k_bias)
            k = k.transpose((0, 2, 1))
        v = P.matmul(v, self.v_weight)
        if self.v_bias is not None:
            v = v.transpose((0, 2, 1))
            v = self.bias_add(v, self.v_bias)
            v = v.transpose((0, 2, 1))

        # check and prep attention mask
        if attn_mask is not None:
            if attn_mask.dtype == ms.uint8:
                warnings.warn("attn_mask tensor dtype should better be bool.")
                attn_mask = self.cast(attn_mask, ms.bool_)
            elif attn_mask.dtype not in (ms.float32, ms.float64, ms.bool_):
                raise TypeError(
                    "attn_mask tensor dtype should be in (ms.float32, ms.float64, ms.bool_,ms.uint8),"
                    "but got {}".format(attn_mask.dtype)
                )
            if attn_mask.ndim == 2:
                if attn_mask.shape != (tgt_len, src_len):
                    raise ValueError(
                        "The shape of 2D attn_mask should be {}, but got {}.".format(
                            (tgt_len, src_len), attn_mask.shape
                        )
                    )
                attn_mask = self.expand_dims(attn_mask, 0)
                # broadcast_to = P.BroadcastTo((batch_size * self.num_heads, tgt_len, src_len))
                # attn_mask = broadcast_to(attn_mask)
            elif attn_mask.ndim == 3:
                size_3d = (batch_size * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != size_3d:
                    raise ValueError(
                        "The shape of 3D attn_mask should be {}, but got {}.".format(size_3d, attn_mask.shape)
                    )
            else:
                raise ValueError("attn_mask's dimension {} is not supported.".format(attn_mask.ndim))

        q = q.reshape((tgt_len, batch_size * self.num_heads, head_dim)).transpose((1, 0, 2))
        k = k.reshape((src_len, batch_size * self.num_heads, head_dim)).transpose((1, 0, 2))
        v = v.reshape((src_len, batch_size * self.num_heads, head_dim)).transpose((1, 0, 2))

        #check and prep key padding mask
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch_size, src_len):
                raise ValueError(
                    "Expecting key_padding_mask shape is {}, but got {}.".format(
                        (batch_size, src_len), key_padding_mask.shape
                    )
                )

            if key_padding_mask.dtype == ms.uint8:
                warnings.warn("key_padding_mask tensor dtype should better be bool.")
                key_padding_mask = self.cast(key_padding_mask, ms.bool_)
            elif key_padding_mask.dtype != ms.bool_:
                raise TypeError(
                    "key_padding_mask tensor dtype should be 'bool' or 'uint8', but got {}.".format(
                        key_padding_mask.dtype
                    )
                )

            key_padding_mask = key_padding_mask.reshape((batch_size, 1, 1, src_len))
            broadcast_to = P.BroadcastTo((batch_size, self.num_heads, 1, src_len))
            key_padding_mask = broadcast_to(key_padding_mask)
            key_padding_mask = key_padding_mask.reshape((batch_size * self.num_heads, 1, src_len))
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == ms.bool_:
                attn_mask = self.logical_or(attn_mask, key_padding_mask)
            else:
                key_padding_mask_inf = np.full_like(key_padding_mask, fill_value=float('-inf'), dtype=ms.float32)
                attn_mask = np.where(key_padding_mask, key_padding_mask_inf, attn_mask)

        if attn_mask is not None and attn_mask.dtype == ms.bool_:
            new_attn_mask_zero = np.zeros_like(attn_mask, dtype=ms.float32)
            new_attn_mask_inf = np.full_like(attn_mask, fill_value=float('-inf'), dtype=ms.float32)
            attn_mask = np.where(attn_mask, new_attn_mask_inf, new_attn_mask_zero)

        q = q / math.sqrt(embed_dim)
        k = k.transpose((0, 2, 1))
        attn = self.bmm(q, k)
        if attn_mask is not None:
            attn += attn_mask
        attn = self.softmax(attn)
        if self.train:
            attn = self.dropout(attn)[0]

        output = self.bmm(attn, v)
        output = output.transpose((1, 0, 2)).reshape((tgt_len, batch_size, embed_dim))
        output = P.matmul(output, self.out_weight)
        if self.out_bias is not None:
            output = output.transpose((0, 2, 1))
            output = self.bias_add(output, self.out_bias)
            output = output.transpose((0, 2, 1))

        if self.batch_first:
            output = output.transpose((1, 0, 2))

        if self.need_weights:
            attn = attn.reshape((batch_size, self.num_heads, tgt_len, src_len))
            attn = self.reduce_sum(attn, 1) / self.num_heads
            return output, attn
        else:
            return output, None


class BinaryDense(Cell):

    def __init__(self, weights, bias):
        super(BinaryDense, self).__init__()
        self.weights = weights
        self.bias = bias

    def construct(self, inputs):
        raise NotImplementedError


class DorefaDense(Cell):

    def __init__(self, weights, bias, bitW, bitA):
        super(DorefaDense, self).__init__()
        self.weights = weights
        self.bias = bias
        self.bitW = bitW
        self.bitA = bitA

    def construct(self, inputs):
        raise NotImplementedError


class TernaryDense(Cell):

    def __init__(self, weights, bias):
        super(TernaryDense, self).__init__()
        self.weights = weights
        self.bias = bias

    def construct(self, inputs):
        raise NotImplementedError


class QuanDense(Cell):

    def __init__(self, weights, bias, bitW, bitA):
        super(QuanDense, self).__init__()
        self.weights = weights
        self.bias = bias
        self.bitW = bitW
        self.bitA = bitA

    def construct(self, inputs):
        raise NotImplementedError


class QuanDenseBn(Cell):

    def __init__(
        self, weights, scale_para, offset_para, moving_mean, moving_variance, decay, bitW, bitA, epsilon, is_train
    ):
        super(QuanDenseBn, self).__init__()
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

    def construct(self, inputs):
        raise NotImplementedError


class TernaryConv(Cell):

    def __init__(self, weights, strides, padding, data_format, dilations):
        self.weights = weights
        self.strides = strides
        self.dilations = dilations
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)

    def construct(self, inputs):
        raise NotImplementedError


class QuanConv(Cell):

    def __init__(self, weights, strides, padding, data_format, dilations, bitW, bitA):
        self.weights = weights
        self.strides = strides
        self.dilations = dilations
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.bitW = bitW
        self.bitA = bitA

    def construct(self, inputs):
        raise NotImplementedError


class QuanConvBn(Cell):

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

    def construct(self, inputs):
        raise NotImplementedError
