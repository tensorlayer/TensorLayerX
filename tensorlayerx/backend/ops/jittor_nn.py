#! /usr/bin/python
# -*- coding: utf-8 -*-


# Unified nn API for TensorLayerX, using Jittor as backend.
# Similar to file ./torch_nn.py and ./oneflow_nn.py

import jittor as jt
import jittor.nn as nn
import collections
from itertools import repeat
from jittor import Module, init , flatten
import math
from abc import abstractmethod

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
    elif isinstance(padding, tuple) or isinstance(padding, int):
        return padding
    else:
        raise Exception("Unsupported padding: " + str(padding))
    return padding

def preprocess_padding(padding, dim='2d'):
    check_padding(padding, dim)
    if dim == '1d':
        out_padding = (0, 0, padding, padding)
    elif dim == '2d':
        if isinstance(padding, tuple):
            out_padding = (padding[0], padding[0], padding[1], padding[1])
        else:
            out_padding = padding
    elif dim == '3d':
        if isinstance(padding, tuple):
            out_padding = (padding[0], padding[0], padding[1], padding[1], padding[2], padding[2])
        else:
            out_padding = padding
    else:
        raise RuntimeError("Unsupported input dimensions.")
    return out_padding



def check_padding(padding, dim='2d'):
    if dim == '1d' and isinstance(object, tuple):
        raise RuntimeError("expected padding to be a single integer value or a list of 1 values to match the convolution dimensions.")
    if dim == '2d' and isinstance(padding, tuple) and len(padding) > 2:
        raise RuntimeError("expected padding to be a single integer value or a list of 2 values to match the convolution dimensions.")
    if dim == '3d' and isinstance(padding, tuple) and len(padding) > 3:
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

    if data_format in ["channels_last", "NHWC"]:
        data_format = "NHWC"
    elif data_format in ["channels_first", "NCHW"]:
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
    shape = x.shape
    if len(shape) == 3:
        x = jt.transpose(x, (0, 2, 1))
    elif len(shape) == 4:
        x = jt.transpose(x, (0, 2, 3, 1))
    elif len(shape) == 5:
        x = jt.transpose(x, (0, 2, 3, 4, 1))
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
    shape = x.shape
    if len(shape) == 3:
        x = jt.transpose(x, (0, 2, 1))
    elif len(shape) == 4:
        x = jt.transpose(x, (0, 3, 1, 2))
    elif len(shape) == 5:
        x = jt.transpose(x, (0, 4, 1, 2, 3))
    # else:
    #     raise Exception("Unsupported dimensions")
    return x

class ReLU(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return nn.relu(x)


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

    return nn.relu(x)


class ELU(object):

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, x):
        return nn.elu(x, alpha=self.alpha)


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

    return nn.elu(x, alpha=alpha)


class ReLU6(object):

    def __call__(self, x):
        return nn.relu6(x)


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

    return nn.relu6(x)


class LeakyReLU(object):
# jittor.nn. leaky_relu ( x , scale = 0.01 )
    def __init__(self, negative_slope=0.01):
        self.negative_slope = negative_slope

    def __call__(self, x):
        return nn.leaky_relu(x, scale=self.negative_slope)


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

    return nn.leaky_relu(x, scale=negative_slope)


class Softplus(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return nn.softplus(x)


class Tanh(object):
# jittor.nn.hardtanh(x, min_val=-1, max_val=1)
    def __init__(self):
        super(Tanh, self).__init__()
        self.tanh = nn.Sigmoid()

    def __call__(self, x):
        return self.tanh(x)


class Sigmoid(object):
# classjittor.nn.Sigmoid
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.sigmoid = nn.Sigmoid()
        pass

    def __call__(self, x):
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
    outputs = nn.Sigmoid()
    return outputs(x)


class Softmax(object):

    def __init__(self, axis=None):
        self.axis = axis

    def __call__(self, x):
        return nn.softmax(x, dim=self.axis)


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

    return nn.softmax(logits, axis)


class GeLU(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return nn.gelu(x)


def gelu(x):

    return nn.gelu(x)


class Dropout(object):

    def __init__(self, p=0.5, seed=0 , is_train=False):
        self.p = p
        self.seed = seed
        self.is_train = is_train
    def __call__(self, inputs):
        return nn.dropout(inputs, p=self.p, is_train=self.is_train)

def dropout(x, p=0.5, is_train=False):
    return nn.dropout(x , p=p, is_train=is_train)


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
        outputs = x + bias
        if len(x.shape) > 2 and self.data_format == 'channels_first':
            outputs = nhwc_to_nchw(outputs)
        return outputs


def bias_add(x, bias, data_format=None):
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

    add_obj = BiasAdd(data_format=data_format)
    return add_obj(x, bias)



class Conv1D(object):

    def __init__(self, stride, padding, data_format='NWC', dilations=None, out_channel=None, k_size=None, groups=1):
        self.stride = stride
        self.dilations = dilations
        self.groups = groups
        self.data_format, self.padding = preprocess_1d_format(data_format, padding)
        # self.conv1d = nn.Conv1d()
    def __call__(self, input, filters):
        if self.data_format == 'NLC':
            input = nhwc_to_nchw(input)
        if self.padding == 'same':
            out = self.conv1d_same_padding(input, filters)
        else:
            
            out = nn.Conv1d(input, filters, stride=self.stride, padding=self.padding,
                           dilation=self.dilations, groups=self.groups)
        if self.data_format == 'NLC':
            out = nchw_to_nhwc(out)

        return out

    def conv1d_same_padding(self, input, filters):
        rows_odd, padding_rows = same_padding(input, filters, self.stride, 1)
        if rows_odd:
            input = nn.pad(input, [0, int(rows_odd)], 'replicate')
        
        return nn.Conv1d(input, filters, stride=self.stride, padding=(padding_rows // 2), groups=self.groups)



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

    return Conv1D(stride=stride, padding=padding, data_format=data_format, dilations=dilations)(input, filters)


def same_padding(input, weight, strides, dilations):
    #                     H(in) + 2* padding[0] - dilation[0] * (Ksize[0] - 1) - 1
    # H(out) = = floor( --------------------------------------------------------------   + 1 )
    #                                        stride[0]
    if isinstance(weight, jt.array):
        if len(input.shape) == 3:
            filter_rows = weight.size(2)
        if len(input.shape) == 4:
            filter_rows = weight.size(2)
            filter_cols = weight.size(3)
        elif len(input.shape) == 5:
            filter_rows = weight.size(2)
            filter_cols = weight.size(3)
            filter_depth = weight.size(4)
    else:
        if len(input.shape) == 3:
            filter_rows = weight[0]
        elif len(input.shape) == 4:
            filter_rows = weight[0]
            filter_cols = weight[1]
        elif len(input.shape) == 5:
            filter_rows = weight[0]
            filter_cols = weight[1]
            filter_depth = weight[2]

    if len(input.shape) == 3:
        input_rows = input.size(2)
        out_rows = (input_rows + strides - 1) // strides
        padding_rows = max(0, (out_rows - 1) * strides + (filter_rows - 1) * dilations + 1 - input_rows)
        rows_odd = (padding_rows % 2 != 0)
        return rows_odd, padding_rows

    if len(input.shape) == 4:
        input_rows = input.size(2)
        input_cols = input.size(3)

        # filter_rows = weight.size(2)
        # filter_cols = weight.size(3)

        out_rows = (input_rows + strides[0] - 1) // strides[0]
        out_cols = (input_cols + strides[1] - 1) // strides[1]

        padding_rows = max(0, (out_rows - 1) * strides[0] + (filter_rows - 1) * dilations[0] + 1 - input_rows)
        padding_cols = max(0, (out_cols - 1) * strides[1] + (filter_cols - 1) * dilations[1] + 1 - input_cols)

        rows_odd = (padding_rows % 2 != 0)
        cols_odd = (padding_cols % 2 != 0)
        return rows_odd, cols_odd, padding_rows, padding_cols

    if len(input.shape) == 5:
        input_rows = input.size(2)
        input_cols = input.size(3)
        input_depth = input.size(4)

        # filter_rows = weight.size(2)
        # filter_cols = weight.size(3)
        # filter_depth = weight.size(4)

        out_rows = (input_rows + strides[0] - 1) // strides[0]
        out_cols = (input_cols + strides[1] - 1) // strides[1]
        out_depth = (input_depth + strides[2] - 1) // strides[2]

        padding_rows = max(0, (out_rows - 1) * strides[0] + (filter_rows - 1) * dilations[0] + 1 - input_rows)
        padding_cols = max(0, (out_cols - 1) * strides[1] + (filter_cols - 1) * dilations[1] + 1 - input_cols)
        padding_depth = max(0, (out_depth - 1) * strides[2] + (filter_depth - 1) * dilations[2] + 1 - input_depth)

        rows_odd = (padding_rows % 2 != 0)
        cols_odd = (padding_cols % 2 != 0)
        depth_odd = (padding_depth % 2 != 0)
        return rows_odd, cols_odd, depth_odd, padding_rows, padding_cols, padding_depth


class Conv2D(object):

    def __init__(self, strides, padding, data_format='NHWC', dilations=None, out_channel=None, k_size=None, groups=1):
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        if self.data_format is 'NHWC':
            self.strides = (strides[1], strides[2])
            self.dilations = (dilations[1], dilations[2])
        elif self.data_format is 'NCHW':
            self.strides = (strides[2], strides[3])
            self.dilations = (dilations[2], dilations[3])
        self.groups = groups

    def __call__(self, input, filters):
        if self.data_format == 'NHWC':
            input = nhwc_to_nchw(input)

        if self.padding == 'same':
            output = self.conv2d_same_padding(input, filters)
        else:
            output = nn.conv2d(input, filters, stride=self.strides, padding=self.padding,
                              dilation=self.dilations, groups=self.groups)

        if self.data_format == 'NHWC':
            output = nchw_to_nhwc(output)
        return output

    def conv2d_same_padding(self, input, weight, bias=None):
        rows_odd, cols_odd, padding_rows, padding_cols = same_padding(input, weight, self.strides, self.dilations)
        if rows_odd or cols_odd:
            input = nn.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

        return nn.conv2d(
            input, weight, bias, self.strides, padding=(padding_rows // 2, padding_cols // 2), dilation=self.dilations,
            groups=self.groups
        )


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

    if data_format == 'NHWC':
        input = nhwc_to_nchw(input)

    output = nn.conv2d(input, filters, stride=strides, padding=padding, dilation=dilations)

    if data_format == 'NHWC':
        output = nchw_to_nhwc(output)
    return output


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
        if self.data_format == 'NDHWC':
            input = nhwc_to_nchw(input)

        if self.padding == 'same':
            out = self.conv3d_same_padding(input, weight=filters)
        else:
            out = nn.conv3d(input, weight=filters, stride=self._strides, padding=self.padding, dilation=self._dilations)

        if self.data_format == 'NDHWC':
            out = nchw_to_nhwc(out)

        return out

    def conv3d_same_padding(self, input, weight, bias=None, groups=1):
        rows_odd, cols_odd, depth_odd, padding_rows, padding_cols, padding_depth = same_padding(input, weight,
                                                                                                self._strides, self._dilations)
        if rows_odd or cols_odd or depth_odd:
            input = nn.pad(input, [0, int(cols_odd), 0, int(rows_odd), 0, int(depth_odd)])

        return nn.conv3d(
            input, weight, bias, self._strides, padding=(padding_rows // 2, padding_cols // 2, padding_depth//2),
            dilation=self._dilations, groups=groups
        )


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

    return Conv3D(strides=strides, padding=padding, data_format=data_format, dilations=dilations)(input, filters)


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

    raise NotImplementedError


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

    raise NotImplementedError


class MaxPool1d(object):

    def __call__():
        return NotImplementedError


class MaxPool(object):

    def __init__(self, ksize, strides, padding, return_mask = False, data_format=None):
        self.ksize = ksize
        self.strides = strides
        self.return_mask = return_mask
        if data_format in ['channels_last', 'NLC', 'NWC', 'NHWC', 'NDHWC']:
            self.data_format = 'channels_last'
        elif data_format in ['channels_first', 'NCL', 'NCW', 'NCHW', 'NCDHW']:
            self.data_format = 'channels_first'
        self.padding = padding
        if self.padding in ['VALID', 'valid']:
            self.padding = 0

    def __call__(self, inputs):
        if self.data_format == 'channels_last':
            inputs = nhwc_to_nchw(inputs)
        if len(inputs.shape) == 2 or len(inputs.shape) == 3:
            raise NotImplementedError
        
        if len(inputs.shape) == 4:
            if self.padding in ['SAME', 'same']:
                out = self.maxpool2d_same_padding(inputs)
            else:
                out = nn.max_pool2d(inputs, self.ksize, self.strides, padding=self.padding,
                            return_indices=self.return_mask)
        if len(inputs.shape) == 5:
            if self.padding in ['SAME', 'same']:
                out = self.maxpool3d_same_padding(inputs)
            else:
                out = nn.max_pool3d(inputs, self.ksize, self.strides, padding=self.padding,
                            return_indices=self.return_mask)

        if self.data_format == 'channels_last':
            if self.return_mask:
                    outputs = [None, None]
                    outputs[0] = nchw_to_nhwc(out[0])
                    outputs[1] = nchw_to_nhwc(out[1])
                    return outputs
            else:
                return nchw_to_nhwc(out)
        else:
            return out


    def maxpool2d_same_padding(self, input):
        rows_odd, cols_odd, padding_rows, padding_cols = same_padding(input, self.ksize, self.strides, (1, 1))
        if rows_odd or cols_odd:
            # TODO The fill value for maxpool is -INF.
            input = nn.pad(input, [0, int(rows_odd), 0, int(cols_odd)], 'constant', float('-inf'))

        return nn.max_pool2d(input, self.ksize, self.strides, padding=(padding_rows // 2, padding_cols // 2),
                            return_indices=self.return_mask)

    def maxpool3d_same_padding(self, input):
        rows_odd, cols_odd, depth_odd, padding_rows, padding_cols, padding_depth = same_padding(
            input, self.ksize, self.strides, (1, 1, 1)
        )
        if rows_odd or cols_odd or depth_odd:
            input = nn.pad(input, [0, int(cols_odd), 0, int(rows_odd), 0, int(depth_odd)], 'constant', float('-inf'))
        return nn.max_pool3d(
                input, self.ksize, self.strides, padding=(padding_rows // 2, padding_cols // 2, padding_depth // 2),
                return_indices=self.return_mask
        )


def max_pool(input, ksize, strides, padding, return_mask, data_format=None):
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

    maxpool_obj = MaxPool(ksize, strides, padding, return_mask, data_format)
    return maxpool_obj(input)

def max_pool1d(input, kernel_size, stride=None, padding=0, return_mask=False, data_format='NCL'):
    raise NotImplementedError

def max_pool2d(input, kernel_size, stride=None, padding=0, return_mask=False, data_format='NCHW'):

    maxpool_obj = MaxPool(kernel_size, stride, padding, return_mask, data_format)
    return maxpool_obj(input)

def max_pool3d(input, kernel_size, stride=None, padding=0, return_mask=False, data_format="NCDHW"):

    maxpool_obj = MaxPool(kernel_size, stride, padding, return_mask, data_format)
    return maxpool_obj(input)


class AvgPool1d(object):

    def __call__(inputs):
        raise NotImplementedError


class AvgPool(object):

    def __init__(self, ksize, strides, padding, data_format=None):
        self.ksize = ksize
        self.strides = strides
        if data_format in ['channels_last', 'NLC', 'NWC', 'NHWC', 'NDHWC']:
            self.data_format = 'channels_last'
        elif data_format in ['channels_first', 'NCL', 'NCW', 'NCHW', 'NCDHW']:
            self.data_format = 'channels_first'
        self.padding = padding
        if self.padding in ['VALID', 'valid']:
            self.padding = 0

    def __call__(self, inputs):
        if self.data_format == 'channels_last':
            inputs = nhwc_to_nchw(inputs)
        if len(inputs.shape) == 2 or len(inputs.shape) == 3:
            raise NotImplementedError
               
        if len(inputs.shape) == 4:
            if self.padding in ['SAME', 'same']:
                out = self.avgpool2d_same_padding(inputs)
            else:
                out = nn.avg_pool2d(inputs, self.ksize, self.strides, padding=self.padding)
        if len(inputs.shape) == 5:
            if self.padding in ['SAME', 'same']:
                out = self.avgpool3d_same_padding(inputs)
            else:
                out = nn.AvgPool2d(inputs, self.ksize, self.strides, padding=self.padding)

        if self.data_format == 'channels_last':
            return nchw_to_nhwc(out)
        else:
            return out


    def avgpool2d_same_padding(self, input):
        rows_odd, cols_odd, padding_rows, padding_cols = same_padding(input, self.ksize, self.strides, (1, 1))
        if rows_odd or cols_odd:
            # TODO The fill value for maxpool is -INF.
            input = nn.pad(input, [0, int(rows_odd), 0, int(cols_odd)], mode='replicate')

        return nn.avg_pool2d(input, self.ksize, self.strides, padding=(padding_rows // 2, padding_cols // 2))

    def avgpool3d_same_padding(self, input):
        rows_odd, cols_odd, depth_odd, padding_rows, padding_cols, padding_depth = same_padding(
            input, self.ksize, self.strides, (1, 1, 1)
        )
        if rows_odd or cols_odd or depth_odd:
            input = nn.pad(input, [0, int(cols_odd), 0, int(rows_odd), 0, int(depth_odd)], mode='replicate')
        return nn.AvgPool3d(
                input, self.ksize, self.strides, padding=(padding_rows // 2, padding_cols // 2, padding_depth // 2)
        )


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

    avg_pool_obj = AvgPool(ksize, strides, padding)
    return avg_pool_obj(input)

def avg_pool1d(input, kernel_size, stride=None, padding=0, data_format='NCL'):
    raise NotImplementedError

def avg_pool2d(input, kernel_size, stride=None, padding=0, data_format='NCHW'):
    data_format, padding = preprocess_2d_format(data_format, padding)
    avg_pool_obj = AvgPool(kernel_size, stride, padding, data_format)
    return avg_pool_obj(input)

def avg_pool3d(input, kernel_size, stride=None, padding=0, data_format='NCDHW'):
    data_format, padding = preprocess_3d_format(data_format, padding)
    avg_pool_obj = AvgPool(kernel_size, stride, padding, data_format)
    return avg_pool_obj(input)

class MaxPool3d(object):

    def __init__(self, ksize, strides, padding, return_mask, data_format=None):
        self.data_format, self.padding = preprocess_3d_format(data_format, padding)
        self.max_pool3d = MaxPool(ksize, strides, padding, return_mask, data_format)

    def __call__(self, inputs):
        return self.max_pool3d(inputs)


# def max_pool3d(input, ksize, strides, padding, data_format=None):
#     """
#     Performs the max pooling on the input.
#
#     Parameters
#     ----------
#     input : tensor
#          A 5-D Tensor of the format specified by data_format.
#     ksize : int or list of ints
#         An int or list of ints that has length 1, 3 or 5.
#         The size of the window for each dimension of the input tensor.
#     strides : int or list of ints
#         An int or list of ints that has length 1, 3 or 5.
#         The stride of the sliding window for each dimension of the input tensor.
#     padding : string
#         'VALID' or 'SAME'. The padding algorithm. See the "returns" section of tf.ops.convolution for details.
#     data_format : string
#          "NDHWC", "NCDHW". Defaults to "NDHWC". The data format of the input and output data.
#          With the default format "NDHWC", the data is stored in the order of: [batch, in_depth, in_height, in_width, in_channels].
#          Alternatively, the format could be "NCDHW", the data storage order is: [batch, in_channels, in_depth, in_height, in_width].
#     name : string
#          A name for the operation (optional).
#
#     Returns
#     -------
#         A Tensor of format specified by data_format. The max pooled output tensor.
#     """
#
#     data_format, padding = preprocess_3d_format(data_format, padding)
#     max_pool3d_obj = MaxPool(ksize, strides, padding, data_format)
#     return max_pool3d_obj(input)


class AvgPool3d(object):

    def __init__(self, ksize, strides, padding, data_format=None):
        self.data_format, self.padding = preprocess_3d_format(data_format, padding)
        self.avg_pool3d_obj = AvgPool(ksize, strides, self.padding, self.data_format)

    def __call__(self, inputs):
        return self.avg_pool3d_obj(inputs)


# def avg_pool3d(input, ksize, strides, padding, data_format=None):
#     """
#     Performs the average pooling on the input.
#
#     Parameters
#     ----------
#     input : tensor
#         A 5-D Tensor of shape [batch, height, width, channels] and type float32, float64, qint8, quint8, or qint32.
#     ksize : int or list of ints
#         An int or list of ints that has length 1, 3 or 5. The size of the window for each dimension of the input tensor.
#     strides : int or list of ints
#         An int or list of ints that has length 1, 3 or 5.
#         The stride of the sliding window for each dimension of the input tensor.
#     padding : string
#         'VALID' or 'SAME'. The padding algorithm. See the "returns" section of tf.ops.convolution for details.
#     data_format : string
#         'NDHWC' and 'NCDHW' are supported.
#     name : string
#         Optional name for the operation.
#
#     Returns
#     -------
#         A Tensor with the same type as value. The average pooled output tensor.
#     """
#
#     avg_pool_obj = AvgPool(ksize, strides, padding, data_format)
#     return avg_pool_obj(input)


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
        pool_obj = MaxPool(window_shape, strides, padding, data_format)
    elif pooling_type in ["AVG", "avg"]:
        pool_obj = AvgPool(window_shape, strides, padding, data_format)
    else:
        raise ValueError('Unsupported pool_mode: ' + str(pooling_type))

    return pool_obj(input)


class DepthwiseConv2d(object):

    def __init__(self, strides, padding, data_format=None, dilations=None, ksize=None, channel_multiplier=1, in_channels=None):
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        if self.data_format is 'NHWC':
            self.strides = (1, strides[0], strides[1], 1)
            self.dilations = (1, dilations[0], dilations[1], 1)
        elif self.data_format is 'NCHW':
            self.strides = (1, 1, strides[0], strides[1])
            self.dilations = (1, 1, dilations[0], dilations[1])
        self.depthwise = Conv2D(padding=self.padding, strides=self.strides, data_format=self.data_format,
                                dilations=self.dilations, groups=in_channels)
        self.pointwise = Conv2D(strides=(1, 1, 1, 1), padding=self.padding, data_format=self.data_format, dilations=self.dilations, k_size=1)

    def __call__(self, input, filter, point_filter=None):
        depthwise_conv = self.depthwise(input, filter)
        pointwise_conv = self.pointwise(depthwise_conv, point_filter)

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

    depthwise_conv2d_obj = DepthwiseConv2d(strides, padding, data_format, dilations)
    return depthwise_conv2d_obj(input, filter)


def same_padding_deconvolution(input, weight, strides, dilations):
    #H(out) = floor((H(in) - 1)*stride[0] - 2* padding[0] + dilation[0] * (ksize[0]-1) + 1)

    if isinstance(weight, jt.array):
        if len(input.shape) == 3:
            filter_rows = weight.size(2)
        if len(input.shape) == 4:
            filter_rows = weight.size(2)
            filter_cols = weight.size(3)
        elif len(input.shape) == 5:
            filter_rows = weight.size(2)
            filter_cols = weight.size(3)
            filter_depth = weight.size(4)
    else:
        if len(input.shape) == 3:
            filter_rows = weight[0]
        elif len(input.shape) == 4:
            filter_rows = weight[0]
            filter_cols = weight[1]
        elif len(input.shape) == 5:
            filter_rows = weight[0]
            filter_cols = weight[1]
            filter_depth = weight[2]

    if len(input.shape) == 3:
        input_rows = input.size(2)
        out_rows = input_rows * strides - strides + 1
        padding_rows = max(0, (input_rows-1) * strides + (filter_rows - 1) * dilations + 1 - out_rows)
        rows_odd = (padding_rows % 2 != 0)
        return rows_odd, padding_rows

    if len(input.shape) == 4:
        input_rows = input.size(2)
        input_cols = input.size(3)

        out_rows = input_rows * strides[0] - strides[0] + 1
        out_cols = input_cols * strides[1] - strides[1] + 1


        padding_rows = max(0, (input_rows - 1) * strides[0] + (filter_rows - 1) * dilations[0] + 1 - out_rows)
        padding_cols = max(0, (input_cols - 1) * strides[1] + (filter_cols - 1) * dilations[1] + 1 - out_cols)

        rows_odd = (padding_rows % 2 != 0)
        cols_odd = (padding_cols % 2 != 0)
        return rows_odd, cols_odd, padding_rows, padding_cols

    if len(input.shape) == 5:
        input_rows = input.size(2)
        input_cols = input.size(3)
        input_depth = input.size(4)

        out_rows = input_rows * strides[0] - strides[0] + 1
        out_cols = input_cols * strides[1] - strides[1] + 1
        out_depth = input_depth * strides[2] - strides[2] + 1

        padding_rows = max(0, (input_rows - 1) * strides[0] + (filter_rows - 1) * dilations[0] + 1 - out_rows)
        padding_cols = max(0, (input_cols - 1) * strides[1] + (filter_cols - 1) * dilations[1] + 1 - out_cols)
        padding_depth = max(0, (input_depth - 1) * strides[2] + (filter_depth - 1) * dilations[2] + 1 - out_depth)

        rows_odd = (padding_rows % 2 != 0)
        cols_odd = (padding_cols % 2 != 0)
        depth_odd = (padding_depth % 2 != 0)
        return rows_odd, cols_odd, depth_odd, padding_rows, padding_cols, padding_depth


class Conv1d_transpose(object):

    # def __init__(
    #     self, stride, padding, data_format='NWC', dilations=None, out_channel=None, k_size=None, in_channels=None
    # ):
    #     self.stride = stride
    #     self.dilations = dilations
    #     self.data_format, self.padding = preprocess_1d_format(data_format, padding)

    def __call__(self, input, filters):
        raise NotImplementedError
#         if self.data_format == 'NLC':
#             input = nhwc_to_nchw(input)
#         if self.padding == 'same':
#             out = self.conv1d_transpose_same_padding(input, filters)
#         else:
#             out = F.conv_transpose1d(
#                 input,
#                 weight=filters,
#                 padding=(0 if isinstance(self.padding, str) else self.padding),
#                 stride=self.stride,
#                 dilation=self.dilations
#             )
#         if self.data_format == 'NLC':
#             out = nchw_to_nhwc(out)
#         return out

#     def conv1d_transpose_same_padding(self, input, filters):
#         rows_odd, padding_rows = same_padding_deconvolution(input, filters, self.stride, 1)
#         if rows_odd:
#             input = F.pad(input, [0, int(rows_odd)])
#             out_padding = 0
#         else:
#             out_padding = 1
#         return F.conv_transpose1d(input, weight=filters, padding=(padding_rows // 2), stride=self.stride,
#                                   dilation=self.dilations, output_padding=out_padding)



# def conv1d_transpose(
#     input, filters, output_shape, strides, padding='SAME', data_format='NWC', dilations=None, name=None
# ):
#     """
#     The transpose of conv1d.

#     Parameters
#     ----------
#     input : tensor
#         A 3-D Tensor of type float and shape [batch, in_width, in_channels]
#         for NWC data format or [batch, in_channels, in_width] for NCW data format.
#     filters : tensor
#         A 3-D Tensor with the same type as value and shape [filter_width, output_channels, in_channels].
#         filter's in_channels dimension must match that of value.
#     output_shape : tensor
#         A 1-D Tensor, containing three elements, representing the output shape of the deconvolution op.
#     strides : list
#         An int or list of ints that has length 1 or 3. The number of entries by which the filter is moved right at each step.
#     padding : string
#         'VALID' or 'SAME'. The padding algorithm. See the "returns" section of tf.ops.convolution for details.
#     data_format : string
#         'NWC' and 'NCW' are supported.
#     dilations : list
#          An int or list of ints that has length 1 or 3 which defaults to 1.
#          The dilation factor for each dimension of input. If set to k > 1,
#          there will be k-1 skipped cells between each filter element on that dimension.
#          Dilations in the batch and depth dimensions must be 1.
#     name : string
#         Optional name for the returned tensor.

#     Returns
#     -------
#         A Tensor with the same type as value.
#     """

#     conv1d_transpose_obj = Conv1d_transpose(strides, padding, data_format, dilations)
#     return conv1d_transpose_obj(input, filters)

def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_single = _ntuple(1, "_single")

class Conv2d_transpose(object):

    def __init__(
        self, strides, padding, data_format='NHWC', dilations=None, name=None, out_channels=None, k_size=None,
        in_channels=None, groups = 1, output_padding = 0,
    ):
        self.strides = strides
        self.dilations = dilations
        self.name = name
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.groups = groups
        self.output_padding = output_padding

    def _output_padding(self, input, output_size,
                        stride, padding, kernel_size,
                        num_spatial_dims, dilation = None):
        if output_size is None:
            ret = _single(self.output_padding)  # converting to list if was not already
        else:
            has_batch_dim = input.dim() == num_spatial_dims + 2
            num_non_spatial_dims = 2 if has_batch_dim else 1
            if len(output_size) == num_non_spatial_dims + num_spatial_dims:
                output_size = output_size[num_non_spatial_dims:]
            if len(output_size) != num_spatial_dims:
                raise ValueError(
                    "ConvTranspose{}D: for {}D input, output_size must have {} or {} elements (got {})"
                    .format(num_spatial_dims, input.dim(), num_spatial_dims,
                            num_non_spatial_dims + num_spatial_dims, len(output_size)))

            min_sizes = []
            max_sizes = []
            for d in range(num_spatial_dims):
                dim_size = ((input.size(d + num_non_spatial_dims) - 1) * stride[d] -
                            2 * padding[d] +
                            (dilation[d] if dilation is not None else 1) * (kernel_size[d] - 1) + 1)
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError((
                        "requested an output size of {}, but valid sizes range "
                        "from {} to {} (for an input of {})").format(
                            output_size, min_sizes, max_sizes, input.size()[2:]))

            res = []
            for d in range(num_spatial_dims):
                res.append(output_size[d] - min_sizes[d])

            ret = res
        return ret

    def __call__(self, input, filters, output_size):
        if self.data_format == 'NHWC':
            input = nhwc_to_nchw(input)
        if self.padding == 'same':
            out = self.conv2d_transpore_same(input, filters)
        else:
            out_padding = self._output_padding(input, output_size, self.strides, (0 if isinstance(self.padding, str) else self.padding),
                                               filters.shape,
                                               2, self.dilations)
            out = nn.conv_transpose2d(
                input,
                weight=filters,
                padding=(0 if isinstance(self.padding, str) else self.padding),
                stride=self.strides,
                dilation=self.dilations,
                output_padding = out_padding,
                groups = self.groups
            )
        if self.data_format == 'NHWC':
            out = nchw_to_nhwc(out)
        return out

    def conv2d_transpore_same(self,input, filters):
        rows_odd, cols_odd, padding_rows, padding_cols = same_padding_deconvolution(input, filters, self.strides, (1, 1))
        if rows_odd or cols_odd:
            input = nn.pad(input, [0, int(rows_odd), 0, int(cols_odd)])
            out_padding = 0
        else:
            out_padding = 1
        out = nn.conv_transpose2d(input, weight=filters, padding=(padding_rows // 2, padding_cols // 2), stride=self.strides,
                                 dilation=self.dilations, output_padding=out_padding, groups=self.groups)
        return out


def conv2d_transpose(x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1, data_format='NCHW', output_size=None):
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
    if isinstance(padding, str):
        raise ValueError("padding should be int or tuple of int.")
    def _output_padding(input, output_size,
                        stride, padding, kernel_size,
                        num_spatial_dims, dilation = None):
        if output_size is None:
            ret = _single(output_padding)  # converting to list if was not already
        else:
            has_batch_dim = input.dim() == num_spatial_dims + 2
            num_non_spatial_dims = 2 if has_batch_dim else 1
            if len(output_size) == num_non_spatial_dims + num_spatial_dims:
                output_size = output_size[num_non_spatial_dims:]
            if len(output_size) != num_spatial_dims:
                raise ValueError(
                    "ConvTranspose{}D: for {}D input, output_size must have {} or {} elements (got {})"
                    .format(num_spatial_dims, input.dim(), num_spatial_dims,
                            num_non_spatial_dims + num_spatial_dims, len(output_size)))

            min_sizes = []
            max_sizes = []
            for d in range(num_spatial_dims):
                dim_size = ((input.size(d + num_non_spatial_dims) - 1) * stride[d] -
                            2 * padding[d] +
                            (dilation[d] if dilation is not None else 1) * (kernel_size[d] - 1) + 1)
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError((
                        "requested an output size of {}, but valid sizes range "
                        "from {} to {} (for an input of {})").format(
                            output_size, min_sizes, max_sizes, input.size()[2:]))

            res = []
            for d in range(num_spatial_dims):
                res.append(output_size[d] - min_sizes[d])

            ret = res
        return ret

    if data_format == 'NHWC':
        x = nhwc_to_nchw(x)

    out_padding = _output_padding(x, output_size, stride,
                                           padding,
                                           weight.shape[2:],
                                           2, dilation)
    out = nn.conv_transpose2d(
            x,
            weight=weight,
            bias = bias,
            padding=padding,
            stride=stride,
            dilation=dilation,
            output_padding=out_padding,
            groups=groups
        )
    if data_format == 'NHWC':
        out = nchw_to_nhwc(out)
    return out

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
            input = nhwc_to_nchw(input)
        if self.padding == 'same':
            out = self.conv3d_transpore_same(input, filters)
        else:
            out = nn.conv_transpose3d(
                input,
                weight=filters,
                padding=(0 if isinstance(self.padding, str) else self.padding),
                stride=self.strides,
                dilation=self.dilations
            )
        if self.data_format == 'NDHWC':
            out = nchw_to_nhwc(out)
        return out

    def conv3d_transpore_same(self,input, filters):
        rows_odd, cols_odd, depth_odd, padding_rows, padding_cols, padding_depth = same_padding_deconvolution(
            input, filters, self.strides, (1, 1, 1))
        if rows_odd or cols_odd or depth_odd:
            input = nn.pad(input, [0, int(rows_odd), 0, int(cols_odd), 0, int(depth_odd)])
            out_padding = 0
        else:
            out_padding = 1
        out = nn.conv_transpose3d(input, weight=filters, padding=(padding_rows // 2, padding_cols // 2, padding_depth // 2),
                                 stride=self.strides, dilation=self.dilations, output_padding=out_padding)
        return out


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
    conv3d_transpose_obj = Conv3d_transpose(strides, padding, data_format, dilations)
    return conv3d_transpose_obj(input, filters)


def _to_channel_first_bias(b):

    raise NotImplementedError


def _bias_scale(x, b, data_format):

    raise NotImplementedError


def _bias_add(x, b, data_format):
    
    raise NotImplementedError

# Batch norms exists for jittor but not added here
def batch_normalization(x, mean, variance, offset, scale, variance_epsilon, data_format, name=None):
    """Data Format aware version of tf.nn.batch_normalization."""
    raise NotImplementedError


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

    >>> net = tlx.layers.Input([None, 50, 50, 32], name='input')
    >>> net = tlx.layers.BatchNorm()(net)

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
        self.decay =  1-decay
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

    def __call__(self, inputs):
        if self.data_format == 'channels_last':
            inputs = nhwc_to_nchw(inputs)

        out = nn.batch_norm(inputs,
                                             running_mean=self.moving_mean,
                                             running_var=self.moving_var,
                                             weight=self.gamma,
                                             bias=self.beta,
                                             training=self.is_train,
                                             momentum=self.decay)
        if self.data_format == 'channels_last':
            out = nchw_to_nhwc(out)
        return out


class GroupConv2D(object):

    def __init__(self, strides, padding, data_format, dilations, out_channel, k_size, groups=1):
        self.groups = groups
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.conv2d = Conv2D(strides, self.padding, self.data_format, dilations, groups=self.groups)

    def __call__(self, input, filters):
        return self.conv2d(input, filters)



class SeparableConv1D(object):

    def __init__(self, stride, padding, data_format, dilations, out_channel, k_size, in_channel, depth_multiplier):
        self.data_format, self.padding = preprocess_1d_format(data_format, padding)
        self.depthwise_conv = Conv1D(stride, self.padding, self.data_format, dilations, groups=in_channel)
        self.pointwise_conv = Conv1D(1, self.padding, self.data_format, 1)


    def __call__(self, inputs, depthwise_filters, pointwise_filters):
        depthwise_conv = self.depthwise_conv(inputs, depthwise_filters)
        pointwise_conv = self.pointwise_conv(depthwise_conv, pointwise_filters)
        return pointwise_conv


class SeparableConv2D(object):

    def __init__(self, strides, padding, data_format, dilations, out_channel, k_size, in_channel, depth_multiplier):
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.depthwise_conv = Conv2D(strides, self.padding, self.data_format, dilations, groups=in_channel)
        self.pointwise_conv = Conv2D((1, 1), self.padding, self.data_format, (1, 1))


    def __call__(self, input, filter, point_filter=None):
        depthwise_conv = self.depthwise_conv(input, filter)
        pointwise_conv = self.pointwise_conv(depthwise_conv, point_filter)
        return pointwise_conv


class AdaptiveMeanPool1D(object):

    # def __init__(self, output_size, data_format):
    #     self.data_format, _ = preprocess_1d_format(data_format, None)
    #     self.op = nn.AdaptiveAvgPool1d(output_size=output_size)

    def __call__():
        raise NotImplementedError
        # if self.data_format == 'NLC':
        #     input = nhwc_to_nchw(input)
        # output = self.op(input)
        # if self.data_format == 'NLC':
        #     output = nchw_to_nhwc(output)
        # return output


class AdaptiveMeanPool2D(object):

    # def __init__(self, output_size, data_format):
    #     self.data_format, _ = preprocess_2d_format(data_format, None)
    #     self.op = nn.AdaptiveMeanPool2d(output_size=output_size)

    def __call__():
        raise NotImplementedError
    #     if self.data_format == 'NHWC':
    #         inputs = nhwc_to_nchw(inputs)
    #     output = self.op(inputs)
    #     if self.data_format == 'NHWC':
    #         output = nchw_to_nhwc(output)
    #     return output


class AdaptiveMeanPool3D(object):

    # def __init__(self, output_size, data_format):
        # self.data_format, _ = preprocess_3d_format(data_format, None)
        # self.op = torch.nn.AdaptiveAvgPool3d(output_size=output_size)

    def __call__():
        raise NotImplementedError
        # if self.data_format == 'NDHWC':
        #     inputs = nhwc_to_nchw(inputs)
        # output = self.op(inputs)
        # if self.data_format == 'NDHWC':
        #     output = nchw_to_nhwc(output)
        # return output


def adaptive_avg_pool1d(input, output_size):

    raise NotImplementedError


def adaptive_avg_pool2d(input, output_size):

    return nn.AdaptiveAvgPool2d(input, output_size)


def adaptive_avg_pool3d(input, output_size):

    return nn.AdaptiveAvgPool3d(input, output_size)


class AdaptiveMaxPool1D(object):

    # def __init__(self, output_size, data_format):
    #     self.data_format, _ = preprocess_1d_format(data_format, None)
    #     self.op = torch.nn.AdaptiveMaxPool1d(output_size=output_size)

    def __call__(self, input):
        raise NotImplementedError
        # if self.data_format == 'NLC':
        #     input = nhwc_to_nchw(input)
        # output = self.op(input)
        # if self.data_format == 'NLC':
        #     output = nchw_to_nhwc(output)
        # return output


class AdaptiveMaxPool2D(object):

    def __init__(self, output_size, data_format):
        self.data_format, _ = preprocess_2d_format(data_format, None)
        self.op = nn.AdaptiveMaxPool2d(output_size=output_size)

    def __call__(self, inputs):
        if self.data_format == 'NHWC':
            inputs = nhwc_to_nchw(inputs)
        output = self.op(inputs)
        if self.data_format == 'NHWC':
            output = nchw_to_nhwc(output)
        return output


class AdaptiveMaxPool3D(object):

    def __init__(self, output_size, data_format):
        self.data_format, _ = preprocess_3d_format(data_format, None)
        self.op = nn.AdaptiveMaxPool3d(output_size=output_size)
    def __call__(self, inputs):
        if self.data_format == 'NDHWC':
            inputs = nhwc_to_nchw(inputs)
        output = self.op(inputs)
        if self.data_format == 'NDHWC':
            output = nchw_to_nhwc(output)
        return output

def adaptive_max_pool1d(input, output_size, return_indices = False):
    raise NotImplementedError
    
def adaptive_max_pool2d(input, output_size, return_indices = False):

    return nn.AdaptiveMaxPool2d(input, output_size, return_indices)

def adaptive_max_pool3d(input, output_size, return_indices=False):

    return nn.AdaptiveMaxPool3d(input, output_size, return_indices)


class BinaryConv2D(object):

    def __init__(self, strides, padding, data_format, dilations, out_channel, k_size, in_channel):
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.strides = strides
        self.dilations = dilations

    def quantize(self, x):
        raise NotImplementedError

    def __call__(self, inputs, filters):
        raise NotImplementedError


class DorefaConv2D(object):

    def __init__(self, bitW, bitA, strides, padding, data_format, dilations, out_channel, k_size, in_channel):
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.strides = strides
        self.dilations = dilations
        self.bitW = bitW
        self.bitA = bitA

    def _quantize_dorefa(self, x, k):
        raise NotImplementedError

    def cabs(self, x):
        raise NotImplementedError

    def quantize_active(self, x, bitA):
        raise NotImplementedError

    def quantize_weight(self, x, bitW, force_quantization=False):
        raise NotImplementedError

    def __call__(self, inputs, filters):
        raise NotImplementedError


class rnncell(object):

    def __init__(self,  input_size , hidden_size , bias = True, nonlinearity='tanh'):
        self.input_size = input_size
        self.hidden_size= hidden_size
        self.bias = bias
        self.act = nonlinearity

    def __call__(self, input, h):
        if self.act == 'tanh':
            h = nn.RNNCell(
                input,
                h,
                bias=self.bias,
                nonlinearity='tanh'
            )
        else:
            h = nn.RNNCell(
                input,
                h,
                bias=self.bias,
                nonlinearity='relu'
            )
        return h, h


class lstmcell(object):

    def __init__(self,  input_size , hidden_size , bias = True, nonlinearity='tanh'):
        self.input_size = input_size
        self.hidden_size= hidden_size
        self.bias = bias

    def __call__(self, input, h, c):
        h = (h, c)
        h, c = nn.LSTMCell(
                input,
                h,
                bias=self.bias
                )
        return h, h, c


class grucell(object):

    def __init__(self,  input_size , hidden_size , bias = True, nonlinearity='tanh'):
        self.input_size = input_size
        self.hidden_size= hidden_size
        self.bias = bias

    def __call__(self, input, h):
        h = nn.GRUCell(
                input,
                h,
                bias=self.bias
        )
        return h, h


class rnnbase(Module):

    def __init__(
        self,
            mode:str,  
            input_size:int,
            hidden_size:int,  
            num_layers:int= 1,
            bias:bool=True,
            batch_first:bool=False ,  
            dropout: float= 0,  
            bidirectional:bool=False,  
            proj_size : int = 0 ,  
            nonlinearity: str = None
    ):
        super(rnnbase, self).__init__()
        self.mode = mode 
        self.input_size = input_size 
        self.hidden_size = hidden_size 
        self.num_layers = num_layers 
        self.bias = bias 
        self.batch_first = batch_first 
        self.dropout = dropout 
        self.bidirectional = bidirectional 
        self.proj_size = proj_size 
        self.nonlinearity = nonlinearity

        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        elif mode == 'GRU':
            gate_size = 3 * hidden_size
        elif mode == 'RNN':
            gate_size = hidden_size
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)

        num_directions = 1 + bidirectional
        k = math.sqrt(1 / hidden_size)

        def build_unit(name, in_channels, out_channels=None):
            if out_channels is not None:
                shape = (in_channels, out_channels)
            else:
                shape = (in_channels,)
            setattr(self, name, init.uniform(shape, 'float32', -k, k))
            if self.bidirectional:
                setattr(self, name + '_reverse', init.uniform(shape, 'float32', -k, k))

        for layer in range(num_layers):
            if layer == 0:
                build_unit(f'weight_ih_l{layer}', gate_size, input_size)
            else:
                if proj_size > 0:
                    build_unit(f'weight_ih_l{layer}', gate_size, num_directions * proj_size)
                else:
                    build_unit(f'weight_ih_l{layer}', gate_size, num_directions * hidden_size)

            if proj_size > 0:
                build_unit(f'weight_hh_l{layer}', gate_size, proj_size)
                build_unit(f'weight_hr_l{layer}', proj_size, hidden_size)
            else:
                build_unit(f'weight_hh_l{layer}', gate_size, hidden_size)

            if bias:
                build_unit(f'bias_ih_l{layer}', gate_size)
                build_unit(f'bias_hh_l{layer}', gate_size)

    def _cudnn_flatten_weights(self, cudnn_mode):
        def copy_to_flatten_weight(param_name, offset_idx, num_gates):
            def copy_to(param_name, offset_idx, idx):
                cur_offset = self._cudnn_weight_offset[offset_idx]
                param = getattr(self, param_name)
                param = param[self.hidden_size * idx: self.hidden_size * (idx + 1)]
                ft_weight[cur_offset:cur_offset + param.numel()] = param.flatten()
                
            if self.bias:
                for idx in range(num_gates):
                    copy_to('weight' + param_name, offset_idx + idx * 2, idx)
                    copy_to('bias' + param_name, offset_idx + idx * 2 + 1, idx)
                return num_gates * 2
            else:
                for idx in range(num_gates):
                    copy_to('weight' + param_name, offset_idx + idx, idx)
                return num_gates

        if jt.flags.use_cuda and jt.cudnn and jt.compiler.is_cuda:
            if getattr(self, '_cudnn_weight_size', None) is None:                
                offset_array = jt.cudnn.cudnn_rnn_weight_offset(
                    cudnn_mode,
                    self.input_size,
                    self.hidden_size, 
                    self.num_layers,
                    self.proj_size,
                    self.bias,
                    self.bidirectional
                )
                self._cudnn_weight_size = offset_array[0]
                self._cudnn_weight_offset = offset_array[1:]
            
            num_gates = {
                "RNN": 1, "LSTM": 4, "GRU": 3
            }[self.mode]
            ft_weight = jt.zeros(self._cudnn_weight_size, dtype=jt.float32)

            cnt = 0
            for layer in range(self.num_layers):
                suffix = ''
                cnt += copy_to_flatten_weight(f'_ih_l{layer}' + suffix, cnt, num_gates)
                cnt += copy_to_flatten_weight(f'_hh_l{layer}' + suffix, cnt, num_gates)
                if self.bidirectional:
                    suffix = '_reverse'
                    cnt += copy_to_flatten_weight(f'_ih_l{layer}' + suffix, cnt, num_gates)
                    cnt += copy_to_flatten_weight(f'_hh_l{layer}' + suffix, cnt, num_gates)
            return ft_weight
        else:
            raise RuntimeError("Not Cudnn found")

    @abstractmethod
    def call_rnn_cell(self, input, hidden, suffix):
        pass

    def call_rnn_sequence(self, input, hidden, suffix):
        if 'reverse' in suffix:
            input = input[::-1]

        output = []
        for s in range(input.shape[0]):
            out, hidden = self.call_rnn_cell(input[s], hidden, suffix)
            output.append(out)

        if 'reverse' in suffix:
            output = output[::-1]
        output = jt.stack(output, dim=0)

        return output, hidden

    def _execute_cudnn_rnn(self, input, hx):
        cudnn_mode = {
            ('RNN', 'tanh'): 'tanh',
            ('RNN', 'relu'): 'relu',
            ('LSTM', None): 'lstm',
            ('GRU', None): 'gru'
        }[(self.mode, self.nonlinearity)]
        ft_weight = self._cudnn_flatten_weights(cudnn_mode)

        if self.mode == 'LSTM':
            ret = jt.cudnn.ops.cudnn_rnn(input, hx[0], hx[1], ft_weight,
                cudnn_mode, self.input_size, self.hidden_size, self.num_layers, 0,
                self.dropout, self.bias, self.bidirectional, self.is_training()
            )
            return ret[0], (ret[1], ret[2])
        else:
            ret = jt.cudnn.ops.cudnn_rnn(input, hx, ft_weight,
                cudnn_mode, self.input_size, self.hidden_size, self.num_layers, 0,
                self.dropout, self.bias, self.bidirectional, self.is_training()
            )
            return ret[0], ret[1]

    def execute(self, input, hx=None):
        if self.batch_first:
            input = input.permute(1, 0, 2)

        num_directions = 2 if self.bidirectional else 1

        if hx is None:
            if self.mode in ['RNN', 'GRU']:
                hx = jt.zeros((num_directions * self.num_layers, input.shape[1], self.hidden_size), dtype=input.dtype)
            elif self.mode == 'LSTM':
                hx = (jt.zeros((num_directions * self.num_layers, input.shape[1], self.hidden_size), dtype=input.dtype),
                      jt.zeros((num_directions * self.num_layers, input.shape[1], self.hidden_size), dtype=input.dtype))

        if jt.flags.use_cuda and jt.cudnn and self.proj_size == 0 and jt.compiler.is_cuda:
            return self._execute_cudnn_rnn(input, hx)
        else:
            hidden_n = []

            for l in range(self.num_layers):
                output = []

                if isinstance(hx, tuple):
                    hidden = [h[l * num_directions] for h in hx]
                else:
                    hidden = hx[l * num_directions]

                output, _hidden = self.call_rnn_sequence(input, hidden, f'l{l}')
                hidden_n.append(_hidden)

                if self.bidirectional:
                    if isinstance(hx, tuple):
                        hidden = [h[l * num_directions + 1] for h in hx]
                    else:
                        hidden = hx[l * num_directions + 1]

                    output_b, _hidden = self.call_rnn_sequence(input, hidden, f'l{l}_reverse')
                    output = jt.concat([output, output_b], dim=-1)
                    hidden_n.append(_hidden)

                if self.dropout > 0:
                    input = dropout(output, p=self.dropout)
                else:
                    input = output

            if isinstance(hx, tuple):
                hidden_n = tuple(jt.stack(hn, dim=0) for hn in zip(*hidden_n))
            else:
                hidden_n = jt.stack(hidden_n, dim=0)

            return output, hidden_n



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

    def __call__(self, input):
        return nn.layer_norm(input, self.normalized_shape, self.gamma, self.beta, self.eps)


class multiheadattention(Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        assert dropout==0, "TODO: dropout>0"

        self.head_dim = embed_dim // num_heads
        assert (self.head_dim * num_heads == self.embed_dim), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, ("Self-attention requires query, key and " "value to be of the same size")

        #TODO: quant_noise
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        assert not add_bias_kv, "TODO: add_bias_kv=True"
        self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def reset_parameters(self):
        '''
        初始化参数

            代码示例:
                >>> multihead_attn = jt.attention.MultiheadAttention(embed_dim, num_heads)
                >>> multihead_attn.reset_parameters()
                
        
        '''
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            init.xavier_uniform_(self.k_proj.weight)
            init.xavier_uniform_(self.v_proj.weight)
            init.xavier_uniform_(self.q_proj.weight)

        # init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            init.xavier_normal_(self.bias_v)



    def execute(
        self,
        query,
        key = None,
        value = None,
        key_padding_mask = None,
        incremental_state = None,
        need_weights = True,
        static_kv = False,
        attn_mask = None,
        before_softmax = False,
        need_head_weights = False,
    ):
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.shape
        assert embed_dim == self.embed_dim
        assert list(query.shape) == [tgt_len, bsz, embed_dim]

        assert incremental_state is None, "TODO: incremental_state is not None"
        saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q = q*self.scaling

        assert self.bias_k is None, "TODO: self.bias_k is not None:"

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)
        if k is not None:
            k = k.view(-1, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)
        if v is not None:
            v = v.view(-1, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)

        assert saved_state is None, "TODO: saved_state is not None"
        assert k is not None
        src_len = k.shape[1]

        assert key_padding_mask is None, "TODO: key_padding_mask is not None"
        assert not self.add_zero_attn, "TODO: self.add_zero_attn=True"

        attn_weights = nn.bmm(q, k.transpose(0, 2, 1))

        assert list(attn_weights.shape) == [bsz * self.num_heads, tgt_len, src_len]

        assert attn_mask is None, "TODO: attn_mask is not None"
        assert key_padding_mask is None, "TODO: key_padding_mask is not None"
        
        if before_softmax:
            return attn_weights, v
        
        attn_weights_float = nn.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)

        assert v is not None
        attn = nn.bmm(attn_weights, v)
        assert list(attn.shape) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.shape[1] == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(1, 0, 2).view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights = None
        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0, 2, 3)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dims=[0])

        return attn, attn_weights

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

        self.data_format = data_format

    def __call__(self, input, weight):
        if self.data_format == 'channels_last' :
            input = nhwc_to_nchw(input)
        output = nn.PReLU(input, weight)
        if self.data_format == 'channels_last':
            output = nchw_to_nhwc(output)
        return output


def prelu(input, weight, data_format):
    if data_format == 'channels_last':
        input = nhwc_to_nchw(input)
    output = nn.PReLU(input, weight)
    if data_format == 'channels_last':
        output = nchw_to_nhwc(output)
    return output

def hardsigmoid(input):

    return NotImplementedError

def hardswish(input):

    return NotImplementedError

def swish(input):

    return NotImplementedError

def linear(input, weight, bias = None):

    return nn.linear(input, weight, bias)

def unfold(input, kernel_size, dilation = 1, padding = 0, stride = 1):

    return nn.unfold(input, kernel_size, stride=stride, padding=padding, dilation=dilation)

