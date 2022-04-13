#! /usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch import _VF
from torch.nn import Module


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

    # if len(x.shape) == 3:
    #     x = torch.transpose(x, 1, 2)
    # elif len(x.shape) == 4:
    #     x = torch.transpose(x, 1, 2)
    #     x = torch.transpose(x, 2, 3)
    # elif len(x.shape) == 5:
    #     x = torch.transpose(x, 1, 2)
    #     x = torch.transpose(x, 2, 3)
    #     x = torch.transpose(x, 3, 4)
    # else:
    #     raise Exception("Unsupported dimensions")
    # return x
    return torch.moveaxis(x, 1, -1)


def nhwc_to_nchw(x):
    """ # TODO permute   x.contiguous
    Channles last to channels first

    Parameters
    ----------
    x : tensor
        channels last tensor data

    Returns
    -------
        channels first tensor data
    """

    # if len(x.shape) == 3:
    #     x = torch.transpose(x, 1, 2)
    # elif len(x.shape) == 4:
    #     x = torch.transpose(x, 2, 3)
    #     x = torch.transpose(x, 1, 2)
    # elif len(x.shape) == 5:
    #     x = torch.transpose(x, 3, 4)
    #     x = torch.transpose(x, 2, 3)
    #     x = torch.transpose(x, 1, 2)
    # else:
    #     raise Exception("Unsupported dimensions")
    return torch.moveaxis(x, -1, 1)


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

    def __init__(self, negative_slope=0.01):
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

    return F.leaky_relu(x, negative_slope=negative_slope)


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
        return torch.sigmoid(x)


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

    def __init__(self, axis=None):
        self.axis = axis

    def __call__(self, x):
        return F.softmax(x, dim=self.axis)


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

    return F.softmax(logits, axis)


class GeLU(object):

    def __init__(self, approximate=False):
        self.approximate = approximate

    def __call__(self, x):
        return F.gelu(x)


def gelu(x, approximate=False):

    return F.gelu(x)


class Dropout(object):

    def __init__(self, p, seed=0):
        self.p = p
        self.seed = seed

    def __call__(self, inputs):
        return F.dropout(inputs, p=self.p)


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
        outputs = torch.add(x, bias)
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

    def __call__(self, input, filters):
        if self.data_format == 'NLC':
            input = nhwc_to_nchw(input)
        if self.padding == 'same':
            out = self.conv1d_same_padding(input, filters)
        else:
            out = F.conv1d(input, filters, stride=self.stride, padding=self.padding,
                           dilation=self.dilations, groups=self.groups)
        if self.data_format == 'NLC':
            out = nchw_to_nhwc(out)

        return out

    def conv1d_same_padding(self, input, filters):
        rows_odd, padding_rows = same_padding(input, filters, self.stride, 1)
        if rows_odd:
            input = F.pad(input, [0, int(rows_odd)], 'replicate')
        return F.conv1d(input, filters, stride=self.stride, padding=(padding_rows // 2), groups=self.groups)



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
    if isinstance(weight, torch.Tensor):
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
        self.strides = (strides[1], strides[2])
        self.dilations = (dilations[1], dilations[2])
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.groups = groups

    def __call__(self, input, filters):
        if self.data_format == 'NHWC':
            input = nhwc_to_nchw(input)

        if self.padding == 'same':
            output = self.conv2d_same_padding(input, filters)
        else:
            output = F.conv2d(input, filters, stride=self.strides, padding=self.padding,
                              dilation=self.dilations, groups=self.groups)

        if self.data_format == 'NHWC':
            output = nchw_to_nhwc(output)
        return output

    def conv2d_same_padding(self, input, weight, bias=None):
        rows_odd, cols_odd, padding_rows, padding_cols = same_padding(input, weight, self.strides, self.dilations)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

        return F.conv2d(
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

    output = F.conv2d(input, filters, stride=strides, padding=padding, dilation=dilations)

    if data_format == 'NHWC':
        output = nchw_to_nhwc(output)
    return output


class Conv3D(object):

    def __init__(self, strides, padding, data_format='NDHWC', dilations=None, out_channel=None, k_size=None):
        if data_format is 'NDHWC':
            self._strides = (strides[1], strides[2], strides[3])
            self._dilations = (dilations[1], dilations[2], dilations[3])
        elif data_format is 'NCDHW':
            self._strides = (strides[2], strides[3], strides[4])
            self._dilations = (dilations[2], dilations[3], dilations[4])
        self.data_format, self.padding = preprocess_3d_format(data_format, padding)

    def __call__(self, input, filters):
        if self.data_format == 'NDHWC':
            input = nhwc_to_nchw(input)

        if self.padding == 'same':
            out = self.conv3d_same_padding(input, weight=filters)
        else:
            out = F.conv3d(input, weight=filters, stride=self._strides, padding=self.padding, dilation=self._dilations)

        if self.data_format == 'NDHWC':
            out = nchw_to_nhwc(out)

        return out

    def conv3d_same_padding(self, input, weight, bias=None, groups=1):
        rows_odd, cols_odd, depth_odd, padding_rows, padding_cols, padding_depth = same_padding(input, weight,
                                                                                                self._strides, self._dilations)
        if rows_odd or cols_odd or depth_odd:
            input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd), 0, int(depth_odd)])

        return F.conv3d(
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

    lrn_obj = torch.nn.LocalResponseNorm(size=depth_radius, alpha=alpha, beta=beta, k=bias)
    return lrn_obj(inputs)


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

    def __init__(self, ksize, strides, padding, data_format=None):
        self.data_format, self.padding = preprocess_1d_format(data_format=data_format, padding=padding)
        self.max_pool1d = MaxPool(ksize, strides, padding, data_format)

    def __call__(self, inputs):
        return self.max_pool1d(inputs)


class MaxPool(object):

    def __init__(self, ksize, strides, padding, data_format=None):
        self.ksize = ksize
        self.strides = strides
        if data_format in ['channels_last', 'NLC', 'NWC', 'NHWC', 'NDHWC']:
            self.data_format = 'channels_last'
        elif data_format in ['channels_first', 'NCL', 'NCW', 'NCHW', 'NCDHW']:
            self.data_format = 'channels_first'
        self.padding = padding

    def __call__(self, inputs):
        if self.data_format == 'channels_last':
            inputs = nhwc_to_nchw(inputs)
        if len(inputs.shape) == 2 or len(inputs.shape) == 3:
            if self.padding in ['SAME', 'same']:
                out = self.maxpool1d_same_padding(inputs)
            else:
                out = F.max_pool1d(inputs, self.ksize, self.strides)
        if len(inputs.shape) == 4:
            if self.padding in ['SAME', 'same']:
                out = self.maxpool2d_same_padding(inputs)
            else:
                out = F.max_pool2d(inputs, self.ksize, (self.strides[1], self.strides[2]))
        if len(inputs.shape) == 5:
            if self.padding in ['SAME', 'same']:
                out = self.maxpool3d_same_padding(inputs)
            else:
                out = F.max_pool3d(inputs, self.ksize, (self.strides[1], self.strides[2], self.strides[3]))

        if self.data_format == 'channels_last':
            return nchw_to_nhwc(out)
        else:
            return out

    def maxpool1d_same_padding(self, input):
        rows_odd, padding_rows = same_padding(input, self.ksize, self.strides, 1)
        if rows_odd:
            input = F.pad(input, [0, int(rows_odd)], 'constant', float('-inf'))
        return F.max_pool1d(input, self.ksize, self.strides, padding=(padding_rows // 2))

    def maxpool2d_same_padding(self, input):
        strides = [self.strides[1], self.strides[2]]
        rows_odd, cols_odd, padding_rows, padding_cols = same_padding(input, self.ksize, strides, (1, 1))
        if rows_odd or cols_odd:
            # TODO The fill value for maxpool is -INF.
            input = F.pad(input, [0, int(rows_odd), 0, int(cols_odd)], 'constant', float('-inf'))

        return F.max_pool2d(input, self.ksize, strides, padding=(padding_rows // 2, padding_cols // 2))

    def maxpool3d_same_padding(self, input):
        strides = [self.strides[1], self.strides[2], self.strides[3]]
        rows_odd, cols_odd, depth_odd, padding_rows, padding_cols, padding_depth = same_padding(
            input, self.ksize, strides, (1, 1, 1)
        )
        if rows_odd or cols_odd or depth_odd:
            input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd), 0, int(depth_odd)], 'constant', float('-inf'))
        return F.max_pool3d(
                input, self.ksize, strides, padding=(padding_rows // 2, padding_cols // 2, padding_depth // 2)
        )


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

    maxpool_obj = MaxPool(ksize, strides, padding, data_format)
    return maxpool_obj(input)


class AvgPool1d(object):

    def __init__(self, ksize, strides, padding, data_format=None):
        self.data_format, self.padding = preprocess_1d_format(data_format=data_format, padding=padding)
        self.avg_poo1d = AvgPool(ksize, strides, padding, data_format)

    def __call__(self, inputs):
        return self.avg_poo1d(inputs)


class AvgPool(object):

    def __init__(self, ksize, strides, padding, data_format=None):
        self.ksize = ksize
        self.strides = strides
        if data_format in ['channels_last', 'NLC', 'NWC', 'NHWC', 'NDHWC']:
            self.data_format = 'channels_last'
        elif data_format in ['channels_first', 'NCL', 'NCW', 'NCHW', 'NCDHW']:
            self.data_format = 'channels_first'
        self.padding = padding

    def __call__(self, inputs):
        if self.data_format == 'channels_last':
            inputs = nhwc_to_nchw(inputs)
        if len(inputs.shape) == 2 or len(inputs.shape) == 3:
            if self.padding in ['SAME', 'same']:
                out = self.avgpool1d_same_padding(inputs)
            else:
                out = F.avg_pool1d(inputs, self.ksize, self.strides)
        if len(inputs.shape) == 4:
            if self.padding in ['SAME', 'same']:
                out = self.avgpool2d_same_padding(inputs)
            else:
                out = F.avg_pool2d(inputs, self.ksize, (self.strides[1], self.strides[2]))
        if len(inputs.shape) == 5:
            if self.padding in ['SAME', 'same']:
                out = self.avgpool3d_same_padding(inputs)
            else:
                out = F.avg_pool3d(inputs, self.ksize, (self.strides[1], self.strides[2], self.strides[3]))

        if self.data_format == 'channels_last':
            return nchw_to_nhwc(out)
        else:
            return out

    def avgpool1d_same_padding(self, input):
        rows_odd, padding_rows = same_padding(input, self.ksize, self.strides, 1)
        if rows_odd:
            input = F.pad(input, [0, int(rows_odd)], 'replicate')
        return F.avg_pool1d(input, self.ksize, self.strides, padding=(padding_rows // 2))

    def avgpool2d_same_padding(self, input):
        strides = [self.strides[1], self.strides[2]]
        rows_odd, cols_odd, padding_rows, padding_cols = same_padding(input, self.ksize, strides, (1, 1))
        if rows_odd or cols_odd:
            # TODO The fill value for maxpool is -INF.
            input = F.pad(input, [0, int(rows_odd), 0, int(cols_odd)], mode='replicate')

        return F.avg_pool2d(input, self.ksize, strides, padding=(padding_rows // 2, padding_cols // 2))

    def avgpool3d_same_padding(self, input):
        strides = [self.strides[1], self.strides[2], self.strides[3]]
        rows_odd, cols_odd, depth_odd, padding_rows, padding_cols, padding_depth = same_padding(
            input, self.ksize, strides, (1, 1, 1)
        )
        if rows_odd or cols_odd or depth_odd:
            input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd), 0, int(depth_odd)], mode='replicate')
        return F.avg_pool3d(
                input, self.ksize, strides, padding=(padding_rows // 2, padding_cols // 2, padding_depth // 2)
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


class MaxPool3d(object):

    def __init__(self, ksize, strides, padding, data_format=None):
        self.data_format, self.padding = preprocess_3d_format(data_format, padding)
        self.max_pool3d = MaxPool(ksize, strides, padding, data_format)

    def __call__(self, inputs):
        return self.max_pool3d(inputs)


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
    max_pool3d_obj = MaxPool(ksize, strides, padding, data_format)
    return max_pool3d_obj(input)


class AvgPool3d(object):

    def __init__(self, ksize, strides, padding, data_format=None):
        self.data_format, self.padding = preprocess_3d_format(data_format, padding)
        self.avg_pool3d_obj = AvgPool(ksize, strides, self.padding, self.data_format)

    def __call__(self, inputs):
        return self.avg_pool3d_obj(inputs)


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

    avg_pool_obj = AvgPool(ksize, strides, padding, data_format)
    return avg_pool_obj(input)


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

    def __init__(self, strides, padding, data_format=None, dilations=None, ksize=None, channel_multiplier=1):
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        if self.data_format == 'NHWC':
            self._stride = (strides[1], strides[2])
        if self.data_format == 'NCHW':
            self._stride = (strides[2], strides[3])
        self.dilations = dilations

    def __call__(self, input, filter, point_filter=None):
        if self.data_format == 'NHWC':
            input = nhwc_to_nchw(input)
        channel = input.shape[1]

        depthwise_conv = F.conv2d(input, filter, bias=None, stride=self._stride, padding=self.padding,
                                  dilation=self.dilations, groups=channel)
        pointwise_conv = F.conv2d(depthwise_conv, point_filter, padding=self.padding)

        if self.data_format == 'NHWC':
            pointwise_conv = nchw_to_nhwc(pointwise_conv)
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

    if isinstance(weight, torch.Tensor):
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
        out_cols = input_rows * strides[1] - strides[1] + 1

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
        out_cols = input_rows * strides[1] - strides[1] + 1
        out_depth = input_rows * strides[2] - strides[2] + 1

        padding_rows = max(0, (input_rows - 1) * strides[0] + (filter_rows - 1) * dilations[0] + 1 - out_rows)
        padding_cols = max(0, (input_cols - 1) * strides[1] + (filter_cols - 1) * dilations[1] + 1 - out_cols)
        padding_depth = max(0, (input_depth - 1) * strides[2] + (filter_depth - 1) * dilations[2] + 1 - out_depth)

        rows_odd = (padding_rows % 2 != 0)
        cols_odd = (padding_cols % 2 != 0)
        depth_odd = (padding_depth % 2 != 0)
        return rows_odd, cols_odd, depth_odd, padding_rows, padding_cols, padding_depth


class Conv1d_transpose(object):

    def __init__(
        self, stride, padding, data_format='NWC', dilations=None, out_channel=None, k_size=None, in_channels=None
    ):
        self.stride = stride
        self.dilations = dilations
        self.data_format, self.padding = preprocess_1d_format(data_format, padding)

    def __call__(self, input, filters):
        if self.data_format == 'NLC':
            input = nhwc_to_nchw(input)
        if self.padding == 'same':
            out = self.conv1d_transpose_same_padding(input, filters)
        else:
            out = F.conv_transpose1d(
                input,
                weight=filters,
                padding=0,
                stride=self.stride,
                dilation=self.dilations
            )
        if self.data_format == 'NLC':
            out = nchw_to_nhwc(out)
        return out

    def conv1d_transpose_same_padding(self, input, filters):
        rows_odd, padding_rows = same_padding_deconvolution(input, filters, self.stride, 1)
        if rows_odd:
            input = F.pad(input, [0, int(rows_odd)])
            out_padding = 0
        else:
            out_padding = 1
        return F.conv_transpose1d(input, weight=filters, padding=(padding_rows // 2), stride=self.stride,
                                  dilation=self.dilations, output_padding=out_padding)



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

    conv1d_transpose_obj = Conv1d_transpose(strides, padding, data_format, dilations)
    return conv1d_transpose_obj(input, filters)


class Conv2d_transpose(object):

    def __init__(
        self, strides, padding, data_format='NHWC', dilations=None, name=None, out_channel=None, k_size=None,
        in_channels=None
    ):
        self.strides = strides
        self.dilations = dilations
        self.name = name
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)

    def __call__(self, input, filters):
        if self.data_format == 'NHWC':
            input = nhwc_to_nchw(input)
        if self.padding == 'same':
            out = self.conv2d_transpore_same(input, filters)
        else:
            out = F.conv_transpose2d(
                input,
                weight=filters,
                padding=0,
                stride=self.strides,
                dilation=self.dilations
            )
        if self.data_format == 'NHWC':
            out = nchw_to_nhwc(out)
        return out

    def conv2d_transpore_same(self,input, filters):
        rows_odd, cols_odd, padding_rows, padding_cols = same_padding_deconvolution(input, filters, self.strides, (1, 1))
        if rows_odd or cols_odd:
            input = F.pad(input, [0, int(rows_odd), 0, int(cols_odd)])
            out_padding = 0
        else:
            out_padding = 1
        out = F.conv_transpose2d(input, weight=filters, padding=(padding_rows // 2, padding_cols // 2), stride=self.strides,
                                 dilation=self.dilations, output_padding=out_padding)
        return out


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

    conv2d_transpose_obj = Conv2d_transpose(strides, padding, data_format, dilations)
    return conv2d_transpose_obj(input, filters)


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
            out = F.conv_transpose3d(
                input,
                weight=filters,
                padding=0,
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
            input = F.pad(input, [0, int(rows_odd), 0, int(cols_odd), 0, int(depth_odd)])
            out_padding = 0
        else:
            out_padding = 1
        out = F.conv_transpose3d(input, weight=filters, padding=(padding_rows // 2, padding_cols // 2, padding_depth // 2),
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
    """Reshape [c] to [c, 1, 1]."""
    raise NotImplementedError


def _bias_scale(x, b, data_format):
    """The multiplication counter part of tf.nn.bias_add."""
    raise NotImplementedError


def _bias_add(x, b, data_format):
    """Alternative implementation of tf.nn.bias_add which is compatiable with tensorRT."""
    raise NotImplementedError


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

        out = torch.nn.functional.batch_norm(inputs,
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

    def __init__(self, output_size, data_format):
        self.data_format, _ = preprocess_1d_format(data_format, None)
        self.op = torch.nn.AdaptiveAvgPool1d(output_size=output_size)

    def __call__(self, input):
        if self.data_format == 'NLC':
            input = nhwc_to_nchw(input)
        output = self.op(input)
        if self.data_format == 'NLC':
            output = nchw_to_nhwc(output)
        return output


class AdaptiveMeanPool2D(object):

    def __init__(self, output_size, data_format):
        self.data_format, _ = preprocess_2d_format(data_format, None)
        self.op = torch.nn.AdaptiveAvgPool2d(output_size=output_size)

    def __call__(self, inputs):
        if self.data_format == 'NHWC':
            inputs = nhwc_to_nchw(inputs)
        output = self.op(inputs)
        if self.data_format == 'NHWC':
            output = nchw_to_nhwc(output)
        return output


class AdaptiveMeanPool3D(object):

    def __init__(self, output_size, data_format):
        self.data_format, _ = preprocess_3d_format(data_format, None)
        self.op = torch.nn.AdaptiveAvgPool3d(output_size=output_size)

    def __call__(self, inputs):
        if self.data_format == 'NDHWC':
            inputs = nhwc_to_nchw(inputs)
        output = self.op(inputs)
        if self.data_format == 'NDHWC':
            output = nchw_to_nhwc(output)
        return output


class AdaptiveMaxPool1D(object):

    def __init__(self, output_size, data_format):
        self.data_format, _ = preprocess_1d_format(data_format, None)
        self.op = torch.nn.AdaptiveMaxPool1d(output_size=output_size)

    def __call__(self, input):
        if self.data_format == 'NLC':
            input = nhwc_to_nchw(input)
        output = self.op(input)
        if self.data_format == 'NLC':
            output = nchw_to_nhwc(output)
        return output


class AdaptiveMaxPool2D(object):

    def __init__(self, output_size, data_format):
        self.data_format, _ = preprocess_2d_format(data_format, None)
        self.op = torch.nn.AdaptiveMaxPool2d(output_size=output_size)

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
        self.op = torch.nn.AdaptiveMaxPool3d(output_size=output_size)
    def __call__(self, inputs):
        if self.data_format == 'NDHWC':
            inputs = nhwc_to_nchw(inputs)
        output = self.op(inputs)
        if self.data_format == 'NDHWC':
            output = nchw_to_nhwc(output)
        return output



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

    def __init__(self, weight_ih, weight_hh, bias_ih, bias_hh, act):
        self.weight_ih = weight_ih
        self.weight_hh = weight_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self.act = act

    def __call__(self, input, h):
        if self.act == 'tanh':
            h = _VF.rnn_tanh_cell(
                input,
                h,
                self.weight_ih,
                self.weight_hh,
                self.bias_ih,
                self.bias_hh,
            )
        else:
            h = _VF.rnn_relu_cell(
                input,
                h,
                self.weight_ih,
                self.weight_hh,
                self.bias_ih,
                self.bias_hh,
            )
        return h, h


class lstmcell(object):

    def __init__(self, weight_ih, weight_hh, bias_ih, bias_hh, act=None):
        self.weight_ih = weight_ih
        self.weight_hh = weight_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh

    def __call__(self, input, h, c):
        h = (h, c)
        h, c = _VF.lstm_cell(
            input,
            h,
            self.weight_ih,
            self.weight_hh,
            self.bias_ih,
            self.bias_hh,
        )
        return h, h, c


class grucell(object):

    def __init__(self, weight_ih, weight_hh, bias_ih, bias_hh, act=None):
        self.weight_ih = weight_ih
        self.weight_hh = weight_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh

    def __call__(self, input, h):
        h = _VF.gru_cell(
            input,
            h,
            self.weight_ih,
            self.weight_hh,
            self.bias_ih,
            self.bias_hh,
        )
        return h, h


class rnnbase(Module):

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
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.rnn_impls = {
            'RNN_TANH': _VF.rnn_tanh,
            'RNN_RELU': _VF.rnn_relu,
            'GRU': _VF.gru,
        }
        self.w_ih = w_ih
        self.w_hh = w_hh
        self.b_ih = b_ih
        self.b_hh = b_hh

        # stdv = 1.0 / np.sqrt(self.hidden_size)
        # _init = tf.random_uniform_initializer(minval=-stdv, maxval=stdv)
        self.proj_size = 0
        self.act_fn = None
        self._flat_weights_names = []
        self._all_weights = []
        cur = 0
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                if bias:
                    layer_params = (w_ih[cur], w_hh[cur], b_ih[cur], b_hh[cur])
                else:
                    layer_params = (w_ih[cur], w_hh[cur])

                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._flat_weights_names.extend(param_names)
                self._all_weights.append(param_names)
                cur += 1
        self._flat_weights = [
            (lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in self._flat_weights_names
        ]
        self.flatten_parameters()

    def flatten_parameters(self):
        if len(self._flat_weights) != len(self._flat_weights_names):
            return

        for w in self._flat_weights:
            if not isinstance(w, torch.Tensor):
                return
        first_fw = self._flat_weights[0]
        dtype = first_fw.dtype
        for fw in self._flat_weights:
            if (not isinstance(fw.data, torch.Tensor) or not (fw.data.dtype == dtype) or not fw.data.is_cuda or
                    not torch.backends.cudnn.is_acceptable(fw.data)):
                return
        unique_data_ptrs = set(p.data_ptr() for p in self._flat_weights)
        if len(unique_data_ptrs) != len(self._flat_weights):
            return

        with torch.cuda.device_of(first_fw):
            import torch.backends.cudnn.rnn as rnn
            with torch.no_grad():
                if torch._use_cudnn_rnn_flatten_weight():
                    num_weights = 4 if self.bias else 2
                    if self.proj_size > 0:
                        num_weights += 1
                    torch._cudnn_rnn_flatten_weight(
                        self._flat_weights, num_weights, self.input_size, rnn.get_cudnn_mode(self.mode),
                        self.hidden_size, self.proj_size, self.num_layers, self.batch_first, bool(self.bidirectional)
                    )

    def _apply(self, fn):
        ret = super(rnnbase, self)._apply(fn)
        self._flat_weights = [
            (lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in self._flat_weights_names
        ]
        self.flatten_parameters()
        return ret

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

    def forward(self, input, states):
        batch_size = input.shape[0] if self.batch_first else input.shape[1]
        input_shape = input.shape
        self.check_input(input_shape)
        if self.mode == 'LSTM':
            if states is None:
                h = torch.zeros(
                    self.num_layers * self.num_directions, batch_size, self.hidden_size, dtype=input.dtype,
                    device=input.device
                )
                c = torch.zeros(
                    self.num_layers * self.num_directions, batch_size, self.hidden_size, dtype=input.dtype,
                    device=input.device
                )
                states = (h, c)
            else:
                h, c = states
                self.check_hidden(h, batch_size)
                self.check_hidden(c, batch_size)
            result = _VF.lstm(
                input, states, self._flat_weights, self.bias, self.num_layers, self.dropout, self.training,
                self.bidirectional, self.batch_first
            )
            return result[0], result[1:]
        else:
            if states is None:
                h = torch.zeros(
                    self.num_layers * self.num_directions, batch_size, self.hidden_size, dtype=input.dtype,
                    device=input.device
                )
                states = h
            else:
                self.check_hidden(states, batch_size)
            impl = self.rnn_impls[self.mode]
            result = impl(
                input, states, self._flat_weights, self.bias, self.num_layers, self.dropout, self.training,
                self.bidirectional, self.batch_first
            )
            return result[0], result[1]


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
        return F.layer_norm(input, self.normalized_shape, self.gamma, self.beta, self.eps)


class multiheadattention(Module):

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
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim_check, 'embed_dim must be divisible by num_heads'
        self.register_parameter('in_proj_weight', None)

        if q_bias is not None:
            self.in_proj_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        else:
            self.register_parameter('in_proj_bias', None)

        self.bias_k = self.bias_v = None
        self.add_zero_attn = False

    def forward(self, q, k, v, attn_mask, key_padding_mask):
        k = q if k is None else k
        v = q if v is None else v
        if self.batch_first:
            q, k, v = [x.transpose(1, 0) for x in (q, k, v)]
        attn_output, attn_output_weights = F.multi_head_attention_forward(
            q, k, v, self.embed_dim_check, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k,
            self.bias_v, self.add_zero_attn, self.dropout, self.out_weight, self.out_bias, training=self.training,
            key_padding_mask=key_padding_mask, need_weights=self.need_weights, attn_mask=attn_mask,
            use_separate_proj_weight=True, q_proj_weight=self.q_weight, k_proj_weight=self.k_weight,
            v_proj_weight=self.v_weight
        )
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


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

        return torch.prelu(input, weight)


def prelu(input, weight, data_format):

    return torch.prelu(input, weight)