#! /usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


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
    raise NotImplementedError


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

    raise NotImplementedError


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

    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x):
        return F.leaky_relu(x, negative_slope=self.alpha)


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

    return F.leaky_relu(x, negative_slope=alpha)


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

    def __init__(self, keep, seed=0):
        self.keep = keep
        self.seed = seed

    def __call__(self, inputs):
        return F.dropout(inputs, p=(1-self.keep))


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

    def __init__(self, data_format=None):
        self.data_format = data_format

    def __call__(self, x, bias):
        return torch.add(x, bias)


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

    raise NotImplementedError


class Conv1D(object):

    def __init__(self, stride, padding, data_format='NWC', dilations=None, out_channel=None, k_size=None):
        self.stride = stride
        self.dilations = dilations
        self.data_format, self.padding = preprocess_1d_format(data_format, padding)

    def __call__(self, input, filters):
        raise NotImplementedError


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

    raise NotImplementedError


def same_padding(input, weight, strides, dilations):
    #                     H(in) + 2* padding[0] - dilation[0] * (Ksize[0] - 1) - 1
    # H(out) = = floor( --------------------------------------------------------------   + 1 )
    #                                        stride[0]
    if isinstance(weight, torch.Tensor):
        if len(input.shape) == 4:
            filter_rows = weight.size(2)
            filter_cols = weight.size(3)
        elif len(input.shape) == 5:
            filter_rows = weight.size(2)
            filter_cols = weight.size(3)
            filter_depth = weight.size(4)
    else:
        if len(input.shape) == 4:
            filter_rows = weight[0]
            filter_cols = weight[1]
        elif len(input.shape) == 5:
            filter_rows = weight[0]
            filter_cols = weight[1]
            filter_depth = weight[2]

    if len(input.shape) == 4:
        input_rows = input.size(2)
        input_cols = input.size(3)

        # filter_rows = weight.size(2)
        # filter_cols = weight.size(3)

        out_rows = (input_rows + strides[0] - 1) // strides[0]
        out_cols = (input_cols + strides[1] - 1) // strides[1]

        padding_rows = max(0, (out_rows - 1) * strides[0] +
                           (filter_rows - 1) * dilations[0] + 1 - input_rows)
        padding_cols = max(0, (out_cols - 1) * strides[1] +
                           (filter_cols - 1) * dilations[1] + 1 - input_cols)

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

        padding_rows = max(0, (out_rows - 1) * strides[0] +
                           (filter_rows - 1) * dilations[0] + 1 - input_rows)
        padding_cols = max(0, (out_cols - 1) * strides[1] +
                           (filter_cols - 1) * dilations[1] + 1 - input_cols)
        padding_depth = max(0, (out_depth - 1) * strides[2] +
                            (filter_depth - 1) * dilations[2] + 1 - input_depth)

        rows_odd = (padding_rows % 2 != 0)
        cols_odd = (padding_cols % 2 != 0)
        depth_odd = (padding_depth % 2 != 0)
        return rows_odd, cols_odd, depth_odd, padding_rows, padding_cols, padding_depth



class Conv2D(object):

    def __init__(self, strides, padding, data_format='NHWC', dilations=None, out_channel=None, k_size=None):
        self.strides = (strides[1], strides[2])
        self.dilations = (dilations[1], dilations[2])
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)

    def __call__(self, input, filters):
        if self.data_format == 'NHWC':
            input = nhwc_to_nchw(input)

        if self.padding == 'same':
            output = self.conv2d_same_padding(input, filters)
        else:
            output = F.conv2d(input, filters, stride=self.strides, padding=self.padding, dilation=self.dilations)

        if self.data_format == 'NHWC':
            output = nchw_to_nhwc(output)
        return output

    def conv2d_same_padding(self, input, weight, bias=None, groups=1):
        rows_odd, cols_odd, padding_rows, padding_cols = same_padding(input, weight, self.strides, self.dilations)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

        return F.conv2d(input, weight, bias, self.strides,
                        padding=(padding_rows // 2, padding_cols // 2),
                        dilation=self.dilations, groups=groups)



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
        self.strides = strides
        self.dilations = dilations
        self.data_format, self.padding = preprocess_3d_format(data_format, padding)

    def __call__(self, input, filters):
        raise NotImplementedError


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

    def __init__(self, ksize, strides, padding, data_format=None):
        self.data_format, self.padding = preprocess_1d_format(data_format=data_format, padding=padding)
        self.ksize = ksize
        self.strides = strides

    def __call__(self, inputs):
        raise NotImplementedError


class MaxPool(object):

    def __init__(self, ksize, strides, padding, data_format=None):
        self.ksize = ksize
        self.strides = (strides[1], strides[2])
        if data_format in ['channels_last', 'NLC', 'NWC', 'NHWC', 'NDHWC']:
            self.data_format = 'channels_last'
        elif data_format in ['channels_first', 'NCL', 'NCW', 'NCHW', 'NCDHW']:
            self.data_format = 'channels_first'
        self.padding = padding

    def __call__(self, inputs):
        if self.data_format == 'channels_last':
            inputs = nhwc_to_nchw(inputs)
        if len(inputs.shape) == 2 or len(inputs.shape) == 3:
            out = F.max_pool1d(inputs, self.ksize, self.strides)
        if len(inputs.shape) == 4:
            if self.padding == 'SAME':
                out = self.maxpool2d_same_padding(inputs)
            else:
                out = F.max_pool2d(inputs, self.ksize, self.strides)
        if len(inputs.shape) == 5:
            if self.padding == 'SAME':
                out = self.maxpool3d_same_padding(inputs)
            else:
                out = F.max_pool3d(inputs, self.ksize, self.strides)

        if self.data_format == 'channels_last':
            return nchw_to_nhwc(out)
        else:
            return out

    def maxpool2d_same_padding(self, input):
        rows_odd, cols_odd, padding_rows, padding_cols = same_padding(input, self.ksize, self.strides, (1, 1))
        if rows_odd or cols_odd:
            # TODO The fill value for maxpool is -INF.
            input = F.pad(input, [0, int(rows_odd), 0, int(cols_odd)], 'constant', float('-inf'))

        return F.max_pool2d(input, self.ksize , self.strides,
                        padding=(padding_rows // 2, padding_cols // 2))

    def maxpool3d_same_padding(self, input):
        rows_odd, cols_odd, depth_odd, padding_rows, padding_cols, padding_depth = same_padding(input, self.ksize,
                                                                                                self.strides, (1, 1, 1))
        if rows_odd or cols_odd:
            input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd), int(depth_odd)])
            return F.max_pool3d(input, self.ksize, self.strides,
                                padding=(padding_rows // 2, padding_cols // 2, padding_depth // 2))



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

    raise NotImplementedError


class AvgPool1d(object):

    def __init__(self, ksize, strides, padding, data_format=None):
        self.data_format, self.padding = preprocess_1d_format(data_format=data_format, padding=padding)
        self.ksize = ksize
        self.strides = strides

    def __call__(self, inputs):
        raise NotImplementedError


class AvgPool(object):

    def __init__(self, ksize, strides, padding, data_format=None):
        self.ksize = ksize
        self.strides = strides
        self.data_format = data_format
        self.padding = padding_format(padding)

    def __call__(self, inputs):
        raise NotImplementedError


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

    raise NotImplementedError


class MaxPool3d(object):

    def __init__(self, ksize, strides, padding, data_format=None):
        self.data_format, self.padding = preprocess_3d_format(data_format, padding)
        self.ksize = ksize
        self.strides = strides

    def __call__(self, inputs):
        raise NotImplementedError


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

    raise NotImplementedError


class AvgPool3d(object):

    def __init__(self, ksize, strides, padding, data_format=None):
        self.data_format, self.padding = preprocess_3d_format(data_format, padding)
        self.ksize = ksize
        self.strides = strides

    def __call__(self, inputs):
        raise NotImplementedError


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

    raise NotImplementedError


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
    raise NotImplementedError


class DepthwiseConv2d(object):

    def __init__(self, strides, padding, data_format=None, dilations=None, ksize=None, channel_multiplier=1):
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.strides = strides
        self.dilations = dilations

    def __call__(self, input, filter, point_filter=None):
        raise NotImplementedError


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

    raise NotImplementedError


class Conv1d_transpose(object):

    def __init__(
        self, stride, padding, data_format='NWC', dilations=None, out_channel=None, k_size=None, in_channels=None
    ):
        self.stride = stride
        self.dilations = dilations
        self.data_format, self.padding = preprocess_1d_format(data_format, padding)

    def __call__(self, input, filters):
        raise NotImplementedError


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

    raise NotImplementedError


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
        raise NotImplementedError


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

    raise NotImplementedError


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
        raise NotImplementedError


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

    raise NotImplementedError


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

    raise NotImplementedError


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
        raise NotImplementedError

    def _check_input_shape(self, inputs):
        if len(inputs.shape) <= 1:
            raise ValueError('expected input at least 2D, but got {}D input'.format(len(inputs.shape)))

    def __call__(self, inputs):
        raise NotImplementedError


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

        raise NotImplementedError


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
        raise NotImplementedError


class SeparableConv2D(object):

    def __init__(self, strides, padding, data_format, dilations, out_channel, k_size, in_channel, depth_multiplier):
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.strides = strides
        self.dilations = (dilations[2], dilations[2])

    def __call__(self, inputs, depthwise_filters, pointwise_filters):
        raise NotImplementedError


class AdaptiveMeanPool1D(object):

    def __init__(self, output_size, data_format):
        self.data_format, _ = preprocess_1d_format(data_format, None)
        self.output_size = output_size

    def __call__(self, input):
        raise NotImplementedError


class AdaptiveMeanPool2D(object):

    def __init__(self, output_size, data_format):
        self.data_format, _ = preprocess_2d_format(data_format, None)
        self.output_size = output_size

    def __call__(self, inputs):
        raise NotImplementedError


class AdaptiveMeanPool3D(object):

    def __init__(self, output_size, data_format):
        self.data_format, _ = preprocess_3d_format(data_format, None)
        self.output_size = output_size

    def __call__(self, inputs):
        raise NotImplementedError


class AdaptiveMaxPool1D(object):

    def __init__(self, output_size, data_format):
        self.data_format, _ = preprocess_1d_format(data_format, None)
        self.output_size = output_size

    def __call__(self, input):
        raise NotImplementedError


class AdaptiveMaxPool2D(object):

    def __init__(self, output_size, data_format):
        self.data_format, _ = preprocess_2d_format(data_format, None)
        self.output_size = output_size

    def __call__(self, inputs):
        raise NotImplementedError


class AdaptiveMaxPool3D(object):

    def __init__(self, output_size, data_format):
        self.data_format, _ = preprocess_3d_format(data_format, None)
        self.output_size = output_size

    def __call__(self, inputs):
        raise NotImplementedError


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

    def __call__(self, input, h, c=None):
        raise NotImplementedError


class lstmcell(object):

    def __init__(self, weight_ih, weight_hh, bias_ih, bias_hh, act=None):
        self.weight_ih = weight_ih
        self.weight_hh = weight_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh

    def __call__(self, input, h, c):
        raise NotImplementedError


class grucell(object):

    def __init__(self, weight_ih, weight_hh, bias_ih, bias_hh, act=None):
        self.weight_ih = weight_ih
        self.weight_hh = weight_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh

    def __call__(self, input, h, c=None):
        raise NotImplementedError


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
        weights_fw,
        weights_bw,
        bias_fw,
        bias_bw,
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

        self.weights_fw = weights_fw
        self.bias_fw = bias_fw
        self.weights_bw = weights_bw
        self.bias_bw = bias_bw

        # stdv = 1.0 / np.sqrt(self.hidden_size)
        # _init = tf.random_uniform_initializer(minval=-stdv, maxval=stdv)

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
        raise NotImplementedError

    def _rnn_forward(self, x, h, c=None):
        raise NotImplementedError

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

    def __call__(self, input, states):
        raise NotImplementedError


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
        raise NotImplementedError

    def __call__(self, input):
        raise NotImplementedError


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
        raise NotImplementedError


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
