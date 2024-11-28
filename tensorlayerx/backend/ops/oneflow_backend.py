#! /usr/bin/python
# -*- coding: utf-8 -*-

# Unified API for TensorLayerX, using OneFlow as backend.
# Similar to file ./torch_backend.py and ./tensorflow_backend.py

from __future__ import absolute_import, division, print_function

from .oneflow_nn import nchw_to_nhwc, nhwc_to_nchw
import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F



import numpy as np
import random

_dtypeDict = {
    'Dtype': flow.dtype,
    'float16': flow.float16,
    'float32': flow.float32,
    'float64': flow.float64,
    'int8': flow.int8,
    'int16': None,
    'int32': flow.int32,
    'int64': flow.int64,
    'uint8': flow.uint8,
    'uint16': None,
    'uint32': None,
    'uint64': None,
    'bool': flow.bool,
    'complex64': None,
    'complex128': None
}

DType = flow.dtype
float16 = flow.float16
float32 = flow.float32
float64 = flow.float64
int8 = flow.int8
int16 = None
int32 = flow.int32
int64 = flow.int64
uint8 = flow.uint8
uint16 = None
uint32 = None
uint64 = None
bool = flow.bool
complex64 = None
complex128 = None


def set_context(**kwargs):
    """Set the context for the backend.
    """
    raise Exception("Using OneFlow backend, set_context is not supported.")


def get_tensor_shape(x):
    """
    Get the shape of tensor

    Parameters
    ----------
    x : tensor
         type float16, float32, float64, int32, complex64, complex128.

    Returns
    -------
        list.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x_in = tlx.layers.Input((32, 3, 3, 32))
    >>> x_shape = tlx.ops.get_tensor_shape(x_in)

    """
    return list(x.shape)


# initializers
def zeros(shape, dtype=None, device=None):
    """
    Creates a tensor with all elements set to zero.

    Parameters
    ----------
    shape : A list of integers
        a tuple of integers, or a 1-D Tensor of type int32.
    dtype : tensor
        The DType of an element in the resulting Tensor

    Returns
    -------
        A Tensor with all elements set to zero.

    """
    if device == 'cpu':
        device = flow.device('cpu')
    else:
        device = flow.device('cuda' if flow.cuda.is_available() else 'cpu')

    return flow.zeros(shape, dtype=_dtypeDict[dtype], device=device)


def ones(shape, dtype=None, device=None):
    """
    Creates a tensor with all elements set to one.

    Parameters
    ----------
    shape : A list of integers
        a tuple of integers, or a 1-D Tensor of type int32.
    dtype : tensor
        The DType of an element in the resulting Tensor

    Returns
    -------
        A Tensor with all elements set to one.

    """
    if device == 'cpu':
        device = flow.device('cpu')
    else:
        device = flow.device('cuda' if flow.cuda.is_available() else 'cpu')

    return flow.ones(shape, dtype=_dtypeDict[dtype], device=device)


def constant(value, shape, dtype=None, device=None):
    """
    Creates a tensor with all elements set to the value.

    Parameters
    ----------
    value : A constant value (or list)
    shape : A list of integers
        a tuple of integers, or a 1-D Tensor of type int32.
    dtype : tensor
        The DType of an element in the resulting Tensor

    Returns
    -------
        A Tensor with all elements set to the value.

    """
    if device == 'cpu':
        device = flow.device('cpu')
    else:
        device = flow.device('cuda' if flow.cuda.is_available() else 'cpu')

    return flow.full(shape, value, dtype=_dtypeDict[dtype], device=device)


def random_uniform(shape, minval=0, maxval=1, dtype=None, seed=None):
    """
    Outputs random values from a uniform distribution.

    Parameters
    ----------
    shape : tuple
        A 1-D integer Tensor or Python array. The shape of the output tensor.
    minval : int
        The lower bound on the range of random values to generate (inclusive). Defaults to 0.
    maxval : int
        The upper bound on the range of random values to generate (exclusive). Defaults to 1 if dtype is floating point.
    dtype : tensor
        The type of the output: float16, float32, float64, int32, or int64.
    seed : int
         Used in combination with tf.random.set_seed to create a reproducible sequence of tensors across multiple calls.
    Returns
    -------
        A tensor of the specified shape filled with random uniform values.

    """

    if seed is not None:
        flow.manual_seed(seed)
    else:
        flow.manual_seed(flow.initial_seed())

    w = flow.randn(shape, dtype=_dtypeDict[dtype])
    out = w.uniform_(minval, maxval)
    return out


def random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    """
    Outputs random values from a normal distribution.

    Parameters
    ----------
    shape : tuple
        A 1-D integer Tensor or Python array. The shape of the output tensor.
    mean : int
        The mean of the normal distribution.
    stddev : int
        The standard deviation of the normal distribution.
    dtype : tensor
        The type of the output: float16, float32, float64, int32, or int64.
    seed : int
         Used in combination with tf.random.set_seed to create a reproducible sequence of tensors across multiple calls.
    Returns
    -------
        A tensor of the specified shape filled with random normal values.

    """

    if seed is not None:
        flow.manual_seed(seed)
    else:
        flow.manual_seed(flow.initial_seed())

    return flow.normal(shape, mean=mean, std=stddev, dtype=_dtypeDict[dtype])


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    """
    Outputs random values from a truncated normal distribution.

    Parameters
    ----------
    shape : tuple
        A 1-D integer Tensor or Python array. The shape of the output tensor.
    mean : int
        The mean of the normal distribution.
    stddev : int
        The standard deviation of the normal distribution.
    dtype : tensor
        The type of the output: float16, float32, float64, int32, or int64.
    seed : int
         Used in combination with tf.random.set_seed to create a reproducible sequence of tensors across multiple calls.
    Returns
    -------
        A tensor of the specified shape filled with random normal values.

    """

    if seed is not None:
        flow.manual_seed(seed)
    else:
        flow.manual_seed(flow.initial_seed())

    w = flow.empty(shape, dtype=_dtypeDict[dtype])
    out = nn.init.truncated_normal_(w, mean=mean, std=stddev)
    return out


def he_normal(shape, dtype=None, seed=None):
    """
    He normal initializer.

    It draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / fan_in) where fan_in is the number of input units in the weight tensor.

    Parameters
    ----------
    shape : tuple
        A 1-D integer Tensor or Python array. The shape of the output tensor.
    dtype : tensor
        The type of the output: float16, float32, float64, int32, or int64.
    seed : int
         Used in combination with tf.random.set_seed to create a reproducible sequence of tensors across multiple calls.
    Returns
    -------
        A tensor of the specified shape filled with random normal values.

    """

    if seed is not None:
        flow.manual_seed(seed)
    else:
        flow.manual_seed(flow.initial_seed())

    w = flow.empty(shape, dtype=_dtypeDict[dtype])
    out = nn.init.kaiming_normal_(w)
    return out


def he_uniform(shape, dtype=None, seed=None):
    """
    He uniform variance scaling initializer.

    It draws samples from a uniform distribution within [-limit, limit] where limit is sqrt(6 / fan_in) where fan_in is the number of input units in the weight tensor.

    Parameters
    ----------
    shape : tuple
        A 1-D integer Tensor or Python array. The shape of the output tensor.
    dtype : tensor
        The type of the output: float16, float32, float64, int32, or int64.
    seed : int
         Used in combination with tf.random.set_seed to create a reproducible sequence of tensors across multiple calls.
    Returns
    -------
        A tensor of the specified shape filled with random uniform values.

    """

    if seed is not None:
        flow.manual_seed(seed)
    else:
        flow.manual_seed(flow.initial_seed())

    w = flow.empty(shape, dtype=_dtypeDict[dtype])
    out = nn.init.kaiming_uniform_(w)
    return out


def xavier_normal(shape, dtype=None, seed=None):
    """
    Xavier normal initializer.

    It draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / (fan_in + fan_out)) where fan_in is the number of input units in the weight tensor and fan_out is the number of output units in the weight tensor.

    Parameters
    ----------
    shape : tuple
        A 1-D integer Tensor or Python array. The shape of the output tensor.
    dtype : tensor
        The type of the output: float16, float32, float64, int32, or int64.
    seed : int
         Used in combination with tf.random.set_seed to create a reproducible sequence of tensors across multiple calls.
    Returns
    -------
        A tensor of the specified shape filled with random normal values.

    """

    if seed is not None:
        flow.manual_seed(seed)
    else:
        flow.manual_seed(flow.initial_seed())

    w = flow.empty(shape, dtype=_dtypeDict[dtype])
    out = nn.init.xavier_normal_(w)
    return out


def xavier_uniform(shape, gain=1.0, dtype=None, seed=None):
    """
    Xavier uniform initializer.

    It draws samples from a uniform distribution within [-limit, limit] where limit is sqrt(6 / (fan_in + fan_out)) where fan_in is the number of input units in the weight tensor and fan_out is the number of output units in the weight tensor.

    Parameters
    ----------
    shape : tuple
        A 1-D integer Tensor or Python array. The shape of the output tensor.
    gain : float
        A multiplicative factor.
    dtype : tensor
        The type of the output: float16, float32, float64, int32, or int64.
    seed : int
         Used in combination with tf.random.set_seed to create a reproducible sequence of tensors across multiple calls.
    Returns
    -------
        A tensor of the specified shape filled with random uniform values.

    """

    if seed is not None:
        flow.manual_seed(seed)
    else:
        flow.manual_seed(flow.initial_seed())

    w = flow.empty(shape, dtype=_dtypeDict[dtype])
    out = nn.init.xavier_uniform_(w, gain=gain)
    return out


def Variable(initial_value, name=None, trainable=True):
    """
    Creates a new Variable.

    Parameters
    ----------
    initial_value : tensor
        A Tensor or Python object convertible to a Tensor which is the initial value for the Variable.
    name : str
        A name for the operation (optional).
    trainable : bool
        If True, also add the variable to the graph collection GraphKeys.TRAINABLE_VARIABLES (see tf.Variable).

    Returns
    -------
        A Variable object.

    """

    return flow.nn.Parameter(initial_value, name=name, requires_grad=trainable)


class MatMul(object):
    def __init__(self, transpose_a=False, transpose_b=False, name=None):
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        self.name = name
        if self.transpose_a or self.transpose_b:
            raise NotImplementedError('keyword argument `transpose_a` or `transpose_b` is not supported.')

    def __call__(self, x, y):
        return flow.matmul(x, y)


def matmul(a, b, transpose_a=False, transpose_b=False):
    """
    Multiplies matrix a by matrix b, producing a * b.

    The inputs must, following any transpositions, be tensors of rank >= 2 where the inner 2 dimensions specify valid matrix multiplication arguments, and any further outer dimensions match.

    Both matrices must be of the same type. The supported types are: float16, float32, float64, int32, complex64, complex128.

    Either matrix can be transposed on the fly by setting one of the corresponding flag to True. This is `False` by default.

    Args:
        a (Tensor): A Tensor. Must be one of the following types: float16, float32, float64, int32, complex64, complex128.
        b (Tensor): A Tensor. Must have the same type as a.
        transpose_a (bool, optional):
            If True, a is transposed before multiplication. Defaults to False.
        transpose_b (bool, optional):
            If True, b is transposed before multiplication. Defaults to False.

    Returns:
        Tensor: The product of the matrix multiplication. Has the same type as a.

    """
    return flow.matmul(a, b)


def add(value, bias):
    """
    Adds bias to value.

    This is a special case of addN where N=1.

    Args:
        value (Tensor): A Tensor. Must be one of the following types: float16, float32, float64, int32, int64, complex64, complex128.
        bias (Tensor): A Tensor. Must have the same type as value.

    Returns:
        Tensor: A Tensor. Has the same type as value.

    """
    return flow.add(value, bias)


def dtypes(dt):
    """
    Returns the data type of dt as a DType.

    Args:
        dt (Tensor): string
         It could be 'uint8', 'int8', 'uint16', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'bool', 'char', 'string', 'complex64', 'complex128', 'int32', 'int64', 'int8', 'uint8', 'float16', 'float32', 'float64', 'complex64', 'complex128', 'bfloat16', 'qint8', 'quint8', 'qint32', 'half', 'resource', 'variant', 'uint32', 'uint64', 'double', 'long', 'int', 'short', 'byte', 'float', 'complex'.

    Returns:
        DType: The data type of dt.

    """
    if dt not in _dtypeDict.keys():
        raise Exception("Unsupported dtype: {}".format(dt))
    return _dtypeDict[dt]


class Maximum(object):
    def __init__(self):
        pass

    def forward(self, x, y):
        return flow.maximum(x, y)


class Minimum(object):
    def __init__(self):
        pass

    def forward(self, x, y):
        return flow.minimum(x, y)


def maximum(x, y):
    """
    Returns the max of x and y (i.e. x > y ? x : y) element-wise.

    Args:
        x (Tensor): A Tensor. Must be one of the following types: float16, float32, float64, int32, int64, complex64, complex128.
        y (Tensor): A Tensor. Must have the same type as x.

    Returns:
        Tensor: A Tensor. Has the same type as x.

    """
    return flow.maximum(x, y)


def minimum(x, y):
    """
    Returns the min of x and y (i.e. x < y ? x : y) element-wise.

    Args:
        x (Tensor): A Tensor. Must be one of the following types: float16, float32, float64, int32, int64, complex64, complex128.
        y (Tensor): A Tensor. Must have the same type as x.

    Returns:
        Tensor: A Tensor. Has the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant([0., 0., 0., 0.])
    >>> y = tlx.ops.constant([-5., -2., 0., 3.])
    >>> z = tlx.ops.minimum(x, y)


    """
    return flow.minimum(x, y)


class FlattenReshape(object):
    def __init__(self):
        pass

    def forward(self, x):
        return flow.reshape(x, (-1,))


class Reshape(object):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        return flow.reshape(x, self.shape)


def reshape(tensor, shape):
    """
    Reshapes a tensor.

    Parameters
    ----------
    tensor : tensor
        A Tensor.
    shape : tensor
         Defines the shape of the output tensor.
    Returns
    -------
        A Tensor. Has the same type as tensor
    """

    return flow.reshape(tensor, shape)


class Concat(object):
    def __init__(self, axis=0):
        self.axis = axis

    def forward(self, values):
        return flow.concat(values, dim=self.axis)


def concat(values, axis=0):
    """
    Concatenates tensors along one dimension.

    Parameters
    ----------
    values : list
         A list of Tensor objects or a single Tensor
    axis : int
        0-D int32 Tensor. Dimension along which to concatenate
    Returns
    -------
        A Tensor resulting from concatenation of the input tensors.
    """

    return flow.concat(values, dim=axis)


def convert_to_tensor(value, dtype=None, device=None):
    """
    Converts the given value to a Tensor.

    Parameters
    ----------
    value : object
        An object whose type has a registered Tensor conversion function.
    dtype : optional
        Optional element type for the returned tensor. If missing, the type is inferred from the type of value.

    Returns
    -------
        A Tensor based on value.
    """

    if isinstance(dtype, str):
        dtype = _dtypeDict[dtype]

    if device == 'cpu':
        device = flow.device('cpu')
    else:
        device = flow.device('cuda' if flow.cuda.is_available() else 'cpu')

    return flow.tensor(value, dtype=dtype, device=device)


def convert_to_numpy(value):
    try:
        return value.numpy()
    except:
        return value.cpu().numpy()


def sqrt(x):
    """
    Computes square root of x element-wise.

    Parameters
    ----------
    x : tensor
         Must be one of the following types: bfloat16, half, float32, float64, complex64, complex128.

    Returns
    -------
        A Tensor. Has the same type as x.
    """

    return flow.sqrt(x)


class ReduceSum(object):

    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def __call__(self, input):
        if self.axis is not None:
            return flow.sum(input, dim=self.axis, keepdim=self.keepdims)
        else:
            return flow.sum(input, keepdim=self.keepdims)

class ReduceMean(object):

    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def __call__(self, inputs):
        if self.axis is not None:
            return flow.mean(input=inputs, dim=self.axis, keepdim=self.keepdims)
        else:
            return flow.mean(inputs)

def reduce_mean(input_tensor, axis=None, keepdims=False):
    """
    Computes the mean of elements across dimensions of a tensor.

    Parameters
    ----------
    input_tensor : tensor
        The tensor to reduce. Should have numeric type.
    axis : list
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range [-rank(input_tensor), rank(input_tensor)).
    name : str
        A name for the operation (optional).

    Returns
    -------
        The reduced tensor.
    """

    if axis is not None:
        return flow.mean(input_tensor, dim=axis, keepdim=keepdims)
    else:
        return flow.mean(input_tensor)


class ReduceMax(object):

    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def __call__(self, inputs):
        if self.axis is not None:
            if isinstance(self.axis, (list, tuple)):
                out = inputs
                for dim in self.axis[::-1]:
                    out = flow.max(out, dim, keepdim=self.keepdims)
                return out
            else:
                return flow.max(inputs, self.axis, keepdim=self.keepdims)
        else:
            return flow.max(inputs)


def reduce_max(input_tensor, axis=None, keepdims=False):
    """
    Computes the maximum of elements across dimensions of a tensor.

    Parameters
    ----------
    input_tensor : tensor
        The tensor to reduce. Should have real numeric type.
    axis : int
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range [-rank(input_tensor), rank(input_tensor)).
    name : str
        A name for the operation (optional).

    Returns
    -------
        The reduced tensor.
    """

    if axis is not None:
        return flow.max(input_tensor, dim=axis, keepdim=keepdims)
    else:
        return flow.max(input_tensor)


def reduce_min(input_tensor, axis=None, keepdims=False):
    """
    Computes the minimum of elements across dimensions of a tensor.

    Parameters
    ----------
    input_tensor : tensor
        The tensor to reduce. Should have real numeric type.
    axis : int
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range [-rank(input_tensor), rank(input_tensor)).
    name : str
        A name for the operation (optional).

    Returns
    -------
        The reduced tensor.
    """

    if axis is not None:
        return flow.min(input_tensor, dim=axis, keepdim=keepdims)
    else:
        return flow.min(input_tensor)


class Pad2d(object):
    def __init__(self, padding, mode='constant', value=0.0, data_format="NCHW", name=None):
        pass

    def __call__(self, x):
        pass


class Pad(object):

    def __init__(self, paddings, mode="REFLECT", constant_values=0.0):
        if mode not in ['CONSTANT', 'REFLECT', 'SYMMETRIC']:
            raise Exception("Unsupported mode: {}".format(mode))
        self.paddings = self.correct_paddings(paddings)
        self.mode = mode.lower()
        self.constant_values = constant_values

    def __call__(self, x):
        if self.mode in ['symmetric', 'reflect']:
            if len(x.shape) == 3 and self.paddings[0:2] + self.paddings[4:] == (0, 0, 0, 0):
                self.paddings = (self.paddings[2], self.paddings[3])
                x = flow.transpose(x, 1, 2)
            elif len(x.shape) == 4 and self.paddings[0:2] + self.paddings[6:] == (0, 0, 0, 0, 0, 0):
                self.paddings = (self.paddings[2], self.paddings[3], self.paddings[4], self.paddings[5])
                x = flow.transpose(x, 1, 3)
            elif len(x.shape) == 5 and self.paddings[0:2] + self.paddings[8:] == (0, 0, 0, 0, 0, 0, 0, 0):
                self.paddings = (self.paddings[2], self.paddings[3], self.paddings[4],
                                 self.paddings[5], self.paddings[6], self.paddings[7])
                x = flow.transpose(x, 1, 4)
            else:
                raise NotImplementedError("Only constant padding is implemented for arbitrary dimensions.")

        out = flow.pad(x, self.paddings, mode=self.mode, value=self.constant_values)

        if self.mode in ['symmetric', 'reflect']:
            if len(x.shape) == 3:
                out = flow.transpose(out, 1, 2)
            elif len(x.shape) == 4:
                out = flow.transpose(out, 1, 3)
            elif len(x.shape) == 5:
                out = flow.transpose(out, 1, 4)

        return out

    def correct_paddings(self, paddings):
        paddings = paddings[::-1]
        _padding = []
        for p_i in paddings:
            for pj in p_i:
                _padding.append(pj)
        return tuple(_padding)


def pad(tensor, paddings, mode='CONSTANT', constant_values=0):
    """
    Pads a tensor.

    Parameters
    ----------
    tensor : tensor
        A Tensor.
    paddings : tensor
        A Tensor of type int32.
    mode : str
        One of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive)
    constant_values : int
        In "CONSTANT" mode, the scalar pad value to use. Must be same type as tensor.

    Returns
    -------
        A Tensor. Has the same type as tensor.
    """
    pad_obj = Pad(paddings, mode, constant_values)
    return pad_obj(tensor)


class Unstack(object):

    def __init__(self, axis, num=None):
        self.axis = axis
        self.num = num

    def __call__(self, values):
        out = []
        for o in flow.chunk(values, chunks=self.num, dim=self.axis):
            out.append(flow.squeeze(o))
        return out


class Stack(object):

    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, values):
        return flow.stack(values, dim=self.axis)


def stack(values, axis=0):
    """
    Stacks a list of rank-R tensors into one rank-(R+1) tensor.

    Parameters
    ----------
    values : list
        A list of Tensor objects with the same shape and type.
    axis : int
        An int. The axis to stack along. Defaults to the first dimension.
        Negative values wrap around, so the valid range is [-(R+1), R+1).

    Returns
    -------
        A stacked Tensor with the same type as values.
    """

    return flow.stack(values, dim=axis)


class Meshgrid(object):

    def __init__(self, indexing='xy'):
        super(Meshgrid, self).__init__()
        self.index = indexing

    def __call__(self, *xi):
        return flow.meshgrid(xi, indexing=self.index)


def meshgrid(*args, **kwargs):
    """
    Broadcasts parameters for evaluation on an N-D grid.

    Parameters
    ----------
    x : tensor
        Tensors with rank 1.
    y : tensor
        Tensors with rank 1.

    Returns
    -------
        A list of N Tensors with rank N.
    """

    return flow.meshgrid(*args, **kwargs)


def arange(start, limit=None, delta=1, dtype=None):
    """
    Creates a sequence of numbers.

    Parameters
    ----------
    start : tensor
        A 0-D Tensor (scalar). Acts as first entry in the range if limit is not None;
        otherwise, acts as range limit and first entry defaults to 0.
    limit : tensor
         A 0-D Tensor (scalar). Upper limit of sequence, exclusive. If None,
         defaults to the value of start while the first entry of the range defaults to 0.
    delta : tensor
        A 0-D Tensor (scalar). Number that increments start. Defaults to 1.
    dtype : None or dtype
        The type of the elements of the resulting tensor.

    Returns
    -------
        An 1-D Tensor of type dtype.
    """

    return flow.arange(start, end=limit, step=delta, dtype=dtype)


class ExpandDims(object):

    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, input):
        return flow.unsqueeze(input, dim=self.axis)


def expand_dims(input, axis):
    """
    Inserts a dimension of 1 into a tensor's shape.

    Parameters
    ----------
    input : tensor
        A Tensor.
    axis : int
        0-D (scalar). Specifies the dimension index at which to expand the shape of input.
        Must be in the range [-rank(input) - 1, rank(input)].

    Returns
    -------
        A Tensor with the same data as input, but its shape has an additional dimension of size 1 added.
    """

    return flow.unsqueeze(input, dim=axis)


class Tile(object):

    def __init__(self):
        pass

    def __call__(self, input, multiples):
        return flow.tile(input, multiples)


def tile(input, multiples):
    """
    Constructs a tensor by tiling a given tensor.

    Parameters
    ----------
    input : tensor
        A Tensor. 1-D or higher.
    multiples : tensor
        Must be one of the following types: int32, int64. 1-D.
        Length must be the same as the number of dimensions in input

    Returns
    -------
        A Tensor. Has the same type as input.
    """

    return flow.tile(input, multiples)


class Cast(object):

    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, x):

        return x.type(self.dtype)


def cast(x, dtype=None):
    """
    Casts a tensor to a new type.

    Parameters
    ----------
    x : tensor
        A Tensor or SparseTensor or IndexedSlices of numeric type.
        It could be uint8, uint16, uint32, uint64, int8, int16, int32, int64, float16, float32, float64.
    dtype : dtpye
         The destination type. The list of supported dtypes is the same as x

    Returns
    -------
        A Tensor or SparseTensor or IndexedSlices with same shape as x and same type as dtype.
    """

    return x.type(dtype)

class Transpose(object):

    def __init__(self, perm, conjugate=False):
        self.perm = perm
        self.conjugate = conjugate

    def __call__(self, a):
        return transpose(a, self.perm, self.conjugate)


def transpose(a, perm=None, conjugate=False):
    """
    Transposes a.

    Parameters
    ----------
    a : tensor
        A Tensor.
    perm : list / int
        A permutation of the dimensions of a.
    conjugate : bool
        Setting it to True is mathematically equivalent to tf.math.conj(tf.transpose(input)).

    Returns
    -------
        A transposed Tensor.
    """
    if perm == None:
        if len(a.shape) <= 2:
            return flow.t(a)
        if len(a.shape) == 3:
            perm = [2, 1, 0]
        if len(a.shape) == 4:
            perm = [3, 2, 1, 0]
        if len(a.shape) == 5:
            perm = [4, 3, 2, 1, 0]

    out = flow.permute(a, perm)
    if conjugate:
        pass  # Not support yet

    return out


def gather_nd(params, indices, batch_dims=0):
    """
    Gather slices from params into a Tensor with shape specified by indices.

    Parameters
    ----------
    params : tensor
        The tensor from which to gather values.
    indices : tensor
        Must be one of the following types: int32, int64. Index tensor.
    batch_dims : int
        An integer or a scalar 'Tensor'. The number of batch dimensions.

    Returns
    -------
        A Tensor. Has the same type as params.
    """

    out_shape = indices.shape[:-1]
    indices = indices.unsqueeze(0).transpose(0, -1)
    ndim = indices.shape[0]
    indices = indices.long()
    idx = flow.zeros_like(indices[0]).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)
    out = flow.gather(params, idx, dim=0)
    return out.reshape(out_shape)


def scatter_nd(indices, updates, shape):
    """
    Scatter updates into a new (initially zero) tensor according to indices.

    Parameters
    ----------
    indices : tensor
        A tensor of indices into a new tensor of shape shape. Must be one of the following types: int32, int64.
    updates : tensor
        A tensor of updated values to scatter into a new tensor. Must have the same type as ref.
    shape : tensor
        A 1-D tensor. The shape of the resulting tensor.

    Returns
    -------
        A Tensor. Has the same type as updates.
    """

    return flow.scatter_nd(indices, updates, shape)


class ClipGradByValue(object):
    def __init__(self, clip_min=-1, clip_max=1):
        self.min = clip_min
        self.max = clip_max

    def __call__(self, inputs):
        flow.nn.utils.clip_grad_value_(inputs, clip_value=self.max)


class ClipGradByNorm(object):
    def __init__(self, clip_norm=0.1):
        self.clip_norm = clip_norm

    def __call__(self, inputs):
        flow.nn.utils.clip_grad_norm_(inputs, max_norm=self.clip_norm, norm_type=2)


class ClipByGlobalNorm(object):
    def __init__(self, clip_norm=0.1):
        self.clip_norm = clip_norm

    def __call__(self, inputs):
        raise NotImplementedError



def clip_by_value(t, clip_value_min, clip_value_max):
    """
    Clips tensor values to a specified min and max.

    Parameters
    ----------
    t : tensor
        A Tensor or IndexedSlices
    clip_value_min : tensor
        A 0-D (scalar) Tensor, or a Tensor with the same shape as t. The minimum value to clip by
    clip_value_max : tensor
        A 0-D (scalar) Tensor, or a Tensor with the same shape as t. The minimum value to clip by

    Returns
    -------
        A clipped Tensor or IndexedSlices.
    """

    t_min = clip_value_min
    t_max = clip_value_max

    result = (t >= t_min) * t + (t < t_min) * t_min
    result = (result <= t_max) * result + (result > t_max) * t_max
    return result


def split(value, num_or_size_splits, axis=0):
    """
    Splits a tensor into sub tensors.

    Parameters
    ----------
    value : tensor
        The Tensor to split.
    num_or_size_splits : list
        Either an integer indicating the number of splits along split_dim or a 1-D integer Tensor or
        Python list containing the sizes of each output tensor along split_dim.
    axis : int
        The dimension along which to split. Must be in the range [-rank(value), rank(value)). Defaults to 0.
    num : int
        used to specify the number of outputs when it cannot be inferred from the shape of size_splits.

    Returns
    -------
        Tensor objects resulting from splitting value.
    """

    return flow.split(value, num_or_size_splits, dim=axis)


class Floor(object):

    def __call__(self, x):
        return flow.floor(x)


def floor(x):
    """
    Returns the floor of a tensor.

    Parameters
    ----------
    x : tensor
        A Tensor or SparseTensor or IndexedSlices of numeric type.

    Returns
    -------
        A Tensor or SparseTensor or IndexedSlices with same shape as x and same type as x.
    """

    return flow.floor(x)


def gather(params, indices, axis=None):
    if axis is None:
        axis = 0
    if axis < 0:
        axis = len(params.shape) + axis
    if axis == 0:
        return params[indices]
    elif axis == 1:
        return params[:, indices]
    elif axis == 2:
        return params[:, :, indices]
    elif axis == 3:
        return params[:, :, :, indices]


def linspace(start, stop, num):
    """
    Returns evenly spaced numbers over a specified interval.

    Parameters
    ----------
    start : tensor
        The starting value of the sequence.
    stop : tensor
        The end value of the sequence, unless endpoint is set to False.
    num : int
        Number of samples to generate.

    Returns
    -------
        A Tensor of one of the following types: float32, float64, int32, int64.
    """
    return flow.linspace(start=start, end=stop, steps=num)


def slice(inputs, starts, sizes):
    '''
    Extracts a slice from a tensor.
    '''

    ends = [starts[i] + sizes[i] for i in range(len(starts))]

    if len(inputs.shape) == 1:
        return inputs[starts[0]: ends[0]]
    if len(inputs.shape) == 2:
        return inputs[starts[0]: ends[0], starts[1]:ends[1]]
    if len(inputs.shape) == 3:
        return inputs[starts[0]: ends[0], starts[1]:ends[1], starts[2]:ends[2]]
    if len(inputs.shape) == 4:
        return inputs[starts[0]: ends[0], starts[1]:ends[1], starts[2]:ends[2], starts[3]:ends[3]]
    if len(inputs.shape) == 5:
        return inputs[starts[0]: ends[0], starts[1]:ends[1], starts[2]:ends[2], starts[3]:ends[3], starts[4]:ends[4]]


def add_n(inputs):
    a = inputs[0]
    for b in inputs[1:]:
        a += b
    return a


class OneHot(object):

    def __init__(self, depth=-1, on_value=None, off_value=None, axis=None, dtype=None):
        self.depth = depth
        self.on_value = on_value
        self.off_value = off_value
        self.axis = axis
        self.dtype = dtype

    def __call__(self, inputs):
        if [self.on_value, self.off_value] == [None, None]:
            return F.one_hot(inputs, self.depth)
        else:
            out = F.one_hot(inputs, self.depth)
            out = cast(out, flow.float64)
            out = flow.where(out == 1, self.on_value, out)
            out = flow.where(out == 0, self.off_value, out)
            out = cast(out, flow.int)
            return out


class L2Normalize(object):

    def __init__(self, axis=None, epsilon=1e-12):
        self.axis = axis
        self.epsilon = epsilon

    def __call__(self, input, *args, **kwargs):

        return F.normalize(input, p=2, dim=self.axis, eps=self.epsilon)


class EmbeddingLookup(object):

    def __init__(self, max_norm=None):
        self.max_norm = max_norm
        self.padding_idx = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False

    def __call__(self, params, ids):
        return F.embedding(
            ids, params, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse
        )


class NCELoss(object):

    def __init__(self, num_true=1, sampled_values=None, remove_accidental_hits=False):
        self.num_true = num_true
        self.sampled_values = sampled_values
        self.remove_accidental_hits = remove_accidental_hits

    def __call__(self, weights, biases, labels, inputs, num_sampled, num_classes):
        raise NotImplementedError


class NotEqual(object):

    def __init__(self):
        pass

    def __call__(self, x, y):
        return flow.not_equal(x, y)


class CountNonzero(object):

    def __init__(self, keepdims=False, dtype="int64"):
        self.keepdims = keepdims

    def __call__(self, input, axis=None):
        return flow.nonzero(input, as_tuple=self.keepdims)


class Resize:

    def __init__(self, scale, method, antialias=False, data_format='channels_last'):
        self.method = method
        self.antialias = antialias
        self.scale = scale
        self.data_format = data_format

    def __call__(self, inputs):
        if self.data_format == "channels_last":
            inputs = nhwc_to_nchw(inputs)
        outputs = F.interpolate(inputs, scale_factor=self.scale, mode=self.method, align_corners=self.antialias)
        if self.data_format == "channels_last":
            outputs = nchw_to_nhwc(outputs)
        return outputs


def resize(inputs, output_size, method, antialias):
    return F.interpolate(inputs, size=output_size, mode=method, align_corners=antialias)


class ZeroPadding1D(object):
    '''
    Pads the 2nd dimension of a 3D tensor.
    '''

    def __init__(self, padding, data_format):
        if data_format == 'channels_first':
            padding = ((0, 0), (0, 0), padding)
        elif data_format == 'channels_last':
            padding = ((0, 0), padding, (0, 0))
        else:
            raise ValueError('data_format must be channels_first or channels_last.')
        self.pad = Pad(paddings=padding)

    def __call__(self, inputs):
        return self.pad(inputs)


class ZeroPadding2D(object):
    '''
    Pads the 2nd and 3rd dimensions of a 4D tensor.
    '''

    def __init__(self, padding, data_format):
        if data_format == 'channels_first':
            padding = ((0, 0), (0, 0), padding[0], padding[1])
        elif data_format == 'channels_last':
            padding = ((0, 0), padding[0], padding[1], (0, 0))
        else:
            raise ValueError('data_format must be channels_first or channels_last.')
        self.pad = Pad(paddings=padding)

    def __call__(self, inputs):
        return self.pad(inputs)


class ZeroPadding3D(object):

    def __init__(self, padding, data_format):
        if data_format == 'channels_first':
            padding = ((0, 0), (0, 0), padding[0], padding[1], padding[2])
        elif data_format == 'channels_last':
            padding = ((0, 0), padding[0], padding[2], padding[1], (0, 0))
        else:
            raise ValueError('data_format must be channels_first or channels_last.')
        self.pad = Pad(paddings=padding)

    def __call__(self, inputs):
        return self.pad(inputs)


class Sign(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return flow.sign(x)


class Ceil(object):

    def __call__(self, x):
        return flow.ceil(x)


def ceil(x):
    return flow.ceil(x)


def multiply(x, y):
    return flow.mul(x, y)


def divide(x, y):
    return flow.div(x, y)


def identity(x):

    raise NotImplementedError

class BatchToSpace(object):

    def __init__(self, block_size, crops):
        self.bolock_size = block_size
        self.crops = crops

    def __call__(self, input_x):
        raise NotImplementedError

class DepthToSpace(object):

    def __init__(self, block_size, data_format):
        self.block_size = block_size
        self.data_format = data_format
        self.pixel_shuffle = nn.PixelShuffle(self.block_size)

    def __call__(self, input):
        if self.data_format == 'channels_last':
            input = nhwc_to_nchw(input)
        output = self.pixel_shuffle(input)

        if self.data_format == 'channels_last':
            output = nchw_to_nhwc(output)
        return output


def triu(data, diagonal=0):

    return flow.triu(data, diagonal=diagonal)


def tril(data, diagonal=0):

    return flow.tril(data, diagonal=diagonal)


def abs(x):
    return flow.abs(x)


def acos(x):
    return flow.acos(x)


def acosh(x):
    return flow.acosh(x)


def angle(x):
    x_np = convert_to_numpy(x)
    return convert_to_tensor(np.angle(x_np))


def argmax(x, axis=None, keepdim=False, dtype='int64'):
    """
    Returns the index with the largest value across axes of a tensor.

    Parameters
    ----------
    x : tensor
        A Tensor
    axis : int
        An integer, the axis to reduce across. Default to 0.
    dtype : tensor or str
        An optional output dtype (nt32 or int64). Defaults to int64.

    Returns
    -------
        A Tensor of type output_type.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=[10, 20, 5, 6, 15])
    >>> y = tlx.ops.argmax(x)

    """

    return flow.argmax(x, axis=axis, dtype=dtype)

def argmin(x, axis=None, dtype='int64'):
    return flow.argmin(x, dim=axis)


def asin(x):
    """
    Returns the index with the smallest value across axes of a tensor.

    Parameters
    ----------
    x : tensor
        Must be one of the following types: bfloat16, half, float32, float64, int8, int16, int32, int64, complex64, complex128.

    Returns
    -------
        A Tensor. Has the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=[10, 20, 5, 6, 15])
    >>> y = tlx.ops.asin(x)

    """

    return flow.asin(x)

# Warps of oneflow functions: asin, asinh, atan, atanh, cos, cosh


def asinh(x):
    return flow.asinh(x)


def atan(x):
    return flow.atan(x)


def atanh(x):
    return flow.atanh(x)


def cos(x):
    return flow.cos(x)


def cosh(x):
    return flow.cosh(x)


def count_nonzero(x, axis=None, keepdims=None, dtype="int64"):
    _nonzero = flow.nonzero(x, as_tuple=True)
    if axis == None:
        return flow.prod(flow.shape(_nonzero[0]))
    x_n = convert_to_numpy(x)
    if isinstance(axis, list):
        axis = tuple(axis)
    non_zero = np.count_nonzero(x_n, axis=axis)
    return convert_to_tensor(non_zero)


def cumprod(x, axis=0, dtype=None, out=None):
    return flow.cumprod(x, dim=axis)


def cumsum(x, axis=0, dtype=None, out=None):
    return flow.cumsum(x, dim=axis)

def equal(x, y):
    return flow.equal(x, y)

def exp(x):
    return flow.exp(x)

def floordiv(x, y):
    return flow.floor_divide(x, y)

def floormod(x, y):
    raise NotImplementedError("floormod is not implemented in oneflow")

def greater(x, y):
    return flow.greater(x, y)

def greater_equal(x, y):
    return flow.greater_equal(x, y)


def is_inf(x):
    return flow.isinf(x)

def is_nan(x):
    return flow.isnan(x)

def l2_normalize(x, axis=None, eps=1e-12):
    axis = 0 if axis is None else axis
    return F.normalize(x, p=2.0, dim=axis, eps=eps)

def less(x, y):
    return flow.lt(x, y)

def less_equal(x, y):
    return flow.le(x, y)

def log(x):
    return flow.log(x)

def log_sigmoid(x):
    return flow.log(1 / (1 + flow.exp(-x)))

def negative(x):
    return flow.negative(x)

def not_equal(x, y):
    return flow.not_equal(x, y)

def pow(x, y):
    return flow.pow(x, y)

def real(x):
    raise NotImplementedError("real is not implemented in oneflow")

def reciprocal(x):
    return flow.reciprocal(x)

def reduce_prod(x, axis=None, keepdims=False):
    if axis is not None:
        return flow.prod(x, dim=axis, keepdim=keepdims)
    else:
        return flow.prod(x)

def reduce_std(x, axis=None, keepdims=False):
    if axis is not None:
        return flow.std(x, dim=axis, keepdim=keepdims)
    else:
        return flow.std(x)

def reduce_sum(x, axis=None, keepdims=False):
    if axis is not None:
        return flow.sum(x, dim=axis, keepdim=keepdims)
    else:
        return flow.sum(x)

def reduce_variance(x, axis=None, keepdims=False):
    if axis is not None:
        return flow.var(x, dim=axis, keepdim=keepdims)
    else:
        return flow.var(x)

def round(x):
    return flow.round(x)

def rsqrt(x):
    return flow.rsqrt(x)


def segment_max(x, segment_ids, num_segments=None):
    segment_ids = flow.Tensor(segment_ids, dtype=flow.int64)
    num_segments = len(flow.unique(segment_ids))

    return unsorted_segment_max(x, segment_ids, num_segments)


def segment_mean(x, segment_ids):
    segment_ids = flow.Tensor(segment_ids, dtype=flow.int64)
    num_segments = len(np.unique(segment_ids.numpy()))
    return unsorted_segment_mean(x, segment_ids, num_segments)


def segment_min(x, segment_ids):
    segment_ids = flow.Tensor(segment_ids, dtype=flow.int64)
    num_segments = len(np.unique(segment_ids.numpy()))
    return unsorted_segment_min(x, segment_ids, num_segments)


def segment_sum(x, segment_ids):
    segment_ids = flow.tensor(segment_ids, dtype=flow.int64)
    num_segments = len(flow.unique(segment_ids))
    return unsorted_segment_sum(x, segment_ids, num_segments)


def segment_prod(x, segment_ids):
    raise NotImplementedError

def sigmoid(x):
    return flow.sigmoid(x)

def sign(x):
    return flow.sign(x)

def sin(x):
    return flow.sin(x)

def sinh(x):
    return flow.sinh(x)

def softplus(x):
    """
    Computes softplus: log(exp(features) + 1).

    Parameters
    ----------
    x : tensor
        Must be one of the following types: half, bfloat16, float32, float64.

    Returns
    -------
        A Tensor. Has the same type as features.
    """

    # Computes softplus: (1/b) * log(1 + exp(features*b)) ; b=1
    return flow.log(1 + flow.exp(x))

def square(x):
    return flow.square(x)

def squared_difference(x, y):
    return flow.square(x - y)

def subtract(x, y):
    return flow.sub(x, y)

def tan(x):
    return flow.tan(x)

def tanh(x):
    """
    Computes hyperbolic tangent of x element-wise.

    Parameters
    ----------
    x : tensor
        Must be one of the following types: bfloat16, half, float32, float64, complex64, complex128.

    Returns
    -------
        A Tensor. Has the same type as x.
    """

    return flow.tanh(x)

def any(x, axis=None, keepdims=False):
    if axis is not None:
        return flow.any(x, dim=axis, keepdim=keepdims)
    else:
        return flow.any(x)

def all(x, axis=None, keepdims=False):
    if axis is not None:
        return flow.all(x, dim=axis, keepdim=keepdims)
    else:
        return flow.all(x)

def logical_and(x, y):
    return flow.logical_and(x, y)

def logical_or(x, y):
    return flow.logical_or(x, y)

def logical_not(x):
    return flow.logical_not(x)

def logical_xor(x, y):
    return flow.logical_xor(x, y)

def argsort(x, axis=-1, descending=False):
    return flow.argsort(x, dim=axis, descending=descending)

def bmm(x, y):
    return flow.bmm(x, y)

def where(condition, x, y):
    return flow.where(condition, x, y)

def ones_like(x, dtype=None):
    return flow.ones_like(x, dtype=dtype)

def zeros_like(x, dtype=None):
    return flow.zeros_like(x, dtype=dtype)

def squeeze(x, axis=None):
    return flow.squeeze(x, dim=axis)

def unsorted_segment_mean(x, segment_ids, num_segments):

    segment_ids = flow.Tensor(segment_ids, dtype=flow.int64)
    assert x.shape[0] == segment_ids.shape[0], "the length of segment_ids should be equal to data.shape[0]."
    if len(segment_ids.shape) == 1:
        s=flow.prod(flow.Tensor(x.shape[1:])).to(flow.int32)
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *x.shape[1:])

    assert x.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(x.shape[1:])
    ones_data = flow.ones_like(x,dtype=x.dtype)
    tensor =  flow.scatter_add(flow.zeros(*shape).to(x.dtype),0,segment_ids, x)
    tensor_nums = flow.scatter_add(flow.zeros(*shape).to(x.dtype),0,segment_ids, ones_data)
    tensor = tensor / tensor_nums
    return tensor


def unsorted_segment_sum(x, segment_ids, num_segments):

    segment_ids = flow.tensor(segment_ids, dtype=flow.int64)
    assert x.shape[0] == segment_ids.shape[0], "the length of segment_ids should be equal to data.shape[0]."
    if len(segment_ids.shape) == 1:
        s = flow.prod(flow.tensor(x.shape[1:])).to(flow.int32)
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *x.shape[1:])

    assert x.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(x.shape[1:])
    tensor = flow.zeros(*shape).to(x.dtype).scatter_add(0, segment_ids, x)
    return tensor


def unsorted_segment_max(x, segment_ids, num_segments):

    segment_ids = flow.Tensor(segment_ids, dtype=flow.int64)

    assert x.shape[0] == segment_ids.shape[0], "the length of segment_ids should be equal to data.shape[0]."
    res = []
    for i in range(num_segments):
        res.append(flow.max(x[segment_ids == i], dim=0)[0])
    return flow.stack(res, dim=0)


def unsorted_segment_min(x, segment_ids, num_segments):

    segment_ids = flow.Tensor(segment_ids, dtype=flow.int64)

    assert x.shape[0] == segment_ids.shape[0], "the length of segment_ids should be equal to data.shape[0]."
    res = []
    for i in range(num_segments):
        res.append(flow.min(x[segment_ids == i], dim=0)[0])
    return flow.stack(res, dim=0)

def set_seed(seed):
    flow.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def is_tensor(obj):
    return isinstance(obj, flow.Tensor)

def tensor_scatter_nd_update(tensor, indices, updates):
    tensor = flow.Tensor(tensor)
    indices = flow.Tensor(indices, dtype=flow.int64)
    updates = flow.Tensor(updates)
    indices = flow.flatten(indices)
    tensor[indices] = updates
    return tensor

def diag(input, diagonal=0):
    return flow.diag(input, diagonal=diagonal)

def mask_select(x, mask, axis = 0):
    if axis is None:
        axis = 0
    if axis < 0:
        axis = len(x.shape) + axis
    if x.shape == mask.shape:
        return flow.masked_select(x, mask)
    if axis == 0:
        return x[mask]
    elif axis == 1:
        return x[:, mask]
    elif axis == 2:
        return x[:, :, mask]
    elif axis == 3:
        return x[:,:,:, mask]

def eye(n, m=None, dtype=flow.float32):
    if m is None:
        m = n
    return flow.eye(n, m, dtype=dtype)

def einsum(equation, *operands):
    return flow.einsum(equation, *operands)

class Einsum(object):
    def __init__(self, equation):
        super(Einsum, self).__init__()
        self.equation = equation

    def __call__(self, *operands):
        return einsum(self.equation, *operands)

def set_device(device = 'GPU', id = 0):
    if device == 'GPU':
        flow.set_default_dtype(flow.float32)
        flow.cuda.set_device(id)

def distributed_init(backend="cncl"):
    raise NotImplementedError("Distributed for this backend is not supported")

def distributed_model(module, device_ids=None, output_device=None, 
                    dim=0, broadcast_buffers=True, process_group=None, bucket_cap_mb=25, 
                    find_unused_parameters=False, check_reduction=False, gradient_as_bucket_view=False):
    raise NotImplementedError("Distributed for this backend is not supported")


def scatter_update(tensor, indices, updates):
    tensor = flow.Tensor(tensor)
    indices = flow.Tensor(indices, dtype=flow.int64)
    updates = flow.Tensor(updates)
    tensor[indices] = updates
    return tensor

def get_device():
    try:
        id = flow.cuda.current_device()
        device = 'GPU:' + str(id)
        return device
    except:
        device = 'CPU'
        return device

def to_device(tensor, device='GPU', id=0):
    device = device.lower()
    if device == 'gpu':
        device = 'cuda' + ':' + str(id)
    tensor = tensor.detach().to(device)
    return tensor

def roll(input, shifts, dims=None):

    return flow.roll(input, shifts, dims)


def logsoftmax(input, dim=None):

    return F.log_softmax(input, dim)


def histogram(input, bins=100, min=0, max=0, name=None):
    raise NotImplementedError


def flatten(x, start_axis=0, stop_axis=-1, name=None):
    raise NotImplementedError


def interpolate(x,
                size=None,
                scale_factor=None,
                mode='nearest',
                align_corners=False,
                align_mode=0,
                data_format='NCHW',
                name=None):
    raise NotImplementedError


def index_select(x, index, axis=0, name=None):
    raise NotImplementedError


def dot(x, y, name=None):
    raise NotImplementedError


class Swish(object):
    def __init__(self):
        pass

    def __call__(self, x):
        raise NotImplementedError

def topk(input, k, dim=-1, largest=True, sorted=True):

    return flow.topk(input, k, dim, largest, sorted)

def numel(input):

    return flow.numel(input)

def expand(x, shape):


    raise NotImplementedError

def unique(x, return_index=False, return_inverse=False, return_counts=False, axis=None, dtype='int64'):

    raise NotImplementedError


def flip(x, axis):

    raise NotImplementedError


def mv(x, vec):

    raise NotImplementedError