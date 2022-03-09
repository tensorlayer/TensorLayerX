#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
from .mindspore_nn import nchw_to_nhwc, nhwc_to_nchw
from .mindspore_nn import preprocess_1d_format, preprocess_2d_format, preprocess_3d_format
from mindspore._c_expression.typing import Type
from mindspore.common import dtype as mstype

from mindspore.common.parameter import Parameter
from mindspore.common.initializer import (
    initializer, Constant, Normal, TruncatedNormal, Initializer, _assignment, _calculate_in_and_out, One, Zero,
    _calculate_fan_in_and_fan_out
)
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
import mindspore.context as context
from mindspore.nn import Cell
import mindspore.numpy as msnp
import mindspore as ms
import numpy as np
from scipy.stats import truncnorm
import random
import math

_dtypeDict = {
    'DType': Type,
    'float16': mstype.float16,
    'float32': mstype.float32,
    'float64': mstype.float64,
    'int8': mstype.int8,
    'int16': mstype.int16,
    'int32': mstype.int32,
    'int64': mstype.int64,
    'uint8': mstype.uint8,
    'uint16': mstype.uint16,
    'uint32': mstype.uint32,
    'uint64': mstype.uint64,
    'bool': mstype.bool_,
    'complex64': None,
    'complex128': None
}

DType = Type
float16 = mstype.float16
float32 = mstype.float32
float64 = mstype.float64
int8 = mstype.int8
int16 = mstype.int16
int32 = mstype.int32
int64 = mstype.int64
uint8 = mstype.uint8
uint16 = mstype.uint16
uint32 = mstype.uint32
uint64 = mstype.uint64
bool = mstype.bool_
complex64 = None
complex128 = None

# isinstance input output
# TensorLike = Tensor_


def set_context(**kwargs):
    return context.set_context(**kwargs)


def get_tensor_shape(x):
    return list(P.Shape()(x))


# initializers
def zeros(shape, dtype=mstype.float32):
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
    # shape = shape[::-1]
    arr = np.ndarray(shape)
    init_obj = Zero()
    init_obj(arr)
    return Tensor(arr, dtype=dtype)


def ones(shape, dtype=mstype.float32):
    """
    Creates a tensor with all elements set to ones.

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
    # shape = shape[::-1]
    arr = np.ndarray(shape)
    init_obj = One()
    init_obj(arr)
    return Tensor(arr, dtype=dtype)


def constant(value, dtype=mstype.float32, shape=None):
    """
    Creates a constant tensor from a tensor-like object.

    Parameters
    ----------
    value : list
        A constant value (or list) of output type dtype.
    dtype : tensor
         The type of the elements of the resulting tensor.
    shape : tuple
        Optional dimensions of resulting tensor.

    Returns
    -------
        A Constant Tensor.

    """
    # shape = shape[::-1]
    arr = np.ndarray(shape)
    Constant(value)(arr=arr)
    return Tensor(arr, dtype=dtype)


class Uniform(Initializer):
    """
    Initialize a uniform array, and obtain values U(-scale, scale) from the uniform distribution
    to fill the input tensor.

    Args:
        minval : int
        The lower bound on the range of random values to generate (inclusive). Defaults to 0.
        maxval : int
        The upper bound on the range of random values to generate (exclusive). Defaults to 1 if dtype is floating point.
        seed : int
         Used in combination with tf.random.set_seed to create a reproducible sequence of tensors across multiple calls.

    Returns:
        Array, uniform array.
    """

    def __init__(self, minval=0, maxval=None, seed=None):
        super(Uniform, self).__init__(minval=minval, maxval=maxval, seed=seed)
        self.minval = minval
        self.maxval = maxval
        self.seed = seed

    def _initialize(self, arr):
        random.seed(self.seed)
        tmp = np.random.uniform(self.minval, self.maxval, arr.shape)
        _assignment(arr, tmp)


def random_uniform(shape, minval=0, maxval=None, dtype=mstype.float32, seed=None):
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
    # shape = shape[::-1]
    arr = np.ndarray(shape)
    init_obj = Uniform(minval=minval, maxval=maxval, seed=seed)
    init_obj(arr)
    return Tensor(arr, dtype=dtype)


class Normal(Initializer):
    """
    Initialize a normal array, and obtain values N(0, sigma) from the uniform distribution
    to fill the input tensor.

    Parameters
    ----------
    mean : float
        The mean of the normal distribution
    stddev : float
        The standard deviation of the normal distribution.
    seed : A Python integer
         Used to create a random seed for the distribution

    Returns:
        Array, normal array.
    """

    def __init__(self, mean=0.0, stddev=0.01, seed=None):
        super(Normal, self).__init__(mean=mean, stddev=stddev)
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def _initialize(self, arr):
        random.seed(self.seed)
        tmp = np.random.normal(self.mean, self.stddev, arr.shape)
        _assignment(arr, tmp)


class RandomNormal(Cell):

    def __init__(self, mean=0.0, stddev=0.01, seed=None):
        super(RandomNormal, self).__init__()
        self.normal = Normal(mean=mean, stddev=stddev, seed=seed)

    def construct(self, shape):
        arr = np.ndarray(shape)
        outputs = self.normal(arr)
        return outputs


def random_normal(shape, mean=0.0, stddev=1.0, dtype=mstype.float32, seed=None):
    """
    Outputs random values from a normal distribution.

    Parameters
    ----------
    shape : tuple
        A 1-D integer Tensor or Python array. The shape of the output tensor.
    mean : float
        The mean of the normal distribution
    stddev : float
        The standard deviation of the normal distribution.
    dtype : tensor
        The type of the output.
    seed : A Python integer
         Used to create a random seed for the distribution

    Returns
    -------
        A tensor of the specified shape filled with random normal values.

    """
    # shape = shape[::-1]
    arr = np.ndarray(shape)
    init_obj = Normal(mean=mean, stddev=stddev, seed=seed)
    init_obj(arr)
    return Tensor(arr, dtype=dtype)


class TruncatedNormal(Initializer):
    """
    Initialize a truncated normal distribution which is a bounded normal distribution within N(low, high).

    Args:
        sigma (float): The sigma of the array. Default: 0.01.

    Returns:
        Array, truncated normal array.
    """

    def __init__(self, mean=0.0, stddev=0.01, seed=None):
        super(TruncatedNormal, self).__init__(mean=mean, stddev=stddev, seed=seed)
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def _initialize(self, arr):
        tmp = truncnorm.rvs(-2, 2, loc=self.mean, scale=self.stddev, size=arr.shape, random_state=None)
        _assignment(arr, tmp)


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=mstype.float32, seed=None):
    """
    Outputs random values from a truncated normal distribution.

    Parameters
    ----------
    shape : tuple
        A 1-D integer Tensor or Python array. The shape of the output tensor.
    mean : float
        The mean of the normal distribution
    stddev : float
        The standard deviation of the normal distribution.
    dtype : tensor
        The type of the output.
    seed : A Python integer
         Used to create a random seed for the distribution

    Returns
    -------
        A tensor of the specified shape filled with random truncated normal values.

    """
    # shape = shape[::-1]
    arr = np.ndarray(shape)
    init_obj = TruncatedNormal(mean=mean, stddev=stddev, seed=seed)
    init_obj(arr)
    return Tensor(arr, dtype=dtype)


class HeNormal(Initializer):
    r"""
    he_normal: It draws samples from a truncated normal distribution centered on 0 with
    stddev = sqrt(2 / fan_in) where fan_in is the number of input units in the weight tensor.

    Args:
        arr (Array): The array to be assigned.

    Returns:
        Array, assigned array.
    """

    def __init__(self, seed=None):
        super(HeNormal, self).__init__(seed=seed)
        self.seed = seed

    def _initialize(self, arr):
        n_in, _ = _calculate_in_and_out(arr)
        boundary = np.sqrt(2.0 / n_in)
        random.seed(self.seed)
        data = np.random.normal(-boundary, boundary, arr.shape)
        _assignment(arr, data)


def he_normal(shape, dtype, seed=None):
    """
    He normal initializer.

    Parameters
    ----------
    seed : A Python integer.
        Used to seed the random generator.
    shape : tuple
        A 1-D integer Tensor or Python array. The shape of the output tensor.
    dtype : tensor
        The type of the output.

    Returns
    -------
        A tensor of the specified shape filled with he normal values.
    """
    # shape = shape[::-1]
    arr = np.ndarray(shape)
    init_obj = HeNormal(seed)
    init_obj(arr)
    return Tensor(arr, dtype=dtype)


class XavierUniform(Initializer):

    def __init__(self, seed=None):
        super(XavierUniform, self).__init__(seed=seed)
        self.seed = seed

    def _initialize(self, arr):
        n_in, n_out = _calculate_fan_in_and_fan_out(arr.shape)
        boundary = math.sqrt(6.0 / (n_in + n_out))
        random.seed(self.seed)
        data = np.random.uniform(-boundary, boundary, arr.shape)
        _assignment(arr, data)


def xavier_uniform(shape, dtype, seed=None):

    arr = np.ndarray(shape)
    init_obj = XavierUniform(seed)
    init_obj(arr)
    return Tensor(arr, dtype=dtype)


class XavierNormal(Initializer):

    def __init__(self, seed=None):
        super(XavierNormal, self).__init__(seed=seed)
        self.seed = seed

    def _initialize(self, arr):
        n_in, n_out = _calculate_fan_in_and_fan_out(arr.shape)
        boundary = math.sqrt(2.0 / (n_in + n_out))
        random.seed(self.seed)
        data = np.random.normal(0, boundary, arr.shape)
        _assignment(arr, data)


def xavier_normal(shape, dtype, seed=None):

    arr = np.ndarray(shape)
    init_obj = XavierNormal(seed)
    init_obj(arr)
    return Tensor(arr, dtype=dtype)


def Variable(initial_value, name, trainable=True):
    """
    Creates a new variable with value initial_value.

    Parameters
    ----------
    initial_value : tensor
        A Tensor, or Python object convertible to a Tensor
    name : str
        Optional name for the variable. Defaults to 'Variable' and gets uniquified automatically.
    Returns
    -------
        Variable
    """

    var = Parameter(initial_value, name=name, requires_grad=trainable)
    return var


class MatMul(Cell):

    def __init__(self):
        super(MatMul, self).__init__()

    def construct(self, a, b):
        return ms.ops.matmul(a, b)


def matmul(a, b):
    """
    Multiplies matrix a by matrix b, producing a * b.

    Parameters
    ----------
    a : tensor
         type float16, float32, float64, int32, complex64, complex128 and rank > 1.
    b : tensor
        with same type and rank as a.

    Returns
    -------
        A Tensor of the same type as a and b
    """

    outputs = ms.ops.matmul(a, b)
    return outputs


def add(value, bias):
    """
    Returns x + y element-wise.

    Parameters
    ----------
    value :  tensor.
        Must be one of the following types: bfloat16, half, float32, float64,
        uint8, int8, int16, int32, int64, complex64, complex128, string.
    bias : tensor
        Must have the same type as a
    name : str
        A name for the operation

    Returns
    -------
        A Tensor. Has the same type as a.
    """

    add_obj = P.Add()
    outputs = add_obj(value, bias)
    return outputs


def dtypes(dt):
    """
    Data dtypes.

    Parameters
    ----------
    dt : string
         It could be 'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16',
         'int32', 'int64', 'float16', 'float32', 'float64', 'DType'.

    Returns
    -------
        Data dtypes
    """

    if dt not in _dtypeDict.keys():
        raise Exception("Unsupported dtype: {}".format(dt))
    return _dtypeDict[dt]


class Maximum(Cell):

    def __init__(self):
        super(Maximum, self).__init__()
        self.maximum = P.Maximum()

    def construct(self, x, y):
        return self.maximum(x, y)


class Minimum(Cell):

    def __init__(self):
        super(Minimum, self).__init__()
        self.minimum = P.Minimum()

    def construct(self, x, y):
        return self.minimum(x, y)


def minimum(x, y):
    """
    Returns the min of x and y (i.e. x < y ? x : y) element-wise.

    Parameters
    ----------
    x : tensor.
        Must be one of the following types: bfloat16, half, float32, float64, int32, int64.
    y : A Tensor.
        Must have the same type as x.
    name : str
        A name for the operation (optional).

    Returns
    -------
        A Tensor. Has the same type as x
    """
    minimum_obj = P.Minimum()
    outputs = minimum_obj(x, y)
    return outputs


class FlattenReshape(Cell):

    def __init__(self):
        super(FlattenReshape, self).__init__()
        self.shape = P.Shape()
        self.reshape = P.Reshape()

    def construct(self, inputs):
        dim = 1
        for d in self.shape(inputs)[1:]:
            dim *= d
        return self.reshape(inputs, (-1, dim))


class Reshape(Cell):

    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.reshape = P.Reshape()
        self.shape = tuple(shape)

    def construct(self, tensor):
        return self.reshape(tensor, self.shape)


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
    reshape_obj = P.Reshape()
    outputs = reshape_obj(tensor, tuple(shape))
    return outputs


class Concat(Cell):

    def __init__(self, axis):
        super(Concat, self).__init__()
        self.concat = P.Concat(axis)

    def construct(self, values):
        return self.concat(values)


def concat(values, axis):
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
    # TODO testing axis
    concat_obj = P.Concat(axis)
    outputs = concat_obj(values)
    return outputs


def convert_to_tensor(value, dtype=None):
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
    return Tensor(value, dtype=dtype)


def convert_to_numpy(value):
    return value.asnumpy()


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
    sqrt_obj = P.Sqrt()
    outputs = sqrt_obj(x)
    return outputs


class ReduceSum(Cell):

    def __init__(self, axis):
        super(ReduceSum, self).__init__()
        self.axis = axis
        self.reduce_sum = P.ReduceSum(keep_dims=False)

    def construct(self, input):
        return self.reduce_sum(input, self.axis)


class ReduceMean(Cell):

    def __init__(self, axis, keepdims=False):
        super(ReduceMean, self).__init__()
        self.axis = axis
        self.reducemean = P.ReduceMean(keep_dims=keepdims)

    def construct(self, inputs):
        output = self.reducemean(inputs, self.axis)
        return output


def reduce_mean(input_tensor, axis=None, keepdims=False):
    """
    Computes the mean of elements across dimensions of a tensor.

    Parameters
    ----------
    input_tensor : tensor
        The tensor to reduce. Should have numeric type.
    axis : int
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range [-rank(input_tensor), rank(input_tensor)).
    name : str
        A name for the operation (optional).

    Returns
    -------
        The reduced tensor.
    """

    Rmean_obj = P.ReduceMean(keep_dims=keepdims)
    outputs = Rmean_obj(input_tensor, axis)
    return outputs


class ReduceMax(Cell):

    def __init__(self, axis=None, keepdims=False):
        super(ReduceMax, self).__init__()
        self.axis = axis
        self.reducemax = P.ReduceMax(keep_dims=keepdims)

    def construct(self, inputs):
        output = self.reducemax(inputs, self.axis)
        return output


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

    Rmax_obj = P.ReduceMax(keep_dims=keepdims)
    outputs = Rmax_obj(input_tensor, axis)
    return outputs


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

    Rmin_obj = P.ReduceMin(keep_dims=keepdims)
    outputs = Rmin_obj(input_tensor, axis)
    return outputs


class Pad(Cell):

    def __init__(self, paddings, mode="REFLECT", constant_values=0):
        super(Pad, self).__init__()
        if mode not in ['CONSTANT', 'REFLECT', 'SYMMETRIC']:
            raise Exception("Unsupported mode: {}".format(mode))
        if mode == 'CONSTANT':
            self.pad = P.Pad(tuple(paddings))
            if constant_values - 0 == 0:
                pass
            else:
                raise NotImplementedError("constant_values can only be equal to 0.")
        else:
            self.pad = P.MirrorPad(mode=mode)
            self.paddings = Tensor(np.array(paddings))
        self.mode = mode

    def construct(self, x):
        if self.mode == 'CONSTANT':
            return self.pad(x)
        else:
            return self.pad(x, self.paddings)


def pad(tensor, paddings, mode='CONSTANT', constant_values=0):
    """
    Pads a tensor.

    Parameters
    ----------
    tensor : tensor
        A Tensor.
    paddings : tuple
        A tuple of type int32.
    mode : str
        One of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive)
    constant_values : int
        In "CONSTANT" mode, the scalar pad value to use. Must be same type as tensor.

    Returns
    -------
        A Tensor. Has the same type as tensor.
    """
    raise NotImplementedError


class Unstack(Cell):

    def __init__(self, axis, num=None):
        super(Unstack, self).__init__()
        if num is not None:
            raise ("The num Parameters do not need to be set.")
        self.unstack = P.Unstack(axis=axis)

    def construct(self, values):
        return list(self.unstack(values))


class Stack(Cell):

    def __init__(self, axis=0):
        super(Stack, self).__init__()
        self.stack = P.Stack(axis=axis)

    def construct(self, values):
        return self.stack(values)


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
    _stack = P.Pack(axis=axis)
    return _stack(values)


class Meshgrid(Cell):

    def __init__(self, indexing='xy'):
        super(Meshgrid, self).__init__()
        self._meshgrid = P.Meshgrid(indexing=indexing)

    def construct(self, *args):
        inputs = tuple(*args)
        return self._meshgrid(inputs)


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

    _meshgrid = P.Meshgrid(**kwargs)
    return _meshgrid(*args)


def range(start, limit=None, delta=1, dtype=None):
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
    dtype : type
        The type of the elements of the resulting tensor.

    Returns
    -------
        An 1-D Tensor of type dtype.
    """

    pass


class ExpandDims(Cell):

    def __init__(self, axis):
        super(ExpandDims, self).__init__()
        self.axis = axis
        self.expand_dims = P.ExpandDims()

    def construct(self, input):
        output = self.expand_dims(input, self.axis)
        return output


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

    expand_obj = P.ExpandDims()
    outputs = expand_obj(input, axis)
    return outputs


class Tile(Cell):

    def __init__(self):
        super(Tile, self).__init__()
        self.tile = P.Tile()

    def construct(self, input, multiples):
        return self.tile(input, tuple(multiples))


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
    tile_obj = P.Tile()
    outputs = tile_obj(input, multiples)
    return outputs


class Cast(Cell):

    def __init__(self, dtype):
        super(Cast, self).__init__()
        self.dtype = dtype
        self.cast = P.Cast()

    def construct(self, input):
        return self.cast(input, self.dtype)


def cast(x, dtype):
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
    cast_obj = P.Cast()
    outputs = cast_obj(x, dtype)
    return outputs


class Transpose(Cell):

    def __init__(self, perm, conjugate=False):
        super(Transpose, self).__init__()
        self.perm = tuple(perm)
        self.conjugate = conjugate
        self.transpose = P.Transpose()
        if self.conjugate:
            raise NotImplementedError("conjugate not implemented")

    def construct(self, a):
        return self.transpose(a, self.perm)


def transpose(a, perm=None, conjugate=False):
    """
    Transposes a.

    Parameters
    ----------
    a : tensor
        A Tensor.
    perm : int
        A permutation of the dimensions of a.
    conjugate : bool
        Setting it to True is mathematically equivalent to ms.math.conj(ms.transpose(input)).

    Returns
    -------
        A transposed Tensor.
    """
    # TODO conjugate
    trans_obj = P.Transpose()
    outputs = trans_obj(a, perm)
    print(outputs)


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

    pass


class ClipGradByValue(object):
    def __init__(self, clip_min=-1, clip_max=1):
        self.min = ms.Tensor(clip_min)
        self.max = ms.Tensor(clip_max)

    def __call__(self, inputs):
        return ms.ops.clip_by_value(inputs, clip_value_max=self.max, clip_value_min=self.min)


class ClipGradByNorm(object):
    def __init__(self, clip_norm=0.1):
        self.clip_norm = clip_norm
        self.clip_by_norm = ms.nn.ClipByNorm()

    def __call__(self, inputs):
        return self.clip_by_norm(inputs, self.clip_norm)


class ClipByGlobalNorm(object):
    def __init__(self, clip_norm):
        self.clip_norm = clip_norm

    def __call__(self, inputs):
        return ms.ops.clip_by_global_norm(inputs, clip_norm=self.clip_norm)


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
    min_value = Tensor(clip_value_min, mstype.float32)
    max_value = Tensor(clip_value_max, mstype.float32)
    output = C.clip_by_value(t, min_value, max_value)
    return output


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
    return msnp.split(value, indices_or_sections=num_or_size_splits, axis=axis)


class Floor(Cell):

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


def floor(x):

    op = P.Floor()
    return op(x)


def gather(params, indices, axis=None):
    op = P.Gather()
    return op(params, indices, axis)


def linspace(start, stop, num):
    return NotImplementedError


def slice(inputs, starts, sizes):
    return NotImplementedError


def add_n(inputs):
    return NotImplementedError


class OneHot(Cell):

    def __init__(self, axis=-1, depth=1, on_value=1.0, off_value=0.0, dtype=mstype.float32):
        super(OneHot, self).__init__()
        self.onehot = P.OneHot(axis)
        self.depth = depth
        self.dtype = dtype
        self.on_value = F.cast(on_value, self.dtype)
        self.off_value = F.cast(off_value, self.dtype)

    def construct(self, indices):
        return self.onehot(indices, self.depth, self.on_value, self.off_value)


class L2Normalize(Cell):

    def __init__(self, axis=None, epsilon=1e-12):
        super(L2Normalize, self).__init__()
        pass

    def construct(self, input, *args, **kwargs):
        pass


class EmbeddingLookup(Cell):

    def __init__(self, max_norm=0):
        super(EmbeddingLookup, self).__init__()
        self.max_norm = max_norm
        self.embedding_lookup = P.EmbeddingLookup()

    def construct(self, params, ids, *args, **kwargs):
        return self.embedding_lookup(params, ids, self.max_norm)


class NCELoss(Cell):

    def __init__(self, num_true=1, sampled_values=None, remove_accidental_hits=False):
        super(NCELoss, self).__init__()
        pass

    def construct(self, weights, biases, labels, inputs, num_sampled, num_classes):
        raise NotImplementedError


class NotEqual(Cell):

    def __init__(self):
        super(NotEqual, self).__init__()
        self.not_equal = P.NotEqual()

    def construct(self, x, y):
        outputs = self.not_equal(x, y)
        return outputs


class CountNonzero(object):

    def __init__(self, keepdims=None, dtype=int64):
        self.keepdims = keepdims
        self.dtype = dtype

    def __call__(self, input, axis=None):
        input = self.convert_dtype(input)
        return ms.ops.count_nonzero(x=input, axis=axis, keep_dims=self.keepdims, dtype=self.dtype)

    def bool_convert_to_tensor(self, x):
        x = x.asnumpy()
        shapes = x.shape
        b = np.ones(shapes)
        if len(shapes) == 1:
            for i in range(shapes - 1):
                if x[i] ==True:
                    b[i] = 1
                else:
                    b[i] = 0
        if len(shapes) == 2:
            for i in range(shapes[0] - 1):
                for j in range(shapes[1] - 1):
                    if x[i][j] ==True:
                        b[i][j] = 1
                    else:
                        b[i][j] = 0
        return Tensor(b, dtype=float32)

    def convert_dtype(self, input):
        if input.shape == 1 and type(input[0]) is bool:
            output = self.bool_convert_to_tensor(input)
        elif input.shape == 2 and type(input[0][0]) is bool:
            output = self.bool_convert_to_tensor(input)
        else:
            output = input
        return output


class Resize(Cell):

    def __init__(self, scale, method, antialias=False, data_format='channels_last'):
        super(Resize, self).__init__()
        self.data_format = data_format
        if method not in ['nearest', 'bilinear']:
            raise ('The method must be "nearest" or "bilinear".')
        self.method = method
        self.antialias = antialias
        self.init_instance = False
        self.scale = scale

    def _initializing_instance(self, out_size):
        if self.method == 'nearest':
            self.resize = P.ResizeNearestNeighbor(size=out_size, align_corners=self.antialias)
        elif self.method == 'bilinear':
            self.resize = P.ResizeBilinear(size=out_size)

    def construct(self, inputs):
        if self.init_instance:
            self._initializing_instance(out_size=self.out_size(inputs))
            self.init_instance = True
        if self.data_format == 'channels_last':
            inputs = nhwc_to_nchw(inputs)
        outputs = self.resize(inputs)
        if self.data_format == 'channels_last':
            outputs = nchw_to_nhwc(outputs)
        return outputs

    def out_size(self, inputs):
        output_size = [int(inputs.shape[1] * self.scale[0]), int(inputs.shape[2] * self.scale[1])]
        return output_size


def resize(inputs, output_size, method, antialias):
    raise NotImplementedError


class ZeroPadding1D(Cell):

    def __init__(self, padding):
        super(ZeroPadding1D, self).__init__()
        padding = ((0, 0), padding, (0, 0))
        self.pad = P.Pad(paddings=padding)

    def construct(self, inputs):
        if len(inputs.shape) == 2:
            raise NotImplementedError("ZeroPadding1D inputs must be 3D.")
        return self.pad(inputs)


class ZeroPadding2D(Cell):

    def __init__(self, padding):
        super(ZeroPadding2D, self).__init__()
        padding = ((0, 0), padding[0], padding[1], (0, 0))
        self.pad = P.Pad(paddings=padding)

    def construct(self, inputs):
        return self.pad(inputs)


class ZeroPadding3D(Cell):

    def __init__(self, padding):
        super(ZeroPadding3D, self).__init__()
        padding = ((0, 0), padding[0], padding[1], padding[2], (0, 0))
        self.pad = P.Pad(paddings=padding)

    def construct(self, inputs):
        return self.pad(inputs)


class Sign(Cell):

    def __init__(self):
        super(Sign, self).__init__()
        self.sign = P.Sign()

    def construct(self, x):
        return self.sign(x)


class Ceil(Cell):

    def __init__(self):
        super(Ceil, self).__init__()
        self.ceil = P.Ceil()

    def construct(self, x):
        return self.ceil(x)


def ceil(x):
    _ceil = P.Ceil()
    return _ceil(x)


def multiply(x, y):
    return ms.numpy.multiply(x, y)


def divide(x, y):
    return msnp.divide(x, y)


def identity(x):
    return ms.numpy.identity(x)


class BatchToSpace(Cell):

    def __init__(self, block_size, crops):
        super(BatchToSpace, self).__init__()
        self.batch_to_space = P.BatchToSpace(block_size=block_size, crops=crops)

    def __call__(self, input_x):
        return self.batch_to_space(input_x)


class DepthToSpace(Cell):

    def __init__(self, block_size, data_format='NHWC'):
        super(DepthToSpace, self).__init__()
        self.data_format, _ = preprocess_2d_format(data_format, None)
        self.depth_to_space = P.DepthToSpace(block_size=block_size)

    def __call__(self, input):
        if self.data_format == 'NHWC':
            input = nhwc_to_nchw(input)

        output = self.depth_to_space(input)

        if self.data_format == 'NHWC':
            output = nchw_to_nhwc(output)

        return output


def triu(data, diagonal=0):

    return msnp.triu(data, k=diagonal)


def tril(data, diagonal=0):

    return msnp.tril(data, k=diagonal)


def abs(x):

    return ms.numpy.abs(x)


def acos(x):
    _acos = ms.ops.ACos()
    return _acos(x)


def acosh(x):
    _acosh = ms.ops.Acosh()
    return _acosh(x)


def angle(x):
    x_np = convert_to_numpy(x)
    return convert_to_tensor(np.angle(x_np))


def argmax(x, axis=None, dtype='int64'):
    return ms.numpy.argmax(x, axis=axis)


def argmin(x, axis=None, dtype='int64'):
    return ms.numpy.argmin(x, axis=axis)


def asin(x):
    _asin = ms.ops.Asin()
    return _asin(x)


def asinh(x):
    _asinh = ms.ops.Asinh()
    return _asinh(x)


def atan(x):
    _atan = ms.ops.Atan()
    return _atan(x)


def atanh(x):
    _atanh = ms.ops.Atanh()
    return _atanh(x)


def cos(x):
    return ms.numpy.cos(x)


def cosh(x):
    return ms.numpy.cosh(x)


def count_nonzero(x, axis=None, keepdims=False, dtype="int64"):
    return ms.numpy.count_nonzero(x, axis=axis, keepdims=keepdims)


def cumprod(x, axis=0, exclusive=False, reverse=False):
    return ms.numpy.cumprod(x, axis=axis)


def cumsum(x, axis=0, exclusive=False, reverse=False):
    return ms.numpy.cumsum(x, axis=axis)


def equal(x, y):
    return ms.numpy.equal(x, y)


def exp(x):
    return ms.numpy.exp(x)


def floordiv(x, y):
    return ms.numpy.floor_divide(x, y)


def floormod(x, y):
    _floormod = ms.ops.FloorMod()
    return _floormod(x, y)


def greater(x, y):
    return ms.numpy.greater(x, y)


def greater_equal(x, y):
    return ms.numpy.greater_equal(x, y)


def is_inf(x):
    return ms.numpy.isinf(x)


def is_nan(x):
    return ms.numpy.isnan(x)


def l2_normalize(x, axis=None, eps=1e-12):
    _l2_normalize = ms.ops.L2Normalize(axis=axis, epsilon=eps)
    return _l2_normalize(x)


def less(x, y):
    return ms.numpy.less(x, y)


def less_equal(x, y):
    return ms.numpy.less_equal(x, y)


def log(x):
    return ms.numpy.log(x)


def log_sigmoid(x):
    _log_sigmoid = ms.nn.LogSigmoid()
    return _log_sigmoid(x)


def maximum(x, y):
    return ms.numpy.maximum(x, y)


def negative(x):
    return ms.numpy.negative(x)


def not_equal(x, y):
    return ms.numpy.not_equal(x, y)


def pow(x, y):
    _pow = ms.ops.Pow()
    return _pow(x, y)


def real(x):
    _real = ms.ops.Real()
    return _real(x)


def reciprocal(x):
    return ms.numpy.reciprocal(x)


def reduce_prod(x, axis=None, keepdims=False):
    op = P.ReduceProd(keep_dims=keepdims)
    return op(x, axis=axis)


def reduce_std(x, axis=None, keepdims=False):
    return msnp.std(x, axis=axis, keepdims=keepdims)


def reduce_sum(x, axis=None, keepdims=False):
    op = P.ReduceSum(keep_dims=keepdims)
    return op(x, axis=axis)


def reduce_variance(x, axis=None, keepdims=False):
    return ms.numpy.var(x, axis=axis, keepdims=keepdims)


def round(x):
    x = convert_to_tensor(x)
    op = P.Round()
    return op(x)


def rsqrt(x):
    x = convert_to_tensor(x, ms.float32)
    op = P.Rsqrt()
    return op(x)


def segment_max(x, segment_ids):
    segment_ids = convert_to_tensor(segment_ids, ms.int32)
    op = P.UnsortedSegmentMax()
    unique = P.Unique()
    num_segments = len(unique(segment_ids))
    return op(x, segment_ids, num_segments)


def segment_mean(x, segment_ids):
    raise NotImplementedError


def segment_min(x, segment_ids):
    segment_ids = convert_to_tensor(segment_ids, ms.int32)
    op = P.UnsortedSegmentMin()
    unique = P.Unique()
    num_segments = len(unique(segment_ids))
    return op(x, segment_ids, num_segments)


def segment_prod(x, segment_ids):
    segment_ids = convert_to_tensor(segment_ids, ms.int32)
    op = P.UnsortedSegmentProd()
    unique = P.Unique()
    num_segments = len(unique(segment_ids))
    return op(x, segment_ids, num_segments)


def segment_sum(x, segment_ids):
    segment_ids = convert_to_tensor(segment_ids, ms.int32)
    op = P.UnsortedSegmentSum()
    unique = P.Unique()
    num_segments = len(unique(segment_ids))
    return op(x, segment_ids, num_segments)


def sigmoid(x):
    op = P.Sigmoid()
    return op(x)


def sign(x):
    op = P.Sign()
    return op(x)


def sin(x):
    op = P.Sin()
    return op(x)


def sinh(x):
    op = P.Sinh()
    return op(x)


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

    obj = ms.ops.Softplus()
    return obj(x)


def square(x):
    op = P.Square()
    return op(x)


def squared_difference(x, y):
    op = P.SquaredDifference()
    return op(x, y)


def subtract(x, y):
    op = P.Sub()
    return op(x, y)


def tan(x):
    op = P.Tan()
    return op(x)


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

    _tanh = ms.ops.Tanh()
    return _tanh(x)


def any(x, axis=None, keepdims=False):
    op = P.ReduceAny(keep_dims=keepdims)
    return op(x, axis)


def all(x, axis=None, keepdims=False):
    op = P.ReduceAll(keep_dims=keepdims)
    return op(x, axis)


def logical_and(x, y):
    op = P.LogicalAnd()
    return op(x, y)


def logical_or(x, y):
    op = P.LogicalOr()
    return op(x, y)


def logical_not(x):
    op = P.LogicalNot()
    return op(x)


def logical_xor(x, y):
    return msnp.logical_xor(x, y)


def argsort(x, axis=-1, descending=False):
    op = P.Sort(axis, descending)
    _, index = op(x)
    return index


def bmm(x, y):
    return ms.ops.matmul(x, y)


def where(condition, x, y):
    return msnp.where(condition, x, y)


def ones_like(x, dtype=None):
    return msnp.ones_like(x, dtype=dtype)


def zeros_like(x, dtype=None):
    return msnp.zeros_like(x, dtype=dtype)


def squeeze(x, axis=None):
    op = P.Squeeze(axis)
    return op(x)


def unsorted_segment_sum(x, segment_ids, num_segments):
    op = P.UnsortedSegmentSum()
    return op(x, segment_ids, num_segments)


def unsorted_segment_mean(x, segment_ids, num_segments):
    raise NotImplementedError


def unsorted_segment_min(x, segment_ids, num_segments):
    op = P.UnsortedSegmentMin()
    return op(x, segment_ids, num_segments)


def unsorted_segment_max(x, segment_ids, num_segments):
    op = P.UnsortedSegmentMax()
    return op(x, segment_ids, num_segments)
