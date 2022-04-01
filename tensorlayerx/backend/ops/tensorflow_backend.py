#! /usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from .tensorflow_nn import nchw_to_nhwc, nhwc_to_nchw, preprocess_1d_format, preprocess_2d_format, preprocess_3d_format
import tensorflow as tf
import random
import numpy as np

_dtypeDict = {
    'DType': tf.DType,
    'float16': tf.float16,
    'float32': tf.float32,
    'float64': tf.float64,
    'int8': tf.int8,
    'int16': tf.int16,
    'int32': tf.int32,
    'int64': tf.int64,
    'uint8': tf.uint8,
    'uint16': tf.uint16,
    'uint32': tf.uint32,
    'uint64': tf.uint64,
    'bool': tf.bool,
    'complex64': tf.complex64,
    'complex128': tf.complex128
}

DType = tf.DType
float16 = tf.float16
float32 = tf.float32
float64 = tf.float64
int8 = tf.int8
int16 = tf.int16
int32 = tf.int32
int64 = tf.int64
uint8 = tf.uint8
uint16 = tf.uint16
uint32 = tf.uint32
uint64 = tf.uint64
bool = tf.bool
complex64 = tf.complex64
complex128 = tf.complex128

# isinstance input output
# TensorLike = tf_ops._TensorLike


def dtype_str(x):
    if isinstance(x, str):
        if x in list(_dtypeDict.keys()):
            return _dtypeDict[x]
        else:
            raise NotImplemented("The input data type is incorrect.")
    else:
        return x


def set_context(**kwargs):
    raise Exception("Using TenosrFlow backend,You don't need to set context")


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

    return x.get_shape().as_list()


# initializers
def zeros(shape, dtype='float32'):
    """
    Creates a tensor with all elements set to zero.

    Parameters
    ----------
    shape : A list of integers
        a tuple of integers, or a 1-D Tensor of type int32.
    dtype : tensor or str
        The DType of an element in the resulting Tensor

    Returns
    -------
        A Tensor with all elements set to zero.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.zeros((32, 3, 3, 32), dtype=tlx.int32)
    >>> y = tlx.ops.zeros((10, 25, 25, 10), dtype='float32')

    """

    return tf.zeros(shape=shape, dtype=dtype_str(dtype))


def ones(shape, dtype='float32'):
    """
    Creates a tensor with all elements set to ones.

    Parameters
    ----------
    shape : A list of integers
        a tuple of integers, or a 1-D Tensor of type int32.
    dtype : tensor or str
        The DType of an element in the resulting Tensor

    Returns
    -------
        A Tensor with all elements set to zero.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.ones((32, 3, 3, 32), dtype=tlx.int32)
    >>> y = tlx.ops.ones((10, 25, 25, 10), dtype='float32')

    """

    return tf.ones(shape=shape, dtype=dtype_str(dtype))


def constant(value, dtype='float32', shape=None):
    """
    Creates a constant tensor from a tensor-like object.

    Parameters
    ----------
    value : list
        A constant value (or list) of output type dtype.
    dtype : tensor or str
         The type of the elements of the resulting tensor.
    shape : tuple
        Optional dimensions of resulting tensor.

    Returns
    -------
        A Constant Tensor.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(0.5, (32, 3, 3, 32), dtype=tlx.int32)
    >>> y = tlx.ops.constant(0.5, (10, 25, 25, 10), dtype='float32')

    """

    return tf.constant(value=value, dtype=dtype_str(dtype), shape=shape)


def random_uniform(shape, minval=0, maxval=None, dtype='float32', seed=None):
    """
    Outputs random values from a uniform distribution.

    Parameters
    ----------
    shape : tuple
        A 1-D integer Tensor or Python array. The shape of the output tensor.
    minval : float
        The lower bound on the range of random values to generate (inclusive). Defaults to 0.
    maxval : float
        The upper bound on the range of random values to generate (exclusive). Defaults to 1 if dtype is floating point.
    dtype : tensor or str
        The type of the output: float16, float32, float64, int32, or int64.
    seed : int
         Used in combination with tf.random.set_seed to create a reproducible sequence of tensors across multiple calls.
    Returns
    -------
        A tensor of the specified shape filled with random uniform values.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.random_uniform((32, 3, 3, 32), maxval=1.0, dtype=tlx.int32)
    >>> y = tlx.ops.random_uniform((10, 25, 25, 10), maxval=1.0, dtype='float32')

    """

    outputs = tf.random.uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype_str(dtype), seed=seed)
    return outputs


def random_normal(shape, mean=0.0, stddev=1.0, dtype='float32', seed=None):
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
    dtype : tensor or str
        The type of the output.
    seed : A Python integer
         Used to create a random seed for the distribution

    Returns
    -------
        A tensor of the specified shape filled with random normal values.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.random_normal((32, 3, 3, 32), dtype=tlx.int32)
    >>> y = tlx.ops.random_normal((10, 25, 25, 10), dtype='float32')

    """

    outputs = tf.random.normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype_str(dtype), seed=seed)
    return outputs


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype='float32', seed=None):
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
    dtype : tensor or str
        The type of the output.
    seed : A Python integer
         Used to create a random seed for the distribution

    Returns
    -------
        A tensor of the specified shape filled with random truncated normal values.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.truncated_normal((32, 3, 3, 32), dtype=tlx.int32)
    >>> y = tlx.ops.truncated_normal((10, 25, 25, 10), dtype='float32')

    """

    outputs = tf.random.truncated_normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype_str(dtype), seed=seed)
    return outputs


def he_normal(shape, dtype='float32', seed=None):
    """
    He normal initializer.

    Parameters
    ----------
    seed : A Python integer.
        Used to seed the random generator.
    shape : tuple
        A 1-D integer Tensor or Python array. The shape of the output tensor.
    dtype : tensor or str
        The type of the output.

    Returns
    -------
        A tensor of the specified shape filled with he normal values.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.he_normal((32, 3, 3, 32), dtype=tlx.int32)
    >>> y = tlx.ops.he_normal((10, 25, 25, 10), dtype='float32')

    """

    return tf.initializers.he_normal(seed)(shape=shape, dtype=dtype_str(dtype))


def xavier_normal(shape, dtype='float32', seed=None):
    """
    Xavier normal.

    Parameters
    ----------
    seed : A Python integer.
        Used to seed the random generator.
    shape : tuple
        A 1-D integer Tensor or Python array. The shape of the output tensor.
    dtype : tensor or str
        The type of the output.

    Returns
    -------
        A tensor of the specified shape filled with xavier normal values.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.xavier_normal((32, 3, 3, 32), dtype=tlx.int32)
    >>> y = tlx.ops.xavier_normal((10, 25, 25, 10), dtype='float32')

    """

    return tf.initializers.glorot_normal(seed)(shape=shape, dtype=dtype_str(dtype))


def xavier_uniform(shape, dtype='float32', seed=None):
    """
    Xavier uniform.

    Parameters
    ----------
    seed : A Python integer.
        Used to seed the random generator.
    shape : tuple
        A 1-D integer Tensor or Python array. The shape of the output tensor.
    dtype : tensor or str
        The type of the output.

    Returns
    -------
        A tensor of the specified shape filled with xavier uniform values.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.xavier_uniform((32, 3, 3, 32), dtype=tlx.int32)
    >>> y = tlx.ops.xavier_uniform((10, 25, 25, 10), dtype='float32')

    """

    return tf.initializers.glorot_uniform(seed)(shape=shape, dtype=dtype_str(dtype))


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

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.Variable(tlx.ops.ones(shape=(10, 20)), name='w')

    """

    var = tf.Variable(initial_value=initial_value, name=name, trainable=trainable)
    return var


class MatMul(object):

    def __init__(self, transpose_a=False, transpose_b=False):
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b

    def __call__(self, a, b):
        return tf.matmul(a, b, transpose_a=self.transpose_a, transpose_b=self.transpose_b)


def matmul(a, b, transpose_a=False, transpose_b=False):
    """
    Multiplies matrix a by matrix b, producing a * b.

    Parameters
    ----------
    a : tensor
         type float16, float32, float64, int32, complex64, complex128 and rank > 1.
    b : tensor
        with same type and rank as a.
    transpose_a : boolean
        If True, a is transposed before multiplication.
    transpose_b : boolean
        If True, b is transposed before multiplication.

    Returns
    -------
        A Tensor of the same type as a and b

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.convert_to_tensor(np.random.random([2,3,2]), dtype="float32")
    >>> y = tlx.convert_to_tensor(np.random.random([2,2,3]), dtype="float32")
    >>> z = tlx.ops.matmul(x, y)
    >>> print(z.shape)
    >>> [2,3,3]
    """

    outputs = tf.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)
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

    Returns
    -------
        A Tensor. Has the same type as a.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> value = tlx.ones(shape=(10, 20))
    >>> bias = tlx.ones(shape=(20))
    >>> x = tlx.ops.add(value, bias)

    """

    outputs = tf.add(value, bias)
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


class Maximum(object):

    def __init__(self):
        pass

    def __call__(self, x, y):
        return tf.maximum(x=x, y=y)


class Minimum(object):

    def __init__(self):
        pass

    def __call__(self, x, y):
        return tf.minimum(x=x, y=y)


def minimum(x, y):
    """
    Returns the min of x and y (i.e. x < y ? x : y) element-wise.

    Parameters
    ----------
    x : tensor.
        Must be one of the following types: bfloat16, half, float32, float64, int32, int64.
    y : A Tensor.
        Must have the same type as x.

    Returns
    -------
        A Tensor. Has the same type as x

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant([0., 0., 0., 0.])
    >>> y = tlx.ops.constant([-5., -2., 0., 3.])
    >>> z = tlx.ops.minimum(x, y)

    """

    return tf.minimum(x=x, y=y)


class FlattenReshape(object):

    def __init__(self):
        pass

    def __call__(self, inputs):
        dim = 1
        for d in get_tensor_shape(inputs)[1:]:
            dim *= d
        return tf.reshape(inputs, [-1, dim])


class Reshape(object):

    def __init__(self, shape):
        self.shape = shape

    def __call__(self, tensor):
        return tf.reshape(tensor, self.shape)


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

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant([0., 1., 2., 3.])
    >>> z = tlx.ops.reshape(x, [2, 2])

    """

    return tf.reshape(tensor, shape)


class Concat(object):

    def __init__(self, axis):
        super(Concat, self).__init__()
        self.axis = axis

    def __call__(self, values):
        return tf.concat(values=values, axis=self.axis)


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

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant([0., 0., 0., 0.])
    >>> y = tlx.ops.constant([-5., -2., 0., 3.])
    >>> z = tlx.ops.concat([x, y], 0)

    """

    return tf.concat(values, axis)


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

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = np.ones(shape=(10, 10))
    >>> y = tlx.ops.convert_to_tensor(x)

    """

    return tf.convert_to_tensor(value, dtype)


def convert_to_numpy(value):
    """
    Converts the given Tensor to a numpy.

    Parameters
    ----------
    value : object
        An object whose type has a registered Tensor conversion function.

    Returns
    -------
        A value based on tensor.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.ones(shape=(10, 10))
    >>> y = tlx.ops.convert_to_numpy(x)

    """

    return value.numpy()


def sqrt(x):
    """
    Computes square root of a tensor element-wise.

    Parameters
    ----------
    x : tensor
         Must be one of the following types: bfloat16, half, float32, float64, complex64, complex128.

    Returns
    -------
        A Tensor. Has the same type as x.


    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.ops.convert_to_tensor(np.array([0.0, 1.0, 4.0]), dtype=tlx.float32)
    >>> x = tlx.ops.sqrt(x)
    >>> print(x)
    >>> [0.0, 1.0, 2.0]

    """
    return tf.sqrt(x)


class ReduceSum(object):

    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def __call__(self, input):
        return tf.reduce_sum(input, axis=self.axis, keepdims=self.keepdims)


class ReduceMean(object):

    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def __call__(self, inputs):
        output = tf.reduce_mean(inputs, self.axis, keepdims=self.keepdims)
        return output


def reduce_mean(input_tensor, axis=None, keepdims=False):
    """
    Computes the mean of elements across dimensions of a tensor.

    Parameters
    ----------
    x : tensor
        The tensor to reduce. Should have numeric type.
    axis : list
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range [-rank(x), rank(x)).
    keepdims : boolean
        If true, keep these reduced dimensions and the length is 1. If false, don’t keep these dimensions.
        Default : False, don’t keep these reduced dimensions.

    Returns
    -------
        The reduced tensor.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.ops.convert_to_tensor(np.random.randn(3, 4))
    >>> x1 = tlx.ops.reduce_mean(x, axis=1, keepdims=False)
    >>> x2 = tlx.ops.reduce_mean(x, axis=1, keepdims=True)

    """

    return tf.reduce_mean(input_tensor, axis=axis, keepdims=keepdims)


class ReduceMax(object):

    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def __call__(self, inputs):
        output = tf.reduce_max(inputs, axis=self.axis, keepdims=self.keepdims)
        return output


def reduce_max(x, axis=None, keepdims=False):
    """
    Computes the maximum of elements across dimensions of a tensor.

    Parameters
    ----------
    x : tensor
        The tensor to reduce. Should have real numeric type.
    axis : int
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range [-rank(x), rank(x)).
    keepdims : boolean
        If true, keep these reduced dimensions and the length is 1. If false, don’t keep these dimensions.
        Default : False, don’t keep these reduced dimensions.


    Returns
    -------
        The reduced tensor.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.ops.convert_to_tensor(np.random.randn(3, 4))
    >>> x1 = tlx.ops.reduce_max(x, axis=1, keepdims=False)
    >>> x2 = tlx.ops.reduce_max(x, axis=1, keepdims=True)


    """

    return tf.reduce_max(x, axis=axis, keepdims=keepdims)


def reduce_min(x, axis=None, keepdims=False):
    """
    Computes the minimum of elements across dimensions of a tensor.

    Parameters
    ----------
    x : tensor
        The tensor to reduce. Should have real numeric type.
    axis : int
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range [-rank(x), rank(x)).
    keepdims : boolean
        If true, keep these reduced dimensions and the length is 1. If false, don’t keep these dimensions.
        Default : False, don’t keep these reduced dimensions.

    Returns
    -------
        The reduced tensor.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.ops.convert_to_tensor(np.random.randn(3, 4))
    >>> x1 = tlx.ops.reduce_min(x, axis=1, keepdims=False)
    >>> x2 = tlx.ops.reduce_min(x, axis=1, keepdims=True)

    """

    return tf.reduce_min(x, axis=axis, keepdims=keepdims)


class Pad(object):

    def __init__(self, paddings, mode="REFLECT", constant_values=0):
        if mode not in ['CONSTANT', 'REFLECT', 'SYMMETRIC']:
            raise Exception("Unsupported mode: {}".format(mode))
        self.paddings = paddings
        self.mode = mode
        self.constant_values = constant_values

    def __call__(self, x):
        outputs = tf.pad(x, self.paddings, mode=self.mode, constant_values=self.constant_values)
        return outputs


def pad(tensor, paddings, mode='CONSTANT', constant_values=0):
    """
    Pads a tensor.

    Parameters
    ----------
    tensor : tensor
        A Tensor.
    paddings : list or tuple
         paddings is an list or tuple with size [n, 2], where n is the rank of tensor.
         For each dimension D of input, paddings[D, :] indicates how many values to add before the contents of tensor in that dimension,
    mode : str
        One of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive), default is "CONSTANT".
    constant_values : int
        In "CONSTANT" mode, the scalar pad value to use. Must be same type as tensor.

    Returns
    -------
        A Tensor. Has the same type as tensor.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant([[1, 2, 3], [4, 5, 6]])
    >>> paddings = [[1,1], [2, 2]]
    >>> res = tlx.ops.pad(x, paddings)
    >>> [[0, 0, 0, 0, 0, 0, 0],
    >>> [0, 0, 1, 2, 3, 0, 0],
    >>> [0, 0, 4, 5, 6, 0, 0],
    >>> [0, 0, 0, 0, 0, 0, 0]]
    """

    if mode not in ['CONSTANT', 'REFLECT', 'SYMMETRIC']:
        raise Exception("Unsupported mode: {}".format(mode))
    outputs = tf.pad(tensor, paddings, mode=mode, constant_values=constant_values)
    return outputs


class Unstack(object):

    def __init__(self, axis, num=None):
        self.axis = axis
        self.num = num

    def __call__(self, values):
        return tf.unstack(values, num=self.num, axis=self.axis)


class Stack(object):

    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, values):
        return tf.stack(values, axis=self.axis)


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

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant([1,2,3])
    >>> y = tlx.ops.constant([1,2,3])
    >>> res = tlx.ops.stack([x, y])
    >>> [[1, 2, 3],
    >>>  [1, 2, 3]]

    """

    return tf.stack(values, axis=axis)


class Meshgrid(object):

    def __init__(self, indexing='xy'):
        super(Meshgrid, self).__init__()
        self.index = indexing

    def __call__(self, inputs):
        return tf.meshgrid(inputs)


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

    return tf.meshgrid(*args, **kwargs)


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
    dtype : type
        The type of the elements of the resulting tensor.

    Returns
    -------
        An 1-D Tensor of type dtype.
    """

    if limit is None:
        outputs = tf.range(start, delta=delta, dtype=dtype)
    else:
        outputs = tf.range(start, limit, delta=delta, dtype=dtype)
    return outputs


class ExpandDims(object):

    def __init__(self, axis):
        self.axis = axis

    def __call__(self, input):
        return tf.expand_dims(input, axis=self.axis)


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

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.ones([1,2,3])
    >>> res = tlx.ops.expand_dims(x, axis=0)
    >>> print(res.shape)
    >>>  [1, 1, 2, 3]

    """

    return tf.expand_dims(input, axis)


class Tile(object):

    def __init__(self):
        pass

    def __call__(self, input, multiples):
        return tile(input, multiples)


def tile(input, multiples):
    """
    Constructs a tensor by tiling a given tensor.

    Parameters
    ----------
    input : tensor
        A Tensor. 1-D or higher.
    multiples : tensor or tuple or list
        The number of repeating times.
        If repeat_times is a list or tuple, all its elements should be integers or 1-D Tensors with the data type int32.
        If repeat_times is a Tensor, it should be an 1-D Tensor with the data type int32.
        Length must be the same as the number of dimensions in input.
    Returns
    -------
        A Tensor. Has the same type as input.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant([[1,2,3],[1,2,3]])
    >>> y = tlx.ops.tile(x, [2, 1])
    >>> [[1, 2, 3],
    >>>  [1, 2, 3],
    >>>  [1, 2, 3],
    >>>  [1, 2, 3]]

    """
    multiples = tf.convert_to_tensor(multiples, tf.int32)
    return tf.tile(input, multiples)


class Cast(object):

    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, x):
        return tf.cast(x, dtype=self.dtype)


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

    return tf.cast(x, dtype=dtype)


class Transpose(object):

    def __init__(self, perm, conjugate=False):
        self.perm = perm
        self.conjugate = conjugate

    def __call__(self, a):
        return tf.transpose(a, self.perm, self.conjugate)


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

    return tf.transpose(a, perm, conjugate)


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

    return tf.gather_nd(params, indices, batch_dims)


class ClipGradByValue(object):
    def __init__(self, clip_min=-1, clip_max=1):
        self.min = clip_min
        self.max = clip_max

    def __call__(self, inputs):
        return tf.clip_by_value(inputs, self.min, self.max)


class ClipGradByNorm(object):
    def __init__(self, clip_norm=0.1):
        self.clip_norm = clip_norm

    def __call__(self, inputs):
        return tf.clip_by_norm(inputs, clip_norm=self.clip_norm)


class ClipByGlobalNorm(object):
    def __init__(self, clip_norm):
        self.clip_norm = clip_norm

    def __call__(self, inputs):
        return tf.clip_by_global_norm(inputs, clip_norm=self.clip_norm)


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

    return tf.clip_by_value(t, clip_value_min, clip_value_max)


def split(value, num_or_size_splits, axis=0):
    """
    Splits a tensor into sub tensors.

    Parameters
    ----------
    value : tensor
        The Tensor to split.
    num_or_size_splits : int or list
        Either an integer indicating the number of splits along split_dim or a 1-D integer Tensor or
        Python list containing the sizes of each output tensor along split_dim.
    axis : int
        The dimension along which to split. Must be in the range [-rank(value), rank(value)). Defaults to 0.


    Returns
    -------
        Tensor objects resulting from splitting value.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.ones([3, 9, 5])
    >>> y1, y2, y3 = tlx.ops.split(x, 3, axis=1)
    >>> y1, y2, y3 = tlx.ops.split(x, [1,3,5], axis=1)

    """

    return tf.split(value=value, num_or_size_splits=num_or_size_splits, axis=axis)


class Floor(object):

    def __call__(self, x):
        return tf.floor(x)


def floor(x):
    """
    Returns element-wise largest integer not greater than x.

    Parameters
    ----------
    x : tensor
        A Tensor. Must be one of the following types: bfloat16, half, float32, float64.

    Returns
    -------
        A Tensor. Has the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=[1.23, 2.56, 3.589])
    >>> y = tlx.ops.floor(x)

    """

    return tf.floor(x)


def gather(params, indices, axis=None):
    """Gather slices from params axis axis according to indices.

    Parameters
    ----------
    params : tensor
        The Tensor from which to gather values. Must be at least rank axis + 1.
    indices : indices
        The index Tensor. Must be one of the following types: int32, int64. The values must be in range [0, params.shape[axis]).
    axis : tensor.
        Must be one of the following types: int32, int64. The axis in params to gather indices from.

    Returns
    -------
        A Tensor. Has the same type as params.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=[[0, 1.0, 2.0],
    >>>                  [10.0, 11.0, 12.0],
    >>>                  [20.0, 21.0, 22.0],
    >>>                  [30.0, 31.0, 32.0]])
    >>> y = tlx.ops.gather(x, indices=[3,1])


    """
    return tf.gather(params, indices, axis=axis)


def linspace(start, stop, num):
    return tf.linspace(start, stop, num)


def slice(inputs, starts, sizes):
    return tf.slice(inputs, starts, sizes)


def add_n(inputs):
    return tf.add_n(inputs)


class OneHot(object):

    def __init__(self, depth, on_value=None, off_value=None, axis=None, dtype=None):
        self.depth = depth
        self.on_value = on_value
        self.off_value = off_value
        self.axis = axis
        self.dtype = dtype

    def __call__(self, inputs):
        outputs = tf.one_hot(
            inputs, self.depth, on_value=self.on_value, off_value=self.off_value, axis=self.axis, dtype=self.dtype
        )
        return outputs


class L2Normalize(object):

    def __init__(self, axis=None, epsilon=1e-12):
        self.axis = axis
        self.epsilon = epsilon

    def __call__(self, input, *args, **kwargs):
        outputs = tf.math.l2_normalize(input, axis=self.axis, epsilon=self.epsilon)
        return outputs


class EmbeddingLookup(object):

    def __init__(self, max_norm=None):
        self.max_norm = max_norm

    def __call__(self, params, ids):
        outputs = tf.nn.embedding_lookup(params=params, ids=ids, max_norm=self.max_norm)
        return outputs


class NCELoss(object):

    def __init__(self, num_true=1, sampled_values=None, remove_accidental_hits=False):
        self.num_true = num_true
        self.sampled_values = sampled_values
        self.remove_accidental_hits = remove_accidental_hits

    def __call__(self, weights, biases, labels, inputs, num_sampled, num_classes):
        outputs = tf.nn.nce_loss(
            weights=weights, biases=biases, inputs=inputs, labels=labels, num_sampled=num_sampled,
            num_classes=num_classes
        )
        return outputs


class NotEqual(object):

    def __init__(self):
        pass

    def __call__(self, x, y):
        return tf.not_equal(x, y)


class CountNonzero(object):

    def __init__(self, keepdims=None, dtype=int64):
        self.keepdims = keepdims
        self.dtype = dtype

    def __call__(self, input, axis=None):
        return tf.math.count_nonzero(input, axis=axis, keepdims=self.keepdims, dtype=self.dtype)


class Resize:

    def __init__(self, scale, method, antialias=False, data_format='channels_last'):
        self.method = method
        self.antialias = antialias
        self.scale = scale
        self.data_format = data_format

    def __call__(self, inputs):
        if self.data_format == 'channels_first':
            inputs = nchw_to_nhwc(inputs)
        if len(get_tensor_shape(inputs)) == 4:
            output_size = [int(inputs.shape[1] * self.scale[0]), int(inputs.shape[2] * self.scale[1])]
        else:
            raise ("The inputs shape must be 4-D Tensor.")
        outputs = tf.image.resize(inputs, size=output_size, method=self.method, antialias=self.antialias)
        if self.data_format == 'channels_first':
            outputs = nhwc_to_nchw(outputs)
        return outputs


def resize(inputs, output_size, method, antialias):
    return tf.image.resize(inputs, size=output_size, method=method, antialias=antialias)


class ZeroPadding1D(object):

    def __init__(self, padding):
        self.zeropad = tf.keras.layers.ZeroPadding1D(padding=padding)

    def __call__(self, inputs):
        return self.zeropad(inputs)


class ZeroPadding2D(object):

    def __init__(self, padding):
        self.zeropad = tf.keras.layers.ZeroPadding2D(padding=padding)

    def __call__(self, inputs):
        return self.zeropad(inputs)


class ZeroPadding3D(object):

    def __init__(self, padding):
        self.zeropad = tf.keras.layers.ZeroPadding3D(padding=padding)

    def __call__(self, inputs):
        return self.zeropad(inputs)


class Sign(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return tf.sign(x)


class Ceil(object):

    def __call__(self, x):
        return tf.math.ceil(x)


def ceil(x):
    """
    Return the ceiling of the input, element-wise.

    Parameters
    ----------
    x : tensor
        Must be one of the following types: bfloat16, half, float32, float64. int32

    Returns
    -------
        A Tensor. Has the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=[0.9142202  0.72091234])
    >>> y = tlx.ops.ceil(x)

    """

    return tf.math.ceil(x)


def multiply(x, y):
    """
    Returns an element-wise x * y.

    Parameters
    ----------
    x : tensor
        Must be one of the following types: bfloat16, half, float32, float64,
        uint8, int8, uint16, int16, int32, int64, complex64, complex128.
    y : tensor
        A Tensor. Must have the same type as x.

    Returns
    -------
        A Tensor. Has the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=[0.9142202  0.72091234])
    >>> y = tlx.ops.multiply(x, x)

    """

    return tf.multiply(x, y)


def divide(x, y):
    """
    Computes Python style division of x by y.

    Parameters
    ----------
    x : tensor
        A Tensor
    y : tensor
        A Tensor

    Returns
    -------
        A Tensor with same shape as input

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=[0.9142202  0.72091234])
    >>> y = tlx.ops.divide(x, x)

    """

    return tf.divide(x, y)


def identity(x):
    return tf.identity(x)


class BatchToSpace(object):

    def __init__(self, block_size, crops):
        self.bolock_size = block_size
        self.crops = crops

    def __call__(self, input_x):
        return tf.batch_to_space(input=input_x, block_shape=self.bolock_size, crops=self.crops)


class DepthToSpace(object):

    def __init__(self, block_size, data_format='NHWC'):
        data_format, _ = preprocess_2d_format(data_format, None)
        self.block_size = block_size
        self.data_format = data_format

    def __call__(self, input):
        return tf.nn.depth_to_space(input, block_size=self.block_size, data_format=self.data_format)


def triu(x, diagonal=0):
    """
    This op returns the upper triangular part of a matrix (2-D tensor) or batch of matrices x,
    the other elements of the result tensor are set to 0.
    The upper triangular part of the matrix is defined as the elements on and above the diagonal.

    Parameters
    ----------
    x : tensor
        The tensor to triu.
    diagonal : int
        The diagonal to consider, default value is 0. If diagonal = 0, all elements on and above the main diagonal are retained.
        A positive value excludes just as many diagonals above the main diagonal, and similarly a negative value includes just as many diagonals below the main diagonal.

    Returns
    -------
        Results of upper triangular operation by the specified diagonal of input tensor x, it’s data type is the same as x’s Tensor.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.convert_to_tensor(np.arange(1, 10, dtype="int32").reshape(3,-1))
    >>> y = tlx.ops.triu(x, diagonal=1)
    >>> print(y)
    >>> [[0, 2, 3],
    >>> [ 0, 0, 6],
    >>> [ 0, 0, 0]]

    """

    return tf.experimental.numpy.triu(x, k=diagonal)


def tril(x, diagonal=0):
    """
    This op returns the lower triangular part of a matrix (2-D tensor) or batch of matrices x, the other elements of the result tensor are set to 0.
    The lower triangular part of the matrix is defined as the elements on and below the diagonal.

    Parameters
    ----------
    x : tensor
        The tensor to tril.
    diagonal : int
        The diagonal to consider, default value is 0. If diagonal = 0, all elements on and below the main diagonal are retained.
        A positive value includes just as many diagonals above the main diagonal, and similarly a negative value excludes just as many diagonals below the main diagonal.

    Returns
    -------
        Results of lower triangular operation by the specified diagonal of input tensor x, it’s data type is the same as x’s Tensor.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.convert_to_tensor(np.arange(1, 10, dtype="int32").reshape(3,-1))
    >>> y = tlx.ops.tril(x, diagonal=1)
    >>> print(y)
    >>> [[0, 0, 0],
    >>> [ 4, 0, 0],
    >>> [ 7, 8, 0]]

    """

    return tf.experimental.numpy.tril(x, k=diagonal)


def abs(x):
    """
    Computes the absolute value of a tensor.

    Parameters
    ----------
    x : tensor
        A Tensor or SparseTensor of type float16, float32, float64, int32, int64, complex64 or complex128.

    Returns
    -------
        A Tensor or SparseTensor of the same size, type and sparsity as x, with absolute values.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.layers.Input((32, 3, 3, 32))
    >>> y = tlx.ops.abs(x)

    """

    return tf.math.abs(x)


def acos(x):
    """
    Computes acos of x element-wise.

    Parameters
    ----------
    x : tensor
        Must be one of the following types: bfloat16, half, float32, float64, uint8, int8, int16, int32, int64, complex64, complex128, string.

    Returns
    -------
        A Tensor. Has the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.layers.Input((32, 3, 3, 32))
    >>> y = tlx.ops.acos(x)

    """

    return tf.math.acos(x)


def acosh(x):
    """
    Computes inverse hyperbolic cosine of x element-wise.

    Parameters
    ----------
    x : tensor
        A Tensor. Must be one of the following types: bfloat16, half, float32, float64, complex64, complex128.

    Returns
    -------
        A Tensor. Has the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.layers.Input((32, 3, 3, 32))
    >>> y = tlx.ops.acosh(x)

    """

    return tf.math.acosh(x)


def angle(x):
    """
    Returns the element-wise argument of a complex (or real) tensor.

    Parameters
    ----------
    x : tensor
        A Tensor. Must be one of the following types: float, double, complex64, complex128.

    Returns
    -------
        A Tensor of type float32 or float64.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=[2.15 + 3.57j, 3.89 + 6.54j])
    >>> y = tlx.ops.angle(x)

    """

    return tf.math.angle(x)


def argmax(x, axis=None, dtype='int64'):
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

    return tf.math.argmax(x, axis=axis, output_type=dtype_str(dtype))


def argmin(x, axis=None, dtype='int64'):
    """
    Returns the index with the smallest value across axes of a tensor.

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
    >>> y = tlx.ops.argmin(x)

    """

    return tf.math.argmin(x, axis=axis, output_type=dtype_str(dtype))


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

    return tf.math.asin(x)


def asinh(x):
    """
    Computes inverse hyperbolic sine of x element-wise.

    Parameters
    ----------
    x : tensor
        Must be one of the following types: bfloat16, half, float32, float64, complex64, complex128.

    Returns
    -------
        A Tensor. Has the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=[0.9142202  0.72091234])
    >>> y = tlx.ops.asinh(x)

    """

    return tf.math.asinh(x)


def atan(x):
    """
    Computes the trignometric inverse tangent of x element-wise.

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
    >>> x = tlx.ops.constant(value=[0.9142202  0.72091234])
    >>> y = tlx.ops.atan(x)

    """

    return tf.math.atan(x)


def atanh(x):
    """
    Computes inverse hyperbolic tangent of x element-wise.

    Parameters
    ----------
    x : tensor
        Must be one of the following types: bfloat16, half, float32, float64, complex64, complex128.

    Returns
    -------
        A Tensor. Has the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=[0.9142202  0.72091234])
    >>> y = tlx.ops.atanh(x)

    """

    return tf.math.atanh(x)


def cos(x):
    """
    Computes cos of x element-wise.

    Parameters
    ----------
    x : tensor
        Must be one of the following types: bfloat16, half, float32, float64, complex64, complex128.

    Returns
    -------
        A Tensor. Has the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=[0.9142202  0.72091234])
    >>> y = tlx.ops.cos(x)

    """

    return tf.math.cos(x)


def cosh(x):
    """
    Computes hyperbolic cosine of x element-wise.

    Parameters
    ----------
    x : tensor
        Must be one of the following types: bfloat16, half, float32, float64, complex64, complex128.

    Returns
    -------
        A Tensor. Has the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=[0.9142202  0.72091234])
    >>> y = tlx.ops.cosh(x)

    """

    return tf.math.cosh(x)


def count_nonzero(x, axis=None, keepdims=None, dtype="int64"):
    """
    Computes number of nonzero elements across dimensions of a tensor.

    Parameters
    ----------
    x : tensor
        The tensor to reduce. Should be of numeric type, bool, or string.
    axis : int
        The dimensions to reduce. If None (the default), reduces all dimensions. Must be in the range [-rank(input), rank(input)).
    keepdims : bool
        If true, retains reduced dimensions with length 1.
    dtype : tensor or str
        The output dtype; defaults to tf.int64.

    Returns
    -------
        The reduced tensor (number of nonzero values).

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=["", "a", "c", "b", " "])
    >>> y = tlx.ops.count_nonzero(x)

    """

    return tf.math.count_nonzero(x, axis=axis, keepdims=keepdims, dtype=dtype_str(dtype))


def cumprod(x, axis=0, exclusive=False, reverse=False):
    """
    Compute the cumulative product of the tensor x along axis.

    Parameters
    ----------
    x : tensor
        Must be one of the following types: float32, float64, int64, int32, uint8, uint16, int16, int8,
         complex64, complex128, qint8, quint8, qint32, half.
    axis : int
        A Tensor of type int32 (default: 0). Must be in the range [-rank(x), rank(x)).
    exclusive : bool
        If True, perform exclusive cumprod.
    reverse : bool
        A bool (default: False).

    Returns
    -------
        A Tensor. Has the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=[3, 2, 1])
    >>> y = tlx.ops.cumprod(x)
    >>> y = tlx.ops.cumprod(x, exclusive=True, reverse=True)

    """

    return tf.math.cumprod(x, axis=axis, exclusive=exclusive, reverse=reverse)


def cumsum(x, axis=0, exclusive=False, reverse=False):
    """
    Compute the cumulative sum of the tensor x along axis.

    Parameters
    ----------
    x : tensor
        Must be one of the following types: float32, float64, int64, int32, uint8, uint16, int16, int8,
         complex64, complex128, qint8, quint8, qint32, half.
    axis : int
        A Tensor of type int32 (default: 0). Must be in the range [-rank(x), rank(x)).
    exclusive : bool
        If True, perform exclusive cumprod.
    reverse : bool
        A bool (default: False).

    Returns
    -------
        A Tensor. Has the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=[1, 2, 3])
    >>> y = tlx.ops.cumsum(x)
    >>> y = tlx.ops.cumsum(x, exclusive=True, reverse=True)

    """

    return tf.math.cumsum(x, axis=axis, exclusive=exclusive, reverse=reverse)


def equal(x, y):
    """
    Returns the truth value of (x == y) element-wise.

    Parameters
    ----------
    x : tensor
        A Tensor or SparseTensor or IndexedSlices.
    y : tensor
        A Tensor or SparseTensor or IndexedSlices.

    Returns
    -------
        A Tensor of type bool with the same size as that of x or y.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=[1, 2, 3])
    >>> y = tlx.ops.equal(x, x)

    """

    return tf.math.equal(x, y)


def exp(x):
    """
    Computes exponential of x element-wise.

    Parameters
    ----------
    x : tensor
        Must be one of the following types: bfloat16, half, float32, float64, complex64, complex128.

    Returns
    -------
        A Tensor. Has the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=[1, 2, 3])
    >>> y = tlx.ops.exp(x)

    """

    return tf.exp(x)


def floordiv(x, y):
    """
    Divides x / y elementwise, rounding toward the most negative integer.

    Parameters
    ----------
    x : tensor
        Tensor numerator of real numeric type.
    y : tensor
        Tensor denominator of real numeric type.

    Returns
    -------
        x / y rounded toward -infinity.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=[1, 2, 3])
    >>> y = tlx.ops.floordiv(x, x)

    """

    return tf.math.floordiv(x, y)


def floormod(x, y):
    """
    Returns element-wise remainder of division.
    When x < 0 xor y < 0 is true, this follows Python semantics in that the result
    here is consistent with a flooring divide. E.g. floor(x / y) * y + mod(x, y) = x.

    Parameters
    ----------
    x : tensor
        Must be one of the following types: int8, int16, int32, int64, uint8,
        uint16, uint32, uint64, bfloat16, half, float32, float64.
    y : tensor
        A Tensor. Must have the same type as x.

    Returns
    -------
        A Tensor. Has the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=[1, 2, 3])
    >>> y = tlx.ops.floormod(x, x)

    """

    return tf.math.floormod(x, y)


def greater(x, y):
    """
    Returns the truth value of (x >= y) element-wise.

    Parameters
    ----------
    x : tensor
        Must be one of the following types: float32, float64, int32, uint8, int16,
        int8, int64, bfloat16, uint16, half, uint32, uint64.
    y : tensor
        A Tensor. Must have the same type as x.

    Returns
    -------
        A Tensor of type bool.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=[1, 2, 3])
    >>> y = tlx.ops.greater(x, x)

    """

    return tf.math.greater(x, y)


def greater_equal(x, y):
    """
    Returns the truth value of (x >= y) element-wise.

    Parameters
    ----------
    x : tensor
        Must be one of the following types: float32, float64, int32, uint8,
        int16, int8, int64, bfloat16, uint16, half, uint32, uint64.
    y : tensor
        A Tensor. Must have the same type as x.

    Returns
    -------
        A Tensor of type bool.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=[1, 2, 3])
    >>> y = tlx.ops.greater_equal(x, x)

    """

    return tf.math.greater_equal(x, y)


def is_inf(x):
    """
    Returns which elements of x are Inf.

    Parameters
    ----------
    x : tensor
        A Tensor. Must be one of the following types: bfloat16, half, float32, float64.

    Returns
    -------
        A Tensor of type bool.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.ops.constant(value=[1, 2, 3, np.inf])
    >>> y = tlx.ops.is_inf(x)

    """

    return tf.math.is_inf(x)


def is_nan(x):
    """
    Returns which elements of x are NaN.

    Parameters
    ----------
    x : tensor
        A Tensor. Must be one of the following types: bfloat16, half, float32, float64.

    Returns
    -------
        A Tensor of type bool.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.ops.constant(value=[1, 2, 3, np.nan])
    >>> y = tlx.ops.is_nan(x)

    """

    return tf.math.is_nan(x)


def l2_normalize(x, axis=None, eps=1e-12):
    """
    Normalizes along dimension axis using an L2 norm.
    For a 1-D tensor with axis = 0, computes output = x / sqrt(max(sum(x**2), epsilon))

    Parameters
    ----------
    x : tensor
        A Tensor
    axis : int
        Dimension along which to normalize. A scalar or a vector of integers.
    eps : float
        A lower bound value for the norm. Will use sqrt(epsilon) as the divisor if norm < sqrt(epsilon).

    Returns
    -------
        A Tensor with the same shape as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.ops.constant(value=[1, 2, 3, np.nan])
    >>> y = tlx.ops.l2_normalize(x)

    """

    return tf.math.l2_normalize(x, axis=axis, epsilon=eps)


def less(x, y):
    """
    Returns the truth value of (x < y) element-wise.

    Parameters
    ----------
    x : tensor
        Must be one of the following types: float32, float64, int32, uint8,
        int16, int8, int64, bfloat16, uint16, half, uint32, uint64.
    y : tensor
        A Tensor. Must have the same type as x.

    Returns
    -------
        A Tensor of type bool.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=[1, 2, 3])
    >>> y = tlx.ops.less(x, x)

    """
    return tf.math.less(x, y)


def less_equal(x, y):
    """
    Returns the truth value of (x <= y) element-wise.

    Parameters
    ----------
    x : tensor
        Must be one of the following types: float32, float64, int32, uint8,
        int16, int8, int64, bfloat16, uint16, half, uint32, uint64.
    y : tensor
        A Tensor. Must have the same type as x.

    Returns
    -------
        A Tensor of type bool.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=[1, 2, 3])
    >>> y = tlx.ops.less_equal(x, x)

    """
    return tf.math.less_equal(x, y)


def log(x):
    """
    Computes natural logarithm of x element-wise.

    Parameters
    ----------
    x : tensor
        Must be one of the following types: bfloat16, half, float32, float64, complex64, complex128.

    Returns
    -------
        A Tensor. Has the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=[1, 2, 3])
    >>> y = tlx.ops.log(x)

    """

    return tf.math.log(x)


def log_sigmoid(x):
    """
    Computes log sigmoid of x element-wise.

    Parameters
    ----------
    x : tensor
        A Tensor with type float32 or float64.

    Returns
    -------
        A Tensor with the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=[1, 2, 3])
    >>> y = tlx.ops.log_sigmoid(x)

    """

    return tf.math.log_sigmoid(x)


def maximum(x, y):
    """
    Returns the max of x and y (i.e. x > y ? x : y) element-wise.

    Parameters
    ----------
    x : tensor
        Must be one of the following types: float32, float64, int32, uint8,
        int16, int8, int64, bfloat16, uint16, half, uint32, uint64.
    y : tensor
        A Tensor. Must have the same type as x.

    Returns
    -------
        A Tensor. Has the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=[1, 2, 3])
    >>> y = tlx.ops.maximum(x, x)

    """

    return tf.math.maximum(x, y)


def negative(x):
    """
    Computes numerical negative value element-wise.

    Parameters
    ----------
    x : tensor
        Must be one of the following types: bfloat16, half, float32, float64, int8,
        int16, int32, int64, complex64, complex128.

    Returns
    -------
        A Tensor. Has the same type as x.
        If x is a SparseTensor, returns SparseTensor(x.indices, tf.math.negative(x.values, ...), x.dense_shape)

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=[1, 2, 3])
    >>> y = tlx.ops.negative(x)

    """

    return tf.math.negative(x)


def not_equal(x, y):
    """
    Returns the truth value of (x != y) element-wise.

    Parameters
    ----------
    x : tensor
        A Tensor or SparseTensor or IndexedSlices.
    y : tensor
        A Tensor or SparseTensor or IndexedSlices.

    Returns
    -------
        A Tensor of type bool with the same size as that of x or y.


    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=[1, 2, 3])
    >>> y = tlx.ops.constant(value=[1, 3, 5])
    >>> x = tlx.ops.not_equal(x, y)

    """

    return tf.math.not_equal(x, y)


def pow(x, y):
    """
    Computes the power of one value to another.

    Parameters
    ----------
    x : tensor
        A Tensor of type float16, float32, float64, int32, int64, complex64, or complex128.
    y : tensor
        A Tensor of type float16, float32, float64, int32, int64, complex64, or complex128.

    Returns
    -------
        A Tensor.


    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=[1, 2, 3])
    >>> y = tlx.ops.constant(value=[1, 3, 5])
    >>> x = tlx.ops.pow(x, y)

    """

    return tf.math.pow(x, y)


def real(x):
    """
    Computes numerical negative value element-wise.

    Parameters
    ----------
    x : tensor
        A Tensor. Must have numeric type.

    Returns
    -------
        A Tensor of type float32 or float64.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=[-2.25 + 4.75j, 3.25 + 5.75j])
    >>> y = tlx.ops.real(x)

    """

    return tf.math.real(x)


def reciprocal(x):
    """
    Computes the reciprocal of x element-wise.

    Parameters
    ----------
    x : tensor
        A Tensor. Must be one of the following types: bfloat16, half, float32, float64,
        int8, int16, int32, int64, complex64, complex128.

    Returns
    -------
        A Tensor. Has the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant(value=[-2.25, 3.25])
    >>> y = tlx.ops.reciprocal(x)

    """

    return tf.math.reciprocal(x)


def reduce_prod(x, axis=None, keepdims=False):
    """
    Computes the multiply of elements across dimensions of a tensor.

    Parameters
    ----------
    x : tensor
        The tensor to reduce. Should have real numeric type.
    axis : int
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range [-rank(x), rank(x)).
    keepdims : boolean
        If true, keep these reduced dimensions and the length is 1. If false, don’t keep these dimensions.
        Default : False, don’t keep these reduced dimensions.

    Returns
    -------
        The reduced tensor.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.ops.convert_to_tensor(np.random.randn(3, 4))
    >>> x1 = tlx.ops.reduce_prod(x, axis=1, keepdims=False)
    >>> x2 = tlx.ops.reduce_prod(x, axis=1, keepdims=True)

    """

    return tf.reduce_prod(x, axis=axis, keepdims=keepdims)


def reduce_std(x, axis=None, keepdims=False):
    """
    Computes the standard deviation of elements across dimensions of a tensor.

    Parameters
    ----------
    x : tensor
        The tensor to reduce. Should have real numeric type.
    axis : int
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range [-rank(x), rank(x)).
    keepdims : boolean
        If true, keep these reduced dimensions and the length is 1. If false, don’t keep these dimensions.
        Default : False, don’t keep these reduced dimensions.

    Returns
    -------
        The reduced tensor.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.ops.convert_to_tensor(np.random.randn(3, 4))
    >>> x1 = tlx.ops.reduce_std(x, axis=1, keepdims=False)
    >>> x2 = tlx.ops.reduce_std(x, axis=1, keepdims=True)

    """

    return tf.math.reduce_std(x, axis=axis, keepdims=keepdims)


def reduce_sum(x, axis=None, keepdims=False):
    """
    Computes the standard deviation of elements across dimensions of a tensor.

    Parameters
    ----------
    x : tensor
        The tensor to reduce. Should have real numeric type.
    axis : int
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range [-rank(x), rank(x)).
    keepdims : boolean
        If true, keep these reduced dimensions and the length is 1. If false, don’t keep these dimensions.
        Default : False, don’t keep these reduced dimensions.

    Returns
    -------
        The reduced tensor.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.ops.convert_to_tensor(np.random.randn(3, 4))
    >>> x1 = tlx.ops.reduce_sum(x, axis=1, keepdims=False)
    >>> x2 = tlx.ops.reduce_sum(x, axis=1, keepdims=True)

    """

    return tf.reduce_sum(x, axis=axis, keepdims=keepdims)


def reduce_variance(x, axis=None, keepdims=False):
    """
    Computes the variance of elements across dimensions of a tensor.

    Parameters
    ----------
    x : tensor
        The tensor to reduce. Should have real numeric type.
    axis : int
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range [-rank(x), rank(x)).
    keepdims : boolean
        If true, keep these reduced dimensions and the length is 1. If false, don’t keep these dimensions.
        Default : False, don’t keep these reduced dimensions.

    Returns
    -------
        The reduced tensor.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.ops.convert_to_tensor(np.random.randn(3, 4))
    >>> x1 = tlx.ops.reduce_variance(x, axis=1, keepdims=False)
    >>> x2 = tlx.ops.reduce_variance(x, axis=1, keepdims=True)

    """
    return tf.math.reduce_variance(x, axis=axis, keepdims=keepdims)


def round(x):
    """
    Rounds the values of a tensor to the nearest integer, element-wise.

    Parameters
    ----------
    x : tensor
        The tensor to round. Should have real numeric type.

    Returns
    -------
        A Tensor of same shape and type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.ops.convert_to_tensor(np.array([0.9, 2.5, 2.3, 1.5, -4.5]))
    >>> x = tlx.ops.round(x)

    """
    return tf.round(x)


def rsqrt(x):
    """
    Computes reciprocal of square root of x element-wise.

    Parameters
    ----------
    x : tensor
        The tensor to rsqrt. Should have real numeric type.

    Returns
    -------
        A Tensor of same shape and type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.ops.convert_to_tensor(np.array([0.9, 2.5, 2.3, 1.5]))
    >>> x = tlx.ops.rsqrt(x)

    """
    return tf.math.rsqrt(x)


def segment_max(x, segment_ids):
    """
    Computes the maximum along segments of a tensor.

    Parameters
    ----------
    x : tensor
        The tensor to segment_max. Should have real numeric type.
    segment_ids : tensor
        A 1-D tensor whose size is equal to the size of data's first dimension. Values should be sorted and can be repeated.

    Returns
    -------
        A Tensor. Has the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.ops.convert_to_tensor(np.array([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]]))
    >>> id = tlx.ops.convert_to_tensor([0, 0, 1])
    >>> x = tlx.ops.segment_max(x, id)
    >>> print(x)
    >>> [[4, 3, 3, 4],
    >>> [5, 6, 7, 8]]

    """

    return tf.math.segment_max(x, segment_ids)


def segment_mean(x, segment_ids):
    """
    Computes the mean along segments of a tensor.

    Parameters
    ----------
    x : tensor
        The tensor to segment_mean. Should have real numeric type.
    segment_ids : tensor
        A 1-D tensor whose size is equal to the size of data's first dimension. Values should be sorted and can be repeated.

    Returns
    -------
        A Tensor. Has the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.ops.convert_to_tensor(np.array([[1.0 , 2, 3, 4], [4, 3, 2, 1], [5, 6, 7, 8]]))
    >>> id = tlx.ops.convert_to_tensor([0, 0, 1])
    >>> x = tlx.ops.segment_mean(x, id)
    >>> print(x)
    >>> [[2.5, 2.5, 2.5, 2.5],
    >>> [5, 6, 7, 8]]

    """

    return tf.math.segment_mean(x, segment_ids)


def segment_min(x, segment_ids):
    """
    Computes the minimum along segments of a tensor.

    Parameters
    ----------
    x : tensor
        The tensor to segment_minimum. Should have real numeric type.
    segment_ids : tensor
        A 1-D tensor whose size is equal to the size of data's first dimension. Values should be sorted and can be repeated.

    Returns
    -------
        A Tensor. Has the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.ops.convert_to_tensor(np.array([[1, 2, 3, 4], [4, 3, 2, 1], [5, 6, 7, 8]]))
    >>> id = tlx.ops.convert_to_tensor([0, 0, 1])
    >>> x = tlx.ops.segment_minimum(x, id)
    >>> print(x)
    >>> [[1, 2, 2, 1],
    >>> [5, 6, 7, 8]]

    """

    return tf.math.segment_minimum(x, segment_ids)


def segment_prod(x, segment_ids):
    """
    Computes the product along segments of a tensor.

    Parameters
    ----------
    x : tensor
        The tensor to segment_prod. Should have real numeric type.
    segment_ids : tensor
        A 1-D tensor whose size is equal to the size of data's first dimension. Values should be sorted and can be repeated.

    Returns
    -------
        A Tensor. Has the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.ops.convert_to_tensor(np.array([[1, 2, 3, 4], [4, 3, 2, 1], [5, 6, 7, 8]]))
    >>> id = tlx.ops.convert_to_tensor([0, 0, 1])
    >>> x = tlx.ops.segment_prod(x, id)
    >>> print(x)
    >>> [[4, 6, 6, 4],
    >>> [5, 6, 7, 8]]

    """

    return tf.math.segment_prod(x, segment_ids)


def segment_sum(x, segment_ids):
    """
    Computes the sum along segments of a tensor.

    Parameters
    ----------
    x : tensor
        The tensor to segment_sum. Should have real numeric type.
    segment_ids : tensor
        A 1-D tensor whose size is equal to the size of data's first dimension. Values should be sorted and can be repeated.

    Returns
    -------
        A Tensor. Has the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.ops.convert_to_tensor(np.array([[1, 2, 3, 4], [4, 3, 2, 1], [5, 6, 7, 8]]))
    >>> id = tlx.ops.convert_to_tensor([0, 0, 1])
    >>> x = tlx.ops.segment_sum(x, id)
    >>> print(x)
    >>> [[5, 5, 5, 5],
    >>> [5, 6, 7, 8]]

    """

    return tf.math.segment_sum(x, segment_ids)


def sigmoid(x):
    """
    Computes sigmoid of a tensor element-wise.

    Parameters
    ----------
    x : tensor
        The tensor to sigmoid.


    Returns
    -------
        A Tensor with the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.ops.convert_to_tensor(np.array([-128.0, 0.0, 128.0]), dtype='float32')
    >>> x = tlx.ops.sigmoid(x)
    >>> print(x)
    >>> [0., 0.5, 1.]

    """
    return tf.sigmoid(x)


def sign(x):
    """
    Computes sign of a tensor element-wise.

    Parameters
    ----------
    x : tensor
        The tensor to sign. y = sign(x) = -1 if x < 0; 0 if x == 0; 1 if x > 0.

    Returns
    -------
        A Tensor with the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.ops.convert_to_tensor(np.array([-128.0, 0.0, 128.0]), dtype='float32')
    >>> x = tlx.ops.sign(x)
    >>> print(x)
    >>> [-1., 0., 1.]

    """
    return tf.sign(x)


def sin(x):
    """
    Computes sine of a tensor element-wise.

    Parameters
    ----------
    x : tensor
        The tensor to sin. Input range is (-inf, inf) and output range is [-1,1].

    Returns
    -------
         A Tensor with the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.ops.convert_to_tensor(np.array([-1.0, 0.0, 1.0]), dtype='float32')
    >>> x = tlx.ops.sin(x)
    >>> print(x)
    >>> [-0.84147096, 0., 0.84147096]

    """
    return tf.math.sin(x)


def sinh(x):
    """
    Computes hyperbolic sine of a tensor element-wise.

    Parameters
    ----------
    x : tensor
        The tensor to hyperbolic sin. Input range is (-inf, inf) and output range is [-inf,inf].

    Returns
    -------
         A Tensor with the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.ops.convert_to_tensor(np.array([-1.0, 0.0, 1.0]), dtype='float32')
    >>> x = tlx.ops.sinh(x)
    >>> print(x)
    >>> [-1.1752012, 0., 1.1752012]

    """
    return tf.math.sinh(x)


def softplus(x):
    """
    Computes softplus of a tensor element-wise.

    Parameters
    ----------
    x : tensor
        The tensor to softplus. softplus(x) = log(exp(x) + 1).

    Returns
    -------
        A Tensor with the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.ops.convert_to_tensor(np.array([-1.0, 0.0, 1.0]), dtype='float32')
    >>> x = tlx.ops.softplus(x)
    >>> print(x)
    >>> [0.3132617, 0.6931472, 1.3132616]
    """

    return tf.math.softplus(x)


def square(x):
    """
    Computes square of a tensor element-wise.

    Parameters
    ----------
    x : tensor
        The tensor to square.

    Returns
    -------
        A Tensor with the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.ops.convert_to_tensor(np.array([-1.0, 0.0, 1.0]), dtype='float32')
    >>> x = tlx.ops.square(x)
    >>> print(x)
    >>> [1.0, 0.0, 1.0]
    """
    return tf.math.square(x)


def squared_difference(x, y):
    """
    Computes difference and square between tensor x and tensor y. return square(x - y)

    Parameters
    ----------
    x : tensor
        A Tensor.
    y : tensor
        A Tensor. Must have the same type as x.

    Returns
    -------
        A Tensor. Has the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.ops.convert_to_tensor(np.array([[1,0,1], [2,3,4]]), dtype='float32')
    >>> y = tlx.ops.convert_to_tensor(np.array([[-1,0,1], [2,3,4]]), dtype='float32')
    >>> res = tlx.ops.squared_difference(x, y)
    >>> print(res)
    >>> [[4.0, 0.0, 0.0],
    >>> [0.0, 0.0, 0.0]]
    """

    return tf.math.squared_difference(x, y)


def subtract(x, y):
    """
    Returns x - y element-wise.

    Parameters
    ----------
    x : tensor
        A tensor.
    y : tensor
        A Tensor. Must have the same type as x.

    Returns
    -------
        A Tensor. Has the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.ops.convert_to_tensor(np.array([[1,0,1], [2,3,4]]), dtype='float32')
    >>> y = tlx.ops.convert_to_tensor(np.array([[-1,0,1], [2,3,4]]), dtype='float32')
    >>> res = tlx.ops.subtract(x, y)
    >>> print(res)
    >>> [[-2.0, 0.0, 0.0],
    >>> [0.0, 0.0, 0.0]]

    """
    return tf.math.subtract(x, y)


def tan(x):
    """
    Computes tan of a tensor element-wise.

    Parameters
    ----------
    x : tensor
        The tensor to tan.

    Returns
    -------
        A Tensor. Has the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.ops.convert_to_tensor(np.array([1,0,1]), dtype='float32')
    >>> res = tlx.ops.tan(x)
    >>> print(res)
    >>> [-1.5574077, 0.0, 1.5574077]

    """

    return tf.math.tan(x)


def tanh(x):
    """
    Computes hyperbolic tangent of a tensor element-wise.

    Parameters
    ----------
    x : tensor
        The tensor to tanh.

    Returns
    -------
        A Tensor. Has the same type as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.ops.convert_to_tensor(np.array([1,0,1]), dtype="float32")
    >>> res = tlx.ops.tanh(x)
    >>> print(res)
    >>> [-0.7615942, 0.0, 0.7615942]
    """

    return tf.math.tanh(x)


def any(x, axis=None, keepdims=False):
    """
    Computes logical_or of a tensor element-wise.

    Parameters
    ----------
    x : tensor
        The boolean tensor to reduce.
    axis : int or None
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range [-rank(x),rank(x)).
    keepdims : boolean
        If true, retains reduced dimensions with length 1.

    Returns
    -------
        The reduced tensor.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.ops.convert_to_tensor(np.array([1,0,1]), dtype='bool')
    >>> res = tlx.ops.any(x, axis = None, keepdims = False)
    >>> print(res)
    >>> True
    """

    return tf.math.reduce_any(x, axis=axis, keepdims=keepdims)


def all(x, axis=None, keepdims=False):
    """
    Computes logical_and of a tensor element-wise.

    Parameters
    ----------
    x : tensor
        The boolean tensor to reduce.
    axis : int or None
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range [-rank(x),rank(x)).
    keepdims : boolean
        If true, retains reduced dimensions with length 1.

    Returns
    -------
        The reduced tensor.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> import numpy as np
    >>> x = tlx.ops.convert_to_tensor(np.array([1,0,1]), dtype='bool')
    >>> res = tlx.ops.all(x, axis = None, keepdims = False)
    >>> print(res)
    >>> False
    """

    return tf.math.reduce_all(x, axis=axis, keepdims=keepdims)


def logical_and(x, y):
    """
    Returns the truth value of x AND y element-wise.

    Parameters
    ----------
    x : tensor
        A tf.Tensor of type bool.
    y : tensor
        A tf.Tensor of type bool.

    Returns
    -------
        A Tensor of type bool.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.constant([False, False, True, True])
    >>> y = tlx.constant([False, True, False, True])
    >>> res = tlx.ops.logical_and(x, y)
    >>> print(res)
    >>> [False, False, False, True]

    """

    return tf.math.logical_and(x, y)


def logical_or(x, y):
    """
    Returns the truth value of x OR y element-wise.

    Parameters
    ----------
    x : tensor
        A tf.Tensor of type bool.
    y : tensor
        A tf.Tensor of type bool.

    Returns
    -------
        A Tensor of type bool.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.constant([False, False, True, True])
    >>> y = tlx.constant([False, True, False, True])
    >>> res = tlx.ops.logical_or(x, y)
    >>> print(res)
    >>> [False, True, True, True]

    """

    return tf.math.logical_or(x, y)


def logical_not(x):
    """
    Returns the truth value of NOT x element-wise.

    Parameters
    ----------
    x : tensor
        A tf.Tensor of type bool.

    Returns
    -------
        A Tensor of type bool.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.constant([False, False, True, True])
    >>> res = tlx.ops.logical_not(x, y)
    >>> print(res)
    >>> [True, True, False, False]

    """

    return tf.math.logical_not(x)


def logical_xor(x, y):
    """
    Returns the truth value of NOT x element-wise. x ^ y = (x | y) & ~(x & y)

    Parameters
    ----------
    x : tensor
        A tf.Tensor of type bool.

    Returns
    -------
        A Tensor of type bool.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.constant([False, False, True, True])
    >>> y = tlx.constant([False, True, False, True])
    >>> res = tlx.ops.logical_xor(x, y)
    >>> print(res)
    >>> [False, True, True, False]

    """

    return tf.math.logical_xor(x, y)


def argsort(x, axis=-1, descending=False):
    """
    Returns the indices of a tensor that give its sorted order along an axis.

    Parameters
    ----------
    x : tensor
        An input N-D Tensor
    axis : int or None
        The axis along which to sort. The default is -1, which sorts the last axis.
    descending : boolean
        Descending is a flag, if set to true, algorithm will sort by descending order, else sort by ascending order. Default is false.

    Returns
    -------
        A Tensor with the same shape as values.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = [1, 10, 26.9, 2.8, 166.32, 62.3]
    >>> y = tlx.ops.argsort(x, descending = False)
    >>> print(y)
    >>> [0, 3, 1, 2, 5, 4]

    """
    direction = 'ASCENDING'
    if descending:
        direction = 'DESCENDING'
    return tf.argsort(x, axis=axis, direction=direction)


def bmm(x, y):
    """
    Applies batched matrix multiplication to two tensors.
    Both of the two input tensors must be three-dementional and share the same batch size.
    if x is a (b, m, k) tensor, y is a (b, k, n) tensor, the output will be a (b, m, n) tensor.

    Parameters
    ----------
    x : tensor
        The input Tensor.
    y : tensor
        The input Tensor.

    Returns
    -------
        The product Tensor.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.convert_to_tensor([[[1.0, 1.0, 1.0],[2.0, 2.0, 2.0]],[[3.0, 3.0, 3.0],[4.0, 4.0, 4.0]]])
    >>> y = tlx.convert_to_tensor([[[1.0, 1.0],[2.0, 2.0],[3.0, 3.0]],[[4.0, 4.0],[5.0, 5.0],[6.0, 6.0]]])
    >>> res = tlx.ops.bmm(x, y)
    >>> print(res)
    >>> [[[6. , 6. ],
    >>> [12., 12.]],
    >>> [[45., 45.],
    >>> [60., 60.]]]

    """
    x_shape = x.shape
    y_shape = y.shape
    if not len(x_shape) == len(y_shape) == 3:
        raise ValueError(
            "x and y should be 3-dimensional. But received x's dimention: {}, y's dimention: {}".format(
                x_shape, y_shape
            )
        )
    if x_shape[2] != y_shape[1]:
        raise ValueError(
            "x's width must be equal with y's height. But received x's shape: {}, y's shape: {}".format(
                x_shape, y_shape
            )
        )
    if x_shape[0] != y_shape[0]:
        raise ValueError(
            "x's batch (shape[0]) must be equal with y's batch (shape[0]). But received x's shape: {}, y's shape: {}".
            format(x_shape, y_shape)
        )

    return tf.matmul(x, y)


def where(condition, x, y):
    """
    Return a tensor of elements selected from either x or y, depending on condition.

    Parameters
    ----------
    condition : tensor of bool
        When True (nonzero), yield x, otherwise yield y
    x : tensor
        values selected at indices where condition is True
    y : tensor
        values selected at indices where condition is False

    Returns
    -------
        A tensor of shape equal to the broadcasted shape of condition, x, y

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.convert_to_tensor([0.9, 0.1, 3.2, 1.2])
    >>> y = tlx.convert_to_tensor([1.0, 1.0, 1.0, 1.0])
    >>> res = tlx.ops.where(x>1, x, y)
    >>> print(res)
    >>> [1.0, 1.0, 3.2, 1.2]

    """

    return tf.where(condition=condition, x=x, y=y)


def ones_like(x, dtype=None):
    """
    This OP returns a Tensor filled with the value 1, with the same shape and data type (use dtype if dtype is not None) as x.

    Parameters
    ----------
    x : tensor
        The input tensor which specifies shape and dtype.
    dtype : str
        A type for the returned Tensor.If dtype is None, the data type is the same as x. Default is None.

    Returns
    -------
        A Tensor filled with the value 1, with the same shape and data type (use dtype if dtype is not None) as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.convert_to_tensor([0.9, 0.1, 3.2, 1.2])
    >>> res = tlx.ops.ones_like(x, dtype="int32")
    >>> print(res)
    >>> [1, 1, 1, 1]

    """

    return tf.ones_like(x, dtype=dtype)


def zeros_like(x, dtype=None):
    """
    This OP returns a Tensor filled with the value 0, with the same shape and data type (use dtype if dtype is not None) as x.

    Parameters
    ----------
    x : tensor
        The input tensor which specifies shape and dtype.
    dtype : str
        A type for the returned Tensor.If dtype is None, the data type is the same as x. Default is None.

    Returns
    -------
        A Tensor filled with the value 0, with the same shape and data type (use dtype if dtype is not None) as x.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.convert_to_tensor([0.9, 0.1, 3.2, 1.2])
    >>> res = tlx.ops.zeros_like(x, dtype="int32")
    >>> print(res)
    >>> [0, 0, 0, 0]

    """

    return tf.zeros_like(x, dtype=dtype)


def squeeze(x, axis=None):
    """
    Removes dimensions of size 1 from the shape of a tensor.

    Parameters
    ----------
    x : tensor
        The input Tensor.
    axis : int or list or tuple
        An integer or list/tuple of integers, indicating the dimensions to be squeezed. Default is None.
        The range of axis is [−ndim(x),ndim(x)). If axis is negative, axis=axis+ndim(x).
        If axis is None, all the dimensions of x of size 1 will be removed.

    Returns
    -------
        Squeezed Tensor with the same data type as input Tensor.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.ones(shape=[1,2,3])
    >>> res = tlx.ops.squeeze(x, axis=0)
    >>> print(res.shape)
    >>> [2, 3]

    """

    return tf.squeeze(x, axis)


def unsorted_segment_sum(x, segment_ids, num_segments):
    """Computes the sum along segments of a tensor.

    Parameters
    ----------
    x : tensor
        A Tensor.
    segment_ids : Tensor or list or tuple
        Must be one of the following types: int32, int64.
    num_segments : int or tensor
        should equal the number of distinct segment IDs.

    Returns
    -------
        A Tensor. Has the same type as data.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant([1,2,3])
    >>> res = tlx.ops.unsorted_segment_sum(x, (0, 0, 1), num_segments=2)
    >>> print(res)
    >>> [2, 3]

    """

    return tf.math.unsorted_segment_sum(x, segment_ids, num_segments)


def unsorted_segment_mean(x, segment_ids, num_segments):
    """Computes the mean along segments of a tensor.

    Parameters
    ----------
    x : tensor
        A Tensor.
    segment_ids : Tensor or list or tuple
        Must be one of the following types: int32, int64.
    num_segments : int or tensor
        should equal the number of distinct segment IDs.

    Returns
    -------
        A Tensor. Has the same type as data.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant([1.,2.,3.])
    >>> res = tlx.ops.unsorted_segment_mean(x, (0, 0, 1), num_segments=2)
    >>> print(res)
    >>> [1.5, 3]

    """

    return tf.math.unsorted_segment_mean(x, segment_ids, num_segments)


def unsorted_segment_min(x, segment_ids, num_segments):
    """Computes the min along segments of a tensor.

    Parameters
    ----------
    x : tensor
        A Tensor.
    segment_ids : Tensor or list or tuple
        Must be one of the following types: int32, int64.
    num_segments : int or tensor
        should equal the number of distinct segment IDs.

    Returns
    -------
        A Tensor. Has the same type as data.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant([1.,2.,3.])
    >>> res = tlx.ops.unsorted_segment_min(x, (0, 0, 1), num_segments=2)
    >>> print(res)
    >>> [1, 3]

    """

    return tf.math.unsorted_segment_min(x, segment_ids, num_segments)


def unsorted_segment_max(x, segment_ids, num_segments):
    """Computes the max along segments of a tensor.

    Parameters
    ----------
    x : tensor
        A Tensor.
    segment_ids : Tensor or list or tuple
        Must be one of the following types: int32, int64.
    num_segments : int or tensor
        should equal the number of distinct segment IDs.

    Returns
    -------
        A Tensor. Has the same type as data.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> x = tlx.ops.constant([1.,2.,3.])
    >>> res = tlx.ops.unsorted_segment_max(x, (0, 0, 1), num_segments=2)
    >>> print(res)
    >>> [2, 3]

    """

    return tf.math.unsorted_segment_max(x, segment_ids, num_segments)

def set_seed(seed):
    """

    Parameters
    ----------
    seed : int
        The random seed to set.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> tlx.ops.set_seed(42)
    """
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def is_tensor(x):
    """

    Parameters
    ----------
    x : input
        A python object to check.

    Returns
    -------
        a bool Value. if x is tensor return True, else return False.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> tlx.ops.is_tensor(a)
    """

    return tf.is_tensor(x)