#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import paddle
import paddle as pd
import paddle.nn as nn
import numpy as np
import paddle.nn.functional as F
from .paddle_nn import nchw_to_nhwc, nhwc_to_nchw, preprocess_2d_format, preprocess_1d_format, preprocess_3d_format
import random
from paddle.fluid.framework import in_dygraph_mode, default_main_program
from paddle.fluid import framework, core, unique_name
from paddle.fluid.core import VarDesc
from paddle.fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype
import math

if paddle.__version__ < '2.2.2':
    _dtypeDict = [
        "float16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool",
        "complex64", "complex128"
    ]
    # TODO NotImplemented
    DType = None
    float16 = "float16"
    float32 = "float32"
    float64 = "float64"
    int8 = "int8"
    int16 = "int16"
    int32 = "int32"
    int64 = "int64"
    uint8 = "uint8"
    uint16 = "uint16"
    uint32 = "uint32"
    uint64 = "uint64"
    bool = "bool"
    complex64 = "complex64"
    complex128 = "complex128"
else:
    _dtypeDict = {
        'DType': paddle.dtype ,
        'float16': paddle.float16,
        'float32': paddle.float32,
        'float64': paddle.float64,
        'int8': paddle.int8,
        'int16': paddle.int16,
        'int32': paddle.int32,
        'int64': paddle.int64,
        'uint8': paddle.uint8,
        'uint16': None,
        'uint32': None,
        'uint64': None,
        'bool': paddle.bool,
        'complex64': paddle.complex64,
        'complex128': paddle.complex128
    }
    # TODO NotImplemented
    DType = paddle.dtype
    float16 = paddle.float16
    float32 = paddle.float32
    float64 = paddle.float64
    int8 = paddle.int8
    int16 = paddle.int16
    int32 = paddle.int32
    int64 = paddle.int64
    uint8 = paddle.uint8
    uint16 = None
    uint32 = None
    uint64 = None
    bool = paddle.bool
    complex64 = paddle.complex64
    complex128 = paddle.complex128


def _getter(init_fn, **kwargs):
    """Return an named eager tensor."""
    raise NotImplementedError


def set_context(**kwargs):
    raise Exception("Using Paddle backend,You don't need to set context")


def get_tensor_shape(x):
    return list(x.shape)


# initializers
def zeros(shape, dtype="float32", device = None):
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
    return pd.zeros(shape=shape, dtype=dtype)


def ones(shape, dtype="float32", device = None):
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
    return pd.ones(shape=shape, dtype=dtype)


def constant(value, dtype="float32", shape=None, device = None):
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
    return pd.full(fill_value=value, dtype=dtype, shape=shape)


def random_uniform(shape, minval=-1.0, maxval=1.0, dtype="float32", seed=0):
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
         Used in combination with dragon.random.set_seed to create a reproducible sequence of tensors across multiple calls.
    Returns
    -------
        A tensor of the specified shape filled with random uniform values.

    """
    return pd.uniform(shape=shape, min=minval, max=maxval, dtype=dtype, seed=seed)


def random_normal(shape, mean=0.0, stddev=1.0, dtype="float32", seed=None):
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
    return pd.normal(mean, stddev, shape)


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype="float32", seed=None):
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
    raise NotImplementedError

def _compute_fans(var):
    shape = var.shape
    if not shape or len(shape) == 0:
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        # This is the case for simple matrix multiply
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        receptive_field_size = np.prod(shape[2:])
        fan_in = shape[1] * receptive_field_size
        fan_out = shape[0] * receptive_field_size

    return fan_in, fan_out

def _check_block(block):
    if block is None:
        block = default_main_program().global_block()
    return block

def calculate_gain(nonlinearity, param=None):
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif isinstance(param, int) or isinstance(param, float):
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))

def he_normal(var, a = 0, mode = 'fan_in', nonlinearity='leaky_relu', dtype='float32', seed=None, block = None):
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
    _uniform = False
    block = _check_block(block)
    assert isinstance(var, framework.Variable)
    assert isinstance(block, framework.Block)
    f_in, f_out = _compute_fans(var)
    correct_fan = f_in if mode == 'fan_in' else f_out
    if seed is None:
        seed = block.program.random_seed

    if var.dtype == VarDesc.VarType.FP16 or (
            var.dtype == VarDesc.VarType.BF16 and not _uniform):
        out_dtype = VarDesc.VarType.FP32
        out_var = block.create_var(
            name=unique_name.generate(".".join(
                ['masra_init', var.name, 'tmp'])),
            shape=var.shape,
            dtype=out_dtype,
            type=VarDesc.VarType.LOD_TENSOR,
            persistable=False)
    else:
        out_dtype = var.dtype
        out_var = var

    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(correct_fan)
    op = block.append_op(
        type="gaussian_random",
        outputs={"Out": out_var},
        attrs={
            "shape": out_var.shape,
            "dtype": int(out_dtype),
            "mean": 0.0,
            "std": std,
            "seed": seed
        },
        stop_gradient=True)

    if var.dtype == VarDesc.VarType.FP16 or (
            var.dtype == VarDesc.VarType.BF16 and not _uniform):
        block.append_op(
            type="cast",
            inputs={"X": out_var},
            outputs={"Out": var},
            attrs={"in_dtype": out_var.dtype,
                   "out_dtype": var.dtype})

    if not framework.in_dygraph_mode():
        var.op = op
    return op

def he_uniform(var, a = 0, mode = 'fan_in', nonlinearity='leaky_relu', dtype='float32', seed=None, block = None):
    _uniform = True
    block = _check_block(block)
    assert isinstance(var, framework.Variable)
    assert isinstance(block, framework.Block)
    f_in, f_out = _compute_fans(var)
    correct_fan = f_in if mode == 'fan_in' else f_out
    if seed is None:
        seed = block.program.random_seed

    if var.dtype == VarDesc.VarType.FP16 or (
            var.dtype == VarDesc.VarType.BF16 and not _uniform):
        out_dtype = VarDesc.VarType.FP32
        out_var = block.create_var(
            name=unique_name.generate(".".join(
                ['masra_init', var.name, 'tmp'])),
            shape=var.shape,
            dtype=out_dtype,
            type=VarDesc.VarType.LOD_TENSOR,
            persistable=False)
    else:
        out_dtype = var.dtype
        out_var = var

    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(correct_fan)
    limit = math.sqrt(3.0) * std
    op = block.append_op(
        type="uniform_random",
        inputs={},
        outputs={"Out": out_var},
        attrs={
            "shape": out_var.shape,
            "dtype": int(out_dtype),
            "min": -limit,
            "max": limit,
            "seed": seed
        },
        stop_gradient=True)

    if var.dtype == VarDesc.VarType.FP16 or (
            var.dtype == VarDesc.VarType.BF16 and not _uniform):
        block.append_op(
            type="cast",
            inputs={"X": out_var},
            outputs={"Out": var},
            attrs={"in_dtype": out_var.dtype,
                   "out_dtype": var.dtype})

    if not framework.in_dygraph_mode():
        var.op = op
    return op

def xavier_normal(var, gain = 1.0, dtype='float32', seed=None, block = None):
    """
    xavier normal initializer.

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
        A tensor of the specified shape filled with xavier normal values.
    """

    block = _check_block(block)
    _uniform = False
    assert isinstance(block, framework.Block)
    check_variable_and_dtype(var, "Out",
                             ["uint16", "float16", "float32", "float64"],
                             "xavier_init")

    fan_in, fan_out = _compute_fans(var)
    if seed is None:
        seed = block.program.random_seed
    if var.dtype == VarDesc.VarType.FP16 or (
            var.dtype == VarDesc.VarType.BF16 and not _uniform):
        out_dtype = VarDesc.VarType.FP32
        out_var = block.create_var(
            name=unique_name.generate(".".join(
                ['xavier_init', var.name, 'tmp'])),
            shape=var.shape,
            dtype=out_dtype,
            type=VarDesc.VarType.LOD_TENSOR,
            persistable=False)
    else:
        out_dtype = var.dtype
        out_var = var

    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    op = block.append_op(
        type="gaussian_random",
        outputs={"Out": out_var},
        attrs={
            "shape": out_var.shape,
            "dtype": out_dtype,
            "mean": 0.0,
            "std": std,
            "seed": seed
        },
        stop_gradient=True)

    if var.dtype == VarDesc.VarType.FP16 or (
            var.dtype == VarDesc.VarType.BF16 and not _uniform):
        block.append_op(
            type="cast",
            inputs={"X": out_var},
            outputs={"Out": var},
            attrs={"in_dtype": out_var.dtype,
                   "out_dtype": var.dtype})

    if not framework.in_dygraph_mode():
        var.op = op
    return op

def xavier_uniform(var, gain = 1.0, dtype='float32', seed=None, block = None):
    """
    xavier uniform initializer.

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
        A tensor of the specified shape filled with xavier uniform values.
    """

    block = _check_block(block)
    _uniform = True
    assert isinstance(block, framework.Block)
    check_variable_and_dtype(var, "Out",
                             ["uint16", "float16", "float32", "float64"],
                             "xavier_init")

    fan_in, fan_out = _compute_fans(var)
    if seed is None:
        seed = block.program.random_seed
    if var.dtype == VarDesc.VarType.FP16 or (
            var.dtype == VarDesc.VarType.BF16 and not _uniform):
        out_dtype = VarDesc.VarType.FP32
        out_var = block.create_var(
            name=unique_name.generate(".".join(
                ['xavier_init', var.name, 'tmp'])),
            shape=var.shape,
            dtype=out_dtype,
            type=VarDesc.VarType.LOD_TENSOR,
            persistable=False)
    else:
        out_dtype = var.dtype
        out_var = var

    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    limit = math.sqrt(3.0) * std
    op = block.append_op(
        type="uniform_random",
        inputs={},
        outputs={"Out": out_var},
        attrs={
            "shape": out_var.shape,
            "dtype": out_dtype,
            "min": -limit,
            "max": limit,
            "seed": seed
        },
        stop_gradient=True)

    if var.dtype == VarDesc.VarType.FP16 or (
            var.dtype == VarDesc.VarType.BF16 and not _uniform):
        block.append_op(
            type="cast",
            inputs={"X": out_var},
            outputs={"Out": var},
            attrs={"in_dtype": out_var.dtype,
                   "out_dtype": var.dtype})

    if not framework.in_dygraph_mode():
        var.op = op
    return op


def Variable(initial_value, name, trainable=None, device = None):
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
    raise NotImplementedError


class MatMul(object):

    def __init__(self, transpose_a=False, transpose_b=False):
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b

    def __call__(self, a, b):
        return pd.matmul(x=a, y=b, transpose_x=self.transpose_a, transpose_y=self.transpose_b)


def matmul(a, b, transpose_a=False, transpose_b=False):
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
    return pd.matmul(x=a, y=b, transpose_x=transpose_a, transpose_y=transpose_b)


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

    return pd.add(value, bias)


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
    return dt.dtype


class Maximum(object):

    def __init__(self):
        pass

    def __call__(self, x, y):
        raise NotImplementedError


class Minimum(object):

    def __init__(self):
        pass

    def __call__(self, x, y):
        return pd.minimum(x, y)


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
    return pd.minimum(x, y)


class FlattenReshape(object):

    def __init__(self):
        pass

    def __call__(self, inputs):
        return pd.flatten(x=inputs, start_axis=1, stop_axis=-1)


class Reshape(object):

    def __init__(self, shape):
        self.shape = shape

    def __call__(self, tensor):
        return pd.reshape(tensor, shape=self.shape)


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
    return pd.reshape(tensor, shape)


class Concat(object):

    def __init__(self, axis=0):
        super(Concat, self).__init__()
        self.axis = axis

    def __call__(self, values):
        return pd.concat(values, axis=self.axis)


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
    return pd.concat(values, axis)


def convert_to_tensor(value, dtype=None, device = None):
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
    return pd.to_tensor(value, dtype=dtype)


def convert_to_numpy(value):
    return value.numpy()


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
    return pd.sqrt(x)


class ReduceSum(object):

    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def construct(self, input):
        return pd.sum(input, axis=self.axis, keepdim=self.keepdims)


class ReduceMean(object):

    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def __call__(self, inputs):
        return pd.mean(inputs, axis=self.axis, keepdim=self.keepdims)


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

    return pd.mean(input_tensor, axis, keepdim=keepdims)


class ReduceMax(object):

    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def __call__(self, inputs):
        return pd.max(inputs, axis=self.axis, keepdim=self.keepdims)


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

    return pd.max(input_tensor, axis, keepdim=keepdims)


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
    return pd.min(input_tensor, axis, keepdim=keepdims)


class Pad(object):

    def __init__(self, paddings, mode="REFLECT", constant_values=0):
        if mode not in ['CONSTANT', 'REFLECT', 'SYMMETRIC']:
            raise Exception("Unsupported mode: {}".format(mode))
        if mode == 'SYMMETRIC':
            raise NotImplementedError
        self.paddings = paddings
        self.mode = mode.lower()
        self.constant_values = constant_values

    def __call__(self, x):
        if len(x.shape) == 3:
            data_format = 'NLC'
            self.paddings = self.correct_paddings(len(x.shape), self.paddings, data_format)
        elif len(x.shape) == 4:
            data_format = 'NHWC'
            self.paddings = self.correct_paddings(len(x.shape), self.paddings, data_format)
        elif len(x.shape) == 5:
            data_format = 'NDHWC'
            self.paddings = self.correct_paddings(len(x.shape), self.paddings, data_format)
        else:
            raise NotImplementedError('Please check the input shape.')
        return pd.nn.functional.pad(x, self.paddings, self.mode, value=self.constant_values, data_format=data_format)

    def correct_paddings(self, in_shape, paddings, data_format):
        if in_shape == 3 and data_format == 'NLC':
            correct_output = [paddings[1][0], paddings[1][1]]
        elif in_shape == 4 and data_format == 'NHWC':
            correct_output = [paddings[2][0], paddings[2][1], paddings[1][0], paddings[1][1]]
        elif in_shape == 5 and data_format == 'NDHWC':
            correct_output = [
                paddings[3][0], paddings[3][1], paddings[2][0], paddings[2][1], paddings[1][0], paddings[1][1]
            ]
        else:
            raise NotImplementedError('Does not support channels first')
        return correct_output


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
    return Pad(paddings, mode, constant_values)(tensor)


class Unstack(object):

    def __init__(self, axis=0, num=None):
        self.axis = axis
        self.num = num

    def __call__(self, values):
        return pd.unstack(values, self.axis, self.num)


class Stack(object):

    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, values):
        return pd.stack(values, self.axis)


def stack(values, axis=0):
    """
    Stacks a list of rank-R tensors into one rank-(R+1) tensor.

    Parameters
    ----------
    values : list or tuple
        A list of Tensor objects with the same shape and type.
    axis : int
        An int. The axis to stack along. Defaults to the first dimension.
        Negative values wrap around, so the valid range is [-(R+1), R+1).

    Returns
    -------
        A stacked Tensor with the same type as values.
    """
    return pd.stack(values, axis=axis)


class Meshgrid(object):

    def __init__(self, indexing='xy'):
        super(Meshgrid, self).__init__()
        self.index = indexing

    def __call__(self, inputs):
        return pd.meshgrid(inputs)


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

    return pd.meshgrid(*args, **kwargs)


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
    return pd.arange(start, end = limit, step=delta, dtype=dtype)


class ExpandDims(object):

    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, input):

        return pd.unsqueeze(input, axis=self.axis)


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

    return pd.unsqueeze(input, axis)


class Tile(object):

    def __init__(self):
        pass

    def __call__(self, input, multiples):
        return pd.tile(input, multiples)


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
    return pd.tile(input, multiples)


class Cast(object):

    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, input):
        return pd.cast(input, self.dtype)


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
    return pd.cast(x, dtype)


class Transpose(object):

    def __init__(self, perm, conjugate=False):
        self.perm = perm
        if conjugate:
            raise ("The conjugate Parameters not supported")

    def __call__(self, a):
        return pd.transpose(a, self.perm)


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
    if perm == None:
        if len(a.shape) <= 2:
            return pd.t(a)
        if len(a.shape) == 3:
            perm = [2, 1, 0]
        if len(a.shape) == 4:
            perm = [3, 2, 1, 0]
        if len(a.shape) == 5:
            perm = [4, 3, 2, 1, 0]
    return pd.transpose(a, perm)


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

    return pd.gather_nd(params, indices)


def scatter_nd(indices, updates, shape):
    raise NotImplementedError


class ClipGradByValue(pd.nn.ClipGradByValue):
    def __init__(self, clip_min=-1, clip_max=1):
        super().__init__(max=clip_max, min=clip_min)


class ClipGradByNorm(pd.nn.ClipGradByNorm):
    def __init__(self, clip_norm=0.1):
        super().__init__(clip_norm=clip_norm)


class ClipByGlobalNorm(pd.nn.ClipGradByGlobalNorm):
    def __init__(self, clip_norm=0.1):
        super().__init__(clip_norm=clip_norm)


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

    return pd.clip(t, clip_value_min, clip_value_max)


def split(value, num_or_size_splits, axis=0):
    """
    Splits a tensor into sub tensors.

    Parameters
    ----------
    value : tensor
        The Tensor to split.
    num_or_size_splits : list or tuple
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
    return pd.split(value, num_or_size_splits, axis)


class Floor(object):

    def __call__(self, x):
        return pd.floor(x)


def floor(x):
    return pd.floor(x)


def gather(params, indices, axis=None):
    if axis is None:
        axis = 0
    if axis < 0:
        axis = len(params.shape) + axis
    return pd.gather(params, indices, axis)


def linspace(start, stop, num):
    return pd.linspace(start, stop, num)


def slice(inputs, axes, starts, sizes):
    return pd.slice(inputs, axes=axes, starts=starts, ends=sizes)


def add_n(inputs):
    return pd.add_n(inputs)


class OneHot(object):

    def __init__(self, axis=-1, depth=1, on_value=1.0, off_value=0.0, dtype="float32"):
        self.depth = depth
        self.dtype = dtype

    def __call__(self, indices):
        output = pd.nn.functional.one_hot(indices, self.depth)
        return output


class L2Normalize(object):

    def __init__(self, axis=None, epsilon=1e-12):
        super(L2Normalize, self).__init__()
        self.axis = axis
        self.epsilon = epsilon

    def __call__(self, input):
        return pd.nn.functional.normalize(x=input, p=2, axis=self.axis, epsilon=self.epsilon)


class EmbeddingLookup(object):

    def __init__(self, max_norm=None):
        self.max_norm = max_norm

    def __call__(self, params, ids):
        return F.embedding(ids, params)


class NCELoss(object):

    def __init__(self, num_true=1, sampled_values=None, remove_accidental_hits=False):
        super(NCELoss, self).__init__()
        self.num_true = num_true
        self.sampled_values = sampled_values
        self.remove_accidental_hits = remove_accidental_hits

    def __call__(self, weights, biases, labels, inputs, num_sampled, num_classes):
        # TODO need to be updated
        if weights or biases is not None:
            raise NotImplementedError("Only Xavier initialization is supported.")
        return pd.static.nn.nce(input=inputs, label=labels, num_total_classes=num_classes)


class NotEqual(object):

    def __init__(self):
        pass

    def __call__(self, x, y):
        return pd.not_equal(x, y)


class CountNonzero(object):

    def __init__(self, keepdims=False, dtype="int64"):
        self.keepdims = keepdims

    def __call__(self, input, axis=None):
        return pd.nonzero(input, as_tuple=self.keepdims)


class Resize:

    def __init__(self, scale, method, antialias=False, data_format='channels_last'):
        if method not in ['nearest', 'linear', 'bilinear']:
            raise ('Current resize does not support this method.')
        self.method = method
        self.antialias = antialias
        self.scale = scale
        self.data_format, _ = preprocess_2d_format(data_format, None)

    def __call__(self, inputs):
        output_size = [int(inputs.shape[1] * self.scale[0]), int(inputs.shape[2] * self.scale[1])]
        out = F.interpolate(
            inputs, size=output_size, mode=self.method, data_format=self.data_format, align_corners=self.antialias
        )
        return out


def resize(inputs, output_size, method, antialias):
    return Resize(output_size, method, antialias)(inputs)


def channels_switching(data_format, dim='2d', padding=None):
    if dim == '1d':
        if data_format == 'channels_first':
            out = 'NCL'
        if data_format == 'channels_last':
            out = 'NLC'
        pads = padding
    if dim == '2d':
        if data_format == 'channels_first':
            out = 'NCHW'
        if data_format == 'channels_last':
            out = 'NHWC'
        pads = [padding[1][0], padding[1][1], padding[0][0], padding[0][1]]
    if dim == '3d':
        if data_format == 'channels_first':
            out = 'NCDHW'
        if data_format == 'channels_last':
            out = 'NDHWC'
        pads = [padding[2][0], padding[2][1],
                padding[1][0], padding[1][1],
                padding[0][0], padding[0][1]]
    return out, pads


class ZeroPadding1D(object):

    def __init__(self, padding, data_format):
        self.padding = padding
        self.data_format = data_format

    def __call__(self, inputs):
        data_format, padding = channels_switching(self.data_format, '1d', self.padding)
        out = pd.nn.functional.pad(inputs, padding, mode='constant', value=0.0, data_format=data_format)
        return out


class ZeroPadding2D(object):

    def __init__(self, padding, data_format):
        self.padding = padding
        self.data_format = data_format

    def __call__(self, inputs):
        data_format, padding = channels_switching(self.data_format, '2d', self.padding)
        out = pd.nn.functional.pad(inputs, padding, mode='constant', value=0.0, data_format=data_format)
        return out


class ZeroPadding3D(object):

    def __init__(self, padding, data_format):
        self.padding = padding
        self.data_format = data_format

    def __call__(self, inputs):
        data_format, padding = channels_switching(self.data_format, '3d', self.padding)
        out = pd.nn.functional.pad(inputs, padding, mode='constant', value=0.0, data_format=data_format)
        return out


class Sign(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return pd.sign(x)


class Ceil(object):

    def __call__(self, x):
        return pd.ceil(x)


def ceil(x):
    return pd.ceil(x)


def multiply(x, y):
    return pd.multiply(x, y)


def divide(x, y):
    return pd.divide(x, y)


def identity(x):

    return x


class BatchToSpace(object):

    def __init__(self, block_size, crops):
        super(BatchToSpace, self).__init__()
        pass

    def __call__(self, input_x):
        raise NotImplementedError


class DepthToSpace(object):

    def __init__(self, block_size, data_format='NHWC'):
        self.block_size = block_size
        self.data_format, _ = preprocess_2d_format(data_format, None)

    def __call__(self, input):

        return pd.nn.functional.pixel_shuffle(input, self.block_size, self.data_format)


def triu(data, diagonal=0):

    return pd.triu(data, diagonal)


def tril(data, diagonal=0):

    return pd.tril(data, diagonal)


def abs(x):

    return pd.abs(x)


def acos(x):

    return pd.acos(x)


def angle(x):
    x_np = convert_to_numpy(x)
    return convert_to_tensor(np.angle(x_np))


def acosh(x):
    return pd.log(x + pd.sqrt(pd.pow(x, 2) - 1))


def argmax(x, axis=None, dtype='int64'):
    return pd.argmax(x, axis=axis, dtype=dtype)


def argmin(x, axis=None, dtype='int64'):
    return pd.argmin(x, axis=axis, dtype=dtype)


def asin(x):
    return pd.asin(x)


def asinh(x):
    return pd.log(x + pd.sqrt(pd.pow(x, 2) + 1))


def atan(x):
    return pd.atan(x)


def atanh(x):
    return 0.5 * pd.log(pd.divide((1.0 + x), (1.0 - x)))


def cos(x):
    return pd.cos(x)


def cosh(x):
    return pd.cosh(x)


def count_nonzero(x, axis=None, keepdims=None, dtype="int64"):
    _nonzero = pd.nonzero(x, as_tuple=True)
    if axis == None:
        return pd.prod(pd.shape(_nonzero[0]))
    x_n = convert_to_numpy(x)
    if isinstance(axis, list):
        axis = tuple(axis)
    non_zero = np.count_nonzero(x_n, axis=axis)
    return convert_to_tensor(non_zero)


def cumprod(x, axis=0, exclusive=False, reverse=False):
    x = convert_to_numpy(x)
    prod = np.cumprod(x, axis=axis)
    return convert_to_tensor(prod)


def cumsum(x, axis=0, exclusive=False, reverse=False):
    return pd.cumsum(x, axis=axis)


def equal(x, y):
    return pd.equal(x, y)


def exp(x):
    return pd.exp(x)


def floordiv(x, y):
    return pd.floor_divide(x, y)


def floormod(x, y):
    return pd.floor_mod(x, y)


def greater(x, y):
    return pd.greater_than(x, y)


def greater_equal(x, y):
    return pd.greater_equal(x, y)


def is_inf(x):
    return pd.isinf(x)


def is_nan(x):
    return pd.isnan(x)


def l2_normalize(x, axis=None, eps=1e-12):
    axis = 0 if axis is None else axis
    return F.normalize(x, p=2.0, axis=axis, epsilon=eps)


def less(x, y):
    return pd.less_than(x, y)


def less_equal(x, y):
    return pd.less_equal(x, y)


def log(x):
    return pd.log(x)


def log_sigmoid(x):
    return pd.log(1 / (1 + pd.exp(-x)))


def maximum(x, y):
    return pd.maximum(x, y)


def negative(x):
    return -x


def not_equal(x, y):
    return pd.not_equal(x, y)


def pow(x, y):
    return pd.pow(x, y)


def real(x):
    return pd.real(x)


def reciprocal(x):
    return pd.reciprocal(x)


def reduce_prod(x, axis=None, keepdims=False):

    return pd.prod(x, axis=axis, keepdim=keepdims)


def reduce_std(x, axis=None, keepdims=False):

    return pd.std(x, axis=axis, keepdim=keepdims)


def reduce_sum(x, axis=None, keepdims=False):

    return pd.sum(x, axis=axis, keepdim=keepdims)


def reduce_variance(x, axis=None, keepdims=False):

    return pd.var(x, axis=axis, keepdim=keepdims)


def round(x):

    return pd.round(x)


def rsqrt(x):
    return pd.rsqrt(x)


def segment_max(x, segment_ids):
    segment_ids = pd.to_tensor(segment_ids, dtype=pd.int32)
    return pd.incubate.segment_max(x, segment_ids)


def segment_mean(x, segment_ids):
    segment_ids = pd.to_tensor(segment_ids, dtype=pd.int32)
    return pd.incubate.segment_mean(x, segment_ids)


def segment_min(x, segment_ids):
    segment_ids = pd.to_tensor(segment_ids, dtype=pd.int32)
    return pd.incubate.segment_min(x, segment_ids)


def segment_prod(x, segment_ids):
    raise NotImplementedError


def segment_sum(x, segment_ids):
    segment_ids = pd.to_tensor(segment_ids, dtype=pd.int32)
    return pd.incubate.segment_sum(x, segment_ids)


def sigmoid(x):
    return pd.nn.functional.sigmoid(x)


def sign(x):
    return pd.sign(x)


def sin(x):
    return pd.sin(x)


def sinh(x):
    return pd.sinh(x)


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

    return F.softplus(x)


def square(x):
    return pd.square(x)


def squared_difference(x, y):
    return pd.square(x - y)


def subtract(x, y):
    return pd.subtract(x, y)


def tan(x):
    return pd.tan(x)


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

    return F.tanh(x)


def any(x, axis=None, keepdims=False):

    return pd.any(x, axis=axis, keepdim=keepdims)


def all(x, axis=None, keepdims=False):

    return pd.all(x, axis=axis, keepdim=keepdims)


def logical_and(x, y):

    return pd.logical_and(x, y)


def logical_or(x, y):

    return pd.logical_or(x, y)


def logical_not(x):

    return pd.logical_not(x)


def logical_xor(x, y):

    return pd.logical_xor(x, y)


def argsort(x, axis=-1, descending=False):

    return pd.argsort(x, axis=axis, descending=descending)


def bmm(x, y):

    return pd.bmm(x, y)


def where(condition, x, y):

    return pd.where(condition, x, y)


def ones_like(x, dtype=None):

    return pd.ones_like(x, dtype)


def zeros_like(x, dtype=None):

    return pd.zeros_like(x, dtype)


def squeeze(x, axis=None):

    return pd.squeeze(x, axis)


def unsorted_segment_sum(x, segment_ids, num_segments):
    segment_ids = pd.to_tensor(segment_ids, dtype=pd.int64)
    assert x.shape[0] == segment_ids.shape[0], "the length of segment_ids should be equal to data.shape[0]."
    res = []
    for i in range(num_segments):
        a = pd.sum(x[segment_ids == i], axis=0)
        res.append(a)
    if res[0].shape == [1]:
        return pd.concat(res, axis = 0)
    else:
        return pd.stack(res, axis=0)


def unsorted_segment_mean(x, segment_ids, num_segments):
    segment_ids = pd.to_tensor(segment_ids, dtype=pd.int64)
    assert x.shape[0] == segment_ids.shape[0], "the length of segment_ids should be equal to data.shape[0]."
    res = []
    for i in range(num_segments):
        a = pd.mean(x[segment_ids == i], axis=0)
        res.append(a)
    if res[0].shape == [1]:
        return pd.concat(res, axis = 0)
    else:
        return pd.stack(res, axis=0)


def unsorted_segment_min(x, segment_ids, num_segments):
    segment_ids = pd.to_tensor(segment_ids, dtype=pd.int64)
    assert x.shape[0] == segment_ids.shape[0], "the length of segment_ids should be equal to data.shape[0]."
    res = []
    for i in range(num_segments):
        a = pd.min(x[segment_ids == i], axis=0)
        res.append(a)
    if res[0].shape == [1]:
        return pd.concat(res, axis = 0)
    else:
        return pd.stack(res, axis=0)


def unsorted_segment_max(x, segment_ids, num_segments):
    segment_ids = pd.to_tensor(segment_ids, dtype=pd.int64)
    assert x.shape[0] == segment_ids.shape[0], "the length of segment_ids should be equal to data.shape[0]."
    res = []
    for i in range(num_segments):
        a = pd.max(x[segment_ids == i], axis=0)
        res.append(a)
    if res[0].shape == [1]:
        return pd.concat(res, axis = 0)
    else:
        return pd.stack(res, axis=0)

def set_seed(seed):

    pd.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def is_tensor(x):

    return pd.is_tensor(x)


def tensor_scatter_nd_update(tensor, indices, updates):
    tensor = paddle.to_tensor(tensor)
    indices = paddle.to_tensor(indices)
    updates = paddle.to_tensor(updates)
    a = pd.scatter_nd(indices, pd.ones_like(updates), tensor.shape)
    a = pd.multiply(tensor, -a)
    tensor = tensor + a
    x = pd.scatter_nd_add(tensor, indices, updates)
    return x

def diag(input, diagonal=0):

    return paddle.diag(input, diagonal)

def mask_select(x, mask, axis = 0):
    def _apply_mask_1d(reshaped_tensor, mask, axis=None):
        indices = paddle.nonzero(paddle.cast(mask, paddle.int32), as_tuple=True)
        return paddle.gather(reshaped_tensor, indices, axis=axis)
    shape_mask = mask.shape
    ndims_mask = len(shape_mask)
    if ndims_mask == 0:
        raise ValueError("mask cannot be scalar.")
    if ndims_mask is None:
        raise ValueError(
                "Number of mask dimensions must be specified, even if some dimensions"
                " are None.  E.g. shape=[None] is ok, but shape=None is not.")
    axis = 0 if axis is None else axis
    leading_size = np.prod(x.shape[axis:axis + ndims_mask], 0)
    tensor = paddle.reshape(
            x,
            list(np.concatenate([
                x.shape[:axis], [leading_size],
                x.shape[axis + ndims_mask:]
            ], 0)))
    mask = paddle.reshape(mask, [-1])
    return _apply_mask_1d(tensor, mask, axis)

def eye(n, m=None, dtype=None):
    return paddle.eye(n, m, dtype)


def einsum(equation, *operands):
    try:
        from paddlenlp.ops import einsum
    except:
        raise Exception("Paddlenlp needs to be installed.")
    return einsum(equation, *operands)


class Einsum(object):
    def __init__(self, equation):
        super(Einsum, self).__init__()
        try:
            from paddlenlp.ops import einsum
        except:
            raise Exception("Paddlenlp needs to be installed.")
        self.equation = equation

    def __call__(self, *args):
        return einsum(self.equation, *args)


def set_device(device = 'GPU', id = 0):
    device = device.lower()
    if device == 'gpu':
        device = device + ':' + str(id)
    paddle.device.set_device(device)

def scatter_update(tensor, indices, updates):

    return pd.scatter(tensor, indices, updates)

def get_device():

    return paddle.device.get_device()

def to_device(tensor, device = 'GPU', id = 0):
    device = device.upper()
    if device == 'GPU':
        return paddle.to_tensor(tensor, place=paddle.CUDAPlace(id))
    if device == 'CPU':
        return paddle.to_tensor(tensor, place=paddle.CPUPlace())

def roll(input, shifts, dims=None):

    return paddle.roll(input, shifts, dims)

def logsoftmax(input, dim = None):
    if dim is None:
        dim = -1
    return F.log_softmax(input, dim)


def topk(input, k, dim=None, largest=True, sorted=True):

    return paddle.topk(input, k, axis=dim, largest=largest, sorted=sorted)

def numel(input):

    return paddle.numel(input)


def histogram(input, bins=100, min=0, max=0, name=None):
    return paddle.histogram(input, bins=bins, min=min, max=max, name=name)


def flatten(x, start_axis=0, stop_axis=-1, name=None):
    return paddle.flatten(x, start_axis=start_axis, stop_axis=stop_axis, name=name)


def interpolate(x,
                size=None,
                scale_factor=None,
                mode='nearest',
                align_corners=False,
                align_mode=0,
                data_format='NCHW',
                name=None):
    return paddle.nn.functional.interpolate(x,
                size=size,
                scale_factor=scale_factor,
                mode=mode,
                align_corners=align_corners,
                align_mode=align_mode,
                data_format=data_format,
                name=name)


def index_select(x, index, axis=0, name=None):
    return paddle.index_select(x, index, axis=axis, name=name)


def dot(x, y, name=None):
    return paddle.dot(x, y, name=name)


class Swish(object):
    def __init__(self):
        pass

    def __call__(self, x):
        return paddle.nn.functional.swish(x)

