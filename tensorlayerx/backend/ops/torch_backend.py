#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
from .torch_nn import nchw_to_nhwc, nhwc_to_nchw
import torch
import torch.nn.functional as F

_dtypeDict = {
    'DType': torch.dtype,
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64,
    'int8': torch.int8,
    'int16': torch.int16,
    'int32': torch.int32,
    'int64': torch.int64,
    'uint8': torch.uint8,
    'uint16': None,
    'uint32': None,
    'uint64': None,
    'bool': torch.bool,
    'complex64': torch.complex64,
    'complex128': torch.complex128
}

DType = torch.dtype
float16 = torch.float16
float32 = torch.float32
float64 = torch.float64
int8 = torch.int8
int16 = torch.int16
int32 = torch.int32
int64 = torch.int64
uint8 = torch.uint8
uint16 = None
uint32 = None
uint64 = None
bool = torch.bool
complex64 = torch.complex64
complex128 = torch.complex128


def set_context(**kwargs):
    raise Exception("Using TenosrFlow backend,You don't need to set context")


def get_tensor_shape(x):
    return list(x.size())


# initializers
def zeros(shape, dtype=None):
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

    return torch.zeros(size=shape, dtype=dtype)


def ones(shape, dtype=None):
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

    return torch.ones(size=shape, dtype=dtype)


def constant(value, dtype=None, shape=None):
    """
    Creates a constant tensor from a tensor-like object.

    Parameters
    ----------
    value : int
        A constant value (or list) of output type dtype.
    dtype : tensor
         The type of the elements of the resulting tensor.
    shape : tuple
        Optional dimensions of resulting tensor.

    Returns
    -------
        A Constant Tensor.

    """

    w = torch.empty(size=shape, dtype=dtype)
    return torch.nn.init.constant_(w, value)


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

    if seed is None:
        torch.random.seed()
    else:
        torch.random.manual_seed(seed)
    w = torch.randn(size=shape, dtype=dtype)
    out = w.uniform_(minval, maxval)
    return out


def random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
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

    if seed is None:
        torch.random.seed()
    else:
        torch.random.manual_seed(seed)
    w = torch.randn(size=shape, dtype=dtype)
    out = w.normal_(mean=mean, std=stddev)
    return out


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
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

    tensor = torch.empty(size=shape, dtype=dtype)
    out = torch.nn.init.trunc_normal_(tensor, mean=mean, std=stddev)
    return out


def he_normal(shape, dtype=None, seed=None):
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

    tensor = torch.empty(size=shape, dtype=dtype)
    out = torch.nn.init.kaiming_normal_(tensor)
    return out


def xavier_normal(shape, dtype=None, seed=None):
    _tensor = torch.empty(size=shape, dtype=dtype)
    return torch.nn.init.xavier_normal_(_tensor)


def xavier_uniform(shape, dtype=None, seed=None):
    _tensor = torch.empty(size=shape, dtype=dtype)
    return torch.nn.init.xavier_uniform_(_tensor)


def Variable(initial_value, name=None, trainable=True):
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
    return torch.nn.Parameter(data=initial_value, requires_grad=trainable)


class MatMul(object):

    def __init__(self):
        pass

    def __call__(self, a, b):
        return torch.matmul(a, b)


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
    return torch.matmul(a, b)


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
    """
    return torch.add(value, bias)


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
        return torch.maximum(x, y)


class Minimum(object):

    def __init__(self):
        pass

    def __call__(self, x, y):
        return torch.minimum(x, y)


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
    """

    return torch.minimum(x, y)


class FlattenReshape(object):

    def __init__(self):
        pass

    def __call__(self, inputs):
        dim = 1
        for d in get_tensor_shape(inputs)[1:]:
            dim *= d
        return torch.reshape(inputs, [-1, dim])


class Reshape(object):

    def __init__(self, shape):
        self.shape = shape

    def __call__(self, tensor):
        return torch.reshape(tensor, self.shape)


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

    return torch.reshape(tensor, shape)


class Concat(object):

    def __init__(self, axis=0):
        super(Concat, self).__init__()
        self.axis = axis

    def __call__(self, values):
        return torch.cat(tensors=values, dim=self.axis)


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

    return torch.cat(values, axis)


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

    return torch.tensor(value, dtype=dtype)


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
    return torch.sqrt(x)


class ReduceSum(object):

    def __init__(self, axis=None):
        self.axis = axis

    def __call__(self, input):
        if self.axis is not None:
            return torch.sum(input=input, dim=self.axis).values
        else:
            return torch.sum(input=input)


class ReduceMean(object):

    def __init__(self, axis=None):
        self.axis = axis

    def __call__(self, inputs):
        if self.axis is not None:
            return torch.mean(input=inputs, dim=self.axis).values
        else:
            return torch.mean(inputs)

def reduce_mean(input_tensor, axis=None):
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
        return torch.mean(input=input_tensor, dim=axis).values
    else:
        return torch.mean(input_tensor)


class ReduceMax(object):

    def __init__(self, axis=None):
        self.axis = axis

    def __call__(self, inputs):
        if self.axis is not None:
            return torch.max(input=inputs, dim=self.axis).values
        else:
            return torch.max(inputs)


def reduce_max(input_tensor, axis=None):
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
        return torch.max(input_tensor, dim=axis).values
    else:
        return torch.max(input_tensor)


def reduce_min(input_tensor, axis=None):
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
        return torch.min(input=input_tensor, dim=axis).values
    else:
        return torch.min(input_tensor)


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
                x = torch.transpose(x, 1, 2)
            elif len(x.shape) == 4 and self.paddings[0:2] + self.paddings[6:] == (0, 0, 0, 0):
                self.paddings = (self.paddings[2:6])[::-1]
                x = torch.transpose(x, 1, 3)
            elif len(x.shape) == 5 and self.paddings[0:2] + self.paddings[8:] == (0, 0, 0, 0):
                self.paddings = (self.paddings[2:8])[::-1]
                x = torch.transpose(x, 1, 4)
                print(self.paddings)
            else:
                raise NotImplementedError("Only constant padding is implemented for arbitrary dimensions.")

        out = torch.nn.functional.pad(x, self.paddings, mode=self.mode, value=self.constant_values)

        if self.mode in ['symmetric', 'reflect']:
            if len(x.shape) == 3:
                out = torch.transpose(out, 1, 2)
            if len(x.shape) == 4:
                out = torch.transpose(out, 1, 3)
            if len(x.shape) == 5:
                out = torch.transpose(out, 1, 4)
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
    pad_obj = Pad(paddings, mode, constant_values=constant_values)
    return pad_obj(tensor)


class Unstack(object):

    def __init__(self, axis, num=None):
        self.axis = axis
        self.num = num

    def __call__(self, values):
        out = []
        for o in torch.chunk(values, chunks=self.num, dim=self.axis):
            out.append(torch.squeeze(o))
        return out


class Stack(object):

    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, values):
        return torch.stack(values, dim=self.axis)


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

    return torch.stack(values, dim=axis)


class Meshgrid(object):

    def __init__(self, indexing='xy'):
        super(Meshgrid, self).__init__()
        self.index = indexing

    def __call__(self, *inputs):
        return torch.meshgrid(*inputs, indexing=self.index)


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

    return torch.meshgrid(*args)


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
    dtype : None or dtype
        The type of the elements of the resulting tensor.

    Returns
    -------
        An 1-D Tensor of type dtype.
    """

    return torch.range(start=start, end=limit, step=delta, dtype=dtype)


class ExpandDims(object):

    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, input):
        return torch.unsqueeze(input=input, dim=self.axis)


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

    return torch.unsqueeze(input, axis)


class Tile(object):

    def __init__(self):
        pass

    def __call__(self, input, multiples):
        return torch.tile(input, dims=multiples)


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

    return torch.tile(input, multiples)


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
            return torch.t(a)
        if len(a.shape) == 3:
            perm = [2, 1, 0]
        if len(a.shape) == 4:
            perm = [3, 2, 1, 0]
        if len(a.shape) == 5:
            perm = [4, 3, 2, 1, 0]
    out = torch.permute(a, perm)
    if conjugate:
        out = torch.conj_physical(out)
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
    idx = torch.zeros_like(indices[0], device=indices.device).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)
    out = torch.take(params, idx)
    return out.view(out_shape)


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

    return torch.split(value, num_or_size_splits, dim=axis)


class Floor(object):

    def __call__(self, x):
        return torch.floor(x)


def floor(x):
    return torch.floor(x)


def gather(params, indices):
    return gather_nd(params, indices)


def linspace(start, stop, num):
    return torch.linspace(start=start, end=stop, steps=num)


def slice(inputs, starts, sizes):
    raise NotImplementedError


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
            return torch.nn.functional.one_hot(inputs, self.depth)
        else:
            out = torch.nn.functional.one_hot(inputs, self.depth)
            out = cast(out, torch.float64)
            out = torch.where(out==1, self.on_value, out)
            out = torch.where(out==0, self.off_value, out)
            out = cast(out, torch.int)
            return out


class L2Normalize(object):

    def __init__(self, axis=None, epsilon=1e-12):
        self.axis = axis
        self.epsilon = epsilon

    def __call__(self, input, *args, **kwargs):
        raise NotImplementedError


class EmbeddingLookup(object):

    def __init__(self, max_norm=None):
        self.max_norm = max_norm

    def __call__(self, params, ids):
        raise NotImplementedError


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
        raise NotImplementedError


class CountNonzero(object):

    def __init__(self, keepdims=None, dtype=None):
        self.keepdims = keepdims
        self.dtype = dtype

    def __call__(self, input, axis=None):
        raise NotImplementedError


class Resize:

    def __init__(self, scale, method, antialias=False, data_format='channels_last'):
        self.method = method
        self.antialias = antialias
        self.scale = scale
        self.data_format = data_format

    def __call__(self, inputs):
        raise NotImplementedError


def resize(inputs, output_size, method, antialias):
    raise NotImplementedError


class ZeroPadding1D(object):

    def __init__(self, padding):
        raise NotImplementedError

    def __call__(self, inputs):
        raise NotImplementedError


class ZeroPadding2D(object):

    def __init__(self, padding):
        raise NotImplementedError

    def __call__(self, inputs):
        raise NotImplementedError


class ZeroPadding3D(object):

    def __init__(self, padding):
        raise NotImplementedError

    def __call__(self, inputs):
        raise NotImplementedError


class Sign(object):

    def __init__(self):
        pass

    def __call__(self, x):
        raise NotImplementedError


class Ceil(object):

    def __call__(self, x):
        raise NotImplementedError


def ceil(x):
    raise NotImplementedError


def multiply(x, y):
    raise NotImplementedError


def divide(x, y):
    raise NotImplementedError


def identity(x):
    raise NotImplementedError


class BatchToSpace(object):

    def __init__(self, block_size, crops):
        self.bolock_size = block_size
        self.crops = crops

    def __call__(self, input_x):
        raise NotImplementedError


class DepthToSpace(object):

    def __init__(self, block_size, data_format='NHWC'):
        self.block_size = block_size
        self.data_format = data_format

    def __call__(self, input):
        raise NotImplementedError


def triu(data, diagonal=0):
    raise NotImplementedError


def tril(data, diagonal=0):
    raise NotImplementedError


def abs(x):
    return torch.abs(x)


def acos(x):
    return torch.acos(x)


def acosh(x):
    return torch.acosh(x)


def angle(x):
    return torch.angle(x)


def argmax(x, axis=None, dtype='int64'):
    return torch.argmax(x, dim=axis)


def argmin(x, axis=None, dtype='int64'):
    return torch.argmin(x, dim=axis)


def asin(x):
    return torch.asin(x)


def asinh(x):
    return torch.asinh(x)


def atan(x):
    return torch.atan(x)


def atanh(x):
    return torch.atanh(x)


def cos(x):
    return torch.cos(x)


def cosh(x):
    return torch.cosh(x)


def count_nonzero(x, axis=None, keepdims=None, dtype="int64"):
    return torch.count_nonzero(x, dim=axis)


def cumprod(x, axis=0, exclusive=False, reverse=False):
    return torch.cumprod(x, dim=axis)


def cumsum(x, axis=0, exclusive=False, reverse=False):
    return torch.cumsum(x, dim=axis)


def equal(x, y):
    return torch.equal(x, y)


def exp(x):
    return torch.exp(x)


def floordiv(x, y):
    return torch.floor_divide(x, y)


def floormod(x, y):
    return torch.fmod(x, y)


def greater(x, y):
    return torch.greater(x, y)


def greater_equal(x, y):
    return torch.greater_equal(x, y)


def is_inf(x):
    return torch.isinf(x)


def is_nan(x):
    return torch.isnan(x)


def l2_normalize(x, axis=None, eps=1e-12):
    if axis == None:
        return torch.divide(x, torch.sqrt(torch.max(torch.sum(torch.pow(x, 2)))))
    return torch.divide(x, torch.sqrt(torch.max(torch.sum(torch.pow(x, 2), dim=axis))))


def less(x, y):
    return torch.less(x, y)


def less_equal(x, y):
    return torch.less_equal(x, y)


def log(x):
    return torch.log(x)


def log_sigmoid(x):
    return torch.log(1 / (1 + torch.exp(-x)))


def maximum(x, y):
    return torch.maximum(x, y)


def negative(x):
    return torch.negative(x)


def not_equal(x, y):
    return torch.not_equal(x, y)


def pow(x, y):
    return torch.pow(x, y)


def real(x):
    return torch.real(x)


def reciprocal(x):
    return torch.reciprocal(x)


def reduce_prod(x, axis=None, keepdims=False):
    raise NotImplementedError


def reduce_std(x, axis=None, keepdims=False):
    raise NotImplementedError


def reduce_sum(x, axis=None, keepdims=False):
    raise NotImplementedError


def reduce_variance(x, axis=None, keepdims=False):
    raise NotImplementedError


def round(x):
    raise NotImplementedError


def rsqrt(x):
    raise NotImplementedError


def segment_max(x, segment_ids):
    raise NotImplementedError


def segment_mean(x, segment_ids):
    raise NotImplementedError


def segment_min(x, segment_ids):
    raise NotImplementedError


def segment_prod(x, segment_ids):
    raise NotImplementedError


def segment_sum(x, segment_ids):
    raise NotImplementedError


def sigmoid(x):
    raise NotImplementedError


def sign(x):
    raise NotImplementedError


def sin(x):
    raise NotImplementedError


def sinh(x):
    raise NotImplementedError


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
    return F.softplus(x)


def square(x):
    raise NotImplementedError


def squared_difference(x, y):
    raise NotImplementedError


def subtract(x, y):
    raise NotImplementedError


def tan(x):
    raise NotImplementedError


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
    raise NotImplementedError


def all(x, axis=None, keepdims=False):
    raise NotImplementedError


def logical_and(x, y):
    raise NotImplementedError


def logical_or(x, y):
    raise NotImplementedError


def logical_not(x):
    raise NotImplementedError


def logical_xor(x, y):
    raise NotImplementedError


def argsort(x, axis=-1, descending=False):
    raise NotImplementedError


def bmm(x, y):
    raise NotImplementedError


def where(condition, x, y):
    raise NotImplementedError


def ones_like(x, dtype=None):
    raise NotImplementedError


def zeros_like(x, dtype=None):
    raise NotImplementedError


def squeeze(x, axis=None):
    raise NotImplementedError


def unsorted_segment_sum(x, segment_ids, num_segments):
    raise NotImplementedError


def unsorted_segment_mean(x, segment_ids, num_segments):
    raise NotImplementedError


def unsorted_segment_min(x, segment_ids, num_segments):
    raise NotImplementedError


def unsorted_segment_max(x, segment_ids, num_segments):
    raise NotImplementedError
