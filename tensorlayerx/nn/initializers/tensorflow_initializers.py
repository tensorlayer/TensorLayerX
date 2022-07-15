#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorlayerx as tlx

__all__ = [
    'Initializer',
    'Zeros',
    'Ones',
    'Constant',
    'RandomUniform',
    'RandomNormal',
    'TruncatedNormal',
    'deconv2d_bilinear_upsampling_initializer',
    'HeNormal',
    'HeUniform',
    'XavierNormal',
    'XavierUniform',
]


class Initializer(object):
    r"""Initializer base class: all initializers inherit from this class.
    """

    def __call__(self, shape, dtype=None):
        r"""Returns a tensor object initialized as specified by the initializer.

        Parameters
        ----------
        shape : tuple of int.
            The shape of the tensor.
        dtype : Optional dtype of the tensor.
            If not provided will return tensor of `tlx.float32`.

        Returns
        -------

        """
        raise NotImplementedError

    def get_config(self):
        r"""Returns the configuration of the initializer as a JSON-serializable dict.

        Returns
        -------
            A JSON-serializable Python dict.
        """
        return {}

    @classmethod
    def from_config(cls, config):
        r"""Instantiates an initializer from a configuration dictionary.

        Parameters
        ----------
        config : A python dictionary.
            It will typically be the output of `get_config`.

        Returns
        -------
            An Initializer instance.
        """
        if 'dtype' in config:
            config.pop('dtype')
        return cls(**config)


class Zeros(Initializer):
    r"""Initializer that generates tensors initialized to 0.

    Examples:
        >>> import tensorlayerx as tlx
        >>> init = tlx.initializers.zeros()
        >>> print(init(shape=(5, 10), dtype=tlx.float32))

    """

    def __call__(self, shape, dtype=tlx.float32):
        return tlx.ops.zeros(shape, dtype=dtype)


class Ones(Initializer):
    r"""Initializer that generates tensors initialized to 1.

    Examples:
        >>> import tensorlayerx as tlx
        >>> init = tlx.initializers.ones()
        >>> print(init(shape=(5, 10), dtype=tlx.float32))

    """

    def __call__(self, shape, dtype=tlx.float32):
        return tlx.ops.ones(shape, dtype=dtype)


class Constant(Initializer):
    r"""Initializer that generates tensors initialized to a constant value.

    Args:
        value : A python scalar or a numpy array.
            The assigned value.

    Examples:
        >>> import tensorlayerx as tlx
        >>> init = tlx.initializers.constant(value=10)
        >>> print(init(shape=(5, 10), dtype=tlx.float32))

    """

    def __init__(self, value=0):
        self.value = value

    def __call__(self, shape, dtype=tlx.float32):
        return tlx.ops.constant(self.value, shape=shape, dtype=dtype)

    def get_config(self):
        return {"value": self.value}


class RandomUniform(Initializer):
    r"""Initializer that generates tensors with a uniform distribution.

    Args:
        minval : A python scalar or a scalar tensor.
            Lower bound of the range of random values to generate.
        maxval : A python scalar or a scalar tensor.
            Upper bound of the range of random values to generate.
        seed : A Python integer.
            Used to seed the random generator.

    Examples :
        >>> import tensorlayerx as tlx
        >>> init = tlx.initializers.random_uniform(minval=-0.05, maxval=0.05)
        >>> print(init(shape=(5, 10), dtype=tlx.float32))

    """

    def __init__(self, minval=-0.05, maxval=0.05, seed=None):
        self.minval = minval
        self.maxval = maxval
        self.seed = seed

    def __call__(self, shape, dtype=tlx.float32):
        return tlx.ops.random_uniform(shape, self.minval, self.maxval, dtype=dtype, seed=self.seed)

    def get_config(self):
        return {"minval": self.minval, "maxval": self.maxval, "seed": self.seed}


class RandomNormal(Initializer):
    r"""Initializer that generates tensors with a normal distribution.

    Args:
        mean : A python scalar or a scalar tensor.
            Mean of the random values to generate.
        stddev : A python scalar or a scalar tensor.
            Standard deviation of the random values to generate.
        seed : A Python integer.
            Used to seed the random generator.


    Examples :
        >>> import tensorlayerx as tlx
        >>> init = tlx.initializers.random_normal(mean=0.0, stddev=0.05)
        >>> print(init(shape=(5, 10), dtype=tlx.float32))

    """

    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(self, shape, dtype=tlx.float32):
        return tlx.ops.random_normal(shape, self.mean, self.stddev, dtype=dtype, seed=self.seed)

    def get_config(self):
        return {"mean": self.mean, "stddev": self.stddev, "seed": self.seed}


class TruncatedNormal(Initializer):
    r"""Initializer that generates a truncated normal distribution.

    These values are similar to values from a `RandomNormal`
    except that values more than two standard deviations from the mean
    are discarded and re-drawn. This is the recommended initializer for
    neural network weights and filters.


    Args:
        mean : A python scalar or a scalar tensor.
            Mean of the random values to generate.
        stddev : A python scalar or a scalar tensor.
            Standard deviation of the andom values to generate.
        seed : A Python integer.
            Used to seed the random generator.

    Examples:

        >>> import tensorlayerx as tlx
        >>> init = tlx.initializers.truncated_normal(mean=0.0, stddev=0.05)
        >>> print(init(shape=(5, 10), dtype=tlx.float32))

    """

    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(self, shape, dtype=tlx.float32):
        return tlx.ops.truncated_normal(shape, self.mean, self.stddev, dtype=dtype, seed=self.seed)

    def get_config(self):
        return {"mean": self.mean, "stddev": self.stddev, "seed": self.seed}


class HeNormal(Initializer):
    r"""He normal initializer.

    The resulting tensor will have values sampled from :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \frac{\text{gain}}{\sqrt{fan\_mode}}

    Args:
        a : int or float
            the negative slope of the rectifier used after this layer (only used with ``'leaky_relu'``)
        mode : str
            either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity : str
            the non-linear function name, recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).
        seed : int
            Used to seed the random generator.

    Examples:

        >>> import tensorlayerx as tlx
        >>> init = tlx.initializers.HeNormal(a=0, mode='fan_out', nonlinearity='relu')
        >>> print(init(shape=(5, 10), dtype=tlx.float32))

    """

    def __init__(self, a=0, mode='fan_in', nonlinearity='leaky_relu', seed=None):
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity
        self.seed = seed

    def __call__(self, shape, dtype=tlx.float32):
        return tlx.ops.he_normal(
            shape=shape, a=self.a, mode=self.mode, nonlinearity=self.nonlinearity, dtype=dtype, seed=self.seed
        )

    def get_config(self):
        return {"a": self.a, "mode ": self.mode, "nonlinearity": self.nonlinearity}


class HeUniform(Initializer):
    r"""He uniform initializer.
    The resulting tensor will have values sampled from :math:`\mathcal{U}(-\text{bound},\text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{fan\_mode}}

    Args:
        a : int or float
            the negative slope of the rectifier used after this layer (only used with ``'leaky_relu'``)
        mode : str
            either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity : str
            the non-linear function name, recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).
        seed : int
            Used to seed the random generator.

    Examples:
        >>> import tensorlayerx as tlx
        >>> init = tlx.initializers.HeUniform(a=0, mode='fan_in', nonlinearity='relu')
        >>> print(init(shape=(5, 10), dtype=tlx.float32))
    """

    def __init__(self, a=0, mode='fan_in', nonlinearity='leaky_relu', seed=None):
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity
        self.seed = seed

    def __call__(self, shape, dtype=tlx.float32):
        return tlx.ops.he_uniform(
            shape=shape, a=self.a, mode=self.mode, nonlinearity=self.nonlinearity, dtype=dtype, seed=self.seed
        )

    def get_config(self):
        return {"a": self.a, "mode ": self.mode, "nonlinearity": self.nonlinearity}


def deconv2d_bilinear_upsampling_initializer(shape):
    r"""Returns the initializer that can be passed to DeConv2dLayer for initializing the
    weights in correspondence to channel-wise bilinear up-sampling.
    Used in segmentation approaches such as [FCN](https://arxiv.org/abs/1605.06211)

    Parameters
    ----------
    shape : tuple of int
        The shape of the filters, [height, width, output_channels, in_channels].
        It must match the shape passed to DeConv2dLayer.

    Returns
    -------
    ``tf.constant_initializer``
        A constant initializer with weights set to correspond to per channel bilinear upsampling
        when passed as W_int in DeConv2dLayer

    """
    if shape[0] != shape[1]:
        raise Exception('deconv2d_bilinear_upsampling_initializer only supports symmetrical filter sizes')

    if shape[3] < shape[2]:
        raise Exception(
            'deconv2d_bilinear_upsampling_initializer behaviour is not defined for num_in_channels < num_out_channels '
        )

    filter_size = shape[0]
    num_out_channels = shape[2]
    num_in_channels = shape[3]

    # Create bilinear filter kernel as numpy array
    bilinear_kernel = np.zeros([filter_size, filter_size], dtype=np.float32)
    scale_factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = scale_factor - 1
    else:
        center = scale_factor - 0.5
    for x in range(filter_size):
        for y in range(filter_size):
            bilinear_kernel[x, y] = (1 - abs(x - center) / scale_factor) * (1 - abs(y - center) / scale_factor)
    weights = np.zeros((filter_size, filter_size, num_out_channels, num_in_channels), dtype=np.float32)
    for i in range(num_out_channels):
        weights[:, :, i, i] = bilinear_kernel

    # assign numpy array to constant_initalizer and pass to get_variable
    return Constant(value=weights)


class XavierNormal(Initializer):
    r"""This class implements the Xavier weight initializer from the paper
    by Xavier Glorot and Yoshua Bengio.using a normal distribution.

    The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{fan\_in + fan\_out}}


    Args:
        gain : float
            an optional scaling factor
        seed : int
            Used to seed the random generator.

    """

    def __init__(self, gain=1.0, seed=None):
        self.gain = gain
        self.seed = seed

    def __call__(self, shape, dtype=tlx.float32):
        return tlx.ops.xavier_normal(shape=shape, gain=self.gain, dtype=dtype, seed=self.seed)

    def get_config(self):
        return {"gain": self.gain}


class XavierUniform(Initializer):
    r"""This class implements the Xavier weight initializer from the paper
    by Xavier Glorot and Yoshua Bengio.using a uniform distribution.

    The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{fan\_mode}}

    Args:
        gain : float
            an optional scaling factor
        seed : int
            Used to seed the random generator.

    """

    def __init__(self, gain=1.0, seed=None):
        self.gain = gain
        self.seed = seed

    def __call__(self, shape, dtype=tlx.float32):
        return tlx.ops.xavier_uniform(shape=shape, gain=self.gain, dtype=dtype, seed=self.seed)

    def get_config(self):
        return {"gain": self.gain}
