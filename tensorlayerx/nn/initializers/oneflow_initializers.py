#! /usr/bin/python
# -*- coding: utf-8 -*-

import oneflow as flow
import tensorlayerx as tlx
import numpy as np

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
    """Initializer base class: all initializers inherit from this class.
    """

    def __call__(self, shape, dtype=None):
        """Returns a tensor object initialized as specified by the initializer.

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
        """Returns the configuration of the initializer as a JSON-serializable dict.

        Returns
        -------
            A JSON-serializable Python dict.
        """
        return {}

    @classmethod
    def from_config(cls, config):
        """Instantiates an initializer from a configuration dictionary.

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
    """Initializer that generates tensors initialized to 0.

    Examples
    --------

    >>> import tensorlayerx as tlx
    >>> init = tlx.initializers.zeros()
    >>> print(init(shape=(5, 10), dtype=tlx.float32))

    """

    def __call__(self, shape, dtype=tlx.float32):
        _tensor = flow.empty(size=shape, dtype=dtype)
        return flow.nn.init.zeros_(_tensor)


class Ones(Initializer):
    """Initializer that generates tensors initialized to 1.

    Examples
    --------

    >>> import tensorlayerx as tlx
    >>> init = tlx.initializers.ones()
    >>> print(init(shape=(5, 10), dtype=tlx.float32))

    """

    def __call__(self, shape, dtype=tlx.float32):
        _tensor = flow.empty(size=shape, dtype=dtype)
        return flow.nn.init.ones_(_tensor)


class Constant(Initializer):
    """Initializer that generates tensors initialized to a constant value.

    Parameters
    ----------
    value : A python scalar or a numpy array.
        The assigned value.

    Examples
    --------

    >>> import tensorlayerx as tlx
    >>> init = tlx.initializers.constant(value=10)
    >>> print(init(shape=(5, 10), dtype=tlx.float32))

    """

    def __init__(self, value=0):
        self.value = value

    def __call__(self, shape, dtype=tlx.float32):
        _tensor = flow.empty(size=shape, dtype=dtype)
        if isinstance(self.value, (int, float)):
            return flow.nn.init.constant_(_tensor, val=self.value)
        elif isinstance(self.value, (flow.Tensor, list, np.ndarray)):
            _tensor.data = flow.as_tensor(self.value)
            return _tensor

    def get_config(self):
        return {"value": self.value}


class RandomUniform(Initializer):
    """Initializer that generates tensors with a uniform distribution.

    Parameters
    ----------
    minval : A python scalar or a scalar tensor.
        Lower bound of the range of random values to generate.
    maxval : A python scalar or a scalar tensor.
        Upper bound of the range of random values to generate.
    seed : A Python integer.
        Used to seed the random generator.

    Examples
    --------

    >>> import tensorlayerx as tlx
    >>> init = tlx.initializers.random_uniform(minval=-0.05, maxval=0.05)
    >>> print(init(shape=(5, 10), dtype=tlx.float32))

    """

    def __init__(self, minval=-0.05, maxval=0.05, seed=None):
        self.minval = minval
        self.maxval = maxval
        self.seed = seed

    def __call__(self, shape, dtype=tlx.float32):
        _tensor = flow.empty(size=shape, dtype=dtype)
        return flow.nn.init.uniform_(_tensor, a=self.minval, b=self.maxval)

    def get_config(self):
        return {"minval": self.minval, "maxval": self.maxval, "seed": self.seed}


class RandomNormal(Initializer):
    """Initializer that generates tensors with a normal distribution.

    Parameters
    ----------
    mean : A python scalar or a scalar tensor.
        Mean of the random values to generate.
    stddev : A python scalar or a scalar tensor.
        Standard deviation of the random values to generate.
    seed : A Python integer.
        Used to seed the random generator.

    minval=-0.05, maxval=0.05

    Examples
    --------

    >>> import tensorlayerx as tlx
    >>> init = tlx.initializers.random_normal(mean=0.0, stddev=0.05)
    >>> print(init(shape=(5, 10), dtype=tlx.float32))

    """

    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(self, shape, dtype=tlx.float32):
        _tensor = flow.empty(size=shape)
        return flow.nn.init.normal_(_tensor, mean=self.mean, std=self.stddev)

    def get_config(self):
        return {"mean": self.mean, "stddev": self.stddev, "seed": self.seed}


class TruncatedNormal(Initializer):
    """Initializer that generates a truncated normal distribution.

    These values are similar to values from a `RandomNormal`
    except that values more than two standard deviations from the mean
    are discarded and re-drawn. This is the recommended initializer for
    neural network weights and filters.


    Parameters
    ----------
    mean : A python scalar or a scalar tensor.
        Mean of the random values to generate.
    stddev : A python scalar or a scalar tensor.
        Standard deviation of the andom values to generate.
    seed : A Python integer.
        Used to seed the random generator.

    Examples
    --------

    >>> import tensorlayerx as tlx
    >>> init = tlx.initializers.truncated_normal(mean=0.0, stddev=0.05)
    >>> print(init(shape=(5, 10), dtype=tlx.float32))

    """

    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(self, shape, dtype=tlx.float32):
        _tensor = flow.empty(size=shape)
        return self._truncated_normal(_tensor, self.mean, self.stddev)

    def _truncated_normal(self, tensor, mean=0, std=0.09):
        with flow.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4, )).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def get_config(self):
        return {"mean": self.mean, "stddev": self.stddev, "seed": self.seed}


class HeNormal(Initializer):
    """He normal initializer.

    Parameters
    ----------
    seed : A Python integer.
        Used to seed the random generator.

    Examples
    --------

    >>> import tensorlayerx as tlx
    >>> init = tlx.initializers.he_normal()
    >>> print(init(shape=(5, 10), dtype=tlx.float32))

    """

    def __init__(self, a=0, mode='fan_in', nonlinearity='leaky_relu', seed=None):
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity
        self.seed = seed

    def __call__(self, shape, dtype=tlx.float32):
        return tlx.ops.he_normal(shape=shape, a=self.a, mode=self.mode, nonlinearity=self.nonlinearity, dtype=dtype)

    def get_config(self):
        return {"a": self.a, "mode ": self.mode, "nonlinearity": self.nonlinearity}


class HeUniform(Initializer):
    """He uniform initializer.

    Parameters
    ----------
    seed : A Python integer.
        Used to seed the random generator.

    Examples
    --------

    >>> import tensorlayerx as tlx
    >>> init = tlx.initializers.he_normal()
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
    """Returns the initializer that can be passed to DeConv2dLayer for initializing the
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
    raise NotImplementedError


class XavierNormal(Initializer):
    """This class implements the Xavier weight initializer from the paper
    by Xavier Glorot and Yoshua Bengio.using a normal distribution.

    Parameters
    ----------
    seed : A Python integer.
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
    """This class implements the Xavier weight initializer from the paper
    by Xavier Glorot and Yoshua Bengio.using a uniform distribution.

    Parameters
    ----------
    seed : A Python integer.
        Used to seed the random generator.

    """

    def __init__(self, gain=1.0, seed=None):
        self.gain = gain
        self.seed = seed

    def __call__(self, shape, dtype=tlx.float32):
        return tlx.ops.xavier_uniform(shape=shape, gain=self.gain, dtype=dtype, seed=self.seed)

    def get_config(self):
        return {"gain": self.gain}
