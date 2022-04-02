#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayerx as tlx
from tensorlayerx import logging
from tensorlayerx.nn.core import Module

__all__ = [
    'GaussianNoise',
]


class GaussianNoise(Module):
    """
    The :class:`GaussianNoise` class is noise layer that adding noise with
    gaussian distribution to the activation.

    Parameters
    ------------
    mean : float
        The mean. Default is 0.0.
    stddev : float
        The standard deviation. Default is 1.0.
    is_always : boolean
        Is True, add noise for train and eval mode. If False, skip this layer in eval mode.
    seed : int or None
        The seed for random noise.
    name : str
        A unique layer name.

    Examples
    --------
    With TensorLayer

    >>> net = tlx.nn.Input([64, 200], name='input')
    >>> net = tlx.nn.Linear(in_features=200, out_features=100, act=tlx.ReLU, name='linear')(net)
    >>> gaussianlayer = tlx.nn.GaussianNoise(name='gaussian')(net)
    >>> print(gaussianlayer)
    >>> output shape : (64, 100)

    """

    def __init__(
        self,
        mean=0.0,
        stddev=1.0,
        is_always=True,
        seed=None,
        name=None,  # 'gaussian_noise',
    ):
        super().__init__(name)
        self.mean = mean
        self.stddev = stddev
        self.seed = seed
        self.is_always = is_always

        self.build()
        self._built = True

        logging.info("GaussianNoise %s: mean: %f stddev: %f" % (self.name, self.mean, self.stddev))

    def __repr__(self):
        s = '{classname}(mean={mean}, stddev={stddev}'
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs=None):
        pass

    def forward(self, inputs):
        if (self.is_train or self.is_always) is False:
            return inputs
        else:
            shapes = tlx.get_tensor_shape(inputs)
            noise = tlx.ops.random_normal(shape=shapes, mean=self.mean, stddev=self.stddev, seed=self.seed)
            outputs = inputs + noise
        return outputs
