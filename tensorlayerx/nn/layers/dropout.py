#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayerx as tlx
from tensorlayerx import logging
from tensorlayerx.nn.core import Module

__all__ = [
    'Dropout',
]


class Dropout(Module):
    """
    The :class:`Dropout` class is a noise layer which randomly set some
    activations to zero according to a keeping probability.

    Parameters
    ----------
    keep : float
        The keeping probability.
        The lower the probability it is, the more activations are set to zero.
    seed : int or None
        The seed for random dropout.
    name : None or str
        A unique layer name.

    Examples
    --------
    >>> net = tlx.nn.Input([10, 200])
    >>> net = tlx.nn.Dropout(keep=0.2)(net)

    """

    def __init__(self, keep, seed=0, name=None):  #"dropout"):
        super(Dropout, self).__init__(name)
        self.keep = keep
        self.seed = seed

        self.build()
        self._built = True

        logging.info("Dropout %s: keep: %f " % (self.name, self.keep))

    def __repr__(self):
        s = ('{classname}(keep={keep}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        self.dropout = tlx.ops.Dropout(keep=self.keep, seed=self.seed)

    # @tf.function
    def forward(self, inputs):
        if self.is_train:
            outputs = self.dropout(inputs)
        else:
            outputs = inputs
        return outputs
