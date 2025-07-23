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
    During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution.
    Each channel will be zeroed out independently on every forward call.

    Parameters
    ----------
    p : float
        probability of an element to be zeroed. Default: 0.5
    seed : int or None
        The seed for random dropout.
    name : None or str
        A unique layer name.

    Examples
    --------
    >>> net = tlx.nn.Input([10, 200])
    >>> net = tlx.nn.Dropout(p=0.2)(net)

    """

    def __init__(self, p=0.5, seed=0, name=None):  #"dropout"):
        super(Dropout, self).__init__(name)
        self.p = p
        self.seed = seed

        self.build()
        self._built = True

        logging.info("Dropout %s: p: %f " % (self.name, self.p))

    def __repr__(self):
        s = ('{classname}(p={p}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        self.dropout = tlx.ops.Dropout(p=self.p, seed=self.seed)

    def forward(self, inputs):
        if self.is_train:
            outputs = self.dropout(inputs)
        else:
            outputs = inputs

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs
