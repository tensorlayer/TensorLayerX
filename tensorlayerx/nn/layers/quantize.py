#! /usr/bin/python
# -*- coding: utf-8 -*-

from tensorlayerx import logging
from tensorlayerx.nn.core import Module
from tensorlayerx.nn.layers.utils import quantize

__all__ = [
    'Sign',
]


class Sign(Module):
    """The :class:`SignLayer` class is for quantizing the layer outputs to -1 or 1 while inferencing.

    Parameters
    ----------
    name : a str
        A unique layer name.

    """

    # @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
        self,
        name=None  # 'sign',
    ):
        super().__init__(name)
        logging.info("Sign  %s" % self.name)

        self.build()
        self._built = True

    def build(self, inputs_shape=None):
        pass

    def __repr__(self):
        s = ('{classname}(')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs):
        outputs = quantize(inputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs
