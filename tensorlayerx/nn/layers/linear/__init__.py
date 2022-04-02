#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
TensorLayer provides rich layer implementations trailed for
various benchmarks and domain-specific problems. In addition, we also
support transparent access to native TensorFlow parameters.
For example, we provide not only layers for local response normalization, but also
layers that allow user to apply ``tf.ops.lrn`` on ``network.outputs``.
More functions can be found in `TensorFlow API <https://www.tensorflow.org/versions/master/api_docs/index.html>`__.
"""

from .base_linear import *
from .binary_linear import *
from .dorefa_linear import *
from .dropconnect import *
from .quan_linear import *
from .quan_linear_bn import *
from .ternary_linear import *

__all__ = [
    'BinaryLinear',
    'Linear',
    'DorefaLinear',
    'DropconnectLinear',
    'TernaryLinear',
    'QuanLinear',
    'QuanLinearWithBN',
]