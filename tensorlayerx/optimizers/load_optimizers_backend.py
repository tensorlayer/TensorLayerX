#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
from tensorlayerx import BACKEND

if BACKEND == 'tensorflow':
    from .tensorflow_optimizers import *
elif BACKEND == 'mindspore':
    from .mindspore_optimizers import *
elif BACKEND == 'paddle':
    from .paddle_optimizers import *
elif BACKEND == 'torch':
    from .torch_optimizers import *
else:
    raise NotImplementedError("This backend is not supported")
