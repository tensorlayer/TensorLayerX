#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
from tensorlayerx.backend import BACKEND

if BACKEND == 'tensorflow':
    from .tensorflow_initializers import *
elif BACKEND == 'mindspore':
    from .mindspore_initializers import *
elif BACKEND == 'paddle':
    from .paddle_initializers import *
elif BACKEND == 'torch':
    from .torch_initializers import *
else:
    raise NotImplementedError("This backend is not supported")
