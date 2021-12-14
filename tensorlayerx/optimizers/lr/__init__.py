#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
from tensorlayerx.backend.ops.load_backend import BACKEND

if BACKEND == 'tensorflow':
    from .tensorflow_lr import *
elif BACKEND == 'mindspore':
    from .mindspore_lr import *
elif BACKEND == 'paddle':
    from .paddle_lr import *
elif BACKEND == 'torch':
    from .torch_lr import *
else:
    raise NotImplementedError("This backend is not supported")
