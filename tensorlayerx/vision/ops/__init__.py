#! /usr/bin/python
# -*- coding: utf-8 -*-

from tensorlayerx.backend import BACKEND

if BACKEND == 'tensorflow':
    from .tensorflow_ops import *
elif BACKEND == 'mindspore':
    from .mindspore_ops import *
elif BACKEND == 'paddle':
    from .paddle_ops import *
elif BACKEND == 'torch':
    from .torch_ops import *
else:
    raise NotImplementedError("This backend is not supported")