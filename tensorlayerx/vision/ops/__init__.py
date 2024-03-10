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
elif BACKEND == 'oneflow':
    from .torch_ops import *
elif BACKEND == 'jittor':
    from .jittor_ops import *#TODO 
else:
    raise NotImplementedError("This backend is not supported")