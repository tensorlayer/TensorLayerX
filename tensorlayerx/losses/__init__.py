#! /usr/bin/python
# -*- coding: utf-8 -*-

from tensorlayerx.backend import BACKEND

if BACKEND == 'tensorflow':
    from .tensorflow_cost import *
elif BACKEND == 'mindspore':
    from .mindspore_cost import *
elif BACKEND == 'paddle':
    from .paddle_cost import *
elif BACKEND == 'torch':
    from .torch_cost import *
elif BACKEND == 'oneflow':
    from .oneflow_cost import *
elif BACKEND == 'jittor':
    from .jittor_cost import *
else:
    raise NotImplementedError("This backend is not supported")
