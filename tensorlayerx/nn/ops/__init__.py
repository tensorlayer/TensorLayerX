#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
from tensorlayerx.backend.load_backend import BACKEND

if BACKEND == 'tensorflow':
    from .tensorflow_nn import *
    from .tensorflow_backend import *
elif BACKEND == 'mindspore':
    from .mindspore_nn import *
    from .mindspore_backend import *
elif BACKEND == 'paddle':
    from .paddle_nn import *
    from .paddle_backend import *
elif BACKEND == 'torch':
    from .torch_nn import *
    from .torch_backend import *
else:
    raise NotImplementedError("This backend is not supported")
