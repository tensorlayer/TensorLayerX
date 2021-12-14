#! /usr/bin/python
# -*- coding: utf-8 -*-

from tensorlayerx.backend import BACKEND

if BACKEND == 'mindspore':
    from .core_mindspore import *
elif BACKEND == 'tensorflow':
    from .core_tensorflow import *
elif BACKEND == 'paddle':
    from .core_paddle import *
elif BACKEND == 'torch':
    from .core_torch import *
else:
    raise ("Unsupported backend:", BACKEND)
