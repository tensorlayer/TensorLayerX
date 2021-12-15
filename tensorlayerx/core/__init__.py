#! /usr/bin/python
# -*- coding: utf-8 -*-

from tensorlayerx.backend import BACKEND
from .common import _save_standard_weights_dict, _load_standard_weights_dict, _save_weights, _load_weights

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
