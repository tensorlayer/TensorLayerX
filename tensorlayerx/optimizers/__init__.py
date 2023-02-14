#! /usr/bin/python
# -*- coding: utf-8 -*-

# from .amsgrad import AMSGrad

# ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD', 'Momentum', 'Lamb', 'LARS']
from .load_optimizers_backend import Adadelta
from .load_optimizers_backend import Adagrad
from .load_optimizers_backend import Adam
from .load_optimizers_backend import Adamax
from .load_optimizers_backend import Ftrl
from .load_optimizers_backend import Nadam
from .load_optimizers_backend import RMSprop
from .load_optimizers_backend import SGD
from .load_optimizers_backend import Momentum
from .load_optimizers_backend import Lamb
from .load_optimizers_backend import LARS
from tensorlayerx.optimizers import lr

__all__ = ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD', 'Momentum', 'Lamb', 'LARS']

