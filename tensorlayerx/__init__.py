#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deep learning and Reinforcement learning library for Researchers and Engineers"""

# import backend
from .backend import *

import os

from tensorlayerx.package_info import (
    VERSION, __contact_emails__, __contact_names__, __description__, __download_url__, __homepage__, __keywords__,
    __license__, __package_name__, __repository_url__, __shortversion__, __version__
)

if 'TENSORLAYER_PACKAGE_BUILDING' not in os.environ:

    from tensorlayerx import nn
    from .nn import core
    from .nn import layers
    from .nn import initializers
    from tensorlayerx import losses
    from tensorlayerx import decorators
    from tensorlayerx import files
    from tensorlayerx import logging
    from tensorlayerx import model
    from tensorlayerx import optimizers
    from tensorlayerx import dataflow
    from tensorlayerx import metrics
    from tensorlayerx import vision

    from tensorlayerx.utils.lazy_imports import LazyImport

    # global vars
    global_flag = {}
    global_dict = {}
