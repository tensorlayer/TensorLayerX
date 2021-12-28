#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deep learning and Reinforcement learning library for Researchers and Engineers"""

# import backend
from .backend import *
# from .backend import ops
# import dataflow
# from .dataflow import *

import os

from tensorlayerx.package_info import (
    VERSION, __contact_emails__, __contact_names__, __description__, __download_url__, __homepage__, __keywords__,
    __license__, __package_name__, __repository_url__, __shortversion__, __version__
)

if 'TENSORLAYER_PACKAGE_BUILDING' not in os.environ:

    # from tensorlayerx import array_ops
    from tensorlayerx import losses
    from tensorlayerx import decorators
    from tensorlayerx import files
    from tensorlayerx import initializers
    from .utils import iterate
    from tensorlayerx import layers
    from tensorlayerx import lazy_imports
    from tensorlayerx import logging
    from tensorlayerx import model
    from tensorlayerx import optimizers
    # from tensorlayerx import rein
    # from tensorlayerx import utils
    from tensorlayerx import dataflow
    from tensorlayerx import metrics
    from tensorlayerx import vision

    from tensorlayerx.lazy_imports import LazyImport

    # Lazy Imports
    db = LazyImport("tensorlayerx.db")
    distributed = LazyImport("tensorlayerx.distributed")
    nlp = LazyImport("tensorlayerx.nlp")
    prepro = LazyImport("tensorlayerx.prepro")
    utils = LazyImport("tensorlayerx.utils")
    visualize = LazyImport("tensorlayerx.visualize")

    # alias
    vis = visualize

    # alphas = array_ops.alphas
    # alphas_like = array_ops.alphas_like

    # global vars
    global_flag = {}
    global_dict = {}
