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
    # from .utils import iterate
    from .utils import lazy_imports
    from . import utils
    # from .text import nlp
    from tensorlayerx import logging
    from tensorlayerx import model
    from tensorlayerx import optimizers
    from tensorlayerx import dataflow
    from tensorlayerx import metrics
    from tensorlayerx import vision

    from tensorlayerx.utils.lazy_imports import LazyImport

    # Lazy Imports
    # db = LazyImport("tensorlayerx.utils.db")
    # distributed = LazyImport("tensorlayerx.utils.distributed")
    # nlp = LazyImport("tensorlayerx.text.nlp")
    prepro = LazyImport("tensorlayerx.utils.prepro")
    utils = LazyImport("tensorlayerx.utils")
    visualize = LazyImport("tensorlayerx.utils.visualize")

    # alias
    vis = visualize

    # alphas = array_ops.alphas
    # alphas_like = array_ops.alphas_like

    # global vars
    global_flag = {}
    global_dict = {}
