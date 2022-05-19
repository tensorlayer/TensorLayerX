#! /usr/bin/python
# -*- coding: utf-8 -*-
"""Deep learning and Reinforcement learning library for Researchers and Engineers."""

MAJOR = 0
MINOR = 5
PATCH = 3
PRE_RELEASE = ''
# Use the following formatting: (major, minor, patch, prerelease)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

__shortversion__ = '.'.join(map(str, VERSION[:3]))
__version__ = '.'.join(map(str, VERSION[:3])) + ''.join(VERSION[3:])

__package_name__ = 'tensorlayerx'
__contact_names__ = 'TensorLayer Contributors'
__contact_emails__ = 'tensorlayerx@gmail.com'
__homepage__ = 'https://tensorlayerx.readthedocs.io/en/latest/'
__repository_url__ = 'https://github.com/tensorlayer/TensorLayerX'
__download_url__ = 'https://github.com/tensorlayer/TensorLayerX'
__description__ = 'High Level Deep Learning Library for Researcher and Engineer.'
__license__ = 'apache'
__keywords__ = 'deep learning, machine learning, computer vision, nlp, '
__keywords__ += 'supervised learning, unsupervised learning, reinforcement learning'
