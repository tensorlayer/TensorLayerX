#! /usr/bin/python
# -*- coding: utf-8 -*-
"""Deep learning and Reinforcement learning library for Researchers and Engineers."""

MAJOR = 1
MINOR = 2
PATCH = 0
PRE_RELEASE = ''
# Use the following formatting: (major, minor, patch, prerelease)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

__shortversion__ = '.'.join(map(str, VERSION[:3]))
__version__ = '.'.join(map(str, VERSION[:3])) + ''.join(VERSION[3:])

__package_name__ = 'tensorlayer3'
__contact_names__ = 'TensorLayer Contributors'
__contact_emails__ = 'tensorlayer@gmail.com'
__homepage__ = 'https://tensorlayer3.readthedocs.io/en/latest/'
__repository_url__ = 'https://git.openi.org.cn/TensorLayer/tensorlayer3.0'
__download_url__ = 'https://git.openi.org.cn/TensorLayer/tensorlayer3.0'
__description__ = 'High Level Tensorflow Deep Learning Library for Researcher and Engineer.'
__license__ = 'apache'
__keywords__ = 'deep learning, machine learning, computer vision, nlp, '
__keywords__ += 'supervised learning, unsupervised learning, reinforcement learning, tensorflow'
