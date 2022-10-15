#! /usr/bin/python
# -*- coding: utf-8 -*-

import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F


def nchw_to_nhwc(x):
    """
    Channels first to channels last

    Parameters
    ----------
    x : tensor
        channels first tensor data

    Returns
    -------
        channels last tensor data
    """

    if len(x.shape) == 3:
        x = flow.transpose(x, 0, 2, 1)
    elif len(x.shape) == 4:
        x = flow.transpose(x, 0, 2, 3, 1)
    elif len(x.shape) == 5:
        x = flow.transpose(x, 0, 2, 3, 4, 1)
    else:
        raise Exception("Not support shape: {}".format(x.shape))
    return x

def nhwc_to_nchw(x):
    """
    Channles last to channels first

    Parameters
    ----------
    x : tensor
        channels last tensor data

    Returns
    -------
        channels first tensor data
    """

    if len(x.shape) == 3:
        x = flow.transpose(x, 0, 2, 1)
    elif len(x.shape) == 4:
        x = flow.transpose(x, 0, 3, 1, 2)
    elif len(x.shape) == 5:
        x = flow.transpose(x, 0, 4, 1, 2, 3)
    else:
        raise Exception("Not support shape: {}".format(x.shape))
    return x
    