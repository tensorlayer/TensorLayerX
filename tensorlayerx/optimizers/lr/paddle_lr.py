#! /usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import paddle

__all__ = [
    'LRScheduler', 'NoamDecay', 'PiecewiseDecay', 'NaturalExpDecay', 'InverseTimeDecay', 'PolynomialDecay',
    'LinearWarmup', 'ExponentialDecay', 'MultiStepDecay', 'StepDecay', 'LambdaDecay', 'ReduceOnPlateau',
    'CosineAnnealingDecay'
]

LRScheduler = paddle.optimizer.lr.LRScheduler
NoamDecay = paddle.optimizer.lr.NoamDecay
PiecewiseDecay = paddle.optimizer.lr.PiecewiseDecay
NaturalExpDecay = paddle.optimizer.lr.NaturalExpDecay
InverseTimeDecay = paddle.optimizer.lr.InverseTimeDecay
PolynomialDecay = paddle.optimizer.lr.PolynomialDecay
LinearWarmup = paddle.optimizer.lr.LinearWarmup
ExponentialDecay = paddle.optimizer.lr.ExponentialDecay
MultiStepDecay = paddle.optimizer.lr.MultiStepDecay
StepDecay = paddle.optimizer.lr.StepDecay
LambdaDecay = paddle.optimizer.lr.LambdaDecay
ReduceOnPlateau = paddle.optimizer.lr.ReduceOnPlateau
CosineAnnealingDecay = paddle.optimizer.lr.CosineAnnealingDecay
