#! /usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import torch
import math
import numpy as np

__all__ = [
    'LRScheduler', 'NoamDecay', 'PiecewiseDecay', 'NaturalExpDecay', 'InverseTimeDecay', 'PolynomialDecay',
    'LinearWarmup', 'ExponentialDecay', 'MultiStepDecay', 'StepDecay', 'LambdaDecay', 'ReduceOnPlateau',
    'CosineAnnealingDecay'
]


class LRScheduler(object):

    def __init__(self, learning_rate=0.1, last_epoch=-1, verbose=False):
        pass

    def __call__(self):

        raise NotImplementedError

    def step(self, epoch=None):
        pass

    def get_lr(self):
        raise NotImplementedError


class StepDecay(LRScheduler):

    def __init__(self, learning_rate, step_size, gamma=0.1, last_epoch=-1, verbose=False):
        pass

    def get_lr(self):
        raise NotImplementedError


class CosineAnnealingDecay(LRScheduler):

    def __init__(self, learning_rate, T_max, eta_min=0, last_epoch=-1, verbose=False):
        pass

    def get_lr(self):
        raise NotImplementedError

    def _get_closed_form_lr(self):
        raise NotImplementedError


class NoamDecay(LRScheduler):

    def __init__(self, d_model, warmup_steps, learning_rate=1.0, last_epoch=-1, verbose=False):
        pass

    def get_lr(self):
        raise NotImplementedError


class PiecewiseDecay(LRScheduler):

    def __init__(self, boundaries, values, last_epoch=-1, verbose=False):
        pass

    def get_lr(self):
        raise NotImplementedError


class NaturalExpDecay(LRScheduler):

    def __init__(self, learning_rate, gamma, last_epoch=-1, verbose=False):
        pass

    def get_lr(self):
        raise NotImplementedError


class InverseTimeDecay(LRScheduler):

    def __init__(self, learning_rate, gamma, last_epoch=-1, verbose=False):
        pass

    def get_lr(self):
        raise NotImplementedError


class PolynomialDecay(LRScheduler):

    def __init__(self, learning_rate, decay_steps, end_lr=0.0001, power=1.0, cycle=False, last_epoch=-1, verbose=False):
        pass

    def get_lr(self):
        raise NotImplementedError


class LinearWarmup(LRScheduler):

    def __init__(self, learning_rate, warmup_steps, start_lr, end_lr, last_epoch=-1, verbose=False):
        pass

    def get_lr(self):
        raise NotImplementedError


class ExponentialDecay(LRScheduler):

    def __init__(self, learning_rate, gamma, last_epoch=-1, verbose=False):
        pass

    def get_lr(self):
        raise NotImplementedError


class MultiStepDecay(LRScheduler):

    def __init__(self, learning_rate, milestones, gamma=0.1, last_epoch=-1, verbose=False):
        pass

    def get_lr(self):
        raise NotImplementedError


class LambdaDecay(LRScheduler):

    def __init__(self, learning_rate, lr_lambda, last_epoch=-1, verbose=False):
        pass

    def get_lr(self):
        raise NotImplementedError


class ReduceOnPlateau(LRScheduler):

    def __init__(
        self, learning_rate, mode='min', factor=0.1, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0,
        min_lr=0, epsilon=1e-8, verbose=False
    ):
        pass

    # "cooldown_counter / best / num_bad_epochs / last_epoch / last_lr" will be stored.
    def step(self, metrics, epoch=None):

        raise NotImplementedError

    def _is_better(self, current, best):
        pass
