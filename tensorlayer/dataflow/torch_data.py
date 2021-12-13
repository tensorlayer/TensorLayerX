#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset as dataset
from torch.utils.data import IterableDataset as iterabledataset

__all__ = [
    'Batch',
    'Concat',
    'FromGenerator',
    'FromSlices',
    'Map',
    'Repeat',
    'Shuffle',
    'Dataloader',
    'Dataset',
    'IterableDataset',
]


class Dataset(dataset):

    def __init__(self):
        pass

    def __getitem__(self, idx):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__getitem__', self.__class__.__name__))

    def __len__(self):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__len__', self.__class__.__name__))


class IterableDataset(iterabledataset):

    def __init__(self):
        pass

    def __iter__(self):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__iter__', self.__class__.__name__))

    def __getitem__(self, idx):
        raise RuntimeError("'{}' should not be called for IterableDataset" \
                "{}".format('__getitem__', self.__class__.__name__))

    def __len__(self):
        raise RuntimeError("'{}' should not be called for IterableDataset" \
                "{}".format('__len__', self.__class__.__name__))


def FromGenerator(generator, output_types=None, column_names=None):

    raise NotImplementedError("Not Implemented.")


def FromSlices(datas, column_names=None):

    raise NotImplementedError("Not Implemented.")


def Concat(datasets):

    raise NotImplementedError("Not Implemented.")


def Zip(datasets):

    raise NotImplementedError("Not Implemented.")


def Dataloader(dataset, batch_size=None, shuffle=False, drop_last=False, shuffle_buffer_size=0):

    raise NotImplementedError("Not Implemented.")


def Batch(dataset, batch_size, drop_last=False):

    raise NotImplementedError("Not Implemented.")


def Shuffle(dataset, buffer_size, seed=None):

    raise NotImplementedError("Not Implemented.")


def Repeat(dataset, count=None):

    raise NotImplementedError("Not Implemented.")


def Map(dataset, map_func, input_columns=None):

    raise NotImplementedError("Not Implemented.")
