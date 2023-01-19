#! /usr/bin/python
# -*- coding: utf-8 -*-
from .dataset import Dataset, IterableDataset
from .sampler import Sampler, SequentialSampler, RandomSampler, BatchSampler, SubsetRandomSampler, WeightedRandomSampler
from .utils import _DatasetKind, _InfiniteIterableSampler
from . import utils
import math
__all__ = [
    'DataLoader',
]


class DataLoader(object):
    """ Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.

    The :class:`tensorlayerx.dataflow.DataLoader` supports both map-style and
    iterable-style datasets with single- or multi-process loading, customizing
    loading order and optional automatic batching

    Parameters
    -----------
    dataset : Dataset
        dataset from which to load the data.
    batch_size : int
        how many samples per batch to load, default is 1.
    shuffle : bool
        set to ``True`` to have the data reshuffled at every epoch, default is ``False``.
    drop_last : bool
        set to ``True`` to drop the last incomplete batch,
        if the dataset size is not divisible by the batch size. If ``False`` and
        the size of dataset is not divisible by the batch size, then the last batch
        will be smaller. default is ``False``.
    sampler : Sampler
        defines the strategy to draw samples from the dataset. If specified, `shuffle` must not be specified.
    batch_sampler : Sampler
        returns a batch of indices at a time. If specified, `shuffle`, `batch_size`, `drop_last`, `sampler` must not be specified.
    num_workers : int
        how many subprocesses to use for data loading. ``0`` means that the data will be loaded in single process. default is ``0``.
    collate_fn : callable
        merges a list of samples to form a mini-batch of Tensor(s).  Used when using batched loading from a map-style dataset.
    time_out : numeric
        if positive, the timeout value for collecting a batch from workers. Should always be non-negative. default is ``0``.
    worker_init_fn : callable
        If not ``None``, this will be called on each worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
        input, after seeding and before data loading. default is ``None``.
    prefetch_factor : int
        Number of samples loaded in advance by each worker.
        ``2`` means there will be a total of 2 * num_workers samples prefetched across all workers. default is ``2``
    persistent_workers : bool
        If ``True``, the data loader will not shutdown the worker processes after a dataset has been consumed once.
        This allows to maintain the workers `Dataset` instances alive. default is ``False``.
    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        time_out=0,
        worker_init_fn=None,
        prefetch_factor=2,
        persistent_workers=False,
    ):
        self.dataset = dataset
        assert num_workers >= 0, "num_workers should be a non_negative integer"
        # if num_workers == 0 and prefetch_factor != 2:
        #     raise ValueError("prefetch_factor option should not be specified, when num_workers is 0.")
        if persistent_workers and num_workers == 0:
            raise ValueError('persistent_workers option needs num_workers > 0')
        self.num_workers = 0 # TODO optimizer multiprocess in multi backends
        self.prefetch_factor = 2
        self.time_out = time_out
        self.worker_init_fn = worker_init_fn
        if isinstance(dataset, IterableDataset):
            self._dataset_kind = _DatasetKind.Iter
            if shuffle is not False:
                raise ValueError("IterableDataset only support 'shuffle=False', but got shuffle={}.".format(shuffle))
            elif sampler is not None:
                raise ValueError("IterableDataset only support 'sampler=None', but got sampler={}.".format(sampler))
            elif batch_sampler is not None:
                raise ValueError(
                    "IterableDataset only support 'batch_sampler=None', "
                    "but got batch_sampler={}.".format(batch_sampler)
                )
        else:
            self._dataset_kind = _DatasetKind.Map

        if sampler is not None and shuffle:
            raise ValueError("sampler option is mutually exclusive with shuffle option.")

        if batch_sampler is not None:
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError(
                    "batch_size, shuffle, sampler, drop_last should not be set, when batch_sampler is specified."
                )
            batch_size = None
            drop_last = False
        elif batch_size is None:
            if drop_last:
                raise ValueError("drop_last should be False, when batch_size is None.")

        if sampler is None:
            if self._dataset_kind == _DatasetKind.Iter:
                sampler = _InfiniteIterableSampler()
            else:
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)

        if batch_size is not None and batch_sampler is None:
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self._iterator = None
        if collate_fn is None:
            if self._is_batch:
                collate_fn = utils.default_collate
            else:
                collate_fn = utils.default_convert

        self.collate_fn = collate_fn
        self.persistent_workers = persistent_workers

    @property
    def _is_batch(self):
        return self.batch_sampler is not None

    @property
    def _index_sampler(self):
        if self._is_batch:
            return self.batch_sampler
        else:
            return self.sampler

    def _get_iterator(self):
        if self.num_workers == 0:
            return utils._SingleProcessDataLoaderIter(self)
        else:
            return utils._MultiProcessingDataLoaderIter(self)

    def __iter__(self):

        if self.persistent_workers and self.num_workers > 0:
            if self._iterator is None:

                self._iterator = self._get_iterator()
            else:
                self._iterator._reset(self)
            return self._iterator
        else:
            return self._get_iterator()

    def __len__(self):
        if self._dataset_kind == _DatasetKind.Iter:
            length = len(self.dataset)
            if self.batch_size is not None:
                if self.drop_last:
                    length = length // self.batch_size
                else:
                    length = math.ceil(length / self.batch_size)
            return length
        else:
            return len(self._index_sampler)
