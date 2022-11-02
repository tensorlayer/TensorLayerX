#! /usr/bin/python
# -*- coding: utf-8 -*-
import os
from tensorlayerx.backend import BACKEND
from .sampler import Sampler
import collections
import numpy as np
import numbers
import itertools
import multiprocessing
import queue
from collections import namedtuple
from dataclasses import dataclass
import sys
import traceback
import re
import tensorlayerx as tlx
np_str_obj_array_pattern = re.compile(r"[SaUO]")
string_classes = (str, bytes)

def default_convert(data):
    elem_type = type(data)
    if elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        # array of string classes and object
        if elem_type.__name__ == 'ndarray' \
                and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return tlx.ops.convert_to_tensor(data)
    elif isinstance(data, collections.abc.Mapping):
        try:
            return elem_type({key: default_convert(data[key]) for key in data})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(default_convert(d) for d in data))
    elif isinstance(data, tuple):
        return [default_convert(d) for d in data]  # Backwards compatibility.
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, string_classes):
        try:
            return elem_type([default_convert(d) for d in data])
        except TypeError:
            # The sequence type may not support `__init__(iterable)` (e.g., `range`).
            return [default_convert(d) for d in data]
    else:
        return data


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

def default_collate_tf(batch):
    import tensorflow as tf
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, tf.Tensor):
        return tf.stack(batch, 0)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([tf.convert_to_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return tf.convert_to_tensor(batch)
    elif isinstance(elem, float):
        return tf.convert_to_tensor(batch, dtype=tf.float64)
    elif isinstance(elem, int):
        return tf.convert_to_tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: default_collate([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [default_collate(samples) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([default_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))

def default_collate_torch(batch):
    import torch
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel, device=elem.device)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: default_collate([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [default_collate(samples) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([default_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def default_collate_paddle(batch):
    import paddle
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, paddle.Tensor):
        return paddle.stack(batch, 0)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([paddle.to_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return paddle.to_tensor(batch)
    elif isinstance(elem, float):
        return paddle.to_tensor(batch, dtype=paddle.float64)
    elif isinstance(elem, int):
        return paddle.to_tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: default_collate([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [default_collate(samples) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([default_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))



def default_collate_ms(batch):
    import mindspore as ms
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, ms.Tensor):
        return ms.ops.stack(batch, 0)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([ms.Tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return ms.Tensor(batch)
    elif isinstance(elem, float):
        return ms.Tensor(batch, dtype=ms.float64)
    elif isinstance(elem, int):
        return ms.Tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: default_collate([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [default_collate(samples) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([default_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def default_collate_flow(batch):
    import oneflow as flow
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, (flow.Tensor, flow._oneflow_internal.Tensor)):
        return flow._C.stack(batch, dim=0)
    elif (
            elem_type.__module__ == "numpy"
            and elem_type.__name__ != "str_"
            and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([flow.tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return flow.tensor(batch)
    elif isinstance(elem, float):
        return flow.tensor(batch, dtype=flow.float64)
    elif isinstance(elem, int):
        return flow.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def default_collate(batch):
    if BACKEND == 'tensorflow':
        return default_collate_tf(batch)
    elif BACKEND == 'torch':
        return default_collate_torch(batch)
    elif BACKEND == 'paddle':
        return default_collate_paddle(batch)
    elif BACKEND == 'mindspore':
        return default_collate_ms(batch)
    elif BACKEND == 'oneflow':
        return default_collate_flow(batch)


class _DatasetKind(object):

    Map = 0
    Iter = 1

    @staticmethod
    def create_fetcher(kind, dataset, is_batch, collate_fn, drop_last):
        if kind == _DatasetKind.Map:
            return _MapDatasetFetcher(dataset, is_batch, collate_fn, drop_last)
        else:
            return _IterableDatasetFetcher(dataset, is_batch, collate_fn, drop_last)


class _BaseDatasetFetcher(object):

    def __init__(self, dataset, is_batch, collate_fn, drop_last):
        self.dataset = dataset
        self.is_batch = is_batch
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def fetch(self, batch_indices):
        raise NotImplementedError("'fetch' not implement for class {}".format(self.__class__.__name__))


class _MapDatasetFetcher(_BaseDatasetFetcher):

    def __init__(self, dataset, is_batch, collate_fn, drop_last):
        super(_MapDatasetFetcher, self).__init__(dataset, is_batch, collate_fn, drop_last)

    def fetch(self, batch_indices):
        if self.is_batch:
            data = [self.dataset[id] for id in batch_indices]
        else:
            data = self.dataset[batch_indices]
        return self.collate_fn(data)


class _IterableDatasetFetcher(_BaseDatasetFetcher):

    def __init__(self, dataset, is_batch, collate_fn, drop_last):
        super(_IterableDatasetFetcher, self).__init__(dataset, is_batch, collate_fn, drop_last)
        self.dataset_iter = iter(dataset)

    def fetch(self, batch_indices):
        if self.is_batch:
            data = []
            for _ in batch_indices:
                try:
                    data.append(next(self.dataset_iter))
                except StopIteration:
                    break
            if len(data) == 0 or (self.drop_last and len(data) < len(batch_indices)):
                raise StopIteration
        else:
            data = next(self.dataset_iter)
        return self.collate_fn(data)


class _InfiniteIterableSampler(Sampler):

    def __init__(self):
        super(_InfiniteIterableSampler, self).__init__()

    def __iter__(self):
        while True:
            yield None


class _BaseDataLoaderIter(object):

    def __init__(self, loader):
        self._dataset = loader.dataset
        self._dataset_kind = loader._dataset_kind
        self._is_batch = loader._is_batch
        self._drop_last = loader.drop_last
        self._index_sampler = loader._index_sampler
        self._num_workers = loader.num_workers
        self._prefetch_factor = loader.prefetch_factor
        self._collate_fn = loader.collate_fn
        self._persistent_workers = loader.persistent_workers
        self._time_out = loader.time_out
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0

    def __iter__(self):
        return self

    def _reset(self, loader, first_iter=False):
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0

    def _next_index(self):
        return next(self._sampler_iter)

    def _next_data(self):

        raise NotImplementedError

    def __next__(self):
        if self._sampler_iter is None:
            self._reset()
        data = self._next_data()
        self._num_yielded += 1
        return data

    def __len__(self):
        return len(self._index_sampler)


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):

    def __init__(self, loader):
        super(_SingleProcessDataLoaderIter, self).__init__(loader)
        assert self._time_out == 0
        assert self._num_workers == 0

        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind, self._dataset, self._is_batch, self._collate_fn, self._drop_last
        )

    def _next_data(self):
        index = self._next_index()
        data = self._dataset_fetcher.fetch(index)
        return data


class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):

    def __init__(self, loader):
        super(_MultiProcessingDataLoaderIter, self).__init__(loader)
        assert self._num_workers > 0
        assert self._prefetch_factor > 0
        self._shutdown = False
        self._worker_init_fn = loader.worker_init_fn
        self._worker_queue_idx_cycle = itertools.cycle(range(self._num_workers))
        self._worker_result_queue = multiprocessing.Queue()
        self._worker_done_event = multiprocessing.Event()
        self._worker_pids_set = False

        self._index_queues = []
        self._workers = []

        for i in range(self._num_workers):
            index_queue = multiprocessing.Queue()
            index_queue.cancel_join_thread()
            w = multiprocessing.Process(
                target=_worker_loop, args=(
                    self._dataset_kind, self._dataset, index_queue, self._worker_result_queue, self._worker_done_event,
                    self._is_batch, self._collate_fn, self._worker_init_fn, i, self._drop_last
                )
            )
            w.daemon = True
            w.start()
            self._index_queues.append(index_queue)
            self._workers.append(w)

        self._data_queue = self._worker_result_queue
        self._reset(loader, first_iter=True)

    def _reset(self, loader, first_iter=False):
        super()._reset(loader, first_iter)
        self._send_idx = 0
        self._rcvd_idx = 0
        self._task_info = {}
        self._tasks_outstanding = 0
        self._workers_status = [True for i in range(self._num_workers)]
        if not first_iter:
            for idx in range(self._num_workers):
                self._index_queues[idx].put(_ResumeIteration())
            resume_iteration_cnt = self._num_workers
            while resume_iteration_cnt > 0:
                return_idx, return_data = self._get_data()
                if isinstance(return_idx, _ResumeIteration):
                    assert return_data is None
                    resume_iteration_cnt -= 1
        for _ in range(self._prefetch_factor * self._num_workers):

            self._try_put_index()

    def _try_get_data(self, timeout=5.0):
        try:
            data = self._data_queue.get(timeout=timeout)
            return (True, data)
        except Exception as e:
            failed_workers = []
            for worker_id, w in enumerate(self._workers):
                if self._workers_status[worker_id] and not w.is_alive():
                    failed_workers.append(w)
                    self._mark_worker_as_unavailable(worker_id)
            if len(failed_workers) > 0:
                pids_str = ', '.join(str(w.pid) for w in failed_workers)
                raise RuntimeError('DataLoader worker (pid(s) {}) exited unexpectedly'.format(pids_str)) from e

            if isinstance(e, queue.Empty):
                return (False, None)

    def _get_data(self):
        if self._time_out > 0:
            success, data = self._try_get_data(self._time_out)
            if success:
                return data
            else:
                raise RuntimeError('DataLoader timed out after {} seconds'.format(self._time_out))
        else:
            while True:
                success, data = self._try_get_data()
                if success:
                    return data

    def _mark_worker_as_unavailable(self, worker_id, shutdown=False):

        assert self._workers_status[worker_id] or (self._persistent_workers and shutdown)
        q = self._index_queues[worker_id]
        q.put(None)
        self._workers_status[worker_id] = False
        assert self._worker_done_event.is_set() == shutdown

    def _try_put_index(self):

        assert self._tasks_outstanding < self._prefetch_factor * self._num_workers

        try:
            index = self._next_index()
        except StopIteration:
            return

        for _ in range(self._num_workers):
            worker_queue_idx = next(self._worker_queue_idx_cycle)
            if self._workers_status[worker_queue_idx]:
                break
        else:
            return

        self._index_queues[worker_queue_idx].put((self._send_idx, index))
        self._task_info[self._send_idx] = (worker_queue_idx, )
        self._tasks_outstanding += 1
        self._send_idx += 1

    def _next_data(self):
        while True:
            while self._rcvd_idx < self._send_idx:
                info = self._task_info[self._rcvd_idx]
                worker_id = info[0]
                if len(info) == 2 or self._workers_status[worker_id]:
                    break
                del self._task_info[self._rcvd_idx]
                self._rcvd_idx += 1
            else:
                if not self._persistent_workers:
                    self._shutdown_workers()
                raise StopIteration

            if len(self._task_info[self._rcvd_idx]) == 2:
                data = self._task_info.pop(self._rcvd_idx)[1]
                return self._process_data(data)

            assert not self._shutdown and self._tasks_outstanding > 0
            idx, data = self._get_data()
            self._tasks_outstanding -= 1
            if self._dataset_kind == _DatasetKind.Iter:
                # Check for _IterableDatasetStopIteration
                if isinstance(data, _IterableDatasetStopIteration):
                    if self._persistent_workers:
                        self._workers_status[data.worker_id] = False
                    else:
                        self._mark_worker_as_unavailable(data.worker_id)
                    self._try_put_index()
                    continue

            if idx != self._rcvd_idx:
                self._task_info[idx] += (data, )
            else:
                del self._task_info[idx]
                return self._process_data(data)

    def _process_data(self, data):
        self._rcvd_idx += 1
        self._try_put_index()
        if isinstance(data, ExceptionWrapper):
            data.reraise()
        return data

    def _shutdown_workers(self):
        if not self._shutdown:
            self._shutdown = True
            try:
                self._worker_done_event.set()
                for worker_id in range(len(self._workers)):
                    if self._persistent_workers or self._workers_status[worker_id]:
                        self._mark_worker_as_unavailable(worker_id, shutdown=True)
                for w in self._workers:
                    w.join(timeout=5.0)
                for q in self._index_queues:
                    q.cancel_join_thread()
                    q.close()
            finally:
                for w in self._workers:
                    if w.is_alive():
                        w.terminate()

    def __del__(self):
        self._shutdown_workers()


_ResumeIteration = namedtuple('_ResumeIteration', [])


class KeyErrorMessage(str):

    def __repr__(self):
        return self


class ExceptionWrapper(object):

    def __init__(self, exc_info=None, where="in background"):
        if exc_info is None:
            exc_info = sys.exc_info()
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))
        self.where = where

    def reraise(self):
        msg = "Caught {} {}.\nOriginal {}".format(self.exc_type.__name__, self.where, self.exc_msg)
        if self.exc_type == KeyError:
            msg = KeyErrorMessage(msg)
        elif getattr(self.exc_type, "message", None):
            raise self.exc_type(message=msg)
        raise self.exc_type(msg)


@dataclass(frozen=True)
class _IterableDatasetStopIteration(object):
    worker_id: int


def _worker_loop(
    dataset_kind, dataset, index_queue, data_queue, done_event, is_batch, collate_fn, init_fn, worker_id, drop_last
):
    try:
        init_exception = None
        try:
            if init_fn is not None:
                init_fn(worker_id)

            fetcher = _DatasetKind.create_fetcher(dataset_kind, dataset, is_batch, collate_fn, drop_last)
        except Exception:
            init_exception = ExceptionWrapper(where="in DataLoader worker process {}".format(worker_id))

        iteration_end = False
        watchdog = ManagerWatchdog()

        while watchdog.is_alive():
            try:
                r = index_queue.get(timeout=5.0)
            except queue.Empty:
                continue
            if isinstance(r, _ResumeIteration):
                data_queue.put(r)
                iteration_end = False
                fetcher = _DatasetKind.create_fetcher(dataset_kind, dataset, is_batch, collate_fn, drop_last)
                continue
            elif r is None:
                assert done_event.is_set() or iteration_end
                break
            elif done_event.is_set() or iteration_end:
                continue
            idx, index = r
            if init_exception is not None:
                data = init_exception
                init_exception = None
            else:
                try:
                    data = fetcher.fetch(index)
                except Exception as e:
                    if isinstance(e, StopIteration) and dataset_kind == _DatasetKind.Iter:
                        data = _IterableDatasetStopIteration(worker_id)
                        iteration_end = True
                    else:
                        # It is important that we don't store exc_info in a variable.
                        # `ExceptionWrapper` does the correct thing.
                        # See NOTE [ Python Traceback Reference Cycle Problem ]
                        data = ExceptionWrapper(
                            where="in DataLoader worker process {}".format(worker_id))
            data_queue.put((idx, data))
            del data, idx, index, r
    except KeyboardInterrupt:
        pass
    if done_event.is_set():
        data_queue.cancel_join_thread()
        data_queue.close()


IS_WINDOWS = sys.platform == 'win32'
if IS_WINDOWS:
    import ctypes
    from ctypes.wintypes import DWORD, BOOL, HANDLE

    # On Windows, the parent ID of the worker process remains unchanged when the manager process
    # is gone, and the only way to check it through OS is to let the worker have a process handle
    # of the manager and ask if the process status has changed.
    class ManagerWatchdog(object):

        def __init__(self):
            self.manager_pid = os.getppid()
            self.kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            self.kernel32.OpenProcess.argtypes = (DWORD, BOOL, DWORD)
            self.kernel32.OpenProcess.restype = HANDLE
            self.kernel32.WaitForSingleObject.argtypes = (HANDLE, DWORD)
            self.kernel32.WaitForSingleObject.restype = DWORD

            SYNCHRONIZE = 0x00100000
            self.manager_handle = self.kernel32.OpenProcess(SYNCHRONIZE, 0, self.manager_pid)

            if not self.manager_handle:
                raise ctypes.WinError(ctypes.get_last_error())
            self.manager_dead = False

        def is_alive(self):
            if not self.manager_dead:
                self.manager_dead = self.kernel32.WaitForSingleObject(self.manager_handle, 0) == 0
            return not self.manager_dead
else:

    class ManagerWatchdog(object):

        def __init__(self):
            self.manager_pid = os.getppid()
            self.manager_dead = False

        def is_alive(self):
            if not self.manager_dead:
                self.manager_dead = os.getppid() != self.manager_pid
            return not self.manager_dead
