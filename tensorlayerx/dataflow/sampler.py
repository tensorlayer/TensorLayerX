#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

__all__ = [
    'Sampler',
    'BatchSampler',
    'RandomSampler',
    'SequentialSampler',
    'WeightedRandomSampler',
    'SubsetRandomSampler',
]


class Sampler(object):
    """Base class for all Samplers.
    All subclasses should implement following methods:
    :code:`__iter__`: providing a way to iterate over indices of dataset element
    :code:`__len__`: the length of the returned iterators.

    Examples
    --------
    With TensorLayerx

    >>> from tensorlayerx.dataflow import Sampler
    >>> class MySampler(Sampler):
    >>>     def __init__(self, data):
    >>>         self.data = data
    >>>     def __iter__(self):
    >>>         return iter(range(len(self.data_source)))
    >>>     def __len__(self):
    >>>         return len(self.data)

    """

    def __init__(self):
        pass

    def __iter__(self):
        raise NotImplementedError


class BatchSampler(Sampler):
    """Wraps another sampler to yield a mini-batch of indices.

    Parameters
    ----------
    sampler : Sampler
        Base sampler.
    batch_size : int
        Size of mini-batch
    drop_last : bool
        If ``True``, the sampler will drop the last batch if its size would be less than ``batch_size``

    Examples
    --------
    With TensorLayerx

    >>> from tensorlayerx.dataflow import BatchSampler, SequentialSampler
    >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
    >>> #[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
    >>> #[[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    """

    def __init__(self, sampler=None, batch_size=1, drop_last=False):
        super(BatchSampler, self).__init__()
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, but got {}.".format(type(batch_size)))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a bool value, but got {}.".format(type(drop_last)))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch_idxs = []
        for index in self.sampler:
            batch_idxs.append(index)
            if len(batch_idxs) == self.batch_size:
                yield batch_idxs
                batch_idxs = []
        if len(batch_idxs) > 0 and not self.drop_last:
            yield batch_idxs

    def __len__(self):
        num_samples = len(self.sampler)
        if self.drop_last:
            return num_samples // self.batch_size
        else:
            return (num_samples + self.batch_size - 1) // self.batch_size


class RandomSampler(Sampler):
    """Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify`num_samples` to draw.

    Parameters
    -------------
    data : Dataset
        dataset to sample
    replacement : bool
        samples are drawn on-demand with replacement if ``True``, default=``False``
    num_samples : int
        number of samples to draw, default=`len(dataset)`. This argument is supposed to be specified only when `replacement` is ``True``.
    generator : Generator
        Generator used in sampling. Default is None.

    Examples
    --------
    With TensorLayerx

    >>> from tensorlayerx.dataflow import RandomSampler, Dataset
    >>> import numpy as np
    >>> class mydataset(Dataset):
    >>>     def __init__(self):
    >>>         self.data = [np.random.random((224,224,3)) for i in range(100)]
    >>>         self.label = [np.random.randint(1, 10, (1,)) for i in range(100)]
    >>>     def __getitem__(self, item):
    >>>         x = self.data[item]
    >>>         y = self.label[item]
    >>>         return x, y
    >>>     def __len__(self):
    >>>         return len(self.data)
    >>> sampler = RandomSampler(data = mydataset())

    """

    def __init__(self, data, replacement=False, num_samples=None, generator=None):
        super(RandomSampler, self).__init__()
        self.data = data
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got " "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("When replacement is False, num_samples should not be specified.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer, "
                "but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        if self._num_samples is None:
            return len(self.data)
        return self._num_samples

    def __iter__(self):
        n = len(self.data)
        if self.generator is None:
            generator = np.random.default_rng()
            if self.replacement:
                for index in generator.choice(np.arange(n), self.num_samples, replace=True).tolist():
                    yield index
            else:
                for index in generator.choice(np.arange(n), n, replace=False).tolist():
                    yield index
        else:
            for i in range(self.num_samples):
                try:
                    index = next(self.generator)
                except StopIteration:
                    return
                yield index

    def __len__(self):
        return self.num_samples


class SequentialSampler(Sampler):
    """Samples elements sequentially, always in the same order.

    Parameters
    ----------
    data : Dataset
        dataset to sample

    Examples
    --------
    With TensorLayerx

    >>> from tensorlayerx.dataflow import SequentialSampler, Dataset
    >>> import numpy as np
    >>> class mydataset(Dataset):
    >>>     def __init__(self):
    >>>         self.data = [np.random.random((224,224,3)) for i in range(100)]
    >>>         self.label = [np.random.randint(1, 10, (1,)) for i in range(100)]
    >>>     def __getitem__(self, item):
    >>>         x = self.data[item]
    >>>         y = self.label[item]
    >>>         return x, y
    >>>     def __len__(self):
    >>>         return len(self.data)
    >>> sampler = SequentialSampler(data = mydataset())

    """

    def __init__(self, data):
        super(SequentialSampler, self).__init__()
        self.data = data

    def __iter__(self):
        return iter(range(len(self.data)))

    def __len__(self):
        return len(self.data)


class WeightedRandomSampler(Sampler):
    """Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Parameters
    -----------
    weights : list or tuple
        a sequence of weights, not necessary summing up to one
    num_samples : int
        number of samples to draw
    replacement : bool
        if ``True``, samples are drawn with replacement.
        If not, they are drawn without replacement, which means that when a sample index is drawn for a row, it cannot be drawn again for that row.

    Examples
    --------
    With TensorLayerx

    >>> from tensorlayerx.dataflow import WeightedRandomSampler, Dataset
    >>> import numpy as np
    >>> sampler = list(WeightedRandomSampler(weights=[0.2,0.3,0.4,0.5,4.0], num_samples=5, replacement=True))
    >>> #[4, 4, 1, 4, 4]
    >>> sampler = list(WeightedRandomSampler(weights=[0.2,0.3,0.4,0.5,0.6], num_samples=5, replacement=False))
    >>> #[4, 1, 3, 0, 2]

    """

    def __init__(self, weights, num_samples, replacement=True):
        super(WeightedRandomSampler, self).__init__()
        if not isinstance(weights, (list, tuple, np.ndarray)):
            raise ValueError("weights should be a list, tuple or numpy.ndarray, but got {}.".format(type(weights)))
        weights = np.asarray(weights, np.float)
        assert len(weights.shape) == 1, "weights should be a 1-D array"
        if np.any(weights < 0.0):
            raise ValueError("weights should be positive value.")
        if not np.sum(weights) > 0.0:
            raise ValueError("The sum of weights should be a positive value.")
        if not replacement:
            if np.sum(weights > 0.0) < num_samples:
                raise ValueError(
                    "when replacement is False, the number of positive values in weights should be greater than numsamples."
                )
        self.weights = weights / weights.sum()
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        index = np.random.choice(len(self.weights), self.num_samples, self.replacement, self.weights)
        return iter(index.tolist())

    def __len__(self):
        return self.num_samples


class SubsetRandomSampler(Sampler):
    """Samples elements randomly from a given list of indices, without replacement.

    Parameters
    ----------
    indices : list or tuple
        sequence of indices

    """

    def __init__(self, indices):
        super(SubsetRandomSampler, self).__init__()
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in np.random.permutation(len(self.indices)))

    def __len__(self):
        return len(self.indices)
