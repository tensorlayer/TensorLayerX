#! /usr/bin/python
# -*- coding: utf-8 -*-

import bisect
import numpy as np

__all__ = [
    'Dataset',
    'IterableDataset',
    'TensorDataset',
    'ChainDataset',
    'ConcatDataset',
    'Subset',
    'random_split',
]


class Dataset(object):
    """An abstract class to encapsulate methods and behaviors of datasets.
    All datasets in map-style(dataset samples can be get by a given key) should be a subclass of 'tensorlayerx.dataflow.Dataset'.
    ALl subclasses should implement following methods:
    :code:`__getitem__`: get sample from dataset with a given index.
    :code:`__len__`: return dataset sample number.
    :code:`__add__`: concat two datasets

    Examples
    --------
    With TensorLayerx

    >>> from tensorlayerx.dataflow import Dataset
    >>> class mnistdataset(Dataset):
    >>>     def __init__(self, data, label,transform):
    >>>         self.data = data
    >>>         self.label = label
    >>>         self.transform = transform
    >>>     def __getitem__(self, index):
    >>>         data = self.data[index].astype('float32')
    >>>         data = self.transform(data)
    >>>         label = self.label[index].astype('int64')
    >>>         return data, label
    >>>     def __len__(self):
    >>>         return len(self.data)
    >>> train_dataset = mnistdataset(data = X_train, label = y_train ,transform = transform)

    """

    def __init__(self):
        pass

    def __getitem__(self, idx):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__getitem__', self.__class__.__name__))

    def __len__(self):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__len__', self.__class__.__name__))

    def __add__(self, other):

        return ConcatDataset([self, other])


class IterableDataset(object):
    """An abstract class to encapsulate methods and behaviors of iterable datasets.
    All datasets in iterable-style (can only get sample one by one sequentially, likea Python iterator) should be a subclass of `tensorlayerx.dataflow.IterableDataset`.
    All subclasses should implement following methods:
    :code:`__iter__`: yield sample sequentially.

    Examples
    --------
    With TensorLayerx

    >>>#example 1:
    >>> from tensorlayerx.dataflow import IterableDataset
    >>> class mnistdataset(IterableDataset):
    >>>     def __init__(self, data, label,transform):
    >>>         self.data = data
    >>>         self.label = label
    >>>         self.transform = transform
    >>>     def __iter__(self):
    >>>         for i in range(len(self.data)):
    >>>             data = self.data[i].astype('float32')
    >>>             data = self.transform(data)
    >>>             label = self.label[i].astype('int64')
    >>>             yield data, label
    >>> train_dataset = mnistdataset(data = X_train, label = y_train ,transform = transform)
    >>>#example 2:
    >>>iterable_dataset_1 = mnistdataset(data_1, label_1, transform_1)
    >>>iterable_dataset_2 = mnistdataset(data_2, label_2, transform_2)
    >>>new_iterable_dataset = iterable_dataset_1 + iterable_dataset_2

    """

    def __init__(self):
        pass

    def __iter__(self):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__iter__', self.__class__.__name__))

    def __add__(self, other):
        return ChainDataset([self, other])


class TensorDataset(Dataset):
    """Generate a dataset from a list of tensors.
    Each sample will be retrieved by indexing tensors along the first dimension.

    Parameters
    ------------
    *tensor : list or tuple of tensors
        tensors that have the same size of the first dimension.

    Examples
    --------
    With TensorLayerx

    >>> import numpy as np
    >>> import tensorlayerx as tlx
    >>> data = np.random.random([10,224,224,3]).astype(np.float32)
    >>> label = np.random.random((10,)).astype(np.int32)
    >>> data = tlx.convert_to_tensor(data)
    >>> label = tlx.convert_to_tensor(label)
    >>> dataset = tlx.dataflow.TensorDataset([data, label])
    >>> for i in range(len(dataset)):
    >>>     x, y = dataset[i]

    """

    def __init__(self, *tensors):
        super(TensorDataset, self).__init__()
        assert all(
            [tensor.shape[0] == tensors[0].shape[0] for tensor in tensors]
        ), "tensors not have same shape of the 1st dimension"
        self.tensors = tensors

    def __getitem__(self, item):

        return tuple(tensor[item] for tensor in self.tensors)

    def __len__(self):

        return self.tensors[0].shape[0]


class ConcatDataset(Dataset):
    """Concat multiple datasets into a new dataset

    Parameters
    --------------
    datasets : list or tuple
        sequence of datasets to be concatenated

    Examples
    --------
    With TensorLayerx

    >>> import numpy as np
    >>> from tensorlayerx.dataflow import Dataset, ConcatDataset
    >>> class mnistdataset(Dataset):
    >>>     def __init__(self, data, label,transform):
    >>>         self.data = data
    >>>         self.label = label
    >>>         self.transform = transform
    >>>     def __getitem__(self, index):
    >>>         data = self.data[index].astype('float32')
    >>>         data = self.transform(data)
    >>>         label = self.label[index].astype('int64')
    >>>         return data, label
    >>>     def __len__(self):
    >>>         return len(self.data)
    >>> train_dataset1 = mnistdataset(data = X_train1, label = y_train1 ,transform = transform1)
    >>> train_dataset2 = mnistdataset(data = X_train2, label = y_train2 ,transform = transform2)
    >>> train_dataset = ConcatDataset([train_dataset1, train_dataset2])

    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable.'
        self.datasets = list(datasets)
        for dataset in self.datasets:
            assert not isinstance(dataset, IterableDataset), "ConcatDataset can not support IterableDataset."
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):

        return self.cumulative_sizes[-1]

    def __getitem__(self, item):
        dataset_id = bisect.bisect_right(self.cumulative_sizes, item)
        if dataset_id == 0:
            sample_id = item
        else:
            sample_id = item - self.cumulative_sizes[dataset_id - 1]
        return self.datasets[dataset_id][sample_id]


class ChainDataset(IterableDataset):
    """A Dataset which chains multiple iterable-tyle datasets.

    Parameters
    ------------
    datasets : list or tuple
        sequence of datasets to be chainned.

    Examples
    --------
    With TensorLayerx

    >>> import numpy as np
    >>> from tensorlayerx.dataflow import IterableDataset, ChainDataset
    >>> class mnistdataset(IterableDataset):
    >>>     def __init__(self, data, label):
    >>>         self.data = data
    >>>         self.label = label
    >>>     def __iter__(self):
    >>>         for i in range(len(self.data)):
    >>>             yield self.data[i] self.label[i]
    >>> train_dataset1 = mnistdataset(data = X_train1, label = y_train1)
    >>> train_dataset2 = mnistdataset(data = X_train2, label = y_train2)
    >>> train_dataset = ChainDataset([train_dataset1, train_dataset2])

    """

    def __init__(self, datasets):
        super(ChainDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable.'
        for dataset in datasets:
            assert isinstance(dataset, IterableDataset), "ChainDataset only supports IterableDataset"
        self.datasets = list(datasets)

    def __iter__(self):
        for dataset in self.datasets:
            for x in dataset:
                yield x

    def __len__(self):
        l = 0
        for dataset in self.datasets:
            l += len(dataset)
        return l


class Subset(Dataset):
    """Subset of a dataset at specified indices.

    Parameters
    -------------
    dataset : Dataset
        The whole Dataset

    indices : list or tuple
        Indices in the whole set selected for subset

    Examples
    --------
    With TensorLayerx

    >>> import numpy as np
    >>> from tensorlayerx.dataflow import Dataset, Subset
    >>> class mnistdataset(Dataset):
    >>>     def __init__(self, data, label):
    >>>         self.data = data
    >>>         self.label = label
    >>>     def __iter__(self):
    >>>         for i in range(len(self.data)):
    >>>             yield self.data[i] self.label[i]
    >>> train_dataset = mnistdataset(data = X_train, label = y_train)
    >>> sub_dataset = Subset(train_dataset, indices=[1,2,3])

    """

    def __init__(self, dataset, indices):
        super(Subset, self).__init__()
        assert not isinstance(dataset, IterableDataset), "Subset does not support IterableDataset."
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, item):
        return self.dataset[self.indices[item]]

    def __len__(self):
        return len(self.indices)


# Taken from python 3.5 docs
def _accumulate(iterable, fn=lambda x, y: x + y):
    'Return running totals'
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total


def random_split(dataset, lengths):
    """Randomly split a dataset into non-overlapping new datasets of given lengths.

    Parameters
    ----------
    dataset : Dataset
        dataset to be split
    lengths : list or tuple
        lengths of splits to be produced

    Examples
    --------
    With TensorLayerx

    >>> import numpy as np
    >>> from tensorlayerx.dataflow import Dataset, Subset
    >>> random_split(range(10), [3, 7])

    """

    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    generator = np.random.default_rng()
    indices = generator.permutation(sum(lengths))
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]
