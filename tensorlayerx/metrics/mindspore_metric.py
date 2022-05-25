#! /usr/bin/python
# -*- coding: utf-8 -*-
import mindspore as ms
import mindspore.nn as nn
import numpy as np
import six
import abc

__all__ = [
    'Metric',
    'Accuracy',
    'Auc',
    'Precision',
    'Recall',
    'acc',
]


@six.add_metaclass(abc.ABCMeta)
class Metric(object):

    def __init__(self):
        pass

    @abc.abstractmethod
    def update(self, *args):
        raise NotImplementedError("function 'update' not implemented in {}.".format(self.__class__.__name__))

    @abc.abstractmethod
    def result(self):
        raise NotImplementedError("function 'reset' not implemented in {}.".format(self.__class__.__name__))

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError("function 'reset' not implemented in {}.".format(self.__class__.__name__))

class Accuracy(object):

    def __init__(self, topk=1):

        self.accuracy = nn.TopKCategoricalAccuracy(k=topk)

    def update(self, y_pred, y_true):

        self.accuracy.update(y_pred, y_true)

    def result(self):

        return self.accuracy.eval()

    def reset(self):

        self.accuracy.clear()


class Auc(object):

    def __init__(
        self,
        curve='ROC',
        num_thresholds=4095,
    ):
        self.curve = curve
        self.num_thresholds = num_thresholds
        self.reset()

    def update(self, y_pred, y_true):
        if isinstance(y_true, ms.Tensor):
            y_true = y_true.asnumpy()
        elif not isinstance(y_pred, np.ndarray):
            raise TypeError("The y_true must be a numpy array or Tensor.")

        if isinstance(y_pred, ms.Tensor):
            y_pred = y_pred.asnumpy()
        elif not isinstance(y_pred, np.ndarray):
            raise TypeError("The y_pred must be a numpy array or Tensor.")

        for i, label in enumerate(y_true):
            value = y_pred[i, 1]  # positive probability
            bin_idx = int(value * self.num_thresholds)
            assert bin_idx <= self.num_thresholds
            if label:
                self._stat_pos[bin_idx] += 1.0
            else:
                self._stat_neg[bin_idx] += 1.0

    @staticmethod
    def trapezoid_area(x1, x2, y1, y2):
        return abs(x1 - x2) * (y1 + y2) / 2.0

    def result(self):
        tot_pos = 0.0
        tot_neg = 0.0
        auc = 0.0
        idx = self.num_thresholds
        while idx > 0:
            tot_pos_prev = tot_pos
            tot_neg_prev = tot_neg
            tot_pos += self._stat_pos[idx]
            tot_neg += self._stat_neg[idx]
            auc += self.trapezoid_area(tot_neg, tot_neg_prev, tot_pos, tot_pos_prev)
            idx -= 1

        return auc / tot_pos / tot_neg if tot_pos > 0.0 and tot_neg > 0.0 else 0.0

    def reset(self):

        _num_pred_buckets = self.num_thresholds + 1
        self._stat_pos = np.zeros(_num_pred_buckets)
        self._stat_neg = np.zeros(_num_pred_buckets)


class Precision(object):

    def __init__(self):
        self.reset()

    def update(self, y_pred, y_true):
        if isinstance(y_true, ms.Tensor):
            y_true = y_true.asnumpy()
        elif not isinstance(y_pred, np.ndarray):
            raise TypeError("The y_true must be a numpy array or Tensor.")

        if isinstance(y_pred, ms.Tensor):
            y_pred = y_pred.asnumpy()
        elif not isinstance(y_pred, np.ndarray):
            raise TypeError("The y_pred must be a numpy array or Tensor.")

        sample_num = y_true.shape[0]
        y_pred = np.rint(y_pred).astype('int32')

        for i in range(sample_num):
            pred = y_pred[i]
            label = y_true[i]
            if pred == 1:
                if pred == label:
                    self.tp += 1
                else:
                    self.fp += 1

    def result(self):

        ap = self.tp + self.fp
        return float(self.tp) / ap if ap != 0 else .0

    def reset(self):
        self.tp = 0
        self.fp = 0


class Recall(object):

    def __init__(self):
        self.reset()

    def update(self, y_pred, y_true):
        if isinstance(y_true, ms.Tensor):
            y_true = y_true.asnumpy()
        elif not isinstance(y_pred, np.ndarray):
            raise TypeError("The y_true must be a numpy array or Tensor.")

        if isinstance(y_pred, ms.Tensor):
            y_pred = y_pred.asnumpy()
        elif not isinstance(y_pred, np.ndarray):
            raise TypeError("The y_pred must be a numpy array or Tensor.")

        sample_num = y_true.shape[0]
        y_pred = np.rint(y_pred).astype('int32')

        for i in range(sample_num):
            pred = y_pred[i]
            label = y_true[i]
            if label == 1:
                if pred == label:
                    self.tp += 1
                else:
                    self.fn += 1

    def result(self):

        recall = self.tp + self.fn
        return float(self.tp) / recall if recall != 0 else .0

    def reset(self):
        self.tp = 0
        self.fn = 0


def acc(predicts, labels, topk=1):
    argsort = ms.ops.Sort(axis=-1, descending=True)
    _, y_pred = argsort(predicts)
    y_pred = y_pred[:, :topk]
    if (len(labels.shape) == 1) or (len(labels.shape) == 2 and labels.shape[-1] == 1):
        y_true = ms.ops.reshape(labels, (-1, 1))
    elif labels.shape[-1] != 1:
        y_true = ms.numpy.argmax(labels, dim=-1)
        y_true = ms.ops.reshape(y_true, (len(y_true), 1))
    correct = y_pred == y_true
    correct = ms.ops.cast(correct, ms.float32)
    correct = correct.asnumpy()
    num_samples = np.prod(np.array(correct.shape[:-1]))
    num_corrects = correct[..., :topk].sum()
    total = num_corrects
    count = num_samples
    return float(total) / count if count > 0 else 0.
