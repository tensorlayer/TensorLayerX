#! /usr/bin/python
# -*- coding: utf-8 -*-
import mindspore as ms
import mindspore.nn as nn
from mindspore.nn.metrics.metric import Metric
import numpy as np
__all__ = [
    'Accuracy',
    'Auc',
    'Precision',
    'Recall',
]


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
        num_thresholds=200,
    ):
        self.curve = curve
        self.num_thresholds = num_thresholds
        self.reset()

    def update(self, y_pred, y_true):
        if isinstance(y_true, ms.Tensor):
            y_true = y_true.numpy()
        elif not isinstance(y_pred, np.ndarray):
            raise TypeError("The y_true must be a numpy array or Tensor.")

        if isinstance(y_pred, ms.Tensor):
            y_pred = y_pred.numpy()
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

        self.precision = nn.Precision(eval_type="classification")

    def update(self, y_pred, y_true):

        self.precision.update(y_pred, y_true)

    def result(self):

        return self.precision.eval()

    def reset(self):

        self.precision.clear()


class Recall(object):

    def __init__(self):

        self.recall = nn.Recall(eval_type="classification")

    def update(self, y_pred, y_true):

        self.recall.update(y_pred, y_true)

    def result(self):

        return self.recall.eval()

    def reset(self):

        self.recall.clear()
