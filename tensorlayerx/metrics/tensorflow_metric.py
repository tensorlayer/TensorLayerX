#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.metrics import Metric
import numpy as np

__all__ = [
    'Accuracy',
    'Auc',
    'Precision',
    'Recall',
    'acc',
]


class Accuracy(object):

    def __init__(self, topk=1):
        self.topk = topk
        if topk == 1:
            self.accuary = tf.keras.metrics.Accuracy()
        else:
            self.accuary = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=topk)

    def update(self, y_pred, y_true):

        if self.topk == 1:
            y_pred = tf.argmax(y_pred, axis=1)
            self.accuary.update_state(y_true, y_pred)
        else:
            self.accuary.update_state(y_true, y_pred)

    def result(self):

        return self.accuary.result().numpy()

    def reset(self):

        self.accuary.reset_states()


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
        if isinstance(y_true, tf.Tensor):
            y_true = y_true.numpy()
        elif not isinstance(y_pred, np.ndarray):
            raise TypeError("The y_true must be a numpy array or Tensor.")

        if isinstance(y_pred, tf.Tensor):
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
        """
        Reset states and result
        """
        _num_pred_buckets = self.num_thresholds + 1
        self._stat_pos = np.zeros(_num_pred_buckets)
        self._stat_neg = np.zeros(_num_pred_buckets)


class Precision(object):

    def __init__(self):

        self.precision = tf.keras.metrics.Precision()

    def update(self, y_pred, y_true):

        self.precision.update_state(y_true, y_pred)

    def result(self):

        return self.precision.result().numpy()

    def reset(self):

        self.precision.reset_states()


class Recall(object):

    def __init__(self):

        self.recall = tf.keras.metrics.Recall()

    def update(self, y_pred, y_true):

        self.recall.update_state(y_true, y_pred)

    def result(self):

        return self.recall.result().numpy()

    def reset(self):

        self.recall.reset_states()


def acc(predicts, labels, topk=1):
    y_pred = tf.argsort(predicts, axis=-1, direction='DESCENDING')
    y_pred = y_pred[:, :topk]
    if (len(labels.shape) == 1) or (len(labels.shape) == 2 and labels.shape[-1] == 1):
        y_true = tf.reshape(labels, (-1, 1))
    elif labels.shape[-1] != 1:
        y_true = tf.argmax(labels, axis=-1)
        y_true = tf.reshape(y_true, shape=(len(y_true), 1))
    correct = y_pred == y_true
    correct = tf.cast(correct, dtype=tf.float32)
    correct = correct.numpy()
    num_samples = np.prod(np.array(correct.shape[:-1]))
    num_corrects = correct[..., :topk].sum()
    total = num_corrects
    count = num_samples
    return float(total) / count if count > 0 else 0.
