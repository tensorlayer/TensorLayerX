#! /usr/bin/python
# -*- coding: utf-8 -*-

import paddle
from paddle.metric.metrics import Metric
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

    def __init__(
        self,
        topk=1,
    ):

        self.topk = topk
        self.accuracy = paddle.metric.Accuracy(topk=(self.topk, ))

    def update(self, y_pred, y_true):

        self.accuracy.update(self.accuracy.compute(y_pred, y_true))

    def result(self):

        return self.accuracy.accumulate()

    def reset(self):

        self.accuracy.reset()


class Auc(object):

    def __init__(self, curve='ROC', num_thresholds=4095):

        self.auc = paddle.metric.Auc(curve=curve, num_thresholds=num_thresholds)

    def update(self, y_pred, y_true):

        self.auc.update(y_pred, y_true)

    def result(self):

        return self.auc.accumulate()

    def reset(self):

        self.auc.reset()


class Precision(object):

    def __init__(self):

        self.precision = paddle.metric.Precision()

    def update(self, y_pred, y_true):

        self.precision.update(y_pred, y_true)

    def result(self):

        return self.precision.accumulate()

    def reset(self):

        self.precision.reset()


class Recall(object):

    def __init__(self):

        self.recall = paddle.metric.Recall()

    def update(self, y_pred, y_true):
        self.recall.update(y_pred, y_true)

    def result(self):
        return self.recall.accumulate()

    def reset(self):
        self.recall.reset()


def acc(predicts, labels, topk=1):

    res = paddle.metric.accuracy(predicts, labels, k=topk)
    return res.numpy()
