#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
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
    """
    Base class for metric

    Methods
    ---------
    __init__()
        Initializing the Metric.
    update()
        Update states for metric.
    result()
        Computes and returns the metric value.
    reset()
        Reset states and result.

    """
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
    """Accuracy metric

    Parameters
    ------------

    topk : int
        Specifies the top-k categorical accuracy to compute. Default is (1,).

    Examples
    -----------
    >>> import tensorlayerx as tlx
    >>> y_pred = tlx.ops.convert_to_tensor(np.array([[0.3, 0.2, 0.1, 0.4], [0.2, 0.2, 0.5, 0.1]]))
    >>> y_true = tlx.ops.convert_to_tensor(np.array([[1], [3]]))
    >>> metric = tlx.metrics.Accuracy()
    >>> metric.update(y_pred, y_true)
    >>> res = metric.result()
    >>> metric.reset()

    """

    def __init__(self, topk=1):
        self.topk = topk
        if topk == 1:
            self.accuary = tf.keras.metrics.Accuracy()
        else:
            self.accuary = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=topk)

    def update(self, y_pred, y_true):
        """
        Updates the internal evaluation result `y_pred` and `y_true`.

        Parameters
        ----------
        y_pred : Tensor
            The predicted value.
        y_true : Tensor
            The ground truth.
        """

        if self.topk == 1:
            y_pred = tf.argmax(y_pred, axis=1)
            self.accuary.update_state(y_true, y_pred)
        else:
            self.accuary.update_state(y_true, y_pred)

    def result(self):
        """
        Computes the top-k categorical accuracy.

        Returns
        -------
            computed result.
        """

        return self.accuary.result().numpy()

    def reset(self):
        """
        Resets all of the metric state.
        """

        self.accuary.reset_states()


class Auc(object):
    """
    The auc metric is for binary classification.

    Parameters
    -----------
    curve : str
        Specifies the mode of the curve to be computed. Only support 'ROC' now.
    num_thresholds : int
        The number of thresholds to use when discretizing the roc curve.

    """

    def __init__(
        self,
        curve='ROC',
        num_thresholds=4095,
    ):
        self.curve = curve
        self.num_thresholds = num_thresholds
        self.reset()

    def update(self, y_pred, y_true):
        """
        Updates the auc curve with `y_pred` and `y_true`.

        Parameters
        ----------
        y_pred : Tensor
            The predicted value.
        y_true : Tensor
            The ground truth.
        """
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
        """
        Return the area (a float score) under auc curve

        Returns
        -------
            computed result.
        """
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
    """
    Precision score for binary classification task.


    Examples
    -----------
    >>> import tensorlayerx as tlx
    >>> y_pred = tlx.ops.convert_to_tensor(np.array([0.3, 0.2, 0.1, 0.7]))
    >>> y_true = tlx.ops.convert_to_tensor(np.array([1, 0, 0, 1]))
    >>> metric = tlx.metrics.Precision()
    >>> metric.update(y_pred, y_true)
    >>> res = metric.result()
    >>> metric.reset()
    """

    def __init__(self):

        self.precision = tf.keras.metrics.Precision()

    def update(self, y_pred, y_true):
        """
        Update the states based on the current mini-batch prediction results.

        Parameters
        ----------
        y_pred : Tensor
            The predicted value.
        y_true : Tensor
            The ground truth.

        """

        self.precision.update_state(y_true, y_pred)

    def result(self):
        """
        Return the precision

        Returns
        -------
            computed result.
        """

        return self.precision.result().numpy()

    def reset(self):
        """
        Resets all of the metric state.
        """
        self.precision.reset_states()


class Recall(object):
    """
    Recall score for binary classification task.

        Examples
    -----------
    >>> import tensorlayerx as tlx
    >>> y_pred = tlx.ops.convert_to_tensor(np.array([0.3, 0.2, 0.1, 0.7]))
    >>> y_true = tlx.ops.convert_to_tensor(np.array([1, 0, 0, 1]))
    >>> metric = tlx.metrics.Recall()
    >>> metric.update(y_pred, y_true)
    >>> res = metric.result()
    >>> metric.reset()
    """

    def __init__(self):

        self.recall = tf.keras.metrics.Recall()

    def update(self, y_pred, y_true):
        """
        Update the states based on the current mini-batch prediction results.

        Parameters
        ----------
        y_pred : Tensor
            The predicted value.
        y_true : Tensor
            The ground truth.

        """
        self.recall.update_state(y_true, y_pred)

    def result(self):
        """
        Return the recall

        Returns
        -------
            computed result.
        """
        return self.recall.result().numpy()

    def reset(self):
        """
        Resets all of the metric state.
        """
        self.recall.reset_states()


def acc(predicts, labels, topk=1):
    """Accuracy function.

    Parameters
    ----------
    predicts : Tensor
        The predicted value.
    labels : Tensor
        The ground truth.
    topk : int
        The top k predictions for each class will be checked.

    Returns
    -------
        The accuracy result.

    Examples
    -----------
    >>> import tensorlayerx as tlx
    >>> y_pred = tlx.ops.convert_to_tensor(np.array([[0.3, 0.2, 0.1, 0.4], [0.2, 0.2, 0.5, 0.1]]))
    >>> y_true = tlx.ops.convert_to_tensor(np.array([[1], [3]]))
    >>> acc = tlx.metrics.acc(y_pred, y_true, topk=1)
    """
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
