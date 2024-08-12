#! /usr/bin/python
# -*- coding: utf-8 -*-

import jittor as jt
import six
import abc
import numpy as np

__all__ = [
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



class Accuracy:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, y_pred, y_true):
        # Step 1: Get the predicted class labels using argmax
        y_pred = jt.argmax(y_pred, dim=-1)
        
        # Step 2: Ensure y_true is reshaped to match y_pred
        y_true = np.reshape(y_true, (-1,))
        
        # Step 3: Compare the predicted labels to the true labels
        correct_predictions = np.equal(y_pred, y_true)
        
        # Step 4: Count the number of correct predictions
        num_correct_predictions = np.sum(correct_predictions).item()
        
        # Update the running totals
        self.correct += num_correct_predictions
        self.total += y_true.shape[0]

    def result(self):
        # Calculate the accuracy
        return self.correct / self.total if self.total > 0 else 0.0

    def reset(self):
        # Reset the counters
        self.correct = 0
        self.total = 0


class Auc:
    def __init__(self, num_thresholds=4095):
        self.num_thresholds = num_thresholds
        self.reset()

    def update(self, y_pred, y_true):
        # Convert Jittor tensors to NumPy arrays if necessary
        if isinstance(y_true, jt.Var):
            y_true = y_true.numpy()
        if isinstance(y_pred, jt.Var):
            y_pred = y_pred.numpy()

        # Flatten y_true to ensure it's 1-dimensional
        y_true = np.reshape(y_true, (-1,))
        
        # Get the positive class probabilities
        pos_prob = y_pred[:, 1]

        # Bin the predictions into thresholds
        bin_idx = np.floor(pos_prob * self.num_thresholds).astype(int)
        bin_idx = np.clip(bin_idx, 0, self.num_thresholds)

        # Update the histogram bins
        for i, label in enumerate(y_true):
            if label:
                self._stat_pos[bin_idx[i]] += 1
            else:
                self._stat_neg[bin_idx[i]] += 1

    @staticmethod
    def trapezoid_area(x1, x2, y1, y2):
        return abs(x1 - x2) * (y1 + y2) / 2.0

    def result(self):
        tot_pos = 0.0
        tot_neg = 0.0
        auc = 0.0

        for idx in range(self.num_thresholds, 0, -1):
            tot_pos_prev = tot_pos
            tot_neg_prev = tot_neg
            tot_pos += self._stat_pos[idx]
            tot_neg += self._stat_neg[idx]
            auc += self.trapezoid_area(tot_neg, tot_neg_prev, tot_pos, tot_pos_prev)

        return auc / (tot_pos * tot_neg) if tot_pos > 0.0 and tot_neg > 0.0 else 0.0

    def reset(self):
        self._stat_pos = np.zeros(self.num_thresholds + 1)
        self._stat_neg = np.zeros(self.num_thresholds + 1)


class Precision:
    def __init__(self):
        self.reset()

    def update(self, y_pred, y_true):
        # Convert Jittor tensors to NumPy arrays if necessary
        if isinstance(y_true, jt.Var):
            y_true = y_true.numpy()
        if isinstance(y_pred, jt.Var):
            y_pred = y_pred.numpy()

        # Ensure y_true is reshaped to match y_pred
        y_true = np.reshape(y_true, (-1,))
        
        # Convert probabilities to class predictions
        y_pred = np.argmax(y_pred, axis=1)

        # Update true positives (tp) and false positives (fp)
        self.tp += np.sum((y_pred == 1) & (y_true == 1))
        self.fp += np.sum((y_pred == 1) & (y_true == 0))

    def result(self):
        ap = self.tp + self.fp
        return float(self.tp) / ap if ap != 0 else 0.0

    def reset(self):
        self.tp = 0
        self.fp = 0


class Recall:
    def __init__(self):
        self.reset()

    def update(self, y_pred, y_true):
        # Convert Jittor tensors to NumPy arrays if necessary
        if isinstance(y_true, jt.Var):
            y_true = y_true.numpy()
        if isinstance(y_pred, jt.Var):
            y_pred = y_pred.numpy()

        # Ensure y_true is reshaped to match y_pred
        y_true = np.reshape(y_true, (-1,))

        # Convert probabilities to class predictions
        y_pred = np.argmax(y_pred, axis=1)

        # Update true positives (tp) and false negatives (fn)
        self.tp += np.sum((y_pred == 1) & (y_true == 1))
        self.fn += np.sum((y_true == 1) & (y_pred == 0))

    def result(self):
        recall = self.tp + self.fn
        return float(self.tp) / recall if recall != 0 else 0.0

    def reset(self):
        self.tp = 0
        self.fn = 0


    def result(self):

        recall = self.tp + self.fn
        return float(self.tp) / recall if recall != 0 else .0

    def reset(self):
        self.tp = 0
        self.fn = 0


def acc(predicts, labels, topk=1):
    y_pred = jt.argsort(predicts, dim=-1, descending=True)
    y_pred = y_pred[:, :topk]
    if (len(labels.shape) == 1) or (len(labels.shape) == 2 and labels.shape[-1] == 1):
        y_true = jt.reshape(labels, (-1, 1))
    elif labels.shape[-1] != 1:
        y_true = jt.argmax(labels, dim=-1, keepdim=True)
    correct = y_pred == y_true
    correct = correct.to(jt.float32)
    correct = correct.cpu().numpy()
    num_samples = np.prod(np.array(correct.shape[:-1]))
    num_corrects = correct[..., :topk].sum()
    total = num_corrects
    count = num_samples
    return float(total) / count if count > 0 else 0.
