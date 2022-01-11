#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections.abc import Iterable
from tensorlayerx.nn.core.common import _save_weights, _load_weights, _save_standard_weights_dict, _load_standard_weights_dict
import tensorlayerx as tl
from tensorlayerx.nn.core import Module
import numpy as np
import time

if tl.BACKEND == 'tensorflow':
    import tensorflow as tf
if tl.BACKEND == 'mindspore':
    from mindspore.ops import composite
    from mindspore.ops import operations as P
    from mindspore.common import ParameterTuple
if tl.BACKEND == 'paddle':
    import paddle as pd
if tl.BACKEND == 'torch':
    import torch

__all__ = ['Model', 'WithLoss', 'WithGrad', 'TrainOneStep']


class Model:
    """
    High-Level API for Training or Testing.

    `Model` groups layers into an object with training and inference features.

    Parameters
    ----------
    network : tensorlayer model
        The training or testing network.
    loss_fn : function
        Objective function
    optimizer : class
        Optimizer for updating the weights
    metrics : class
        Dict or set of metrics to be evaluated by the model during

    Methods
    ---------
    trin()
        Model training.
    eval()
        Model prediction.
    save_weights()
        Input file_path, save model weights into a file of given format.
        Use load_weights() to restore.
    load_weights()
        Load model weights from a given file, which should be previously saved by save_weights().

    Examples
    --------
    >>> import tensorlayerx as tl
    >>> class Net(Module):
    >>>     def __init__(self):
    >>>         super(Net, self).__init__()
    >>>         self.conv = tl.layers.Conv2d(n_filter=32, filter_size=(3, 3), strides=(2, 2), in_channels=5, name='conv2d')
    >>>         self.bn = tl.layers.BatchNorm2d(num_features=32, act=tl.ReLU)
    >>>         self.flatten = tl.layers.Flatten()
    >>>         self.fc = tl.layers.Dense(n_units=12, in_channels=32*224*224) # padding=0
    >>>
    >>>     def construct(self, x):
    >>>         x = self.conv(x)
    >>>         x = self.bn(x)
    >>>         x = self.flatten(x)
    >>>         out = self.fc(x)
    >>>         return out
    >>>
    >>> net = Net()
    >>> loss = tl.losses.softmax_cross_entropy_with_logits
    >>> optim = tl.optimizers.Momentum(params=net.trainable_weights, learning_rate=0.1, momentum=0.9)
    >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None)
    >>> dataset = get_dataset()
    >>> model.train(2, dataset)

    """

    def __init__(self, network, loss_fn=None, optimizer=None, metrics=None, **kwargs):
        self.network = network
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics
        self.all_weights = network.all_weights
        self.train_weights = self.network.trainable_weights

    def train(self, n_epoch, train_dataset=None, test_dataset=False, print_train_batch=False, print_freq=5):
        if not isinstance(train_dataset, Iterable):
            raise Exception("Expected type in (train_dataset, Iterable), but got {}.".format(type(train_dataset)))

        if tl.BACKEND == 'tensorflow':
            self.tf_train(
                n_epoch=n_epoch, train_dataset=train_dataset, network=self.network, loss_fn=self.loss_fn,
                train_weights=self.train_weights, optimizer=self.optimizer, metrics=self.metrics,
                print_train_batch=print_train_batch, print_freq=print_freq, test_dataset=test_dataset
            )
        elif tl.BACKEND == 'mindspore':
            self.ms_train(
                n_epoch=n_epoch, train_dataset=train_dataset, network=self.network, loss_fn=self.loss_fn,
                train_weights=self.train_weights, optimizer=self.optimizer, metrics=self.metrics,
                print_train_batch=print_train_batch, print_freq=print_freq, test_dataset=test_dataset
            )
        elif tl.BACKEND == 'paddle':
            self.pd_train(
                n_epoch=n_epoch, train_dataset=train_dataset, network=self.network, loss_fn=self.loss_fn,
                train_weights=self.train_weights, optimizer=self.optimizer, metrics=self.metrics,
                print_train_batch=print_train_batch, print_freq=print_freq, test_dataset=test_dataset
            )

    def eval(self, test_dataset):
        self.network.set_eval()
        test_loss, test_acc, n_iter = 0, 0, 0
        for X_batch, y_batch in test_dataset:
            _logits = self.network(X_batch)
            test_loss += self.loss_fn(_logits, y_batch)
            if self.metrics:
                try:
                    test_acc += self.metrics(_logits, y_batch)
                except:
                    self.metrics.update(_logits, y_batch)
                    test_acc += self.metrics.result()
                    self.metrics.reset()
            else:
                test_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            n_iter += 1
        print("   test loss: {}".format(test_loss / n_iter))
        print("   test acc:  {}".format(test_acc / n_iter))

    def save_standard_weights(self, file_path):
        """Save to standard format parameter format as {conv.filters: filters_param, conv.biases: biases_parm,
        dense.weights: weights_parm ...}

        Parameters
        ----------
        file_path : str
            Name of the saved file

        """

        _save_standard_weights_dict(self, file_path)

    def load_standard_weights(self, file_path, skip=False, reshape=False, format='npz_dict'):
        """

        Parameters
        ----------
        file_path : str
            Name of the saved file.
        skip : boolean
            If 'skip' == True, loaded layer whose name is not found in 'layers' will be skipped. If 'skip' is False,
            error will be raised when mismatch is found. Default False.
        reshape : boolean
            This parameter needs to be set to True when importing parameters from tensorflow training to paddle/mindspore/pytorch,
            and similarly when importing parameters from paddle/mindspore/pytorch training to tensorflow.
            This parameter does not need to be set between paddle/mindspore/pytorch.

        """

        _load_standard_weights_dict(self, file_path, skip, reshape, format)

    def save_weights(self, file_path, format=None):
        """Input file_path, save model weights into a file of given format.
            Use self.load_weights() to restore.

        Parameters
        ----------
        file_path : str
            Filename to which the model weights will be saved.
        format : str or None
            Saved file format.
            Value should be None, 'hdf5', 'npz', 'npz_dict' or 'ckpt'. Other format is not supported now.
            1) If this is set to None, then the postfix of file_path will be used to decide saved format.
            If the postfix is not in ['h5', 'hdf5', 'npz', 'ckpt'], then file will be saved in hdf5 format by default.
            2) 'hdf5' will save model weights name in a list and each layer has its weights stored in a group of
            the hdf5 file.
            3) 'npz' will save model weights sequentially into a npz file.
            4) 'npz_dict' will save model weights along with its name as a dict into a npz file.
            5) 'ckpt' will save model weights into a tensorflow ckpt file.

            Default None.

        Examples
        --------
        1) Save model weights in hdf5 format by default.
        >>> net = vgg16()
        >>> optimizer = tl.optimizers.Adam(learning_rate=0.001)
        >>> metrics = tl.metrics.Accuracy()
        >>> model = tl.model.Model(network=net, loss_fn=tl.losses.softmax_cross_entropy_with_logits, optimizer=optimizer, metrics=metrics)
        >>> model.save_weights('./model.h5')
        ...
        >>> model.load_weights('./model.h5')

        2) Save model weights in npz/npz_dict format
        >>> model.save_weights('./model.npz')
        >>> model.save_weights('./model.npz', format='npz_dict')

        """

        _save_weights(net=self, file_path=file_path, format=format)

    def load_weights(self, file_path, format=None, in_order=True, skip=False):
        """Load model weights from a given file, which should be previously saved by self.save_weights().

        Parameters
        ----------
        file_path : str
            Filename from which the model weights will be loaded.
        format : str or None
            If not specified (None), the postfix of the file_path will be used to decide its format. If specified,
            value should be 'hdf5', 'npz', 'npz_dict' or 'ckpt'. Other format is not supported now.
            In addition, it should be the same format when you saved the file using self.save_weights().
            Default is None.
        in_order : bool
            Allow loading weights into model in a sequential way or by name. Only useful when 'format' is 'hdf5'.
            If 'in_order' is True, weights from the file will be loaded into model in a sequential way.
            If 'in_order' is False, weights from the file will be loaded into model by matching the name
            with the weights of the model, particularly useful when trying to restore model in eager(graph) mode from
            a weights file which is saved in graph(eager) mode.
            Default is True.
        skip : bool
            Allow skipping weights whose name is mismatched between the file and model. Only useful when 'format' is
            'hdf5' or 'npz_dict'. If 'skip' is True, 'in_order' argument will be ignored and those loaded weights
            whose name is not found in model weights (self.all_weights) will be skipped. If 'skip' is False, error will
            occur when mismatch is found.
            Default is False.

        Examples
        --------
        1) load model from a hdf5 file.
        >>> net = vgg16()
        >>> optimizer = tl.optimizers.Adam(learning_rate=0.001)
        >>> metrics = tl.metrics.Accuracy()
        >>> model = tl.model.Model(network=net, loss_fn=tl.losses.softmax_cross_entropy_with_logits, optimizer=optimizer, metrics=metrics)
        >>> model.load_weights('./model_graph.h5', in_order=False, skip=True) # load weights by name, skipping mismatch
        >>> model.load_weights('./model_eager.h5') # load sequentially

        2) load model from a npz file
        >>> model.load_weights('./model.npz')

        3) load model from a npz file, which is saved as npz_dict previously
        >>> model.load_weights('./model.npz', format='npz_dict')

        Notes
        -------
        1) 'in_order' is only useful when 'format' is 'hdf5'. If you are trying to load a weights file which is
           saved in a different mode, it is recommended to set 'in_order' be True.
        2) 'skip' is useful when 'format' is 'hdf5' or 'npz_dict'. If 'skip' is True,
           'in_order' argument will be ignored.

        """

        _load_weights(net=self, file_path=file_path, format=format, in_order=in_order, skip=skip)

    def tf_train(
        self, n_epoch, train_dataset, network, loss_fn, train_weights, optimizer, metrics, print_train_batch,
        print_freq, test_dataset
    ):
        for epoch in range(n_epoch):
            start_time = time.time()

            train_loss, train_acc, n_iter = 0, 0, 0
            for X_batch, y_batch in train_dataset:
                network.set_train()

                with tf.GradientTape() as tape:
                    # compute outputs
                    _logits = network(X_batch)
                    # compute loss and update model
                    _loss_ce = loss_fn(_logits, y_batch)

                grad = tape.gradient(_loss_ce, train_weights)
                optimizer.apply_gradients(zip(grad, train_weights))

                train_loss += _loss_ce
                if metrics:
                    metrics.update(_logits, y_batch)
                    train_acc += metrics.result()
                    metrics.reset()
                else:
                    train_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
                n_iter += 1

                if print_train_batch:
                    print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                    print("   train loss: {}".format(train_loss / n_iter))
                    print("   train acc:  {}".format(train_acc / n_iter))

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                print("   train loss: {}".format(train_loss / n_iter))
                print("   train acc:  {}".format(train_acc / n_iter))

            if test_dataset:
                # use training and evaluation sets to evaluate the model every print_freq epoch
                if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                    network.set_eval()
                    val_loss, val_acc, n_iter = 0, 0, 0
                    for X_batch, y_batch in test_dataset:
                        _logits = network(X_batch)  # is_train=False, disable dropout
                        val_loss += loss_fn(_logits, y_batch, name='eval_loss')
                        if metrics:
                            metrics.update(_logits, y_batch)
                            val_acc += metrics.result()
                            metrics.reset()
                        else:
                            val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
                        n_iter += 1
                    print("   val loss: {}".format(val_loss / n_iter))
                    print("   val acc:  {}".format(val_acc / n_iter))

    def ms_train(
        self, n_epoch, train_dataset, network, loss_fn, train_weights, optimizer, metrics, print_train_batch,
        print_freq, test_dataset
    ):
        net_with_criterion = WithLoss(network, loss_fn)
        train_network = GradWrap(net_with_criterion, network.trainable_weights)
        train_network.set_train()
        for epoch in range(n_epoch):
            start_time = time.time()
            train_loss, train_acc, n_iter = 0, 0, 0
            for X_batch, y_batch in train_dataset:
                output = network(X_batch)
                loss_output = loss_fn(output, y_batch)
                grads = train_network(X_batch, y_batch)
                success = optimizer.apply_gradients(zip(grads, train_weights))
                loss = loss_output.asnumpy()
                train_loss += loss
                if metrics:
                    metrics.update(output, y_batch)
                    train_acc += metrics.result()
                    metrics.reset()
                else:
                    train_acc += np.mean((P.Equal()(P.Argmax(axis=1)(output), y_batch).asnumpy()))
                n_iter += 1

                if print_train_batch:
                    print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                    print("   train loss: {}".format(train_loss / n_iter))
                    print("   train acc:  {}".format(train_acc / n_iter))

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                print("   train loss: {}".format(train_loss / n_iter))
                print("   train acc:  {}".format(train_acc / n_iter))

            if test_dataset:
                # use training and evaluation sets to evaluate the model every print_freq epoch
                if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                    network.set_eval()
                    val_loss, val_acc, n_iter = 0, 0, 0
                    for X_batch, y_batch in test_dataset:
                        _logits = network(X_batch)
                        val_loss += loss_fn(_logits, y_batch, name='eval_loss')
                        if metrics:
                            metrics.update(_logits, y_batch)
                            val_acc += metrics.result()
                            metrics.reset()
                        else:
                            val_acc += np.mean((P.Equal()(P.Argmax(axis=1)(_logits), y_batch).asnumpy()))
                        n_iter += 1
                    print("   val loss: {}".format(val_loss / n_iter))
                    print("   val acc:  {}".format(val_acc / n_iter))

    def pd_train(
        self, n_epoch, train_dataset, network, loss_fn, train_weights, optimizer, metrics, print_train_batch,
        print_freq, test_dataset
    ):
        for epoch in range(n_epoch):
            start_time = time.time()

            train_loss, train_acc, n_iter = 0, 0, 0
            for X_batch, y_batch in train_dataset:
                network.set_train()

                output = network(X_batch)
                loss = loss_fn(output, y_batch)
                loss_ce = loss.numpy()
                grads = optimizer.gradient(loss, train_weights)
                optimizer.apply_gradients(zip(grads, train_weights))

                train_loss += loss_ce
                if metrics:
                    metrics.update(output, y_batch)
                    train_acc += metrics.result()
                    metrics.reset()
                else:
                    train_acc += pd.metric.accuracy(output, y_batch)
                n_iter += 1

                if print_train_batch:
                    print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                    print("   train loss: {}".format(train_loss / n_iter))
                    print("   train acc:  {}".format(train_acc / n_iter))

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                print("   train loss: {}".format(train_loss / n_iter))
                print("   train acc:  {}".format(train_acc / n_iter))

            if test_dataset:
                # use training and evaluation sets to evaluate the model every print_freq epoch
                if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                    network.set_eval()
                    val_loss, val_acc, n_iter = 0, 0, 0
                    for X_batch, y_batch in test_dataset:
                        _logits = network(X_batch)  # is_train=False, disable dropout
                        val_loss += loss_fn(_logits, y_batch, name='eval_loss')
                        if metrics:
                            metrics.update(_logits, y_batch)
                            val_acc += metrics.result()
                            metrics.reset()
                        else:
                            val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
                        n_iter += 1
                    print("   val loss: {}".format(val_loss / n_iter))
                    print("   val acc:  {}".format(val_acc / n_iter))

    def th_train(
            self, n_epoch, train_dataset, network, loss_fn, train_weights, optimizer, metrics, print_train_batch,
            print_freq, test_dataset
    ):
        for epoch in range(n_epoch):
            start_time = time.time()

            train_loss, train_acc, n_iter = 0, 0, 0
            for X_batch, y_batch in train_dataset:
                network.set_train()

                output = network(X_batch)
                loss = loss_fn(output, y_batch)

                grads = optimizer.gradient(loss, train_weights)
                optimizer.apply_gradients(zip(grads, train_weights))

                train_loss += loss
                if metrics:
                    pass
                    # metrics.update(output, y_batch)
                    # train_acc += metrics.result()
                    # metrics.reset()
                else:
                    train_acc += (output.argmax(1) == y_batch).type(torch.float).sum().item()
                n_iter += 1

                if print_train_batch:
                    print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                    print("   train loss: {}".format(train_loss / n_iter))
                    print("   train acc:  {}".format(train_acc / n_iter))

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                print("   train loss: {}".format(train_loss / n_iter))
                print("   train acc:  {}".format(train_acc / n_iter))


class WithLoss(Module):
    """
    High-Level API for Training or Testing.

    Wraps the network with loss function. This Module accepts data and label as inputs and
    the computed loss will be returned.

    Parameters
    ----------
    backbone : tensorlayer model
        The tensorlayer network.
    loss_fn : function
        Objective function

    Methods
    ---------
    forward()
        Model inference.

    Examples
    --------
    >>> import tensorlayerx as tl
    >>> net = vgg16()
    >>> loss_fn = tl.losses.softmax_cross_entropy_with_logits
    >>> net_with_loss = tl.model.WithLoss(net, loss_fn)

    """

    def __init__(self, backbone, loss_fn):
        super(WithLoss, self).__init__()
        self._backbone = backbone
        self._loss_fn = loss_fn

    def forward(self, data, label):
        out = self._backbone(data)
        return self._loss_fn(out, label)

    @property
    def backbone_network(self):
        return self._backbone


class GradWrap(Module):
    """ GradWrap definition """

    def __init__(self, network, trainable_weights):
        super(GradWrap, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(trainable_weights)

    def forward(self, x, label):
        return composite.GradOperation(get_by_list=True)(self.network, self.weights)(x, label)


class WithGradMS(Module):
    "Module that returns the gradients."
    def __init__(self, network, loss_fn=None, sens=None, optimizer=None):
        super(WithGradMS, self).__init__()
        self.network = network
        self.loss_fn = loss_fn
        self.weights = ParameterTuple(network.trainable_weights)
        self.grad = composite.GradOperation(get_by_list=True, sens_param=(sens is not None))
        self.sens = sens
        self.optimizer = optimizer
        if self.loss_fn is None:
            self.network_with_loss = network
        else:
            self.network_with_loss = WithLoss(self.network, self.loss_fn)
        self.network.set_train()

    def forward(self, inputs, label):
        grads = self.grad(self.network_with_loss, self.weights)(inputs, label)
        return grads


class WithGradTF(object):
    def __init__(self, network, loss_fn=None, optimizer=None):
        self.network = network
        self.loss_fn = loss_fn
        self.train_weights = self.network.trainable_weights
        self.optimizer = optimizer
        if loss_fn is None:
            self.network_with_loss = network
        else:
            self.network_with_loss = WithLoss(self.network, self.loss_fn)
        self.network.set_train()

    def __call__(self, inputs, label):
        with tf.GradientTape() as tape:
            loss = self.network_with_loss(inputs, label)
        grads = tape.gradient(loss, self.train_weights)
        return grads


class WithGradPD(object):
    def __init__(self, network, loss_fn=None, optimizer=None):
        self.network = network
        self.loss_fn = loss_fn
        self.train_weights = self.network.trainable_weights
        self.optimizer = optimizer
        if loss_fn is None:
            self.network_with_loss = network
        else:
            self.network_with_loss = WithLoss(self.network, self.loss_fn)
        self.network.set_train()

    def __call__(self, inputs, label):
        loss = self.network_with_loss(inputs, label)
        grads = self.optimizer.gradient(loss, self.train_weights)
        return grads


class TrainOneStepWithTF(object):

    def __init__(self, net_with_loss, optimizer, train_weights):
        self.net_with_loss = net_with_loss
        self.optimzer = optimizer
        self.train_weights = train_weights

    def __call__(self, data, label):
        with tf.GradientTape() as tape:
            loss = self.net_with_loss(data, label)
        grad = tape.gradient(loss, self.train_weights)
        self.optimzer.apply_gradients(zip(grad, self.train_weights))
        return loss


class TrainOneStepWithMS(object):

    def __init__(self, net_with_loss, optimizer, train_weights):
        self.net_with_loss = net_with_loss
        self.optimizer = optimizer
        self.train_weights = train_weights
        self.net_with_loss = net_with_loss
        self.train_network = GradWrap(net_with_loss, train_weights)

    def __call__(self, data, label):
        loss = self.net_with_loss(data, label)
        grads = self.train_network(data, label)
        self.optimizer.apply_gradients(zip(grads, self.train_weights))
        loss = loss.asnumpy()
        return loss


class TrainOneStepWithPD(object):

    def __init__(self, net_with_loss, optimizer, train_weights):
        self.net_with_loss = net_with_loss
        self.optimizer = optimizer
        self.train_weights = train_weights

    def __call__(self, data, label):
        loss = self.net_with_loss(data, label)
        grads = self.optimizer.gradient(loss, self.train_weights)
        self.optimizer.apply_gradients(zip(grads, self.train_weights))
        return loss.numpy()


class TrainOneStepWithTH(object):
    def __init__(self, net_with_loss, optimizer, train_weights):
        self.net_with_loss = net_with_loss
        self.optimizer = optimizer
        self.train_weights = train_weights

    def __call__(self, data, label):
        loss = self.net_with_loss(data, label)
        grads = self.optimizer.gradient(loss, self.train_weights)
        self.optimizer.apply_gradients(zip(grads, self.train_weights))
        return loss


class WithGrad(object):
    """Module that returns the gradients.

    Parameters
    ----------
    network : tensorlayer model
        The tensorlayer network.
    loss_fn : function
        Objective function
    optimizer : class
        Optimizer for updating the weights

    Examples
    --------
    >>> import tensorlayerx as tl
    >>> net = vgg16()
    >>> loss_fn = tl.losses.softmax_cross_entropy_with_logits
    >>> optimizer = tl.optimizers.Adam(learning_rate=1e-3)
    >>> net_with_grad = tl.model.WithGrad(net, loss_fn, optimizer)
    >>> inputs, labels = tl.layers.Input((128, 784), dtype=tl.float32), tl.layers.Input((128, 1), dtype=tl.int32)
    >>> net_with_grad(inputs, labels)

    """
    def __init__(self, network, loss_fn=None, optimizer=None):
        if tl.BACKEND == 'tensorflow':
            self.net_with_grad = WithGradTF(network, loss_fn, optimizer)
        elif tl.BACKEND == 'mindspore':
            self.net_with_grad = WithGradMS(network, loss_fn, optimizer)
        elif tl.BACKEND == 'paddle':
            self.net_with_grad = WithGradPD(network, loss_fn, optimizer)
        else:
            raise NotImplementedError("This backend is not supported")

    def __call__(self, data, label):
        grad = self.net_with_grad(data, label)
        return grad


class TrainOneStep(object):
    """
    High-Level API for Training One Step.

    Wraps the network with an optimizer. It can be trained in one step using the optimizer to get the loss.

    Parameters
    ----------
    net_with_loss : tensorlayer WithLoss
        The training or testing network.
    optimizer : class
        Optimizer for updating the weights
    train_weights : class
        Dict or set of metrics to be evaluated by the model during

    Examples
    --------
    >>> import tensorlayerx as tl
    >>> net = vgg16()
    >>> train_weights = net.trainable_weights
    >>> loss_fn = tl.losses.softmax_cross_entropy_with_logits
    >>> optimizer = tl.optimizers.Adam(learning_rate=1e-3)
    >>> net_with_loss = tl.model.WithLoss(net, loss_fn)
    >>> train_one_step = tl.model.TrainOneStep(net_with_loss, optimizer, train_weights)
    >>> inputs, labels = tl.layers.Input((128, 784), dtype=tl.float32), tl.layers.Input((128, 1), dtype=tl.int32)
    >>> train_one_step(inputs, labels)

    """

    def __init__(self, net_with_loss, optimizer, train_weights):
        if tl.BACKEND == 'tensorflow':
            self.net_with_train = TrainOneStepWithTF(net_with_loss, optimizer, train_weights)
        elif tl.BACKEND == 'mindspore':
            self.net_with_train = TrainOneStepWithMS(net_with_loss, optimizer, train_weights)
        elif tl.BACKEND == 'paddle':
            self.net_with_train = TrainOneStepWithPD(net_with_loss, optimizer, train_weights)
        elif tl.BACKEND == 'torch':
            self.net_with_train = TrainOneStepWithTH(net_with_loss, optimizer, train_weights)
        else:
            raise NotImplementedError("This backend is not supported")

    def __call__(self, data, label):
        loss = self.net_with_train(data, label)
        return loss
