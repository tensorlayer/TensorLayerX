#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections.abc import Iterable

from tensorlayerx.nn.core.common import _save_weights, _load_weights, \
    _save_standard_weights_dict, _load_standard_weights_dict
from .utils import WithLoss, WithGradPD, WithGradMS, WithGradTF,WithGradJT, TrainOneStepWithPD, \
    TrainOneStepWithMS, TrainOneStepWithTH,TrainOneStepWithJT, TrainOneStepWithTF, GradWrap, \
    TrainOneStepWithGradientClippingTF,TrainOneStepWithGradientClippingJT, TrainOneStepWithGradientClippingPD, TrainOneStepWithGradientClippingTH
import tensorlayerx as tlx
from tensorlayerx.nn import Module
import numpy as np
import time

from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

if tlx.BACKEND == 'tensorflow':
    import tensorflow as tf
if tlx.BACKEND == 'mindspore':
    from mindspore.ops import operations as P
if tlx.BACKEND == 'paddle':
    import paddle as pd
if tlx.BACKEND == 'torch':
    import torch
if tlx.BACKEND == 'jittor':
    import jittor as jt
__all__ = ['Model', 'WithLoss', 'WithGrad', 'TrainOneStep', 'TrainOneStepWithGradientClipping']


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
    >>> import tensorlayerx as tlx
    >>> class Net(Module):
    >>>     def __init__(self):
    >>>         super(Net, self).__init__()
    >>>         self.conv = tlx.nn.Conv2d(out_channels=32, kernel_size=(3, 3), stride=(2, 2), in_channels=5, name='conv2d')
    >>>         self.bn = tlx.nn.BatchNorm2d(num_features=32, act=tlx.ReLU)
    >>>         self.flatten = tlx.nn.Flatten()
    >>>         self.fc = tlx.nn.Linear(out_features=12, in_features=32*224*224) # padding=0
    >>>
    >>>     def construct(self, x):
    >>>         x = self.conv(x)
    >>>         x = self.bn(x)
    >>>         x = self.flatten(x)
    >>>         out = self.fc(x)
    >>>         return out
    >>>
    >>> net = Net()
    >>> loss = tlx.losses.softmax_cross_entropy_with_logits
    >>> optim = tlx.optimizers.Momentum(params=net.trainable_weights, learning_rate=0.1, momentum=0.9)
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

        if tlx.BACKEND == 'tensorflow':
            self.tf_train(
                n_epoch=n_epoch, train_dataset=train_dataset, network=self.network, loss_fn=self.loss_fn,
                train_weights=self.train_weights, optimizer=self.optimizer, metrics=self.metrics,
                print_train_batch=print_train_batch, print_freq=print_freq, test_dataset=test_dataset
            )
        elif tlx.BACKEND == 'mindspore':
            self.ms_train(
                n_epoch=n_epoch, train_dataset=train_dataset, network=self.network, loss_fn=self.loss_fn,
                train_weights=self.train_weights, optimizer=self.optimizer, metrics=self.metrics,
                print_train_batch=print_train_batch, print_freq=print_freq, test_dataset=test_dataset
            )
        elif tlx.BACKEND == 'paddle':
            self.pd_train(
                n_epoch=n_epoch, train_dataset=train_dataset, network=self.network, loss_fn=self.loss_fn,
                train_weights=self.train_weights, optimizer=self.optimizer, metrics=self.metrics,
                print_train_batch=print_train_batch, print_freq=print_freq, test_dataset=test_dataset
            )
        elif tlx.BACKEND == 'torch':
            self.th_train(
                n_epoch=n_epoch, train_dataset=train_dataset, network=self.network, loss_fn=self.loss_fn,
                train_weights=self.train_weights, optimizer=self.optimizer, metrics=self.metrics,
                print_train_batch=print_train_batch, print_freq=print_freq, test_dataset=test_dataset
            )

        elif tlx.BACKEND == "oneflow":
            self.of_train(
                n_epoch=n_epoch, train_dataset=train_dataset, network=self.network, loss_fn=self.loss_fn,
                train_weights=self.train_weights, optimizer=self.optimizer, metrics=self.metrics,
                print_train_batch=print_train_batch, print_freq=print_freq, test_dataset=test_dataset,
            )

        elif tlx.BACKEND == "jittor":
            self.jt_train(
                n_epoch=n_epoch, train_dataset=train_dataset, network=self.network, loss_fn=self.loss_fn,
                train_weights=self.train_weights, optimizer=self.optimizer, metrics=self.metrics,
                print_train_batch=print_train_batch, print_freq=print_freq, test_dataset=test_dataset,
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
        linear.weights: weights_parm ...}

        Parameters
        ----------
        file_path : str
            Name of the saved file

        """

        _save_standard_weights_dict(self.network, file_path)

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

        _load_standard_weights_dict(self.network, file_path, skip, reshape, format)

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
        >>> optimizer = tlx.optimizers.Adam(learning_rate=0.001)
        >>> metrics = tlx.metrics.Accuracy()
        >>> model = tlx.model.Model(network=net, loss_fn=tlx.losses.softmax_cross_entropy_with_logits, optimizer=optimizer, metrics=metrics)
        >>> model.save_weights('./model.h5')
        ...
        >>> model.load_weights('./model.h5')

        2) Save model weights in npz/npz_dict format
        >>> model.save_weights('./model.npz')
        >>> model.save_weights('./model.npz', format='npz_dict')

        """

        _save_weights(net=self.network, file_path=file_path, format=format)

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
        >>> optimizer = tlx.optimizers.Adam(learning_rate=0.001)
        >>> metrics = tlx.metrics.Accuracy()
        >>> model = tlx.model.Model(network=net, loss_fn=tlx.losses.softmax_cross_entropy_with_logits, optimizer=optimizer, metrics=metrics)
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

        _load_weights(net=self.network, file_path=file_path, format=format, in_order=in_order, skip=skip)

    def tf_train(
        self, n_epoch, train_dataset, network, loss_fn, train_weights, optimizer, metrics, print_train_batch,
        print_freq, test_dataset
    ):
        with Progress(TextColumn("[progress.description]{task.description}"),
                      BarColumn(),
                      TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                      TimeRemainingColumn(),
                      TimeElapsedColumn()) as progress:

            n_batch = len(train_dataset)
            epoch_tqdm = progress.add_task(description="[red]Epoch progress 0/{}".format(n_epoch), total=n_epoch)
            batch_tqdm = progress.add_task(description="[green]Batch progress 0/{}".format(n_batch), total=n_batch)

            for epoch in range(n_epoch):
                start_time = time.time()

                train_loss, train_acc, n_iter = 0, 0, 0
                for batch, (X_batch, y_batch) in enumerate(train_dataset):
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
                    progress.advance(batch_tqdm, advance=1)
                    progress.update(batch_tqdm, description="[green]Batch progress {}/{}".format(batch + 1, n_batch))

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
                            val_loss += loss_fn(_logits, y_batch)
                            if metrics:
                                metrics.update(_logits, y_batch)
                                val_acc += metrics.result()
                                metrics.reset()
                            else:
                                val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
                            n_iter += 1
                        print("   val loss: {}".format(val_loss / n_iter))
                        print("   val acc:  {}".format(val_acc / n_iter))
                progress.update(epoch_tqdm, description="[red]Epoch progress {}/{}".format(epoch + 1, n_epoch))
                progress.advance(epoch_tqdm, advance=1)
                progress.reset(batch_tqdm)


    def ms_train(
        self, n_epoch, train_dataset, network, loss_fn, train_weights, optimizer, metrics, print_train_batch,
        print_freq, test_dataset
    ):
        net_with_criterion = WithLoss(network, loss_fn)
        train_network = GradWrap(net_with_criterion, network.trainable_weights)
        train_network.set_train()

        with Progress(TextColumn("[progress.description]{task.description}"),
                      BarColumn(),
                      TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                      TimeRemainingColumn(),
                      TimeElapsedColumn()) as progress:

            n_batch = len(train_dataset)
            epoch_tqdm = progress.add_task(description="[red]Epoch progress 0/{}".format(n_epoch), total=n_epoch)
            batch_tqdm = progress.add_task(description="[green]Batch progress 0/{}".format(n_batch), total=n_batch)

            for epoch in range(n_epoch):
                start_time = time.time()
                train_loss, train_acc, n_iter = 0, 0, 0
                for batch, (X_batch, y_batch) in enumerate(train_dataset):
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
                    progress.advance(batch_tqdm, advance=1)
                    progress.update(batch_tqdm, description="[green]Batch progress {}/{}".format(batch + 1, n_batch))

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
                            val_loss += loss_fn(_logits, y_batch)
                            if metrics:
                                metrics.update(_logits, y_batch)
                                val_acc += metrics.result()
                                metrics.reset()
                            else:
                                val_acc += np.mean((P.Equal()(P.Argmax(axis=1)(_logits), y_batch).asnumpy()))
                            n_iter += 1
                        print("   val loss: {}".format(val_loss / n_iter))
                        print("   val acc:  {}".format(val_acc / n_iter))

                progress.update(epoch_tqdm, description="[red]Epoch progress {}/{}".format(epoch + 1, n_epoch))
                progress.advance(epoch_tqdm, advance=1)
                progress.reset(batch_tqdm)


    def pd_train(
        self, n_epoch, train_dataset, network, loss_fn, train_weights, optimizer, metrics, print_train_batch,
        print_freq, test_dataset
    ):
        with Progress(TextColumn("[progress.description]{task.description}"),
                      BarColumn(),
                      TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                      TimeRemainingColumn(),
                      TimeElapsedColumn()) as progress:

            n_batch = len(train_dataset)
            epoch_tqdm = progress.add_task(description="[red]Epoch progress 0/{}".format(n_epoch), total=n_epoch)
            batch_tqdm = progress.add_task(description="[green]Batch progress 0/{}".format(n_batch), total=n_batch)

            for epoch in range(n_epoch):
                start_time = time.time()

                train_loss, train_acc, n_iter = 0, 0, 0
                for batch, (X_batch, y_batch) in enumerate(train_dataset):
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
                    progress.advance(batch_tqdm, advance=1)
                    progress.update(batch_tqdm, description="[green]Batch progress {}/{}".format(batch + 1, n_batch))

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
                            val_loss += loss_fn(_logits, y_batch)
                            if metrics:
                                metrics.update(_logits, y_batch)
                                val_acc += metrics.result()
                                metrics.reset()
                            else:
                                val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
                            n_iter += 1
                        print("   val loss: {}".format(val_loss / n_iter))
                        print("   val acc:  {}".format(val_acc / n_iter))
                progress.update(epoch_tqdm, description="[red]Epoch progress {}/{}".format(epoch + 1, n_epoch))
                progress.advance(epoch_tqdm, advance=1)
                progress.reset(batch_tqdm)


    def th_train(
        self, n_epoch, train_dataset, network, loss_fn, train_weights, optimizer, metrics, print_train_batch,
        print_freq, test_dataset
    ):
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # network = network.to(device)
        with Progress(TextColumn("[progress.description]{task.description}"),
                      BarColumn(),
                      TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                      TimeRemainingColumn(),
                      TimeElapsedColumn()) as progress:

            n_batch = len(train_dataset)
            epoch_tqdm = progress.add_task(description="[red]Epoch progress 0/{}".format(n_epoch), total=n_epoch)
            batch_tqdm = progress.add_task(description="[green]Batch progress 0/{}".format(n_batch), total=n_batch)

            for epoch in range(n_epoch):
                start_time = time.time()

                train_loss, train_acc, n_iter = 0, 0, 0
                for batch, (X_batch, y_batch) in enumerate(train_dataset):
                    network.set_train()
                    output = network(X_batch)
                    loss = loss_fn(output, y_batch)
                    grads = optimizer.gradient(loss, train_weights)
                    optimizer.apply_gradients(zip(grads, train_weights))

                    train_loss += loss
                    if metrics:
                        metrics.update(output, y_batch)
                        train_acc += metrics.result()
                        metrics.reset()
                    else:
                        train_acc += (output.argmax(1) == y_batch).type(torch.float).mean().item()
                    n_iter += 1

                    if print_train_batch:
                        print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                        print("   train loss: {}".format(train_loss / n_iter))
                        print("   train acc:  {}".format(train_acc / n_iter))
                    progress.advance(batch_tqdm, advance=1)
                    progress.update(batch_tqdm, description="[green]Batch progress {}/{}".format(batch + 1, n_batch))

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
                            val_loss += loss_fn(_logits, y_batch)
                            if metrics:
                                metrics.update(_logits, y_batch)
                                val_acc += metrics.result()
                                metrics.reset()
                            else:
                                val_acc += (_logits.argmax(1) == y_batch).type(torch.float).mean().item()
                            n_iter += 1
                        print("   val loss: {}".format(val_loss / n_iter))
                        print("   val acc:  {}".format(val_acc / n_iter))
                progress.update(epoch_tqdm, description="[red]Epoch progress {}/{}".format(epoch + 1, n_epoch))
                progress.advance(epoch_tqdm, advance=1)
                progress.reset(batch_tqdm)



    def of_train(
        self, n_epoch, train_dataset, network, loss_fn, train_weights, optimizer, metrics, print_train_batch,
        print_freq, test_dataset
    ):
        with Progress(TextColumn("[progress.description]{task.description}"),
                      BarColumn(),
                      TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                      TimeRemainingColumn(),
                      TimeElapsedColumn()) as progress:

            n_batch = len(train_dataset)
            epoch_tqdm = progress.add_task(description="[red]Epoch progress 0/{}".format(n_epoch), total=n_epoch)
            batch_tqdm = progress.add_task(description="[green]Batch progress 0/{}".format(n_batch), total=n_batch)

            for epoch in range(n_epoch):
                start_time = time.time()

                train_loss, train_acc, n_iter = 0, 0, 0
                for batch, (X_batch, y_batch) in enumerate(train_dataset):
                    network.set_train()
                    output = network(X_batch)
                    loss = loss_fn(output, y_batch)
                    grads = optimizer.gradient(loss, train_weights)
                    optimizer.apply_gradients(zip(grads, train_weights))

                    train_loss += loss
                    if metrics:
                        metrics.update(output, y_batch)
                        train_acc += metrics.result()
                        metrics.reset()
                    else:
                        train_acc += (output.argmax(1) == y_batch).type(torch.float).mean().item()
                    n_iter += 1

                    if print_train_batch:
                        print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                        print("   train loss: {}".format(train_loss / n_iter))
                        print("   train acc:  {}".format(train_acc / n_iter))
                    progress.advance(batch_tqdm, advance=1)
                    progress.update(batch_tqdm, description="[green]Batch progress {}/{}".format(batch + 1, n_batch))

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
                            val_loss += loss_fn(_logits, y_batch)
                            if metrics:
                                metrics.update(_logits, y_batch)
                                val_acc += metrics.result()
                                metrics.reset()
                            else:
                                val_acc += (_logits.argmax(1) == y_batch).type(torch.float).mean().item()
                            n_iter += 1
                        print("   val loss: {}".format(val_loss / n_iter))
                        print("   val acc:  {}".format(val_acc / n_iter))
                progress.update(epoch_tqdm, description="[red]Epoch progress {}/{}".format(epoch + 1, n_epoch))
                progress.advance(epoch_tqdm, advance=1)
                progress.reset(batch_tqdm)


    def jt_train(
        self, n_epoch, train_dataset, network, loss_fn, train_weights, optimizer, metrics, print_train_batch,
        print_freq, test_dataset
    ):
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # network = network.to(device)
        with Progress(TextColumn("[progress.description]{task.description}"),
                      BarColumn(),
                      TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                      TimeRemainingColumn(),
                      TimeElapsedColumn()) as progress:

            n_batch = len(train_dataset)
            epoch_tqdm = progress.add_task(description="[red]Epoch progress 0/{}".format(n_epoch), total=n_epoch)
            batch_tqdm = progress.add_task(description="[green]Batch progress 0/{}".format(n_batch), total=n_batch)

            for epoch in range(n_epoch):
                start_time = time.time()

                train_loss, train_acc, n_iter = 0, 0, 0
                for batch, (X_batch, y_batch) in enumerate(train_dataset):
                    network.set_train()
                    output = network(X_batch)
                    loss = loss_fn(output, y_batch)
                    # optimizer.apply_gradients(loss, train_weights)
                    # grads = optimizer.gradient(loss, train_weights)
                    # optimizer.apply_gradients(zip(grads, train_weights))

                    optimizer.set(train_weights)
                    optimizer.zero_grad()
                    optimizer.step(loss)
                    train_loss += loss.item()
               
                    if metrics:
                        metrics.update(y_pred=output,y_true= y_batch)
                        train_acc += metrics.result() 
                        metrics.reset()
                    else:
                        train_acc += np.mean(np.equal(np.argmax(output, axis=1), y_batch))
                    n_iter += 1

                    if print_train_batch:
                        print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                        print("   train loss: {}".format(train_loss / n_iter))
                        print("   train acc:  {}".format(train_acc / n_iter))
                    progress.advance(batch_tqdm, advance=1)
                    progress.update(batch_tqdm, description="[green]Batch progress {}/{}".format(batch + 1, n_batch))

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
                            val_loss += loss_fn(_logits, y_batch)
                            if metrics:
                                metrics.update(_logits, y_batch)
                                val_acc += metrics.result()
                                metrics.reset()
                            else:
                                val_acc += (_logits.argmax(1) == y_batch).type(jt.float).mean().item()
                            n_iter += 1
                        print("   val loss: {}".format(val_loss / n_iter))
                        print("   val acc:  {}".format(val_acc / n_iter))
                progress.update(epoch_tqdm, description="[red]Epoch progress {}/{}".format(epoch + 1, n_epoch))
                progress.advance(epoch_tqdm, advance=1)
                progress.reset(batch_tqdm)



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
    >>> import tensorlayerx as tlx
    >>> net = vgg16()
    >>> loss_fn = tlx.losses.softmax_cross_entropy_with_logits
    >>> optimizer = tlx.optimizers.Adam(learning_rate=1e-3)
    >>> net_with_grad = tlx.model.WithGrad(net, loss_fn, optimizer)
    >>> inputs, labels = tlx.nn.Input((128, 784), dtype=tlx.float32), tlx.nn.Input((128, 1), dtype=tlx.int32)
    >>> net_with_grad(inputs, labels)

    """

    def __init__(self, network, loss_fn=None, optimizer=None):
        if tlx.BACKEND == 'tensorflow':
            self.net_with_grad = WithGradTF(network, loss_fn, optimizer)
        elif tlx.BACKEND == 'mindspore':
            self.net_with_grad = WithGradMS(network, loss_fn, optimizer)
        elif tlx.BACKEND == 'paddle':
            self.net_with_grad = WithGradPD(network, loss_fn, optimizer)
        elif tlx.BACKEND == 'jittor':
            self.net_with_grad = WithGradJT(network, loss_fn, optimizer)            
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
    >>> import tensorlayerx as tlx
    >>> net = vgg16()
    >>> train_weights = net.trainable_weights
    >>> loss_fn = tlx.losses.softmax_cross_entropy_with_logits
    >>> optimizer = tlx.optimizers.Adam(learning_rate=1e-3)
    >>> net_with_loss = tlx.model.WithLoss(net, loss_fn)
    >>> train_one_step = tlx.model.TrainOneStep(net_with_loss, optimizer, train_weights)
    >>> inputs, labels = tlx.nn.Input((128, 784), dtype=tlx.float32), tlx.nn.Input((128, 1), dtype=tlx.int32)
    >>> train_one_step(inputs, labels)

    """

    def __init__(self, net_with_loss, optimizer, train_weights):
        if tlx.BACKEND == 'tensorflow':
            self.net_with_train = TrainOneStepWithTF(net_with_loss, optimizer, train_weights)
        elif tlx.BACKEND == 'mindspore':
            self.net_with_train = TrainOneStepWithMS(net_with_loss, optimizer, train_weights)
        elif tlx.BACKEND == 'paddle':
            self.net_with_train = TrainOneStepWithPD(net_with_loss, optimizer, train_weights)
        elif tlx.BACKEND == 'torch':
            self.net_with_train = TrainOneStepWithTH(net_with_loss, optimizer, train_weights)
        elif tlx.BACKEND == 'jittor':
            self.net_with_train = TrainOneStepWithJT(net_with_loss, optimizer, train_weights)
        else:
            raise NotImplementedError("This backend is not supported")

    def __call__(self, data, label, *args, **kwargs):
        loss = self.net_with_train(data, label, *args, **kwargs)
        return loss


class TrainOneStepWithGradientClipping(object):
    """
    High-Level API for Training One Step, And  do gradient clipping.

    Wraps the network with an optimizer. It can be trained in one step using the optimizer to get the loss.
    It can do gradient clipping using the clipping function.

    Parameters
    ----------
    net_with_loss : tensorlayer WithLoss
        The training or testing network.
    optimizer : class
        Optimizer for updating the weights
    train_weights : class
        Dict or set of metrics to be evaluated by the model during
    gradient_clipping : class
        Clips gradient norm of an iterable of parameters.

    Examples
    --------
    >>> import tensorlayerx as tlx
    >>> net = vgg16()
    >>> train_weights = net.trainable_weights
    >>> loss_fn = tlx.losses.softmax_cross_entropy_with_logits
    >>> optimizer = tlx.optimizers.Adam(learning_rate=1e-3)
    >>> net_with_loss = tlx.model.WithLoss(net, loss_fn)
    >>> train_one_step_with_clip = tlx.model.TrainOneStepWithGradientClipping(net_with_loss, optimizer, train_weights, tlx.ops.ClipByGlobalNorm(0.1))
    >>> inputs, labels = tlx.nn.Input((128, 784), dtype=tlx.float32), tlx.nn.Input((128, 1), dtype=tlx.int32)
    >>> train_one_step_with_clip(inputs, labels)

    """

    def __init__(self, net_with_loss, optimizer, train_weights, gradient_clipping=tlx.ops.ClipByGlobalNorm(0.1)):
        if gradient_clipping is None:
            raise Exception("This method must input the gradient clipping function, eg tlx.ops.ClipByGlobalNorm(0.1).")

        if tlx.BACKEND == 'tensorflow':
            self.net_weith_train = TrainOneStepWithGradientClippingTF(
                net_with_loss, optimizer, train_weights, gradient_clipping)
        elif tlx.BACKEND == 'paddle':
            self.net_weith_train = TrainOneStepWithGradientClippingPD(
                net_with_loss, optimizer, train_weights, gradient_clipping)
        elif tlx.BACKEND == 'torch':
            self.net_weith_train = TrainOneStepWithGradientClippingTH(
                net_with_loss, optimizer, train_weights, gradient_clipping)
        elif tlx.BACKEND == 'jittor':
            self.net_weith_train = TrainOneStepWithGradientClippingJT(
                net_with_loss, optimizer, train_weights, gradient_clipping)
        else:
            raise NotImplementedError("This backend is not supported")

    def __call__(self, data, label):
        loss = self.net_weith_train(data, label)
        return loss
