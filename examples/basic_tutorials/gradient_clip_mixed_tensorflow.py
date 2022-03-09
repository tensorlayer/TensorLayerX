#! /usr/bin/python
# -*- coding: utf-8 -*-
# The tensorlayerx and tensorflow operators can be mixed
import os
os.environ['TL_BACKEND'] = 'tensorflow'
# os.environ['TL_BACKEND'] = 'paddle'
# os.environ['TL_BACKEND'] = 'torch'

import numpy as np
import time

import tensorflow as tf
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import Dense
from tensorlayerx.model import WithGrad

X_train, y_train, X_val, y_val, X_test, y_test = tlx.files.load_mnist_dataset(shape=(-1, 784))


class CustomModel(Module):

    def __init__(self):
        super(CustomModel, self).__init__()
        self.dense1 = Dense(n_units=800, in_channels=784)
        self.dense2 = Dense(n_units=800, act=tlx.ReLU, in_channels=800)
        self.dense3 = Dense(n_units=10, act=tlx.ReLU, in_channels=800)

    def forward(self, x, foo=None):
        z = self.dense1(x)
        z = self.dense2(z)
        out = self.dense3(z)
        if foo is not None:
            out = tlx.relu(out)
        return out


MLP = CustomModel()
n_epoch = 50
batch_size = 500
print_freq = 5
train_weights = MLP.trainable_weights
optimizer = tlx.optimizers.Adam(learning_rate=0.0001)

net_with_loss = tlx.model.WithLoss(backbone=MLP, loss_fn=tlx.losses.softmax_cross_entropy_with_logits)
net_with_grad_train = tlx.model.TrainOneStepWithGradientClipping(net_with_loss, optimizer, train_weights, gradient_clipping=tlx.ops.ClipGradByValue())

# net_with_grad = WithGrad(network=MLP, loss_fn=tlx.losses.softmax_cross_entropy_with_logits, optimizer=optimizer)

for epoch in range(n_epoch):  ## iterate the dataset n_epoch times
    start_time = time.time()
    ## iterate over the entire training set once (shuffle the data via training)
    for X_batch, y_batch in tlx.utils.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
        MLP.set_train()  # enable dropout
        # X_batch = tlx.convert_to_tensor(X_batch)
        # y_batch = tlx.convert_to_tensor(y_batch, dtype=tlx.int64)
        loss = net_with_grad_train(X_batch, y_batch)

    # use training and evaluation sets to evaluate the model every print_freq epoch
    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
        train_loss, train_acc, n_iter = 0, 0, 0
        for X_batch, y_batch in tlx.utils.iterate.minibatches(X_train, y_train, batch_size, shuffle=False):

            X_batch = tlx.convert_to_tensor(X_batch)
            y_batch = tlx.convert_to_tensor(y_batch, dtype=tlx.int64)

            _logits = MLP(X_batch)
            train_loss += tlx.losses.softmax_cross_entropy_with_logits(_logits, y_batch)
            train_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            n_iter += 1
        print("   train loss: {}".format(train_loss / n_iter))
        print("   train acc:  {}".format(train_acc / n_iter))
