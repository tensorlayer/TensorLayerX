#! /usr/bin/python
# -*- coding: utf-8 -*-
# The tensorlayerx and tensorflow operators can be mixed
import os
os.environ['TL_BACKEND'] = 'tensorflow'

import numpy as np
import time

import tensorflow as tf
import tensorlayerx as tl
from tensorlayerx.nn import Module
from tensorlayerx.nn import Dense
from tensorlayerx.model import WithGrad

X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))


class CustomModel(Module):

    def __init__(self):
        super(CustomModel, self).__init__()
        self.dense1 = Dense(n_units=800, in_channels=784)
        self.dense2 = Dense(n_units=800, act=tl.ReLU, in_channels=800)
        self.dense3 = Dense(n_units=10, act=tl.ReLU, in_channels=800)

    def forward(self, x, foo=None):
        z = self.dense1(x)
        z = self.dense2(z)
        out = self.dense3(z)
        if foo is not None:
            out = tl.ops.relu(out)
        return out


MLP = CustomModel()
n_epoch = 50
batch_size = 500
print_freq = 5
train_weights = MLP.trainable_weights
optimizer = tl.optimizers.Adam(lr=0.0001)

net_with_grad = WithGrad(network=MLP, loss_fn=tl.losses.softmax_cross_entropy_with_logits, optimizer=optimizer)

for epoch in range(n_epoch):  ## iterate the dataset n_epoch times
    start_time = time.time()
    ## iterate over the entire training set once (shuffle the data via training)
    for X_batch, y_batch in tl.utils.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
        MLP.set_train()  # enable dropout
        grad = net_with_grad(X_batch, y_batch)
        clip_grad, _ = tf.clip_by_global_norm(grad, 0.1)
        optimizer.apply_gradients(zip(clip_grad, train_weights))

    ## use training and evaluation sets to evaluate the model every print_freq epoch
    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
        train_loss, train_acc, n_iter = 0, 0, 0
        for X_batch, y_batch in tl.utils.iterate.minibatches(X_train, y_train, batch_size, shuffle=False):
            _logits = MLP(X_batch)
            train_loss += tl.losses.softmax_cross_entropy_with_logits(_logits, y_batch, name='eval_loss')
            train_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            n_iter += 1
        print("   train loss: {}".format(train_loss / n_iter))
        print("   train acc:  {}".format(train_acc / n_iter))
