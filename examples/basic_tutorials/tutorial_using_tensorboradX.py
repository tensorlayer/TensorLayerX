#! /usr/bin/python
# -*- coding: utf-8 -*-
# This code describes how TensoLayerX uses TensorBoard to monitor training.
# TensorLayerX uses tensorboardX to monitor the training situation, so it is necessary to install tensorboardX, The version of tensorboardX installed needs to match with TensorFlow.
# tensorboardX repo: https://github.com/lanpa/tensorboardX/blob/master/README.md

# Use the steps Description

# Step 1: install tensorboardX.
# pip install tensorboardX
# or build from source:
# pip install 'git+https://github.com/lanpa/tensorboardX'
# You can optionally install crc32c to speed up.
# pip install crc32c
# Starting from tensorboardX 2.1, You need to install soundfile for the add_audio() function (200x speedup).
# pip install soundfile

# Step 2: Creates writer1 object.The log will be saved in 'runs/mlp'
# writer = SummaryWriter('runs/mlp')

# Step 3:Use the add_scalar to record numeric constants.
# writer.add_scalar('train acc', train_acc / n_iter, train_batch)

# Step 4:start tensorboard on the command line
# tensorboard --logdir=<your_log_dir>
# eg. tensorboard --logdir=runs/mlp

# Step 5:viewing the content in a browser.
# Enter the http://localhost:6006 in the browser.


import os
os.environ['TL_BACKEND'] = 'tensorflow'

import numpy as np
import time

import tensorflow as tf
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import Linear, Dropout, BatchNorm1d
from tensorboardX import SummaryWriter


X_train, y_train, X_val, y_val, X_test, y_test = tlx.files.load_mnist_dataset(shape=(-1, 784))


class CustomModel(Module):

    def __init__(self):
        super(CustomModel, self).__init__()
        self.dropout1 = Dropout(p=0.2)
        self.linear1 = Linear(out_features=800, in_features=784)
        self.batchnorm = BatchNorm1d(act=tlx.ReLU, num_features=800)
        self.dropout2 = Dropout(p=0.2)
        self.linear2 = Linear(out_features=800, act=tlx.ReLU, in_features=800)
        self.dropout3 = Dropout(p=0.2)
        self.linear3 = Linear(out_features=10, act=tlx.ReLU, in_features=800)

    def forward(self, x, foo=None):
        z = self.dropout1(x)
        z = self.linear1(z)
        z = self.batchnorm(z)
        z = self.dropout2(z)
        z = self.linear2(z)
        z = self.dropout3(z)
        out = self.linear3(z)
        if foo is not None:
            out = tlx.relu(out)
        return out


MLP = CustomModel()
n_epoch = 50
batch_size = 500
print_freq = 1
train_weights = MLP.trainable_weights
optimizer = tlx.optimizers.Adam(learning_rate=0.0001)
train_batch = 0
test_batch = 0

writer = SummaryWriter('runs/mlp')

for epoch in range(n_epoch):  ## iterate the dataset n_epoch times
    start_time = time.time()
    ## iterate over the entire training set once (shuffle the data via training)
    for X_batch, y_batch in tlx.utils.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
        MLP.set_train()  # enable dropout
        with tf.GradientTape() as tape:
            ## compute outputs
            _logits = MLP(X_batch)
            ## compute loss and update model
            _loss = tlx.losses.softmax_cross_entropy_with_logits(_logits, y_batch)
        grad = tape.gradient(_loss, train_weights)
        optimizer.apply_gradients(zip(grad, train_weights))

    ## use training and evaluation sets to evaluate the model every print_freq epoch
    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
        train_loss, train_acc, n_iter = 0, 0, 0
        for X_batch, y_batch in tlx.utils.iterate.minibatches(X_train, y_train, batch_size, shuffle=False):
            train_batch += 1
            _logits = MLP(X_batch)
            train_loss += tlx.losses.softmax_cross_entropy_with_logits(_logits, y_batch)
            train_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            n_iter += 1

            print("   train loss: {}".format(train_loss / n_iter))
            print("   train acc:  {}".format(train_acc / n_iter))

            writer.add_scalar('train loss', tlx.ops.convert_to_numpy(train_loss / n_iter), train_batch)
            writer.add_scalar('train acc', train_acc / n_iter, train_batch)

        val_loss, val_acc, n_iter = 0, 0, 0
        for X_batch, y_batch in tlx.utils.iterate.minibatches(X_val, y_val, batch_size, shuffle=False):
            test_batch += 1
            _logits = MLP(X_batch)  # is_train=False, disable dropout
            val_loss += tlx.losses.softmax_cross_entropy_with_logits(_logits, y_batch)
            val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            n_iter += 1

            print("   val loss: {}".format(val_loss / n_iter))
            print("   val acc:  {}".format(val_acc / n_iter))

            writer.add_scalar('val loss', tlx.ops.convert_to_numpy(val_loss / n_iter), test_batch)
            writer.add_scalar('val acc', val_acc / n_iter, test_batch)

writer.export_scalars_to_json("./all_scalars.json")
writer.close()

