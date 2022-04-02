#! /usr/bin/python
# -*- coding: utf-8 -*-
import os
os.environ['TL_BACKEND'] = 'tensorflow'

import numpy as np
import time
import tensorflow as tf
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import Linear, Dropout, BatchNorm1d
from tensorlayerx.dataflow import Dataset, DataLoader

X_train, y_train, X_val, y_val, X_test, y_test = tlx.files.load_mnist_dataset(shape=(-1, 784))


class CustomModel(Module):

    def __init__(self):
        super(CustomModel, self).__init__()
        self.dropout1 = Dropout(p=0.2)
        self.linear1 = Linear(out_features=800)
        self.batchnorm = BatchNorm1d(act=tlx.ReLU)
        self.dropout2 = Dropout(p=0.2)
        self.linear2 = Linear(out_features=800, act=tlx.ReLU)
        self.dropout3 = Dropout(p=0.2)
        self.linear3 = Linear(out_features=10, act=tlx.ReLU)

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


class mnistdataset(Dataset):

    def __init__(self, data=X_train, label=y_train):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = self.data[index].astype('float32')
        label = self.label[index].astype('int64')
        return data, label

    def __len__(self):
        return len(self.data)

MLP = CustomModel()
# Automatic inference input of shape.
# If Layer has no input in_channels, init_build(input) must be called to initialize the weights.
MLP.init_build(tlx.nn.Input(shape=(1, 784)))

n_epoch = 50
batch_size = 500
print_freq = 5
train_weights = MLP.trainable_weights
optimizer = tlx.optimizers.Adam(learning_rate=0.0001)
train_dataset = mnistdataset(data=X_train, label=y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = mnistdataset(data=X_val, label=y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataset = mnistdataset(data=X_test, label=y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


for epoch in range(n_epoch):  ## iterate the dataset n_epoch times
    start_time = time.time()
    ## iterate over the entire training set once (shuffle the data via training)
    for X_batch, y_batch in train_loader:
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
        for X_batch, y_batch in train_loader:
            _logits = MLP(X_batch)
            train_loss += tlx.losses.softmax_cross_entropy_with_logits(_logits, y_batch)
            train_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            n_iter += 1
        print("   train loss: {}".format(train_loss / n_iter))
        print("   train acc:  {}".format(train_acc / n_iter))

        val_loss, val_acc, n_iter = 0, 0, 0
        for X_batch, y_batch in val_loader:
            _logits = MLP(X_batch)  # is_train=False, disable dropout
            val_loss += tlx.losses.softmax_cross_entropy_with_logits(_logits, y_batch)
            val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            n_iter += 1
        print("   val loss: {}".format(val_loss / n_iter))
        print("   val acc:  {}".format(val_acc / n_iter))

## use testing data to evaluate the model
MLP.set_eval()
test_loss, test_acc, n_iter = 0, 0, 0
for X_batch, y_batch in test_loader:
    _logits = MLP(X_batch, foo=1)
    test_loss += tlx.losses.softmax_cross_entropy_with_logits(_logits, y_batch)
    test_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
    n_iter += 1
print("   test foo=1 loss: {}".format(test_loss / n_iter))
print("   test foo=1 acc:  {}".format(test_acc / n_iter))
