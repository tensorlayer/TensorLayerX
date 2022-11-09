#! /usr/bin/python
# -*- coding: utf-8 -*-
import os
os.environ['TL_BACKEND'] = 'tensorflow'

import time
import numpy as np
import tensorflow as tf

from tensorlayerx.nn import Module, Sequential
import tensorlayerx as tlx
from tensorlayerx.nn import (Conv2d, Linear, Flatten, MaxPool2d, BatchNorm2d, Elementwise)
from tensorlayerx.dataflow import Dataset, DataLoader
from tensorlayerx.vision.transforms import (
    Compose, Resize, RandomFlipHorizontal, RandomContrast, RandomBrightness, StandardizePerImage, RandomCrop
)


X_train, y_train, X_test, y_test = tlx.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)


class Block(Module):

    def __init__(self, in_features):
        super(Block, self).__init__()
        self.linear1 = Linear(in_features=in_features, out_features=256)
        self.linear2 = Linear(in_features=256, out_features=384)
        self.linear3 = Linear(in_features=in_features, out_features=384)
        self.concat = Elementwise(combine_fn=tlx.ops.add)

    def forward(self, inputs):
        z = self.linear1(inputs)
        z1 = self.linear2(z)

        z2 = self.linear3(inputs)
        out = self.concat([z1, z2])
        return out


class CNN(Module):

    def __init__(self):
        super(CNN, self).__init__()
        # weights init
        W_init = tlx.nn.initializers.truncated_normal(stddev=5e-2)
        W_init2 = tlx.nn.initializers.truncated_normal(stddev=0.04)
        b_init2 = tlx.nn.initializers.constant(value=0.1)

        self.conv1 = Conv2d(64, (5, 5), (1, 1), padding='SAME', W_init=W_init, b_init=None, name='conv1', in_channels=3)
        self.bn = BatchNorm2d(num_features=64, act=tlx.nn.ReLU)
        self.maxpool1 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool1')

        self.conv2 = Conv2d(
            64, (5, 5), (1, 1), padding='SAME', act=tlx.nn.ReLU, W_init=W_init, name='conv2', in_channels=64
        )
        self.maxpool2 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool2')

        self.flatten = Flatten(name='flatten')
        self.linear1 = Linear(384, act=tlx.nn.ReLU, W_init=W_init2, b_init=b_init2, name='linear1relu', in_features=2304)
        self.linear_add = self.make_layer(in_channel=384)

        self.linear2 = Linear(192, act=tlx.nn.ReLU, W_init=W_init2, b_init=b_init2, name='linear2relu', in_features=384)
        self.linear3 = Linear(10, act=None, W_init=W_init2, name='output', in_features=192)

    def forward(self, x):
        z = self.conv1(x)
        z = self.bn(z)
        z = self.maxpool1(z)
        z = self.conv2(z)
        z = self.maxpool2(z)
        z = self.flatten(z)
        z = self.linear1(z)
        z = self.linear_add(z)

        z = self.linear2(z)
        z = self.linear3(z)
        return z

    def make_layer(self, in_channel):
        layers = []

        _block = Block(in_channel)
        layers.append(_block)

        for _ in range(1, 3):
            range_block = Block(in_channel)
            layers.append(range_block)

        return Sequential(layers)


# get the network
net = CNN()
print(net)
# training settings
batch_size = 128
n_epoch = 500
learning_rate = 0.001
print_freq = 5
n_step_epoch = int(len(y_train) / batch_size)
n_step = n_epoch * n_step_epoch
shuffle_buffer_size = 128

train_weights = net.trainable_weights
optimizer = tlx.optimizers.Adam(learning_rate)

class make_dataset(Dataset):

    def __init__(self, data, label, transforms):
        self.data = data
        self.label = label
        self.transforms = transforms

    def __getitem__(self, idx):
        x = self.data[idx].astype('uint8')
        y = self.label[idx].astype('int64')
        x = self.transforms(x)

        return x, y

    def __len__(self):

        return len(self.label)


train_transforms = Compose(
    [
        RandomCrop(size=[24, 24]),
        RandomFlipHorizontal(),
        RandomBrightness(brightness_factor=(0.5, 1.5)),
        RandomContrast(contrast_factor=(0.5, 1.5)),
        StandardizePerImage()
    ]
)

test_transforms = Compose([Resize(size=(24, 24)), StandardizePerImage()])

train_dataset = make_dataset(data=X_train, label=y_train, transforms=train_transforms)
test_dataset = make_dataset(data=X_test, label=y_test, transforms=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

for epoch in range(n_epoch):
    start_time = time.time()

    train_loss, train_acc, n_iter = 0, 0, 0
    for X_batch, y_batch in train_loader:
        net.set_train()

        with tf.GradientTape() as tape:
            # compute outputs
            _logits = net(X_batch)
            # compute loss and update model
            _loss_ce = tlx.losses.softmax_cross_entropy_with_logits(_logits, y_batch)

        grad = tape.gradient(_loss_ce, train_weights)
        optimizer.apply_gradients(zip(grad, train_weights))

        train_loss += _loss_ce
        train_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
        n_iter += 1

        print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
        print("   train loss: {}".format(train_loss / n_iter))
        print("   train acc:  {}".format(train_acc / n_iter))

    # use training and evaluation sets to evaluate the model every print_freq epoch
    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:

        net.set_eval()
        val_loss, val_acc, n_iter = 0, 0, 0
        for X_batch, y_batch in test_loader:
            _logits = net(X_batch)  # is_train=False, disable dropout
            val_loss += tlx.losses.softmax_cross_entropy_with_logits(_logits, y_batch)
            val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            n_iter += 1
        print("   val loss: {}".format(val_loss / n_iter))
        print("   val acc:  {}".format(val_acc / n_iter))

# use testing data to evaluate the model
net.set_eval()
test_loss, test_acc, n_iter = 0, 0, 0
for X_batch, y_batch in test_loader:
    _logits = net(X_batch)
    test_loss += tlx.losses.softmax_cross_entropy_with_logits(_logits, y_batch)
    test_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
    n_iter += 1
print("   test loss: {}".format(test_loss / n_iter))
print("   test acc:  {}".format(test_acc / n_iter))
