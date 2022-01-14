#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ['TL_BACKEND'] = 'mindspore'

import time
import numpy as np
from tensorlayerx.nn import Module
import tensorlayerx as tlx
from tensorlayerx.nn import (Conv2d, Dense, Flatten, MaxPool2d, BatchNorm2d)
from tensorlayerx.vision.transforms import (
    Compose, Resize, RandomFlipHorizontal, RandomContrast, RandomBrightness, StandardizePerImage, RandomCrop, HWC2CHW
)
from tensorlayerx.dataflow import Dataset, Dataloader
from mindspore.nn import WithLossCell, Adam
from mindspore import ParameterTuple
import mindspore.nn as nn
from mindspore.ops import composite as C
import mindspore.ops.operations as P

# enable debug logging
tlx.logging.set_verbosity(tlx.logging.DEBUG)


class CNN(Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = Conv2d(
            64, (5, 5), (2, 2), b_init=None, name='conv1', in_channels=3, act=tlx.ReLU, data_format='channels_first'
        )
        self.bn = BatchNorm2d(num_features=64, act=tlx.ReLU, data_format='channels_first')
        self.maxpool1 = MaxPool2d((3, 3), (2, 2), name='pool1', data_format='channels_first')
        self.conv2 = Conv2d(
            128, (5, 5), (2, 2), act=tlx.ReLU, b_init=None, name='conv2', in_channels=64, data_format='channels_first'
        )
        self.maxpool2 = MaxPool2d((3, 3), (2, 2), name='pool2', data_format='channels_first')

        self.flatten = Flatten(name='flatten')
        self.dense1 = Dense(120, act=tlx.ReLU, name='dense1relu', in_channels=512)
        self.dense2 = Dense(84, act=tlx.ReLU, name='dense2relu', in_channels=120)
        self.dense3 = Dense(10, act=None, name='output', in_channels=84)

    def forward(self, x):
        z = self.conv1(x)
        z = self.bn(z)
        z = self.maxpool1(z)
        z = self.conv2(z)
        z = self.maxpool2(z)
        z = self.flatten(z)
        z = self.dense1(z)
        z = self.dense2(z)
        z = self.dense3(z)
        return z


# training settings
batch_size = 128
n_epoch = 500
shuffle_buffer_size = 128

# prepare cifar10 data
X_train, y_train, X_test, y_test = tlx.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)


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
        StandardizePerImage(),
        HWC2CHW()
    ]
)

test_transforms = Compose([Resize(size=(24, 24)), StandardizePerImage(), HWC2CHW()])

train_dataset = make_dataset(data=X_train, label=y_train, transforms=train_transforms)
test_dataset = make_dataset(data=X_test, label=y_test, transforms=test_transforms)

train_dataset = tlx.dataflow.FromGenerator(
    train_dataset, output_types=(tlx.float32, tlx.int64), column_names=['data', 'label']
)
test_dataset = tlx.dataflow.FromGenerator(
    test_dataset, output_types=(tlx.float32, tlx.int64), column_names=['data', 'label']
)

train_dataset = Dataloader(train_dataset, batch_size=batch_size, shuffle=True, shuffle_buffer_size=128)
test_dataset = Dataloader(test_dataset, batch_size=batch_size)


class GradWrap(Module):
    """ GradWrap definition """

    def __init__(self, network):
        super(GradWrap, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(filter(lambda x: x.requires_grad, network.get_parameters()))

    def forward(self, x, label):
        return C.GradOperation(get_by_list=True)(self.network, self.weights)(x, label)


# get the network
net = CNN()
train_weights = net.trainable_weights
optimizer = Adam(train_weights, learning_rate=0.01)
# optimizer = Momentum(train_weights, 0.01, 0.5)
criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
net_with_criterion = WithLossCell(net, criterion)
train_network = GradWrap(net_with_criterion)
train_network.set_train()
# print(train_weights)
for epoch in range(n_epoch):
    start_time = time.time()
    train_network.set_train()
    train_loss, train_acc, n_iter = 0, 0, 0
    for X_batch, y_batch in train_dataset:
        output = net(X_batch)
        loss_output = criterion(output, y_batch)
        grads = train_network(X_batch, y_batch)
        success = optimizer(grads)
        loss = loss_output.asnumpy()
        train_loss += loss
        n_iter += 1
        train_acc += np.mean((P.Equal()(P.Argmax(axis=1)(output), y_batch).asnumpy()))
        print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
        print("   train loss: {}".format(train_loss / n_iter))
        print("   train acc:  {}".format(train_acc / n_iter))
        print(" loss ", loss)
