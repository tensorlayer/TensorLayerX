#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ['TL_BACKEND'] = 'paddle'
# os.environ['TL_BACKEND'] = 'tensorflow'
# os.environ['TL_BACKEND'] = 'mindspore'
# os.environ['TL_BACKEND'] = 'torch'

import time
from tensorlayerx.dataflow import Dataset, DataLoader
from tensorlayerx.vision.transforms import (
    Compose, Resize, RandomFlipHorizontal, RandomContrast, RandomBrightness, StandardizePerImage, RandomCrop
)
from tensorlayerx.model import TrainOneStep
from tensorlayerx.nn import Module
import tensorlayerx as tlx
from tensorlayerx.nn import (Conv2d, Linear, Flatten, MaxPool2d, BatchNorm2d)
# enable debug logging
tlx.logging.set_verbosity(tlx.logging.DEBUG)

# prepare cifar10 data
X_train, y_train, X_test, y_test = tlx.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)


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
            64, (5, 5), (1, 1), padding='SAME', act=tlx.nn.ReLU, W_init=W_init, b_init=None, name='conv2', in_channels=64
        )
        self.maxpool2 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool2')

        self.flatten = Flatten(name='flatten')
        self.linear1 = Linear(384, act=tlx.nn.ReLU, W_init=W_init2, b_init=b_init2, name='linear1relu', in_features=2304)
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
        z = self.linear2(z)
        z = self.linear3(z)
        return z


# get the network
net = CNN()

# training settings
batch_size = 128
n_epoch = 500
learning_rate = 0.0001
print_freq = 5
n_step_epoch = int(len(y_train) / batch_size)
n_step = n_epoch * n_step_epoch
shuffle_buffer_size = 128

train_weights = net.trainable_weights
optimizer = tlx.optimizers.Adam(learning_rate)
metrics = tlx.metrics.Accuracy()


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

loss_fn=tlx.losses.softmax_cross_entropy_with_logits

model = tlx.model.Model(network=net, loss_fn=loss_fn, optimizer=optimizer, metrics=metrics)
model.train(n_epoch=n_epoch, train_dataset=train_loader, test_dataset=test_loader, print_freq=print_freq, print_train_batch=True)
