#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
# os.environ['TL_BACKEND'] = 'tensorflow'
# os.environ['TL_BACKEND'] = 'mindspore'
# os.environ['TL_BACKEND'] = 'paddle'
# os.environ['TL_BACKEND'] = 'jittor'
os.environ['TL_BACKEND'] = 'torch'


import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import Linear, Flatten
from tensorlayerx.vision.transforms import Normalize, Compose
from tensorlayerx.dataflow import Dataset, IterableDataset

transform = Compose([Normalize(mean=[127.5], std=[127.5], data_format='HWC')])

print('download training data and load training data')

X_train, y_train, X_val, y_val, X_test, y_test = tlx.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
X_train = X_train * 255

print('load finished')


class mnistdataset(Dataset):

    def __init__(self, data=X_train, label=y_train, transform=transform):
        self.data = data
        self.label = label
        self.transform = transform

    def __getitem__(self, index):
        data = self.data[index].astype('float32')
        data = self.transform(data)
        label = self.label[index].astype('int64')

        return data, label

    def __len__(self):

        return len(self.data)


class mnistdataset1(IterableDataset):

    def __init__(self, data=X_train, label=y_train, transform=transform):
        self.data = data
        self.label = label
        self.transform = transform

    def __iter__(self):

        for i in range(len(self.data)):
            data = self.data[i].astype('float32')
            data = self.transform(data)
            label = self.label[i].astype('int64')
            yield data, label


class MLP(Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = Linear(out_features=120, in_features=784, act=tlx.nn.ReLU)
        self.linear2 = Linear(out_features=84, in_features=120, act=tlx.nn.ReLU)
        self.linear3 = Linear(out_features=10, in_features=84)
        self.flatten = Flatten()

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


train_dataset = mnistdataset1(data=X_train, label=y_train, transform=transform)
train_loader = tlx.dataflow.DataLoader(train_dataset, batch_size=128, shuffle=False)

for i in train_loader:
    print(i[0].shape, i[1])
