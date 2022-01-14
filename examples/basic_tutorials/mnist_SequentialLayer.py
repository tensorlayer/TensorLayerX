#! /usr/bin/python
# -*- coding: utf-8 -*-
import os
os.environ['TL_BACKEND'] = 'tensorflow'
# os.environ['TL_BACKEND'] = 'mindspore'
# os.environ['TL_BACKEND'] = 'paddle'

from tensorlayerx.nn import SequentialLayer
from tensorlayerx.nn import Dense
import tensorlayerx as tlx
from tensorlayerx.dataflow import Dataset

layer_list = []
layer_list.append(Dense(n_units=800, act=tlx.ReLU, in_channels=784, name='Dense1'))
layer_list.append(Dense(n_units=800, act=tlx.ReLU, in_channels=800, name='Dense2'))
layer_list.append(Dense(n_units=10, act=tlx.ReLU, in_channels=800, name='Dense3'))
MLP = SequentialLayer(layer_list)

X_train, y_train, X_val, y_val, X_test, y_test = tlx.files.load_mnist_dataset(shape=(-1, 784))

# def generator_train():
#     inputs = X_train
#     targets = y_train
#     if len(inputs) != len(targets):
#         raise AssertionError("The length of inputs and targets should be equal")
#     for _input, _target in zip(inputs, targets):
#         yield (_input, np.array(_target))


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


n_epoch = 50
batch_size = 128
print_freq = 2
shuffle_buffer_size = 128

train_weights = MLP.trainable_weights
print(train_weights)
optimizer = tlx.optimizers.Momentum(0.05, 0.9)
train_dataset = mnistdataset(data=X_train, label=y_train)
train_dataset = tlx.dataflow.FromGenerator(
    train_dataset, output_types=[tlx.float32, tlx.int64], column_names=['data', 'label']
)
train_loader = tlx.dataflow.Dataloader(train_dataset, batch_size=batch_size, shuffle=True)
metric = tlx.metrics.Accuracy()
model = tlx.model.Model(
    network=MLP, loss_fn=tlx.losses.softmax_cross_entropy_with_logits, optimizer=optimizer, metrics=metric
)
model.train(n_epoch=n_epoch, train_dataset=train_loader, print_freq=print_freq, print_train_batch=False)
model.save_weights('./model.npz', format='npz_dict')
model.load_weights('./model.npz', format='npz_dict')
