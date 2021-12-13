#! /usr/bin/python
# -*- coding: utf-8 -*-

# The same set of code can switch the backend with one line
import os
os.environ['TL_BACKEND'] = 'tensorflow'
# os.environ['TL_BACKEND'] = 'mindspore'
# os.environ['TL_BACKEND'] = 'paddle'

import tensorlayer as tl
from tensorlayer.layers import Module
from tensorlayer.layers import Dense, LSTM, Embedding
from tensorlayer.dataflow import Dataset
import numpy as np

X_train, y_train, X_test, y_test = tl.files.load_imdb_dataset('data', nb_words=20000, test_split=0.2)

Seq_Len = 200
vocab_size = len(X_train) + 1


class imdbdataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):

        data = self.X[index]
        data = np.concatenate([data[:Seq_Len], [0] * (Seq_Len - len(data))]).astype('int64')  # set
        label = self.y[index].astype('int64')
        return data, label

    def __len__(self):

        return len(self.y)


class ImdbNet(Module):

    def __init__(self):
        super(ImdbNet, self).__init__()
        self.embedding = Embedding(vocabulary_size=vocab_size, embedding_size=64)
        self.lstm = LSTM(input_size=64, hidden_size=64)
        self.dense1 = Dense(in_channels=64, n_units=64, act=tl.ReLU)
        self.dense2 = Dense(in_channels=64, n_units=2)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = tl.ops.reduce_mean(x, axis=1)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


n_epoch = 5
batch_size = 64
print_freq = 2

train_dataset = imdbdataset(X=X_train, y=y_train)
train_dataset = tl.dataflow.FromGenerator(
    train_dataset, output_types=[tl.int64, tl.int64], column_names=['data', 'label']
)
train_loader = tl.dataflow.Dataloader(train_dataset, batch_size=batch_size, shuffle=True)

net = ImdbNet()
train_weights = net.trainable_weights
optimizer = tl.optimizers.Adam(1e-3)
metric = tl.metric.Accuracy()
loss_fn = tl.cost.softmax_cross_entropy_with_logits
model = tl.models.Model(network=net, loss_fn=loss_fn, optimizer=optimizer, metrics=metric)
model.train(n_epoch=n_epoch, train_dataset=train_loader, print_freq=print_freq, print_train_batch=True)
