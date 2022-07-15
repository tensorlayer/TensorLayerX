#! /usr/bin/python
# -*- coding: utf-8 -*-

# The same set of code can switch the backend with one line
import os
# os.environ['TL_BACKEND'] = 'tensorflow'
# os.environ['TL_BACKEND'] = 'mindspore'
os.environ['TL_BACKEND'] = 'paddle'
# os.environ['TL_BACKEND'] = 'torch'
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import Linear, LSTM, Embedding, RNN
from tensorlayerx.dataflow import Dataset
import numpy as np
prev_h = np.random.random([1, 200, 64]).astype(np.float32)
prev_h = tlx.convert_to_tensor(prev_h)

X_train, y_train, X_test, y_test = tlx.files.load_imdb_dataset('data', nb_words=20000, test_split=0.2)

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
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=64)
        self.lstm = LSTM(input_size=64, hidden_size=64)
        self.linear1 = Linear(in_features=64, out_features=64, act=tlx.nn.ReLU)
        self.linear2 = Linear(in_features=64, out_features=2)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x, [prev_h, prev_h])
        x = tlx.reduce_mean(x, axis=1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


n_epoch = 5
batch_size = 64
print_freq = 2

train_dataset = imdbdataset(X=X_train, y=y_train)
train_loader = tlx.dataflow.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

net = ImdbNet()
train_weights = net.trainable_weights
optimizer = tlx.optimizers.Adam(1e-3)
metric = tlx.metrics.Accuracy()
loss_fn = tlx.losses.softmax_cross_entropy_with_logits
model = tlx.model.Model(network=net, loss_fn=loss_fn, optimizer=optimizer, metrics=metric)
model.train(n_epoch=n_epoch, train_dataset=train_loader, print_freq=print_freq, print_train_batch=True)
