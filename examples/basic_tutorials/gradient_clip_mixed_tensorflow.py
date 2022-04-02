#! /usr/bin/python
# -*- coding: utf-8 -*-
# The tensorlayerx and tensorflow operators can be mixed
import os
os.environ['TL_BACKEND'] = 'tensorflow'
# os.environ['TL_BACKEND'] = 'paddle'
# os.environ['TL_BACKEND'] = 'torch'


import time

import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import Linear
from tensorlayerx.dataflow import Dataset, DataLoader

X_train, y_train, X_val, y_val, X_test, y_test = tlx.files.load_mnist_dataset(shape=(-1, 784))


class CustomModel(Module):

    def __init__(self):
        super(CustomModel, self).__init__()
        self.linear1 = Linear(out_features=800, in_features=784)
        self.linear2 = Linear(out_features=800, act=tlx.ReLU, in_features=800)
        self.linear3 = Linear(out_features=10, act=tlx.ReLU, in_features=800)

    def forward(self, x, foo=None):
        z = self.linear1(x)
        z = self.linear2(z)
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
n_epoch = 50
batch_size = 500
print_freq = 5
train_weights = MLP.trainable_weights
train_dataset = mnistdataset(data=X_train, label=y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
optimizer = tlx.optimizers.Adam(learning_rate=0.0001, weight_decay= 0.001, grad_clip=tlx.ops.ClipGradByValue())

net_with_loss = tlx.model.WithLoss(backbone=MLP, loss_fn=tlx.losses.softmax_cross_entropy_with_logits)
net_with_grad_train = tlx.model.TrainOneStep(net_with_loss, optimizer, train_weights)
metrics = tlx.metrics.Accuracy()

for epoch in range(n_epoch):  ## iterate the dataset n_epoch times
    start_time = time.time()
    ## iterate over the entire training set once (shuffle the data via training)
    for X_batch, y_batch in train_loader:
        MLP.set_train()  # enable dropout
        loss = net_with_grad_train(X_batch, y_batch)

    # use training and evaluation sets to evaluate the model every print_freq epoch
    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
        train_loss, train_acc, n_iter = 0, 0, 0
        for X_batch, y_batch in train_loader:
            _logits = MLP(X_batch)
            train_loss += tlx.losses.softmax_cross_entropy_with_logits(_logits, y_batch)
            metrics.update(_logits, y_batch)
            train_acc += metrics.result()
            metrics.reset()
            n_iter += 1
        print("   train loss: {}".format(train_loss / n_iter))
        print("   train acc:  {}".format(train_acc / n_iter))
