#! /usr/bin/python
# -*- coding: utf-8 -*-

# The same set of code can switch the backend with one line
import os
# os.environ['TL_BACKEND'] = 'tensorflow'
# os.environ['TL_BACKEND'] = 'mindspore'
# os.environ['TL_BACKEND'] = 'paddle'
os.environ['TL_BACKEND'] = 'torch'

import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import Linear, Dropout
from tensorlayerx.dataflow import Dataset, DataLoader

X_train, y_train, X_val, y_val, X_test, y_test = tlx.files.load_mnist_dataset(shape=(-1, 784))


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


class CustomModel(Module):

    def __init__(self):
        super(CustomModel, self).__init__()
        self.dropout1 = Dropout(p=0.2)
        self.linear1 = Linear(out_features=800, act=tlx.ReLU, in_features=784)
        self.dropout2 = Dropout(p=0.2)
        self.linear2 = Linear(out_features=800, act=tlx.ReLU, in_features=800)
        self.dropout3 = Dropout(p=0.2)
        self.linear3 = Linear(out_features=10, act=tlx.ReLU, in_features=800)

    def forward(self, x, foo=None):
        z = self.dropout1(x)
        z = self.linear1(z)
        z = self.dropout2(z)
        z = self.linear2(z)
        z = self.dropout3(z)
        out = self.linear3(z)
        if foo is not None:
            out = tlx.relu(out)
        return out


MLP = CustomModel()
n_epoch = 50
batch_size = 128
print_freq = 2

train_weights = MLP.trainable_weights
optimizer = tlx.optimizers.Momentum(0.05, 0.9)
metric = tlx.metrics.Accuracy()
loss_fn = tlx.losses.softmax_cross_entropy_with_logits
train_dataset = mnistdataset(data=X_train, label=y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = tlx.model.Model(network=MLP, loss_fn=loss_fn, optimizer=optimizer, metrics=metric)
model.train(n_epoch=n_epoch, train_dataset=train_loader, print_freq=print_freq, print_train_batch=False)
model.save_weights('./model.npz', format='npz_dict')
model.load_weights('./model.npz', format='npz_dict')

################################ TensorLayerX and TensorFlow can be mixed programming. #################################
# import os
# os.environ['TL_BACKEND'] = 'tensorflow'
#
# import numpy as np
# import time
#
# import tensorflow as tf
# import tensorlayerx as tlx
# from tensorlayerx.nn import Module
# from tensorlayerx.nn import Linear, Dropout, BatchNorm1d
#
# X_train, y_train, X_val, y_val, X_test, y_test = tlx.files.load_mnist_dataset(shape=(-1, 784))
#
#
# class CustomModel(Module):
#
#     def __init__(self):
#         super(CustomModel, self).__init__()
#         self.dropout1 = Dropout(p=0.2)
#         self.linear1 = Linear(out_features=800, in_features=784)
#         self.batchnorm = BatchNorm1d(act=tlx.ReLU, num_features=800)
#         self.dropout2 = Dropout(p=0.2)
#         self.linear2 = Linear(out_features=800, act=tlx.ReLU, in_features=800)
#         self.dropout3 = Dropout(p=0.2)
#         self.linear3 = Linear(out_features=10, act=tlx.ReLU, in_features=800)
#
#     def forward(self, x, foo=None):
#         z = self.dropout1(x)
#         z = self.linear1(z)
#         z = self.batchnorm(z)
#         z = self.dropout2(z)
#         z = self.linear2(z)
#         z = self.dropout3(z)
#         out = self.linear3(z)
#         if foo is not None:
#             out = tlx.relu(out)
#         return out
#
#
# MLP = CustomModel()
# n_epoch = 50
# batch_size = 500
# print_freq = 5
# train_weights = MLP.trainable_weights
# optimizer = tlx.optimizers.Adam(lr=0.0001)
#
# for epoch in range(n_epoch):  ## iterate the dataset n_epoch times
#     start_time = time.time()
#     ## iterate over the entire training set once (shuffle the data via training)
#     for X_batch, y_batch in tlx.utils.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
#         MLP.set_train()  # enable dropout
#         with tf.GradientTape() as tape:
#             ## compute outputs
#             _logits = MLP(X_batch)
#             ## compute loss and update model
#             _loss = tlx.losses.softmax_cross_entropy_with_logits(_logits, y_batch)
#         grad = tape.gradient(_loss, train_weights)
#         optimizer.apply_gradients(zip(grad, train_weights))
#
#     ## use training and evaluation sets to evaluate the model every print_freq epoch
#     if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
#         print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
#         train_loss, train_acc, n_iter = 0, 0, 0
#         for X_batch, y_batch in tlx.utils.iterate.minibatches(X_train, y_train, batch_size, shuffle=False):
#             _logits = MLP(X_batch)
#             train_loss += tlx.losses.softmax_cross_entropy_with_logits(_logits, y_batch)
#             train_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
#             n_iter += 1
#         print("   train loss: {}".format(train_loss / n_iter))
#         print("   train acc:  {}".format(train_acc / n_iter))
#
#         val_loss, val_acc, n_iter = 0, 0, 0
#         for X_batch, y_batch in tlx.utils.iterate.minibatches(X_val, y_val, batch_size, shuffle=False):
#             _logits = MLP(X_batch)  # is_train=False, disable dropout
#             val_loss += tlx.losses.softmax_cross_entropy_with_logits(_logits, y_batch)
#             val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
#             n_iter += 1
#         print("   val loss: {}".format(val_loss / n_iter))
#         print("   val acc:  {}".format(val_acc / n_iter))
#
# ## use testing data to evaluate the model
# MLP.set_eval()
# test_loss, test_acc, n_iter = 0, 0, 0
# for X_batch, y_batch in tlx.utils.iterate.minibatches(X_test, y_test, batch_size, shuffle=False):
#     _logits = MLP(X_batch, foo=1)
#     test_loss += tlx.losses.softmax_cross_entropy_with_logits(_logits, y_batch)
#     test_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
#     n_iter += 1
# print("   test foo=1 loss: {}".format(test_loss / n_iter))
# print("   test foo=1 acc:  {}".format(test_acc / n_iter))


################################ TensorLayerX and MindSpore can be mixed programming. #################################
# import os
# os.environ['TL_BACKEND'] = 'mindspore'
#
# import mindspore.ops.operations as P
# from mindspore.ops import composite as C
# from mindspore import ParameterTuple
# from mindspore.nn import Momentum, WithLossCell
#
# import numpy as np
# import tensorlayerx as tlx
# import mindspore as ms
# import tensorflow as tf
# import time
# from tensorlayerx.nn import Module
# from tensorlayerx.nn import Linear
# import mindspore.nn as nn
#
#
# class MLP(Module):
#
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.linear1 = Linear(out_features=800, act=tlx.ReLU, in_features=784)
#         self.linear2 = Linear(out_features=800, act=tlx.ReLU, in_features=800)
#         self.linear3 = Linear(out_features=10, act=tlx.ReLU, in_features=800)
#
#     def forward(self, x):
#         z = self.linear1(x)
#         z = self.linear2(z)
#         out = self.linear3(z)
#         return out
#
#
# class GradWrap(Module):
#     """ GradWrap definition """
#
#     def __init__(self, network):
#         super(GradWrap, self).__init__(auto_prefix=False)
#         self.network = network
#         self.weights = ParameterTuple(filter(lambda x: x.requires_grad, network.get_parameters()))
#
#     def forward(self, x, label):
#         return C.GradOperation(get_by_list=True)(self.network, self.weights)(x, label)
#
#
# def generator_train():
#     inputs = X_train
#     targets = y_train
#     if len(inputs) != len(targets):
#         raise AssertionError("The length of inputs and targets should be equal")
#     for _input, _target in zip(inputs, targets):
#         yield _input, _target
#
#
# net = MLP()
# train_weights = list(filter(lambda x: x.requires_grad, net.get_parameters()))
# optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.15, 0.8)
#
# criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
# net_with_criterion = WithLossCell(net, criterion)
# train_network = GradWrap(net_with_criterion)
# train_network.set_train()
#
# X_train, y_train, X_val, y_val, X_test, y_test = tlx.files.load_mnist_dataset(shape=(-1, 784))
# train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32, tf.int32))
# shuffle_buffer_size = 128
# batch_size = 128
# train_ds = train_ds.shuffle(shuffle_buffer_size)
# train_ds = train_ds.batch(batch_size)
# n_epoch = 50
#
# for epoch in range(n_epoch):
#     start_time = time.time()
#     train_network.set_train()
#     train_loss, train_acc, n_iter = 0, 0, 0
#     for X_batch, y_batch in train_ds:
#         X_batch = ms.Tensor(X_batch.numpy(), dtype=ms.float32)
#         y_batch = ms.Tensor(y_batch.numpy(), dtype=ms.int32)
#         output = net(X_batch)
#         loss_output = criterion(output, y_batch)
#         grads = train_network(X_batch, y_batch)
#         success = optimizer(grads)
#         loss = loss_output.asnumpy()
#         train_loss += loss
#         n_iter += 1
#         train_acc += np.mean((P.Equal()(P.Argmax(axis=1)(output), y_batch).asnumpy()))
#         print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
#         print("   train loss: {}".format(train_loss / n_iter))
#         print("   train acc:  {}".format(train_acc / n_iter))
#         print(" loss ", loss)


################################## TensorLayerX and Paddle can be mixed programming. ##################################
# import os
# os.environ['TL_BACKEND'] = 'paddle'
#
# import tensorlayerx as tlx
# from tensorlayerx.nn import Module
# from tensorlayerx.nn import Linear, Flatten
# import paddle
# from paddle.io import TensorDataset
#
# print('download training data and load training data')
#
# X_train, y_train, X_val, y_val, X_test, y_test = tlx.files.load_mnist_dataset(shape=(-1, 784))
#
# print('load finished')
# X_train = paddle.to_tensor(X_train.astype('float32'))
# y_train = paddle.to_tensor(y_train.astype('int64'))
#
#
# class MLP(Module):
#
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.linear1 = Linear(out_features=120, in_features=784, act=tlx.ReLU)
#         self.linear2 = Linear(out_features=84, in_features=120, act=tlx.ReLU)
#         self.linear3 = Linear(out_features=10, in_features=84)
#         self.flatten = Flatten()
#
#     def forward(self, x):
#         x = self.flatten(x)
#         x = self.linear1(x)
#         x = self.linear2(x)
#         x = self.linear3(x)
#         return x
#
#
# traindataset = paddle.io.TensorDataset([X_train, y_train])
# train_loader = paddle.io.DataLoader(traindataset, batch_size=64, shuffle=True)
# net = MLP()
#
# optimizer = tlx.optimizers.Adam(learning_rate=0.001)
# metric = tlx.metrics.Accuracy()
# model = tlx.model.Model(
#     network=net, loss_fn=tlx.losses.softmax_cross_entropy_with_logits, optimizer=optimizer, metrics=metric
# )
# model.train(n_epoch=2, train_dataset=train_loader, print_freq=5, print_train_batch=True)
# model.save_weights('./model_mlp.npz', format='npz_dict')
# model.load_weights('./model_mlp.npz', format='npz_dict')
# # model.eval(train_loader)

################################## TensorLayerX and Torch can be mixed programming. ##################################
# import os
# os.environ['TL_BACKEND'] = 'torch'
#
# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.transforms import ToTensor
#
# from tensorlayerx.nn import Module, Linear
# import tensorlayerx as tlx
#
# # Download training data from open datasets.
# training_data = datasets.FashionMNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor(),
# )
#
# # Download test data from open datasets.
# test_data = datasets.FashionMNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor(),
# )
#
# batch_size = 64
#
# # Create data loaders.
# train_dataloader = DataLoader(training_data, batch_size=batch_size)
# test_dataloader = DataLoader(test_data, batch_size=batch_size)
#
# for X, y in test_dataloader:
#     print("Shape of X [N, C, H, W]: ", X.shape)
#     print("Shape of y: ", y.shape, y.dtype)
#     break
#
# # Get cpu or gpu device for training.
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using {} device".format(device))
#
# # Define model
# # class NeuralNetwork(nn.Module):
# #     def __init__(self):
# #         super(NeuralNetwork, self).__init__()
# #         self.flatten = nn.Flatten()
# #         self.linear_relu_stack = nn.Sequential(
# #             nn.Linear(28*28, 512),
# #             nn.ReLU(),
# #             nn.Linear(512, 512),
# #             nn.ReLU(),
# #             nn.Linear(512, 10)
# #         )
# #
# #     def forward(self, x):
# #         x = self.flatten(x)
# #         logits = self.linear_relu_stack(x)
# #         return logits
#
#
# class NeuralNetwork(Module):
#
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear1 = Linear(in_features=28 * 28, out_features=512)
#         self.linear2 = Linear(in_features=512, out_features=512)
#         self.linear3 = Linear(in_features=512, out_features=10)
#
#     def forward(self, x):
#         x = self.flatten(x)
#         x = self.linear1(x)
#         x = self.linear2(x)
#         x = self.linear3(x)
#         return x
#
#
# model = NeuralNetwork().to(device)
#
# # loss_fn = nn.CrossEntropyLoss()
# loss_fn = tlx.losses.softmax_cross_entropy_with_logits
#
# # optimizer = torch.optim.SGD(model.trainable_weights, lr=1e-3)
# optimizer = tlx.optimizers.SGD(learning_rate=1e-3)
#
#
# def train(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     model.train()
#     for batch, (X, y) in enumerate(dataloader):
#         X, y = X.to(device), y.to(device)
#
#         # Compute prediction error
#         pred = model(X)
#         loss = loss_fn(pred, y)
#
#         # Backpropagation
#         # optimizer.zero_grad()
#         # loss.backward()
#         # optimizer.step()
#         grads = optimizer.gradient(loss, model.trainable_weights)
#         optimizer.apply_gradients(zip(grads, model.trainable_weights))
#
#         if batch % 100 == 0:
#             loss, current = loss.item(), batch * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
#
#
# def test(dataloader, model, loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.eval()
#     test_loss, correct = 0, 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
#
#
# epochs = 5
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(train_dataloader, model, loss_fn, optimizer)
#     test(test_dataloader, model, loss_fn)
# print("Done!")
