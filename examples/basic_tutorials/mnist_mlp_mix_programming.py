################################ TensorLayerX and TensorFlow can be mixed programming. #################################
import os
os.environ['TL_BACKEND'] = 'tensorflow'

import numpy as np
import time

import tensorflow as tf
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import Linear, Dropout

# Load MNIST data by TensorLayerX
X_train, y_train, X_val, y_val, X_test, y_test = tlx.files.load_mnist_dataset(shape=(-1, 784))

def generator_train():
    inputs = X_train
    targets = y_train
    if len(inputs) != len(targets):
        raise AssertionError("The length of inputs and targets should be equal")
    for _input, _target in zip(inputs, targets):
        yield _input, _target

# Make Dataset by TensorFlow
train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32, tf.int32))
shuffle_buffer_size = 128
batch_size = 128
train_ds = train_ds.shuffle(shuffle_buffer_size)
train_ds = train_ds.batch(batch_size)


# Define the network through tensorlayerx
class CustomModel(Module):

    def __init__(self):
        super(CustomModel, self).__init__()
        self.dropout1 = Dropout(p=0.2)
        self.linear1 = Linear(out_features=800, in_features=784)
        self.dropout2 = Dropout(p=0.2)
        self.linear2 = Linear(out_features=800, act=tlx.nn.ReLU, in_features=800)
        self.dropout3 = Dropout(p=0.2)
        self.linear3 = Linear(out_features=10, act=tlx.nn.ReLU, in_features=800)

    def forward(self, x):
        z = self.dropout1(x)
        z = self.linear1(z)
        z = self.dropout2(z)
        z = self.linear2(z)
        z = self.dropout3(z)
        out = self.linear3(z)
        return out


MLP = CustomModel()
n_epoch = 50
batch_size = 500
print_freq = 1
train_weights = MLP.trainable_weights
# Define the optimizer through tensorlayerx
optimizer = tlx.optimizers.Adam(lr=0.0001)

for epoch in range(n_epoch):  ## iterate the dataset n_epoch times
    start_time = time.time()
    ## iterate over the entire training set once (shuffle the data via training)
    for X_batch, y_batch in train_ds :
        MLP.set_train()  # enable dropout
        with tf.GradientTape() as tape: # use tf.GradientTape() to record gradient
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
        for X_batch, y_batch in train_ds :
            _logits = MLP(X_batch)
            train_loss += tlx.losses.softmax_cross_entropy_with_logits(_logits, y_batch)
            train_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            n_iter += 1
        print("   train loss: {}".format(train_loss / n_iter))
        print("   train acc:  {}".format(train_acc / n_iter))

        val_loss, val_acc, n_iter = 0, 0, 0
        for X_batch, y_batch in train_ds:
            _logits = MLP(X_batch)  # is_train=False, disable dropout
            val_loss += tlx.losses.softmax_cross_entropy_with_logits(_logits, y_batch)
            val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            n_iter += 1
        print("   val loss: {}".format(val_loss / n_iter))
        print("   val acc:  {}".format(val_acc / n_iter))

################################ TensorLayerX and MindSpore can be mixed programming. #################################
# import os
# os.environ['TL_BACKEND'] = 'mindspore'
#
# import time
# import numpy as np
# import tensorlayerx as tlx
# import mindspore as ms
# import mindspore.ops.operations as P
# from mindspore.ops import composite as C
# from mindspore import ParameterTuple
# import mindspore.nn as nn
# from mindspore.nn import Momentum, WithLossCell
# from tensorlayerx.nn import Module
# from tensorlayerx.nn import Linear, Dropout
# from tensorlayerx.dataflow import Dataset, DataLoader
#
# # Load MNIST data by TensorLayerX
# X_train, y_train, X_val, y_val, X_test, y_test = tlx.files.load_mnist_dataset(shape=(-1, 784))
#
# # Make Dataset by TensorLayerX
# class mnistdataset(Dataset):
#
#     def __init__(self, data=X_train, label=y_train):
#         self.data = data
#         self.label = label
#
#     def __getitem__(self, index):
#         data = self.data[index].astype('float32')
#         label = self.label[index].astype('int64')
#         return data, label
#
#     def __len__(self):
#         return len(self.data)
#
# train_dataset = mnistdataset(data=X_train, label=y_train)
# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
#
# # Define the network through TensorLayerX
# class MLP(Module):
#
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.dropout1 = Dropout(p=0.2)
#         self.linear1 = Linear(out_features=800, act=tlx.nn.ReLU, in_features=784)
#         self.dropout2 = Dropout(p=0.2)
#         self.linear2 = Linear(out_features=800, act=tlx.nn.ReLU, in_features=800)
#         self.dropout3 = Dropout(p=0.2)
#         self.linear3 = Linear(out_features=10, act=tlx.nn.ReLU, in_features=800)
#
#     def forward(self, x):
#         z = self.dropout1(x)
#         z = self.linear1(z)
#         z = self.dropout2(z)
#         z = self.linear2(z)
#         z = self.dropout3(z)
#         out = self.linear3(z)
#         return out
#
# # Gradient calculation through MindSpore
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
# net = MLP()
# train_weights = list(filter(lambda x: x.requires_grad, net.get_parameters()))
# # Define the optimizer and loss funciton through MindSpore
# optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.15, 0.8)
# criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
# net_with_criterion = WithLossCell(net, criterion)
# train_network = GradWrap(net_with_criterion)
# train_network.set_train()
#
# n_epoch = 50
# # We use tlx's Model and Dataset and MindSpore's optimizer, loss function to train the network
# for epoch in range(n_epoch):
#     start_time = time.time()
#     train_network.set_train()
#     train_loss, train_acc, n_iter = 0, 0, 0
#     for X_batch, y_batch in train_loader:
#         X_batch = X_batch
#         y_batch = y_batch
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


################################## TensorLayerX and Paddle can be mixed programming. ##################################
# import os
# os.environ['TL_BACKEND'] = 'paddle'
#
# import time
# import paddle
# import tensorlayerx as tlx
# from tensorlayerx.nn import Module
# from tensorlayerx.nn import Linear, Dropout
#
# # Load MNIST data by TensorLayerX
# X_train, y_train, X_val, y_val, X_test, y_test = tlx.files.load_mnist_dataset(shape=(-1, 784))
#
# print('load finished')
# # Make Dataset by Paddle
# X_train = paddle.to_tensor(X_train.astype('float32'))
# y_train = paddle.to_tensor(y_train.astype('int64'))
# traindataset = paddle.io.TensorDataset([X_train, y_train])
# train_loader = paddle.io.DataLoader(traindataset, batch_size=64, shuffle=True)
#
# # Define the network through TensorLayerX
# class MLP(Module):
#
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.dropout1 = Dropout(p=0.2)
#         self.linear1 = Linear(out_features=800, act=tlx.nn.ReLU, in_features=784)
#         self.dropout2 = Dropout(p=0.2)
#         self.linear2 = Linear(out_features=800, act=tlx.nn.ReLU, in_features=800)
#         self.dropout3 = Dropout(p=0.2)
#         self.linear3 = Linear(out_features=10, act=tlx.nn.ReLU, in_features=800)
#
#     def forward(self, x):
#         z = self.dropout1(x)
#         z = self.linear1(z)
#         z = self.dropout2(z)
#         z = self.linear2(z)
#         z = self.dropout3(z)
#         out = self.linear3(z)
#         return out
#
# net = MLP()
# # Define the optimizer through Paddle
# optimizer = paddle.optimizer.Adam(0.001, parameters=net.trainable_weights)
#
#
# # We use tlx's Model, loss function and Paddle's optimizer, Dataset to train the network
# n_epoch = 50
# for epoch in range(n_epoch):
#     n_iter = 0
#     start_time = time.time()
#     for X_batch, y_batch in train_loader:
#         predict = net(X_batch)
#         loss = tlx.losses.softmax_cross_entropy_with_logits(predict, y_batch)
#         loss.backward()
#         optimizer.step()
#         optimizer.clear_grad()
#         n_iter += 1
#         acc = paddle.metric.accuracy(predict, y_batch, k=1)
#         print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
#         print("   train loss: {}".format(loss.numpy()))
#         print("   train acc:  {}".format(acc.numpy()))


################################## TensorLayerX and Torch can be mixed programming. ##################################
# import os
# os.environ['TL_BACKEND'] = 'torch'
#
# import torch
# from tensorlayerx.nn import Module, Linear, Dropout
# import tensorlayerx as tlx
# from tensorlayerx.dataflow import Dataset, DataLoader
#
# # Get cpu or gpu device for training.
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using {} device".format(device))
#
# # Load MNIST data and make Dataset by TensorLayerX
# X_train, y_train, X_val, y_val, X_test, y_test = tlx.files.load_mnist_dataset(shape=(-1, 784))
#
# class mnistdataset(Dataset):
#
#     def __init__(self, data=X_train, label=y_train):
#         self.data = data
#         self.label = label
#
#     def __getitem__(self, index):
#         data = self.data[index].astype('float32')
#         label = self.label[index].astype('int64')
#         return data, label
#
#     def __len__(self):
#         return len(self.data)
#
# train_dataset = mnistdataset(data=X_train, label=y_train)
# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
#
# # Define the network through TensorLayerX
# class MLP(Module):
#
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.dropout1 = Dropout(p=0.2)
#         self.linear1 = Linear(out_features=800, act=tlx.nn.ReLU, in_features=784)
#         self.dropout2 = Dropout(p=0.2)
#         self.linear2 = Linear(out_features=800, act=tlx.nn.ReLU, in_features=800)
#         self.dropout3 = Dropout(p=0.2)
#         self.linear3 = Linear(out_features=10, act=tlx.nn.ReLU, in_features=800)
#
#     def forward(self, x):
#         z = self.dropout1(x)
#         z = self.linear1(z)
#         z = self.dropout2(z)
#         z = self.linear2(z)
#         z = self.dropout3(z)
#         out = self.linear3(z)
#         return out
#
#
# model = MLP().to(device)
#
# # Define the loss fucntion through TensorLayerX
# loss_fn = tlx.losses.softmax_cross_entropy_with_logits
# # Define the optimizer through torch
# optimizer = torch.optim.SGD(lr=0.05, momentum=0.9, params=model.trainable_weights)
#
# n_epoch = 50
# size = len(train_loader.dataset)
# model.train()
#
# # We use tlx's Model, loss function, Dataset and torch's optimizer to train the network
# for epoch in range(n_epoch):
#     for batch, (X, y) in enumerate(train_loader):
#         X, y = X.to(device), y.to(device)
#
#         # Compute prediction error
#         pred = model(X)
#         loss = loss_fn(pred, y)
#         acc = tlx.metrics.acc(pred, y)
#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if batch % 100 == 0:
#             loss, current = loss.item(), batch * len(X)
#             print(f"loss: {loss:>7f} acc: {acc:>7f}  [{current:>5d}/{size:>5d}] [{epoch} / {n_epoch}epoch]")
