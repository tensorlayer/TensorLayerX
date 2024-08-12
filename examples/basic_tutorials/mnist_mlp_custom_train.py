#! /usr/bin/python
# -*- coding: utf-8 -*-

# The same set of code can switch the backend with one line
import os
# os.environ['TL_BACKEND'] = 'tensorflow'
# os.environ['TL_BACKEND'] = 'mindspore'
# os.environ['TL_BACKEND'] = 'paddle'
os.environ['TL_BACKEND'] = 'jittor'
# os.environ['TL_BACKEND'] = 'oneflow'
# os.environ['TL_BACKEND'] = 'torch'

import time
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import Linear, Dropout
from tensorlayerx.dataflow import Dataset, DataLoader
from tensorlayerx.model import TrainOneStep

# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location and loading it into numpy arrays
X_train, y_train, X_val, y_val, X_test, y_test = tlx.files.load_mnist_dataset(shape=(-1, 784))

# ################## MNIST dataset ##################
# We define a Dataset class for Loading MNIST images and labels.
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

# We use DataLoader to batch and shuffle data, and make data into iterators.
train_dataset = mnistdataset(data=X_train, label=y_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

class CustomModel(Module):

    def __init__(self):
        super(CustomModel, self).__init__()
        # It applies 20% dropout to each Linear layer.
        self.dropout1 = Dropout(p=0.2)
        # Linear layer with 800 units, using ReLU for output.
        self.linear1 = Linear(out_features=800, act=tlx.nn.ReLU, in_features=784)
        self.dropout2 = Dropout(p=0.2)
        # Linear layer with 800 units, using ReLU for output.
        self.linear2 = Linear(out_features=800, act=tlx.nn.ReLU, in_features=800)
        self.dropout3 = Dropout(p=0.2)
        # Linear layer with 10 units, using ReLU for output.
        self.linear3 = Linear(out_features=10, act=tlx.nn.ReLU, in_features=800)

    # We define the forward computation process.
    def forward(self, x):
        z = self.dropout1(x)
        z = self.linear1(z)
        z = self.dropout2(z)
        z = self.linear2(z)
        z = self.dropout3(z)
        out = self.linear3(z)
        return out


# We initialize the network
MLP = CustomModel()
# Set the number of training cycles
n_epoch = 50
# set print frequency.
print_freq = 2

# Get training parameters
train_weights = MLP.trainable_weights
# Define the optimizer, use the Momentum optimizer, and set the learning rate to 0.05, momentum to 0.9
optimizer = tlx.optimizers.Momentum(0.05, 0.9)
# Define evaluation metrics.
metric = tlx.metrics.Accuracy()
# Define loss function, this operator implements the cross entropy loss function with softmax. This function
# combines the calculation of the softmax operation and the cross entropy loss function
# to provide a more numerically stable computing.
loss_fn = tlx.losses.softmax_cross_entropy_with_logits



# Define the loss calculation process
class WithLoss(Module):

    def __init__(self, net, loss_fn):
        super(WithLoss, self).__init__()
        # Get network and loss function
        self._net = net
        self._loss_fn = loss_fn

    def forward(self, data, label):
        # Define calculation process
        out = self._net(data)
        loss = self._loss_fn(out, label)
        return loss


net_with_loss = WithLoss(MLP, loss_fn=tlx.losses.softmax_cross_entropy_with_logits)
# Initialize one-step training
net_with_train = TrainOneStep(net_with_loss, optimizer, train_weights)


# Custom training loops
for epoch in range(n_epoch):
    start_time = time.time()
    # Set the network to training state
    MLP.set_train()
    train_loss, train_acc, n_iter = 0, 0, 0
    # Get training data and labels
    for X_batch, y_batch in train_loader:

        # Calculate the loss value, and automatically complete the gradient update
        _loss_ce = net_with_train(X_batch, y_batch)
        train_loss += _loss_ce

        n_iter += 1
        _logits = MLP(X_batch)
        # Calculate accuracy
        metric.update(_logits, y_batch)
        train_acc += metric.result()
        metric.reset()
        print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
        print("   train loss: {}".format(train_loss / n_iter))
        print("   train acc:  {}".format(train_acc / n_iter))