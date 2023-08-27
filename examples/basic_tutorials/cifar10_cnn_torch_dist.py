#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
# os.environ['TL_BACKEND'] = 'paddle'
# os.environ['TL_BACKEND'] = 'tensorflow'
# os.environ['TL_BACKEND'] = 'mindspore'
os.environ['TL_BACKEND'] = 'torch'

import time
from tensorlayerx.dataflow import Dataset, DataLoader
from tensorlayerx.vision.transforms import (
    Compose, Resize, RandomFlipHorizontal, RandomContrast, RandomBrightness, StandardizePerImage, RandomCrop
)
from tensorlayerx.model import TrainOneStep
from tensorlayerx.nn import Module
import tensorlayerx as tlx
from tensorlayerx.nn import (Conv2d, Linear, Flatten, MaxPool2d, BatchNorm2d)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
args = parser.parse_args()
# enable debug logging
tlx.logging.set_verbosity(tlx.logging.DEBUG)

tlx.ops.set_device(device = 'MLU', id = args.local_rank)
tlx.ops.distributed_init(backend="cncl")
# ################## Download and prepare the CIFAR10 dataset ##################
# This is just some way of getting the CIFAR10 dataset from an online location
# and loading it into numpy arrays with shape [32,32,3]
X_train, y_train, X_test, y_test = tlx.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)

# ################## CIFAR10 dataset ##################
# We define a Dataset class for Loading CIFAR10 images and labels.
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

# We define the CIFAR10 iamges preprocessing pipeline.
train_transforms = Compose( # Combining multiple operations sequentially
    [
        RandomCrop(size=[24, 24]), #random crop from images to shape [24, 24]
        RandomFlipHorizontal(), # random invert each image horizontally by probability
        RandomBrightness(brightness_factor=(0.5, 1.5)), # Within the range of values (0.5, 1.5), adjust brightness randomly
        RandomContrast(contrast_factor=(0.5, 1.5)), # Within the range of values (0.5, 1.5), adjust contrast randomly
        StandardizePerImage() #Normalize the values of each image to [-1, 1]
    ]
)

test_transforms = Compose([Resize(size=(24, 24)), StandardizePerImage()])

# We use DataLoader to batch and shuffle data, and make data into iterators.
train_dataset = make_dataset(data=X_train, label=y_train, transforms=train_transforms)
test_dataset = make_dataset(data=X_test, label=y_test, transforms=test_transforms)

train_dataset = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataset = DataLoader(test_dataset, batch_size=128)

# ################## CNN network ##################
class CNN(Module):

    def __init__(self):
        super(CNN, self).__init__()
        # Parameter initialization method
        W_init = tlx.nn.initializers.truncated_normal(stddev=5e-2)
        W_init2 = tlx.nn.initializers.truncated_normal(stddev=0.04)
        b_init2 = tlx.nn.initializers.constant(value=0.1)

        # 2D Convolutional Neural Network, Set padding method "SAME", convolutional kernel size [5,5], stride [1,1], in channels, out channels
        self.conv1 = Conv2d(64, (5, 5), (1, 1), padding='SAME', W_init=W_init, b_init=None, name='conv1', in_channels=3)
        # Add 2D BatchNormalize, using ReLU for output.
        self.bn = BatchNorm2d(num_features=64, act=tlx.nn.ReLU)
        # Add 2D Max pooling layer.
        self.maxpool1 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool1')

        self.conv2 = Conv2d(
            64, (5, 5), (1, 1), padding='SAME', act=tlx.nn.ReLU, W_init=W_init, name='conv2', in_channels=64
        )
        self.maxpool2 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool2')
        # Flatten 2D data to 1D data
        self.flatten = Flatten(name='flatten')
        # Linear layer with 384 units, using ReLU for output.
        self.linear1 = Linear(384, act=tlx.nn.ReLU, W_init=W_init2, b_init=b_init2, name='linear1relu', in_features=2304)
        self.linear2 = Linear(192, act=tlx.nn.ReLU, W_init=W_init2, b_init=b_init2, name='linear2relu', in_features=384)
        self.linear3 = Linear(10, act=None, W_init=W_init2, name='output', in_features=192)

    # We define the forward computation process.
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
n_epoch = 500
learning_rate = 0.0001
print_freq = 5
n_step_epoch = int(len(y_train) / 128)
n_step = n_epoch * n_step_epoch
shuffle_buffer_size = 128
# Get training parameters
train_weights = net.trainable_weights
# Define the optimizer, use the Adam optimizer.
optimizer = tlx.optimizers.Adam(learning_rate)
# Define evaluation metrics.
metrics = tlx.metrics.Accuracy()

# Define the loss calculation process
class WithLoss(Module):

    def __init__(self, net, loss_fn):
        super(WithLoss, self).__init__()
        self._net = net
        self._loss_fn = loss_fn

    def forward(self, data, label):
        out = self._net(data)
        loss = self._loss_fn(out, label)
        return loss


net_with_loss = WithLoss(net.mlu(), loss_fn=tlx.losses.softmax_cross_entropy_with_logits).mlu()
model = tlx.ops.distributed_model(net_with_loss, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
# Initialize one-step training
#net_with_train = TrainOneStep(net_with_loss, optimizer, train_weights)
net_with_train = TrainOneStep(model, optimizer, train_weights)

# Custom training loops
for epoch in range(n_epoch):
    start_time = time.time()
    # Set the network to training state
    net.set_train()
    train_loss, train_acc, n_iter = 0, 0, 0
    # Get training data and labels
    for X_batch, y_batch in train_dataset:
        # Calculate the loss value, and automatically complete the gradient update
        _loss_ce = net_with_train(X_batch.mlu(), y_batch.mlu())
        train_loss += _loss_ce

        n_iter += 1
        _logits = net(X_batch.mlu())
        # Calculate accuracy
        metrics.update(_logits, y_batch.mlu())
        train_acc += metrics.result()
        metrics.reset()
        if (n_iter  % 100 == 0):
          print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
          print("rank {} train loss: {}".format(args.local_rank,train_loss / n_iter))
          print("rank {} train acc:  {}".format(args.local_rank,train_acc / n_iter))

