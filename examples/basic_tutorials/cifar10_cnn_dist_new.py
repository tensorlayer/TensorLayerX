#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ['TL_BACKEND'] = 'paddle'
# os.environ['TL_BACKEND'] = 'jittor'
# os.environ['TL_BACKEND'] = 'tensorflow'
# os.environ['TL_BACKEND'] = 'mindspore'
# os.environ['TL_BACKEND'] = 'torch'

from tensorlayerx.dataflow import Dataset, DataLoader, DistributedBatchSampler
from tensorlayerx.vision.transforms import (
    Compose, Resize, RandomFlipHorizontal, RandomContrast, RandomBrightness, StandardizePerImage, RandomCrop
)
from tensorlayerx.nn import Module
import tensorlayerx as tlx
from tensorlayerx.nn import (Conv2d, Linear, Flatten, MaxPool2d, BatchNorm2d)
# enable debug logging
tlx.logging.set_verbosity(tlx.logging.DEBUG)

# paddle.disable_static()
tlx.ops.set_device('gpu')
print(tlx.ops.get_device())
tlx.ops.distributed_init()
print(tlx.is_distributed())
# ################## Download and prepare the CIFAR10 dataset ##################
# This is just some way of getting the CIFAR10 dataset from an online location
# and loading it into numpy arrays with shape [32,32,3]
X_train, y_train, X_test, y_test = tlx.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)

# training settings
batch_size = 128
n_epoch = 10
learning_rate = 0.0001
print_freq = 5
n_step_epoch = int(len(y_train) / batch_size)
n_step = n_epoch * n_step_epoch
shuffle_buffer_size = 128

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

train_sampler = DistributedBatchSampler(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
valid_sampler = DistributedBatchSampler(test_dataset, batch_size=batch_size, drop_last=True)

train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=2)
valid_loader = DataLoader(test_dataset, batch_sampler=valid_sampler, num_workers=2)


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
        self.bn = BatchNorm2d(num_features=64, act=tlx.ReLU)
        # Add 2D Max pooling layer.
        self.maxpool1 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool1')

        self.conv2 = Conv2d(
            64, (5, 5), (1, 1), padding='SAME', act=tlx.ReLU, W_init=W_init, name='conv2', in_channels=64
        )
        self.maxpool2 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool2')
        # Flatten 2D data to 1D data
        self.flatten = Flatten(name='flatten')
        # Linear layer with 384 units, using ReLU for output.
        self.linear1 = Linear(384, act=tlx.ReLU, W_init=W_init2, b_init=b_init2, name='linear1relu', in_features=2304)
        self.linear2 = Linear(192, act=tlx.ReLU, W_init=W_init2, b_init=b_init2, name='linear2relu', in_features=384)
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

# Define the loss function, use the softmax cross entropy loss.
loss_fn = tlx.losses.softmax_cross_entropy_with_logits
# Define the optimizer, use the Adam optimizer.
optimizer = tlx.optimizers.Adam(learning_rate)
metrics = tlx.metrics.Accuracy()

# Wrap the network with distributed_model
dp_layer = tlx.ops.distributed_model(net)

print("模型已转换为分布式")

# 使用高级 API 构建可训练模型
net_with_train = tlx.model.Model(network=dp_layer, loss_fn=loss_fn, optimizer=optimizer, metrics=metrics)

#执行训练
import time
t0 = time.time()

net_with_train.train(n_epoch=n_epoch, train_dataset=train_loader, print_freq=print_freq, print_train_batch=False)

t1 = time.time()
training_time = t1 - t0
import datetime
def format_time(time):
    elapsed_rounded = int(round((time)))
    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
training_time = format_time(training_time)
print(training_time)
