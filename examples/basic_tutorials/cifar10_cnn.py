#! /usr/bin/python
# -*- coding: utf-8 -*-

################################ TensorLayerX and Jittor. #################################

import os
import time
import tensorlayerx as tlx
from tensorlayerx.dataflow import Dataset, DataLoader
from tensorlayerx.vision.transforms import (
    Compose, Resize, RandomFlipHorizontal, RandomContrast, RandomBrightness, StandardizePerImage, RandomCrop
)
from tensorlayerx.nn import Conv2d, Linear, Flatten, Module, MaxPool2d, BatchNorm2d
from tensorlayerx.optimizers import Adam
from tqdm import tqdm

# Enable debug logging
tlx.logging.set_verbosity(tlx.logging.DEBUG)

os.environ['TL_BACKEND'] = 'jittor'

# Download and prepare the CIFAR10 dataset
print("Downloading CIFAR10 dataset...")
X_train, y_train, X_test, y_test = tlx.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)

# Define the CIFAR10 dataset
class CIFAR10Dataset(Dataset):
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

# Define the CIFAR10 images preprocessing pipeline
train_transforms = Compose([
    RandomCrop(size=[24, 24]),
    RandomFlipHorizontal(),
    RandomBrightness(brightness_factor=(0.5, 1.5)),
    RandomContrast(contrast_factor=(0.5, 1.5)),
    StandardizePerImage()
])

test_transforms = Compose([Resize(size=(24, 24)), StandardizePerImage()])

# Create DataLoaders for training and testing
print("Processing CIFAR10 dataset...")
train_dataset = CIFAR10Dataset(data=X_train, label=y_train, transforms=train_transforms)
test_dataset = CIFAR10Dataset(data=X_test, label=y_test, transforms=test_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128)


class SimpleCNN(Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = Conv2d(16, (3, 3), (1, 1), padding='SAME', act=tlx.nn.ReLU, in_channels=3)
        self.conv2 = Conv2d(32, (3, 3), (1, 1), padding='SAME', act=tlx.nn.ReLU, in_channels=16)
        self.maxpool1 = MaxPool2d((2, 2), (2, 2), padding='SAME')
        self.conv3 = Conv2d(64, (3, 3), (1, 1), padding='SAME', act=tlx.nn.ReLU, in_channels=32)
        self.bn1 = BatchNorm2d(num_features=64, act=tlx.nn.ReLU)
        self.conv4 = Conv2d(128, (3, 3), (1, 1), padding='SAME', act=tlx.nn.ReLU, in_channels=64)
        self.maxpool2 = MaxPool2d((2, 2), (2, 2), padding='SAME')
        self.flatten = Flatten()
        self.fc1 = Linear(out_features=128, act=tlx.nn.ReLU, in_features=128 * 6 * 6)
        self.fc2 = Linear(out_features=64, act=tlx.nn.ReLU, in_features=128)
        self.fc3 = Linear(out_features=10, act=None, in_features=64)

    def forward(self, x):
        z = self.conv1(x)
        z = self.conv2(z)
        z = self.maxpool1(z)
        z = self.conv3(z)
        z = self.bn1(z)
        z = self.conv4(z)
        z = self.maxpool2(z)
        z = self.flatten(z)
        z = self.fc1(z)
        z = self.fc2(z)
        z = self.fc3(z)
        return z


# Instantiate the model
model = SimpleCNN()

# Define the optimizer
optimizer = Adam(lr=0.001)
# optimizer = Adam(lr=0.001, params=model.trainable_weights )

# Define the loss function
loss_fn = tlx.losses.softmax_cross_entropy_with_logits

# Use the built-in training method
metric = tlx.metrics.Recall()
tlx_model = tlx.model.Model(network=model, loss_fn=loss_fn, optimizer=optimizer, metrics=metric)
tlx_model.train(n_epoch=2, train_dataset=train_dataloader, print_freq=1, print_train_batch=True)


################################ TensorLayerX and Torch. #################################



# import os
# # os.environ['TL_BACKEND'] = 'paddle'
# # os.environ['TL_BACKEND'] = 'tensorflow'
# # os.environ['TL_BACKEND'] = 'mindspore'
# os.environ['TL_BACKEND'] = 'torch'


# import time
# from tensorlayerx.dataflow import Dataset, DataLoader
# from tensorlayerx.vision.transforms import (
#     Compose, Resize, RandomFlipHorizontal, RandomContrast, RandomBrightness, StandardizePerImage, RandomCrop
# )
# from tensorlayerx.model import TrainOneStep
# from tensorlayerx.nn import Module
# import tensorlayerx as tlx
# from tensorlayerx.nn import (Conv2d, Linear, Flatten, MaxPool2d, BatchNorm2d)
# # enable debug logging
# tlx.logging.set_verbosity(tlx.logging.DEBUG)

# # ################## Download and prepare the CIFAR10 dataset ##################
# # This is just some way of getting the CIFAR10 dataset from an online location
# # and loading it into numpy arrays with shape [32,32,3]
# X_train, y_train, X_test, y_test = tlx.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)

# # ################## CIFAR10 dataset ##################
# # We define a Dataset class for Loading CIFAR10 images and labels.
# class make_dataset(Dataset):

#     def __init__(self, data, label, transforms):
#         self.data = data
#         self.label = label
#         self.transforms = transforms

#     def __getitem__(self, idx):
#         x = self.data[idx].astype('uint8')
#         y = self.label[idx].astype('int64')
#         x = self.transforms(x)

#         return x, y

#     def __len__(self):

#         return len(self.label)

# # We define the CIFAR10 iamges preprocessing pipeline.
# train_transforms = Compose( # Combining multiple operations sequentially
#     [
#         RandomCrop(size=[24, 24]), #random crop from images to shape [24, 24]
#         RandomFlipHorizontal(), # random invert each image horizontally by probability
#         RandomBrightness(brightness_factor=(0.5, 1.5)), # Within the range of values (0.5, 1.5), adjust brightness randomly
#         RandomContrast(contrast_factor=(0.5, 1.5)), # Within the range of values (0.5, 1.5), adjust contrast randomly
#         StandardizePerImage() #Normalize the values of each image to [-1, 1]
#     ]
# )

# test_transforms = Compose([Resize(size=(24, 24)), StandardizePerImage()])

# # We use DataLoader to batch and shuffle data, and make data into iterators.
# train_dataset = make_dataset(data=X_train, label=y_train, transforms=train_transforms)
# test_dataset = make_dataset(data=X_test, label=y_test, transforms=test_transforms)

# train_dataset = DataLoader(train_dataset, batch_size=128, shuffle=True)
# test_dataset = DataLoader(test_dataset, batch_size=128)

# # ################## CNN network ##################
# class CNN(Module):

#     def __init__(self):
#         super(CNN, self).__init__()
#         # Parameter initialization method
#         W_init = tlx.nn.initializers.truncated_normal(stddev=5e-2)
#         W_init2 = tlx.nn.initializers.truncated_normal(stddev=0.04)
#         b_init2 = tlx.nn.initializers.constant(value=0.1)

#         # 2D Convolutional Neural Network, Set padding method "SAME", convolutional kernel size [5,5], stride [1,1], in channels, out channels
#         self.conv1 = Conv2d(64, (5, 5), (1, 1), padding='SAME', W_init=W_init, b_init=None, name='conv1', in_channels=3)
#         # Add 2D BatchNormalize, using ReLU for output.
#         self.bn = BatchNorm2d(num_features=64, act=tlx.nn.ReLU)
#         # Add 2D Max pooling layer.
#         self.maxpool1 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool1')

#         self.conv2 = Conv2d(
#             64, (5, 5), (1, 1), padding='SAME', act=tlx.nn.ReLU, W_init=W_init, name='conv2', in_channels=64
#         )
#         self.maxpool2 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool2')
#         # Flatten 2D data to 1D data
#         self.flatten = Flatten(name='flatten')
#         # Linear layer with 384 units, using ReLU for output.
#         self.linear1 = Linear(384, act=tlx.nn.ReLU, W_init=W_init2, b_init=b_init2, name='linear1relu', in_features=2304)
#         self.linear2 = Linear(192, act=tlx.nn.ReLU, W_init=W_init2, b_init=b_init2, name='linear2relu', in_features=384)
#         self.linear3 = Linear(10, act=None, W_init=W_init2, name='output', in_features=192)

#     # We define the forward computation process.
#     def forward(self, x):
#         z = self.conv1(x)
#         z = self.bn(z)
#         z = self.maxpool1(z)
#         z = self.conv2(z)
#         z = self.maxpool2(z)
#         z = self.flatten(z)
#         z = self.linear1(z)
#         z = self.linear2(z)
#         z = self.linear3(z)
#         return z


# # get the network
# net = CNN()

# # training settings
# n_epoch = 500
# learning_rate = 0.0001
# print_freq = 5
# n_step_epoch = int(len(y_train) / 128)
# n_step = n_epoch * n_step_epoch
# shuffle_buffer_size = 128
# # Get training parameters
# train_weights = net.trainable_weights
# # Define the optimizer, use the Adam optimizer.
# optimizer = tlx.optimizers.Adam(learning_rate)
# # Define evaluation metrics.
# metrics = tlx.metrics.Accuracy()

# # Define the loss calculation process
# class WithLoss(Module):

#     def __init__(self, net, loss_fn):
#         super(WithLoss, self).__init__()
#         self._net = net
#         self._loss_fn = loss_fn

#     def forward(self, data, label):
#         out = self._net(data)
#         loss = self._loss_fn(out, label)
#         return loss


# net_with_loss = WithLoss(net, loss_fn=tlx.losses.softmax_cross_entropy_with_logits)
# # Initialize one-step training
# net_with_train = TrainOneStep(net_with_loss, optimizer, train_weights)

# # Custom training loops
# for epoch in range(n_epoch):
#     start_time = time.time()
#     # Set the network to training state
#     net.set_train()
#     train_loss, train_acc, n_iter = 0, 0, 0
#     # Get training data and labels
#     for X_batch, y_batch in train_dataset:
#         # Calculate the loss value, and automatically complete the gradient update
#         _loss_ce = net_with_train(X_batch, y_batch)
#         train_loss += _loss_ce

#         n_iter += 1
#         _logits = net(X_batch)
#         # Calculate accuracy
#         metrics.update(_logits, y_batch)
#         train_acc += metrics.result()
#         metrics.reset()
#         print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
#         print("   train loss: {}".format(train_loss / n_iter))
#         print("   train acc:  {}".format(train_acc / n_iter))


################################ TensorLayerX and TensorFlow can be mixed programming. #################################
# import os
# os.environ['TL_BACKEND'] = 'tensorflow'
#
# import time
# import numpy as np
# import multiprocessing
# import tensorflow as tf
#
# from tensorlayerx.nn import Module
# import tensorlayerx as tlx
# from tensorlayerx.nn import (Conv2d, Linear, Flatten, MaxPool2d, BatchNorm2d)
#
# # enable debug logging
# tlx.logging.set_verbosity(tlx.logging.DEBUG)
#
# # prepare cifar10 data
# X_train, y_train, X_test, y_test = tlx.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)
#
#
# class CNN(Module):
#
#     def __init__(self):
#         super(CNN, self).__init__()
#         # weights init
#         W_init = tlx.nn.initializers.truncated_normal(stddev=5e-2)
#         W_init2 = tlx.nn.initializers.truncated_normal(stddev=0.04)
#         b_init2 = tlx.nn.initializers.constant(value=0.1)
#
#         self.conv1 = Conv2d(64, (5, 5), (1, 1), padding='SAME', W_init=W_init, b_init=None, name='conv1', in_channels=3)
#         self.bn = BatchNorm2d(num_features=64, act=tlx.nn.ReLU)
#         self.maxpool1 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool1')
#
#         self.conv2 = Conv2d(
#             64, (5, 5), (1, 1), padding='SAME', act=tlx.nn.ReLU, W_init=W_init, b_init=None, name='conv2', in_channels=64
#         )
#         self.maxpool2 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool2')
#
#         self.flatten = Flatten(name='flatten')
#         self.linear1 = Linear(384, act=tlx.nn.ReLU, W_init=W_init2, b_init=b_init2, name='linear1relu', in_channels=2304)
#         self.linear2 = Linear(192, act=tlx.nn.ReLU, W_init=W_init2, b_init=b_init2, name='linear2relu', in_channels=384)
#         self.linear3 = Linear(10, act=None, W_init=W_init2, name='output', in_channels=192)
#
#     def forward(self, x):
#         z = self.conv1(x)
#         z = self.bn(z)
#         z = self.maxpool1(z)
#         z = self.conv2(z)
#         z = self.maxpool2(z)
#         z = self.flatten(z)
#         z = self.linear1(z)
#         z = self.linear2(z)
#         z = self.linear3(z)
#         return z
#
#
# # get the network
# net = CNN()
#
# # training settings
# batch_size = 128
# n_epoch = 500
# learning_rate = 0.0001
# print_freq = 5
# n_step_epoch = int(len(y_train) / batch_size)
# n_step = n_epoch * n_step_epoch
# shuffle_buffer_size = 128
#
# train_weights = net.trainable_weights
# optimizer = tlx.optimizers.Adam(learning_rate)
# # looking for decay learning rate? see https://github.com/tensorlayer/srgan/blob/master/train.py
#
#
# def generator_train():
#     inputs = X_train
#     targets = y_train
#     if len(inputs) != len(targets):
#         raise AssertionError("The length of inputs and targets should be equal")
#     for _input, _target in zip(inputs, targets):
#         # yield _input.encode('utf-8'), _target.encode('utf-8')
#         yield _input, _target
#
#
# def generator_test():
#     inputs = X_test
#     targets = y_test
#     if len(inputs) != len(targets):
#         raise AssertionError("The length of inputs and targets should be equal")
#     for _input, _target in zip(inputs, targets):
#         # yield _input.encode('utf-8'), _target.encode('utf-8')
#         yield _input, _target
#
#
# def _map_fn_train(img, target):
#     # 1. Randomly crop a [height, width] section of the image.
#     img = tf.image.random_crop(img, [24, 24, 3])
#     # 2. Randomly flip the image horizontally.
#     img = tf.image.random_flip_left_right(img)
#     # 3. Randomly change brightness.
#     img = tf.image.random_brightness(img, max_delta=63)
#     # 4. Randomly change contrast.
#     img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
#     # 5. Subtract off the mean and divide by the variance of the pixels.
#     img = tf.image.per_image_standardization(img)
#     target = tf.reshape(target, ())
#     return img, target
#
#
# def _map_fn_test(img, target):
#     # 1. Crop the central [height, width] of the image.
#     img = tf.image.resize_with_pad(img, 24, 24)
#     # 2. Subtract off the mean and divide by the variance of the pixels.
#     img = tf.image.per_image_standardization(img)
#     img = tf.reshape(img, (24, 24, 3))
#     target = tf.reshape(target, ())
#     return img, target
#
#
# # dataset API and augmentation
# train_ds = tf.data.Dataset.from_generator(
#     generator_train, output_types=(tf.float32, tf.int32)
# )  # , output_shapes=((24, 24, 3), (1)))
# train_ds = train_ds.map(_map_fn_train, num_parallel_calls=multiprocessing.cpu_count())
# # train_ds = train_ds.repeat(n_epoch)
# train_ds = train_ds.shuffle(shuffle_buffer_size)
# train_ds = train_ds.prefetch(buffer_size=4096)
# train_ds = train_ds.batch(batch_size)
# # value = train_ds.make_one_shot_iterator().get_next()
#
# test_ds = tf.data.Dataset.from_generator(
#     generator_test, output_types=(tf.float32, tf.int32)
# )  # , output_shapes=((24, 24, 3), (1)))
# # test_ds = test_ds.shuffle(shuffle_buffer_size)
# test_ds = test_ds.map(_map_fn_test, num_parallel_calls=multiprocessing.cpu_count())
# # test_ds = test_ds.repeat(n_epoch)
# test_ds = test_ds.prefetch(buffer_size=4096)
# test_ds = test_ds.batch(batch_size)
# # value_test = test_ds.make_one_shot_iterator().get_next()
#
# for epoch in range(n_epoch):
#     start_time = time.time()
#
#     train_loss, train_acc, n_iter = 0, 0, 0
#     for X_batch, y_batch in train_ds:
#         net.set_train()
#
#         with tf.GradientTape() as tape:
#             # compute outputs
#             _logits = net(X_batch)
#             # compute loss and update model
#             _loss_ce = tlx.losses.softmax_cross_entropy_with_logits(_logits, y_batch)
#
#         grad = tape.gradient(_loss_ce, train_weights)
#         optimizer.apply_gradients(zip(grad, train_weights))
#
#         train_loss += _loss_ce
#         train_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
#         n_iter += 1
#
#         print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
#         print("   train loss: {}".format(train_loss / n_iter))
#         print("   train acc:  {}".format(train_acc / n_iter))
#
#     # use training and evaluation sets to evaluate the model every print_freq epoch
#     if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
#
#         net.set_eval()
#         val_loss, val_acc, n_iter = 0, 0, 0
#         for X_batch, y_batch in test_ds:
#             _logits = net(X_batch)  # is_train=False, disable dropout
#             val_loss += tlx.losses.softmax_cross_entropy_with_logits(_logits, y_batch)
#             val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
#             n_iter += 1
#         print("   val loss: {}".format(val_loss / n_iter))
#         print("   val acc:  {}".format(val_acc / n_iter))
#
# # use testing data to evaluate the model
# net.set_eval()
# test_loss, test_acc, n_iter = 0, 0, 0
# for X_batch, y_batch in test_ds:
#     _logits = net(X_batch)
#     test_loss += tlx.losses.softmax_cross_entropy_with_logits(_logits, y_batch)
#     test_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
#     n_iter += 1
# print("   test loss: {}".format(test_loss / n_iter))
# print("   test acc:  {}".format(test_acc / n_iter))

################################ TensorLayerX and MindSpore can be mixed programming. #################################
# import os
# os.environ['TL_BACKEND'] = 'mindspore'
#
# import time
# import numpy as np
# from tensorlayerx.nn import Module
# import tensorlayerx as tlx
# from tensorlayerx.nn import (Conv2d, Linear, Flatten, MaxPool2d, BatchNorm2d)
# from tensorlayerx.vision.transforms import (
#     Compose, Resize, RandomFlipHorizontal, RandomContrast, RandomBrightness, StandardizePerImage, RandomCrop, HWC2CHW
# )
# from tensorlayerx.dataflow import Dataset, DataLoader
# from mindspore.nn import WithLossCell, Adam
# from mindspore import ParameterTuple
# import mindspore.nn as nn
# from mindspore.ops import composite as C
# import mindspore.ops.operations as P
#
# # enable debug logging
# tlx.logging.set_verbosity(tlx.logging.DEBUG)
#
#
# class CNN(Module):
#
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = Conv2d(
#             64, (5, 5), (2, 2), b_init=None, name='conv1', in_channels=3, act=tlx.nn.ReLU, data_format='channels_first'
#         )
#         self.bn = BatchNorm2d(num_features=64, act=tlx.nn.ReLU, data_format='channels_first')
#         self.maxpool1 = MaxPool2d((3, 3), (2, 2), name='pool1', data_format='channels_first')
#         self.conv2 = Conv2d(
#             128, (5, 5), (2, 2), act=tlx.nn.ReLU, b_init=None, name='conv2', in_channels=64, data_format='channels_first'
#         )
#         self.maxpool2 = MaxPool2d((3, 3), (2, 2), name='pool2', data_format='channels_first')
#
#         self.flatten = Flatten(name='flatten')
#         self.linear1 = Linear(120, act=tlx.nn.ReLU, name='linear1relu', in_channels=512)
#         self.linear2 = Linear(84, act=tlx.nn.ReLU, name='linear2relu', in_channels=120)
#         self.linear3 = Linear(10, act=None, name='output', in_channels=84)
#
#     def forward(self, x):
#         z = self.conv1(x)
#         z = self.bn(z)
#         z = self.maxpool1(z)
#         z = self.conv2(z)
#         z = self.maxpool2(z)
#         z = self.flatten(z)
#         z = self.linear1(z)
#         z = self.linear2(z)
#         z = self.linear3(z)
#         return z
#
#
# # training settings
# batch_size = 128
# n_epoch = 500
# shuffle_buffer_size = 128
#
# # prepare cifar10 data
# X_train, y_train, X_test, y_test = tlx.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)
#
#
# class make_dataset(Dataset):
#
#     def __init__(self, data, label, transforms):
#         self.data = data
#         self.label = label
#         self.transforms = transforms
#
#     def __getitem__(self, idx):
#         x = self.data[idx].astype('uint8')
#         y = self.label[idx].astype('int64')
#         x = self.transforms(x)
#
#         return x, y
#
#     def __len__(self):
#
#         return len(self.label)
#
#
# train_transforms = Compose(
#     [
#         RandomCrop(size=[24, 24]),
#         RandomFlipHorizontal(),
#         RandomBrightness(brightness_factor=(0.5, 1.5)),
#         RandomContrast(contrast_factor=(0.5, 1.5)),
#         StandardizePerImage(),
#         HWC2CHW()
#     ]
# )
#
# test_transforms = Compose([Resize(size=(24, 24)), StandardizePerImage(), HWC2CHW()])
#
# train_dataset = make_dataset(data=X_train, label=y_train, transforms=train_transforms)
# test_dataset = make_dataset(data=X_test, label=y_test, transforms=test_transforms)
#
# train_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_dataset = DataLoader(test_dataset, batch_size=batch_size)
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
# # get the network
# net = CNN()
# train_weights = net.trainable_weights
# optimizer = Adam(train_weights, learning_rate=0.01)
# # optimizer = Momentum(train_weights, 0.01, 0.5)
# criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
# net_with_criterion = WithLossCell(net, criterion)
# train_network = GradWrap(net_with_criterion)
# train_network.set_train()
# # print(train_weights)
# for epoch in range(n_epoch):
#     start_time = time.time()
#     train_network.set_train()
#     train_loss, train_acc, n_iter = 0, 0, 0
#     for X_batch, y_batch in train_dataset:
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

################################### TensorLayerX and Paddle can be mixed programming. ##################################
# import os
# os.environ['TL_BACKEND'] = 'paddle'

# import time
# import paddle as pd
# from tensorlayerx.nn import Module
# import tensorlayerx as tlx
# from tensorlayerx.dataflow import Dataset, DataLoader
# from tensorlayerx.nn import (Conv2d, Linear, Flatten, MaxPool2d, BatchNorm2d)
# from tensorlayerx.vision.transforms import (
#     Compose, Resize, RandomFlipHorizontal, RandomContrast, RandomBrightness, StandardizePerImage, RandomCrop
# )
# # enable debug logging
# tlx.logging.set_verbosity(tlx.logging.DEBUG)

# # prepare cifar10 data
# X_train, y_train, X_test, y_test = tlx.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)


# class CNN(Module):

#     def __init__(self):
#         super(CNN, self).__init__()
#         # weights init
#         W_init = tlx.nn.initializers.truncated_normal(stddev=5e-2)
#         W_init2 = tlx.nn.initializers.truncated_normal(stddev=0.04)
#         b_init2 = tlx.nn.initializers.constant(value=0.1)

#         self.conv1 = Conv2d(64, (5, 5), (1, 1), padding='SAME', W_init=W_init, b_init=None, name='conv1', in_channels=3)
#         self.bn1 = BatchNorm2d(num_features=64, act=tlx.nn.ReLU)
#         self.maxpool1 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool1')

#         self.conv2 = Conv2d(
#             64, (5, 5), (1, 1), padding='SAME', W_init=W_init, b_init=None, name='conv2', in_channels=64
#         )
#         self.bn2 = BatchNorm2d(num_features=64, act=tlx.nn.ReLU)
#         self.maxpool2 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool2')

#         self.flatten = Flatten(name='flatten')
#         self.linear1 = Linear(384, act=tlx.nn.ReLU, W_init=W_init2, b_init=b_init2, name='linear1relu', in_channels=2304)
#         self.linear2 = Linear(192, act=tlx.nn.ReLU, W_init=W_init2, b_init=b_init2, name='linear2relu', in_channels=384)
#         self.linear3 = Linear(10, act=None, W_init=W_init2, name='output', in_channels=192)

#     def forward(self, x):
#         z = self.conv1(x)
#         z = self.bn1(z)
#         z = self.maxpool1(z)
#         z = self.conv2(z)
#         z = self.bn2(z)
#         z = self.maxpool2(z)
#         z = self.flatten(z)
#         z = self.linear1(z)
#         z = self.linear2(z)
#         z = self.linear3(z)
#         return z


# # get the network
# net = CNN()

# # training settings
# batch_size = 128
# n_epoch = 500
# learning_rate = 0.0001
# print_freq = 5
# shuffle_buffer_size = 128
# metrics = tlx.metrics.Accuracy()

# train_weights = net.trainable_weights
# optimizer = tlx.optimizers.Adam(learning_rate)
# # looking for decay learning rate? see https://github.com/tensorlayer/srgan/blob/master/train.py


# class make_dataset(Dataset):

#     def __init__(self, data, label, transforms):
#         self.data = data
#         self.label = label
#         self.transforms = transforms

#     def __getitem__(self, idx):
#         x = self.data[idx].astype('uint8')
#         y = self.label[idx].astype('int64')
#         x = self.transforms(x)

#         return x, y

#     def __len__(self):

#         return len(self.label)


# train_transforms = Compose(
#     [
#         RandomCrop(size=[24, 24]),
#         RandomFlipHorizontal(),
#         RandomBrightness(brightness_factor=(0.5, 1.5)),
#         RandomContrast(contrast_factor=(0.5, 1.5)),
#         StandardizePerImage()
#     ]
# )

# test_transforms = Compose([Resize(size=(24, 24)), StandardizePerImage()])

# train_dataset = make_dataset(data=X_train, label=y_train, transforms=train_transforms)
# test_dataset = make_dataset(data=X_test, label=y_test, transforms=test_transforms)

# train_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_dataset = DataLoader(test_dataset, batch_size=batch_size)

# for epoch in range(n_epoch):
#     train_loss, train_acc, n_iter = 0, 0, 0
#     start_time = time.time()
#     for X_batch, y_batch in train_dataset:
#         net.set_train()
#         output = net(X_batch)
#         loss = pd.nn.functional.cross_entropy(output, y_batch)
#         loss_ce = loss.numpy()
#         grads = optimizer.gradient(loss, train_weights)
#         optimizer.apply_gradients(zip(grads, train_weights))

#         train_loss += loss_ce

#         if metrics:
#             metrics.update(output, y_batch)
#             train_acc += metrics.result()
#             metrics.reset()
#         n_iter += 1

#         print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
#         print("   train loss: {}".format(train_loss / n_iter))
#         print("   train acc:  {}".format(train_acc / n_iter))
