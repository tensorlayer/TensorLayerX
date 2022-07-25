#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ['TL_BACKEND'] = 'tensorflow'
# os.environ['TL_BACKEND'] = 'paddle'
# os.environ['TL_BACKEND'] = 'mindspore'
# os.environ['TL_BACKEND'] = 'torch'

import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import Linear, Dropout, Conv2d, MaxPool2d, Flatten

class CustomModel(Module):

    def __init__(self):
        super(CustomModel, self).__init__()
        self.dropout1 = Dropout(p=0.2)
        self.linear1 = Linear(out_features=800, act=tlx.nn.ReLU, in_features=784, name='linear1')
        self.dropout2 = Dropout(p=0.8)
        self.linear2 = Linear(out_features=800, act=tlx.nn.ReLU, in_features=800, name='linear2')
        self.dropout3 = Dropout(p=0.8)
        self.linear3 = Linear(out_features=10, act=tlx.nn.ReLU, in_features=800, name='linear3')

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


class CNN(Module):

    def __init__(self):
        super(CNN, self).__init__()
        # weights init
        W_init = tlx.nn.initializers.truncated_normal(stddev=5e-2)
        W_init2 = tlx.nn.initializers.truncated_normal(stddev=0.04)
        b_init2 = tlx.nn.initializers.constant(value=0.1)

        self.conv1 = Conv2d(64, (5, 5), (2, 2), padding='SAME', W_init=W_init, name='conv1', in_channels=3)
        # self.bn = BatchNorm2d(num_features=64, act=tlx.nn.ReLU)
        self.maxpool1 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool1')

        self.conv2 = Conv2d(
            64, (5, 5), (2, 2), padding='SAME', act=tlx.nn.ReLU, W_init=W_init, b_init=None, name='conv2', in_channels=64
        )
        self.maxpool2 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool2')

        self.flatten = Flatten(name='flatten')
        self.linear1 = Linear(384, act=tlx.nn.ReLU, W_init=W_init2, b_init=b_init2, name='linear1', in_features=256)
        self.linear2 = Linear(192, act=tlx.nn.ReLU, W_init=W_init2, b_init=b_init2, name='linear2', in_features=384)
        self.linear3 = Linear(10, act=None, W_init=W_init2, name='linear3', in_features=192)

    def forward(self, x):
        z = self.conv1(x)
        print("conv1 outputs:", z[1, :, :, 1])
        z = self.maxpool1(z)
        print("maxpool outputs:", z[1, :, :, 1])
        z = self.conv2(z)
        print("conv2 outputs:", z[1, :, :, 1])
        z = self.maxpool2(z)
        print("max2 outputs:", z[1, :, :, 1])
        z = self.flatten(z)
        z = self.linear1(z)
        z = self.linear2(z)
        z = self.linear3(z)
        return z


# # TODO The MLP model was saved to the standard npz_dict format after training at the TensorFlow backend
# #  and imported into TensorFlow/PyTorch/PaddlePaddle/MindSpore.
# MLP = CustomModel()
# # MLP.save_standard_weights('./model.npz')
# MLP.load_standard_weights('./model.npz', weights_from='tensorflow', weights_to='mindspore')
# MLP.set_eval()
# inputs = tlx.layers.Input(shape=(10, 784))
# output = MLP(inputs)
# print(output)

# TODO The CNN model was saved to the standard npz_dict format after training at the TensorFlow backend
#  and imported into TensorFlow/PyTorch/PaddlePaddle/MindSpore.
cnn = CNN()
# cnn.save_standard_weights('./cnn.npz')
cnn.load_standard_weights('./cnn.npz', weights_from='torch', weights_to='tensorflow')
cnn.set_eval()

inputs = tlx.nn.Input(shape=(10, 28, 28, 3), dtype=tlx.float32)
outputs = cnn(inputs)
# print(outputs)
