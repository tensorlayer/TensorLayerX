#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import Linear, Conv2d, BatchNorm2d, MaxPool2d, Flatten

class CNN(Module):

    def __init__(self):
        super(CNN, self).__init__()
        # weights init
        W_init = tlx.nn.initializers.truncated_normal(stddev=5e-2)
        W_init2 = tlx.nn.initializers.truncated_normal(stddev=0.04)
        b_init2 = tlx.nn.initializers.constant(value=0.1)

        self.conv1 = Conv2d(64, (5, 5), (1, 1), padding='SAME', W_init=W_init, b_init=None, name='conv1', in_channels=3, act=tlx.nn.ReLU)
        self.bn = BatchNorm2d(num_features=64, act=tlx.nn.ReLU)
        self.maxpool1 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool1')

        self.conv2 = Conv2d(
            64, (5, 5), (1, 1), padding='SAME', act=tlx.nn.ReLU, W_init=W_init, b_init=None, name='conv2', in_channels=64
        )
        self.maxpool2 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool2')

        self.flatten = Flatten(name='flatten')
        self.linear1 = Linear(384, act=tlx.nn.ReLU, W_init=W_init2, b_init=b_init2, name='linear1relu', in_features=2304)
        self.linear2 = Linear(192, act=tlx.nn.ReLU, W_init=W_init2, b_init=b_init2, name='linear2relu', in_features=384)
        self.linear3 = Linear(10, act=None, W_init=W_init2, name='output1', in_features=192)
        self.linear4 = Linear(20, act=None, W_init=W_init2, name='output2', in_features=192)
        self.concat = tlx.nn.Concat(name='concat')

    def forward(self, x):
        z = self.conv1(x)
        z = self.bn(z)
        z = self.maxpool1(z)
        z = self.conv2(z)
        z = self.maxpool2(z)
        z = self.flatten(z)
        z = self.linear1(z)
        z = self.linear2(z)
        z1 = self.linear3(z)
        z2 = self.linear4(z)
        z = self.concat([z1, z2])
        return z

model = CNN()
inputs = tlx.nn.Input(shape=(3, 24, 24, 3))
outputs = model(inputs)

node_by_depth, all_layers = model.build_graph(inputs)

for depth, nodes in enumerate(node_by_depth):
    if depth == 0:
        if isinstance(inputs, list):
            assert len(inputs) == len(nodes)
            for idx, node in enumerate(nodes):
                print(node.node_name, node.layer)
        else:
            print(nodes[0].node_name, nodes[0].layer)
    else:
        for node in nodes:
            print(node.node_name, node.layer)