#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
# os.environ['TL_BACKEND'] = 'tensorflow'
# os.environ['TL_BACKEND'] = 'mindspore'
os.environ['TL_BACKEND'] = 'paddle'

from tensorlayerx.nn import Module, LayerList, Dense
import tensorlayerx as tlx

d1 = Dense(n_units=800, act=tlx.ReLU, in_channels=784, name='Dense1')
d2 = Dense(n_units=800, act=tlx.ReLU, in_channels=800, name='Dense2')
d3 = Dense(n_units=10, act=tlx.ReLU, in_channels=800, name='Dense3')

layer_list = LayerList([d1, d2])
# Inserts a given d2 before a given index in the list
layer_list.insert(1, d2)
layer_list.insert(2, d2)
# Appends d2 from a Python iterable to the end of the list.
layer_list.extend([d2])
# Appends a given d3 to the end of the list.
layer_list.append(d3)

print(layer_list)


class model(Module):

    def __init__(self):
        super(model, self).__init__()
        self._list = layer_list

    def forward(self, inputs):
        output = self._list[0](inputs)
        for i in range(1, len(self._list)):
            output = self._list[i](output)
        return output


net = model()
print(net.trainable_weights)
print(net)
print(net(tlx.nn.Input((10, 784))))
