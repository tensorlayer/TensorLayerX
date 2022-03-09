#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
# os.environ['TL_BACKEND'] = 'paddle'
# os.environ['TL_BACKEND'] = 'tensorflow'
os.environ['TL_BACKEND'] = 'torch'

import tensorlayerx as tlx

inputs1 = tlx.nn.Input(shape=(2, 2))

class model(tlx.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.fc = tlx.nn.Dropout(keep=0.3)

    def forward(self, input):
        x = self.fc(input)
        return x

net = model()
net.set_eval()
print(net(inputs1))


# inputs2 = tlx.nn.Input(shape=(2, 2))
# stac = tlx.ops.Stack()
# print(stac([inputs1, inputs2]).shape)
# ustac = tlx.ops.Unstack(axis=0, num=2)
# print(ustac(stac([inputs1, inputs2])))

# import torch
# import torch.nn.functional as F
#
# t4d = torch.empty(3, 3, 4)
# p1d = (1, 1) # pad last dim by 1 on each side
# out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
# print(out.size())
# p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
# out = F.pad(t4d, p2d, "constant", 0)
# print(out.size())
# t4d = torch.empty(3, 3, 4, 2)
# p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
# out = F.pad(t4d, p3d, "constant", 0)
# print(out.size())