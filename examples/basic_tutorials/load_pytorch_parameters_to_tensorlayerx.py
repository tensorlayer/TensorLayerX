#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
# os.environ['TL_BACKEND'] = 'tensorflow'
os.environ['TL_BACKEND'] = 'paddle'
import numpy as np

import torch
import tensorlayerx as tlx
from tensorlayerx import nn

"""
Save pytorch parameters to the a.pth
"""
from torch import nn as th_nn
class B(th_nn.Module):
    def __init__(self):
        super(B, self).__init__()
        self.conv1 = th_nn.Conv2d(3, 16, kernel_size=1)
        self.conv2 = th_nn.Conv2d(16, 16, kernel_size=1)
        self.bn1 = th_nn.BatchNorm2d(16)
        self.act = th_nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn1(self.conv2(self.conv1(x))))

"""
Load the pytorch parameters a.pth to TensorLayerX
"""
class A(nn.Module):
    def __init__(self):
        super(A, self).__init__()
        self.conv1 = nn.Conv2d(16, kernel_size=1, in_channels=3, data_format='channels_first')
        self.conv2 = nn.Conv2d(16, kernel_size=1, in_channels=16, data_format='channels_first')
        self.bn1 = nn.BatchNorm2d(num_features=16, data_format='channels_first')
        self.act = nn.activation.ReLU()

    def forward(self, x):
        return self.act(self.bn1(self.conv2(self.conv1(x))))


def pth2npz(pth_path, npz_path):
    tl_npz = {}
    temp = torch.load(pth_path)
    print("Pytorch parameter names and parameter shapes:")
    for key in temp.keys():
        print(key, temp[key].shape)

    print("Parameter names and parameter shapes of the renamed PyTorch:")
    for key in temp.keys():
        tl_npz[def_rename_torch_key(key)] = def_torch_weight_reshape(temp[key])
        print(def_rename_torch_key(key), def_torch_weight_reshape(temp[key]).shape)
    np.savez(npz_path, **tl_npz)


def def_rename_torch_key(key):
    # Define parameter naming rules that convert the parameter names of PyTorch to TensorLayerX.
    # Only the name changes of Conv2d and BatchNorm2d in this example are given.
    # Different code styles may not be applicable, so you need to customize the here.
    split_key = key.split('.')
    if 'conv' in key and 'weight' in split_key[1]:
        key = 'conv2d_' + key.split('.')[0][-1] + '/' + 'filters'
    if 'conv' in key and 'bias' in split_key[1]:
        key = 'conv2d_' + key.split('.')[0][-1] + '/' + 'biases'
    if 'bn' in key and 'weight' in split_key[1]:
        key = 'batchnorm2d_' + key.split('.')[0][-1] + '/' + 'gamma'
    if 'bn' in key and 'bias' in split_key[1]:
        key = 'batchnorm2d_' + key.split('.')[0][-1] + '/' + 'beta'
    if 'bn' in key and 'running_mean' in split_key[1]:
        key = 'batchnorm2d_' + key.split('.')[0][-1] + '/' + 'moving_mean'
    if 'bn' in key and 'running_var' in split_key[1]:
        key = 'batchnorm2d_' + key.split('.')[0][-1] + '/' + 'moving_var'
    return key

def def_torch_weight_reshape(weight):
    # The shape of the TensorFlow parameter is [ksize, ksize, in_channel, out_channel]
    if tlx.BACKEND == 'tensorflow':
        if isinstance(weight, int):
            return weight
        if len(weight.shape) == 4:
            weight = torch.moveaxis(weight, (1, 0), (2, 3))
        if len(weight.shape) == 5:
            weight = np.moveaxis(weight, (1, 0), (3, 4))
    return weight

if __name__ == '__main__':
    # Step1: save pytorch model parameters to a.pth
    # On the first run, uncomment lines 90 and 91.
    # b = B()
    # torch.save(a.state_dict(), 'a.pth')

    a = A()
    # Step2: Converts pytorch a.pth to the model parameter format of tensorlayerx
    pth2npz('a.pth', 'a.npz')
    # View the parameter name and size of the tensorlayerx
    print("TensorLayer parameter names and parameter shapes:")
    for w in a.all_weights:
        print(w.name, w.shape)

    # Step3: Load model parameters to tensorlayerx
    tlx.files.load_and_assign_npz_dict('a.npz', a, skip=True)
    a.set_eval()

    # Perform tensorlayerx inference to output the value at position [0][0]
    print("TensorLayerX outputs[0][0]:", a(tlx.nn.Input(shape=(5, 3, 3, 3)))[0][0])

    # load torch parameters
    b = B()
    b.eval()
    weights = torch.load('a.pth')
    b.load_state_dict(weights)
    # Perform pytorch inference to output the value at position [0][0]
    print("PyTorch outputs[0][0]：", b(torch.ones((5, 3, 3, 3)))[0][0])
