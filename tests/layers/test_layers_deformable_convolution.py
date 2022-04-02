#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorlayerx
import tensorlayerx as tlx
from tests.utils import CustomTestCase


class Layer_Convolution_2D_Test(CustomTestCase):

    @classmethod
    def setUpClass(self):
        print("\n#################################")

        self.batch_size = 5
        self.inputs_shape = [self.batch_size, 10, 10, 16]
        self.input_layer = tlx.nn.Input(self.inputs_shape, name='input_layer')

        self.offset1 = tlx.nn.Conv2d(
            out_channels=18, kernel_size=(3, 3), stride=(1, 1), padding='SAME', name='offset1'
        )(self.input_layer)
        self.init_deformconv1 = tlx.nn.DeformableConv2d(
            offset_layer=self.offset1, out_channels=32, kernel_size=(3, 3), act='relu', name='deformable1'
        )
        self.deformconv1 = self.init_deformconv1(self.input_layer)
        self.offset2 = tlx.nn.Conv2d(
            out_channels=18, kernel_size=(3, 3), stride=(1, 1), padding='SAME', name='offset2'
        )(self.deformconv1)
        self.deformconv2 = tlx.nn.DeformableConv2d(
            offset_layer=self.offset2, out_channels=64, kernel_size=(3, 3), act='relu', name='deformable2'
        )(self.deformconv1)

    @classmethod
    def tearDownClass(self):
        pass

    def test_layer_n1(self):

        self.assertEqual(len(self.init_deformconv1.all_weights), 2)
        self.assertEqual(tlx.get_tensor_shape(self.deformconv1)[1:], [10, 10, 32])

    def test_layer_n2(self):
        self.assertEqual(tlx.get_tensor_shape(self.deformconv2)[1:], [10, 10, 64])


if __name__ == '__main__':

    tlx.logging.set_verbosity(tlx.logging.DEBUG)

    unittest.main()
