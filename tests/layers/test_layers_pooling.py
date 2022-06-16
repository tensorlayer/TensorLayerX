#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorlayerx as tlx
import tensorlayerx
from tensorlayerx.nn import Input
from tests.utils import CustomTestCase


class Layer_Pooling_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        ## 1D ========================================================================

        x_1_input_shape = [None, 100, 1]
        nin_1 = tlx.layers.Input(x_1_input_shape, name='test_in1')

        n1 = tlx.nn.Conv1d(out_channels=32, kernel_size=5, stride=2, name='test_conv1d')(nin_1)
        n2 = tlx.nn.MaxPool1d(kernel_size=3, stride=2, padding='SAME', name='test_maxpool1d')(n1)
        n3 = tlx.nn.AvgPool1d(kernel_size=3, stride=2, padding='SAME', name='test_meanpool1d')(n1)
        n4 = tlx.nn.GlobalMaxPool1d(name='test_maxpool1d')(n1)
        n5 = tlx.nn.GlobalAvgPool1d(name='test_meanpool1d')(n1)
        n16 = tlx.nn.MaxPool1d(kernel_size=3, stride=1, padding='VALID', name='test_maxpool1d')(n1)
        n17 = tlx.nn.AvgPool1d(kernel_size=3, stride=1, padding='VALID', name='test_meanpool1d')(n1)
        n19 = tlx.nn.AdaptiveAvgPool1d(output_size=44, name='test_adaptivemeanpool1d')(n1)
        n20 = tlx.nn.AdaptiveMaxPool1d(output_size=44, name='test_adaptivemaxpool1d')(n1)

        cls.n1_shape = tlx.get_tensor_shape(n1)
        cls.n2_shape = tlx.get_tensor_shape(n2)
        cls.n3_shape = tlx.get_tensor_shape(n3)
        cls.n4_shape = tlx.get_tensor_shape(n4)
        cls.n5_shape = tlx.get_tensor_shape(n5)
        cls.n16_shape = tlx.get_tensor_shape(n16)
        cls.n17_shape = tlx.get_tensor_shape(n17)
        cls.n19_shape = tlx.get_tensor_shape(n19)
        cls.n20_shape = tlx.get_tensor_shape(n20)

        ## 2D ========================================================================

        x_2_input_shape = [None, 100, 100, 3]
        nin_2 = Input(x_2_input_shape, name='test_in2')
        n6 = tlx.nn.Conv2d(out_channels=32, kernel_size=(3, 3), stride=(2, 2), name='test_conv2d')(nin_2)
        n7 = tlx.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding='SAME',
                                           name='test_maxpool2d')(n6)
        n8 = tlx.nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2), padding='SAME',
                                            name='test_meanpool2d')(n6)
        n9 = tlx.nn.GlobalMaxPool2d(name='test_maxpool2d')(n6)
        n10 = tlx.nn.GlobalAvgPool2d(name='test_meanpool2d')(n6)
        # n15 = tlx.nn.PoolLayer(name='test_pool2d')(n6)
        # n18 = tlx.layers.CornerPool2d('TopLeft', name='test_cornerpool2d')(n6)
        n21 = tlx.nn.AdaptiveAvgPool2d(output_size=(45, 32), name='test_adaptivemeanpool2d')(n6)
        n22 = tlx.nn.AdaptiveMaxPool2d(output_size=(45, 32), name='test_adaptivemaxpool2d')(n6)

        cls.n6_shape = tlx.get_tensor_shape(n6)
        cls.n7_shape = tlx.get_tensor_shape(n7)
        cls.n8_shape = tlx.get_tensor_shape(n8)
        cls.n9_shape = tlx.get_tensor_shape(n9)
        cls.n10_shape = tlx.get_tensor_shape(n10)
        # cls.n15_shape = tlx.get_tensor_shape(n15)
        cls.n21_shape = tlx.get_tensor_shape(n21)
        cls.n22_shape = tlx.get_tensor_shape(n22)

        ## 3D ========================================================================

        x_3_input_shape = [None, 100, 100, 100, 3]
        nin_3 = Input(x_3_input_shape, name='test_in3')

        n11 = tlx.nn.AvgPool3d(
            kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME', name='test_meanpool3d'
        )(nin_3)
        n12 = tlx.nn.GlobalMaxPool3d(name='test_maxpool3d')(nin_3)
        n13 = tlx.nn.GlobalAvgPool3d(name='test_meanpool3d')(nin_3)
        n14 = tlx.nn.MaxPool3d(
            kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME', name='test_maxpool3d'
        )(nin_3)

        n23 = tlx.nn.AdaptiveAvgPool3d(output_size=(45, 32, 55), name='test_adaptivemeanpool3d')(nin_3)
        n24 = tlx.nn.AdaptiveMaxPool3d(output_size=(45, 32, 55), name='test_adaptivemaxpool3d')(nin_3)

        cls.n11_shape = n11.get_shape().as_list()
        cls.n12_shape = n12.get_shape().as_list()
        cls.n13_shape = n13.get_shape().as_list()
        cls.n14_shape = n14.get_shape().as_list()
        cls.n21_shape = n21.get_shape().as_list()
        cls.n22_shape = n22.get_shape().as_list()
        cls.n23_shape = n23.get_shape().as_list()
        cls.n24_shape = n24.get_shape().as_list()

    @classmethod
    def tearDownClass(cls):
        pass
        # tf.reset_default_graph()

    def test_n1_shape(self):
        self.assertEqual(self.n1_shape[1:3], [50, 32])

    def test_n2_shape(self):
        self.assertEqual(self.n2_shape[1:3], [25, 32])

    def test_n3_shape(self):
        self.assertEqual(self.n3_shape[1:3], [25, 32])

    def test_n4_shape(self):
        self.assertEqual(self.n4_shape[-1], 32)

    def test_n5_shape(self):
        self.assertEqual(self.n5_shape[-1], 32)

    def test_n6_shape(self):
        self.assertEqual(self.n6_shape[1:4], [50, 50, 32])

    def test_n7_shape(self):
        self.assertEqual(self.n7_shape[1:4], [25, 25, 32])

    def test_n8_shape(self):
        self.assertEqual(self.n8_shape[1:4], [25, 25, 32])

    def test_n9_shape(self):
        self.assertEqual(self.n9_shape[-1], 32)

    def test_n10_shape(self):
        self.assertEqual(self.n10_shape[-1], 32)

    def test_n11_shape(self):
        self.assertEqual(self.n11_shape[1:5], [50, 50, 50, 3])

    def test_n12_shape(self):
        self.assertEqual(self.n12_shape[-1], 3)

    def test_n13_shape(self):
        self.assertEqual(self.n13_shape[-1], 3)

    def test_n14_shape(self):
        self.assertEqual(self.n14_shape[1:5], [50, 50, 50, 3])

    # def test_n15_shape(self):
    #     self.assertEqual(self.n15_shape[1:4], [25, 25, 32])

    def test_n16_shape(self):
        self.assertEqual(self.n16_shape[1:4], [48, 32])

    def test_n17_shape(self):
        self.assertEqual(self.n17_shape[1:4], [48, 32])

    def test_n19_shape(self):
        self.assertEqual(self.n19_shape[1:3], [44, 32])

    def test_n20_shape(self):
        self.assertEqual(self.n20_shape[1:3], [44, 32])

    def test_n21_shape(self):
        self.assertEqual(self.n21_shape[1:4], [45, 32, 32])

    def test_n22_shape(self):
        self.assertEqual(self.n22_shape[1:4], [45, 32, 32])

    def test_n23_shape(self):
        self.assertEqual(self.n23_shape[1:5], [45, 32, 55, 3])

    def test_n24_shape(self):
        self.assertEqual(self.n24_shape[1:5], [45, 32, 55, 3])

    # def test_n18_shape(self):
    #     self.assertEqual(self.n18_shape[1:], [50, 50, 32])


if __name__ == '__main__':

    tlx.logging.set_verbosity(tlx.logging.DEBUG)

    unittest.main()
