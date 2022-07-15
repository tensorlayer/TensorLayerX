#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorlayerx as tlx
import tensorlayerx
from tests.utils import CustomTestCase


class Layer_Padding_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        ## 1D
        cls.input_layer1 = tlx.nn.Input([None, 100, 1], name='input_layer1')

        n2 = tlx.nn.ZeroPad1d(padding=(2, 3))(cls.input_layer1)

        cls.n2_shape = n2.get_shape().as_list()

        ## 2D
        cls.input_layer2 = tlx.nn.Input([None, 100, 100, 3], name='input_layer2')

        n0 = tlx.nn.PadLayer([[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT", name='inpad')(cls.input_layer2)
        n5 = tlx.nn.ZeroPad2d(padding=((3, 3), (4, 4)), data_format='channels_last')(cls.input_layer2)

        cls.n0_shape = n0.get_shape().as_list()
        cls.n5_shape = n5.get_shape().as_list()

        ## 3D
        cls.input_layer3 = tlx.nn.Input([None, 100, 100, 100, 3], name='input_layer3')

        n8 = tlx.nn.ZeroPad3d(padding=((3, 3), (4, 4), (5, 5)))(cls.input_layer3)

        cls.n8_shape = n8.get_shape().as_list()

    @classmethod
    def tearDownClass(cls):
        pass

    def test_n0_shape(self):
        self.assertEqual(self.n0_shape[1:], [106, 106, 3])

    def test_n2_shape(self):
        self.assertEqual(self.n2_shape[1:], [105, 1])

    def test_n5_shape(self):
        self.assertEqual(self.n5_shape[1:], [106, 108, 3])

    def test_n8_shape(self):
        self.assertEqual(self.n8_shape[1:], [106, 108, 110, 3])


if __name__ == '__main__':

    tlx.logging.set_verbosity(tlx.logging.DEBUG)

    unittest.main()
