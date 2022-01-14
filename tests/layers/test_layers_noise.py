#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorlayerx as tlx
import tensorlayerx
from tests.utils import CustomTestCase


class Layer_Convolution_1D_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        print("\n#################################")

        cls.batch_size = 8
        cls.inputs_shape = [cls.batch_size, 200]
        cls.input_layer = tlx.layers.Input(cls.inputs_shape, name='input_layer')

        cls.dense = tlx.nn.Dense(n_units=100, act=tlx.ReLU, in_channels=200)(cls.input_layer)

        cls.noiselayer = tlx.nn.GaussianNoise(name='gaussian')(cls.dense)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_layer_n1(self):
        self.assertEqual(self.noiselayer.get_shape().as_list()[1:], [100])


if __name__ == '__main__':

    # tlx.logging.set_verbosity(tlx.logging.DEBUG)

    unittest.main()
