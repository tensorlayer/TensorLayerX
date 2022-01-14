#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorlayerx
import tensorlayerx as tlx

from tests.utils import CustomTestCase
import numpy as np


class Layer_BinaryDense_Test(CustomTestCase):

    @classmethod
    def setUpClass(self):
        print("-" * 20, "Layer_BinaryDense_Test", "-" * 20)
        self.batch_size = 4
        self.inputs_shape = [self.batch_size, 10]

        self.ni = tlx.nn.Input(self.inputs_shape, name='input_layer')
        self.layer1 = tlx.nn.BinaryDense(n_units=5)

        self.layer2 = tlx.nn.BinaryDense(n_units=5, in_channels=10)

        self.n1 = self.layer1(self.ni)
        self.n2 = self.layer2(self.ni)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_layer_n1(self):
        print(self.n1[0])
        self.assertEqual(tlx.ReduceSum()(self.n1).numpy() % 1, 0.0)  # should be integer

    def test_layer_n2(self):
        print(self.n2[0])
        self.assertEqual(tlx.ReduceSum()(self.n2).numpy() % 1, 0.0)  # should be integer


class Layer_DorefaDense_Test(CustomTestCase):

    @classmethod
    def setUpClass(self):
        print("-" * 20, "Layer_DorefaDense_Test", "-" * 20)
        self.batch_size = 4
        self.inputs_shape = [self.batch_size, 10]

        self.ni = tlx.nn.Input(self.inputs_shape, name='input_layer')
        self.layer1 = tlx.nn.DorefaDense(n_units=5)
        self.layer2 = tlx.nn.DorefaDense(n_units=5, in_channels=10)

        self.n1 = self.layer1(self.ni)
        self.n2 = self.layer2(self.ni)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_layer_n1(self):
        print(self.n1[0])

    def test_layer_n2(self):
        print(self.n2[0])


class Layer_DropconnectDense_Test(CustomTestCase):

    @classmethod
    def setUpClass(self):
        print("-" * 20, "Layer_DropconnectDense_Test", "-" * 20)
        self.batch_size = 4
        self.inputs_shape = [self.batch_size, 10]

        self.ni = tlx.nn.Input(self.inputs_shape, name='input_layer')
        self.layer1 = tlx.nn.DropconnectDense(n_units=5, keep=1.0)

        self.layer2 = tlx.nn.DropconnectDense(n_units=5, in_channels=10, keep=0.01)

        self.n1 = self.layer1(self.ni)
        self.n2 = self.layer2(self.ni)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_layer_n1(self):
        print(self.n1[0])

    def test_layer_n2(self):
        print(self.n2[0])


class Layer_QuanDense_Test(CustomTestCase):

    @classmethod
    def setUpClass(self):
        print("-" * 20, "Layer_QuanDense_Test", "-" * 20)
        self.batch_size = 4
        self.inputs_shape = [self.batch_size, 10]

        self.ni = tlx.nn.Input(self.inputs_shape, name='input_layer')
        self.layer1 = tlx.nn.QuanDense(n_units=5)

        self.layer2 = tlx.nn.QuanDense(n_units=5, in_channels=10)

        self.n1 = self.layer1(self.ni)
        self.n2 = self.layer2(self.ni)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_layer_n1(self):
        print(self.n1[0])

    def test_layer_n2(self):
        print(self.n2[0])


class Layer_QuanDenseWithBN_Test(CustomTestCase):

    @classmethod
    def setUpClass(self):
        print("-" * 20, "Layer_QuanDenseWithBN_Test", "-" * 20)
        self.batch_size = 4
        self.inputs_shape = [self.batch_size, 10]

        self.inputs = tensorlayerx.nn.initializers.TruncatedNormal()(shape=self.inputs_shape)
        self.layer1 = tlx.nn.QuanDenseWithBN(n_units=5)
        self.layer2 = tlx.nn.QuanDenseWithBN(n_units=5, in_channels=10)

        self.n1 = self.layer1(self.inputs)
        self.n2 = self.layer2(self.inputs)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_layer_n1(self):
        print(self.n1[0])

    def test_layer_n2(self):
        print(self.n2[0])


class Layer_TernaryDense_Test(CustomTestCase):

    @classmethod
    def setUpClass(self):
        print("-" * 20, "Layer_BinaryDense_Test", "-" * 20)
        self.batch_size = 4
        self.inputs_shape = [self.batch_size, 10]

        self.inputs = tlx.nn.Input(self.inputs_shape, name='input_layer')
        self.layer1 = tlx.nn.TernaryDense(n_units=5)
        self.layer2 = tlx.nn.TernaryDense(n_units=5, in_channels=10)

        self.n1 = self.layer1(self.inputs)
        self.n2 = self.layer2(self.inputs)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_layer_n1(self):
        print(np.unique(self.n1.numpy().reshape(-1)))
        print(self.n1[0])

    def test_layer_n2(self):
        print(np.unique(self.n2.numpy().reshape(-1)))
        print(self.n2[0])


if __name__ == '__main__':

    tlx.logging.set_verbosity(tlx.logging.DEBUG)

    unittest.main()
