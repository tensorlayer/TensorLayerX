#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorlayerx
import tensorlayerx as tlx

from tests.utils import CustomTestCase
import numpy as np


class Layer_BinaryLinear_Test(CustomTestCase):

    @classmethod
    def setUpClass(self):
        print("-" * 20, "Layer_BinaryLinear_Test", "-" * 20)
        self.batch_size = 4
        self.inputs_shape = [self.batch_size, 10]

        self.ni = tlx.nn.Input(self.inputs_shape, name='input_layer')
        self.layer1 = tlx.nn.BinaryLinear(out_features=5)

        self.layer2 = tlx.nn.BinaryLinear(out_features=5, in_features=10)

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


class Layer_DorefaLinear_Test(CustomTestCase):

    @classmethod
    def setUpClass(self):
        print("-" * 20, "Layer_DorefaLinear_Test", "-" * 20)
        self.batch_size = 4
        self.inputs_shape = [self.batch_size, 10]

        self.ni = tlx.nn.Input(self.inputs_shape, name='input_layer')
        self.layer1 = tlx.nn.DorefaLinear(out_features=5)
        self.layer2 = tlx.nn.DorefaLinear(out_features=5, in_features=10)

        self.n1 = self.layer1(self.ni)
        self.n2 = self.layer2(self.ni)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_layer_n1(self):
        print(self.n1[0])

    def test_layer_n2(self):
        print(self.n2[0])


class Layer_DropconnectLinear_Test(CustomTestCase):

    @classmethod
    def setUpClass(self):
        print("-" * 20, "Layer_DropconnectLinear_Test", "-" * 20)
        self.batch_size = 4
        self.inputs_shape = [self.batch_size, 10]

        self.ni = tlx.nn.Input(self.inputs_shape, name='input_layer')
        self.layer1 = tlx.nn.DropconnectLinear(out_features=5, keep=1.0)

        self.layer2 = tlx.nn.DropconnectLinear(out_features=5, in_features=10, keep=0.01)

        self.n1 = self.layer1(self.ni)
        self.n2 = self.layer2(self.ni)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_layer_n1(self):
        print(self.n1[0])

    def test_layer_n2(self):
        print(self.n2[0])


class Layer_QuanLinear_Test(CustomTestCase):

    @classmethod
    def setUpClass(self):
        print("-" * 20, "Layer_QuanLinear_Test", "-" * 20)
        self.batch_size = 4
        self.inputs_shape = [self.batch_size, 10]

        self.ni = tlx.nn.Input(self.inputs_shape, name='input_layer')
        self.layer1 = tlx.nn.QuanLinear(out_features=5)

        self.layer2 = tlx.nn.QuanLinear(out_features=5, in_features=10)

        self.n1 = self.layer1(self.ni)
        self.n2 = self.layer2(self.ni)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_layer_n1(self):
        print(self.n1[0])

    def test_layer_n2(self):
        print(self.n2[0])


class Layer_QuanLinearWithBN_Test(CustomTestCase):

    @classmethod
    def setUpClass(self):
        print("-" * 20, "Layer_QuanLinearWithBN_Test", "-" * 20)
        self.batch_size = 4
        self.inputs_shape = [self.batch_size, 10]

        self.inputs = tensorlayerx.nn.initializers.TruncatedNormal()(shape=self.inputs_shape)
        self.layer1 = tlx.nn.QuanLinearWithBN(out_features=5)
        self.layer2 = tlx.nn.QuanLinearWithBN(out_features=5, in_features=10)

        self.n1 = self.layer1(self.inputs)
        self.n2 = self.layer2(self.inputs)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_layer_n1(self):
        print(self.n1[0])

    def test_layer_n2(self):
        print(self.n2[0])


class Layer_TernaryLinear_Test(CustomTestCase):

    @classmethod
    def setUpClass(self):
        print("-" * 20, "Layer_BinaryLinear_Test", "-" * 20)
        self.batch_size = 4
        self.inputs_shape = [self.batch_size, 10]

        self.inputs = tlx.nn.Input(self.inputs_shape, name='input_layer')
        self.layer1 = tlx.nn.TernaryLinear(out_features=5)
        self.layer2 = tlx.nn.TernaryLinear(out_features=5, in_features=10)

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
