#!/usr/bin/env python
# -*- coding: utf-8 -*-\
import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorlayerx
import tensorlayerx as tlx
import numpy as np

from tests.utils import CustomTestCase


class Layer_nested(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        print("##### begin testing nested layer #####")

    @classmethod
    def tearDownClass(cls):
        pass

    def test_nested_layer_with_inchannels(cls):

        class MyLayer(tensorlayerx.nn.Module):

            def __init__(self, name=None):
                super(MyLayer, self).__init__(name=name)
                self.input_layer = tlx.nn.Linear(in_features=50, out_features=20)
                self.build(None)
                self._built = True

            def build(self, inputs_shape=None):
                self.W = self._get_weights('weights', shape=(20, 10))

            def forward(self, inputs):
                inputs = self.input_layer(inputs)
                output = tlx.ops.matmul(inputs, self.W)
                return output

        class model(tensorlayerx.nn.Module):

            def __init__(self, name=None):
                super(model, self).__init__(name=name)
                self.layer = MyLayer()

            def forward(self, inputs):
                return self.layer(inputs)

        input = tlx.nn.Input(shape=(100, 50))
        model_dynamic = model()
        model_dynamic.set_train()
        cls.assertEqual(model_dynamic(input).shape, (100, 10))
        cls.assertEqual(len(model_dynamic.all_weights), 3)
        cls.assertEqual(len(model_dynamic.trainable_weights), 3)

        model_dynamic.layer.input_layer.biases.assign_add(tlx.ones((20, )))
        cls.assertEqual(np.sum(model_dynamic.all_weights[-1].numpy() - tlx.ones(20, ).numpy()), 0)


if __name__ == '__main__':

    tlx.logging.set_verbosity(tlx.logging.DEBUG)

    unittest.main()
