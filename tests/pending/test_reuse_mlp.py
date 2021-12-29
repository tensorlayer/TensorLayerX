#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayerx as tl

from tests.utils import CustomTestCase


# define the network
def mlp(x, is_train=True, reuse=False):
    with tf.variable_scope("MLP", reuse=reuse):
        tensorlayerx.layers.set_name_reuse(reuse)  # print warning
        network = tensorlayerx.layers.InputLayer(x, name='input')
        network = tensorlayerx.layers.DropoutLayer(network, keep=0.8, is_fix=True, is_train=is_train, name='drop1')
        network = tensorlayerx.layers.DenseLayer(network, n_units=800, act=tf.nn.relu, name='relu1')
        network = tensorlayerx.layers.DropoutLayer(network, keep=0.5, is_fix=True, is_train=is_train, name='drop2')
        network = tensorlayerx.layers.DenseLayer(network, n_units=800, act=tf.nn.relu, name='relu2')
        network = tensorlayerx.layers.DropoutLayer(network, keep=0.5, is_fix=True, is_train=is_train, name='drop3')
        network = tensorlayerx.layers.DenseLayer(network, n_units=10, name='output')
    return network


class MLP_Reuse_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        # define placeholder
        cls.x = tf.placeholder(tf.float32, shape=[None, 784], name='x')

        # define inferences
        mlp(cls.x, is_train=True, reuse=False)
        mlp(cls.x, is_train=False, reuse=True)

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_reuse(self):

        with self.assertRaises(Exception):
            mlp(self.x, is_train=False, reuse=False)  # Already defined model with the same var_scope


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
