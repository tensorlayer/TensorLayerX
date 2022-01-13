#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorlayerx as tlx
import tensorlayerx
from tests.utils import CustomTestCase


class Layer_RNN_Test(CustomTestCase):

    @classmethod
    def setUpClass(self):
        self.rnncell_input = tlx.nn.Input([4, 16], name='input')
        self.rnncell_prev_h = tlx.nn.Input([4,32])
        self.rnncell = tlx.nn.RNNCell(input_size=16, hidden_size=32, bias=True, act='tanh', name='rnncell_1')
        self.rnncell_out, _ = self.rnncell(self.rnncell_input, self.rnncell_prev_h)

        self.rnn_input = tlx.nn.Input([23, 32, 16], name='input1')
        self.rnn_prev_h = tlx.nn.Input([4, 32, 32])
        self.rnn = tlx.nn.RNN(
            input_size=16, hidden_size=32, bias=True, num_layers=2, bidirectional = True, act='tanh',
            batch_first=False, dropout=0, name='rnn_1')

        self.rnn_out, _ = self.rnn(self.rnn_input, self.rnn_prev_h)

        self.lstmcell_input = tlx.nn.Input([4, 16], name='input')
        self.lstmcell_prev_h = tlx.nn.Input([4, 32])
        self.lstmcell_prev_c = tlx.nn.Input([4, 32])
        self.lstmcell = tlx.nn.LSTMCell(input_size=16, hidden_size=32, bias=True, name='lstmcell_1')
        self.lstmcell_out, (h, c) = self.lstmcell(self.lstmcell_input, (self.lstmcell_prev_h, self.lstmcell_prev_c))

        self.lstm_input = tlx.nn.Input([23, 32, 16], name='input')
        self.lstm_prev_h = tlx.nn.Input([4, 32, 32])
        self.lstm_prev_c = tlx.nn.Input([4, 32, 32])
        self.lstm = tlx.nn.LSTM(input_size=16, hidden_size=32, bias=True, num_layers=2, bidirectional=True,
                              batch_first=False, dropout=0, name='lstm_1')
        self.lstm_out, (h, c) = self.lstm(self.lstm_input, (self.lstm_prev_h, self.lstm_prev_c))

        self.grucell_input = tlx.nn.Input([4, 16], name='input')
        self.grucell_prev_h = tlx.nn.Input([4, 32])
        self.grucell = tlx.nn.GRUCell(input_size=16, hidden_size=32, bias=True, name='grucell_1')
        self.grucell_out, h = self.grucell(self.grucell_input, self.grucell_prev_h)

        self.gru_input = tlx.nn.Input([23, 32, 16], name='input')
        self.gru_prev_h = tlx.nn.Input([4, 32, 32])
        self.gru = tlx.nn.GRU(input_size=16, hidden_size=32, bias=True, num_layers=2, bidirectional=True,
                             batch_first=False, dropout=0, name='GRU_1')
        self.gru_out, h = self.gru(self.gru_input, self.gru_prev_h)

    @classmethod
    def tearDownClass(self):
        pass

    def test_layer_n1(self):

        self.assertEqual(tlx.get_tensor_shape(self.rnncell_out), [4, 32])

    def test_layer_n2(self):

        self.assertEqual(tlx.get_tensor_shape(self.rnn_out), [23, 32, 64])

    def test_layer_n3(self):

        self.assertEqual(tlx.get_tensor_shape(self.lstmcell_out), [4, 32])

    def test_layer_n4(self):

        self.assertEqual(tlx.get_tensor_shape(self.lstm_out), [23, 32, 64])

    def test_layer_n5(self):

        self.assertEqual(tlx.get_tensor_shape(self.grucell_out), [4, 32])

    def test_layer_n6(self):

        self.assertEqual(tlx.get_tensor_shape(self.gru_out), [23, 32, 64])


class Layer_Transformer_Test(CustomTestCase):

    @classmethod
    def setUpClass(self):
        self.multiheadattention_q = tlx.nn.Input(shape=(4,2,128),init=tlx.initializers.ones())
        self.multiheadattention_attn_mask = tlx.convert_to_tensor(np.zeros((4,4)),dtype='bool')
        self.multiheadattention = tlx.nn.MultiheadAttention(embed_dim=128, num_heads=4)
        self.multiheadattention_out = self.multiheadattention(
            self.multiheadattention_q, attn_mask=self.multiheadattention_attn_mask
        )

        self.transformerencoderLayer_q = tlx.nn.Input(shape=(4, 2, 128), init=tlx.initializers.ones())
        self.transformerencoderLayer_attn_mask = tlx.convert_to_tensor(np.zeros((4, 4)), dtype='bool')
        self.encoder = tlx.nn.TransformerEncoderLayer(128, 2, 256)
        self.encoderlayer_out = self.encoder(self.transformerencoderLayer_q, src_mask=self.transformerencoderLayer_attn_mask)

        self.transformerdecoderLayer_q = tlx.nn.Input(shape=(4, 2, 128), init=tlx.initializers.ones())
        self.encoder_layer = tlx.nn.TransformerDecoderLayer(128, 2, 256)
        self.decoderlayer_out = self.encoder_layer(self.transformerdecoderLayer_q, self.transformerdecoderLayer_q)

        self.transformerencoder_q = tlx.nn.Input(shape=(4, 2, 128), init=tlx.initializers.ones())
        self.transformerencoder_attn_mask = tlx.convert_to_tensor(np.zeros((4, 4)), dtype='bool')
        self.encoder_layer = tlx.nn.TransformerEncoderLayer(128, 2, 256)
        self.encoder = tlx.nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.encoder_out = self.encoder(self.transformerencoder_q, mask=self.transformerencoder_attn_mask)

        self.transformeradecoder_q = tlx.nn.Input(shape=(4, 2, 128), init=tlx.initializers.ones())
        self.decoder_layer = tlx.nn.TransformerDecoderLayer(128, 2, 256)
        self.decoder = tlx.nn.TransformerDecoder(self.decoder_layer, num_layers=3)
        self.decoder_out = self.decoder(self.transformeradecoder_q, self.transformeradecoder_q)

        self.src = tlx.nn.Input(shape=(4, 2, 128), init=tlx.initializers.ones())
        self.tgt = tlx.nn.Input(shape=(4, 2, 128), init=tlx.initializers.ones())
        self.layer = tlx.nn.Transformer(d_model=128, nhead=4)
        self.out = self.layer(self.src, self.tgt)

    @classmethod
    def tearDownClass(self):
        pass

    def test_layer_n7(self):

        self.assertEqual(tlx.get_tensor_shape(self.multiheadattention_out[0]), [4, 2, 128])

    def test_layer_n8(self):

        self.assertEqual(tlx.get_tensor_shape(self.encoderlayer_out), [4, 2, 128])

    def test_layer_n9(self):

        self.assertEqual(tlx.get_tensor_shape(self.decoderlayer_out), [4, 2, 128])

    def test_layer_n10(self):

        self.assertEqual(tlx.get_tensor_shape(self.encoder_out), [4, 2, 128])

    def test_layer_n11(self):

        self.assertEqual(tlx.get_tensor_shape(self.decoder_out), [4, 2, 128])

    def test_layer_n12(self):

        self.assertEqual(tlx.get_tensor_shape(self.out), [4, 2, 128])


if __name__ == '__main__':

    unittest.main()
