#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['TL_BACKEND'] = 'mindspore'

import tensorlayerx as tlx

def test_conv1d():
    input_layer = tlx.nn.Input([8, 100, 1])
    conv1dlayer1 = tlx.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2, data_format='channels_last')
    n1 = conv1dlayer1(input_layer)
    print("Conv1D channels last: ", n1.shape)

    input_layer = tlx.nn.Input([8, 1, 100])
    conv1dlayer2 = tlx.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2, data_format='channels_first')
    n2 = conv1dlayer2(input_layer)
    print("Conv1D channels first: ", n2.shape)

    input_layer = tlx.nn.Input([1, 3, 50])
    dconv1dlayer1 = tlx.nn.ConvTranspose1d(out_channels=64, in_channels=3, kernel_size=4, data_format='channels_first')
    n3 = dconv1dlayer1(input_layer)
    print("DConv1D channels first: ", n3.shape)

    # input_layer = tlx.nn.Input([8, 50, 3])
    # dconv1dlayer2 = tlx.nn.ConvTranspose1d(out_channels=64, in_channels=3, kernel_size=4, data_format='channels_last')
    # n4 = dconv1dlayer2(input_layer)
    # print("Deconv1D channels last", n4.shape)

    input_layer = tlx.nn.Input([8, 50, 1])
    separableconv1d1 = tlx.nn.SeparableConv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2, data_format='channels_last')
    n5 = separableconv1d1(input_layer)
    print("SeparableConv1d: ", n5.shape)

    input_layer = tlx.nn.Input([8, 50, 1])
    separableconv1d2 = tlx.nn.SeparableConv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2, depth_multiplier=4)
    n6 = separableconv1d2(input_layer)
    print("SeparableConv1d: ", n6.shape)

    input_layer = tlx.nn.Input([8, 1, 50])
    separableconv1d3 = tlx.nn.SeparableConv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2, data_format='channels_first')
    n7 = separableconv1d3(input_layer)
    print("SeparableConv1d: ", n7.shape)


def test_conv2d():
    input_layer = tlx.nn.Input([5, 400, 400, 3])
    conv2dlayer1 = tlx.nn.Conv2d(
        out_channels=32, in_channels=3, stride=(2, 2), kernel_size=(5, 5), padding='SAME',
        b_init=tlx.nn.initializers.truncated_normal(0.01), name='conv2dlayer'
    )
    n1 = conv2dlayer1(input_layer)
    print("Conv2d", n1.shape)

    input_layer = tlx.nn.Input([5, 400, 400, 3])
    conv2dlayer2 = tlx.nn.Conv2d(
        out_channels=32, in_channels=3, kernel_size=(3, 3), stride=(2, 2), act=None, name='conv2d'
    )
    n2 = conv2dlayer2(input_layer)
    print("Conv2d", n2.shape)

    input_layer = tlx.nn.Input([5, 400, 400, 32])
    conv2dlayer3 = tlx.nn.Conv2d(
        in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2), act=tlx.ReLU, b_init=None,
        name='conv2d_no_bias'
    )
    n3 = conv2dlayer3(input_layer)
    print("Conv2d", n3.shape)

    input_layer = tlx.nn.Input([5, 32, 400, 400])
    dconv2dlayer = tlx.nn.ConvTranspose2d(
        out_channels=32, in_channels=32, kernel_size=(5, 5), stride=(2, 2), name='deconv2dlayer', data_format='channels_first'
    )
    n4 = dconv2dlayer(input_layer)
    print("ConvTranspose2d", n4.shape)

    # input_layer = tlx.nn.Input([5, 400, 400, 3])
    # dwconv2dlayer = tlx.nn.DepthwiseConv2d(
    #     in_channels=3, kernel_size=(3, 3), stride=(1, 1), dilation=(2, 2), act=tlx.ReLU, depth_multiplier=2,
    #     name='depthwise'
    # )
    # n5 = dwconv2dlayer(input_layer)
    # print("DepthwiseConv2d", n5.shape)

    input_layer = tlx.nn.Input([5, 400, 400, 3])
    separableconv2d = tlx.nn.SeparableConv2d(
        in_channels=3, kernel_size=(3, 3), stride=(2, 2), dilation=(2, 2), act=tlx.ReLU, depth_multiplier=3,
        name='separableconv2d'
    )
    n6 = separableconv2d(input_layer)
    print("SeparableConv2d", n6.shape)
    
def test_conv3d():
    batch_size = 5
    inputs_shape = [batch_size, 3, 20, 20, 20]
    input_layer = tlx.nn.Input(inputs_shape, name='input_layer')

    conv3dlayer1 = tlx.nn.Conv3d(
        out_channels=32, in_channels=3, kernel_size=(2, 2, 2), stride=(2, 2, 2), data_format='channels_first'
    )
    n1 = conv3dlayer1(input_layer)
    print("Conv3d", n1.shape)

    input_layer = tlx.nn.Input([8, 3, 20, 20, 20])
    deconv3dlayer = tlx.nn.ConvTranspose3d(
        out_channels=128, in_channels=3, kernel_size=(2, 2, 2), stride=(2, 2, 2), data_format='channels_first'
    )
    n2 = deconv3dlayer(input_layer)
    print("Deconv3d", n2.shape)

    input_layer = tlx.nn.Input([8, 3, 20, 20, 20])
    conv3dlayer2 = tlx.nn.Conv3d(
        out_channels=64, in_channels=3, kernel_size=(3, 3, 3), stride=(3, 3, 3), act=tlx.ReLU, b_init=None, data_format='channels_first',
        name='conv3d_no_bias'
    )
    n3 = conv3dlayer2(input_layer)
    print("Conv3d", n3.shape)

def test_pooling():
    # 1d pool
    x_1_input_shape = [3, 100, 1]
    nin_1 = tlx.layers.Input(x_1_input_shape, name='test_in1')

    n1 = tlx.nn.Conv1d(out_channels=32, kernel_size=5, stride=2, name='test_conv1d')(nin_1)
    n2 = tlx.nn.MaxPool1d(kernel_size=3, stride=2, padding='SAME', name='test_maxpool1d')(n1)
    n3 = tlx.nn.MeanPool1d(kernel_size=3, stride=2, padding='SAME', name='test_meanpool1d')(n1)
    n4 = tlx.nn.GlobalMaxPool1d(name='test_maxpool1d')(n1)
    n5 = tlx.nn.GlobalMeanPool1d(name='test_meanpool1d')(n1)
    n16 = tlx.nn.MaxPool1d(kernel_size=3, stride=1, padding='VALID', name='test_maxpool1d')(n1)
    n17 = tlx.nn.MeanPool1d(kernel_size=3, stride=1, padding='VALID', name='test_meanpool1d')(n1)
    n19 = tlx.nn.AdaptiveMeanPool1d(output_size=44, name='test_adaptivemeanpool1d')(n1)
    n20 = tlx.nn.AdaptiveMaxPool1d(output_size=44, name='test_adaptivemaxpool1d')(n1)

    n1_shape = tlx.get_tensor_shape(n1)
    n2_shape = tlx.get_tensor_shape(n2)
    n3_shape = tlx.get_tensor_shape(n3)
    n4_shape = tlx.get_tensor_shape(n4)
    n5_shape = tlx.get_tensor_shape(n5)
    n16_shape = tlx.get_tensor_shape(n16)
    n17_shape = tlx.get_tensor_shape(n17)
    n19_shape = tlx.get_tensor_shape(n19)
    n20_shape = tlx.get_tensor_shape(n20)
    print("pooling 1d ", n1_shape, n2_shape, n3_shape, n4_shape, n5_shape, n16_shape, n17_shape, n19_shape, n20_shape)

    # 2d pool
    x_2_input_shape = [3, 100, 100, 3]
    nin_2 = tlx.nn.Input(x_2_input_shape, name='test_in2')

    n6 = tlx.nn.Conv2d(out_channels=32, kernel_size=(3, 3), stride=(2, 2), name='test_conv2d')(nin_2)
    n7 = tlx.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding='SAME',
                          name='test_maxpool2d')(n6)
    n8 = tlx.nn.MeanPool2d(kernel_size=(3, 3), stride=(2, 2), padding='SAME',
                           name='test_meanpool2d')(n6)
    n9 = tlx.nn.GlobalMaxPool2d(name='test_maxpool2d')(n6)
    n10 = tlx.nn.GlobalMeanPool2d(name='test_meanpool2d')(n6)
    n15 = tlx.nn.PoolLayer(name='test_pool2d')(n6)
    n21 = tlx.nn.AdaptiveMeanPool2d(output_size=(45, 32), name='test_adaptivemeanpool2d')(n6)
    n22 = tlx.nn.AdaptiveMaxPool2d(output_size=(45, 32), name='test_adaptivemaxpool2d')(n6)

    n6_shape = tlx.get_tensor_shape(n6)
    n7_shape = tlx.get_tensor_shape(n7)
    n8_shape = tlx.get_tensor_shape(n8)
    n9_shape = tlx.get_tensor_shape(n9)
    n10_shape = tlx.get_tensor_shape(n10)
    n15_shape = tlx.get_tensor_shape(n15)
    n21_shape = tlx.get_tensor_shape(n21)
    n22_shape = tlx.get_tensor_shape(n22)
    print("2d pooling", n6_shape, n7_shape, n8_shape, n9_shape, n10_shape, n15_shape, n21_shape, n22_shape)

    # 3d pool
    # x_3_input_shape = [3, 3, 100, 100, 100]
    # nin_3 = tlx.nn.Input(x_3_input_shape, name='test_in3')

    # Currently support Ascend
    # n11 = tlx.nn.MeanPool3d(
    #     kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME', name='test_meanpool3d', data_format='channels_first'
    # )(nin_3)

    input_layer = tlx.nn.Input([3, 3, 100, 100, 100])
    n12 = tlx.nn.GlobalMaxPool3d(name='test_maxpool3d', data_format='channels_first')(input_layer)
    n13 = tlx.nn.GlobalMeanPool3d(name='test_meanpool3d', data_format='channels_first')(input_layer)

    n14 = tlx.nn.MaxPool3d(
        kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME', name='test_maxpool3d', data_format='channels_first'
    )(input_layer)

    # n23 = tlx.nn.AdaptiveMeanPool3d(output_size=(45, 32, 55), name='test_adaptivemeanpool3d', data_format='channels_first')(input_layer)
    # n24 = tlx.nn.AdaptiveMaxPool3d(output_size=(45, 32, 55), name='test_adaptivemaxpool3d', data_format='channels_first')(input_layer)
    print("3d pooling", n12.shape, n13.shape, n14.shape)

def test_dense():
    input_layer = tlx.nn.Input([10, 30])
    n1 = tlx.nn.Linear(out_features=100, in_features=30, b_init=tlx.initializers.truncated_normal())(input_layer)
    n2 = tlx.nn.Linear(out_features=10, name='none inchannels')(input_layer)
    print("Dense :", n1.shape, n2.shape)

def test_normalization():
    ## Base
    ni_2 = tlx.nn.Input([10, 3, 25, 25])
    nn_2 = tlx.nn.Conv2d(out_channels=32, in_channels=3, data_format='channels_first', kernel_size=(3, 3), stride=(2, 2), name='test_conv2d')(ni_2)
    n2_b = tlx.nn.BatchNorm(name='test_bn2d')(nn_2)
    print(n2_b.shape)

    ni_3 = tlx.nn.Input([10, 3, 25, 25, 25])
    nn_3 = tlx.nn.Conv3d(out_channels=32, in_channels=3, kernel_size=(3, 3, 3), stride=(2, 2, 2), name='test_conv3d', data_format='channels_first')(ni_3)
    n3_b = tlx.nn.BatchNorm(name='test_bn3d')(nn_3)
    print(n3_b.shape)

def test_rnn():
    rnncell_input = tlx.nn.Input([4, 16], name='input')
    rnncell_prev_h = tlx.nn.Input([4, 32])
    rnncell = tlx.nn.RNNCell(input_size=16, hidden_size=32, bias=True, act='tanh', name='rnncell_1')
    rnncell_out, _ = rnncell(rnncell_input, rnncell_prev_h)

    rnn_input = tlx.nn.Input([23, 32, 16], name='input1')
    rnn_prev_h = tlx.nn.Input([4, 32, 32])
    rnn = tlx.nn.RNN(
        input_size=16, hidden_size=32, bias=True, num_layers=2, bidirectional=True, act='tanh',
        batch_first=False, dropout=0, name='rnn_1')

    rnn_out, _ = rnn(rnn_input, rnn_prev_h)

    lstmcell_input = tlx.nn.Input([4, 16], name='input')
    lstmcell_prev_h = tlx.nn.Input([4, 32])
    lstmcell_prev_c = tlx.nn.Input([4, 32])
    lstmcell = tlx.nn.LSTMCell(input_size=16, hidden_size=32, bias=True, name='lstmcell_1')
    lstmcell_out, (h, c) = lstmcell(lstmcell_input, (lstmcell_prev_h, lstmcell_prev_c))

    lstm_input = tlx.nn.Input([23, 32, 16], name='input')
    lstm_prev_h = tlx.nn.Input([4, 32, 32])
    lstm_prev_c = tlx.nn.Input([4, 32, 32])
    lstm = tlx.nn.LSTM(input_size=16, hidden_size=32, bias=True, num_layers=2, bidirectional=True,
                            batch_first=False, dropout=0, name='lstm_1')
    lstm_out, (h, c) = lstm(lstm_input, (lstm_prev_h, lstm_prev_c))

    grucell_input = tlx.nn.Input([4, 16], name='input')
    grucell_prev_h = tlx.nn.Input([4, 32])
    grucell = tlx.nn.GRUCell(input_size=16, hidden_size=32, bias=True, name='grucell_1')
    grucell_out, h = grucell(grucell_input, grucell_prev_h)

    gru_input = tlx.nn.Input([23, 32, 16], name='input')
    gru_prev_h = tlx.nn.Input([4, 32, 32])
    gru = tlx.nn.GRU(input_size=16, hidden_size=32, bias=True, num_layers=2, bidirectional=True,
                          batch_first=False, dropout=0, name='GRU_1')
    gru_out, h = gru(gru_input, gru_prev_h)




if __name__ == '__main__':
    test_conv1d()
    test_conv2d()
    test_conv3d()
    test_pooling()
    test_dense()
    test_normalization()
    # test_rnn()
