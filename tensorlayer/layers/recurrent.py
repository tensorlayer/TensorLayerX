#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.backend.ops.load_backend import BACKEND
from tensorlayer.layers.core import Module

__all__ = [
    'RNN',
    'RNNCell',
    'GRU',
    'LSTM',
    'GRUCell',
    'LSTMCell',
]


class RNNCell(Module):
    """An Elman RNN cell with tanh or ReLU non-linearity.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input `x`
    hidden_size : int
        The number of features in the hidden state `h`
    bias : bool
        If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`. Default: ``True``
    act : activation function
        The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'
    name : None or str
        A unique layer name

    Returns
    ----------
    outputs : tensor
        A tensor with shape `[batch_size, hidden_size]`.
    states : tensor
        A tensor with shape `[batch_size, hidden_size]`.
        Tensor containing the next hidden state for each element in the batch

    """

    def __init__(
        self,
        input_size,
        hidden_size,
        bias=True,
        act='tanh',
        name=None,
    ):
        super(RNNCell, self).__init__(name)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        if act not in ('relu', 'tanh'):
            raise ValueError("Activation should be 'tanh' or 'relu'.")
        self.act = act
        self.build(None)
        logging.info("RNNCell %s: input_size: %d hidden_size: %d  act: %s" % (self.name, input_size, hidden_size, act))

    def __repr__(self):
        actstr = self.act
        s = ('{classname}(input_size={input_size}, hidden_size={hidden_size}')
        s += ', bias=True' if self.bias else ', bias=False'
        s += (',' + actstr)
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def check_input(self, input_shape):
        if input_shape[1] != self.input_size:
            raise ValueError(
                'input should have consistent input_size. But got {}, expected {}'.format(
                    input_shape[1], self.input_size
                )
            )

    def check_hidden(self, input_shape, h_shape, hidden_label):
        if input_shape[0] != h_shape[0]:
            raise ValueError(
                'input batch size{} should match hidden{} batch size{}.'.format(
                    input_shape[0], hidden_label, h_shape[0]
                )
            )
        if h_shape[1] != self.hidden_size:
            raise ValueError(
                'hidden{} should have consistent hidden_size. But got {},  expected {}.'.format(
                    hidden_label, h_shape[1], self.hidden_size
                )
            )

    def build(self, inputs_shape):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        _init = tl.initializers.RandomUniform(minval=-stdv, maxval=stdv)
        self.weight_ih_shape = (self.hidden_size, self.input_size)
        self.weight_hh_shape = (self.hidden_size, self.hidden_size)
        self.weight_ih = self._get_weights("weight_ih", shape=self.weight_ih_shape, init=_init)
        self.weight_hh = self._get_weights("weight_hh", shape=self.weight_hh_shape, init=_init)

        if self.bias:
            self.bias_ih_shape = (self.hidden_size, )
            self.bias_hh_shape = (self.hidden_size, )
            self.bias_ih = self._get_weights('bias_ih', shape=self.bias_ih_shape, init=_init)
            self.bias_hh = self._get_weights('bias_hh', shape=self.bias_hh_shape, init=_init)
        else:
            self.bias_ih = None
            self.bias_hh = None
        self.rnncell = tl.ops.rnncell(
            weight_ih=self.weight_ih, weight_hh=self.weight_hh, bias_ih=self.bias_ih, bias_hh=self.bias_hh, act=self.act
        )

    def forward(self, inputs, states=None):
        """

        Parameters
        ----------
        inputs : tensor
            A tensor with shape `[batch_size, input_size]`.
        states : tensor or None
            A tensor with shape `[batch_size, hidden_size]`. When states is None, zero state is used. Defaults to None.

        Examples
        --------
        With TensorLayer

        >>> input = tl.layers.Input([4, 16], name='input')
        >>> prev_h = tl.layers.Input([4,32])
        >>> cell = tl.layers.RNNCell(input_size=16, hidden_size=32, bias=True, act='tanh', name='rnncell_1')
        >>> y, h = cell(input, prev_h)
        >>> print(y.shape)

        """
        input_shape = tl.get_tensor_shape(inputs)
        self.check_input(input_shape)
        if states is None:
            states = tl.zeros(shape=(input_shape[0], self.hidden_size), dtype=inputs.dtype)
        states_shape = tl.get_tensor_shape(states)
        self.check_hidden(input_shape, states_shape, hidden_label='h')
        output, states = self.rnncell(inputs, states)
        return output, states


class LSTMCell(Module):
    """A long short-term memory (LSTM) cell.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input `x`
    hidden_size : int
        The number of features in the hidden state `h`
    bias : bool
        If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`. Default: ``True``
    name : None or str
        A unique layer name

    Returns
    ----------
    outputs : tensor
        A tensor with shape `[batch_size, hidden_size]`.
    states : tensor
        A tuple of two tensor `(h, c)`, each of shape `[batch_size, hidden_size]`.
        Tensors containing the next hidden state and next cell state for each element in the batch.

    """

    def __init__(
        self,
        input_size,
        hidden_size,
        bias=True,
        name=None,
    ):
        super(LSTMCell, self).__init__(name)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.build(None)
        logging.info("LSTMCell %s: input_size: %d hidden_size: %d " % (self.name, input_size, hidden_size))

    def __repr__(self):
        s = ('{classname}(input_size={input_size}, hidden_size={hidden_size}')
        s += ', bias=True' if self.bias else ', bias=False'
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def check_input(self, input_shape):
        if input_shape[1] != self.input_size:
            raise ValueError(
                'input should have consistent input_size. But got {}, expected {}'.format(
                    input_shape[1], self.input_size
                )
            )

    def check_hidden(self, input_shape, h_shape, hidden_label):
        if input_shape[0] != h_shape[0]:
            raise ValueError(
                'input batch size{} should match hidden{} batch size{}.'.format(
                    input_shape[0], hidden_label, h_shape[0]
                )
            )
        if h_shape[1] != self.hidden_size:
            raise ValueError(
                'hidden{} should have consistent hidden_size. But got {},  expected {}.'.format(
                    hidden_label, h_shape[1], self.hidden_size
                )
            )

    def build(self, inputs_shape):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        _init = tl.initializers.RandomUniform(minval=-stdv, maxval=stdv)
        self.weight_ih_shape = (4 * self.hidden_size, self.input_size)
        self.weight_hh_shape = (4 * self.hidden_size, self.hidden_size)
        self.weight_ih = self._get_weights("weight_ih", shape=self.weight_ih_shape, init=_init)
        self.weight_hh = self._get_weights("weight_hh", shape=self.weight_hh_shape, init=_init)

        if self.bias:
            self.bias_ih_shape = (4 * self.hidden_size, )
            self.bias_hh_shape = (4 * self.hidden_size, )
            self.bias_ih = self._get_weights('bias_ih', shape=self.bias_ih_shape, init=_init)
            self.bias_hh = self._get_weights('bias_hh', shape=self.bias_hh_shape, init=_init)
        else:
            self.bias_ih = None
            self.bias_hh = None

        self.lstmcell = tl.ops.lstmcell(
            weight_ih=self.weight_ih, weight_hh=self.weight_hh, bias_ih=self.bias_ih, bias_hh=self.bias_hh
        )

    def forward(self, inputs, states=None):
        """

        Parameters
        ----------
        inputs : tensor
            A tensor with shape `[batch_size, input_size]`.
        states : tuple or None
            A tuple of two tensor `(h, c)`, each of shape `[batch_size, hidden_size]`. When states is None, zero state is used. Defaults: None.

        Examples
        --------
        With TensorLayer

        >>> input = tl.layers.Input([4, 16], name='input')
        >>> prev_h = tl.layers.Input([4,32])
        >>> prev_c = tl.layers.Input([4,32])
        >>> cell = tl.layers.LSTMCell(input_size=16, hidden_size=32, bias=True, name='lstmcell_1')
        >>> y, (h, c)= cell(input, (prev_h, prev_c))
        >>> print(y.shape)

        """
        input_shape = tl.get_tensor_shape(inputs)
        self.check_input(input_shape)
        if states is not None:
            h, c = states
        else:
            h = tl.zeros(shape=(input_shape[0], self.hidden_size), dtype=inputs.dtype)
            c = tl.zeros(shape=(input_shape[0], self.hidden_size), dtype=inputs.dtype)
        h_shape = tl.get_tensor_shape(h)
        c_shape = tl.get_tensor_shape(c)
        self.check_hidden(input_shape, h_shape, hidden_label='h')
        self.check_hidden(input_shape, c_shape, hidden_label='c')
        output, new_h, new_c = self.lstmcell(inputs, h, c)
        return output, (new_h, new_c)


class GRUCell(Module):
    """A gated recurrent unit (GRU) cell.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input `x`
    hidden_size : int
        The number of features in the hidden state `h`
    bias : bool
        If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`. Default: ``True``
    name : None or str
        A unique layer name

    Returns
    ----------
    outputs : tensor
        A tensor with shape `[batch_size, hidden_size]`.
    states : tensor
        A tensor with shape `[batch_size, hidden_size]`.
        Tensor containing the next hidden state for each element in the batch

    """

    def __init__(
        self,
        input_size,
        hidden_size,
        bias=True,
        name=None,
    ):
        super(GRUCell, self).__init__(name)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.build(None)
        logging.info("GRUCell %s: input_size: %d hidden_size: %d " % (self.name, input_size, hidden_size))

    def __repr__(self):
        s = ('{classname}(input_size={input_size}, hidden_size={hidden_size}')
        s += ', bias=True' if self.bias else ', bias=False'
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def check_input(self, input_shape):
        if input_shape[1] != self.input_size:
            raise ValueError(
                'input should have consistent input_size. But got {}, expected {}'.format(
                    input_shape[1], self.input_size
                )
            )

    def check_hidden(self, input_shape, h_shape, hidden_label):
        if input_shape[0] != h_shape[0]:
            raise ValueError(
                'input batch size{} should match hidden{} batch size{}.'.format(
                    input_shape[0], hidden_label, h_shape[0]
                )
            )
        if h_shape[1] != self.hidden_size:
            raise ValueError(
                'hidden{} should have consistent hidden_size. But got {},  expected {}.'.format(
                    hidden_label, h_shape[1], self.hidden_size
                )
            )

    def build(self, inputs_shape):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        _init = tl.initializers.RandomUniform(minval=-stdv, maxval=stdv)
        self.weight_ih_shape = (3 * self.hidden_size, self.input_size)
        self.weight_hh_shape = (3 * self.hidden_size, self.hidden_size)
        self.weight_ih = self._get_weights("weight_ih", shape=self.weight_ih_shape, init=_init)
        self.weight_hh = self._get_weights("weight_hh", shape=self.weight_hh_shape, init=_init)

        if self.bias:
            self.bias_ih_shape = (3 * self.hidden_size, )
            self.bias_hh_shape = (3 * self.hidden_size, )
            self.bias_ih = self._get_weights('bias_ih', shape=self.bias_ih_shape, init=_init)
            self.bias_hh = self._get_weights('bias_hh', shape=self.bias_hh_shape, init=_init)
        else:
            self.bias_ih = None
            self.bias_hh = None

        self.grucell = tl.ops.grucell(
            weight_ih=self.weight_ih, weight_hh=self.weight_hh, bias_ih=self.bias_ih, bias_hh=self.bias_hh
        )

    def forward(self, inputs, states=None):
        """

        Parameters
        ----------
        inputs : tensor
            A tensor with shape `[batch_size, input_size]`.
        states : tensor or None
            A tensor with shape `[batch_size, hidden_size]`. When states is None, zero state is used. Defaults: `None`.

        Examples
        --------
        With TensorLayer

        >>> input = tl.layers.Input([4, 16], name='input')
        >>> prev_h = tl.layers.Input([4,32])
        >>> cell = tl.layers.GRUCell(input_size=16, hidden_size=32, bias=True, name='grucell_1')
        >>> y, h= cell(input, prev_h)
        >>> print(y.shape)

        """
        input_shape = tl.get_tensor_shape(inputs)
        self.check_input(input_shape)
        if states is None:
            states = tl.zeros(shape=(input_shape[0], self.hidden_size), dtype=inputs.dtype)
        states_shape = tl.get_tensor_shape(states)
        self.check_hidden(input_shape, states_shape, hidden_label='h')
        output, states = self.grucell(inputs, states)
        return output, states


class RNNBase(Module):
    """
    RNNBase class for RNN networks. It provides `forward` and other common methods for RNN, LSTM and GRU.
    """

    def __init__(
        self,
        mode,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0.0,
        bidirectional=False,
        name=None,
    ):
        super(RNNBase, self).__init__(name)
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.build(None)

        logging.info(
            "%s: %s: input_size: %d hidden_size: %d  num_layers: %d " %
            (self.mode, self.name, input_size, hidden_size, num_layers)
        )

    def __repr__(self):
        s = (
            '{classname}(input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}'
            ', dropout={dropout}'
        )
        s += ', bias=True' if self.bias else ', bias=False'
        s += ', bidirectional=True' if self.bidirectional else ', bidirectional=False'
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if BACKEND == 'tensorflow':
            bidirect = 2 if self.bidirectional else 1
            self.weights_fw = []
            self.bias_fw = []
            self.weights_bw = []
            self.bias_bw = []
            stdv = 1.0 / np.sqrt(self.hidden_size)
            _init = tl.initializers.RandomUniform(minval=-stdv, maxval=stdv)
            if self.mode == 'LSTM':
                gate_size = 4 * self.hidden_size
            elif self.mode == 'GRU':
                gate_size = 3 * self.hidden_size
            else:
                gate_size = self.hidden_size
            for layer in range(self.num_layers):
                for direction in range(bidirect):
                    layer_input_size = self.input_size if layer == 0 else self.hidden_size * bidirect
                    if direction == 0:
                        self.w_ih = self._get_weights(
                            'weight_ih_l' + str(layer), shape=(gate_size, layer_input_size), init=_init
                        )
                        self.w_hh = self._get_weights(
                            'weight_ih_l' + str(layer), shape=(gate_size, self.hidden_size), init=_init
                        )
                        self.weights_fw.append(self.w_ih)
                        self.weights_fw.append(self.w_hh)
                        if self.bias:
                            self.b_ih = self._get_weights('bias_ih_l' + str(layer), shape=(gate_size, ), init=_init)
                            self.b_hh = self._get_weights('bias_hh_l' + str(layer), shape=(gate_size, ), init=_init)
                            self.bias_fw.append(self.b_ih)
                            self.bias_fw.append(self.b_hh)
                    else:
                        self.w_ih = self._get_weights(
                            'weight_ih_l' + str(layer) + '_reverse', shape=(gate_size, layer_input_size), init=_init
                        )
                        self.w_hh = self._get_weights(
                            'weight_hh_l' + str(layer) + '_reverse', shape=(gate_size, self.hidden_size), init=_init
                        )
                        self.weights_bw.append(self.w_ih)
                        self.weights_bw.append(self.w_hh)
                        if self.bias:
                            self.b_ih = self._get_weights(
                                'bias_ih_l' + str(layer) + '_reverse', shape=(gate_size, ), init=_init
                            )
                            self.b_hh = self._get_weights(
                                'bias_hh_l' + str(layer) + '_reverse', shape=(gate_size, ), init=_init
                            )
                            self.bias_bw.append(self.b_ih)
                            self.bias_bw.append(self.b_hh)

            self.rnn = tl.ops.rnnbase(
                mode=self.mode, input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                bias=self.bias, batch_first=self.batch_first, dropout=self.dropout, bidirectional=self.bidirectional,
                is_train=self.is_train, weights_fw=self.weights_fw, weights_bw=self.weights_bw, bias_fw=self.bias_fw,
                bias_bw=self.bias_bw
            )
        else:
            self.rnn = tl.ops.rnnbase(
                mode=self.mode,
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                bias=self.bias,
                batch_first=self.batch_first,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
                is_train=self.is_train,
            )

    def forward(self, input, states=None):

        output, new_states = self.rnn(input, states)
        return output, new_states


class RNN(RNNBase):
    """Multilayer Elman network(RNN). It takes input sequences and initial
    states as inputs, and returns the output sequences and the final states.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input `x`
    hidden_size : int
        The number of features in the hidden state `h`
    num_layers : int
        Number of recurrent layers.  Default: 1
    bias : bool
        If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`. Default: ``True``
    batch_first : bool
        If ``True``, then the input and output tensors are provided as `[batch_size, seq, input_size]`, Default: ``False``
    dropout : float
        If non-zero, introduces a `Dropout` layer on the outputs of each RNN layer except the last layer,
        with dropout probability equal to `dropout`. Default: 0
    bidirectional : bool
        If ``True``, becomes a bidirectional RNN. Default: ``False``
    act : activation function
        The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'
    name : None or str
        A unique layer name

    Returns
    ----------
    outputs : tensor
        the output sequence. if `batch_first` is True, the shape is `[batch_size, seq, num_directions * hidden_size]`,
        else, the shape is `[seq, batch_size, num_directions * hidden_size]`.
    final_states : tensor
        final states. The shape is `[num_layers * num_directions, batch_size, hidden_size]`. Note that if the RNN is Bidirectional, the forward states are (0,2,4,6,...) and
        the backward states are (1,3,5,7,....).

    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0.0,
        bidirectional=False,
        act='tanh',
        name=None,
    ):
        if act == 'tanh':
            mode = 'RNN_TANH'
        elif act == 'relu':
            mode = 'RNN_RELU'
        else:
            raise ValueError("act should be in ['tanh', 'relu'], but got {}.".format(act))
        super(RNN, self
             ).__init__(mode, input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, name)

    def forward(self, input, states=None):
        """

        Parameters
        ----------
        inputs : tensor
            the input sequence. if `batch_first` is True, the shape is `[batch_size, seq, input_size]`, else, the shape is `[seq, batch_size, input_size]`.
        initial_states : tensor or None
            the initial states. The shape is `[num_layers * num_directions, batch_size, hidden_size]`.If initial_state is not given, zero initial states are used.
            If the RNN is Bidirectional, num_directions should be 2, else it should be 1. Default: None.

        Examples
        --------
        With TensorLayer

        >>> input = tl.layers.Input([23, 32, 16], name='input')
        >>> prev_h = tl.layers.Input([4, 32, 32])
        >>> cell = tl.layers.RNN(input_size=16, hidden_size=32, bias=True, num_layers=2, bidirectional = True, act='tanh', batch_first=False, dropout=0, name='rnn_1')
        >>> y, h= cell(input, prev_h)
        >>> print(y.shape)

        """

        output, new_states = self.rnn(input, states)
        return output, new_states



class LSTM(RNNBase):
    """Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input `x`
    hidden_size : int
        The number of features in the hidden state `h`
    num_layers : int
        Number of recurrent layers.  Default: 1
    bias : bool
        If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`. Default: ``True``
    batch_first : bool
        If ``True``, then the input and output tensors are provided as `[batch_size, seq, input_size]`, Default: ``False``
    dropout : float
        If non-zero, introduces a `Dropout` layer on the outputs of each LSTM layer except the last layer,
        with dropout probability equal to `dropout`. Default: 0
    bidirectional : bool
        If ``True``, becomes a bidirectional LSTM. Default: ``False``
    name : None or str
        A unique layer name

    Returns
    ----------
    outputs : tensor
        the output sequence. if `batch_first` is True, the shape is `[batch_size, seq, num_directions * hidden_size]`,
        else, the shape is `[seq, batch_size, num_directions * hidden_size]`.
    final_states : tensor
        final states. A tuple of two tensor. The shape of each is `[num_layers * num_directions, batch_size, hidden_size]`. Note that if the LSTM is Bidirectional, the forward states are (0,2,4,6,...) and
        the backward states are (1,3,5,7,....).

    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0.0,
        bidirectional=False,
        name=None,
    ):
        super(LSTM, self
             ).__init__('LSTM', input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, name)

    def forward(self, input, states=None):
        """

        Parameters
        ----------
        inputs : tensor
            the input sequence. if `batch_first` is True, the shape is `[batch_size, seq, input_size]`, else, the shape is `[seq, batch_size, input_size]`.
        initial_states : tensor or None
            the initial states. A tuple of tensor (h, c), the shape of each is `[num_layers * num_directions, batch_size, hidden_size]`.If initial_state is not given, zero initial states are used.
            If the LSTM is Bidirectional, num_directions should be 2, else it should be 1. Default: None.

        Examples
        --------
        With TensorLayer

        >>> input = tl.layers.Input([23, 32, 16], name='input')
        >>> prev_h = tl.layers.Input([4, 32, 32])
        >>> prev_c = tl.layers.Input([4, 32, 32])
        >>> cell = tl.layers.LSTM(input_size=16, hidden_size=32, bias=True, num_layers=2, bidirectional = True,  batch_first=False, dropout=0, name='lstm_1')
        >>> y, (h, c)= cell(input, (prev_h, prev_c))
        >>> print(y.shape)

        """

        output, new_states = self.rnn(input, states)
        return output, new_states

class GRU(RNNBase):
    """Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input `x`
    hidden_size : int
        The number of features in the hidden state `h`
    num_layers : int
        Number of recurrent layers. Default: 1
    bias : bool
        If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`. Default: ``True``
    batch_first : bool
        If ``True``, then the input and output tensors are provided as `[batch_size, seq, input_size]`, Default: ``False``
    dropout : float
        If non-zero, introduces a `Dropout` layer on the outputs of each GRU layer except the last layer,
        with dropout probability equal to `dropout`. Default: 0
    bidirectional : bool
        If ``True``, becomes a bidirectional LSTM. Default: ``False``
    name : None or str
        A unique layer name

    Returns
    ----------
    outputs : tensor
        the output sequence. if `batch_first` is True, the shape is `[batch_size, seq, num_directions * hidden_size]`,
        else, the shape is `[seq, batch_size, num_directions * hidden_size]`.
    final_states : tensor
        final states. A tuple of two tensor. The shape of each is `[num_layers * num_directions, batch_size, hidden_size]`. Note that if the GRU is Bidirectional, the forward states are (0,2,4,6,...) and
        the backward states are (1,3,5,7,....).

    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0.0,
        bidirectional=False,
        name=None,
    ):
        super(GRU, self
             ).__init__('GRU', input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, name)

    def forward(self, input, states=None):
        """

        Parameters
        ----------
        inputs : tensor
            the input sequence. if `batch_first` is True, the shape is `[batch_size, seq, input_size]`, else, the shape is `[seq, batch_size, input_size]`.
        initial_states : tensor or None
            the initial states. A tuple of tensor (h, c), the shape of each is `[num_layers * num_directions, batch_size, hidden_size]`.If initial_state is not given, zero initial states are used.
            If the GRU is Bidirectional, num_directions should be 2, else it should be 1. Default: None.

        Examples
        --------
        With TensorLayer

        >>> input = tl.layers.Input([23, 32, 16], name='input')
        >>> prev_h = tl.layers.Input([4, 32, 32])
        >>> cell = tl.layers.GRU(input_size=16, hidden_size=32, bias=True, num_layers=2, bidirectional = True,  batch_first=False, dropout=0, name='GRU_1')
        >>> y, h= cell(input, prev_h)
        >>> print(y.shape)

        """

        output, new_states = self.rnn(input, states)
        return output, new_states
