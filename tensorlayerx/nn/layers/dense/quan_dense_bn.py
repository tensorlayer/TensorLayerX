#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayerx as tlx
from tensorlayerx import logging
from tensorlayerx.nn.core import Module

__all__ = [
    'QuanDenseWithBN',
]


class QuanDenseWithBN(Module):
    """The :class:`QuanDenseWithBN` class is a quantized fully connected layer with BN, which weights are 'bitW' bits and the output of the previous layer
    are 'bitA' bits while inferencing.
    # TODO The QuanDenseWithBN only supports TensorFlow backend.
    Parameters
    ----------
    out_features : int
        The number of units of this layer.
    act : activation function
        The activation function of this layer.
    decay : float
        A decay factor for `ExponentialMovingAverage`.
        Suggest to use a large value for large dataset.
    epsilon : float
        Eplison.
    is_train : boolean
        Is being used for training or inference.
    beta_init : initializer or None or str
        The initializer for initializing beta, if None, skip beta.
        Usually you should not skip beta unless you know what happened.
    gamma_init : initializer or None or str
        The initializer for initializing gamma, if None, skip gamma.
    bitW : int
        The bits of this layer's parameter
    bitA : int
        The bits of the output of previous layer
    use_gemm : boolean
        If True, use gemm instead of ``tf.matmul`` for inferencing. (TODO).
    W_init : initializer or str
        The initializer for the the weight matrix.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    in_features: int
        The number of channels of the previous layer.
        If None, it will be automatically detected when the layer is forwarded for the first time.
    name : a str
        A unique layer name.

    Examples
    ---------
    >>> import tensorlayerx as tlx
    >>> net = tlx.nn.Input([50, 256])
    >>> layer = tlx.nn.QuanDenseWithBN(128, act='relu', name='qdbn1')(net)
    >>> net = tlx.nn.QuanDenseWithBN(256, act='relu', name='qdbn2')(net)
    """

    def __init__(
        self,
        out_features=100,
        act=None,
        decay=0.9,
        epsilon=1e-5,
        is_train=False,
        bitW=8,
        bitA=8,
        gamma_init='truncated_normal',
        beta_init='truncated_normal',
        use_gemm=False,
        W_init='truncated_normal',
        W_init_args=None,
        in_features=None,
        name=None,  # 'quan_dense_with_bn',
    ):
        super(QuanDenseWithBN, self).__init__(act=act, W_init_args=W_init_args, name=name)
        self.out_features = out_features
        self.decay = decay
        self.epsilon = epsilon
        self.is_train = is_train
        self.bitW = bitW
        self.bitA = bitA
        self.gamma_init = self.str_to_init(gamma_init)
        self.beta_init = self.str_to_init(beta_init)
        self.use_gemm = use_gemm
        self.W_init = self.str_to_init(W_init)
        self.in_features = in_features

        if self.in_features is not None:
            self.build((None, self.in_features))
            self._built = True

        logging.info(
            "QuanDenseLayerWithBN  %s: %d %s" %
            (self.name, out_features, self.act.__class__.__name__ if self.act is not None else 'No Activation')
        )

    def __repr__(self):
        actstr = self.act.__class__.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(out_features={out_features}, ' + actstr)
        s += ', bitW={bitW}, bitA={bitA}'
        if self.in_features is not None:
            s += ', in_features=\'{in_features}\''
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if self.in_features is None and len(inputs_shape) != 2:
            raise Exception("The input dimension must be rank 2, please reshape or flatten it")

        if self.in_features is None:
            self.in_features = inputs_shape[1]

        if self.use_gemm:
            raise Exception("TODO. The current version use tf.matmul for inferencing.")

        n_in = inputs_shape[-1]
        self.W = self._get_weights("weights", shape=(n_in, self.out_features), init=self.W_init)

        para_bn_shape = (self.out_features, )
        if self.gamma_init:
            self.scale_para = self._get_weights("gamm_weights", shape=para_bn_shape, init=self.gamma_init)
        else:
            self.scale_para = None

        if self.beta_init:
            self.offset_para = self._get_weights("beta_weights", shape=para_bn_shape, init=self.beta_init)
        else:
            self.offset_para = None

        self.moving_mean = self._get_weights(
            "moving_mean", shape=para_bn_shape, init=tlx.nn.initializers.constant(1.0), trainable=False
        )
        self.moving_variance = self._get_weights(
            "moving_variacne", shape=para_bn_shape, init=tlx.nn.initializers.constant(1.0), trainable=False
        )

        self.quan_dense_bn = tlx.ops.QuanDenseBn(
            self.W, self.scale_para, self.offset_para, self.moving_mean, self.moving_variance, self.decay, self.bitW,
            self.bitA, self.epsilon, self.is_train
        )

    def forward(self, inputs):
        if self._forward_state == False:
            if self._built == False:
                self.build(tlx.get_tensor_shape(inputs))
                self._built = True
            self._forward_state = True

        outputs = self.quan_dense_bn(inputs)

        if self.act:
            outputs = self.act(outputs)
        else:
            outputs = outputs
        return outputs
