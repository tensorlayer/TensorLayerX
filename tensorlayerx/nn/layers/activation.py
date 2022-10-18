#! /usr/bin/python
# -*- coding: utf-8 -*-

from tensorlayerx import logging
import tensorlayerx as tlx
from tensorlayerx.nn.core import Module

__all__ = [
    'ELU', 'PRelu', 'PRelu6', 'PTRelu6', 'ReLU', 'ReLU6', 'Softplus', 'LeakyReLU', 'LeakyReLU6', 'LeakyTwiceRelu6',
    'Ramp', 'Swish', 'HardTanh', 'Mish', 'Tanh', 'Sigmoid', 'Softmax', 'LogSoftmax', 'HardSigmoid', 'Hardswish'
]


class PRelu(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{PReLU}(x) = \max(0,x) + a * \min(0,x)

    Parameters
    ----------
    num_parameters : int
        number of `a` to learn.  1, or the number of channels at input. Default: 1
    init : float
        the initial value of `a`. Default: 0.25
    data_format : str
        Data format that specifies the layout of input. It may be 'channels_last' or 'channels_first'. Default is 'channels_last'.
    name : None or str
        A unique layer name.

    Examples
    -----------
    >>> inputs = tlx.nn.Input([10, 5, 10])
    >>> prelulayer = tlx.nn.PRelu(num_parameters=5, init=0.25, data_format='channels_first')(inputs)

    References
    -----------
    - `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification <http://arxiv.org/abs/1502.01852>`__
    - `Convolutional Deep Belief Networks on CIFAR-10 [A. Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`__

    """

    def __init__(
        self, num_parameters = 1, init=0.25,  data_format='channels_last', name=None,
    ):

        super(PRelu, self).__init__(name)
        self.num_parameters = num_parameters
        self.init = init
        self.data_format = data_format

        logging.info("PRelu %s: num_parameters: %s" % (self.name, self.num_parameters))

    def __repr__(self):
        s = ('{classname}(')
        s += 'num_parameters={num_parameters},'
        s += 'init={init},'
        s += 'name={name}'
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        dim = len(inputs_shape)
        if self.data_format == 'channels_last':
            w_shape = (self.num_parameters, )
        elif self.data_format == 'channels_first':
            if dim == 4:
                w_shape = (1, self.num_parameters, 1, 1)
            elif dim == 3:
                w_shape = (1, self.num_parameters, 1)
            elif dim == 5:
                w_shape = (1, self.num_parameters, 1, 1, 1)
            elif dim < 3:
                w_shape = (self.num_parameters, )

        self.alpha = self._get_weights("alpha", shape=w_shape, init=tlx.initializers.constant(value=self.init))
        self.prelu = tlx.ops.PReLU(data_format = self.data_format)

    def forward(self, inputs):
        if self._forward_state == False:
            if self._built == False:
                self.build(tlx.get_tensor_shape(inputs))
                self._built = True
            self._forward_state = True

        output = self.prelu(inputs, self.alpha)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, output)
            self._nodes_fixed = True
        return output


class PRelu6(Module):
    r"""
    The :class:`PRelu6` class is Parametric Rectified Linear layer integrating ReLU6 behaviour.

    This activation layer use a modified version :func:`tlx.nn.LeakyReLU` introduced by the following paper:
    `Rectifier Nonlinearities Improve Neural Network Acoustic Models [A. L. Maas et al., 2013] <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`__

    This activation function also use a modified version of the activation function :func:`tf.nn.relu6` introduced by the following paper:
    `Convolutional Deep Belief Networks on CIFAR-10 [A. Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`__

    This activation layer push further the logic by adding `leaky` behaviour both below zero and above six.

    The function return the following results:
      - When x < 0: ``f(x) = alpha_low * x``.
      - When x in [0, 6]: ``f(x) = x``.
      - When x > 6: ``f(x) = 6``.

    Parameters
    ----------
    channel_shared : boolean
        If True, single weight is shared by all channels.
    in_channels: int
        The number of channels of the previous layer.
        If None, it will be automatically detected when the layer is forwarded for the first time.
    a_init : initializer or str
        The initializer for initializing the alpha(s).
    name : None or str
        A unique layer name.

    Examples
    -----------
    >>> inputs = tlx.nn.Input([10, 5])
    >>> prelulayer = tlx.nn.PRelu6(channel_shared=True, in_channels=5)(inputs)

    References
    -----------
    - `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification <http://arxiv.org/abs/1502.01852>`__
    - `Rectifier Nonlinearities Improve Neural Network Acoustic Models [A. L. Maas et al., 2013] <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`__
    - `Convolutional Deep Belief Networks on CIFAR-10 [A. Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`__

    """

    def __init__(
        self,
        channel_shared=False,
        in_channels=None,
        a_init='truncated_normal',
        name=None,  # "prelu6"
        data_format='channels_last',
        dim=2
    ):

        super(PRelu6, self).__init__(name)
        self.channel_shared = channel_shared
        self.in_channels = in_channels
        self.a_init = self.str_to_init(a_init)
        self.data_format = data_format
        self.dim = dim

        if self.channel_shared:
            self.build((None, ))
            self._built = True
        elif self.in_channels is not None:
            self.build((None, self.in_channels))
            self._built = True

        logging.info("PRelu6 %s: channel_shared: %s" % (self.name, self.channel_shared))

    def __repr__(self):
        s = ('{classname}(')
        s += 'channel_shared={channel_shared},'
        s += 'in_channels={in_channels},'
        s += 'name={name}'
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if self.in_channels is None:
            if self.data_format == 'channels_last':
                self.in_channels = inputs_shape[-1]
            else:
                self.in_channels = inputs_shape[1]
        if self.channel_shared:
            w_shape = (1, )
        elif self.data_format == 'channels_last':
            w_shape = (self.in_channels, )
        elif self.data_format == 'channels_first':
            if self.dim == 2:
                w_shape = (1, self.in_channels, 1, 1)
            elif self.dim == 1:
                w_shape = (1, self.in_channels, 1)
            elif self.dim == 3:
                w_shape = (1, self.in_channels, 1, 1, 1)
            else:
                raise Exception("Dim should be equal to 1, 2 or 3")
        self.alpha = self._get_weights("alpha", shape=w_shape, init=self.a_init)
        self.sigmoid = tlx.ops.Sigmoid()
        self.relu = tlx.ops.ReLU()

    def forward(self, inputs):
        if self._forward_state == False:
            if self._built == False:
                self.build(tlx.get_tensor_shape(inputs))
                self._built = True
            self._forward_state = True

        alpha_var_constrained = self.sigmoid(self.alpha)
        pos = self.relu(inputs)
        pos_6 = -self.relu(inputs - 6)
        neg = -alpha_var_constrained * self.relu(-inputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, pos + pos_6 + neg)
            self._nodes_fixed = True
        return pos + pos_6 + neg


class PTRelu6(Module):
    r"""
    The :class:`PTRelu6` class is Parametric Rectified Linear layer integrating ReLU6 behaviour.

    This activation layer use a modified version :func:`tlx.nn.LeakyReLU` introduced by the following paper:
    `Rectifier Nonlinearities Improve Neural Network Acoustic Models [A. L. Maas et al., 2013] <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`__

    This activation function also use a modified version of the activation function :func:`tf.nn.relu6` introduced by the following paper:
    `Convolutional Deep Belief Networks on CIFAR-10 [A. Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`__

    This activation layer push further the logic by adding `leaky` behaviour both below zero and above six.

    The function return the following results:
      - When x < 0: ``f(x) = alpha_low * x``.
      - When x in [0, 6]: ``f(x) = x``.
      - When x > 6: ``f(x) = 6 + (alpha_high * (x-6))``.

    This version goes one step beyond :class:`PRelu6` by introducing leaky behaviour on the positive side when x > 6.

    Parameters
    ----------
    channel_shared : boolean
        If True, single weight is shared by all channels.
    in_channels: int
        The number of channels of the previous layer.
        If None, it will be automatically detected when the layer is forwarded for the first time.
    a_init : initializer or str
        The initializer for initializing the alpha(s).
    name : None or str
        A unique layer name.

    Examples
    -----------
    >>> inputs = tlx.nn.Input([10, 5])
    >>> prelulayer = tlx.nn.PTRelu6(channel_shared=True, in_channels=5)(inputs)

    References
    -----------
    - `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification <http://arxiv.org/abs/1502.01852>`__
    - `Convolutional Deep Belief Networks on CIFAR-10 [A. Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`__
    - `Rectifier Nonlinearities Improve Neural Network Acoustic Models [A. L. Maas et al., 2013] <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`__

    """

    def __init__(
        self,
        channel_shared=False,
        in_channels=None,
        data_format='channels_last',
        a_init='truncated_normal',
        name=None  # "ptrelu6"
    ):

        super(PTRelu6, self).__init__(name)
        self.channel_shared = channel_shared
        self.in_channels = in_channels
        self.data_format = data_format
        self.a_init = self.str_to_init(a_init)

        if self.channel_shared:
            self.build((None, ))
            self._built = True
        elif self.in_channels:
            self.build((None, self.in_channels))
            self._built = True

        logging.info("PTRelu6 %s: channel_shared: %s" % (self.name, self.channel_shared))

    def __repr__(self):
        s = ('{classname}(')
        s += 'channel_shared={channel_shared},'
        s += 'in_channels={in_channels},'
        s += 'name={name}'
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if self.in_channels is None:
            if self.data_format == 'channels_last':
                self.in_channels = inputs_shape[-1]
            else:
                self.in_channels = inputs_shape[1]

        if self.channel_shared:
            w_shape = (1, )
        elif self.data_format == 'channels_last':
            w_shape = (self.in_channels, )
        elif self.data_format == 'channels_first':
            if self.dim == 2:
                w_shape = (1, self.in_channels, 1, 1)
            elif self.dim == 1:
                w_shape = (1, self.in_channels, 1)
            elif self.dim == 3:
                w_shape = (1, self.in_channels, 1, 1, 1)
            else:
                raise Exception("Dim should be equal to 1, 2 or 3")

        # Alpha for outputs lower than zeros
        self.alpha_low = self._get_weights("alpha_low", shape=w_shape, init=self.a_init)
        self.sigmoid = tlx.ops.Sigmoid()
        self.relu = tlx.ops.ReLU()
        # Alpha for outputs higher than 6
        self.alpha_high = self._get_weights("alpha_high", shape=w_shape, init=self.a_init)

    def forward(self, inputs):
        if self._forward_state == False:
            if self._built == False:
                self.build(tlx.get_tensor_shape(inputs))
                self._built = True
            self._forward_state = True

        alpha_low_constrained = self.sigmoid(self.alpha_low)
        alpha_high_constrained = self.sigmoid(self.alpha_high)
        pos = self.relu(inputs)
        pos_6 = -self.relu(inputs - 6) + alpha_high_constrained * self.relu(inputs - 6)
        neg = -alpha_low_constrained * self.relu(-inputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, pos + pos_6 + neg)
            self._nodes_fixed = True
        return pos + pos_6 + neg


class Ramp(Module):
    r"""Ramp activation function.

    Reference: [tf.clip_by_value]<https://www.tensorflow.org/api_docs/python/tf/clip_by_value>

    Parameters
    ----------
    x : Tensor
        input.
    v_min : float
        cap input to v_min as a lower bound.
    v_max : float
        cap input to v_max as a upper bound.

    Returns
    -------
    Tensor
        A ``Tensor`` in the same type as ``x``.

    Examples
    -----------
    >>> inputs = tlx.nn.Input([10, 5])
    >>> prelulayer = tlx.nn.Ramp()(inputs)

    """

    def __init__(self, v_min=0, v_max=1):
        super(Ramp, self).__init__()
        self._built = True
        self.v_min = v_min
        self.v_max = v_max

    def forward(self, x):
        outputs = tlx.ops.clip_by_value(x, clip_value_min=self.v_min, clip_value_max=self.v_max)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(x, outputs)
            self._nodes_fixed = True
        return outputs


class ELU(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{ELU}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
        \end{cases}

    Parameters
    ----------
    alpha : float
        the :math:`\alpha` value for the ELU formulation. Default: 1.0
    name : str
        The function name (optional).

    Returns
    -------
    Tensor
        A ``Tensor`` in the same type as ``x``.

    Examples
    --------
    >>> net = tlx.nn.Input([10, 200])
    >>> net = tlx.nn.ELU(alpha=0.5)(net)

    """

    def __init__(self, alpha=1.0):
        super(ELU, self).__init__()
        self._built = True
        self.alpha = alpha
        self._elu = tlx.ops.ELU(alpha=alpha)

    def forward(self, x):
        outputs = self._elu(x)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(x, outputs)
            self._nodes_fixed = True
        return outputs


class Softmax(Module):
    """Applies the Softmax function to an n-dimensional input Tensor rescaling them so that the elements of the n-dimensional output Tensor lie in the range [0,1] and sum to 1.

    Parameters
    ----------
    axis : int
         A dimension along which Softmax will be computed

    Examples
    --------
    >>> net = tlx.nn.Input([10, 200])
    >>> net = tlx.nn.Softmax()(net)

    Returns
    -------
    Tensor
        A ``Tensor`` in the same type as ``x``.

    """

    def __init__(self, axis = -1):
        super(Softmax, self).__init__()
        self._built = True
        self.axis = axis
        self._softmax = tlx.ops.Softmax(axis)

    def forward(self, x):
        outputs = self._softmax(x)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(x, outputs)
            self._nodes_fixed = True
        return outputs


class Sigmoid(Module):
    """Computes sigmoid of x element-wise.
    Formula for calculating sigmoid(x) = 1/(1+exp(-x))

    Parameters
    ----------
    x : Tensor
        Support input type ``float``, ``double``, ``int32``, ``int64``, ``uint8``, ``int16``, or ``int8``.

    Examples
    --------
    >>> net = tlx.nn.Input([10, 200])
    >>> net = tlx.nn.Sigmoid()(net)

    Returns
    -------
    Tensor
        A ``Tensor`` in the same type as ``x``.

    """

    def __init__(self):
        super(Sigmoid, self).__init__()
        self._built = True
        self._sigmoid = tlx.ops.Sigmoid()

    def forward(self, x):
        outputs = self._sigmoid(x)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(x, outputs)
            self._nodes_fixed = True
        return outputs


class Tanh(Module):
    """Applies the Hyperbolic Tangent (Tanh) function element-wise.

    Parameters
    ----------
    x : Tensor
        Support input type ``float``, ``double``, ``int32``, ``int64``, ``uint8``, ``int16``, or ``int8``.

    Examples
    --------
    >>> net = tlx.nn.Input([10, 200])
    >>> net = tlx.nn.Tanh()(net)

    Returns
    -------
    Tensor
        A ``Tensor`` in the same type as ``x``.

    """

    def __init__(self):
        super(Tanh, self).__init__()
        self._built = True
        self._tanh = tlx.ops.Tanh()

    def forward(self, x):
        outputs = self._tanh(x)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(x, outputs)
            self._nodes_fixed = True
        return outputs


class Softplus(Module):
    """This function is Softplus.

    The function return the following results:
      - softplus(x) = log(exp(x) + 1).

    Parameters
    ----------
    x : Tensor
        Support input type ``float``, ``double``, ``int32``, ``int64``, ``uint8``, ``int16``, or ``int8``.

    Examples
    --------
    >>> net = tlx.nn.Input([10, 200])
    >>> net = tlx.nn.Softplus()(net)

    Returns
    -------
    Tensor
        A ``Tensor`` in the same type as ``x``.

    """

    def __init__(self):
        super(Softplus, self).__init__()
        self._built = True
        self._softplus = tlx.ops.Softplus()

    def forward(self, x):
        outputs = self._softplus(x)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(x, outputs)
            self._nodes_fixed = True
        return outputs


class ReLU6(Module):
    """This function is ReLU6.

    The function return the following results:
      - ReLU6(x)=min(max(0,x),6)

    Parameters
    ----------
    x : Tensor
        Support input type ``float``, ``double``, ``int32``, ``int64``, ``uint8``, ``int16``, or ``int8``.

    Examples
    --------
    >>> net = tlx.nn.Input([10, 200])
    >>> net = tlx.nn.ReLU6()(net)

    Returns
    -------
    Tensor
        A ``Tensor`` in the same type as ``x``.

    """

    def __init__(self):
        super(ReLU6, self).__init__()
        self._built = True
        self._relu6 = tlx.ops.ReLU6()

    def forward(self, x):
        outputs = self._relu6(x)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(x, outputs)
            self._nodes_fixed = True
        return outputs


class ReLU(Module):
    """This function is ReLU.

    The function return the following results:
      - When x < 0: ``f(x) = 0``.
      - When x >= 0: ``f(x) = x``.

    Parameters
    ----------
    x : Tensor
        Support input type ``float``, ``double``, ``int32``, ``int64``, ``uint8``, ``int16``, or ``int8``.

    Examples
    --------
    >>> net = tlx.nn.Input([10, 200])
    >>> net = tlx.nn.ReLU()(net)

    Returns
    -------
    Tensor
        A ``Tensor`` in the same type as ``x``.

    """

    def __init__(self):
        super(ReLU, self).__init__()
        self._built = True
        self._relu = tlx.ops.ReLU()

    def forward(self, x):
        outputs = self._relu(x)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(x, outputs)
            self._nodes_fixed = True
        return outputs


class LeakyReLU(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{LeakyReLU}(x) = \max(0, x) + negative\_slope * \min(0, x)

    Parameters
    ----------
    negative_slope : float
        Controls the angle of the negative slope. Default: 1e-2
    name : str
        The function name (optional).

    Examples
    --------
    >>> net = tlx.nn.Input([10, 200])
    >>> net = tlx.nn.LeakyReLU(alpha=0.5)(net)

    Returns
    -------
    Tensor
        A ``Tensor`` in the same type as ``x``.

    References
    ----------
    - `Rectifier Nonlinearities Improve Neural Network Acoustic Models [A. L. Maas et al., 2013] <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`__

    """

    def __init__(self, negative_slope=0.01):
        super(LeakyReLU, self).__init__()
        self._built = True
        self.negative_slope = negative_slope
        self._leakyrelu = tlx.ops.LeakyReLU(negative_slope=self.negative_slope)

    def forward(self, x):
        outputs = self._leakyrelu(x)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(x, outputs)
            self._nodes_fixed = True
        return outputs


class LeakyReLU6(Module):
    r"""
        This activation function is a modified version :func:`leaky_relu` introduced by the following paper:
        `Rectifier Nonlinearities Improve Neural Network Acoustic Models [A. L. Maas et al., 2013] <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`__

        This activation function also follows the behaviour of the activation function :func:`tf.ops.relu6` introduced by the following paper:
        `Convolutional Deep Belief Networks on CIFAR-10 [A. Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`__

        The function return the following results:
          - When x < 0: ``f(x) = alpha_low * x``.
          - When x in [0, 6]: ``f(x) = x``.
          - When x > 6: ``f(x) = 6``.

        Parameters
        ----------
        x : Tensor
            Support input type ``float``, ``double``, ``int32``, ``int64``, ``uint8``, ``int16``, or ``int8``.
        alpha : float
            Slope.
        name : str
            The function name (optional).

        Examples
        --------
        >>> net = tlx.nn.Input([10, 200])
        >>> net = tlx.nn.LeakyReLU6(alpha=0.5)(net)

        Returns
        -------
        Tensor
            A ``Tensor`` in the same type as ``x``.

        References
        ----------
        - `Rectifier Nonlinearities Improve Neural Network Acoustic Models [A. L. Maas et al., 2013] <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`__
        - `Convolutional Deep Belief Networks on CIFAR-10 [A. Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`__
    """

    def __init__(self, alpha=0.2):
        super(LeakyReLU6, self).__init__()
        self._built = True
        if not (0 < alpha <= 1):
            raise ValueError("`alpha` value must be in [0, 1]`")

        self.alpha = alpha
        self.minimum = tlx.ops.Minimum()
        self.maximum = tlx.ops.Maximum()

    def forward(self, x):
        outputs = self.minimum(self.maximum(x, self.alpha * x), 6)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(x, outputs)
            self._nodes_fixed = True
        return outputs


class LeakyTwiceRelu6(Module):
    """

        This activation function is a modified version :func:`leaky_relu` introduced by the following paper:
        `Rectifier Nonlinearities Improve Neural Network Acoustic Models [A. L. Maas et al., 2013] <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`__

        This activation function also follows the behaviour of the activation function :func:`tf.ops.relu6` introduced by the following paper:
        `Convolutional Deep Belief Networks on CIFAR-10 [A. Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`__

        This function push further the logic by adding `leaky` behaviour both below zero and above six.

        The function return the following results:
          - When x < 0: ``f(x) = alpha_low * x``.
          - When x in [0, 6]: ``f(x) = x``.
          - When x > 6: ``f(x) = 6 + (alpha_high * (x-6))``.

        Parameters
        ----------
        x : Tensor
            Support input type ``float``, ``double``, ``int32``, ``int64``, ``uint8``, ``int16``, or ``int8``.
        alpha_low : float
            Slope for x < 0: ``f(x) = alpha_low * x``.
        alpha_high : float
            Slope for x < 6: ``f(x) = 6 (alpha_high * (x-6))``.
        name : str
            The function name (optional).

        Examples
        --------
        >>> net = tlx.nn.Input([10, 200])
        >>> net = tlx.nn.LeakyTwiceRelu6(alpha_low=0.5, alpha_high=0.2)(net)

        Returns
        -------
        Tensor
            A ``Tensor`` in the same type as ``x``.

        References
        ----------
        - `Rectifier Nonlinearities Improve Neural Network Acoustic Models [A. L. Maas et al., 2013] <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`__
        - `Convolutional Deep Belief Networks on CIFAR-10 [A. Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`__

    """

    def __init__(self, alpha_low=0.2, alpha_high=0.2):
        super(LeakyTwiceRelu6, self).__init__()
        self._built = True
        if not (0 < alpha_high <= 1):
            raise ValueError("`alpha_high` value must be in [0, 1]`")

        if not (0 < alpha_low <= 1):
            raise ValueError("`alpha_low` value must be in [0, 1]`")

        self.alpha_low = alpha_low
        self.alpha_high = alpha_high
        self.minimum = tlx.ops.Minimum()
        self.maximum = tlx.ops.Maximum()

    def forward(self, x):
        x_is_above_0 = self.minimum(x, 6 * (1 - self.alpha_high) + self.alpha_high * x)
        x_is_below_0 = self.minimum(self.alpha_low * x, 0)
        outputs = self.maximum(x_is_above_0, x_is_below_0)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(x, outputs)
            self._nodes_fixed = True
        return outputs


class Swish(Module):
    """Swish function.

         See `Swish: a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941>`__.

        Parameters
        ----------
        x : Tensor
            input.
        name: str
            function name (optional).

        Examples
        --------
        >>> net = tlx.nn.Input([10, 200])
        >>> net = tlx.nn.Swish()(net)

        Returns
        -------
        Tensor
            A ``Tensor`` in the same type as ``x``.

    """

    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = tlx.ops.Sigmoid()
        self._built = True

    def forward(self, x):
        outputs = self.sigmoid(x) * x

        if not self._nodes_fixed and self._build_graph:
            self._add_node(x, outputs)
            self._nodes_fixed = True
        return outputs


class HardTanh(Module):
    """Hard tanh activation function.

        Which is a ramp function with low bound of -1 and upper bound of 1, shortcut is `htanh`.

        Parameters
        ----------
        x : Tensor
            input.
        name : str
            The function name (optional).

        Examples
        --------
        >>> net = tlx.nn.Input([10, 200])
        >>> net = tlx.nn.HardTanh()(net)

        Returns
        -------
        Tensor
            A ``Tensor`` in the same type as ``x``.

    """

    def __init__(self):
        super(HardTanh, self).__init__()
        self._built = True

    def forward(self, x):
        outputs = tlx.ops.clip_by_value(x, -1, 1)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(x, outputs)
            self._nodes_fixed = True
        return outputs


class Mish(Module):
    r"""Applies the Mish function, element-wise. Mish: A Self Regularized Non-Monotonic Neural Activation Function.

    .. math::
        \text{Mish}(x) = x * \text{Tanh}(\text{Softplus}(x))

    Reference: [Mish: A Self Regularized Non-Monotonic Neural Activation Function .Diganta Misra, 2019]<https://arxiv.org/abs/1908.08681>

    Parameters
    ----------
    x : Tensor
        input.

    Examples
    --------
    >>> net = tlx.nn.Input([10, 200])
    >>> net = tlx.nn.Mish()(net)

    Returns
    -------
    Tensor
        A ``Tensor`` in the same type as ``x``.

    """

    def __init__(self):
        super(Mish, self).__init__()
        self._tanh = tlx.ops.Tanh()
        self._softplus = tlx.ops.Softplus()
        self._built = True

    def forward(self, x):
        outputs = x * self._tanh(self._softplus(x))

        if not self._nodes_fixed and self._build_graph:
            self._add_node(x, outputs)
            self._nodes_fixed = True
        return outputs

class LogSoftmax(Module):
    r"""Applies a softmax followed by a logarithm.

    .. math::
        \text{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right)


    Parameters
    ----------
    x : Tensor
        input.
    dim : int
        A dimension along which LogSoftmax will be computed.

    Examples
    --------
    >>> net = tlx.nn.Input([10, 200])
    >>> net = tlx.nn.LogSoftmax()(net)

    Returns
    -------
    Tensor
        A ``Tensor`` in the same type as ``x``.

    """

    def __init__(self, dim = None):
        super(LogSoftmax, self).__init__()
        self.dim = dim
        self._built = True

    def forward(self, x):
        outputs = tlx.ops.logsoftmax(x, self.dim)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(x, outputs)
            self._nodes_fixed = True
        return outputs

class HardSigmoid(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            1 & \text{if~} x \ge +3, \\
            x / 6 + 1 / 2 & \text{otherwise}
        \end{cases}


    Parameters
    ----------
    x : Tensor
        input.

    Examples
    --------
    >>> net = tlx.nn.Input([10, 200])
    >>> net = tlx.nn.HardSigmoid()(net)

    Returns
    -------
    Tensor
        A ``Tensor`` in the same type as ``x``.

    """

    def __init__(self):
        super(HardSigmoid, self).__init__()
        self._built = True

    def forward(self, x):
        outputs = tlx.ops.hardsigmoid(x)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(x, outputs)
            self._nodes_fixed = True
        return outputs

class Hardswish(Module):
    r"""Applies the hardswish function, element-wise, as described in the paper:

    `Searching for MobileNetV3`_.

    .. math::
        \text{Hardswish}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            x & \text{if~} x \ge +3, \\
            x \cdot (x + 3) /6 & \text{otherwise}
        \end{cases}


    Parameters
    ----------
    x : Tensor
        input.

    Examples
    --------
    >>> net = tlx.nn.Input([10, 200])
    >>> net = tlx.nn.Hardswish()(net)

    Returns
    -------
    Tensor
        A ``Tensor`` in the same type as ``x``.

    """

    def __init__(self):
        super(Hardswish, self).__init__()
        self._built = True

    def forward(self, x):
        outputs = tlx.ops.hardswish(x)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(x, outputs)
            self._nodes_fixed = True
        return outputs