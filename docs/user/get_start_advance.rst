.. _getstartadvance:

==================
Advanced features
==================


Customizing layer
==================

Layers with weights
----------------------

The fully-connected layer is `a = f(x*W+b)`, the most simple implementation is as follow.

.. code-block:: python

  import tensorlayerx as tlx
  from tensorlayerx.nn import Module

  class Linear(Module):
    """The :class:`Linear` class is a fully connected layer.
    
    Parameters
    ----------
    out_features : int
        The number of units of this layer.
    act : activation function
        The activation function of this layer.
    W_init : initializer or str
        The initializer for the weight matrix.
    b_init : initializer or None or str
        The initializer for the bias vector. If None, skip biases.
    in_features: int
        The number of channels of the previous layer.
        If None, it will be automatically detected when the layer is forwarded for the first time.
    name : None or str
        A unique layer name. If None, a unique name will be automatically generated.
    """
    
    def __init__(
        self,
        out_features,
        act=None,
        W_init='truncated_normal',
        b_init='constant',
        in_features=None,
        name=None,  # 'linear',
    ):
        super(Linear, self).__init__(name, act=act) # auto naming, linear_1, linear_2 ...
        self.out_features = out_features
        self.in_features = in_features
        self.W_init = self.str_to_init(W_init)
        self.b_init = self.str_to_init(b_init)
        self.build()
        self._built = True
        
    def build(self): # initialize the model weights here
        shape = [self.in_features, self.out_features]
        self.W = self._get_weights("weights", shape=tuple(shape), init=self.W_init)
        self.b = self._get_weights("biases", shape=(self.out_features, ), init=self.b_init)

    def forward(self, inputs): # call function
        z = tlx.matmul(inputs, self.W) + self.b
        if self.act: # is not None
            z = self.act(z)
        return z

The full implementation is as follow, which supports both automatic inference input and dynamic models and allows users to control whether to use the bias, how to initialize the weight values.

.. code-block:: python


  class Linear(Module):
    """The :class:`Linear` class is a fully connected layer.

    Parameters
    ----------
    out_features : int
        The number of units of this layer.
    act : activation function
        The activation function of this layer.
    W_init : initializer or str
        The initializer for the weight matrix.
    b_init : initializer or None or str
        The initializer for the bias vector. If None, skip biases.
    in_features: int
        The number of channels of the previous layer.
        If None, it will be automatically detected when the layer is forwarded for the first time.
    name : None or str
        A unique layer name. If None, a unique name will be automatically generated.

    Examples
    --------
    With TensorLayerx

    >>> net = tlx.nn.Input([100, 50], name='input')
    >>> linear = tlx.nn.Linear(out_features=800, act=tlx.nn.ReLU, in_features=50, name='linear_1')
    >>> tensor = tlx.nn.Linear(out_features=800, act=tlx.nn.ReLU, name='linear_2')(net)

    Notes
    -----
    If the layer input has more than two axes, it needs to be flatten by using :class:`Flatten`.

    """

    def __init__(
        self,
        out_features,
        act=None,
        W_init='truncated_normal',
        b_init='constant',
        in_features=None,
        name=None,  # 'linear',
    ):

        super(Linear, self).__init__(name, act=act)

        self.out_features = out_features
        self.W_init = self.str_to_init(W_init)
        self.b_init = self.str_to_init(b_init)
        self.in_features = in_features

        if self.in_features is not None:
            self.build(self.in_features)
            self._built = True

        logging.info(
            "Linear  %s: %d %s" %
            (self.name, self.out_features, self.act.__class__.__name__ if self.act is not None else 'No Activation')
        )

    def __repr__(self):
        actstr = self.act.__class__.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(out_features={out_features}, ' + actstr)
        if self.in_features is not None:
            s += ', in_features=\'{in_features}\''
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if self.in_features is None and len(inputs_shape) < 2:
            raise AssertionError("The dimension of input should not be less than 2")
        if self.in_features:
            shape = [self.in_features, self.out_features]
        else:
            self.in_features = inputs_shape[-1]
            shape = [self.in_features, self.out_features]

        self.W = self._get_weights("weights", shape=tuple(shape), init=self.W_init)

        self.b_init_flag = False
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.out_features, ), init=self.b_init)
            self.b_init_flag = True
            self.bias_add = tlx.ops.BiasAdd(data_format='NHWC')

        self.act_init_flag = False
        if self.act:
            self.act_init_flag = True

        self.matmul = tlx.ops.MatMul()

    def forward(self, inputs):
        if self._forward_state == False:
            if self._built == False:
                self.build(tlx.get_tensor_shape(inputs))
                self._built = True
            self._forward_state = True

        z = self.matmul(inputs, self.W)
        if self.b_init_flag:
            z = self.bias_add(z, self.b)
        if self.act_init_flag:
            z = self.act(z)
        return z



Layers with train/test modes
------------------------------

We use Dropout as an example here:

.. code-block:: python
  
  class Dropout(Module):
    """
    The :class:`Dropout` class is a noise layer which randomly set some
    activations to zero according to a probability.

    Parameters
    ----------
    p : float
        probability of an element to be zeroed. Default: 0.5
    seed : int or None
        The seed for random dropout.
    name : None or str
        A unique layer name.

    Examples
    --------
    >>> net = tlx.nn.Input([10, 200])
    >>> net = tlx.nn.Dropout(p=0.2)(net)

    """

    def __init__(self, p=0.5, seed=0, name=None):  #"dropout"):
        super(Dropout, self).__init__(name)
        self.p = p
        self.seed = seed

        self.build()
        self._built = True

        logging.info("Dropout %s: p: %f " % (self.name, self.p))

    def __repr__(self):
        s = ('{classname}(p={p}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        self.dropout = tlx.ops.Dropout(p=self.p, seed=self.seed)

    def forward(self, inputs):
        if self.is_train:
            outputs = self.dropout(inputs)
        else:
            outputs = inputs
        return outputs

Pre-trained CNN
================

Get entire CNN
---------------

.. code-block:: python


  import tensorlayerx as tlx
  import numpy as np
  from tensorlayerx.models.imagenet_classes import class_names
  from examples.model_zoo import vgg16

  vgg = vgg16(pretrained=True)
  img = tlx.vision.load_image('data/tiger.jpeg')
  img = tlx.utils.functional.resize(img, (224, 224), method='bilinear')
  img = tlx.ops.convert_to_tensor(img, dtype = 'float32') / 255.
  output = vgg(img, is_train=False)

