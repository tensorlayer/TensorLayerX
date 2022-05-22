.. _getstartmodel:

===============
定义一个模型
===============

TensorLayerX 提供了两种定义模型的方式。
序列式模型可以让您顺畅的搭建一个模型，而动态模型则可以让您全面掌控模型前向传播过程。

序列式模型
===================

.. code-block:: python

  from tensorlayerx.nn import SequentialLayer
  from tensorlayerx.nn import Linear
  import tensorlayerx as tlx

  def get_model():
      layer_list = []
      layer_list.append(Linear(out_features=800, act=tlx.ReLU, in_features=784, name='Linear1'))
      layer_list.append(Linear(out_features=800, act=tlx.ReLU, in_features=800, name='Linear2'))
      layer_list.append(Linear(out_features=10, act=tlx.ReLU, in_features=800, name='Linear3'))
      MLP = SequentialLayer(layer_list)
      return MLP



动态模型
=======================


这种情况下，您需要手动将上一层的输出形状提供给新的层。

.. code-block:: python

  import tensorlayerx as tlx
  from tensorlayerx.nn import Module
  from tensorlayerx.nn import Dropout, Linear
  class CustomModel(Module):

      def __init__(self):
          super(CustomModel, self).__init__()

          self.dropout1 = Dropout(p=0.2)
          self.linear1 = Linear(out_features=800, act=tlx.ReLU, in_features=784)
          self.dropout2 = Dropout(p=0.2)
          self.linear2 = Linear(out_features=800, act=tlx.ReLU, in_features=800)
          self.dropout3 = Dropout(p=0.2)
          self.linear3 = Linear(out_features=10, act=None, in_features=800)

      def forward(self, x, foo=False):
          z = self.dropout1(x)
          z = self.linear1(z)
          z = self.dropout2(z)
          z = self.linear2(z)
          z = self.dropout3(z)
          out = self.linear3(z)
          if foo:
              out = tlx.softmax(out)
          return out

  MLP = CustomModel()
  MLP.set_eval()
  outputs = MLP(data, foo=True) # controls the forward here
  outputs = MLP(data, foo=False)
  
  
动态模型不需要提供前一层的输出形状
=========================================================


这种情况下，您不需要手动将上一层的输出形状提供给新的层。

.. code-block:: python

  import tensorlayerx as tlx
  from tensorlayerx.nn import Module
  from tensorlayerx.nn import Dropout, Linear
  class CustomModel(Module):

      def __init__(self):
          super(CustomModel, self).__init__()

          self.dropout1 = Dropout(p=0.2)
          self.linear1 = Linear(out_features=800, act=tlx.ReLU)
          self.dropout2 = Dropout(p=0.2)
          self.linear2 = Linear(out_features=800, act=tlx.ReLU)
          self.dropout3 = Dropout(p=0.2)
          self.linear3 = Linear(out_features=10, act=None)

      def forward(self, x, foo=False):
          z = self.dropout1(x)
          z = self.linear1(z)
          z = self.dropout2(z)
          z = self.linear2(z)
          z = self.dropout3(z)
          out = self.linear3(z)
          if foo:
              out = tlx.softmax(out)
          return out

  MLP = CustomModel()
  MLP.init_build(tlx.nn.Input(shape=(1, 784))) # init_build must be called to initialize the weights.
  MLP.set_eval()
  outputs = MLP(data, foo=True) # controls the forward here
  outputs = MLP(data, foo=False)

切换训练/测试模式
=============================

.. code-block:: python

  # method 1: 在前向传播前切换
  MLP.set_train() # enable dropout, batch norm moving avg ...
  output = MLP(train_data)
  ... # training code here
  Model.set_eval()  # disable dropout, batch norm moving avg ...
  output = MLP(test_data)
  ... # testing code here
  
  # method 2: 使用封装的训练模块
  model = tlx.model.Model(network=MLP, loss_fn=tlx.losses.softmax_cross_entropy_with_logits, optimizer=optimizer)
  model.train(n_epoch=n_epoch, train_dataset=train_ds)

权重的复用
=======================

对于动态模型，只需要在前向传播函数中多次调用layer对象即可

.. code-block:: python

  import tensorlayerx as tlx
  from tensorlayerx.nn import Module, Linear, Concat
  class MyModel(Module):
      def __init__(self):
          super(MyModel, self).__init__()
          self.linear_shared = Linear(out_features=800, act=tlx.ReLU, in_features=784)
          self.linear1 = Linear(out_features=10, act=tlx.ReLU, in_features=800)
          self.linear2 = Linear(out_features=10, act=tlx.ReLU, in_features=800)
          self.cat = Concat()

      def forward(self, x):
          x1 = self.linear_shared(x) # call dense_shared twice
          x2 = self.linear_shared(x)
          x1 = self.linear1(x1)
          x2 = self.linear2(x2)
          out = self.cat([x1, x2])
          return out

  model = MyModel()

模型打印相关函数
=======================

.. code-block:: python

  print(MLP) # 只需要简单的调用print函数

  # Model(
  #   (_inputlayer): Input(shape=[None, 784], name='_inputlayer')
  #   (dropout): Dropout(p=0.8, name='dropout')
  #   (linear): Linear(out_features=800, relu, in_features='784', name='linear')
  #   (dropout_1): Dropout(p=0.8, name='dropout_1')
  #   (linear_1): Linear(out_features=800, relu, in_features='800', name='linear_1')
  #   (dropout_2): Dropout(p=0.8, name='dropout_2')
  #   (linear_2): Linear(out_features=10, None, in_features='800', name='linear_2')
  # )

访问指定权重
=======================

We can get the specific weights by indexing or naming.

.. code-block:: python

  # indexing
  all_weights = MLP.all_weights
  some_weights = MLP.all_weights[1:3]

保存和恢复模型
=======================

我们提供了两种模型保存和恢复的方式


仅保存权重参数
------------------

.. code-block:: python

  MLP.save_weights('./model_weights.npz') # by default, file will be in hdf5 format
  MLP.load_weights('./model_weights.npz')

保存模型权重
-----------------------------------------------

.. code-block:: python

  # 在使用封装的训练模块时，可用如下方式保存和加载模型。
  model = tlx.model.Model(network=MLP, loss_fn=tlx.losses.softmax_cross_entropy_with_logits, optimizer=optimizer)
  model.train(n_epoch=n_epoch, train_dataset=train_ds)
  model.save_weights('./model.npz', format='npz_dict')
  model.load_weights('./model.npz', format='npz_dict')

