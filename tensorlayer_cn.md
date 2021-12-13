TensorLayer3.0是一款兼容多种深度学习框架后端的深度学习库，支持TensorFlow, MindSpore, PaddlePaddle为后端计算引擎。TensorLayer3.0使用方式简单，并且在选定运算后端后能和该后端的算子混合使用。TensorLayer3.0提供了数据处理、模型构建、模型训练等深度学习全流程API，同一套代码可以通过一行代码切换后端，减少框架之间算法迁移需要重构代码的繁琐工作。

## 一、TensorLayer安装

TensorLayer安装前置条件包括TensorFlow, numpy, matplotlib等，如果你需要使用GPU加速还需要安装CUDA和cuDNN。

### 1.1 安装后端

TensorLayer支持多种后端，默认为TensorFlow，也支持MindSpore和PaddlePaddle，PaddlePaddle目前只支持少量Layer，后续新版本中会持续更新。

安装TensorFlow

```python
pip3 install tensorflow-gpu # GPU version
pip3 install tensorflow # CPU version
```
如果你想使用MindSpore后端还需要安装MindSpore1.2.0,下面给出了MindSpore1.2.0GPU版本的安装，如果需要安装CPU或者Ascend可以参考MindSpore官网。
```python
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.2.1/MindSpore/gpu/ubuntu_x86/cuda-10.1/mindspore_gpu-1.2.1-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```
如果想要使用PaddlePaddle后端，还需要安装PaddlePaddle2.0，下面给出了PaddlePaddle2.0GPU版本的安装，其他平台请参考PaddlePaddle官网。
```python
python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```
### 1.2 安装TensorLayer

通过PIP安装稳定版本

```plain
pip install tensorlayer3
pip install tensorlayer3 -i https://pypi.tuna.tsinghua.edu.cn/simple  (faster in China)
```
如果要获得最新开发版本可以通过下载源码安装
```plain
pip3 install git+https://git.openi.org.cn/TensorLayer/tensorlayer3.0.git
```
## 二、TensorLayer3.0特性

TensorLayer3.0版本主要设计目标如下，我们将会支持TensorFlow, Pytorch, MindSpore, PaddlePaddle作为计算引擎。在API层提供深度学习模型构建组件(Layers)，数据处理(DataFlow), 激活函数（Activations），参数初始化函数，代价函数，模型优化函数以及一些常用操作函数。在最上层我们利用TensorLayer开发了一些例子和预训练模型。
![](https://content.markdowner.net/pub/BMDV5X-aoB1EmR)

### 2.1 可扩展性

TensorLayer3.0的相比于TensorLayer2.0之前的版本对后端进行了解耦。在之前的版本中我们设计的Layer直接使用了TensorFlow的算子，这为后续扩展后端带来不便。为此在新的版本中，我们将所有后端算子均封装在backend层，并且对不同框架之间的接口进行了统一，在构建Layer时均调用统一的接口来达到兼容多框架的目的。

### 2.2 简易性

TensorLayer3.0使用简单，我们设计了两种构建模型的方式，对于顺序连贯的模型我们提供了SequentialLayer来构建，对于复杂模型可以通过SubClass的方式继承Module来构建。在TensorLayer3.0中构建的网络模型可以当成Layer在__init__中初始化在forward中调用。TensorLayer3.0构建网络时可以无需计算上一层的输出（不用输入in_channels参数），通过最后init_build操作来完成参数初始化自动推断模型输出大小。

### 2.3 兼容性

TensorLayer3.0构建的模型能直接在TensorFlow, MindSpore, PaddlePaddle中使用，可以混合对应框架的算子进行使用。例如用TensorLayer搭建网络，使用TensorFlow后端，那么在数据处理和模型训练时可以直接用TensorFlow提供的算子完成。

## 三、数据集加载

TensorLayer内置了一些常见的数据集例如mnist, cifar10。这里加载手写数字识别数据集，用来模型训练和评估。

```python
import tensorlayer as tl
X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))
```
## 四、数据预处理

TensorLayer提供了大量数据处理操作，也可以直接使用对应框架数据处理操作完成你的数据构建。

Tensorlayer目前拥有完善的图像预处理操作。为了满足开发者习惯，集成以TensorFlow、MindSpore、PaddlePaddle为后端的图像算子。图像算子主要基于各框架本身tensor操作以及PIL、opencv库完成，并且能够自动根据全局后端环境变量将图像矩阵数据转换为后端框架对应的数据格式。为了图像算子在各框架后端保持一致，TensorLayer综合考虑TensorFlow、Mindspore、PaddlePaddle框架各自图像算子功能及参数，增加和调整不同后端框架源码扩展了图像处理功能。以PyTorch为后端的图像算子将在未来开发中更新。

TensorLayer的图像数据预处理例子如下：

```python
import tensorlayer as tl
import numpy as np
image=(np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
transform=tl.vision.transforms.Resize(size(100,100),interpolation='bilinear')
image=transform(image)
print(image.shape)
#image shape:(100, 100, 3)
image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
transform=tl.vision.transforms.Pad(padding=10,padding_value=0,mode='constant')
image = transform(image)
print(image.shape)
#image shape : (244, 244, 3)
```
## 五、模型构建

### 5.1 SequentialLayer构建

针对有顺序的线性网络结构，你可以通过SequentialLayer来快速构建模型，可以减少定义网络等代码编写，具体如下：我们构建一个多层感知机模型。

```python
import tensorlayer as tl
from tensorlayer.layers import Dense
layer_list = []
layer_list.append(Dense(n_units=800, act=tl.ReLU, in_channels=784, name='Dense1'))
layer_list.append(Dense(n_units=800, act=tl.ReLU, in_channels=800, name='Dense2'))
layer_list.append(Dense(n_units=10, act=tl.ReLU, in_channels=800, name='Dense3'))
MLP = SequentialLayer(layer_list)
```
### 5.2 继承基类Module构建

针对较为复杂的网络，可以使用Module子类定义的方式来进行模型构建，在__init__对Layer进行声明，在forward里使用声明的Layer进行前向计算。这种方式中声明的Layer可以进行复用，针对相同的Layer构造一次，在forward可以调用多次。同样我们构建一个多层感知机模型。

```python
import tensorlayer as tl
from tensorlayer.layers import Module, Dropout, Dense
class MLP(Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.dense1 = Dense(n_units=800, act=tl.ReLU, in_channels=784)
        self.dense2 = Dense(n_units=800, act=tl.ReLU, in_channels=800)
        self.dense3 = Dense(n_units=10, act=tl.ReLU, in_channels=800)
    def forward(self, x):
        z = self.dense1(z)
        z = self.dense2(z)
        out = self.dense3(z)
        return out
```
### 5.3 构建复杂网络结构

在构建网络时，我们经常遇到一些模块重复使用多次，可以通过循环来构建。

例如在网络中需要将感知机当成一个Block，并且使用三次，我们先定义要多次调用的Block

```python
import tensorlayer as tl
from tensorlayer.layers import Module, Dense, Elementwise
class Block(Module):
    def __init__(self, in_channels):
        super(Block, self).__init__()
        self.dense1 = Dense(in_channels=in_channels, n_units=256)
        self.dense2 = Dense(in_channels=256, n_units=384)
        self.dense3 = Dense(in_channels=in_channels, n_units=384)
        self.concat = Elementwise(combine_fn=tl.ops.add)
        
    def forward(self, inputs):
        z = self.dense1(inputs)
        z1 = self.dense2(z)
        z2 = self.dense3(inputs)
        out = self.concat([z1, z2])
        return out
```
定义好Block后我们通过SequentialLayer和Module构建网络
```python
class CNN(Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.flatten = Flatten(name='flatten')
        self.dense1 = Dense(384, act=tl.ReLU, in_channels=2304)
        self.dense_add = self.make_layer(in_channel=384)
        self.dense2 = Dense(192, act=tl.ReLU, n_channels=384)
        self.dense3 = Dense(10, act=None, in_channels=192)
        
    def forward(self, x):
        z = self.flatten(z)
        z = self.dense1(z)
        z = self.dense_add(z)
        z = self.dense2(z)
        z = self.dense3(z)
        return z
        
    def make_layer(self, in_channel):
        layers = []
        _block = Block(in_channel)
        layers.append(_block)
        for _ in range(1, 3):
            range_block = Block(in_channel)
            layers.append(range_block)
        return SequentialLayer(layers)
```
### 5.4 自动推断上一层输出大小

我们构建网络时经常需要手动输入上一层的输出大小，作为下一层的输入，也就是每个Layer中的in_channels参数。在TensoLayer中也可以无需输入in_channels，构建网络后给定网络的输入调用一次参数初始化即可。

```python
import tensorlayer as tl
from tensorlayer.layers import Module, Dense
class CustomModel(Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.dense1 = Dense(n_units=800)
        self.dense2 = Dense(n_units=800, act=tl.ReLU)
        self.dense3 = Dense(n_units=10, act=tl.ReLU)
        
    def forward(self, x):
        z = self.dense1(z)
        z = self.dense2(z)
        out = self.dense3(z)
        return out
MLP = CustomModel()
input = tl.layers.Input(shape=(1, 784))
MLP.init_build(input)
```
## 六、模型训练

TensorLayer提供了模型训练模块，可以直接调用进行训练。TensorLayer构建的模型也能支持在其他框架中直接使用，如用TensorLayer构建MLP模型，使用的是TensorFlow后端，那么可以使用TensoFlow的算子完成模型训练。

### 6.1 调用模型训练模块训练

调用封装好的models模块进行训练。

```python
import tensorlayer as tl
optimizer = tl.optimizers.Momentum(0.05, 0.9)
model = tl.models.Model(network=MLP, loss_fn=tl.cost.softmax_cross_entropy_with_logits, optimizer=optimizer)
model.train(n_epoch=500, train_dataset=train_ds, print_freq=2)
```
### 6.2 混合对应框架算子进行训练

混合TensorFlow进行训练。下面例子中optimizer和loss均可以使用TensorFlow的算子

```python
import tensorlayer as tl
import tensorflow as tf
optimizer = tl.optimizers.Momentum(0.05, 0.9)
# optimizer = tf.optimizers.Momentum(0.05, 0.9)
for epoch in range(n_epoch):
    for X_batch, y_batch in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
        MLP.set_train()
        with tf.GradientTape() as tape:
            _logits = MLP(X_batch)
            _loss = tl.cost.softmax_cross_entropy_with_logits(_logits, y_batch)
        grad = tape.gradient(_loss, train_weights)
        optimizer.apply_gradients(zip(grad, train_weights))
```

## 七、完整实例

同一套代码通过设置后端进行切换不同后端训练，无需修改代码。在os.environ['TL_BACKEND']中可以设置为'tensorflow’,'mindspore', 'paddle'。

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ['TL_BACKEND'] = 'tensorflow'
# os.environ['TL_BACKEND'] = 'mindspore'
# os.environ['TL_BACKEND'] = 'paddle'
import numpy as np
import tensorlayer as tl
from tensorlayer.layers import Module
from tensorlayer.layers import Dense, Dropout
X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))
class CustomModel(Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.dropout1 = Dropout(keep=0.8)
        self.dense1 = Dense(n_units=800, act=tl.ReLU, in_channels=784)
        self.dropout2 = Dropout(keep=0.8)
        self.dense2 = Dense(n_units=800, act=tl.ReLU, in_channels=800)
        self.dropout3 = Dropout(keep=0.8)
        self.dense3 = Dense(n_units=10, act=tl.ReLU, in_channels=800)
    def forward(self, x, foo=None):
        z = self.dropout1(x)
        z = self.dense1(z)
        # z = self.bn(z)
        z = self.dropout2(z)
        z = self.dense2(z)
        z = self.dropout3(z)
        out = self.dense3(z)
        if foo is not None:
            out = tl.ops.relu(out)
        return out
def generator_train():
    inputs = X_train
    targets = y_train
    if len(inputs) != len(targets):
        raise AssertionError("The length of inputs and targets should be equal")
    for _input, _target in zip(inputs, targets):
        yield (_input, np.array(_target))
MLP = CustomModel()
n_epoch = 50
batch_size = 128
print_freq = 2
shuffle_buffer_size = 128
train_weights = MLP.trainable_weights
optimizer = tl.optimizers.Momentum(0.05, 0.9)
train_ds = tl.dataflow.FromGenerator(
    generator_train, output_types=(tl.float32, tl.int32) , column_names=['data', 'label']
)
train_ds = tl.dataflow.Shuffle(train_ds,shuffle_buffer_size)
train_ds = tl.dataflow.Batch(train_ds,batch_size)
model = tl.models.Model(network=MLP, loss_fn=tl.cost.softmax_cross_entropy_with_logits, optimizer=optimizer)
model.train(n_epoch=n_epoch, train_dataset=train_ds, print_freq=print_freq, print_train_batch=False)
model.save_weights('./model.npz', format='npz_dict')
model.load_weights('./model.npz', format='npz_dict')
```
## 八、预训练模型

在TensorLayer中我们将持续提供了丰富的预训练模型，和应用。例如VGG16, VGG19, ResNet50, YOLOv4.下面例子展示了在MS-COCO数据集中利用YOLOv4进行目标检测，对应预训练模型和数据可以从examples/model_zoo中找到。

```python
import numpy as np
import cv2
from PIL import Image
from examples.model_zoo.common import yolo4_input_processing, yolo4_output_processing, \
    result_to_json, read_class_names, draw_boxes_and_labels_to_image_with_json
from examples.model_zoo.yolo import YOLOv4
import tensorlayer as tl
tl.logging.set_verbosity(tl.logging.DEBUG)
INPUT_SIZE = 416
image_path = './data/kite.jpg'
class_names = read_class_names('./model/coco.names')
original_image = cv2.imread(image_path)
image = cv2.cvtColor(np.array(original_image), cv2.COLOR_BGR2RGB)
model = YOLOv4(NUM_CLASS=80, pretrained=True)
model.set_eval()
batch_data = yolo4_input_processing(original_image)
feature_maps = model(batch_data)
pred_bbox = yolo4_output_processing(feature_maps)
json_result = result_to_json(image, pred_bbox)
image = draw_boxes_and_labels_to_image_with_json(image, json_result, class_names)
image = Image.fromarray(image.astype(np.uint8))
image.show()
```
## 九、自定义Layer

在TensorLayer中自定以Layer需要继承Module，在build中我们对训练参数进行定义，在forward中我们定义前向计算。下面给出用TensorFlow后端时，定义全连接层$$a=f(x*W + b)$$如果你想定义其他后端的Dense需要将算子换成对应后端。

如果要定义一个通用的Layer则要把算子接口进行统一后封装在backend中，具体可以参考tensorlayer/layers中的Layer。

```python
from tensorlayer.layers import Module
class Dense(Module):
  def __init__(
          self,
          n_units,
          act=None,  
          name=None, 
          in_channels = None
  ):
      super(Dense, self).__init__(name, act=act)
      self.n_units = n_units
      self.in_channels = in_channels
      self.build()
      self._built = True
  def build(self): # initialize the model weights here
      shape = [self.in_channels, self.n_units]
      self.W = self._get_weights("weights", shape=tuple(shape), init=self.W_init)
      self.b = self._get_weights("biases", shape=(self.n_units, ), init=self.b_init)
  def forward(self, inputs): # call function
      z = tf.matmul(inputs, self.W) + self.b
      if self.act: # is not None
          z = self.act(z)
      return z
```

