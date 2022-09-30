# 使用卷积神经网络进行图像分类
**摘要:** 本示例教程将会演示如何使用TensorLayerX的卷积神经网络来完成图像分类任务。这是一个较为简单的示例，将会使用一个由三个卷积层组成的网络完成cifar10数据集的图像分类任务。

## 一、环境配置
本教程基于TensorLayerX 0.5.6 编写，如果你的环境不是本版本，请先参考官网[安装](https://tensorlayerx.readthedocs.io/en/latest/user/installation.html)。  
TensorlayerX目前支持包括TensorFlow、Pytorch、PaddlePaddle、MindSpore作为计算后端，指定计算后端的方法也非常简单，只需要设置环境变量即可
```{.python}
import os
os.environ['TL_BACKEND'] = 'paddle'
# os.environ['TL_BACKEND'] = 'tensorflow'
# os.environ['TL_BACKEND'] = 'mindspore'
# os.environ['TL_BACKEND'] = 'torch'
```

引入需要的模块
```{.python}
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import (Conv2d, Linear, Flatten, MaxPool2d, BatchNorm2d)

from tensorlayerx.dataflow import Dataset, DataLoader
from tensorlayerx.vision.transforms import (
    Compose, Resize, RandomFlipHorizontal, RandomContrast, RandomBrightness, StandardizePerImage, RandomCrop
)
```
## 二、加载数据集
本案例将会使用TensorLayerX提供的API完成数据集的下载并为后续的训练任务准备好数据迭代器。 
cifar10数据集由60000张大小为32 * 32的彩色图片组成，其中有50000张图片组成了训练集，另外10000张图片组成了测试集。这些图片分为10个类别，将训练一个模型能够把图片进行正确的分类。

```{.python}
# prepare cifar10 data
X_train, y_train, X_test, y_test = tlx.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)

class CIFAR10Dataset(Dataset):

    def __init__(self, data, label, transforms):
        self.data = data
        self.label = label
        self.transforms = transforms

    def __getitem__(self, idx):
        x = self.data[idx].astype('uint8')
        y = self.label[idx].astype('int64')
        x = self.transforms(x)

        return x, y

    def __len__(self):

        return len(self.label)

#设置数据处理函数
train_transforms = Compose(
    [
        RandomCrop(size=[24, 24]),
        RandomFlipHorizontal(),
        RandomBrightness(brightness_factor=(0.5, 1.5)),
        RandomContrast(contrast_factor=(0.5, 1.5)),
        StandardizePerImage()
    ]
)

test_transforms = Compose([Resize(size=(24, 24)), StandardizePerImage()])

#构建数据集和加载器
train_dataset = CIFAR10Dataset(data=X_train, label=y_train, transforms=train_transforms)
test_dataset = CIFAR10Dataset(data=X_test, label=y_test, transforms=test_transforms)

```

# 三、组建网络
接下来使用TensorLayerX定义一个使用了三个二维卷积（ `Conv2D` ) 且每次卷积之后使用 `relu` 激活函数，两个二维池化层（ `MaxPool2D` ），和两个线性变换层组成的分类网络，来把一个`(32, 32, 3)`形状的图片通过卷积神经网络映射为10个输出，这对应着10个分类的类别。

```{.python}
class CNN(Module):

    def __init__(self):
        super(CNN, self).__init__()
        # weights init
        W_init = tlx.nn.initializers.truncated_normal(stddev=5e-2)
        W_init2 = tlx.nn.initializers.truncated_normal(stddev=0.04)
        b_init2 = tlx.nn.initializers.constant(value=0.1)

        self.conv1 = Conv2d(32, (3, 3), (1, 1), padding='SAME', W_init=W_init, b_init=None, name='conv1', in_channels=3)
        self.maxpool1 = MaxPool2d((2, 2), (2, 2), padding='SAME', name='pool1')

        self.conv2 = Conv2d(
            64, (3, 3), (1, 1), padding='SAME', act=tlx.nn.ReLU, W_init=W_init, b_init=None, name='conv2', in_channels=32
        )

        self.conv3 = Conv2d(
            64, (3, 3), (1, 1), padding='SAME', act=tlx.nn.ReLU, W_init=W_init, b_init=None, name='conv3', in_channels=64
        )
        self.maxpool2 = MaxPool2d((2, 2), (2, 2), padding='SAME', name='pool2')

        self.flatten = Flatten(name='flatten')
        self.linear1 = Linear(1024, act=tlx.nn.ReLU, W_init=W_init2, b_init=b_init2, name='linear1relu', in_features=2304)
        self.linear2 = Linear(10, act=None, W_init=W_init2, name='output', in_features=1024)

    def forward(self, x):
        z = self.conv1(x)
        z = self.maxpool1(z)
        z = self.conv2(z)
        z = self.maxpool2(z)
        z = self.flatten(z)
        z = self.linear1(z)
        z = self.linear2(z)
        return z


# get the network
net = CNN()
```

打印模型结构
```
[TL] Conv2d conv1: out_channels : 32 kernel_size: (3, 3) stride: (1, 1) pad: SAME act: No Activation
[TL] MaxPool2d pool1: kernel_size: (2, 2) stride: (2, 2) padding: SAME
[TL] Conv2d conv2: out_channels : 64 kernel_size: (3, 3) stride: (1, 1) pad: SAME act: ReLU
[TL] Conv2d conv3: out_channels : 64 kernel_size: (3, 3) stride: (1, 1) pad: SAME act: ReLU
[TL] MaxPool2d pool2: kernel_size: (2, 2) stride: (2, 2) padding: SAME
[TL] Flatten flatten:
[TL] Linear  linear1relu: 1024 ReLU
[TL] Linear  output: 10 No Activation
```

## 四、模型训练&预测  
接下来，用Model高级接口来快速开始模型的训练，将会:

* 使用 `tlx.optimizers.Adam` 优化器来进行优化。

* 使用 `tlx.losses.softmax_cross_entropy_with_logits` 来计算损失值。

* 使用 `tensorlayerx.dataflow.DataLoader` 来加载数据并组建batch。

* 使用 `tlx.model.Model` 高级模型接口构建用于训练的模型

```
#构建batch数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

#使用高级API构建可训练模型
model = tlx.model.Model(network=net, loss_fn=loss_fn, optimizer=optimizer, metrics=metrics)

#执行训练
model.train(n_epoch=n_epoch, train_dataset=train_loader, test_dataset=test_loader, print_freq=print_freq, print_train_batch=True)
```

```
Epoch 1 of 500 took 2.037001371383667
   train loss: [2.4006615]
   train acc:  0.0546875
Epoch 1 of 500 took 2.0650012493133545
   train loss: [2.3827682]
   train acc:  0.05859375
      ......

Epoch 30 of 500 took 11.347046375274658
   train loss: [0.8383277]
   train acc:  0.7114410166240409
   val loss: Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=False,
       [0.87068987])
   val acc:  0.7016416139240507
```

## The End
从上面的示例可以看到，在cifar10数据集上，使用简单的卷积神经网络，用TensorLayerX可以达到70%以上的准确率。你也可以通过调整网络结构和参数，达到更好的效果。