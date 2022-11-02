# 使用卷积神经网络进行图像分类
**摘要:** 本教程将通过一个示例对 GAN 进行介绍。 在向其展示许多真实手写数字的照片后，我们将训练一个生成对抗网络（GAN）来产生新手写数字。 此处的大多数代码来自[examples\basic_tutorials\mnist_gan.py](examples\basic_tutorials\mnist_gan.py)中的 gan 实现，并且本文档将对该实现进行详尽的解释，并阐明此模型的工作方式和原因。 但请放心，不需要 GAN 的先验知识，但这可能需要新手花一些时间来推理幕后实际发生的事情。 

## 生成对抗网络
### 什么是 GAN？
GAN 是用于教授 DL 模型以捕获训练数据分布的框架，因此我们可以从同一分布中生成新数据。 GAN 由 Ian Goodfellow 于 2014 年发明，并在论文[《生成对抗网络》](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)中首次进行了描述。 它们由两个不同的模型组成：生成器和判别器。 生成器的工作是生成看起来像训练图像的“假”图像。 判别器的工作是查看图像并从生成器输出它是真实的训练图像还是伪图像。 在训练过程中，生成器不断尝试通过生成越来越好的伪造品而使判别器的表现超过智者，而判别器正在努力成为更好的侦探并正确地对真实和伪造图像进行分类。 博弈的平衡点是当生成器生成的伪造品看起来像直接来自训练数据时，而判别器则总是猜测生成器输出是真实还是伪造品的 50% 置信度。

现在，让我们从判别器开始定义一些在整个教程中使用的符号。 令`x`为代表图像的数据。` D(x)`是判别器网络，其输出`x`来自训练数据而不是生成器的（标量）概率。 在这里，由于我们要处理图像，因此`D(x)`的输入是 `CHW `大小为`3x64x64`的图像。 直观地，当`x`来自训练数据时，`D(x)`应该为高，而当`x`来自生成器时，它应该为低。 `D(x)`也可以被认为是传统的二分类器。

对于生成器的表示法，令`z`是从标准正态分布中采样的潜在空间向量。 `G(z)`表示将隐向量z映射到数据空间的生成器函数。 `G`的目标是估计训练数据来自`p_data`的分布，以便它可以从该估计分布（`p_g）`生成假样本。

因此，`D(G(z))`是生成器G的输出是真实图像的概率（标量）。 如 [Goodfellow](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) 的论文中所述，`D`和`G`玩一个 `minimax` 游戏，其中D试图最大化其正确分类实物和假物`log D(x)`，并且`G`尝试最小化`D`预测其输出为假的概率`log(1 - D(G(g(x))))`。 从本文来看，`GAN `损失函数为:   
![](images\gan_loss.gif)  
从理论上讲，此极小极大游戏的解决方案是`p_g = p_data`，判别器会随机猜测输入是真实的还是假的。 但是，`GAN` 的收敛理论仍在积极研究中，实际上，模型并不总是能达到这一目的。
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
import time
import numpy as np
import tensorlayerx as tlx
from tensorlayerx.nn import Module, Linear
from tensorlayerx.dataflow import Dataset
from tensorlayerx.model import TrainOneStep
```
## 二、加载数据集
本案例将会使用TensorLayerX提供的API完成数据集的下载并为后续的训练任务准备好数据迭代器。 
MNIST手写数字识别数据集由60000张大小为28 * 28的黑白图片组成。这些图片分为10个类别，分别对应数字0-9，将训练一个模型能够把图片进行正确的分类。

```{.python}
# prepare cifar10 data
X_train, y_train, X_val, y_val, X_test, y_test = tlx.files.load_mnist_dataset(shape=(-1, 784))


class mnistdataset(Dataset):

    def __init__(self, data=X_train, label=y_train):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = self.data[index].astype('float32')
        label = self.label[index].astype('int64')
        return data, label

    def __len__(self):
        return len(self.data)


#构建数据集和加载器
train_dataset = mnistdataset(data=X_train, label=y_train)

batch_size = 128
train_loader = tlx.dataflow.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

```

# 三、组建网络

## 生成器
接下来使用TensorLayerX定义一个使用了三个全连接层（ `Linear` ) 且前两层使用 `relu` 激活函数，最后一层使用值域为`-1~1`的`Tanh`作为激活函数神经网络作为GAN中的生成器网络`G`，来把一个`(1,100)`形状的随机噪音向量通过全连接层映射为`28*28=784`维度的向量，相当于生成一个28*28的手写图片。

```{.python}
class Generator(Module):

    def __init__(self):
        super(generator, self).__init__()
        self.g_fc1 = Linear(out_features=256, in_features=100, act=tlx.nn.ReLU)
        self.g_fc2 = Linear(out_features=256, in_features=256, act=tlx.nn.ReLU)
        self.g_fc3 = Linear(out_features=784, in_features=256, act=tlx.nn.Tanh)

    def forward(self, x):
        out = self.g_fc1(x)
        out = self.g_fc2(out)
        out = self.g_fc3(out)
        return out
```

## 判别器
接下来使用TensorLayerX定义一个使用了三个全连接层（ `Linear` ) 且前两层使用 `relu` 激活函数，最后一层使用值域为`0~1`的`Sigmoid`作为激活函数神经网络作为GAN中的判别器网络`D`。它接受来自生成器网络`G`或真实手写图片的一个`(1,784)`形状的由`28*28 `的图像折叠而成的向量，通过全连接层映射为`(1,1)`的值域为`0~1`的向量，对应真实/伪造的二分类。

```{.python}
class Discriminator(Module):

    def __init__(self):
        super(discriminator, self).__init__()
        self.d_fc1 = Linear(out_features=256, in_features=784, act=tlx.LeakyReLU)
        self.d_fc2 = Linear(out_features=256, in_features=256, act=tlx.LeakyReLU)
        self.d_fc3 = Linear(out_features=1, in_features=256, act=tlx.Sigmoid)

    def forward(self, x):
        out = self.d_fc1(x)
        out = self.d_fc2(out)
        out = self.d_fc3(out)
        return out
```

打印模型结构
```
Generator<
  (g_fc1): Linear(out_features=256, ReLU, in_features='100', name='linear_1')
  (g_fc2): Linear(out_features=256, ReLU, in_features='256', name='linear_2')
  (g_fc3): Linear(out_features=784, Tanh, in_features='256', name='linear_3')
  >
Discriminator<
  (d_fc1): Linear(out_features=256, LeakyReLU, in_features='784', name='linear_4')
  (d_fc2): Linear(out_features=256, LeakyReLU, in_features='256', name='linear_5')
  (d_fc3): Linear(out_features=1, Sigmoid, in_features='256', name='linear_6')
  >
```

## 四、模型训练&预测  
接下来，由于生成器`G`和判别器`D`两个网络的训练过程是互相依赖的，我们要将他们计算损失函数过程也包装到一个Module对象中。  
`G`和`D`网络本身就是`Module`对象，他们也可以作为上一级的`Module`对象的组件进行使用，就像使用别的层作为组件一样。
```{.python}
class WithLossG(Module):

    def __init__(self, G, D, loss_fn):
        super(WithLossG, self).__init__()
        self.g_net = G
        self.d_net = D
        self.loss_fn = loss_fn

    def forward(self, g_data, label):
        fake_image = self.g_net(g_data)
        logits_fake = self.d_net(fake_image)
        valid = tlx.convert_to_tensor(np.ones(logits_fake.shape), dtype=tlx.float32)
        loss = self.loss_fn(logits_fake, valid)
        return loss


class WithLossD(Module):

    def __init__(self, G, D, loss_fn):
        super(WithLossD, self).__init__()
        self.g_net = G
        self.d_net = D
        self.loss_fn = loss_fn

    def forward(self, real_data, g_data):
        logits_real = self.d_net(real_data)
        fake_image = self.g_net(g_data)
        logits_fake = self.d_net(fake_image)

        valid = tlx.convert_to_tensor(np.ones(logits_real.shape), dtype=tlx.float32)
        fake = tlx.convert_to_tensor(np.zeros(logits_fake.shape), dtype=tlx.float32)

        loss = self.loss_fn(logits_real, valid) + self.loss_fn(logits_fake, fake)
        return loss
```
然后我们用`TrainOneStep`单步接口来开始模型的训练，将会:

* 分别使用 `tlx.optimizers.Adam` 优化器来对`G`和`D`网络进行优化。

* 使用 `tlx.losses.mean_squared_error` 来计算损失值。

* 使用 `tensorlayerx.dataflow.DataLoader` 来加载数据并组建batch。

* 使用 `tlx.model.TrainOneStep` 单步训练接口构建用于训练的模型

```{.python}
loss_fn = tlx.losses.mean_squared_error
optimizer_g = tlx.optimizers.Adam(lr=3e-4, beta_1=0.5, beta_2=0.999)
optimizer_d = tlx.optimizers.Adam(lr=3e-4)

g_weights = G.trainable_weights
d_weights = D.trainable_weights
net_with_loss_G = WithLossG(G, D, loss_fn)
net_with_loss_D = WithLossD(G, D, loss_fn)
train_one_step_g = TrainOneStep(net_with_loss_G, optimizer_g, g_weights)
train_one_step_d = TrainOneStep(net_with_loss_D, optimizer_d, d_weights)
```

接下来我们写一个循环，从数据集中加载数据，并对`train_one_step_g`，`train_one_step_d`两个网络进行训练
```{.python}
for epoch in range(n_epoch):
    d_loss, g_loss = 0.0, 0.0
    n_iter = 0
    start_time = time.time()
    for data, label in train_loader:
        noise = tlx.convert_to_tensor(np.random.random(size=(batch_size, 100)), dtype=tlx.float32)

        _loss_d = train_one_step_d(data, noise)
        _loss_g = train_one_step_g(noise, label)
        d_loss += _loss_d
        g_loss += _loss_g

        n_iter += 1
        print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
        print("   d loss: {}".format(d_loss / n_iter))
        print("   g loss:  {}".format(g_loss / n_iter))
    fake_image = G(tlx.convert_to_tensor(np.random.random(size=(36, 100)), dtype=tlx.float32))
    plot_fake_image(fake_image, 36)
```

```
Epoch 1 of 50 took 1.3067221641540527
   d loss: 0.5520201059612068
   g loss:  0.19243632538898572

...
```
初始生成的图片：
![](images\fake_mnist_1.png)  
最终结果：
![](images\fake_mnist_final.png)  

## The End
从上面的示例可以看到，在MNIST数据集上，使用简单的GAN神经网络，用TensorLayerX可以生成逼真的手写数字图片。你也可以通过调整网络结构和参数，达到更好的效果。