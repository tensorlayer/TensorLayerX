# TensorlayerX目前支持包括TensorFlow、Pytorch、PaddlePaddle、MindSpore作为计算后端，指定计算后端的方法也非常简单，只需要设置环境变量即可
import os
# os.environ['TL_BACKEND'] = 'tensorflow'
os.environ['TL_BACKEND'] = 'jittor'
# os.environ['TL_BACKEND'] = 'torch'
# os.environ['TL_BACKEND'] = 'mindspore'
# os.environ['TL_BACKEND'] = 'paddle'


import tensorlayerx as tlx


from tensorlayerx.nn import Module
from tensorlayerx.nn import Sequential #序列式模型

from tensorlayerx.nn import (Conv2d, Linear, Flatten, MaxPool2d, BatchNorm2d)

from tensorlayerx.dataflow import Dataset, DataLoader
from tensorlayerx.vision.transforms import (
    Compose, Resize, RandomFlipHorizontal, RandomContrast, RandomBrightness, StandardizePerImage, RandomCrop
)

layer_list = [] #空的层列表
#依次添加各层
layer_list.append(Linear(out_features=800, act=tlx.ReLU, in_features=784, name='linear1'))
layer_list.append(Linear(out_features=800, act=tlx.ReLU, in_features=800, name='linear2'))
layer_list.append(Linear(out_features=10, act=tlx.ReLU, in_features=800, name='linear3'))
MLP = Sequential(layer_list)


class CNN(Module):

    def __init__(self):
        super(CNN, self).__init__()
        W_init = tlx.nn.initializers.truncated_normal(stddev=5e-2)
        W_init2 = tlx.nn.initializers.truncated_normal(stddev=0.04)
        b_init = tlx.nn.initializers.constant(value=0.1)
        b_init2 = tlx.nn.initializers.constant(value=0.1)

        self.conv1 = Conv2d(32, (5, 5), (1, 1), padding='SAME', W_init=W_init, b_init=b_init, name='conv1', in_channels=3)
        self.bn1 = BatchNorm2d(num_features=32, act=tlx.nn.ReLU)
        self.maxpool1 = MaxPool2d((2, 2), (2, 2), padding='SAME', name='pool1')

        self.conv2 = Conv2d(64, (5, 5), (1, 1), padding='SAME', act=tlx.nn.ReLU, W_init=W_init, b_init=b_init, name='conv2', in_channels=32)
        self.bn2 = BatchNorm2d(num_features=64, act=tlx.nn.ReLU)
        self.maxpool2 = MaxPool2d((2, 2), (2, 2), padding='SAME', name='pool2')

        self.flatten = Flatten(name='flatten')
        self.linear1 = Linear(1024, act=tlx.nn.ReLU, W_init=W_init2, b_init=b_init2, name='linear1relu', in_features=2304)
																													   
        self.linear2 = Linear(10, act=None, W_init=W_init2, b_init=b_init2, name='output', in_features=1024)

    def forward(self, x):
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.maxpool1(z)
        z = self.conv2(z)
        z = self.bn2(z)
        z = self.maxpool2(z)
        z = self.flatten(z)
        z = self.linear1(z)
        z = self.linear2(z)						   
        return z

X_train, y_train, X_test, y_test = tlx.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)

class cifar10_dataset(Dataset):

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


train_transforms = Compose(
    [
        RandomCrop(size=[24, 24]),
        RandomFlipHorizontal(),
        RandomBrightness(brightness_factor=(0.5, 1.5)),
        RandomContrast(contrast_factor=(0.5, 1.5)),
        StandardizePerImage()
    ]
)
# 设置训练参数
batch_size = 128
n_epoch = 500
learning_rate = 0.0001
print_freq = 5
n_step_epoch = int(len(y_train) / batch_size)
n_step = n_epoch * n_step_epoch
shuffle_buffer_size = 128

test_transforms = Compose([Resize(size=(24, 24)), StandardizePerImage()])

train_dataset = cifar10_dataset(data=X_train, label=y_train, transforms=train_transforms)
test_dataset = cifar10_dataset(data=X_test, label=y_test, transforms=test_transforms)

train_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = DataLoader(test_dataset, batch_size=batch_size)

# 搭建网络
net = CNN()

# 定义损失函数、优化器等
optimizer = tlx.optimizers.Adam(learning_rate)
metrics = tlx.metrics.Accuracy()
loss_fn = tlx.losses.softmax_cross_entropy_with_logits

#使用高级API构建可训练模型
net_with_train = tlx.model.Model(
    network=net, loss_fn=loss_fn, optimizer=optimizer, metrics=metrics
)

#执行训练
net_with_train.train(n_epoch=n_epoch, train_dataset=train_dataset, print_freq=print_freq, print_train_batch=False)
