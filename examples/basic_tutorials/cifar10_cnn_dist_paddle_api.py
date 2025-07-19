#! /usr/bin/python
# -*- coding: utf-8 -*-
import os
os.environ['TL_BACKEND'] = 'paddle'

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.io import DataLoader, DistributedBatchSampler
from tensorlayerx.vision.transforms import (
    Compose, Resize, RandomFlipHorizontal, RandomContrast, RandomBrightness, StandardizePerImage, RandomCrop
)
from tensorlayerx.dataflow import Dataset
from tensorlayerx.nn import Module
import tensorlayerx as tlx
from tensorlayerx.nn import (Conv2d, Linear, Flatten, MaxPool2d, BatchNorm2d)
# enable debug logging
tlx.logging.set_verbosity(tlx.logging.DEBUG)

# paddle.disable_static()
paddle.set_device('gpu')
print(paddle.get_device())
fleet.init(is_collective=True)
# print(tlx.is_distributed())
# prepare cifar10 data
X_train, y_train, X_test, y_test = tlx.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)

# training settings
batch_size = 128
n_epoch = 10
learning_rate = 0.0001
print_freq = 5
n_step_epoch = int(len(y_train) / batch_size)
n_step = n_epoch * n_step_epoch
shuffle_buffer_size = 128

class make_dataset(Dataset):

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

# 构建分布式训练使用的数据集和加载器
train_dataset = make_dataset(data=X_train, label=y_train, transforms=train_transforms)
test_dataset = make_dataset(data=X_test, label=y_test, transforms=test_transforms)

# 五、构建分布式训练使用的数据集
train_sampler = DistributedBatchSampler(train_dataset, batch_size, shuffle=True, drop_last=True)
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=2)

valid_sampler = DistributedBatchSampler(test_dataset, batch_size, drop_last=True)
valid_loader = DataLoader(test_dataset, batch_sampler=valid_sampler, num_workers=2)


class CNN(Module):

    def __init__(self):
        super(CNN, self).__init__()
        # weights init
        W_init = tlx.nn.initializers.truncated_normal(stddev=5e-2)
        W_init2 = tlx.nn.initializers.truncated_normal(stddev=0.04)
        b_init2 = tlx.nn.initializers.constant(value=0.1)

        self.conv1 = Conv2d(64, (5, 5), (1, 1), padding='SAME', W_init=W_init, b_init=None, name='conv1', in_channels=3)
        self.bn = BatchNorm2d(num_features=64, act=tlx.ReLU)
        self.maxpool1 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool1')

        self.conv2 = Conv2d(
            64, (5, 5), (1, 1), padding='SAME', act=tlx.ReLU, W_init=W_init, name='conv2', in_channels=64
        )
        self.maxpool2 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool2')

        self.flatten = Flatten(name='flatten')
        self.linear1 = Linear(384, act=tlx.ReLU, W_init=W_init2, b_init=b_init2, name='linear1relu', in_features=2304)
        self.linear2 = Linear(192, act=tlx.ReLU, W_init=W_init2, b_init=b_init2, name='linear2relu', in_features=384)
        self.linear3 = Linear(10, act=None, W_init=W_init2, name='output', in_features=192)

    def forward(self, x):
        z = self.conv1(x)
        z = self.bn(z)
        z = self.maxpool1(z)
        z = self.conv2(z)
        z = self.maxpool2(z)
        z = self.flatten(z)
        z = self.linear1(z)
        z = self.linear2(z)
        z = self.linear3(z)
        return z
        

# get the network
net = CNN()

# 获取分布式 model，用于支持分布式训练
dp_layer = fleet.distributed_model(net)

# 定义损失函数、优化器等
loss_fn = tlx.losses.softmax_cross_entropy_with_logits
optimizer = paddle.optimizer.Adam(learning_rate, parameters=dp_layer.parameters())
optimizer = fleet.distributed_optimizer(optimizer)
metrics = tlx.metrics.Accuracy()

print("模型已转换为分布式")

val_acc_history = []
val_loss_history = []

#执行训练
import time
t0 = time.time()

for epoch in range(n_epoch):
    dp_layer.train()
    for batch_id, data in enumerate(train_loader()):
        x_data = data[0]
        y_data = paddle.to_tensor(data[1])
        y_data = paddle.unsqueeze(y_data, 1)

        logits = dp_layer(x_data)
        loss = loss_fn(logits, y_data)

        if batch_id % 1000 == 0:
            print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, loss.numpy()))
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

    # dp_layer.eval()
    # accuracies = []
    # losses = []
    # for batch_id, data in enumerate(valid_loader()):
    #     x_data = data[0]
    #     y_data = paddle.to_tensor(data[1])
    #     y_data = paddle.unsqueeze(y_data, 1)

    #     logits = dp_layer(x_data)
    #     loss = loss_fn(logits, y_data)
    #     acc = paddle.metric.accuracy(logits, y_data)
    #     accuracies.append(acc.numpy())
    #     losses.append(loss.numpy())

    # avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
    # print("[validation] accuracy/loss: {}/{}".format(avg_acc, avg_loss))
    # val_acc_history.append(avg_acc)
    # val_loss_history.append(avg_loss)

t1 = time.time()
training_time = t1 - t0
import datetime
def format_time(time):
    elapsed_rounded = int(round((time)))
    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
training_time = format_time(training_time)
print(training_time)
