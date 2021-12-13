#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
# os.environ['TL_BACKEND'] = 'paddle'
os.environ['TL_BACKEND'] = 'tensorflow'
# os.environ['TL_BACKEND'] = 'mindspore'

import time
import numpy as np
import tensorlayer as tl
from tensorlayer.layers import Module, Dense
from tensorlayer.dataflow import Dataset
from tensorlayer.models import TrainOneStep

X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))


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


batch_size = 128
train_dataset = mnistdataset(data=X_train, label=y_train)
train_dataset = tl.dataflow.FromGenerator(
    train_dataset, output_types=[tl.float32, tl.int64], column_names=['data', 'label']
)
train_loader = tl.dataflow.Dataloader(train_dataset, batch_size=batch_size, shuffle=True)


class generator(Module):

    def __init__(self):
        super(generator, self).__init__()
        self.g_fc1 = Dense(n_units=256, in_channels=100, act=tl.ReLU)
        self.g_fc2 = Dense(n_units=256, in_channels=256, act=tl.ReLU)
        self.g_fc3 = Dense(n_units=784, in_channels=256, act=tl.Tanh)

    def forward(self, x):
        out = self.g_fc1(x)
        out = self.g_fc2(out)
        out = self.g_fc3(out)
        return out


class discriminator(Module):

    def __init__(self):
        super(discriminator, self).__init__()
        self.d_fc1 = Dense(n_units=256, in_channels=784, act=tl.LeakyReLU)
        self.d_fc2 = Dense(n_units=256, in_channels=256, act=tl.LeakyReLU)
        self.d_fc3 = Dense(n_units=1, in_channels=256, act=tl.Sigmoid)

    def forward(self, x):
        out = self.d_fc1(x)
        out = self.d_fc2(out)
        out = self.d_fc3(out)
        return out


G = generator()
D = discriminator()


class WithLossG(Module):

    def __init__(self, G, D, loss_fn):
        super(WithLossG, self).__init__()
        self.g_net = G
        self.d_net = D
        self.loss_fn = loss_fn

    def forward(self, g_data, label):
        fake_image = self.g_net(g_data)
        logits_fake = self.d_net(fake_image)
        valid = tl.convert_to_tensor(np.ones(logits_fake.shape), dtype=tl.float32)
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

        valid = tl.convert_to_tensor(np.ones(logits_real.shape), dtype=tl.float32)
        fake = tl.convert_to_tensor(np.zeros(logits_fake.shape), dtype=tl.float32)

        loss = self.loss_fn(logits_real, valid) + self.loss_fn(logits_fake, fake)
        return loss


# loss_fn = tl.cost.sigmoid_cross_entropy
# optimizer = tl.optimizers.Momentum(learning_rate=5e-4, momentum=0.5)
loss_fn = tl.cost.mean_squared_error
optimizer_g = tl.optimizers.Adam(learning_rate=3e-4, beta_1=0.5, beta_2=0.999)
optimizer_d = tl.optimizers.Adam(learning_rate=3e-4)

g_weights = G.trainable_weights
d_weights = D.trainable_weights
net_with_loss_G = WithLossG(G, D, loss_fn)
net_with_loss_D = WithLossD(G, D, loss_fn)
train_one_setp_g = TrainOneStep(net_with_loss_G, optimizer_g, g_weights)
train_one_setp_d = TrainOneStep(net_with_loss_D, optimizer_d, d_weights)
n_epoch = 50


def plot_fake_image(fake_image, num):
    fake_image = tl.reshape(fake_image, shape=(num, 28, 28))
    fake_image = tl.convert_to_numpy(fake_image)
    import matplotlib.pylab as plt
    for i in range(num):
        plt.subplot(int(np.sqrt(num)), int(np.sqrt(num)), i + 1)
        plt.imshow(fake_image[i])
    plt.show()


for epoch in range(n_epoch):
    d_loss, g_loss = 0.0, 0.0
    n_iter = 0
    start_time = time.time()
    for data, label in train_loader:
        noise = tl.convert_to_tensor(np.random.random(size=(batch_size, 100)), dtype=tl.float32)

        _loss_d = train_one_setp_d(data, noise)
        _loss_g = train_one_setp_g(noise, label)
        d_loss += _loss_d
        g_loss += _loss_g

        n_iter += 1
        print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
        print("   d loss: {}".format(d_loss / n_iter))
        print("   g loss:  {}".format(g_loss / n_iter))
    fake_image = G(tl.convert_to_tensor(np.random.random(size=(36, 100)), dtype=tl.float32))
    plot_fake_image(fake_image, 36)
