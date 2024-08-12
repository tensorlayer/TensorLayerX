
# """"
# Here we have a Tutorial of Jittor backend being used with several different models, which includes:

# """
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++ Jittor CIFAR CNN ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import time
import tensorlayerx as tlx
from tensorlayerx.dataflow import Dataset, DataLoader
from tensorlayerx.vision.transforms import (
    Compose, Resize, RandomFlipHorizontal, RandomContrast, RandomBrightness, StandardizePerImage, RandomCrop
)
from tensorlayerx.nn import Conv2d, Linear, Flatten, Module, MaxPool2d, BatchNorm2d
from tensorlayerx.optimizers import Adam
from tqdm import tqdm

# Enable debug logging
tlx.logging.set_verbosity(tlx.logging.DEBUG)

os.environ['TL_BACKEND'] = 'jittor'

# Download and prepare the CIFAR10 dataset
print("Downloading CIFAR10 dataset...")
X_train, y_train, X_test, y_test = tlx.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)

# Define the CIFAR10 dataset
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

# Define the CIFAR10 images preprocessing pipeline
train_transforms = Compose([
    RandomCrop(size=[24, 24]),
    RandomFlipHorizontal(),
    RandomBrightness(brightness_factor=(0.5, 1.5)),
    RandomContrast(contrast_factor=(0.5, 1.5)),
    StandardizePerImage()
])

test_transforms = Compose([Resize(size=(24, 24)), StandardizePerImage()])

# Create DataLoaders for training and testing
print("Processing CIFAR10 dataset...")
train_dataset = CIFAR10Dataset(data=X_train, label=y_train, transforms=train_transforms)
test_dataset = CIFAR10Dataset(data=X_test, label=y_test, transforms=test_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128)


class SimpleCNN(Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = Conv2d(16, (3, 3), (1, 1), padding='SAME', act=tlx.nn.ReLU, in_channels=3)
        self.conv2 = Conv2d(32, (3, 3), (1, 1), padding='SAME', act=tlx.nn.ReLU, in_channels=16)
        self.maxpool1 = MaxPool2d((2, 2), (2, 2), padding='SAME')
        self.conv3 = Conv2d(64, (3, 3), (1, 1), padding='SAME', act=tlx.nn.ReLU, in_channels=32)
        self.bn1 = BatchNorm2d(num_features=64, act=tlx.nn.ReLU)
        self.conv4 = Conv2d(128, (3, 3), (1, 1), padding='SAME', act=tlx.nn.ReLU, in_channels=64)
        self.maxpool2 = MaxPool2d((2, 2), (2, 2), padding='SAME')
        self.flatten = Flatten()
        self.fc1 = Linear(out_features=128, act=tlx.nn.ReLU, in_features=128 * 6 * 6)
        self.fc2 = Linear(out_features=64, act=tlx.nn.ReLU, in_features=128)
        self.fc3 = Linear(out_features=10, act=None, in_features=64)

    def forward(self, x):
        z = self.conv1(x)
        z = self.conv2(z)
        z = self.maxpool1(z)
        z = self.conv3(z)
        z = self.bn1(z)
        z = self.conv4(z)
        z = self.maxpool2(z)
        z = self.flatten(z)
        z = self.fc1(z)
        z = self.fc2(z)
        z = self.fc3(z)
        return z


# Instantiate the model
model = SimpleCNN()

# Define the optimizer
optimizer = Adam(lr=0.001)
# optimizer = Adam(lr=0.001, params=model.trainable_weights )

# Define the loss function
loss_fn = tlx.losses.softmax_cross_entropy_with_logits

# Use the built-in training method
metric = tlx.metrics.Accuracy()
tlx_model = tlx.model.Model(network=model, loss_fn=loss_fn, optimizer=optimizer, metrics=metric)
tlx_model.train(n_epoch=2, train_dataset=train_dataloader, print_freq=1, print_train_batch=True)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++ Jittor IMDB LSTM ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# import os
# import sys
# import tensorlayerx as tlx
# from tensorlayerx.nn import Module, LSTM, Embedding, Linear
# from tensorlayerx.dataflow import Dataset
# import numpy as np

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# os.environ['TL_BACKEND'] = 'jittor'
# sys.setrecursionlimit(10000)  # Increase recursion limit

# # Set parameters
# max_features = 20000
# maxlen = 200
# prev_h = np.random.random([1, 200, 64]).astype(np.float32)
# prev_h = tlx.convert_to_tensor(prev_h)
# X_train, y_train, X_test, y_test = tlx.files.load_imdb_dataset('data', nb_words=20000, test_split=0.2)
# vocab_size = max_features
# seq_Len = 200


# class ImdbDataset(Dataset):

#     def __init__(self, X, y):
#         self.X = X
#         self.y = y

#     def __getitem__(self, index):
#         data = self.X[index]
#         data = np.concatenate([data[:seq_Len], [0] * (seq_Len - len(data))]).astype('int64')  # set
#         label = self.y[index].astype('int64')
#         return data, label

#     def __len__(self):
#         return len(self.y)


# class ImdbNet(Module):

#     def __init__(self):
#         super(ImdbNet, self).__init__()
#         self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=64)
#         self.lstm = LSTM(input_size=64, hidden_size=64)
#         self.linear1 = Linear(in_features=64, out_features=64, act=tlx.nn.ReLU)
#         self.linear2 = Linear(in_features=64, out_features=2)
#     def forward(self, x):
#         x = self.embedding(x)
#         x, _ = self.lstm(x)
#         x = tlx.reduce_mean(x, axis=1)
#         x = self.linear1(x)
#         x = self.linear2(x)
#         return x

#     def __repr__(self):
#         return "ImdbNet(embedding_dim=64, hidden_size=64, num_classes=2)"

#     def __str__(self):
#         return self.__repr__()

# # Training settings
# n_epoch = 1
# batch_size = 64
# print_freq = 2

# # Create DataLoader
# train_dataset = ImdbDataset(X=X_train, y=y_train)
# train_loader = tlx.dataflow.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# # Initialize the network
# net = ImdbNet()
# print(net)

# # Define optimizer, metric, and loss function using TLX functions
# optimizer = tlx.optimizers.Adam(lr=1e-3)
# metric = tlx.metrics.Accuracy()
# loss_fn = tlx.losses.softmax_cross_entropy_with_logits

# # Create and train the model
# model = tlx.model.Model(network=net, loss_fn=loss_fn, optimizer=optimizer, metrics=metric)
# model.train(n_epoch=n_epoch, train_dataset=train_loader, print_freq=print_freq, print_train_batch=True)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++ Jittor MNIST MLP ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# # ! /usr/bin/python
# # -*- coding: utf-8 -*-

# # The same set of code can switch the backend with one line
# import os
# os.environ['TL_BACKEND'] = 'jittor'
# import tensorlayerx as tlx
# from tensorlayerx.nn import Module
# from tensorlayerx.nn import Linear, Dropout
# from tensorlayerx.dataflow import Dataset, DataLoader

# # ################## Download and prepare the MNIST dataset ##################
# # This is just some way of getting the MNIST dataset from an online location and loading it into numpy arrays
# X_train, y_train, X_val, y_val, X_test, y_test = tlx.files.load_mnist_dataset(shape=(-1, 784))

# # ################## MNIST dataset ##################
# # We define a Dataset class for Loading MNIST images and labels.
# class mnistdataset(Dataset):

#     def __init__(self, data=X_train, label=y_train):
#         self.data = data
#         self.label = label

#     def __getitem__(self, index):
#         data = self.data[index].astype('float32')
#         label = self.label[index].astype('int64')
#         return data, label

#     def __len__(self):
#         return len(self.data)

# # We use DataLoader to batch and shuffle data, and make data into iterators.
# train_dataset = mnistdataset(data=X_train, label=y_train)
# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# # ##################### Build the neural network model #######################
# # This creates an MLP of  two hidden Linear layers of 800 units each, followed by a Linear output layer of 10 units.

# class CustomModel(Module):

#     def __init__(self):
#         super(CustomModel, self).__init__()
#         # It applies 20% dropout to each Linear layer.
#         self.dropout1 = Dropout(p=0.2)
#         # Linear layer with 800 units, using ReLU for output.
#         self.linear1 = Linear(out_features=800, act=tlx.nn.ReLU, in_features=784)
#         self.dropout2 = Dropout(p=0.2)
#         # Linear layer with 800 units, using ReLU for output.
#         self.linear2 = Linear(out_features=800, act=tlx.nn.ReLU, in_features=800)
#         self.dropout3 = Dropout(p=0.2)
#         # Linear layer with 10 units, using ReLU for output.
#         self.linear3 = Linear(out_features=10, act=tlx.nn.ReLU, in_features=800)

#     # We define the forward computation process.
#     def forward(self, x):
#         z = self.dropout1(x)
#         z = self.linear1(z)
#         z = self.dropout2(z)
#         z = self.linear2(z)
#         z = self.dropout3(z)
#         out = self.linear3(z)
#         return out

# # We initialize the network
# MLP = CustomModel()
# # Set the number of training cycles
# n_epoch = 50
# # set print frequency.
# print_freq = 2

# # Get training parameters
# train_weights = MLP.trainable_weights
# # Define the optimizer, use the Momentum optimizer, and set the learning rate to 0.05, momentum to 0.9
# optimizer = tlx.optimizers.Momentum(lr=0.05, momentum= 0.9 )
# # Define evaluation metrics.
# metric = tlx.metrics.Accuracy()
# # Define loss function, this operator implements the cross entropy loss function with softmax. This function
# # combines the calculation of the softmax operation and the cross entropy loss function
# # to provide a more numerically stable computing.
# loss_fn = tlx.losses.softmax_cross_entropy_with_logits

# # Using a simple training method without custom trianing loops.
# model = tlx.model.Model(network=MLP, loss_fn=loss_fn, optimizer=optimizer, metrics=metric)
# model.train(n_epoch=n_epoch, train_dataset=train_loader, print_freq=print_freq, print_train_batch=False)

# # Optionally, you could now dump the network weights to a file like this:
# # model.save_weights('./model.npz', format='npz_dict')
# # model.load_weights('./model.npz', format='npz_dict')


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++ Jittor MNIST Sequential ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ! /usr/bin/python
# -*- coding: utf-8 -*-
# import os
# os.environ['TL_BACKEND'] = 'jittor'

# # os.environ['TL_BACKEND'] = 'torch'

# from tensorlayerx.nn import Sequential
# from tensorlayerx.nn import Linear
# import tensorlayerx as tlx
# from tensorlayerx.dataflow import Dataset

# layer_list = []
# layer_list.append(Linear(out_features=800, act=tlx.nn.ReLU, in_features=784, name='linear1'))
# layer_list.append(Linear(out_features=800, act=tlx.nn.ReLU, in_features=800, name='linear2'))
# layer_list.append(Linear(out_features=10, act=tlx.nn.ReLU, in_features=800, name='linear3'))
# MLP = Sequential(layer_list)

# X_train, y_train, X_val, y_val, X_test, y_test = tlx.files.load_mnist_dataset(shape=(-1, 784))


# class mnistdataset(Dataset):

#     def __init__(self, data=X_train, label=y_train):
#         self.data = data
#         self.label = label

#     def __getitem__(self, index):
#         data = self.data[index].astype('float32')
#         label = self.label[index].astype('int64')

#         return data, label

#     def __len__(self):

#         return len(self.data)


# n_epoch = 1
# batch_size = 128
# print_freq = 2
# shuffle_buffer_size = 128

# train_weights = MLP.trainable_weights
# optimizer = tlx.optimizers.Momentum(lr=0.05,momentum= 0.9)
# train_dataset = mnistdataset(data=X_train, label=y_train)
# train_loader = tlx.dataflow.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# metric = tlx.metrics.Accuracy()
# model = tlx.model.Model(
#     network=MLP, loss_fn=tlx.losses.softmax_cross_entropy_with_logits, optimizer=optimizer, metrics=metric
# )
# model.train(n_epoch=n_epoch, train_dataset=train_loader, print_freq=print_freq, print_train_batch=False)
# model.save_weights('./model.npz', format='npz_dict')
# model.load_weights('./model.npz', format='npz_dict', skip=True)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++ Jittor MNIST GAN ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#! /usr/bin/python
# -*- coding: utf-8 -*-

# import os
# os.environ['TL_BACKEND'] = 'jittor'

# import time
# import numpy as np
# import tensorlayerx as tlx
# from tensorlayerx.nn import Module, Linear
# from tensorlayerx.dataflow import Dataset
# from tensorlayerx.model import TrainOneStep

# # ################## Download and prepare the MNIST dataset ##################
# # This is just some way of getting the MNIST dataset from an online location and loading it into numpy arrays
# X_train, y_train, X_val, y_val, X_test, y_test = tlx.files.load_mnist_dataset(shape=(-1, 784))

# # ################## MNIST dataset ##################
# # We define a Dataset class for Loading MNIST images and labels.
# class mnistdataset(Dataset):

#     def __init__(self, data=X_train, label=y_train):
#         self.data = data
#         self.label = label

#     def __getitem__(self, index):
#         data = self.data[index].astype('float32')
#         label = self.label[index].astype('int64')
#         return data, label

#     def __len__(self):
#         return len(self.data)

# # We use DataLoader to batch and shuffle data, and make data into iterators.
# batch_size = 128
# train_dataset = mnistdataset(data=X_train, label=y_train)
# train_loader = tlx.dataflow.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# # We define generator network.
# class generator(Module):

#     def __init__(self):
#         super(generator, self).__init__()
#         # Linear layer with 256 units, using ReLU for output.
#         self.g_fc1 = Linear(out_features=256, in_features=100, act=tlx.nn.ReLU)
#         self.g_fc2 = Linear(out_features=256, in_features=256, act=tlx.nn.ReLU)
#         self.g_fc3 = Linear(out_features=784, in_features=256, act=tlx.nn.Tanh)

#     def forward(self, x):
#         out = self.g_fc1(x)
#         out = self.g_fc2(out)
#         out = self.g_fc3(out)
#         return out

# # We define discriminator network.
# class discriminator(Module):

#     def __init__(self):
#         super(discriminator, self).__init__()
#         # Linear layer with 256 units, using ReLU for output.
#         self.d_fc1 = Linear(out_features=256, in_features=784, act=tlx.LeakyReLU)
#         self.d_fc2 = Linear(out_features=256, in_features=256, act=tlx.LeakyReLU)
#         self.d_fc3 = Linear(out_features=1, in_features=256, act=tlx.Sigmoid)

#     def forward(self, x):
#         out = self.d_fc1(x)
#         out = self.d_fc2(out)
#         out = self.d_fc3(out)
#         return out


# G = generator()
# D = discriminator()

# # Define the generator network loss calculation process
# class WithLossG(Module):

#     def __init__(self, G, D, loss_fn):
#         super(WithLossG, self).__init__()
#         self.g_net = G
#         self.d_net = D
#         self.loss_fn = loss_fn

#     def forward(self, g_data, label):
#         fake_image = self.g_net(g_data)
#         logits_fake = self.d_net(fake_image)
#         valid = tlx.convert_to_tensor(np.ones(logits_fake.shape), dtype=tlx.float32)
#         loss = self.loss_fn(logits_fake, valid)
#         return loss

# # Define the discriminator network loss calculation process
# class WithLossD(Module):

#     def __init__(self, G, D, loss_fn):
#         super(WithLossD, self).__init__()
#         self.g_net = G
#         self.d_net = D
#         self.loss_fn = loss_fn

#     def forward(self, real_data, g_data):
#         logits_real = self.d_net(real_data)
#         fake_image = self.g_net(g_data)
#         logits_fake = self.d_net(fake_image)

#         valid = tlx.convert_to_tensor(np.ones(logits_real.shape), dtype=tlx.float32)
#         fake = tlx.convert_to_tensor(np.zeros(logits_fake.shape), dtype=tlx.float32)

#         loss = self.loss_fn(logits_real, valid) + self.loss_fn(logits_fake, fake)
#         return loss


# # loss_fn = tlx.losses.sigmoid_cross_entropy
# # optimizer = tlx.optimizers.Momentum(learning_rate=5e-4, momentum=0.5)
# loss_fn = tlx.losses.mean_squared_error
# # Define the optimizers, use the Adam optimizer.
# optimizer_g = tlx.optimizers.Adam(lr=3e-4, beta_1=0.5, beta_2=0.999)
# optimizer_d = tlx.optimizers.Adam(lr=3e-4)
# # Get training parameters
# g_weights = G.trainable_weights
# d_weights = D.trainable_weights
# net_with_loss_G = WithLossG(G, D, loss_fn)
# net_with_loss_D = WithLossD(G, D, loss_fn)
# # Initialize one-step training
# train_one_step_g = TrainOneStep(net_with_loss_G, optimizer_g, g_weights)
# train_one_step_d = TrainOneStep(net_with_loss_D, optimizer_d, d_weights)
# n_epoch = 2


# def plot_fake_image(fake_image, num):
#     fake_image = tlx.reshape(fake_image, shape=(num, 28, 28))
#     fake_image = tlx.convert_to_numpy(fake_image)
#     import matplotlib.pylab as plt
#     for i in range(num):
#         plt.subplot(int(np.sqrt(num)), int(np.sqrt(num)), i + 1)
#         plt.imshow(fake_image[i])
#     plt.show()

# # Custom training loops
# for epoch in range(n_epoch):
#     d_loss, g_loss = 0.0, 0.0
#     n_iter = 0
#     start_time = time.time()
#     # Get training data and labels
#     for data, label in train_loader:
#         noise = tlx.convert_to_tensor(np.random.random(size=(batch_size, 100)), dtype=tlx.float32)
#         # Calculate the loss value, and automatically complete the gradient update for discriminator
#         _loss_d = train_one_step_d(data, noise)
#         # Calculate the loss value, and automatically complete the gradient update for generator
#         _loss_g = train_one_step_g(noise, label)
#         d_loss += _loss_d
#         g_loss += _loss_g

#         n_iter += 1
#         print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
#         print("   d loss: {}".format(d_loss / n_iter))
#         print("   g loss:  {}".format(g_loss / n_iter))
#     fake_image = G(tlx.convert_to_tensor(np.random.random(size=(36, 100)), dtype=tlx.float32))
#     plot_fake_image(fake_image, 36)



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++ Jittor IMDB RNN +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# import os
# import sys
# import tensorlayerx as tlx
# from tensorlayerx.nn import Module, RNN, Embedding, Linear
# from tensorlayerx.dataflow import Dataset
# import numpy as np
					 

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# os.environ['TL_BACKEND'] = 'jittor'
# sys.setrecursionlimit(10000)  # Increase recursion limit

# # Set parameters
# max_features = 20000
# maxlen = 200
# prev_h = np.random.random([1, 200, 64]).astype(np.float32)
# prev_h = tlx.convert_to_tensor(prev_h)
# X_train, y_train, X_test, y_test = tlx.files.load_imdb_dataset('data', nb_words=20000, test_split=0.2)
# vocab_size = max_features
# seq_Len = 200

		
# class ImdbDataset(Dataset):
					   
#     def __init__(self, X, y):
#         self.X = X
#         self.y = y

#     def __getitem__(self, index):
#         data = self.X[index]
#         data = np.concatenate([data[:seq_Len], [0] * (seq_Len - len(data))]).astype('int64')  # set
#         label = self.y[index].astype('int64')
#         return data, label

#     def __len__(self):
#         return len(self.y)


# class ImdbNet(Module):

#     def __init__(self):
#         super(ImdbNet, self).__init__()
#         self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=64)
#         self.rnn = RNN(input_size=64, hidden_size=64)
#         self.linear1 = Linear(in_features=64, out_features=64, act=tlx.nn.ReLU)
#         self.linear2 = Linear(in_features=64, out_features=2)

#     def forward(self, x):
#         x = self.embedding(x)
#         x, _ = self.rnn(x)
#         x = tlx.reduce_mean(x, axis=1)
#         x = self.linear1(x)
#         x = self.linear2(x)
#         return x

#     def __repr__(self):
#         return "ImdbNet(embedding_dim=64, hidden_size=64, num_classes=2)"

#     def __str__(self):
#         return self.__repr__()

# # Training settings
# n_epoch = 1
# batch_size = 64
# print_freq = 2

# # Create DataLoader
# train_dataset = ImdbDataset(X=X_train, y=y_train)
# train_loader = tlx.dataflow.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# # Initialize the network
# net = ImdbNet()
# print(net)

# # Define optimizer, metric, and loss function using TLX functions
# optimizer = tlx.optimizers.Adam(lr=1e-3)
# metric = tlx.metrics.Accuracy()
# loss_fn = tlx.losses.softmax_cross_entropy_with_logits

# # Create and train the model
# model = tlx.model.Model(network=net, loss_fn=loss_fn, optimizer=optimizer, metrics=metric)
# model.train(n_epoch=n_epoch, train_dataset=train_loader, print_freq=print_freq, print_train_batch=True)
# Optionally, you could now dump the network weights to a file like this:
# model.save_weights('./rnn_model.npz', format='npz_dict')
# model.load_weights('./rnn_model.npz', format='npz_dict', skip= True)

