#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
# os.environ['TL_BACKEND'] = 'paddle'
# os.environ['TL_BACKEND'] = 'tensorflow'
# os.environ['TL_BACKEND'] = 'mindspore'
os.environ['TL_BACKEND'] = 'torch'

import time
from tensorlayerx.vision.transforms import (
    Compose, Resize, RandomFlipHorizontal, RandomContrast, RandomBrightness, StandardizePerImage, RandomCrop
)
from tensorlayerx.nn import Module
import tensorlayerx as tlx
from tensorlayerx.nn import (Conv2d, Linear, Flatten, MaxPool2d, BatchNorm2d)
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# enable debug logging
tlx.logging.set_verbosity(tlx.logging.DEBUG)


local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
init_process_group(backend="nccl")
print(f"Process {local_rank} using GPU: {torch.cuda.get_device_name(local_rank)}")

# ################## Download and prepare the CIFAR10 dataset ##################
# This is just some way of getting the CIFAR10 dataset from an online location
# and loading it into numpy arrays with shape [32,32,3]
X_train, y_train, X_test, y_test = tlx.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)

# training settings
n_epoch = 5
learning_rate = 0.0001
print_freq = 5
n_step_epoch = int(len(y_train) / 128)
n_step = n_epoch * n_step_epoch
shuffle_buffer_size = 128
batch_size = 128

# ################## CIFAR10 dataset ##################
# We define a Dataset class for Loading CIFAR10 images and labels.
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

# We define the CIFAR10 iamges preprocessing pipeline.
train_transforms = Compose( # Combining multiple operations sequentially
    [
        RandomCrop(size=[24, 24]), #random crop from images to shape [24, 24]
        RandomFlipHorizontal(), # random invert each image horizontally by probability
        RandomBrightness(brightness_factor=(0.5, 1.5)), # Within the range of values (0.5, 1.5), adjust brightness randomly
        RandomContrast(contrast_factor=(0.5, 1.5)), # Within the range of values (0.5, 1.5), adjust contrast randomly
        StandardizePerImage() #Normalize the values of each image to [-1, 1]
    ]
)

test_transforms = Compose([Resize(size=(24, 24)), StandardizePerImage()])

# We use DataLoader to batch and shuffle data, and make data into iterators.
train_dataset = make_dataset(data=X_train, label=y_train, transforms=train_transforms)
test_dataset = make_dataset(data=X_test, label=y_test, transforms=test_transforms)
train_dataset = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(train_dataset)
    )

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        # save_every: int,
        # snapshot_path: str,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        # self.save_every = save_every
        self.epochs_run = 0
        # self.snapshot_path = snapshot_path
        # if os.path.exists(snapshot_path):
        #     print("Loading snapshot")
        #     self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            # if self.gpu_id == 0 and epoch % self.save_every == 0:
            #     self._save_snapshot(epoch)


# ################## CNN network ##################
class CNN(Module):

    def __init__(self):
        super(CNN, self).__init__()
        # Parameter initialization method
        W_init = tlx.nn.initializers.truncated_normal(stddev=5e-2)
        W_init2 = tlx.nn.initializers.truncated_normal(stddev=0.04)
        b_init2 = tlx.nn.initializers.constant(value=0.1)

        # 2D Convolutional Neural Network, Set padding method "SAME", convolutional kernel size [5,5], stride [1,1], in channels, out channels
        self.conv1 = Conv2d(64, (5, 5), (1, 1), padding='SAME', W_init=W_init, b_init=None, name='conv1', in_channels=3)
        # Add 2D BatchNormalize, using ReLU for output.
        self.bn = BatchNorm2d(num_features=64, act=tlx.nn.ReLU)
        # Add 2D Max pooling layer.
        self.maxpool1 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool1')

        self.conv2 = Conv2d(
            64, (5, 5), (1, 1), padding='SAME', act=tlx.nn.ReLU, W_init=W_init, name='conv2', in_channels=64
        )
        self.maxpool2 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool2')
        # Flatten 2D data to 1D data
        self.flatten = Flatten(name='flatten')
        # Linear layer with 384 units, using ReLU for output.
        self.linear1 = Linear(384, act=tlx.nn.ReLU, W_init=W_init2, b_init=b_init2, name='linear1relu', in_features=2304)
        self.linear2 = Linear(192, act=tlx.nn.ReLU, W_init=W_init2, b_init=b_init2, name='linear2relu', in_features=384)
        self.linear3 = Linear(10, act=None, W_init=W_init2, name='output', in_features=192)

    # We define the forward computation process.
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

# Get training parameters
train_weights = net.trainable_weights
# Define the optimizer, use the Adam optimizer.
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
trainer = Trainer(net, train_dataset, optimizer)

# Custom training loops
t0 = time.time()

trainer.train(n_epoch)

t1 = time.time()
training_time = t1 - t0
import datetime
def format_time(time):
    elapsed_rounded = int(round((time)))
    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
training_time = format_time(training_time)
print(training_time)

destroy_process_group()
