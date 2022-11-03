#! /usr/bin/python
# -*- coding: utf-8 -*-
"""VGG-16 for ImageNet using TL model."""

import time

import numpy as np
import tensorflow as tf

import tensorlayerx as tlx
from examples.model_zoo.imagenet_classes import class_names
from examples.model_zoo.vgg import vgg16

tlx.logging.set_verbosity(tlx.logging.DEBUG)

# get the whole model
vgg = vgg16(pretrained=True)
vgg.set_eval()

img = tlx.vision.load_image('data/tiger.jpeg')
img = tlx.vision.transforms.transforms.Resize((224, 224))(img).astype(np.float32) / 255

start_time = time.time()
output = vgg(img)
probs = tf.nn.softmax(output)[0].numpy()
print("  End time : %.5ss" % (time.time() - start_time))
preds = (np.argsort(probs)[::-1])[0:5]
for p in preds:
    print(class_names[p], probs[p])
