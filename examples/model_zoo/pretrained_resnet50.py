#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
ResNet50 for ImageNet using TL model

"""

import time
import numpy as np
import tensorlayerx as tlx
from examples.model_zoo.imagenet_classes import class_names
from examples.model_zoo.resnet import ResNet50

tlx.logging.set_verbosity(tlx.logging.DEBUG)

# get the whole model
resnet = ResNet50(pretrained=True)
resnet.set_eval()

img1 = tlx.vision.load_image('data/tiger.jpeg')
img1 = tlx.vision.transforms.Resize((224, 224))(img1)[:, :, ::-1]
img1 = img1 - np.array([103.939, 116.779, 123.68]).reshape((1, 1, 3))

img1 = img1.astype(np.float32)[np.newaxis, ...]

start_time = time.time()
output = resnet(img1)
prob = tlx.softmax(output)[0].numpy()
print("  End time : %.5ss" % (time.time() - start_time))
preds = (np.argsort(prob)[::-1])[0:5]
for p in preds:
    print(class_names[p], prob[p])
