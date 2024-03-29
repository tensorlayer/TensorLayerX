#! /usr/bin/python
# -*- coding: utf-8 -*-

import os

import numpy as np

from tensorlayerx import logging
from tensorlayerx.vision import load_images
from tensorlayerx.files.utils import (del_file, folder_exists, load_file_list, maybe_download_and_extract)
logging.set_verbosity(logging.INFO)
__all__ = ['load_cyclegan_dataset']


def load_cyclegan_dataset(filename='summer2winter_yosemite', path='data'):
    """Load images from CycleGAN's database, see `this link <https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/>`__.

    Parameters
    ------------
    filename : str
        The dataset you want, see `this link <https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/>`__.
    path : str
        The path that the data is downloaded to, defaults is `data/cyclegan`

    Examples
    ---------
    >>> im_train_A, im_train_B, im_test_A, im_test_B = load_cyclegan_dataset(filename='summer2winter_yosemite')

    """
    path = os.path.join(path, 'cyclegan')
    url = 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/'

    logging.info("If can't download this dataset automatically, "
                  "please download it from the official website manually."
                  "cyclegan Dataset <https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/>."
                  "Please place dataset under 'data/cyclegan/' by default.")

    if folder_exists(os.path.join(path, filename)) is False:
        logging.info("[*] {} is nonexistent in {}".format(filename, path))
        maybe_download_and_extract(filename + '.zip', path, url, extract=True)
        del_file(os.path.join(path, filename + '.zip'))

    def load_image_from_folder(path):
        return load_images(path=path, n_threads=10)

    im_train_A = load_image_from_folder(os.path.join(path, filename, "trainA"))
    im_train_B = load_image_from_folder(os.path.join(path, filename, "trainB"))
    im_test_A = load_image_from_folder(os.path.join(path, filename, "testA"))
    im_test_B = load_image_from_folder(os.path.join(path, filename, "testB"))

    def if_2d_to_3d(images):  # [h, w] --> [h, w, 3]
        for i, _v in enumerate(images):
            if len(images[i].shape) == 2:
                images[i] = images[i][:, :, np.newaxis]
                images[i] = np.tile(images[i], (1, 1, 3))
        return images

    im_train_A = if_2d_to_3d(im_train_A)
    im_train_B = if_2d_to_3d(im_train_B)
    im_test_A = if_2d_to_3d(im_test_A)
    im_test_B = if_2d_to_3d(im_test_B)

    return im_train_A, im_train_B, im_test_A, im_test_B
