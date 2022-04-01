#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import os

__all__ = [
    'load_image',
    'save_image',
    'load_images',
    'save_images',
]

def load_image(path):
    '''Load an image

    Parameters
    ----------
    path : str
        path of the image.

    Returns : numpy.ndarray
    -------
        a numpy RGB image


    Examples
    ----------
    With TensorLayerX

    >>> import tensorlayerx as tlx
    >>> path = './data/1.png'
    >>> image = tlx.vision.load_image(path)
    >>> print(image)

    '''
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Please check 'path'. 'Path' cannot contain Chinese")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_image(image, file_name, path):
    '''Save an image

    Parameters
    ----------
    image : numpy.ndarray
        The image to save
    file_name : str
        image name to save
    path : str
        path to save image

    Examples
    ----------
    With TensorLayerX

    >>> import tensorlayerx as tlx
    >>> load_path = './data/1.png'
    >>> save_path = './test/'
    >>> image = tlx.vision.load_image(path)
    >>> tlx.vision.save_image(image, file_name='1.png',path=save_path)

    '''
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(path, file_name), image)

def load_images(path):
    '''Load images from file

    Parameters
    ----------
    path : str
        path of the images.

    Returns : list
    -------
        a list of numpy RGB images

    Examples
    ----------
    With TensorLayerX

    >>> import tensorlayerx as tlx
    >>> load_path = './data/'
    >>> image = tlx.vision.load_images(path)
    '''
    images = []
    files = sorted(os.listdir(path))
    for file in files:
        image = cv2.imread(os.path.join(path, file))
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    return images



def save_images(images, file_names, path):
    '''Save images

    Parameters
    ----------
    images : list
        a list of numpy RGB images
    file_names : list
        a list of image names to save
    path : str
        path to save images

    Examples
    ----------
    With TensorLayerX

    >>> import tensorlayerx as tlx
    >>> load_path = './data/'
    >>> save_path = './test/'
    >>> images = tlx.vision.load_images(path)
    >>> name_list = user_define
    >>> tlx.vision.save_images(images, file_names=name_list,path=save_path)
    '''
    if len(images) != len(file_names):
        raise ValueError(" The number of images should be equal to the number of file names.")
    for i in range(len(file_names)):
        images[i] = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(path, str(file_names[i])), images[i])