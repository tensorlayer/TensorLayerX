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
    '''

    Parameters
    ----------
    path : str
        path of the image.

    Returns : numpy.ndarray
    -------
        a numpy RGB image

    '''
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Please check 'path'. 'Path' cannot contain Chinese")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_image(image, file_name, path):
    '''

    Parameters
    ----------
    image : numpy.ndarray
        The image to save
    file_name : str
        image name to save
    path : str
        path to save image

    '''
    cv2.imwrite(os.path.join(path, file_name), image)

def load_images(path):
    '''

    Parameters
    ----------
    path : str
        path of the images.


    Returns : list
    -------
        a list of numpy RGB images
    '''
    images = []
    files = os.listdir(path)
    for file in files:
        image = cv2.imread(os.path.join(path, file))
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    return images



def save_images(images, file_names, path):

    '''

    Parameters
    ----------
    images : list
        a list of numpy RGB images
    file_names : list
        a list of image names to save
    path : str
        path to save images

    '''
    if len(images) != len(file_names):
        raise ValueError(" The number of images should be equal to the number of file names.")
    for i in range(len(file_names)):
        cv2.imwrite(os.path.join(path, str(file_names[i])), images[i])