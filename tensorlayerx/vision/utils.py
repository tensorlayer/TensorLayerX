#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
import threading


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

def load_images(path, n_threads = 10):
    '''Load images from file

    Parameters
    ----------
    path : str
        path of the images.
    n_threads : int
        The number of threads to read image.

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
    for file in range(0, len(files), n_threads):
        image_list = files[file:file + n_threads]
        image = threading_data(image_list, fn=load_image, path = path)
        images.extend(image)
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


def threading_data(data=None, fn=None, thread_count=None, path = None):
    """Process a batch of data by given function by threading.

    Usually be used for data augmentation.

    Parameters
    -----------
    data : numpy.array or others
        The data to be processed.
    thread_count : int
        The number of threads to use.
    fn : function
        The function for data processing.
    more args : the args for `fn`
        Ssee Examples below.

    Returns
    -------
    list or numpyarray
        The processed results.

    References
    ----------
    - `python queue <https://pymotw.com/2/Queue/index.html#module-Queue>`__
    - `run with limited queue <http://effbot.org/librarybook/queue.htm>`__

    """

    def apply_fn(results, i, data, path):
        path = os.path.join(path, data)
        results[i] = fn(path)

    if thread_count is None:
        results = [None] * len(data)
        threads = []
        # for i in range(len(data)):
        #     t = threading.Thread(name='threading_and_return', target=apply_fn, args=(results, i, data[i], kwargs))
        for i, d in enumerate(data):
            t = threading.Thread(name='threading_and_return', target=apply_fn, args=(results, i, d, path))
            t.start()
            threads.append(t)
    else:
        divs = np.linspace(0, len(data), thread_count + 1)
        divs = np.round(divs).astype(int)
        results = [None] * thread_count
        threads = []
        for i in range(thread_count):
            t = threading.Thread(
                name='threading_and_return', target=apply_fn, args=(results, i, data[divs[i]:divs[i + 1]], path)
            )
            t.start()
            threads.append(t)

    for t in threads:
        t.join()

    if thread_count is None:
        try:
            return np.asarray(results, dtype=object)
        except Exception:
            return results
    else:
        return np.concatenate(results)