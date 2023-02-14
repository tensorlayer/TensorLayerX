#! /usr/bin/python
# -*- coding: utf-8 -*-

import os

from tensorlayerx import logging
from tensorlayerx.files.utils import maybe_download_and_extract
logging.set_verbosity(logging.INFO)
__all__ = ['load_nietzsche_dataset']


def load_nietzsche_dataset(path='data'):
    """Load Nietzsche dataset.

    Parameters
    ----------
    path : str
        The path that the data is downloaded to, defaults is ``data/nietzsche/``.

    Returns
    --------
    str
        The content.

    Examples
    --------
    >>> see tutorial_generate_text.py
    >>> words = tlx.files.load_nietzsche_dataset()
    >>> words = basic_clean_str(words)
    >>> words = words.split()

    """
    logging.info("If can't download this dataset automatically, "
                 "please download it from the official website manually."
                 "nietzsche Dataset <https://s3.amazonaws.com/text-datasets/nietzsche.txt>."
                 "Please place dataset under 'data/nietzsche/' by default.")
    path = os.path.join(path, 'nietzsche')

    filename = "nietzsche.txt"
    url = 'https://s3.amazonaws.com/text-datasets/'
    filepath = maybe_download_and_extract(filename, path, url)

    with open(filepath, "r") as f:
        words = f.read()
        return words
