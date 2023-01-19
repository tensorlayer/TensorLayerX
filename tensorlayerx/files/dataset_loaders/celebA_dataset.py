#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import zipfile
from tensorlayerx import logging
from tensorlayerx.files.utils import (download_file_from_google_drive, exists_or_mkdir, load_file_list)
logging.set_verbosity(logging.INFO)
__all__ = ['load_celebA_dataset']


def load_celebA_dataset(path='data'):
    """Load CelebA dataset

    Return a list of image path.

    Parameters
    -----------
    path : str
        The path that the data is downloaded to, defaults is ``data/celebA/``.

    """
    logging.info("The dataset is stored on google drive, if you can't download it from google drive, "
                  "please download it from the official website manually. " 
                  "Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>. "
                  "Please place dataset 'img_align_celeba.zip' under 'data/celebA/' by default.")

    data_dir = 'celebA'
    filename, drive_id = "img_align_celeba.zip", "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
    file_path = os.path.join(path, data_dir)
    image_path = os.path.join(path, data_dir, "img_align_celeba")
    save_path = os.path.join(path, data_dir, filename)
    if os.path.exists(image_path):
        logging.info('[*] {} already exists'.format(image_path))
    else:
        if not os.path.exists(save_path):
            exists_or_mkdir(file_path)
            download_file_from_google_drive(drive_id, save_path)
            zip_dir = ''
        with zipfile.ZipFile(save_path) as zf:
            zip_dir = zf.namelist()[0]
            zf.extractall(file_path)
        # os.remove(save_path)
        # os.rename(os.path.join(path, zip_dir), image_path)

    data_files = load_file_list(path=image_path, regx='\\.jpg', printable=False)
    for i, _v in enumerate(data_files):
        data_files[i] = os.path.join(image_path, data_files[i])
    return data_files
