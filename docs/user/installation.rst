.. _installation:

============
Installation
============

TensorLayerX has some prerequisites that need to be installed first, including NumPy, Matplotlib and one of the frameworks (
`TensorFlow`_ , `MindSpore <https://www.mindspore.cn/>`__, `PaddlePaddle <https://www.paddlepaddle.org.cn/>`__,  `PyTorch <https://pytorch.org/>`__). For GPU
support CUDA and cuDNN are required.

If you run into any trouble, please check the `TensorFlow installation
instructions <https://www.tensorflow.org/versions/master/get_started/os_setup.html>`__,
`MindSpore installation instructions <https://www.mindspore.cn/install>`__,
`PaddlePaddle installation instructions <https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/windows-pip.html>`__,
`PyTorch installation instructions <https://pytorch.org/get-started/locally/>`__,
which cover installing the frameworks for a range of operating systems including
Mac OX, Linux and Windows, or ask for help on `tensorlayer@gmail.com <tensorlayer@gmail.com>`_
or `FAQ <http://tensorlayer.readthedocs.io/en/latest/user/more.html>`_.

Install TensorLayerX
=========================

For stable version:

.. code-block:: bash

  pip3 install tensorlayerx
  
  pip install tensorlayerx -i https://pypi.tuna.tsinghua.edu.cn/simple  (faster in China)

For latest version, please install from github.

.. code-block:: bash

  pip3 install git+https://github.com/tensorlayer/TensorLayerX.git

For developers, you should clone the folder to your local machine and put it along with your project scripts.

.. code-block:: bash

  git clone https://github.com/tensorlayer/TensorLayerX.git


Alternatively, you can build from the source.

.. code-block:: bash

  # First clone the repository and change the current directory to the newly cloned repository
  git clone https://github.com/tensorlayer/TensorLayerX.git
  cd tensorlayerx
  python setup.py install

This command will run the ``setup.py`` to install TensorLayerX.

.. _TensorFlow: https://www.tensorflow.org/versions/master/get_started/os_setup.html
.. _GitHub: https://github.com/tensorlayer/tensorlayer
.. _TensorLayer: https://github.com/tensorlayer/tensorlayer/