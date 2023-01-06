.. _contributing:

===============
Contributing
===============

TensorLayerX is a major ongoing research project in Peking University and Pengcheng Laboratory, the first version was established at Imperial College London in 2016. The goal of the project is to develop a compositional language that is compatible with multiple deep learning frameworks,
while complex learning systems can be built through composition of neural network modules.

Numerous contributors come from various horizons such as: Peking University, Pengcheng Laboratory.

You can easily open a Pull Request (PR) on `GitHub <https://github.com/tensorlayer/TensorLayerX>`__, every little step counts and will be credited.
As an open-source project, we highly welcome and value contributions!

**If you are interested in working with us, please contact us at:** `tensorlayer@gmail.com <tensorlayer@gmail.com>`_.

.. image:: my_figs//join_slack.png
  :width: 30 %
  :align: center
  :target: https://join.slack.com/t/tensorlayer/shared_invite/enQtMjUyMjczMzU2Njg4LWI0MWU0MDFkOWY2YjQ4YjVhMzI5M2VlZmE4YTNhNGY1NjZhMzUwMmQ2MTc0YWRjMjQzMjdjMTg2MWQ2ZWJhYzc


Project Maintainers
--------------------------

TensorLayerX is a major ongoing research project in Peking University and Pengcheng Laboratory.

For TensorLayerX, it is now actively developing and maintaining by the following people *(in alphabetical order)*:

- **Cheng Lai** (`@Laicheng0830 <https://github.com/Laicheng0830>`_) - `<https://Laicheng0830.github.io>`_
- **Hao Dong** (`@zsdonghao <https://github.com/zsdonghao>`_) - `<https://zsdonghao.github.io>`_
- **Jiarong Han** (`@hanjr92 <https://github.com/hanjr92>`_) - `<https://hanjr92.github.io>`_

Numerous other contributors can be found in the `Github Contribution Graph <https://github.com/tensorlayer/TensorLayerX/graphs/contributors>`_.


What to contribute
------------------

Your method and example
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have a new method or example in terms of Deep learning or Reinforcement learning, you are welcome to contribute.

* Provide your layers or examples, so everyone can use it.
* Explain how it would work, and link to a scientific paper if applicable.
* Keep the scope as narrow as possible, to make it easier to implement.


Report bugs
~~~~~~~~~~~

Report bugs at the `GitHub <https://github.com/tensorlayer/TensorLayerX>`__  or `OpenI <https://git.openi.org.cn/OpenI/TensorLayerX>`__, we normally will fix it in 5 hours.
If you are reporting a bug, please include:

* your TensorLayerX, TensorFlow, MindSpore, PaddlePaddle, PyTorch and Python version.
* steps to reproduce the bug, ideally reduced to a few Python commands.
* the results you obtain, and the results you expected instead.

If you are unsure whether the behavior you experience is a bug, or if you are
unsure whether it is related to TensorLayer or TensorFlow, MindSpore, PaddlePaddle, PyTorch, please just ask on `our
mailing list`_ first.


Fix bugs
~~~~~~~~

Look through the GitHub issues or OpenI issues for bug reports. Anything tagged with "bug" is
open to whoever wants to implement it. If you discover a bug in TensorLayerX you can
fix yourself, by all means feel free to just implement a fix and not report it
first.


Write documentation
~~~~~~~~~~~~~~~~~~~

Whenever you find something not explained well, misleading, glossed over or
just wrong, please update it! The *Edit on Github* link on the top right of
every documentation page and the *[source]* link for every documented entity
in the API reference will help you to quickly locate the origin of any text.



How to contribute
-----------------

Edit on Github
~~~~~~~~~~~~~~

As a very easy way of just fixing issues in the documentation, use the
*Edit on Github* link on the top right of a documentation page or the *[source]* link
of an entity in the API reference to open the corresponding source file in
Github, then click the *Edit this file* link to edit the file in your browser
and send us a Pull Request. All you need for this is a free Github account.

For any more substantial changes, please follow the steps below to setup
TensorLayerX for development.


Documentation
~~~~~~~~~~~~~

The documentation is generated with `Sphinx
<http://sphinx-doc.org/latest/index.html>`_. To build it locally, run the
following commands:

.. code:: bash

    pip install Sphinx
    sphinx-quickstart

    cd docs
    make html

If you want to re-generate the whole docs, run the following commands:

.. code :: bash

    cd docs
    make clean
    make html


To write the docs, we recommend you to install `Local RTD VM <http://docs.readthedocs.io/en/latest/custom_installs/local_rtd_vm.html>`_.




Afterwards, open ``docs/_build/html/index.html`` to view the documentation as
it would appear on `readthedocs <http://tensorlayer.readthedocs.org/>`_. If you
changed a lot and seem to get misleading error messages or warnings, run
``make clean html`` to force Sphinx to recreate all files from scratch.

When writing docstrings, follow existing documentation as much as possible to
ensure consistency throughout the library. For additional information on the
syntax and conventions used, please refer to the following documents:

* `reStructuredText Primer <http://sphinx-doc.org/rest.html>`_
* `Sphinx reST markup constructs <http://sphinx-doc.org/markup/index.html>`_
* `A Guide to NumPy/SciPy Documentation <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_


Testing
~~~~~~~

TensorLayerX has a code coverage of 100%, which has proven very helpful in the past,
but also creates some duties:

* Whenever you change any code, you should test whether it breaks existing
  features by just running the test scripts.
* Every bug you fix indicates a missing test case, so a proposed bug fix should
  come with a new test that fails without your fix.


Sending Pull Requests
~~~~~~~~~~~~~~~~~~~~~

When you're satisfied with your addition, the tests pass and the documentation
looks good without any markup errors, commit your changes to a new branch, push
that branch to your fork and send us a Pull Request via Github's web interface.


When filing your Pull Request, please include a description of what it does, to
help us reviewing it. If it is fixing an open issue, say, issue #123, add
*Fixes #123*, *Resolves #123* or *Closes #123* to the description text, so
Github will close it when your request is merged.


.. _Release: https://github.com/tensorlayer/TensorLayerX/releases
.. _OpenI: https://git.openi.org.cn/OpenI/TensorLayerX
.. _our mailing list: hao.dong11@imperial.ac.uk
