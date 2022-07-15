API - Metrics
==================

The tensorlayerx.metrics directory contians Accuracy, Auc, Precision and Recall.
For more complex metrics, you can encapsulates metric logic and APIs by base class.


.. automodule:: tensorlayerx.metrics

Metric list
-------------

.. autosummary::

   Metric
   Accuracy
   Auc
   Precision
   Recall
   acc



Metric
^^^^^^^^^^^^^^^^
.. autoclass:: Metric


Accuracy
""""""""""""""""""""""""""
.. autoclass:: Accuracy
    :members:


Auc
""""""""""""""""""""""""""
.. autoclass:: Auc
    :members:


Precision
""""""""""""""""""""""""""
.. autoclass:: Precision
    :members:


Recall
""""""""""""""""""""""""""
.. autoclass:: Recall
    :members:


acc
""""""""""""""""""""""""""
.. autofunction:: acc
