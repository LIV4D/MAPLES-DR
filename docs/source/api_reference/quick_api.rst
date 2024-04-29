********************
MAPLES-DR Quick API
********************

Downloading, formating and saving MAPLES-DR data in memory or on the disk can be done quite simply using `maples_dr` quick API.

All the following function are imported either from `maples_dr.quick_api` or directly from the `maples_dr` module:

.. code-block:: python

    from maples_dr import *
    # or
    from maples_dr import configure, export_train_set, export_test_set


----------------

.. autofunction:: maples_dr.configure

.. autofunction:: maples_dr.load_train_set

.. autofunction:: maples_dr.load_test_set

.. autofunction:: maples_dr.load_dataset

.. autofunction:: maples_dr.export_train_set

.. autofunction:: maples_dr.export_test_set

.. autofunction:: maples_dr.clear_cache

.. autofunction:: maples_dr.clear_download_cache
