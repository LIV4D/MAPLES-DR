************
Quick Start
************

The MAPLES-DR dataset is available for download on `Figshare <https://doi.org/10.6084/m9.figshare.24328660>`_. But in the context of training machine learning algorithms we strongly recommend users to install the `maples_dr` python package to easily download, format and manipulate the dataset.

This page will guide you through the installation process and basic usage of the package: loading the dataset in memory or saving it in a local folder.

Installation
============

The `maples-dr` package is available on PyPI and can be installed using pip:

.. code-block:: console

    $ pip install maples-dr


Basic Usage
============
Once imported, MAPLES-DR train or test sets can be loaded in memory with a single line of Python code.

.. code-block:: python

    import maples_dr
    train_set = maples_dr.load_train_set()
    test_set = maples_dr.load_test_set()

If necessary, the dataset archive is automatically downloaded from `Figshare <https://doi.org/10.6084/m9.figshare.24328660>`_, extracted and cached locally. The data is then returned as a :class:`maples_dr.Dataset` which takes the form of a list of dictionaries containing all MAPLES-DR labels. (For more information, see the :doc:`../api_reference/dataset` class documentation). 

For example, the vessel map of the first sample of the train set can be accessed with:

.. code-block:: python

    vessel_map = train_set[0]['vessels']

------------

Alternatively, if you'd rather rely on your own code for data loading, MAPLES-DR images can be saved in local folders:

.. code-block:: python

    maples_dr.export_train_set('MAPLES-DR/train/')
    maples_dr.export_test_set('MAPLES-DR/test/')

As a result of these commands, all the labels of MAPLES-DR are saved as image files in their appropriate folders:
::

    MAPLES-DR/train/
    ├── bright_uncertains/
    │    ├── 20051019_38557_0100_PP.png
    │    ├── 20051020_55346_0100_PP.png
    │    └── ... (138 image files)
    ├── cotton_wool_spots/
    │    └── ... (138 image files)
    ├── drusens/
    ├── exudates/
    ├── hemorrhages/
    ├── macula/
    ├── microaneurysms/
    ├── neovascularization/
    ├── optic_cup/
    ├── optic_disc/
    ├── red_uncertains/
    └── vessels/

::

    MAPLES-DR/test/
    ├── bright_uncertains/
    │    ├── 20051019_38557_0100_PP.png
    │    └── ... (60 image files)
    ├── cotton_wool_spots/
    └── ...
    

Configure the datasets behavior
===============================

The dataset behavior can be tailored to ease the integration with your code or specific application. For instance, you might need the images and biomarker maps to have a specific resolution, a specific format (PIL image or numpy array), a specific channel order (`rgb` or `bgr`)...  The default behavior of the library is configured with the :func:`maples_dr.configure` method, and the configuration options are detailed in :class:`maples_dr.config.DatasetConfig` documentation.

The following example shows how to configure the dataset to return images as numpy arrays (instead of PIL images) and with a resolution of 512x512 pixels:

.. code-block:: python

    maples_dr.configure(resize=512, image_format="rgb")



The same method can be used to specify a local path where the library should read MAPLES-DR data, instead of downloading them from Figshare.

.. code-block:: python

    maples_dr.configure(
        maples_dr_path="path/to/MAPLES-DR/AdditionalData.zip",
        maples_dr_diagnosis_path="path/to/MAPLES-DR/diagnosis.xls"
    )

Finally, a local path to the MESSIDOR dataset can also be specified with this function in order to include the fundus images from MESSIDOR along with the MAPLES-DR labels. (See :doc:`../welcome/messidor` for more details.)

.. code-block:: python

    maples_dr.configure(messidor_path="path/to/Messidor/directory/")

------------

For more information on all the methods presented in this quick start, please refer to :doc:`../api_reference/quick_api` documentation.
