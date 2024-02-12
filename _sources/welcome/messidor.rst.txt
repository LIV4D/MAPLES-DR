****************************************
Using MESSIDOR fundus images
****************************************

The fundus images corresponding to the diagnostic labels and biomarker segmentation maps of MAPLES-DR are the property of the MESSIDOR consortium and are therefore not included in the dataset. However they are freely available to any research team who requires them on the `Consortium's website <https://www.adcis.net/en/third-party/messidor/>`_. This page will guide you through the setup of including MESSIDOR fundus images along MAPLES-DR labels.

Download MESSIDOR fundus images
================================

The MESSIDOR fundus images can be downloaded from the `Consortium's website <https://www.adcis.net/en/third-party/messidor/>`_. The Consortium requires you to fill a form with your personal informations, agree to the terms of license agreement, and verify your email address before downloading the images. 

The download page contains links to several zip files. The 12 links named `Base__ images (zip)` must be downloaded to the same directory. This directory should therefore contains the 12 archives: `Base11.zip`, `Base12.zip`, ...,  `Base34.zip`.


Configure MESSIDOR local path
=============================

Once you have downloaded the images, you can include them in the dataset by calling:

.. code-block:: python

    from maples_dr import configure
    configure(messidor_path='/path/to/messidor/download/directory/')


Upon calling :func:`maples_dr.load_train_set` or :func:`maples_dr.load_test_set`, the 200 fundus images related to MAPLES-DR will be extracted, cropped and resized to the appropriate resolution, and finally included along the MAPLES-DR biomarkers under the name `fundus`.

.. code-block:: python

    from maples_dr import load_train_set
    train_set = load_train_set()
    fundus = train_set[0]['fundus']


Similarly, if you export the dataset to local files with :func:`maples_dr.export_train_set` or  :func:`maples_dr.export_test_set`, a folder named `fundus` will be created in the same directory as the other biomarker, containing the fundus images.