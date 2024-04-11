****************************************
Using MESSIDOR fundus images
****************************************

The fundus images corresponding to the diagnostic labels and biomarker segmentation maps of MAPLES-DR are the property of the MESSIDOR consortium and are therefore not included in the dataset. However they are freely available to any research team who requires them on the `Consortium's website <https://www.adcis.net/en/third-party/messidor/>`_. This page will guide you through the setup of including MESSIDOR fundus images along MAPLES-DR labels.

Download MESSIDOR fundus images
================================

The MESSIDOR fundus images can be downloaded from the `Consortium's website <https://www.adcis.net/en/third-party/messidor/>`_. The Consortium requires you to fill a form with your personal informations, agree to the terms of license agreement, and verify your email address before downloading the images. 

The download page contains links to several files. The fundus images are divided into 12 zip archives named ``Base__ images (zip)``. Make sure to download all 12 archives (`Base11.zip`, `Base12.zip`, ...,  `Base34.zip`) to the same directory.


Configure MESSIDOR local path
=============================

Once you have downloaded the images, you can include them in the dataset by calling:

.. code-block:: python

    from maples_dr import configure
    configure(messidor_path='/path/to/messidor/download/directory/')


Upon calling :func:`maples_dr.load_train_set` or :func:`maples_dr.load_test_set`, the 200 fundus images related to MAPLES-DR will be extracted, cropped and resized to match MAPLES-DR resolution. They will be accessible under the field ``"fundus"`` along the other MAPLES-DR biomarkers.

.. code-block:: python

    from maples_dr import load_train_set
    train_set = load_train_set()
    fundus = train_set[0]["fundus"]


Similarly, if you export the dataset to local files with :func:`maples_dr.export_train_set` or  :func:`maples_dr.export_test_set`, a folder named ``fundus`` will be created in the same directory as the other biomarkers, containing the fundus images at the appropriate resolution.

Acknowledgements
================

The MESSIDOR dataset was kindly provided by the Messidor program partners (see `https://www.adcis.net/en/third-party/messidor/ <https://www.adcis.net/en/third-party/messidor/>`_).

We remind that users of the MESSIDOR database (and therefore MAPLES-DR users) are encouraged to reference the following article::

    Decencière et al.. Feedback on a publicly distributed database: the Messidor database.
    Image Analysis & Stereology, v. 33, n. 3, p. 231-234, aug. 2014. ISSN 1854-5165.
    Available at: http://www.ias-iss.org/ojs/IAS/article/view/1155 or http://dx.doi.org/10.5566/ias.1155.