.. MAPLES-DR documentation master file, created by
   sphinx-quickstart on Wed Jan 17 09:27:07 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MAPLES-DR's documentation!
=====================================

**MAPLES-DR** *(MESSIDOR Anatomical and Pathological Labels for Explainable Screening of Diabetic Retinopathy)* is a public dataset which provides diagnoses for DR and ME as well as pixel-wise segmentation maps for 10 retinal structures of 198 fundus images from the public dataset MESSIDOR.

.. figure:: _static/MAPLES-DR_Overview.svg
   :width: 800px
   :align: center

|br|
The `maples_dr` packages documented here provides a python library to ease the use of MAPLES-DR dataset, especially for machine learning applications.

-----

.. toctree::
   :maxdepth: 1
   :caption: Welcome

   welcome/quickstart
   welcome/messidor

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/visualisation


.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api_reference/quick_api
   api_reference/loader
   api_reference/dataset



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
