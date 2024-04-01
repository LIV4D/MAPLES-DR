*************************************
Welcome to MAPLES-DR's documentation!
*************************************

**MAPLES-DR** *(MESSIDOR Anatomical and Pathological Labels for Explainable Screening of Diabetic Retinopathy)* is a public dataset which provides diagnoses for DR and ME as well as pixel-wise segmentation maps for 10 retinal structures of 198 fundus images from the public dataset MESSIDOR.


The `maples_dr <https://github.com/LIV4D/MAPLES-DR/>`_ package documented here provides a python library to ease the use of MAPLES-DR dataset, especially for machine learning applications.

MAPLES-DR Dataset Content
=========================

.. figure:: _static/MAPLES-DR_Overview.svg
   :width: 800px
   :align: center

If you use the MAPLES-DR dataset, please cite the following paper:
``https://arxiv.org/abs/2402.04258``.


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
   Configuration <api_reference/config>
   api_reference/dataset
   api_reference/loader
   api_reference/preprocessing



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
