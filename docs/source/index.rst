*******************************
MAPLES-DR Dataset Documentation
*******************************


**MAPLES-DR** *(MESSIDOR Anatomical and Pathological Labels for Explainable Screening of Diabetic Retinopathy)* is a public dataset which provides expert-level diagnosis of DR and pixel-wise segmentation maps of 10 retinal structures.

For **198 fundus image** of the public dataset :doc:`MESSIDOR <welcome/messidor>`, our team of seven canadian senior retinologists graded :abbr:`DR (Diabetic Retinopathy)` and :abbr:`ME (Macular Edema)`, and segmented ten retinal structures related to those pathologies: **optic disc** and **cup**, **macula**, **vessels**, **micro-aneurysms**, **hemorrhages**, **neo-vessels**, **exudates**, **cotton wool spots** and **drusens**. A detailed description of those biomarkers and their implication in the diagnosis of DR can be found in the :doc:`dataset description section <welcome/dataset_description>` of this documentation. 
By releasing this dataset, we hope to help the AI community improves the explainability and reliability of machine learning models for DR screening.

.. figure:: _static/MAPLES-DR_Overview.svg
   :width: 800px
   :align: center

   Overview of MAPLES-DR content and annotation process. (Credit: :cite:t:`maples_dr`)


Note that, in the interest of time, some of the retinal structures were annotated by correcting AI generated segmentation map instead of labelling them from scratch. The complete annotation process is documented in `this paper <https://arxiv.org/abs/2402.04258>`_ :cite:`maples_dr` *(the URL currently refers to a temporary arxiv preprint, while our manuscript is under revisions.)*.


Usage
=====

The dataset is freely available for download through `MAPLES-DR Figshare repository <https://doi.org/10.6084/m9.figshare.24328660>`_. 

However, for machine learning usage we encourage researcher to directly download MAPLES-DR labels through ``maples_dr`` :doc:`python library <welcome/python_library>`. This library provides a simple API to load MAPLES-DR labels, and ease their integration with MESSIDOR original fundus images by automating the process of matching, cropping and resizing them to a uniform format.




If you wish to use this dataset in an academic work, we kindly ask you to cite the following `paper <https://arxiv.org/abs/2402.04258>`_ :cite:`maples_dr`::

      @article{maples_dr,
         title={MAPLES-DR: MESSIDOR Anatomical and Pathological Labels for Explainable Screening of Diabetic Retinopathy}, 
         author={Gabriel Lepetit-Aimon and Clément Playout and Marie Carole Boucher and Renaud Duval and Michael H Brent and Farida Cheriet},
         year={2024},
         eprint={2402.04258},
         archivePrefix={arXiv},
         primaryClass={eess.IV},
         doi={10.48550/arXiv.2402.04258}
      }

*(This citation currently refers to a temporary arxiv preprint while our manuscript is under revisions.)*

Additional Resources
====================

Segmentation Models
*******************

The segmentation models used to generate MAPLES-DR pre-annotation were originally implemented using Theano and we sadly can't publish them as they are unusable now. 

However we've publicly released some improved versions of those models as two python libraries, which bundle the weights and pytorch code required to automatically segment retinal vessels and lesions. These libraries were design to be used by researcher or clinician without deep learning expertise. They are available on github:

 - `fundus-vessels-toolkit <https://github.com/gabriel-lepetitaimon/fundus-vessels-toolkit>`_ for automatic segmentation and graph extraction of the retinal vasculature; 
 - `fundus-lesions-toolkit <https://github.com/ClementPla/fundus-lesions-toolkit>`_ for automatic semantic segmentation of microaneurysms, hemorrhages, exudates and :abbr:`CWS (Cotton Wool Spots)`.


Annotation Platform
*******************
The web-based annotation platform used to annotate MAPLES-DR is available on `github <https://github.com/LIV4D/AnnotationPlatform>`_.


Reference
=========

.. bibliography::
   :filter: docname in docnames

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Welcome

   Dataset Description <welcome/dataset_description>
   Python library <welcome/python_library>
   welcome/quickstart
   welcome/messidor

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Examples

   examples/visualisation


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API Reference

   api_reference/quick_api
   Configuration <api_reference/config>
   api_reference/dataset
   api_reference/loader
   api_reference/preprocessing

