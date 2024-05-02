*******************************
MAPLES-DR Dataset Documentation
*******************************



**MAPLES-DR** *(MESSIDOR Anatomical and Pathological Labels for Explainable Screening of Diabetic Retinopathy)* is a public dataset which provides expert-level diagnosis of |DR| and pixel-wise segmentation maps of 10 retinal structures.

For **198 fundus image** of the public dataset :doc:`MESSIDOR <welcome/messidor>` :footcite:`MESSIDOR`, our team of seven Canadian senior retinologists graded :abbr:`DR (Diabetic Retinopathy)` and :abbr:`ME (Macular Edema)`, and segmented ten retinal structures related to those pathologies: **optic disc** and **cup**, **macula**, **vessels**, **micro-aneurysms**, **hemorrhages**, **neo-vessels**, **exudates**, **cotton wool spots** and **drusens**. A detailed description of those biomarkers and their role in the diagnosis of DR can be found in the :doc:`dataset description section <welcome/dataset_description>` of this documentation. 
By releasing this dataset, we hope to help the AI community improves the explainability and reliability of machine learning models for DR screening.

.. figure:: _static/MAPLES-DR_Overview.svg
   :width: 800px
   :align: center

   Overview of |MAPLES-DR| content and annotation process. (Credit: :footcite:t:`maples_dr`)


The annotation procedure relied on AI generated pre-segmentation of some retinal structures and a custom web-based annotation platform. The complete annotation process is documented in `this paper <https://arxiv.org/abs/2402.04258>`_ :footcite:`maples_dr` *(the URL currently refers to a temporary arxiv preprint, while the manuscript is under revision.)*.


Usage
=====

The dataset is freely available for download from the `MAPLES-DR Figshare repository <https://doi.org/10.6084/m9.figshare.24328660>`_. 

However, for machine learning usage we encourage researchers to directly download MAPLES-DR labels through :doc:`the python library <welcome/python_library>`: ``maples_dr``. This library provides a simple API to load MAPLES-DR labels, and eases their integration with the original fundus images of MESSIDOR, by automating the process of matching, cropping and resizing them to a uniform format. 

Note that the fundus images are the property of the MESSIDOR program partners and are not included in the MAPLES-DR dataset, but they are available to any research teams who requires them on `Messidor website <https://www.adcis.net/en/third-party/messidor/>`_. Follow the instructions in :doc:`MESSIDOR section <welcome/messidor>` to integrate them with |MAPLES-DR| labels.

If you wish to use this dataset in an academic work, we kindly ask you to footcite the following `paper <https://arxiv.org/abs/2402.04258>`_ :footcite:`maples_dr`::

      @article{maples_dr,
         title={MAPLES-DR: MESSIDOR Anatomical and Pathological Labels for Explainable Screening of Diabetic Retinopathy}, 
         author={Gabriel Lepetit-Aimon and Clément Playout and Marie Carole Boucher and Renaud Duval and Michael H Brent and Farida Cheriet},
         year={2024},
         eprint={2402.04258},
         archivePrefix={arXiv},
         doi={10.48550/arXiv.2402.04258}
      }

*(This citation currently refers to a temporary arxiv preprint while our manuscript is under revisions.)*

Additional Resources
====================

Segmentation Models
*******************
We released the segmentation models developed and trained on MAPLES-DR as two Python libraries: they bundle the weights and PyTorch code required to automatically segment retinal vessels and lesions. These libraries were designed to be used by researchers or clinicians without deep learning expertise. They are available on GitHub:

 - `fundus-vessels-toolkit <https://github.com/gabriel-lepetitaimon/fundus-vessels-toolkit>`_ for automatic segmentation and graph extraction of the retinal vasculature; 
 - `fundus-lesions-toolkit <https://github.com/ClementPla/fundus-lesions-toolkit>`_ for automatic semantic segmentation of microaneurysms, hemorrhages, exudates and |CWS|.


Annotation Platform
*******************
The web-based annotation platform used to annotate MAPLES-DR is also available on `GitHub <https://github.com/LIV4D/AnnotationPlatform>`_.

Reference
=========

.. footbibliography::


Acknowledgements
================
The LIV4D laboratory would like to thank Dr. Marie Carole Boucher, Dr. Michael H Brent, Dr. Renaud Duval as well as Dr. Karim Hammamji, Dr. Ananda Kalevar, Dr. Cynthia Qian, and  Dr. David Wong for their time and effort labeling the MAPLES-DR dataset. We also thank Dr. Fares Antaky and Dr. Daniel Milad for participating in a inter-observer variability study that helped us assess the quality of lesions segmentations of MAPLES-DR.

This study was funded by the Natural Science and Engineering Research Council of Canada as well as Diabetes Action Canada and FROUM (Fonds de recherche en ophtalmologie de l'Université de Montréal).

The original MESSIDOR dataset  was kindly provided by the Messidor program partners (see `https://www.adcis.net/en/third-party/messidor/ <https://www.adcis.net/en/third-party/messidor/>`_).

.. toctree::
   :hidden:

   MAPLES-DR Dataset <self>

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Documentation

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