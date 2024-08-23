# MAPLES-DR

[![Version](https://img.shields.io/pypi/v/maples_dr.svg?logo=pypi)](https://pypi.python.org/pypi/maples_dr)
[![doc](https://github.com/LIV4D/MAPLES-DR/actions/workflows/documentation.yml/badge.svg?branch=dev)](https://liv4d.github.io/MAPLES-DR/en/)


**[MAPLES-DR](https://liv4d.github.io/MAPLES-DR/en/) _(MESSIDOR Anatomical and Pathological Labels for Explainable Screening of Diabetic Retinopathy)_** is a public dataset which provides diagnoses for DR and ME as well as pixel-wise segmentation maps for 10 retinal structures for 198 images of MESSIDOR. This repository provides a python library of utility codes to ease the use of the database.

## MAPLES-DR Dataset Content
![Overview of the content of the MAPLES-DR dataset.](docs/source/_static/MAPLES-DR_Overview.svg)

MAPLES-DR dataset is available for download on [figshare](https://doi.org/10.6084/m9.figshare.24328660). The fundus images are not included in MAPLES-DR but one can download them from [MESSIDOR Consortium's website](https://www.adcis.net/fr/logiciels-tiers/messidor-fr/).

If you wish to use this dataset in an academic work, we kindly ask you to cite the following [paper](https://doi.org/10.1038/s41597-024-03739-6):

```bibtex
@article{maples_dr,
   title={MAPLES-DR: MESSIDOR Anatomical and Pathological Labels for Explainable Screening of Diabetic Retinopathy},
   author={Gabriel Lepetit-Aimon and Clément Playout and Marie Carole Boucher and Renaud Duval and Michael H Brent and Farida Cheriet},
   journal={Scientific Data},
   volume={11},
   number={1},
   pages={914},
   year={2024},
   month={08},
   publisher={Nature Publishing Group UK London},
   doi={10.1038/s41597-024-03739-6}
}
```


## MAPLES-DR Python Library

The `maples_dr` python library provides a simple way to download and use the dataset. It was designed with machine learning applications in mind.

### Install

```bash
pip install maples-dr
```

If you plan to run the examples provided as Jupyter Notebooks in the `examples/` folder, you should 
also install their dependencies:
```bash
pip install maples-dr[examples]
```

### Simple Usage

#### Load MAPLES-DR in memory

Once imported, MAPLES-DR train or test sets can be loaded in memory with a single line of Python code.

```python
import maples_dr
train_set = maples_dr.load_train_set()
test_set = maples_dr.load_test_set()
```

If necessary, the dataset archive is automatically downloaded from [Figshare](https://doi.org/10.6084/m9.figshare.24328660), extracted and cached locally. The data is then returned as a [`Dataset`](https://liv4d.github.io/MAPLES-DR/api_reference/dataset.html) object similar to a list of samples stored as dictionaries containing all MAPLES-DR labels. 

For example, the vessel map of the first sample of the train set can be accessed with:

```python
vessel_map = train_set[0]['vessels']
```

#### Export MAPLES-DR to a local folder
Alternatively, MAPLES-DR images can be saved in local folders:

```python
maples_dr.export_train_set('MAPLES-DR/train/')
maples_dr.export_test_set('MAPLES-DR/test/')
```


For more information on how to configure image resolution and format, on how to include the fundus images along the labels, and more advanced functionalities, have a look at [the documentation of `maples_dr`](https://liv4d.github.io/MAPLES-DR/en/welcome/python_library.html)!

### Examples
The `examples/` folder contains Jupyter Notebooks that demonstrate how to use the `maples_dr` python library.
 - [`article_figures.ipynb`](examples/article_figures.ipynb) generates all the figures of MAPLES-DR paper.
 - [`display_biomarkers.ipynb`](examples/display_biomarkers.ipynb) uses `maples-dr` to download and visualize the anatomical and pathological structures of the retina on top of the fundus images from MAPLES-DR.

## Acknowledgements
The LIV4D laboratory would like to thank Dr. Marie Carole Boucher, Dr. Michael H Brent, Dr. Renaud Duval as well as Dr. Karim Hammamji, Dr. Ananda Kalevar, Dr. Cynthia Qian, and  Dr. David Wong for their time and effort labeling the MAPLES-DR dataset. We also thank Dr. Fares Antaky and Dr. Daniel Milad for participating in a inter-observer variability study that helped us assess the quality of lesions segmentations of MAPLES-DR.

This study was funded by the Natural Science and Engineering Research Council of Canada as well as Diabetes Action Canada and FROUM (Fonds de recherche en ophtalmologie de l'Université de Montréal).

The original MESSIDOR dataset  was kindly provided by the Messidor program partners (see [https://www.adcis.net/en/third-party/messidor/](https://www.adcis.net/en/third-party/messidor/) ).