# MAPLES-DR

[![doc](https://github.com/LIV4D/MAPLES-DR/actions/workflows/documentation.yml/badge.svg?branch=dev)](https://liv4d.github.io/MAPLES-DR/index.html)

**MAPLES-DR _(MESSIDOR Anatomical and Pathological Labels for Explainable Screening of Diabetic Retinopathy)_** is a public dataset which provides diagnoses for DR and ME as well as pixel-wise segmentation maps for 10 retinal structures for 198 images of MESSIDOR. This repository provides a python library of utility codes to ease the use of the database.

## MAPLES-DR Dataset Content
![Overview of the content of the MAPLES-DR dataset.](docs/source/_static/MAPLES-DR_Overview.svg)

MAPLES-DR dataset is available for download on figshare: [DOI:10.6084/m9.figshare.24328660](https://doi.org/10.6084/m9.figshare.24328660). However, `maples_dr` python library provides a simple way to download and read the dataset.

The fundus images are not included on MAPLES-DR because they are the property of the MESSIDOR consortium. One can download them from the [Consortium's website](https://www.adcis.net/fr/logiciels-tiers/messidor-fr/).

If you use the MAPLES-DR dataset, please cite the following paper:
```https://arxiv.org/abs/2402.04258```.

## MAPLES-DR Python Library

_The MAPLES-DR python library is still under developpement and will be released once MAPLES-DR paper is published._

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
Once imported, MAPLES-DR train or test sets can be loaded in memory with a single line of Python code.

```python
import maples_dr
train_set = maples_dr.load_train_set()
test_set = maples_dr.load_test_set()
```

If necessary, the dataset archive is automatically downloaded from [Figshare](https://doi.org/10.6084/m9.figshare.24328660), extracted and cached locally. The data is then returned as a `Dataset` object assimilable to a list of samples stored as dictionnaries containing all MAPLES-DR labels. 

For example, the vessel map of the first sample of the train set can be accessed with:

```python
    vessel_map = train_set[0]['vessels']
```

---

Alternatively, MAPLES-DR images can be saved in local folders:

```python
    maples_dr.export_train_set('MAPLES-DR/train/')
    maples_dr.export_test_set('MAPLES-DR/test/')
```

---

For more information, please refer to the [documentation](https://liv4d.github.io/MAPLES-DR/index.html).

### Examples
The `examples/` folder contains Jupyter Notebooks that demonstrate how to use the MAPLES-DR python library.
 - `article_figures.ipynb` generates all the figures of MAPLES-DR paper.
 - `visualisation.ipynb` uses `maples-dr` to download and visualize the dataset labels.
