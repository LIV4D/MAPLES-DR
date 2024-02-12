# MAPLES-DR

**MAPLES-DR _(MESSIDOR Anatomical and Pathological Labels for Explainable Screening of Diabetic Retinopathy)_** is a public dataset which provides diagnoses for DR and ME as well as pixel-wise segmentation maps for 10 retinal structures for 198 images of MESSIDOR. This repository provides a python library of utility codes to ease the use of the database.

## MAPLES-DR Dataset Content
![Overview of the content of the MAPLES-DR dataset.](docs/source/_static/MAPLES-DR_Overview.svg)

MAPLES-DR dataset is available for download on figshare: [DOI:10.6084/m9.figshare.24328660](https://doi.org/10.6084/m9.figshare.24328660).

The fundus images are not included on MAPLES-DR because they are the property of the MESSIDOR consortium. One can download them from the [Consortium's website](https://www.adcis.net/fr/logiciels-tiers/messidor-fr/).

If you use the MAPLES-DR dataset, please cite the following paper:
```https://arxiv.org/abs/2402.04258```.

## MAPLES-DR Python Library

_The MAPLES-DR python library is still under developpement and will be released once MAPLES-DR paper is published._

### Install

```bash
git clone https://github.com/LIV4D/MAPLES-DR.git
pip install -e MAPLES-DR
```

If you plan to run the examples provided as Jupyter Notebooks in the `examples/` folder, you should 
also install their dependencies:
```bash
pip install -e "MAPLES-DR[examples]"
```

### Examples

 - `article_figures.ipynb` generates all the figures of MAPLES-DR paper.
