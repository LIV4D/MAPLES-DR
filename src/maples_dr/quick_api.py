__all__ = ["configure", "export_test_set", "export_train_set", "load_test_set", "load_train_set"]
from pathlib import Path
from typing import Dict, List, Optional

from .dataset import Dataset, ImageField
from .loader import DatasetLoader, DatasetSubset

GLOBAL_LOADER = DatasetLoader()

configure = GLOBAL_LOADER.configure


def load_train_set() -> Dataset:
    """Load the training set.

    MAPLES-DR training set contains labels of pathological and anatomical structures for 138 fundus images.
    The dataset is return as a :class:`Dataset` object, which is equivalent to a list of samples.
    Each sample being stored as a dictionary with the following fields:

    - ``"fundus"``: The fundus image. (MESSIDOR path must have been configured!)

    - ``"optic_cup"``: The optic cup mask.
    - ``"optic_disc"``: The optic disc mask.
    - ``"macula"``: The macula mask.
    - ``"vessels"``: The red lesions mask.
    - ``"cotton_wool_spots"``: The cotton wool spots mask.
    - ``"exudates"``: The exudates mask.
    - ``"drusens"``: The drusens mask.
    - ``"bright_uncertain"``: The mask of bright lesions with uncertain type (CWS, exudates or drusens).
    - ``"microaneurysms"``: The microaneurysms mask.
    - ``"hemorrhages"``: The hemorrhages mask.
    - ``"neovascularization"``: The neovascularization mask.
    - ``"red_uncertains"``: The mask of red lesions with uncertain type (microaneuryms or hemorrhages).
    - ``"dr"``: The Diabetic Retinopathy grade (``'R0'``, ``'R1'``, ``'R2'``, ``'R3'``, ``'R4A'``).
      See :ref:`dr-me-grades` for more information.
    - ``"me"``: The Macular Edema grade (``'M0'``, ``'M1'``, ``'M2'``).
      See :ref:`dr-me-grades` for more information.


    Returns
    -------
    Dataset
        The training set.

    Examples
    --------
    >>> import maples_dr
    >>> maples_dr.configure(messidor_path="path/to/messidor.zip")
    >>> train_set = maples_dr.load_train_set()
    >>> for sample in train_set:
    >>>     fundus = sample["fundus"]           # The fundus image
    >>>     vessels = sample["vessels"]         # The vessels mask
    >>>     dr = sample["dr"]                   # The Diabetic Retinopathy grade
    """
    return GLOBAL_LOADER.load_dataset(DatasetSubset.TRAIN)


def load_test_set() -> Dataset:
    """Load the testing set.

    See :func:`load_train_set` for more details.

    Returns
    -------
    Dataset
        The testing set.
    """
    return GLOBAL_LOADER.load_dataset(DatasetSubset.TEST)


def load_dataset(subset: DatasetSubset | str | list[str] = DatasetSubset.ALL) -> Dataset:
    """Load specific subset of the dataset (by default load the whole dataset without the duplicates).

    Parameters
    ----------
    subset :
        The subset to load (see :class:`DatasetSubset` for the available options), or a list of image names.



    Returns
    -------
    Dataset
        The corresponding subset of MAPLES-DR. See :func:`load_train_set` for more details on the data format.
    """
    return GLOBAL_LOADER.load_dataset(subset)


def export_train_set(
    path: str | Path, fields: Optional[ImageField | List[ImageField] | Dict[ImageField, str]] = None
) -> None:
    """Save the training set to a folder.



    Parameters
    ----------
    path :
        The path to the directory where
    fields :
        The field or list of fields to save. If None (by default), all fields will be saved.

        See :class:`maples_dr.dataset.ImageField` for the list of available fields.


    Examples
    --------
    >>> import maples_dr
    >>> maples_dr.configure(messidor_path="path/to/messidor.zip")
    >>> export_train_set("path/to/save", fields=["fundus", "red_lesions", "vessels"])
    """
    return GLOBAL_LOADER.load_dataset("train").export(path, fields)


def export_test_set(path: str | Path, fields: Optional[ImageField | List[ImageField] | Dict[ImageField, str]] = None):
    """Save the testing set to a folder.



    Parameters
    ----------
    path :
        The path to save the dataset to.
    fields :
        The field or list of fields to save. If None (by default), all fields will be saved.

        See :class:`maples_dr.dataset.ImageField` for the list of available fields.


    Examples
    --------
    >>> import maples_dr
    >>> maples_dr.configure(messidor_path="path/to/messidor.zip")
    >>> export_test_set("path/to/save", fields=["fundus", "red_lesions", "vessels"])
    """
    return GLOBAL_LOADER.load_dataset("test").export(path, fields)


def clear_cache():
    """Clear the cache directory."""
    return GLOBAL_LOADER.clear_cache()


def clear_download_cache():
    """Clear the download cache directory."""
    return GLOBAL_LOADER.clear_download_cache()
