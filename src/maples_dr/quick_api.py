__all__ = ["configure", "export_test_set", "export_train_set", "load_test_set", "load_train_set"]
from pathlib import Path
from typing import Optional

from .dataset import Dataset, ImageField
from .loader import DatasetLoader

GLOBAL_LOADER = DatasetLoader()

configure = GLOBAL_LOADER.configure


def load_train_set() -> Dataset:
    """Load the training set in memory.

    MAPLES-DR training set contains labels of pathological and anatomical structures for 138 fundus images.
    The dataset is return as a :class:`Dataset` object, which is equivalent to a list of samples.
    Each sample being stored as a dictionary with the following fields:
    - "fundus": The fundus image. (MESSIDOR path must have been configured!)

    - "optic_cup": The optic cup mask.
    - "optic_disc": The optic disc mask.
    - "macula": The macula mask.
    - "vessels": The red lesions mask.

    - "cotton_wool_spots": The cotton wool spots mask.
    - "exudates": The exudates mask.
    - "drusens": The drusens mask.
    - "bright_uncertain": The mask of bright lesions with uncertain type (CWS, exudates or drusens).

    - "microaneurysms": The microaneurysms mask.
    - "hemorrhages": The hemorrhages mask.
    - "neovascularization": The neovascularization mask.
    - "red_uncertains": The mask of red lesions with uncertain type (microaneuryms or hemorrhages).

    - "dr": The Diabetic Retinopathy grade ('R0', 'R1', 'R2', 'R3', 'R4A').
    - "me": The Macular Edema grade ('M0', 'M1', 'M2').


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
    return GLOBAL_LOADER.dataset("train")


def load_test_set() -> Dataset:
    """Load the testing set in memory.

    See :func:`load_train_set` for more details.

    Returns
    -------
    Dataset
        The testing set.
    """
    return GLOBAL_LOADER.dataset("test")


def export_train_set(
    path: str | Path, fields: Optional[ImageField | list[ImageField] | dict[ImageField, str]] = None
) -> None:
    """Save the training set to a folder.



    Parameters
    ----------
    path : str | Path
        The path to the directory where
    fields : Optional[ImageField  |  list[ImageField]  |  dict[ImageField, str]], optional
        The fields to save. If None, all fields will be saved.
        By default None.


    Examples
    --------
    >>> import maples_dr
    >>> maples_dr.configure(messidor_path="path/to/messidor.zip")
    >>> export_train_set("path/to/save", fields=["fundus", "red_lesions", "vessels"])
    """
    return GLOBAL_LOADER.dataset("train").export(path, fields)


def export_test_set(path: str | Path, fields: Optional[ImageField | list[ImageField] | dict[ImageField, str]] = None):
    """Save the testing set to a folder.



    Parameters
    ----------
    path : str | Path
        The path to save the dataset to.
    fields : Optional[ImageField  |  list[ImageField]  |  dict[ImageField, str]], optional
        The fields to save. If None, all fields will be saved.
        By default None.


    Examples
    --------
    >>> import maples_dr
    >>> maples_dr.configure(messidor_path="path/to/messidor.zip")
    >>> export_test_set("path/to/save", fields=["fundus", "red_lesions", "vessels"])
    """
    return GLOBAL_LOADER.dataset("test").export(path, fields)