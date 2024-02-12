from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypeAlias, Union

import numpy as np
import pandas as pd
from PIL import Image

from .config import DatasetConfig, ImageFormat, Preprocessing
from .utilities import Rect, RichProgress


class BiomarkerField(str, Enum):
    """Valid name of MAPLES-DR biomarkers

    - ``bright_lesions`` : Union of all the bright lesions masks (CWS, exudates, drusens).
    - ``bright_uncertains`` : The mask of bright lesions with uncertain type (CWS, exudates or drusens).
    - ``cotton_wool_spots`` : The cotton wool spots mask.
    - ``drusens`` : The drusens mask.
    - ``exudates`` : The exudates mask.
    - ``hemorrhages`` : The hemorrhages mask.
    - ``macula`` : The macula mask.
    - ``microaneurysms`` : The microaneurysms mask.
    - ``neovascularization`` : The neovascularization mask.
    - ``optic_cup`` : The optic cup mask.
    - ``optic_disc`` : The optic disc mask.
    - ``red_lesions`` : Union of all the red lesions masks (microaneuryms, hemorrhages).
    - ``red_uncertains`` : The mask of red lesions with uncertain type (microaneuryms, hemorrhages).
    - ``vessels`` : The vessels mask.
    """

    BRIGHT_LESIONS = "bright_lesions"
    BRIGHT_UNCERTAINS = "bright_uncertains"
    COTTON_WOOL_SPOTS = "cotton_wool_spots"
    DRUSENS = "drusens"
    EXUDATES = "exudates"
    HEMORRHAGES = "hemorrhages"
    MACULA = "macula"
    MICROANEURYSMS = "microaneurysms"
    NEOVASCULARIZATION = "neovascularization"
    OPTIC_CUP = "optic_cup"
    OPTIC_DISC = "optic_disc"
    RED_LESIONS = "red_lesions"
    RED_UNCERTAINS = "red_uncertains"
    VESSELS = "vessels"


# Correspondance between Biomarkers and their effective MAPLES-DR labels.
maples_dr_labels_correspondances: Dict[BiomarkerField, Tuple[str]] = {
    BiomarkerField.BRIGHT_LESIONS: ("CottonWoolSpots", "Exudates", "Drusens", "BrightUncertains"),
    BiomarkerField.BRIGHT_UNCERTAINS: ("BrightUncertains",),
    BiomarkerField.COTTON_WOOL_SPOTS: ("CottonWoolSpots",),
    BiomarkerField.DRUSENS: ("Drusens",),
    BiomarkerField.EXUDATES: ("Exudates",),
    BiomarkerField.HEMORRHAGES: ("Hemorrhages",),
    BiomarkerField.MACULA: ("Macula",),
    BiomarkerField.MICROANEURYSMS: ("Microaneurysms",),
    BiomarkerField.NEOVASCULARIZATION: ("Neovascularization",),
    BiomarkerField.OPTIC_CUP: ("OpticCup",),
    BiomarkerField.OPTIC_DISC: ("OpticDisc",),
    BiomarkerField.RED_LESIONS: ("Hemorrhages", "Microaneurysms", "RedUncertains"),
    BiomarkerField.RED_UNCERTAINS: ("RedUncertains",),
    BiomarkerField.VESSELS: ("Vessels",),
}


class FundusField(str, Enum):
    """
    Valid name for fields concerning fundus images

    - ``fundus`` : The preprocessed fundus image.
    - ``raw_fundus`` : The raw fundus image.
    """

    FUNDUS = "fundus"
    RAW_FUNDUS = "raw_fundus"


class DiagnosisField(str, Enum):
    """
    Valid name for fields concerning MAPLES-DR diagnosis labels

    - ``dr`` : The Diabetic Retinopathy grade. ('R0', 'R1', 'R2', 'R3', 'R4A')
    - ``me`` : The Macular Edema grade. ('M0', 'M1', 'M2')
    """

    DR = "dr"
    ME = "me"


#: The name of a valid image field in MAPLES-DR.
#:
#: This type alias is a union of :class:`BiomarkerField` and :class:`FundusField`.
ImageField: TypeAlias = Union[FundusField, BiomarkerField]

#: The name of a valid field in MAPLES-DR.
#:
#: This type alias is a union of :class:`DiagnosisField`, :class:`BiomarkerField`, and :class:`FundusField`.
Field: TypeAlias = Union[DiagnosisField, ImageField]


class Dataset:
    """A set of samples from the MAPLES-DR dataset.

    Datasets are a utility class to access and export samples from the MAPLES-DR dataset.

    They are equivalent to a list of samples, each sample being stored as a dictionary­.
    See :obj:`Field` for the list of available fields.
    """

    def __init__(self, data: pd.DataFrame, cfg: DatasetConfig, rois: Optional[Dict[str, Rect]] = None):
        """Create a new dataset. (Internal use only)

        :meta private:

        Parameters
        ----------
        data : pd.DataFrame
            The data of the dataset including the name of the samples and the paths to the images.
        cfg : DatasetConfig
            The configuration of the dataset.
        rois : Optional[Dict[str, Rect]], optional
            The region of interest in MESSIDOR fundus images.
        """
        self._cfg = cfg
        self._data: pd.DataFrame = data
        self._rois = rois

    def __getitem__(self, idx: int | str) -> dict:
        """Get a sample from the dataset.

        This method is a shortcut for :meth:`read_sample`.
        It returns the sample as a dictionary with the keys:
            - ``name`` containing the name of the image (e.g. "20051116_44816_0400_PP").
            - every :class:`Field` except ``bright_lesions`` and ``red_lesions``.

        The image are formatted following the default configuration (see :func:`maples_dr.configure`).

        Parameters
        ----------
        idx : int | str
            Index of the sample. Can be an integer or the name of the sample (e.g. "20051116_44816_0400_PP").

        Returns
        -------
        dict
            The sample as a dictionary.
        """
        return self.read_sample(idx)

    def __len__(self) -> int:
        return len(self._data)

    def available_fields(
        self, biomarkers=None, aggregated_biomarkers=None, diagnosis=None, fundus=None, raw_fundus=None
    ):
        any_true = (
            biomarkers is True
            or aggregated_biomarkers is True
            or diagnosis is True
            or fundus is True
            or raw_fundus is True
        )
        if not any_true:
            biomarkers = biomarkers is None
            aggregated_biomarkers = aggregated_biomarkers is None
            diagnosis = diagnosis is None
            fundus = fundus is None
            raw_fundus = raw_fundus is None

        fields = []
        if biomarkers:
            fields += [
                "optic_disc",
                "optic_cup",
                "macula",
                "vessels",
                "bright_uncertains",
                "exudates",
                "cotton_wool_spots",
                "drusens",
                "red_uncertains",
                "neovascularization",
                "microaneurysms",
                "hemorrhages",
            ]
        if aggregated_biomarkers:
            fields += ["red_lesions", "bright_lesions"]
        if diagnosis:
            fields += ["dr", "me"]
        if "fundus" in self._data.columns:
            if fundus:
                fields.append("fundus")
            if raw_fundus and self._cfg.preprocessing != "none":
                fields.append("raw_fundus")
        return fields

    def read_fundus(self, idx: str | int, preprocess=True, image_format: Optional[ImageFormat] = None):
        """Read a fundus image from the dataset.

        This requires the path to the MESSIDOR dataset to be configured.
        The image is cropped and resized to the size defined in the configuration (default: 1500x1500).

        Parameters
        ----------
        idx :
            Index of the image. Can be an integer or the name of the image.
        preprocess :
            If True, the image is preprocessed with the preprocessing defined in the configuration.
        image_format :
            Format of the image to return. If None, use the format defined in the configuration.
        """
        if "fundus" not in self._data.columns:
            raise RuntimeError(
                "Impossible to read fundus images, path to MESSIDOR dataset is unkown.\n"
                "Download MESSIDOR and use maples_dr.configure(messidor_path='...')."
            )

        # Read the image
        path = self.get_sample_infos(idx)["fundus"]
        img = Image.open(path)

        # Crop the image to the proper region of interest (stored in AdditionalData.zip/MESSIDOR-rois.yaml )
        if img.height != img.width:
            roi = self._rois[path.stem]
            img = img.crop(roi.box())

        # Resize the image
        img = img.resize((self._cfg.resize,) * 2)

        if preprocess:
            img = self.preprocess_fundus(img)

        # Apply format conversion
        if image_format is None:
            image_format = self._cfg.image_format
        if image_format == "PIL":
            return img
        else:
            img = np.array(img)
            if image_format == "rgb":
                return img
            elif image_format == "bgr":
                return img[..., ::-1]

        return img

    def preprocess_fundus(self, img: Image.Image, preprocessing: Optional[Preprocessing] = None):
        """Preprocess a fundus image.

        Parameters
        ----------
        img :
            Image to preprocess.
        preprocessing :
            Name of the preprocessing to apply.
            If None, use the preprocessing defined in the configuration.
        """
        if preprocessing is None:
            preprocessing = self._cfg.preprocessing
        if preprocessing == "none":
            return img
        elif preprocessing == "clahe":
            ...
        elif preprocessing == "median":
            ...
        else:
            raise ValueError(f"Unknown preprocessing: {preprocessing}. Must be either 'none', 'clahe' or 'median'.")

        return img

    def read_biomarkers(
        self,
        idx: str | int,
        biomarkers: Optional[
            BiomarkerField | List[BiomarkerField] | Dict[int, BiomarkerField | List[BiomarkerField]]
        ] = None,
        image_format: Optional[ImageFormat] = None,
    ):
        """
        Read biomarkers from the dataset.

        """

        # Read the paths of the biomarkers
        paths = self.get_sample_infos(idx)

        # Initialize the resulting biomarkers map

        # Cast the biomarkers definition to a dictionary
        if biomarkers is None:
            biomarkers = {i + 1: b for i, b in enumerate(self.available_fields(biomarkers=True))}
        if isinstance(biomarkers, (list, str)):
            biomarkers = {1: biomarkers}
            biomarkers_map = np.zeros(self._target_image_size, dtype=bool)
        elif isinstance(biomarkers, dict):
            biomarkers_map = np.zeros(self._target_image_size, dtype=np.uint8)
        else:
            raise ValueError(
                f"Unknown biomarkers: {biomarkers}." "Must be either a string, a list of strings or a dictionary."
            )

        # Check the validity of the biomarkers definition and expand merged biomarkers)
        valid_biomarkers = {}
        for i, ith_biomarkers in biomarkers.items():
            ith_valid_biomarkers = set()
            if isinstance(ith_biomarkers, str):
                ith_biomarkers = [ith_biomarkers]
            for b in ith_biomarkers:
                if b not in maples_dr_labels_correspondances:
                    raise ValueError(f"Unknown biomarker: {b}.")
                ith_valid_biomarkers.update(maples_dr_labels_correspondances[b])
            valid_biomarkers[i] = ith_valid_biomarkers

        # Read the biomarkers
        for i, ith_biomarkers in valid_biomarkers.items():
            for b in ith_biomarkers:
                path = paths[b]
                img = Image.open(path).resize(self._target_image_size)
                biomarkers_map[np.array(img) > 0] = i

        # Apply format conversion
        if image_format is None:
            image_format = self._cfg.image_format
        if image_format == "PIL":
            return Image.fromarray(biomarkers_map)
        else:
            return biomarkers_map

    def read_sample(
        self, idx: str | int, field: Optional[Field | List[Field]] = None, image_format: Optional[ImageFormat] = None
    ) -> dict:
        """
        Read a sample from the dataset.

        Parameters
        ----------
        idx :
            Index of the sample.
        field :
            Name of the field to read. If None, read the whole sample.
        """
        if field is None:
            field = self.available_fields(aggregated_biomarkers=False)

        fundus = None
        sample = {"name": self.get_sample_infos(idx).name}
        for f in field:
            if f in self.available_fields(biomarkers=True):
                sample[f] = self.read_biomarkers(idx, f, image_format=image_format)
            elif f == "fundus":
                if fundus is None:
                    fundus = self.read_fundus(idx, preprocess=False, image_format=image_format)
                sample[f] = self.preprocess_fundus(fundus)
            elif f == "raw_fundus":
                if fundus is None:
                    fundus = self.read_fundus(idx, preprocess=False, image_format=image_format)
                sample[f] = fundus
            elif f in ["dr", "me"]:
                sample[f] = self.get_sample_infos(idx)[f]
            else:
                raise ValueError(f"Unknown field: {f}.")
        return sample

    def get_sample_infos(self, idx: str | int) -> pd.Series:
        """
        Get the informations of a sample.

        :param idx: Index of the sample.

        :meta private:
        """
        if isinstance(idx, int):
            return self._data.iloc[idx]
        else:
            infos = self._data.loc[self._data.index == idx]
            if len(infos) == 0:
                raise ValueError(f"Unknown sample: {idx}.")
            return infos.iloc[0]

    def export(self, path: str | Path, fields: Optional[ImageField | List[ImageField] | Dict[ImageField, str]] = None):
        """Export the dataset to a folder.

        Parameters
        ----------
        path :
            Path of the folder where to export the dataset.
        fields :
            Fields to export. If None, export the whole dataset.
            If ``fields`` is a string or list, export only the given fields.
            If ``fields`` is a dictionary, export the fields given by the keys
            and rename their folder to their corresponding dictionnary values.
        """
        if fields is None:
            fields = {f: f for f in self.available_fields(biomarkers=True, fundus=True, raw_fundus=True)}
        elif isinstance(fields, str):
            fields = {fields: fields}
        elif isinstance(fields, list):
            fields = {f: f for f in fields}

        if isinstance(path, str):
            path = Path(path)

        with RichProgress.iteration("Exporting Maples-DR...", len(self), "Maples-DR exported in {t} seconds.") as p:
            for i in range(len(self)):
                sample = self.read_sample(i, list(fields.values()), image_format="PIL")

                for field, img in sample.items():
                    if field not in fields:
                        continue

                    field_folder = path / field
                    field_folder.mkdir(parents=True, exist_ok=True)

                    opt = {}
                    if field not in ("fundus", "raw_fundus"):
                        opt["bits "] = 1
                        opt["optimize"] = True

                    img.save(field_folder / f"{sample['name']}.png", **opt)
                    p.update(1 / len(fields))

    @property
    def _target_image_size(self):
        """
        Size of the images in the dataset.
        """
        return (self._cfg.resize,) * 2
