from __future__ import annotations

import logging
import multiprocessing
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Tuple, TypeAlias, Union

import numpy as np
import pandas as pd
from PIL import Image
from PIL.Image import Resampling

from .config import DatasetConfig, ImageFormat, Preprocessing
from .preprocessing import fundus_roi, preprocess_fundus
from .utilities import Point, Rect, RichProgress, case_insensitive_parsing


class BiomarkerField(str, Enum):
    """String Enum of MAPLES-DR biomarkers fields."""

    #: The optic cup mask.
    OPTIC_CUP = "opticCup"

    #: The optic disc mask.
    OPTIC_DISC = "opticDisc"

    #: The macula mask.
    MACULA = "macula"

    #: The vessels mask.
    VESSELS = "vessels"

    #: Union of all the bright lesions masks (CWS, exudates, drusens and uncertain).
    BRIGHT_LESIONS = "brightLesions"

    #: The cotton wool spots mask.
    COTTON_WOOL_SPOTS = "cottonWoolSpots"

    #: The drusens mask.
    DRUSENS = "drusens"

    #: The exudates mask.
    EXUDATES = "exudates"

    #: The mask of bright lesions with uncertain type (either CWS, exudates or drusens).
    BRIGHT_UNCERTAINS = "brightUncertains"

    #: Union of all the red lesions masks (microaneurysm, hemorrhages and uncertain).
    RED_LESIONS = "redLesions"

    #: The hemorrhages mask.
    HEMORRHAGES = "hemorrhages"

    #: The microaneurysms mask.
    MICROANEURYSMS = "microaneurysms"

    #: The neovascularization mask.
    NEOVASCULARIZATION = "neovascularization"

    #: The mask of red lesions with uncertain type (either microaneurysm or hemorrhages).
    RED_UNCERTAINS = "redUncertains"

    @classmethod
    def parse(cls, biomarker: str | BiomarkerField) -> BiomarkerField:
        """Parse a biomarker from a string.

        This method accept case insensitive biomarker names and the following aliases:
        - ``red`` for :attr:`BiomarkerField.RED_LESIONS`;
        - ``bright`` for :attr:`BiomarkerField.BRIGHT_LESIONS`;
        - ``cup`` for :attr:`BiomarkerField.OPTIC_CUP`;
        - ``opticDisk``, ``disc`` or ``disk``  for :attr:`BiomarkerField.OPTIC_DISC`;
        - ``cws`` for :attr:`BiomarkerField.COTTON_WOOL_SPOTS`;
        - ``neovessels`` for :attr:`BiomarkerField.NEOVASCULARIZATION`.

        Parameters
        ----------
        biomarker :
            The name of the biomarker.

        Returns
        -------
        BiomarkerField
            The corresponding biomarker field.

        :meta private:
        """
        return case_insensitive_parsing(
            cls,
            biomarker,
            alias={
                "red": BiomarkerField.RED_LESIONS,
                "bright": BiomarkerField.BRIGHT_LESIONS,
                "cup": BiomarkerField.OPTIC_CUP,
                "opticdisk": BiomarkerField.OPTIC_DISC,
                "disc": BiomarkerField.OPTIC_DISC,
                "disk": BiomarkerField.OPTIC_DISC,
                "cws": BiomarkerField.COTTON_WOOL_SPOTS,
                "neovessels": BiomarkerField.NEOVASCULARIZATION,
            },
        )


AGGREGATED_BIOMARKERS = {
    BiomarkerField.BRIGHT_LESIONS: (BiomarkerField.COTTON_WOOL_SPOTS, BiomarkerField.EXUDATES, BiomarkerField.DRUSENS),
    BiomarkerField.RED_LESIONS: (BiomarkerField.MICROANEURYSMS, BiomarkerField.HEMORRHAGES),
}


class FundusField(str, Enum):
    """String Enum of MAPLES-DR fields associated with fundus images.

    .. warning::
        Path to MESSIDOR fundus images **must** be configured to use these fields!
        See :func:`maples_dr.configure` for more information.
    """

    #: The preprocessed fundus image (or the original fundus image if no preprocessing is applied).
    FUNDUS = "fundus"

    #: The raw fundus image. (If no preprocessing is applied, this is the same as :attr:`FundusField.FUNDUS`.)
    RAW_FUNDUS = "rawFundus"

    #: The mask of the fundus image.
    MASK = "mask"

    @classmethod
    def parse(cls, field: str | FundusField) -> FundusField:
        """Parse a field from a string.

        Parameters
        ----------
        field :
            The name of the field.

        Returns
        -------
        FundusField
            The corresponding field.

        :meta private:
        """
        return case_insensitive_parsing(cls, field, alias={"roi": FundusField.MASK})


class DiagnosisField(str, Enum):
    """String Enum of MAPLES-DR diagnosis fields."""

    #: The Diabetic Retinopathy grade.
    #:
    #:  - ``R0`` : No DR.
    #:  - ``R1`` : Mild Non-Proliferative DR.
    #:  - ``R2`` : Moderate Non-Proliferative DR.
    #:  - ``R3`` : Severe Non-Proliferative DR.
    #:  - ``R4A`` : Proliferative DR.
    #:  - ``R4S`` : Treated and Stable Proliferative DR.
    #:  - ``R6`` : Insufficient quality to grade.
    DR = "dr"

    #: The Macular Edema grade.
    #:
    #:  - ``M0`` : No ME.
    #:  - ``M1`` : ME without center involvement.
    #:  - ``M2`` : ME with center involvement.
    #:  - ``M6`` : Insufficient quality to grade.
    ME = "me"

    @classmethod
    def parse(cls, field: str | DiagnosisField) -> DiagnosisField:
        """Parse a field from a string.

        Parameters
        ----------
        field :
            The name of the field.

        Returns
        -------
        DiagnosisField
            The corresponding field.

        :meta private:
        """
        return case_insensitive_parsing(cls, field)


class BiomarkersAnnotationInfos(str, Enum):
    """Valid name for the additional information collected during the biomarkers annotations process."""

    #: The name of the retinologist who annotated the series of biomarkers.
    RETINOLOGIST = "retinologist"

    #: A comment from the retinologist.
    COMMENT = "comment"

    #: The time at which the annotation was made.
    ANNOTATION_TIME = "annotationTime"

    #: The id of the annotation. (1 is the first image to be annotated, 200 is the last one.)
    ANNOTATION_ID = "annotationID"

    @classmethod
    def parse(cls, infos: str | BiomarkersAnnotationInfos) -> BiomarkersAnnotationInfos:
        """Parse a field from a string.

        Parameters
        ----------
        infos :
            The name of the infos field.

        Returns
        -------
        BiomarkersAnnotationInfos
            The corresponding infos field.

        :meta private:
        """
        return case_insensitive_parsing(cls, infos)


class BiomarkersAnnotationTasks(str, Enum):
    """Valid name for the different annotation tasks in MAPLES-DR."""

    #: The annotation task for bright lesions (CWS, exudates, drusens).
    BRIGHT = "brightLesions"

    #: The annotation task for red lesions (microaneurysm, hemorrhages, neovessels).
    RED = "redLesions"

    #: The annotation task for the optic disc, optic cup and the macula.
    DISC_MACULA = "discMacula"

    #: The annotation task for the vessels.
    VESSELS = "vessels"


class Pathology(str, Enum):
    """Valid name for the different pathology in MAPLES-DR."""

    #: The Diabetic Retinopathy.
    DR = "dr"

    #: The Macular Edema.
    ME = "me"


#: The name of a valid image field in MAPLES-DR.
#:
#: This type alias is a union of :class:`BiomarkerField <maples_dr.dataset.BiomarkerField>` and :class:`FundusField <,maples_dr.dataset.FundusField>`.
ImageField: TypeAlias = Union[FundusField, BiomarkerField]

#: The name of a valid field in MAPLES-DR.
#:
#: This type alias is a union of :class:`DiagnosisField <maples_dr.dataset.DiagnosisField>`, :class:`BiomarkerField <maples_dr.dataset.BiomarkerField>`, and :class:`FundusField <maples_dr.dataset.FundusField>`.
Field: TypeAlias = Union[DiagnosisField, ImageField]


def check_field(field: Field | str) -> Field:
    try:
        return DiagnosisField.parse(field)
    except ValueError:
        pass
    try:
        return BiomarkerField.parse(field)
    except ValueError:
        pass
    try:
        return FundusField.parse(field)
    except ValueError:
        pass

    raise ValueError(
        f"Unknown field: {field}. Must be one of:\n"
        f"  - DiagnosisField: {', '.join(_.value for _ in DiagnosisField)}\n"
        f"  - BiomarkerField: {', '.join(_.value for _ in BiomarkerField)}\n"
        f"  - FundusField: {', '.join(_.value for _ in FundusField)}\n"
    )


def field2folder_name(field: str | Field | Path) -> str:
    if isinstance(field, Field):
        return field.value
    if isinstance(field, Path):
        field = str(field.resolve().absolute())
    assert isinstance(field, str), "Invalid folder name."
    return field


class Dataset(Sequence):
    """A set of samples from the MAPLES-DR dataset.

    Datasets are a utility class to access and export samples from the MAPLES-DR dataset.

    They are equivalent to a list of samples, each sample being stored as a dictionary­.
    See :class:`Field <maples_dr.dataset.Field>` for the list of available fields.
    """

    def __init__(self, data: pd.DataFrame, cfg: DatasetConfig, messidor_rois: pd.DataFrame):
        """Create a new dataset. (Internal use only)

        :meta private:

        Parameters
        ----------
        data : pd.DataFrame
            The data of the dataset. It should match :attr:`Dataset.data` format.

        cfg : DatasetConfig
            The configuration of the dataset.
        messidor_rois : pd.DataFrame
            A dataframe with the same index as ``data`` containing the columns:
            - x0: the x coordinate of the top-left corner of the region of interest;
            - y0: the y coordinate of the top-left corner of the region of interest;
            - x1: the x coordinate of the bottom-right corner of the region of interest;
            - y1: the y coordinate of the bottom-right corner of the region of interest;
            - W: the width of the original MESSIDOR fundus image;
            - H: the height of the original MESSIDOR fundus image.

        """
        self._cfg: DatasetConfig = cfg

        assert isinstance(data, pd.DataFrame), "Invalid type for 'data'."
        assert all(isinstance(_, str) for _ in data.index), "Invalid index type for 'data'."
        assert all(
            col.value in data.columns for col in BiomarkerField if col not in AGGREGATED_BIOMARKERS
        ), "Missing biomarker columns in the dataset."
        assert all(col.value in data.columns for col in DiagnosisField), "Missing diagnosis columns in the dataset."
        self._data: pd.DataFrame = data

        if messidor_rois is not None:
            assert isinstance(messidor_rois, pd.DataFrame), "Invalid type for 'messidor_rois'."
            assert data.index.isin(messidor_rois.index).all(), "Invalid index for 'messidor_rois'."
            assert "x0" in messidor_rois.columns, "Missing 'x0' column in the MESSIDOR ROIs dataframe."
            assert "y0" in messidor_rois.columns, "Missing 'y0' column in the MESSIDOR ROIs dataframe."
            assert "x1" in messidor_rois.columns, "Missing 'x1' column in the MESSIDOR ROIs dataframe."
            assert "y1" in messidor_rois.columns, "Missing 'y1' column in the MESSIDOR ROIs dataframe."
            assert "W" in messidor_rois.columns, "Missing 'W' column in the MESSIDOR ROIs dataframe."
            assert "H" in messidor_rois.columns, "Missing 'H' column in the MESSIDOR ROIs dataframe."
        self._rois: pd.DataFrame = messidor_rois

    def __getitem__(self, idx: int | str | slice[int] | list[str]) -> DataSample:
        """Get a sample from the dataset.

        The sample is returned as a :class:`DataSample <maples_dr.dataset.DataSample>`.


        Parameters
        ----------
        idx : int | str
            Index of the sample. Can be an integer or the name of the sample (e.g. "20051116_44816_0400_PP").

        Returns
        -------
        DataSample
            The sample.
        """
        if isinstance(idx, (str, int)):
            sample_infos = self.get_sample_infos(idx)
            rois = self._rois.loc[sample_infos.name]
            return DataSample(sample_infos, self._cfg, rois)
        elif isinstance(idx, slice):
            return self.subset(start=idx.start, end=idx.stop, step=idx.step)
        elif isinstance(idx, list):
            return self.subset_by_name(idx)
        else:
            raise ValueError(f"Invalid index type: {type(idx)}.")

    def __contains__(self, idx: str | int) -> bool:
        try:
            self.get_sample_infos(idx)
            return True
        except ValueError:
            return False

    def __len__(self) -> int:
        return len(self._data)

    @property
    def data(self) -> pd.DataFrame:
        """The data of the dataset.

        A dataframe containing the information of each sample. It has the following columns:

            - index: the name of the sample.
            - ``fundus``: the path to the fundus image.
            - :class:`BiomarkerField <maples_dr.dataset.BiomarkerField>` (accept aggregated): the path to the biomarkers masks.
            - :class:`BiomarkersAnnotationTasks <maples_dr.dataset.BiomarkersAnnotationTasks>` _ :class:`BiomarkersAnnotationInfos <maples_dr.dataset.BiomarkersAnnotationInfos>`: the additional annotations informations.
            - ``dr``: the consensus DR grade.
            - ``me``: the consensus ME grade.
            - ``dr_{A|B|C}``: the DR grade given by one retinologist.
            - ``me_{A|B|C}``: the ME grade given by one retinologist.
            - ``dr_{A|B|C}_comment``: comments from the retinologist when grading the DR diagnosis.
        """
        return self._data

    def keys(self) -> List[str]:
        """Get the names of the samples in the dataset.

        Returns
        -------
        List[str]
            The names of the samples.
        """
        return self._data.index.tolist()

    def subset(
        self, *arg, start: Optional[int] = None, end: Optional[int] = None, step: Optional[int] = None
    ) -> Dataset:
        """Get a subset of the dataset.

        Parameters
        ----------
        start : int
            The index of the first sample of the subset.
        end : int
            The index of the last sample of the subset.

        Returns
        -------
        Dataset
            The subset of the dataset.
        """
        if start is None and end is None:
            if step is None and len(arg) == 3:
                start, end, step = arg
            elif len(arg) == 2:
                start, end = arg
            elif len(arg) == 1:
                start = 0
                end = arg[0]
            else:
                raise ValueError("Invalid number of arguments: only expect start and end indexes.")
        elif start is None and end is not None and len(arg) == 1:
            start = arg[0]
        elif len(arg) != 0:
            raise ValueError("Invalid number of arguments: only expect start, end and step.")

        if step is None:
            step = 1

        subset_data = self._data.iloc[start:end:step]
        subset_rois = self._rois.loc[subset_data.index]
        return Dataset(subset_data, self._cfg, subset_rois)

    def subset_by_name(self, names: List[str]) -> Dataset:
        """Get a subset of the dataset by names.

        Parameters
        ----------
        names : List[str]
            The names of the samples to keep.

        Returns
        -------
        Dataset
            The subset of the dataset.
        """
        subset_data = self._data.loc[names]
        subset_rois = self._rois.loc[names]
        return Dataset(subset_data, self._cfg, subset_rois)

    def diagnosis(self, pathology: Optional[Pathology] = None):
        """Get the diagnosis of the dataset.

        Parameters
        ----------
        pathology : Pathology, optional
            The pathology to get the diagnosis for.

            If None, get the diagnosis for both pathologies.

        Returns
        -------
        pd.Series
            The diagnosis of the dataset.
        """
        retinologist = ("A", "B", "C")

        if pathology is None:
            diagnosis_fields = ["dr"] + [f"dr_{r}" for r in retinologist]
            diagnosis_fields += ["me"] + [f"me_{r}" for r in retinologist]
            diagnosis_fields += [f"dr_{r}_comment" for r in retinologist]
            return self._data[diagnosis_fields]

        pathology = Pathology(pathology)
        if pathology is Pathology.DR:
            diagnosis_fields = {"dr": "Consensus"} | {f"dr_{r}": r for r in retinologist}
        else:
            diagnosis_fields = {"me": "Consensus"} | {f"me_{r}": r for r in retinologist}
        diagnosis_fields |= {f"dr_{r}_comment": f"{r}_comment" for r in retinologist}

        return self._data[list(diagnosis_fields.keys())].rename(columns=diagnosis_fields)

    def available_fields(
        self, biomarkers=None, aggregated_biomarkers=None, diagnosis=None, fundus=None, raw_fundus=None
    ) -> List[Field]:
        any_true = biomarkers is True or aggregated_biomarkers is True or diagnosis is True or fundus is True
        if not any_true:
            biomarkers = biomarkers is None
            aggregated_biomarkers = aggregated_biomarkers is None
            diagnosis = diagnosis is None
            fundus = fundus is None

        fields = []
        if biomarkers:
            fields.extend(
                field
                for field in BiomarkerField
                if field not in AGGREGATED_BIOMARKERS and self._data[field.value] is not None
            )
        if aggregated_biomarkers:
            fields.extend(AGGREGATED_BIOMARKERS.keys())
        if diagnosis:
            fields.extend(DiagnosisField)
        if "fundus" in self._data.columns:
            if fundus:
                fields.extend(FundusField)
        return fields

    def get_sample_infos(self, idx: str | int) -> pd.Series:
        """Get the information of a sample.

        Parameters
        ----------
        idx : str | int
            Index of the sample. Can be an integer or the name of the sample (i.e. "20051116_44816_0400_PP").

        Returns
        -------
        pd.Series
            The information of the sample.


        Raises
        ------
        IndexError
            If the index is out of range.

        KeyError
            If the image name is unknown.
        """
        if isinstance(idx, int):
            if idx < -len(self) or idx >= len(self):
                raise IndexError(f"Unknown sample: {idx}.")
            return self._data.iloc[idx]
        else:
            infos = self._data.loc[self._data.index == idx]
            if len(infos) == 0:
                raise KeyError(f"Unknown sample: {idx}.")
            return infos.iloc[0]

    def annotations_infos(self) -> pd.DataFrame:
        """Get the annotations information of the dataset.

        Returns
        -------
        pd.DataFrame
            The annotations information of the dataset.
        """
        tasks = [_.value for _ in BiomarkersAnnotationTasks]
        infos = [_.value for _ in BiomarkersAnnotationInfos]
        columns = pd.MultiIndex.from_product([tasks, infos], names=["Task", "Infos"])

        df = pd.DataFrame(index=self._data.index, columns=columns)
        for task in tasks:
            for info in infos:
                df[(task, info)] = self._data[task + "_" + info]
                if info == "comment":
                    df[(task, info)] = df[(task, info)].astype(str).replace("nan", "")

        return df

    def export(
        self,
        path: str | Path,
        fields: Optional[ImageField | List[ImageField] | Dict[ImageField, str]] = None,
        fundus_as_jpg: bool = True,
        missing_as_blank: bool = False,
        pre_annotation: bool = False,
        n_workers=None,
    ):
        """Export the dataset to a folder.

        Parameters
        ----------
        path :
            Path of the folder where to export the dataset.
        fields :
            The fields to be exported.

            - If None, export the whole dataset.
            - If ``fields`` is a string or list, export only the given fields.
            - If ``fields`` is a dictionary, export the fields given by the keys
            and rename their folder to their corresponding dictionary values.
        fundus_as_jpg :
            If True, save the fundus images (raw and preprocessed) as JPEG images. This format will drastically reduce load times but may introduce compression artifacts.
        pre_annotation :
            If set to ``True``, write the pre-annotation biomarkers instead of the reviewed ones.
        missing_as_blank :
            If set to ``True``, missing biomarkers will be exported as blank images.
        n_workers :
            The number of workers to use for the export.

            If None, use the number of CPUs available.
        """
        # === Parse the fields argument ===
        if fields is None:
            fields = {
                f: field2folder_name(f) for f in self.available_fields(diagnosis=False, aggregated_biomarkers=False)
            }
        elif isinstance(fields, (str, Field)):
            fields = {check_field(fields): field2folder_name(fields)}
        elif isinstance(fields, list):
            fields = {check_field(f): field2folder_name(f) for f in fields}
        if isinstance(fields, dict):
            fields = {check_field(f): field2folder_name(folder) for f, folder in fields.items()}
        else:
            raise ValueError("Invalid fields argument.")

        # === Parse the path argument ===
        if isinstance(path, str):
            path = Path(path)

        export_opts = dict(
            path=path,
            fields=fields,
            fundus_as_jpg=fundus_as_jpg,
            pre_annotation=pre_annotation,
            missing_as_blank=missing_as_blank,
        )

        with RichProgress.iteration("Exporting Maples-DR...", len(self), "Maples-DR exported in {t} seconds.") as p:
            if n_workers == 1:
                for i, sample in enumerate(self):
                    for _ in sample._export_no_check(yield_on_save=True, **export_opts):
                        p.update(1 / len(fields))
                    p.update(completed=i + 1)
            else:
                if n_workers is None:
                    n_workers = multiprocessing.cpu_count()
                with multiprocessing.Manager() as manager:
                    tasks_progress = []
                    futures = []

                    with ProcessPoolExecutor(max_workers=n_workers) as executor:
                        for subset in np.array_split(np.arange(len(self)), n_workers):
                            task_progress = manager.dict({"completed": 0})
                            future = executor.submit(self._export_worker, subset, task_progress, **export_opts)
                            tasks_progress.append(task_progress)
                            futures.append(future)

                        while sum(future.done() for future in futures) < len(futures):
                            n = sum([progress_dict["completed"] for progress_dict in tasks_progress])
                            p.update(completed=n, total=len(self))

                        # raise any errors:
                        for future in futures:
                            future.result()

                p.update(completed=True)

    def _export_worker(self, indices, task_progress, fields=None, **export_opts):
        for i, index in enumerate(indices):
            sample = self[int(index)]
            for _ in sample._export_no_check(fields=fields, **export_opts, yield_on_save=True):
                task_progress["completed"] += 1 / len(fields)
            task_progress["completed"] = i + 1


class DataSample(Mapping):
    """A sample from the MAPLES-DR dataset.

    A sample is a dictionary containing the information of a single sample from the dataset.

    """

    def __init__(
        self,
        data: pd.Series,
        cfg: DatasetConfig,
        roi: pd.Series,
    ):
        self._data: pd.Series = data
        self._cfg: DatasetConfig = cfg
        self._fundus: Optional[np.ndarray] = None
        self._roi: Rect = Rect.from_points(int(roi["y0"]), int(roi["x0"]), int(roi["y1"]), int(roi["x1"]))
        self._messidor_shape: Point = Point(int(roi["H"]), int(roi["W"]))
        self._keys: Optional[List[Field]] = None

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: Field | str) -> Union[Image.Image, np.ndarray, str]:
        return self.read_field(field=key)

    def available_fields(
        self,
        biomarkers=None,
        aggregated_biomarkers=None,
        missing_biomarkers=None,
        diagnosis=None,
        fundus=None,
    ):
        any_true = (
            biomarkers is True
            or aggregated_biomarkers is True
            or missing_biomarkers is True
            or diagnosis is True
            or fundus is True
        )
        if not any_true:
            biomarkers = biomarkers is None
            aggregated_biomarkers = aggregated_biomarkers is None
            missing_biomarkers = missing_biomarkers is None
            diagnosis = diagnosis is None
            fundus = fundus is None

        fields = []
        if biomarkers:
            fields.extend(
                field
                for field in BiomarkerField
                if field not in AGGREGATED_BIOMARKERS and self._data[field.value] is not None
            )
        if aggregated_biomarkers:
            fields.extend(AGGREGATED_BIOMARKERS.keys())
        if missing_biomarkers:
            fields.extend(
                field
                for field in BiomarkerField
                if field not in AGGREGATED_BIOMARKERS and self._data[field.value] is None
            )
        if diagnosis:
            fields.extend(DiagnosisField)
        if "fundus" in self._data:
            if fundus:
                fields.extend(FundusField)
        return fields

    def keys(self) -> List[Field]:
        if self._keys is None:
            self._keys = self.available_fields(aggregated_biomarkers=False, missing_biomarkers=False)
        return self._keys

    def __iter__(self) -> Generator[Field]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def __repr__(self) -> str:
        return f"<DataSample name={self.name}>"

    @property
    def name(self) -> str:
        """The name of the sample."""
        return self._data.name

    def read_field(
        self,
        field: Field | str,
        image_format: Optional[ImageFormat] = None,
        resize: Optional[int | bool] = None,
        pre_annotation: bool = False,
    ):
        """Read one field of the sample.

        This function is similar to __getitem__ but provides more options to format the result (resize, image format...).

        Parameters
        ----------
        field : Field | str
            Any field from:

            - :class:`BiomarkerField <maples_dr.dataset.BiomarkerField>`: a biomarker name, possible values are: ``'opticCup'``, ``'opticDisc'``, ``'macula'``, ``'vessels'``, ``'brightLesions'``, ``'cottonWoolSpots'``, ``'drusens'``, ``'exudates'``, ``'brightUncertains'``, ``'redLesions'``, ``'hemorrhages'``, ``'microaneurysms'``, ``'neovascularization'``, ``'redUncertains'``.
            - :class:`DiagnosisField <maples_dr.dataset.DiagnosisField>`: a diagnosis name, possible values are: ``'dr'``, ``'me'``.
            - :class:`FundusField <maples_dr.dataset.FundusField>`: a fundus field, possible values are: ``'fundus'``, ``'rawFundus'``, ``'mask'``.

        image_format :
            Format of the image to return.

            If ``None`` (by default), use the format defined in the configuration.

        resize :
            Resize the image to the given size.

            - If ``resize`` is an int, crop the image to a square ROI and resize it to the shape ``(resize, resize)``;
            - If ``True``, keep the original MAPLES-DR resolution of 1500x1500 px;
            - If ``False``, use the original MESSIDOR resolution if MESSIDOR path is configured, otherwise fallback to MAPLES-DR original resolution.
            - If ``None`` (by default), use the size defined in the configuration.

        pre_annotation :
            If set to ``True``, read the pre-annotation biomarkers instead of the final ones.

            .. warning::
                Only hemorrhages, microaneurysms, exudates and vessels have pre-annotations.

        Returns
        -------
        Image.Image | np.ndarray | str
            The field under the format specified.
        """
        field = check_field(field)
        if isinstance(field, BiomarkerField):
            return self.read_biomarker(field, image_format=image_format, resize=resize, pre_annotation=pre_annotation)
        elif isinstance(field, DiagnosisField):
            return self._data[field.value]
        elif field is FundusField.FUNDUS:
            return self.read_fundus(image_format=image_format, resize=resize)
        elif field is FundusField.RAW_FUNDUS:
            return self.read_fundus(image_format=image_format, resize=resize, preprocess=False)
        elif field is FundusField.MASK:
            return self.read_roi_mask(image_format=image_format, resize=resize)

    def read_biomarker(
        self,
        biomarkers: BiomarkerField | str | Iterable[BiomarkerField | str],
        image_format: Optional[ImageFormat] = None,
        resize: Optional[int | bool] = None,
        pre_annotation: bool = False,
        no_cache: bool = False,
    ) -> any:
        """Read a biomarker from the sample.

        Parameters
        ----------
        biomarkers :
            Name of the biomarker(s) to read. Possible values are: ``'opticCup'``, ``'opticDisc'``, ``'macula'``,
            ``'vessels'``, ``'brightLesions'``, ``'cottonWoolSpots'``, ``'drusens'``, ``'exudates'``,
            ``'brightUncertains'``, ``'redLesions'``, ``'hemorrhages'``, ``'microaneurysms'``,
            ``'neovascularization'``, ``'redUncertains'`` (see :class:`BiomarkerField <maples_dr.dataset.BiomarkerField>` for more details).

            If multiple biomarkers are given, the corresponding masks will be merged.

        image_format :
            Format of the image to return. Possible values are: ``'PIL'``, ``'BGR'`` or ``'RGB'``
            (see :class:`ImageFormat <maples_dr.config.ImageFormat>` for more details.).

            If ``None`` (by default), use the format defined in the configuration.

        resize :
            Resize the image to the given size.

            - If ``resize`` is an int, crop the image to a square ROI and resize it to the shape ``(resize, resize)``;
            - If ``True``, keep the original MAPLES-DR resolution of 1500x1500 px;
            - If ``False``, use the original MESSIDOR resolution if MESSIDOR path is configured, otherwise fallback to MAPLES-DR original resolution.
            - If ``None`` (by default), use the size defined in the configuration.

        pre_annotation :
            If set to ``True``, read the pre-annotation biomarkers instead of the final ones.

            .. warning::
                Only hemorrhages, microaneurysms, exudates and vessels have pre-annotations.

        no_cache :
            If set to ``True``, the cache will not be used to read the biomarker, regardless of the configuration.

        Returns
        -------
            The biomarker mask under the format specified.
        """
        # Check arguments validity
        image_format = self._check_image_format(image_format)
        target_size, crop = self._target_size(resize)

        if isinstance(biomarkers, (str, BiomarkerField)):
            biomarkers = [biomarkers]
        biomarkers = [BiomarkerField.parse(b) for b in biomarkers]
        assert len(biomarkers), "No biomarker to read."

        # === Try to read the biomarker from the cache ===
        if not no_cache and self._cfg.cache_path is not None:
            if (
                (crop and target_size == (1500, 1500))
                and len(biomarkers) == 1
                and biomarkers[0] not in AGGREGATED_BIOMARKERS
            ):
                # If the biomarker is not aggregated and not resized, use the direct path
                key = biomarkers[0].value + ("_pre" if pre_annotation else "")
                path = self._data.get(key, None)
                if path is None or not Path(path).exists():
                    # For missing biomarkers, return an empty image
                    return Image.new("1", target_size) if image_format is ImageFormat.PIL else np.zeros(target_size)

                img = Image.open(path)
                return img if image_format is ImageFormat.PIL else np.array(img)

            # If the cache exists
            cache_path = self._cfg.biomarkers_cache_path(biomarkers, pre_annotation, resize) / f"{self.name}.png"
            if cache_path.exists():
                # Read the cached image
                img = Image.open(cache_path)
            else:
                # Otherwise read the image with the normal method and save it to the cache
                img = self.read_biomarker(
                    biomarkers=biomarkers,
                    resize=resize,
                    image_format=ImageFormat.PIL,
                    pre_annotation=pre_annotation,
                    no_cache=True,
                )
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(cache_path, bits=1)

            return img if image_format is ImageFormat.PIL else np.array(img)

        # === Read the biomarker from the original files ===
        # Expand aggregated biomarkers
        for i in range(len(biomarkers) - 1, -1, -1):
            bio = biomarkers[i]
            if bio in AGGREGATED_BIOMARKERS:
                del biomarkers[i]
                biomarkers.extend(AGGREGATED_BIOMARKERS[bio])

        # Read the paths of the biomarkers
        paths = []
        for bio in biomarkers:
            key = bio.value + ("_pre" if pre_annotation else "")
            if key in self._data:
                bio_path = self._data[key]
                if bio_path is not None:
                    paths.append(bio_path)
                else:
                    if bio is BiomarkerField.MACULA:
                        logging.warning(
                            f"The macula is not segmented on image {self.name}!\n"
                            "(The corresponding fundus image is centered on the optic disc, the macula is not visible)."
                            "\n Use maples_dr.configure(exclude_missing_macula=True) to exclude this image."
                        )
                    if bio is BiomarkerField.OPTIC_CUP:
                        logging.warning(
                            f"The optic cup is not segmented on image {self.name}!\n"
                            "(The cup boundaries are too fuzzy to be annotated.).\n"
                            "Use maples_dr.configure(exclude_missing_optic_cup=True) to exclude this image."
                        )

        # Read the biomarkers
        biomarkers_map = np.zeros(target_size, dtype=bool)
        window = biomarkers_map

        if not crop:
            # Align the image to the original MESSIDOR resolution
            window = biomarkers_map[self._roi.slice()]
            target_size = self._roi.shape

        for path in paths:
            if Path(path).exists():
                img = Image.open(path).resize(target_size.xy(), Resampling.NEAREST)
                window[np.array(img) > 0] = 1

        # Apply format conversion
        if image_format is ImageFormat.PIL:
            return Image.fromarray(biomarkers_map)
        else:
            return biomarkers_map

    def read_multiple_biomarkers(
        self,
        biomarkers: Mapping[int, BiomarkerField | str | List[BiomarkerField | str]],
        image_format: Optional[ImageFormat] = None,
        pre_annotation: bool = False,
        resize: Optional[int | bool] = None,
    ):
        """
        Read multiple biomarkers at once, assigning a class to a biomarker or a group of them.

        Parameters
        ----------
        biomarkers :
            Name of the biomarker(s) to read. Possible values are: ``'opticCup'``, ``'opticDisc'``, ``'macula'``,
            ``'vessels'``, ``'brightLesions'``, ``'cottonWoolSpots'``, ``'drusens'``, ``'exudates'``,
            ``'brightUncertains'``, ``'redLesions'``, ``'hemorrhages'``, ``'microaneurysms'``,
            ``'neovascularization'``, ``'redUncertains'`` (see :class:`BiomarkerField <maples_dr.dataset.BiomarkerField>` for more details).

        image_format :
            Format of the image to return. Possible values are: ``'PIL'``, ``'BGR'`` or ``'RGB'``
            (see :class:`ImageFormat <maples_dr.config.ImageFormat>` for more details.).

            If None (by default), use the format defined in the configuration.

        pre_annotation :
            If set to ``True``, read the pre-annotation biomarkers instead of the final ones.

            .. warning::
                Only hemorrhages, microaneurysms, exudates and vessels have pre-annotations.

        """
        image_format = self._check_image_format(image_format)
        target_size, _ = self._target_size(resize)

        biomarkers_map = np.zeros(target_size, dtype=np.uint8)
        for i, bio in biomarkers.items():
            biomarkers_map[self.read_biomarker(bio, image_format="bgr", pre_annotation=pre_annotation)] = i

        # Apply format conversion
        if image_format is ImageFormat.PIL:
            return Image.fromarray(biomarkers_map)
        else:
            return biomarkers_map

    def read_fundus(
        self,
        preprocess: Optional[Preprocessing | str | bool] = None,
        image_format: Optional[ImageFormat] = None,
        resize: Optional[int | bool] = None,
        no_cache: bool = False,
    ) -> Union[Image.Image, np.ndarray]:
        """Read the fundus image of the sample.

        Parameters
        ----------
        preprocess :
            Preprocessing to apply to the image.

            - If a :class:`Preprocessing <maples_dr.config.Preprocessing>` (or an equivalent string), the image is preprocessed with the given preprocessing;
            - if ``False``, the image is not preprocessed.
            - if ``None`` (by default) or ``True``, use the preprocessing defined in the configuration.

        image_format :
            Format of the image to return. Possible values are: ``'PIL'``, ``'BGR'`` or ``'RGB'``
            (see :class:`ImageFormat <maples_dr.config.ImageFormat>` for more details.).

            If ``None`` (by default), use the format defined in the configuration.

        resize :
            Resize the image to the given size.

            - If ``resize`` is an int, crop the image to a square ROI and resize it to the shape ``(resize, resize)``;
            - If ``True``, use the original MAPLES-DR resolution of 1500x1500 px;
            - If ``False``, keep the original MESSIDOR resolution.
            - If ``None`` (by default), use the size defined in the configuration.

        no_cache :
            If set to ``True``, the cache will not be used to read the fundus image, regardless of the configuration.

        Returns
        -------
            The fundus image under the format specified.
        """

        assert "fundus" in self._data, "Impossible to read fundus images, path to MESSIDOR dataset is unknown."

        # Check arguments
        preprocess = self._check_preprocessing(preprocess)
        image_format = self._check_image_format(image_format)
        target_size, crop = self._target_size(resize)

        # Check if the result is cached
        cache_path = self._cfg.cache_path
        if not no_cache and cache_path is not None:
            if not crop and preprocess is Preprocessing.NONE:
                img = Image.open(self._data[FundusField.FUNDUS.value])
            else:
                cache_path = self._cfg.fundus_cache_path(resize, preprocess) / f"{self.name}.jpg"

                if not cache_path.exists():
                    img = self.read_fundus(
                        preprocess=preprocess, resize=resize, image_format=ImageFormat.PIL, no_cache=True
                    )
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    img.save(cache_path, quality=98, subsampling=0)

                # Even if we just created the cache, we reload it to ensure consistency despite compression losses
                img = Image.open(cache_path)

            return self._apply_image_format(img, ImageFormat.PIL, image_format)

        # Read the image
        if self._fundus is None:
            path = self._data[FundusField.FUNDUS.value]
            fundus = Image.open(path)
            fundus_format = ImageFormat.PIL

            # Crop the image to the proper region of interest
            if crop and fundus.height != fundus.width:
                fundus = fundus.crop(self._roi.box())
        else:
            fundus = self._fundus
            fundus_format = ImageFormat.BGR

        # Resize the image
        fundus_shape = fundus.size if fundus_format is ImageFormat.PIL else fundus.shape[:2]
        if fundus_shape != target_size:
            resampling_method = Resampling.LANCZOS if target_size.y < 1000 else Resampling.BILINEAR
            if fundus_format is not ImageFormat.PIL:
                fundus = self._apply_image_format(fundus, fundus_format, ImageFormat.PIL)
                fundus_format = ImageFormat.PIL
            fundus = fundus.resize(target_size.xy(), resampling_method)

        # Preprocess the image
        if preprocess is not Preprocessing.NONE:
            # If required, cast the image to a numpy array
            fundus = self._apply_image_format(fundus, fundus_format, ImageFormat.BGR)
            fundus_format = ImageFormat.BGR
            self._fundus = fundus

            fundus = preprocess_fundus(fundus, preprocess)

        return self._apply_image_format(fundus, fundus_format, image_format)

    def read_roi_mask(
        self, image_format: Optional[ImageFormat] = None, resize: Optional[int | bool] = None
    ) -> np.ndarray | Image.Image:
        """Read the region of interest of the fundus image.

        Parameters
        ----------

        image_format :
            Format of the image to return. Possible values are: ``'PIL'``, ``'BGR'`` or ``'RGB'``
            (see :class:`ImageFormat <maples_dr.config.ImageFormat>` for more details.).

            If ``None`` (by default), use the format defined in the configuration.

        resize :
            Resize the image to the given size.

            - If ``resize`` is an int, crop the image to a square ROI and resize it to the shape ``(resize, resize)``;
            - If ``True``, use the original MAPLES-DR resolution of 1500x1500 px;
            - If ``False``, keep the original MESSIDOR resolution.
            - If ``None`` (by default), use the size defined in the configuration.

        Returns
        -------
        np.ndarray
            The region of interest of the fundus image.
        """
        fundus = self.read_fundus(preprocess=False, image_format=ImageFormat.BGR, resize=resize)
        mask = fundus_roi(fundus)

        return self._apply_image_format(mask, ImageFormat.BGR, image_format)

    def export(
        self,
        path: str | Path,
        fields: Optional[ImageField | List[ImageField] | Dict[ImageField, str]] = None,
        fundus_as_jpg: bool = True,
        *,
        pre_annotation: bool = False,
        missing_as_blank=False,
    ) -> Mapping[Field, Optional[str]]:
        """Export the sample to a folder.

        Parameters
        ----------
        path :
            Path of the folder where to export the sample.
        fields :
            The fields to be exported.

            - If None, export all available fields.
            - If ``fields`` is a string or list, export only the given fields.
            - If ``fields`` is a dictionary, export the fields given by the keys
            and rename their folder to their corresponding dictionary values.
        fundus_as_jpg :
            If True, save the fundus images (raw and preprocessed) as JPEG images. This format will drastically reduce load times but may introduce compression artifacts.
        pre_annotation :
            If set to ``True``, write the pre-annotation biomarkers instead of the reviewed ones.
        missing_as_blank :
            If set to ``True``, missing biomarkers will be exported as blank images.

        Returns
        -------
        Mapping[Field, Optional[str]]
            A mapping of the exported fields to their paths (or None if the field was not exported).
        """

        # === Parse the fields argument ===
        if fields is None:
            fields = {
                f: field2folder_name(f)
                for f in self.available_fields(
                    aggregated_biomarkers=False, diagnosis=False, missing_biomarkers=not missing_as_blank
                )
            }
        elif isinstance(fields, (str, Field)):
            fields = {check_field(fields): field2folder_name(fields)}
        elif isinstance(fields, list):
            fields = {check_field(f): field2folder_name(f) for f in fields}
        if isinstance(fields, dict):
            fields = {check_field(f): field2folder_name(folder) for f, folder in fields.items()}
        else:
            raise ValueError("Invalid fields argument.")

        # === Parse the path argument ===
        if isinstance(path, str):
            path = Path(path)

        # === Export the fields ===
        return {
            k: v
            for k, v in self._export_no_check(
                path=path,
                fields=fields,
                fundus_as_jpg=fundus_as_jpg,
                pre_annotation=pre_annotation,
                missing_as_blank=missing_as_blank,
                yield_on_save=True,
            )
        }

    def _export_no_check(
        self, path, fields, fundus_as_jpg, pre_annotation, missing_as_blank, yield_on_save=False
    ) -> Generator[Tuple[Field, Optional[str]]]:
        available_fields = self.available_fields(missing_biomarkers=False, diagnosis=False)

        for field, folder in fields.items():
            field_folder = path / folder
            field_folder.mkdir(parents=True, exist_ok=True)
            filename = str((field_folder / f"{self.name}.png").resolve().absolute())

            if field in available_fields:
                img = self.read_field(field, image_format=ImageFormat.PIL, pre_annotation=pre_annotation)

                if field in (FundusField.FUNDUS, FundusField.RAW_FUNDUS):
                    if fundus_as_jpg:
                        img.save(filename[:-3] + "jpg", quality=98, subsampling=0)
                    else:
                        img.save(filename)
                else:
                    img.save(filename, bits=1, optimize=True)

            else:
                if missing_as_blank:
                    img = Image.new("1", self._target_size()[0])
                    img.save(filename, bits=1)
                else:
                    yield field, None
                    continue

            if yield_on_save:
                yield field, filename

    def _apply_image_format(
        self, image: Image.Image | np.ndarray, input_format: ImageFormat, target_format: str | ImageFormat | None
    ) -> Image.Image | np.ndarray:
        target_format = self._check_image_format(target_format)
        input_format = ImageFormat(input_format)

        if input_format is ImageFormat.PIL:
            if target_format is ImageFormat.PIL:
                return image
            else:  # Target format is numpy array
                image = np.array(image)
                if len(image.shape) == 2 or target_format is ImageFormat.RGB:
                    return image
                else:
                    return image[..., ::-1]
        else:  # Input format is numpy array
            if target_format is ImageFormat.PIL:
                return Image.fromarray(image)
            else:  # Target format is numpy array
                same_channel_order = len(image.shape) == 2 or target_format is input_format
                return image if same_channel_order else image[..., ::-1]

    def _check_image_format(self, image_format: Optional[ImageFormat | str] = None) -> ImageFormat:
        if image_format is None:
            return ImageFormat(self._cfg.image_format)
        return ImageFormat(image_format)

    def _check_preprocessing(self, preprocess: str | Preprocessing | bool | None) -> Preprocessing:
        if preprocess is None or preprocess is True:
            return Preprocessing(self._cfg.preprocessing)
        if preprocess is False:
            return Preprocessing.NONE
        return Preprocessing(preprocess)

    def _target_size(self, size: Optional[int | bool] = None) -> Tuple[Point, bool]:
        if size is None:
            size = self._cfg.resize

        if size is False:
            return self._messidor_shape, False
        elif size is True:
            return Point(1500, 1500), True
        else:
            return Point(size, size), True
