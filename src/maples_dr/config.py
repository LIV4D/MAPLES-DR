from __future__ import annotations

__all__ = ["DatasetConfig", "InvalidConfigError", "ImageFormat", "Preprocessing"]
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional


class ImageFormat(str, Enum):
    """String Enum of possible image formats for fundus images and biomarker masks."""

    #: Images are formatted as :class:`PIL.Image.Image`.
    PIL = "PIL"

    #: Images are formatted as numpy array.
    #: Biomarkers masks have a shape of ``(height, width)`` and
    #: fundus image have a shape of ``(height, width, 3)`` with a ``RGB`` channel order.
    RGB = "rgb"

    #: Images are formatted as numpy array.
    #: Biomarkers masks have a shape of ``(height, width)`` and
    #: fundus image have a shape of ``(height, width, 3)`` with a ``BGR`` channel order.
    BGR = "bgr"


class Preprocessing(str, Enum):
    """String Enum of possible preprocessing algorithms applied on the fundus images."""

    #: No preprocessing.
    NONE = "none"

    #: Preprocessing based on Contrast Limited Adaptive Histogram Equalization (CLAHE).
    #:
    #: See :func:`maples_dr.preprocessing.clahe_preprocessing`.
    CLAHE = "clahe"

    #: Preprocessing based on median filtering.
    #:
    #: See :func:`maples_dr.preprocessing.median_preprocessing`.
    MEDIAN = "median"


@dataclass
class DatasetConfig:
    """
    Dataclass storing the configuration of the dataset.
    """

    #: Size of the generated images.
    #:
    #: - If ``int``: the images are resized to a square of the given size;
    #: - if ``True``: the original MAPLES-DR resolution of 1500x1500 px is kept;
    #: - if ``False``: keep the original MESSIDOR resolution if MESSIDOR path is configured,
    #:   otherwise fallback to MAPLES-DR original resolution.
    #:
    #: By default: ``True`` (use MAPLES-DR 1500x1500 px resolution).
    resize: int | bool = True

    #: Python format of the generated images. See :class:`ImageFormat` for the available options.
    #:
    #: By default: :attr:`ImageFormat.PIL` is used.
    image_format: ImageFormat = ImageFormat.PIL

    #: Preprocessing algorithm applied on the fundus images.
    #: See :class:`Preprocessing` for the available options.
    #:
    #: By default: :attr:`Preprocessing.NONE` (no preprocessing) is applied.
    preprocessing: Preprocessing = Preprocessing.NONE

    _cache: str | bool = False

    def update(self, cfg: Mapping[str, Any]):
        """
        Update the configuration with the given values.

        Parameters
        ----------
        cfg:
            Configuration to update with.

        :meta private:
        """
        for k, v in cfg.items():
            if v is not None:
                setattr(self, k, v)

    @property
    def cache_path(self) -> Optional[Path]:
        """
        Return the path to the cache directory.

        Returns
        -------
            The path to the cache directory or None if the cache is disabled.
        """
        path = self._cache
        if path is False:
            return None
        return Path(path)

    def biomarkers_cache_path(
        self, biomarkers, pre_annotation: bool = False, resize: Optional[int | bool] = None
    ) -> Path:
        """Return the path to the cache directory for biomarkers masks.

        Parameters
        ----------
        biomarker: BiomarkerField | list(BiomarkerField)
            Name of the biomarker.
        pre_annotation:
            If True, the path to the cache directory for pre-annotated biomarkers masks is returned.
        resize:
            Size of the generated images. If not provided, the current size is used.

        Returns
        -------
            The path to the cache directory for biomarkers masks.

        :meta private:
        """
        from .dataset import BiomarkerField

        if isinstance(biomarkers, BiomarkerField):
            biomarkers = [biomarkers]
        folder = "+".join(b.value for b in sorted(biomarkers))
        if pre_annotation:
            folder += "_pre"
        return self.cache_path / self._resize_cache_folder_name(resize) / folder

    def fundus_cache_path(
        self, resize: Optional[int | bool] = None, preprocess: Optional[Preprocessing] = None
    ) -> Path:
        """
        Return the path to the cache directory for fundus images.

        Parameters
        ----------
        resize:
            Size of the generated images. If not provided, the current size is used.
        preprocess:
            Preprocessing algorithm applied on the fundus images. If not provided, the current preprocessing is used.

        Returns
        -------
            The path to the cache directory for fundus images.

        :meta private:
        """
        preprocess = Preprocessing(preprocess) or self.preprocessing
        preprocess_folder = "fundus" if preprocess is Preprocessing.NONE else "fundus_" + preprocess.value
        return self.cache_path / self._resize_cache_folder_name(resize) / preprocess_folder

    def _resize_cache_folder_name(self, resize: Optional[int | bool] = None) -> str:
        """
        Return the name of the cache folder for the given resize value.

        Parameters
        ----------
        resize:
            Size of the generated images. If not provided, the current size is used.

        Returns
        -------
            The name of the cache folder for the given resize value.

        :meta private:
        """
        resize = resize if resize is not None else self.resize
        if resize is False:
            return "mes"
        elif resize is True:
            return "1500"
        return str(resize)


class InvalidConfigError(Exception):
    """
    Exception raised when the configuration is invalid.
    """

    pass
