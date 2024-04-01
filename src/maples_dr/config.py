from __future__ import annotations

__all__ = ["DatasetConfig", "InvalidConfigError", "ImageFormat", "Preprocessing"]
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


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


DOWNLOAD = "DOWNLOAD"


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

    #: Path to permanently cache the formatted dataset. If False (by default), then the cache is disabled.
    cache: str | bool = False

    def update(self, cfg: Mapping[str, Any]):
        """
        Update the configuration with the given values.

        :param kwargs: Configuration to update with.

        :meta private:
        """
        for k, v in cfg.items():
            if v is not None:
                setattr(self, k, v)


class InvalidConfigError(Exception):
    """
    Exception raised when the configuration is invalid.
    """

    pass
