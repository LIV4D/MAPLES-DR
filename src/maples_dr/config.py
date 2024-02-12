from __future__ import annotations

__all__ = ["DatasetConfig", "InvalidConfigError", "ImageFormat", "Preprocessing"]
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ImageFormat(str, Enum):
    """Image formats for fundus images.

    - ``PIL`` : Images are formatted as :class:`PIL.Image.Image`.
    - ``rgb`` : Images are formatted as numpy array of shape: (height, width, 3). The channel order is RGB.
    - ``bgr`` : Images are formatted as numpy array of shape: (height, width, 3). The channel order is BGR.

    """

    PIL = "PIL"
    RGB = "rgb"
    BGR = "bgr"


class Preprocessing(str, Enum):
    """Preprocessing algorithms for fundus images.

    - ``none``: No preprocessing is applied.
    - ``clahe``: Contrast Limited Adaptive Histogram Equalization.
    - ``median``: Median filter.

    """

    NONE = "none"
    CLAHE = "clahe"
    MEDIAN = "median"


DOWNLOAD = "DOWNLOAD"


@dataclass
class DatasetConfig:
    """
    Configuration of the MAPLES-DR dataset.
    """

    #: Size of the generated images.
    #: By default, keep the original image size of 1500x1500.
    resize: Optional[int] = None

    #: Python format of the generated images. Must be either "PIL", "rgb" or "bgr".
    #: If "rgb" or "bgr" is selected, images will be formatted as numpy array of shape: (height, width, channel).
    #: By default, "PIL" is used.
    image_format: Optional[ImageFormat] = None

    #: Preprocessing aglorithm applied on the fundus images.
    #: Must be either "clahe", "median" or None (no preprocessing).
    #: By default, no preprocessing is applied.
    preprocessing: Optional[Preprocessing] = None

    #: Path to permanently cache the formatted dataset. If None (by default), then the cache is disabled.
    cache: Optional[str] = None

    def update(self, cfg: Optional[DatasetConfig] = None, **kwargs):
        """
        Update the configuration with the given values.

        :param cfg: Configuration to update with.
        :param kwargs: Configuration to update with.

        :meta private:
        """
        if cfg is not None:
            for k, v in cfg.__dict__.items():
                if v is not None:
                    setattr(self, k, v)
        for k, v in kwargs.items():
            if v is not None:
                setattr(self, k, v)


class InvalidConfigError(Exception):
    """
    Exception raised when the configuration is invalid.
    """

    pass
