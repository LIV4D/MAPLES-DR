from typing import Optional

import numpy as np

from .config import Preprocessing


def preprocess_fundus(fundus: np.ndarray, preprocessing: Preprocessing | str) -> np.ndarray:
    """Preprocess a fundus image.

    Args:
        fundus: The fundus image to preprocess.
        preprocessing: The preprocessing algorithm to apply.

            See :class:`maples_dr.config.Preprocessing` for the available options.

    Returns:
        The preprocessed fundus image.

    """
    preprocessing = Preprocessing(preprocessing)

    if preprocessing is Preprocessing.NONE:
        return fundus
    elif preprocessing is Preprocessing.CLAHE:
        return clahe_preprocessing(fundus)
    elif preprocessing is Preprocessing.MEDIAN:
        return median_preprocessing(fundus)


def clahe_preprocessing(fundus: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Preprocessing based on Contrast Limited Adaptive Histogram Equalization (CLAHE).

    This algorithm was used to annotate MAPLES-DR anatomical and pathological structures.

    Args:
        fundus: The fundus image as a BGR numpy array (height, width, 3).

    Returns:
        The preprocessed fundus image.

    """
    # Initial checks
    assert len(fundus.shape) == 3 and fundus.shape[2] == 3, "Fundus image must be a 3-channel RGB image."
    mask = fundus_roi(fundus) if mask is None else mask
    assert mask.shape == fundus.shape[:2], "Mask must have the same shape as the fundus image."

    # CV2 is required for this preprocessing
    import cv2

    # Preprocessing
    mean_b = np.median(fundus[..., 0][mask])
    mean_g = np.median(fundus[..., 1][mask])
    mean_r = np.median(fundus[..., 2][mask])
    mean_channels = np.asarray([mean_b, mean_g, mean_r])

    mask = np.expand_dims(mask, 2)

    bgr = fundus.astype(np.float32)
    bgr += (mean_channels - cv2.medianBlur(fundus, 151)) * mask
    bgr = bgr.clip(0, 255).astype(np.uint8)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_layers = list(cv2.split(lab))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_layers[0] = clahe.apply(lab_layers[0])
    lab = cv2.merge(lab_layers)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return bgr * mask


def median_preprocessing(fundus: np.ndarray) -> np.ndarray:
    """Preprocessing based on median filtering.

    This algorithm is often used as a preprocessing step for automatic vessel segmentation.

    Args:
        fundus: The fundus image as a BGR numpy array (height, width, 3).

    Returns:
        The preprocessed fundus image.

    """
    import cv2

    k = np.max(fundus.shape) // 20 * 2 + 1
    bg = cv2.medianBlur(fundus, k)
    return cv2.addWeighted(fundus, 4, bg, -4, 128)


def fundus_roi(fundus: np.ndarray) -> np.ndarray:
    """Compute the region of interest (ROI) of a fundus image.

    Args:
        fundus: The fundus image.

    Returns:
        The ROI mask.

    """
    import cv2

    g = fundus[..., 1]
    return cv2.medianBlur(g, 5) > 10
