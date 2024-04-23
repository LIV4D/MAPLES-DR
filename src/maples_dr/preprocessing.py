from typing import Optional

import numpy as np

from .config import Preprocessing

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from skimage import measure as skmeasure
    from skimage import morphology as skmorph
except ImportError:
    skmeasure = None
    skmorph = None


__all__ = ["preprocess_fundus", "clahe_preprocessing", "median_preprocessing", "fundus_roi"]


def ensure_imports(cv2_needed=False, skimage_needed=False):
    if cv2_needed and skimage_needed and (cv2 is None and skmeasure is None):
        raise ImportError(
            "OpenCV and Scikit-Image are required for this function.\n"
            "Please install them using 'pip install opencv-python-headless scikit-image'."
        )
    if cv2_needed and not cv2:
        raise ImportError(
            "OpenCV is required for this function." "Please install it using 'pip install opencv-python-headless'."
        )
    if skimage_needed and not skmeasure:
        raise ImportError(
            "Scikit-Image is required for this function." "Please install it using 'pip install scikit-image'."
        )


def preprocess_fundus(fundus: np.ndarray, preprocessing: Preprocessing | str) -> np.ndarray:
    """Preprocess a fundus image.

    Parameters
    ----------
    fundus:
        The fundus image to preprocess.
    preprocessing:
        The preprocessing algorithm to apply.
        See :class:`maples_dr.config.Preprocessing` for the available options.

    Returns
    -------
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

    Parameters
    ----------
    fundus:
        The fundus image as a BGR numpy array (height, width, 3).

    Returns
    -------
        The preprocessed fundus image.

    """
    # Initial checks
    assert isinstance(fundus, np.ndarray), f"Fundus image must be a numpy array instead of {type(fundus)}."
    assert fundus.dtype == np.uint8, f"Fundus image must be a 8-bit unsigned integer array instead of {fundus.dtype}."
    assert (
        len(fundus.shape) == 3 and fundus.shape[2] == 3
    ), f"Fundus image must be a 3-channel RGB image with shape (height, width, 3) instead of {fundus.shape}."
    mask = fundus_roi(fundus) if mask is None else mask
    assert mask.shape == fundus.shape[:2], "Mask must have the same shape as the fundus image."

    # CV2 is required for this preprocessing
    ensure_imports(cv2_needed=True)

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

    Parameters
    ----------
    fundus:
        The fundus image as a BGR numpy array (height, width, 3).

    Returns
    -------
        The preprocessed fundus image.

    """
    assert isinstance(fundus, np.ndarray), f"Fundus image must be a numpy array instead of {type(fundus)}."
    assert fundus.dtype == np.uint8, f"Fundus image must be a 8-bit unsigned integer array instead of {fundus.dtype}."
    assert (
        len(fundus.shape) == 3 and fundus.shape[2] == 3
    ), "Fundus image must be a 3-channel RGB image with shape (height, width, 3) instead of {fundus.shape}."

    # CV2 is required for this preprocessing
    ensure_imports(cv2_needed=True)

    k = np.max(fundus.shape) // 20 * 2 + 1
    bg = cv2.medianBlur(fundus, k)
    return cv2.addWeighted(fundus, 4, bg, -4, 128)


def fundus_roi(
    fundus: np.ndarray, blur_radius=5, morphological_clean=False, smoothing_radius=0, final_erosion=4
) -> np.ndarray:
    """Compute the region of interest (ROI) of a fundus image.

    Parameters:
    -----------
    fundus:
        The fundus image.

    blur_radius:
        The radius of the median blur filter.

        By default: 5.

    morphological_clean:
        Whether to perform morphological cleaning. (small objects removal and filling of the holes not on the border)

        By default: False.

    smoothing_radius:
        The radius of the Gaussian blur filter.

        By default: 0.

    final_erosion:
        The radius of the disk used for the final erosion.

        By default: 4.

    Returns:
        The ROI mask.

    """
    ensure_imports(cv2_needed=True, skimage_needed=morphological_clean or final_erosion > 0)

    fundus = cv2.medianBlur(fundus[..., 1], blur_radius * 2 - 1)
    mask = fundus > 10

    if morphological_clean:
        # Remove small objects
        mask = skmorph.remove_small_objects(mask, 5000)

        # Remove holes that are not on the border
        MASK_BORDER = np.zeros_like(mask)
        MASK_BORDER[0, :] = 1
        MASK_BORDER[-1, :] = 1
        MASK_BORDER[:, 0] = 1
        MASK_BORDER[:, -1] = 1
        labelled_holes = skmeasure.label(mask == 0)
        for i in range(1, labelled_holes.max() + 1):
            hole_mask = labelled_holes == i
            if not np.any(MASK_BORDER & hole_mask):
                mask[hole_mask] = 1

    if smoothing_radius > 0:
        mask = (
            cv2.GaussianBlur(
                mask.astype(np.uint8) * 255,
                (smoothing_radius * 6 + 1, smoothing_radius * 6 + 1),
                smoothing_radius,
                borderType=cv2.BORDER_CONSTANT,
            )
            > 125
        )

    if final_erosion > 0:
        skmorph.binary_erosion(mask, skmorph.disk(final_erosion), out=mask)

    return mask
