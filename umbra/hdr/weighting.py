"""How much each stack contributes to each pixel of the composite."""
import numpy as np


def in_range(img: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    Mask of the pixels of a colour image whose every channel lies strictly within ``[low, high]``.

    Each bound is tested against the channel most likely to violate it: the dimmest one for
    the noise floor, the brightest one for saturation.
    """
    return (img.min(axis=2) > low) & (img.max(axis=2) < high)


def saturation_weighting(img: np.ndarray, low: float, high: float, low_smoothness: float, high_smoothness: float) -> np.ndarray:
    """
    Smooth counterpart of :func:`in_range`, feathering both bounds instead of cutting them.

    The weight reaches 1 over ``[low, high]`` and falls linearly to 0 over the
    ``low_smoothness`` below ``low`` and the ``high_smoothness`` above ``high``. The two
    bounds are combined multiplicatively, so a pixel violating either one weighs nothing.
    """
    low_weights = np.clip((img.min(axis=2) + low_smoothness - low) / low_smoothness, 0, 1)
    high_weights = np.clip((- img.max(axis=2) + high_smoothness + high) / high_smoothness, 0, 1)
    return low_weights * high_weights


def running_composite(hdr_img: np.ndarray, sum_weights: np.ndarray) -> np.ndarray:
    """Normalize the partial composite for display, leaving the pixels no stack has covered yet black."""
    weights = sum_weights[:, :, None]
    return np.divide(hdr_img, weights, out=np.zeros_like(hdr_img), where=weights > 0)
