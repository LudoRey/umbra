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
    Smooth counterpart of :func:`in_range`, feathering both bounds inwards instead of cutting them.

    Nothing outside ``[low, high]`` weighs anything: the bounds are hard limits, and the two
    smoothnesses only say how far inside them the weight takes to reach 1. A smoothness of 0
    turns its bound into a step, which is how a bound that should not bite is expressed. The
    two bounds are combined multiplicatively, so a pixel violating either one weighs nothing.
    """
    low_weights = _ramp(img.min(axis=2) - low, low_smoothness)
    high_weights = _ramp(high - img.max(axis=2), high_smoothness)
    return low_weights * high_weights


def _ramp(depth: np.ndarray, smoothness: float) -> np.ndarray:
    """Weight of a pixel lying ``depth`` inside a bound: 0 outside it, 1 once ``smoothness`` past it."""
    if smoothness == 0:
        return (depth >= 0).astype(depth.dtype)
    return np.clip(depth / smoothness, 0, 1)


def running_composite(hdr_img: np.ndarray, sum_weights: np.ndarray) -> np.ndarray:
    """Normalize the partial composite for display, leaving the pixels no stack has covered yet black."""
    weights = sum_weights[:, :, None]
    return np.divide(hdr_img, weights, out=np.zeros_like(hdr_img), where=weights > 0)
