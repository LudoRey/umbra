"""How much each stack contributes to each pixel of the composite."""
import numpy as np

from umbra.common import channels


def in_range(img: np.ndarray, low: float, high: float) -> np.ndarray:
    """Mask of the pixels whose every channel lies strictly within ``[low, high]``."""
    lowest, highest = _channel_bounds(img)
    return (lowest > low) & (highest < high)


def saturation_weighting(img: np.ndarray, low: float, low_smoothness: float, high: float, high_smoothness: float) -> np.ndarray:
    """
    Smooth counterpart of :func:`in_range`, feathering both bounds inwards instead of cutting them.

    Nothing outside ``[low, high]`` weighs anything: the bounds are hard limits, and the two
    smoothnesses only say how far inside them the weight takes to reach 1. A smoothness of 0
    turns its bound into a step, which is how a bound that should not bite is expressed.
    """
    lowest, highest = _channel_bounds(img)
    low_weights = _ramp(lowest - low, low_smoothness)
    high_weights = _ramp(high - highest, high_smoothness)
    return low_weights * high_weights


def _channel_bounds(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The per-pixel minimum and maximum over the channels, which a mono image does not have."""
    if img.ndim == 2:
        return img, img
    return img.min(axis=2), img.max(axis=2)


def _ramp(depth: np.ndarray, smoothness: float) -> np.ndarray:
    """Weight of a pixel lying ``depth`` inside a bound: 0 outside it, 1 once ``smoothness`` past it."""
    if smoothness == 0:
        return (depth >= 0).astype(depth.dtype)
    return np.clip(depth / smoothness, 0, 1)


def running_composite(hdr_img: np.ndarray, sum_weights: np.ndarray) -> np.ndarray:
    """Normalize the partial composite for display, leaving the pixels no stack has covered yet black."""
    weights = channels.broadcast_over_channels(sum_weights, hdr_img)
    return np.divide(hdr_img, weights, out=np.zeros_like(hdr_img), where=weights > 0)
