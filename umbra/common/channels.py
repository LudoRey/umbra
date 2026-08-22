"""Handling images that may or may not carry a trailing channel axis."""
import numpy as np


def broadcast_over_channels(pixel_map: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Reshape a per-pixel map so that it broadcasts over the channels of ``img``, if it has any."""
    return pixel_map.reshape(*pixel_map.shape, *(1,) * (img.ndim - pixel_map.ndim))
