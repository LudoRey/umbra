import numpy as np
import bottleneck as bn
from umbra.common import trackers
from umbra.common.terminal import cprint

@trackers.track_info
def weighted_average(
    stack: np.ndarray,
    weights: np.ndarray
) -> np.ndarray:
    """
    Compute the weighted average of a stack of images.

    Parameters
    ----------
    stack : np.ndarray
        Array of shape (N, H, W, C) representing the stack of images.
    weights : np.ndarray
        Array of shape (N, H, W) representing the weights for each pixel in the stack.

    Returns
    -------
    np.ndarray
        Array of shape (H, W, C) representing the weighted average image.
    """
    weighted_sum = np.einsum('nhwc,nhwc->hwc', stack, weights)
    sum_of_weights = np.sum(weights, axis=0)
    return weighted_sum / sum_of_weights

@trackers.track_info
def average_ignore_nan(
    stack: np.ndarray
) -> np.ndarray:
    """
    Compute the mean of an array while ignoring NaN values.

    Parameters
    ----------
    stack : np.ndarray
        Input array.
    axis : int
        Axis along which to compute the mean.

    Returns
    -------
    np.ndarray
        Array with the mean computed along the specified axis, ignoring NaNs.
    """
    cprint("Stacking...", end=" ", flush=True)
    img = bn.nanmean(stack, axis=0)
    cprint("Done.")
    return img