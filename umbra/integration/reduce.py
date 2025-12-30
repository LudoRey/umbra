import numpy as np
import bottleneck as bn
from umbra.common import trackers
from umbra.common.terminal import cprint
import memory_profiler

@trackers.track_info
@memory_profiler.profile
def weighted_average_ignore_nan(
    stack: np.ndarray,
    weights: np.ndarray,
    out_img: np.ndarray,
    out_total_weights: np.ndarray
) -> None:
    """
    Compute the weighted average of a stack of images.
    Wheere pixels are NaN in the stack, they are ignored in the computation (i.e., treated as zero weight).
    We assume that for each pixel there is at least one image in the stack with a non-zero weight.

    Parameters
    ----------
    stack : np.ndarray
        Array of shape (N, H, W, C) representing the stack of images.
    weights : np.ndarray
        Array of shape (N, H, W) representing the weights for each pixel in the stack.
    out_img : np.ndarray
        Array of shape (H, W, C) representing the weighted average image.
    out_total_weights : np.ndarray
        Array of shape (H, W, C) representing the average weights used per pixel.
    """
    N, H, W, C = stack.shape
    mask = np.isnan(stack)
    stack[mask] = 0
    weights_c = weights.copy()
    for c in range(C):
        weights_c[:] = weights
        weights_c[mask[..., c]] = 0
        np.einsum('nhw,nhw->hw', stack[..., c], weights_c, out=out_img[..., c])
        np.sum(weights_c, axis=0, out=out_total_weights[..., c])
        out_img[..., c] /= out_total_weights[..., c]
    out_total_weights /= N

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