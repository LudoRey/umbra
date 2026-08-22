import numpy as np
import bottleneck as bn

def weighted_average_ignore_nan(
    stack: np.ndarray,
    weights: np.ndarray,
    out_img: np.ndarray,
    out_total_weights: np.ndarray,
) -> None:
    """
    Compute the weighted average of a stack of images.
    Wheere pixels are NaN in the stack, they are ignored in the computation (i.e., treated as zero weight).
    We assume that for each pixel there is at least one image in the stack with a non-zero weight.

    Parameters
    ----------
    stack : np.ndarray
        Array of shape (N, H, W) or (N, H, W, C) representing the stack of images.
    weights : np.ndarray
        Array of shape (N, H, W) representing the weights for each pixel in the stack.
    out_img : np.ndarray
        Output, modified in-place. Array of shape (H, W) or (H, W, C) representing the weighted average image.
    out_total_weights : np.ndarray
        Output, modified in-place. Array of shape (H, W) or (H, W, C) representing the average weights used per pixel.
    """
    print("Computing weighted average...", end=" ", flush=True)
    N = stack.shape[0]
    if stack.ndim == 3:
        # A mono frame is one channel that carries no axis of its own. These are views, so the
        # loop below still writes through to the caller's arrays.
        stack, out_img, out_total_weights = stack[..., None], out_img[..., None], out_total_weights[..., None]
    C = stack.shape[3]
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
    print("Done.")

def average_ignore_nan(
    stack: np.ndarray,
    out_img: np.ndarray,
    out_total_weights: np.ndarray,
) -> None:
    """
    Compute the average of a stack of images, ignoring NaN pixels.

    Parameters
    ----------
    stack : np.ndarray
        Array of shape (N, H, W) or (N, H, W, C) representing the stack of images.
    out_img : np.ndarray
        Output, modified in-place. Array of shape (H, W) or (H, W, C) representing the average image.
    out_total_weights : np.ndarray
        Output, modified in-place. Array of shape (H, W) or (H, W, C) holding the fraction of non-NaN
        (i.e. non-rejected) frames per pixel, in [0, 1]. This doubles as a rejection map.
    """
    print("Computing average...", end=" ", flush=True)
    N = stack.shape[0]
    out_img[:] = bn.nanmean(stack, axis=0)
    np.sum(~np.isnan(stack), axis=0, out=out_total_weights)
    out_total_weights /= N
    print("Done.")
