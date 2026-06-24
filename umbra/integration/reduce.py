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
    Pixels whose total weight is zero (e.g. NaN across the whole stack) are set to 0.

    Parameters
    ----------
    stack : np.ndarray
        Array of shape (N, H, W, C) representing the stack of images.
    weights : np.ndarray
        Array of shape (N, H, W) representing the weights for each pixel in the stack.
    out_img : np.ndarray
        Output, modified in-place. Array of shape (H, W, C) representing the weighted average image.
    out_total_weights : np.ndarray
        Output, modified in-place. Array of shape (H, W, C) representing the average weights used per pixel.
    """
    print("Computing weighted average...", end=" ", flush=True)
    N, H, W, C = stack.shape
    mask = np.isnan(stack)
    stack[mask] = 0
    weights_c = weights.copy()
    for c in range(C):
        weights_c[:] = weights
        weights_c[mask[..., c]] = 0
        np.einsum('nhw,nhw->hw', stack[..., c], weights_c, out=out_img[..., c])
        np.sum(weights_c, axis=0, out=out_total_weights[..., c])
        # Where total weight is 0, out_img is already 0 (sum of 0*0); skip the division to avoid 0/0 NaN.
        np.divide(out_img[..., c], out_total_weights[..., c], out=out_img[..., c],
                  where=out_total_weights[..., c] != 0)
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
        Array of shape (N, H, W, C) representing the stack of images.
    out_img : np.ndarray
        Output, modified in-place. Array of shape (H, W, C) representing the average image.
    out_total_weights : np.ndarray
        Output, modified in-place. Array of shape (H, W, C) holding the fraction of non-NaN
        (i.e. non-rejected) frames per pixel, in [0, 1]. This doubles as a rejection map.
    """
    print("Computing average...", end=" ", flush=True)
    N = stack.shape[0]
    out_img[:] = bn.nanmean(stack, axis=0)
    np.sum(~np.isnan(stack), axis=0, out=out_total_weights)
    # Pixels that are NaN across the whole stack have a count of 0; set them to 0.
    out_img[out_total_weights == 0] = 0
    out_total_weights /= N
    print("Done.")
