import astropy.stats
import numpy as np

from umbra.common import disk
from umbra.common.utils import cprint

def outlier_mad_rejection(
    stack: np.ndarray | np.ma.MaskedArray,
    rejection_threshold: float
) -> np.ma.MaskedArray:
    """
    Perform median absolute deviation (MAD) clipping on a stack of images.

    Parameters
    ----------
    stack : np.ndarray or np.ma.MaskedArray
        Array of shape (N, H, W, C) representing the image stack.
    rejection_threshold : float
        Threshold in units of MAD for clipping.

    Returns
    -------
    masked_stack : np.ma.MaskedArray
        Masked array of shape (N, H, W, C). The data shares memory with the input array.
    """
    cprint(f"Rejecting outliers with MAD threshold...", end=' ', flush=True)
    masked_stack = astropy.stats.sigma_clip(
        stack, 
        sigma=rejection_threshold, 
        stdfunc='mad_std', 
        axis=0, 
        copy=False, 
        maxiters=1
    )
    print("Done.")
    return masked_stack

def compute_moon_weights(
    stack: np.ndarray | np.ma.MaskedArray,
    header_list: list[dict],
    extra_radius: int,
    smoothness: float,
) -> np.ndarray:
    """
    Reject the pixels that correspond to the moon.
    
    Parameters
    ----------
    stack : np.ndarray or np.ma.MaskedArray
        Array of shape (N, H, W, C) representing the image stack.
    extra_radius : int
        Extra amount of pixels added to the radius of the moon mask. Increasing this parameter will lead to fewer artifacts at the cost of worse SNR: it should be as close to 0 as possible.
    smoothness : float
        Smoothness of the mask in pixels. Increasing this parameter leads to a smoother transition at the cost of worse SNR.

    Returns
    -------
    weights : np.ndarray
        Array of shape (N, H, W, C) of representing the weights for each pixel in the stack. The dtype is np.uint8
    """
    cprint(f"Rejecting moon pixels...", end=' ', flush=True)
    shape = stack.shape[1:4]  # (H, W, C)
    weights = np.zeros(stack.shape, dtype=np.uint8)
    for i, header in enumerate(header_list):
        moon_x_c, moon_y_c, moon_radius = header["MOON-X"], header["MOON-Y"], header["MOON-R"]
        moon_mask, dist_to_moon_center = disk.linear_falloff_disk(
            moon_x_c, 
            moon_y_c, 
            moon_radius + extra_radius, 
            shape[0:2], 
            smoothness=smoothness, 
            return_dist=True
        )
        # TODO
        