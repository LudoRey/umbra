from typing import Callable
import numpy as np
import bottleneck as bn

from umbra.common import disk, coords
from umbra.common.terminal import cprint

def outlier_rejection(
    stack: np.ndarray,
    outlier_threshold: float,
    *,
    checkstate: Callable[[], None]
) -> None:
    """
    Perform iterative sigma clipping on a stack of images.

    Parameters
    ----------
    stack : np.ndarray
        Array of shape (N, H, W, C) representing the image stack. May contain NaNs.
    outlier_threshold : float
        Threshold in units of standard deviation for clipping.

    Returns
    -------
    None
        The function modifies the input stack in place, setting outlier pixels to NaN.
        
    Notes
    ------
    Temporary memory requirement (excluding input stack) in bytes:
    S is the number of bytes of a pixel in the input array (e.g., 4 bytes for float32):
    - S * 2 * H * W * C (for the median, std, and bounds)
    - 2 * N * H * W * C (for the lower, upper and combined masks)
    Total: (S * 2 + 2 * N ) * H * W * C bytes.
    For float32 : (8 + 2N) * H * W * C bytes.
    """
    print(f"Rejecting outliers...", end=' ', flush=True)
    tmp = bn.nanmedian(stack, axis=0)
    std = bn.nanstd(stack, axis=0)
    tmp -= outlier_threshold * std
    mask = stack < tmp
    tmp += (2 * outlier_threshold) * std
    mask |= stack > tmp
    stack[mask] = np.nan
    num_rejected = np.sum(mask)
    checkstate()
    print(f"Found {num_rejected} pixels.")

def moon_rejection(
    stack: np.ndarray,
    headers: list[dict],
    extra_radius_pixels: float,
    smoothness: float,
    region: coords.Region = None,
    *,
    checkstate: Callable[[], None]
) -> np.ndarray:
    """
    Smooth reject the pixels that correspond to the moon.
    
    Parameters
    ----------
    stack : np.ndarray
        Array of shape (N, H, W, C) representing the image stack.
    headers : list of dict
        List of FITS headers corresponding to each image in the stack. Each header must contain the keywords "MOON-X", "MOON-Y", and "MOON-R".
    extra_radius_pixels : float
        Additional radius in pixels to add to the moon radius for rejection.
    smoothness : float
        Smoothness parameter in pixels for the weight transition.
    region : coords.Region, optional
        The region covered by the stack.

    Returns
    -------
    weights : np.ndarray
        Array of shape (N, H, W) representing the moon rejection weights. The stack pixels with zero weight are set to NaN in place.
        
    Notes
    ------
    The returned array uses S * N * H * W bytes of memory, where S is the number of bytes of a pixel in the input array (e.g., 4 bytes for float32).
    """
    cprint(f"Rejecting the moon...", end=' ', flush=True)
    N, H, W = stack.shape[0:3]
    if region is None:
        region = coords.Region(width=W, height=H, left=0, top=0)
    # Track preferred index map to fill in all-masked pixels later
    max_dist_map = np.zeros((H, W), dtype=stack.dtype)
    preferred_idx_map = np.zeros((H, W), dtype=np.int16)
    weights = np.zeros((N, H, W), dtype=stack.dtype)
    for i, header in enumerate(headers):
        # Compute moon weights and distance map
        moon_x, moon_y, radius = header["MOON-X"], header["MOON-Y"], header["MOON-R"] + extra_radius_pixels
        dist_map = disk.distance_map(coords.Point(moon_x, moon_y), region)
        if smoothness == 0:
            weights[i] = (dist_map > radius)
        else:
            weights[i] = np.clip((dist_map - radius) / smoothness, 0, 1)
        # Update preferred index map
        update_mask = dist_map > max_dist_map
        preferred_idx_map[update_mask] = i
        max_dist_map[update_mask] = dist_map[update_mask]
    # Ensure no pixel has all weights zero by assigning weight 1 to preferred index
    all_zero_pixels = np.nonzero(np.all(weights == 0, axis=0))
    preferred_image_indices = preferred_idx_map[all_zero_pixels]
    weights[(preferred_image_indices, *all_zero_pixels)] = 1.0
    # Set fully rejected pixels to NaN in the stack
    mask = weights == 0
    stack[mask] = np.nan
    num_rejected = np.sum(mask)
    checkstate()
    print(f"Found {num_rejected} pixels.")
    return weights
