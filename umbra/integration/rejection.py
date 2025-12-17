import numpy as np
import bottleneck as bn

from umbra.common import disk, trackers, coords
from umbra.common.terminal import cprint

@trackers.track_info
def outlier_rejection(
    stack: np.ndarray,
    outlier_threshold: float,
) -> np.ndarray:
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
    stack : np.ndarray
        Array of shape (N, H, W, C) after outlier rejection. Shares memory with the input array.
        
    Notes
    ------
    Temporary memory requirement per pixel. S is the number of bytes of a pixel in the input array (e.g., 4 bytes for float32):
    - S * N (for the input stack)
    - S * 2 (for the median, std, and bounds)
    - 2 * N (for the lower, upper and combined masks)
    Total: S * N + S * 2 + 2 * N bytes per pixel.
    For float32 : 4N + 8 + 2N = 6N + 8 bytes per pixel.
    """
    print(f"Rejecting outliers...", end=' ', flush=True)
    tmp = bn.nanmedian(stack, axis=0)
    std = bn.nanstd(stack, axis=0)
    tmp -= outlier_threshold * std
    mask = stack < tmp
    tmp += (2 * outlier_threshold) * std
    mask |= stack > tmp
    stack[mask] = np.nan
    print("Done.")
    print(f"Rejected {np.sum(mask)} outliers.")
    return stack, mask

@trackers.track_info
def moon_rejection(
    stack: np.ndarray,
    headers: list[dict],
    extra_radius_pixels: float,
    smoothness: float,
    region: coords.Region = None,
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
        Array of shape (N, H, W) representing the moon rejection weights.
    """
    cprint(f"Rejecting moon pixels...", end=' ', flush=True)
    N, H, W, C = stack.shape
    if region is None:
        region = coords.Region(width=W, height=H, left=0, top=0)
    # Track preferred index map to fill in all-masked pixels later
    max_dist_map = np.zeros((H, W))
    preferred_idx_map = np.zeros((H, W), dtype=np.int16)
    weights = np.zeros((N, H, W, C), dtype=np.float32)
    for i, header in enumerate(headers):
        # Compute moon weights and distance map
        moon_x, moon_y, radius = header["MOON-X"], header["MOON-Y"], header["MOON-R"] + extra_radius_pixels
        dist_map = disk.distance_map(coords.Point(moon_x, moon_y), region)
        if smoothness == 0:
            weights[i] = (dist_map > radius)[..., None]
        else:
            weights[i] = np.clip((dist_map - radius) / smoothness, 0, 1)[..., None]
        # Update preferred index map
        update_mask = dist_map > max_dist_map
        preferred_idx_map[update_mask] = i
        max_dist_map[update_mask] = dist_map[update_mask]
    # Unmask preferred pixels if all are masked
    all_masked = np.all(weights == 0, axis=0)
    h_idx, w_idx, c_idx = np.nonzero(all_masked)
    n_idx = preferred_idx_map[h_idx, w_idx]
    weights[n_idx, h_idx, w_idx, c_idx] = 1
    # Apply mask to stack
    stack[weights == 0] = np.nan
    print("Done.")
    return stack, weights

@trackers.track_info
def compute_rejection_map(
    stack: np.ndarray,
) -> np.ndarray:
    """
    Compute a rejection map indicating which pixels were rejected (NaN) in any image of the stack.

    Parameters
    ----------
    stack : np.ndarray
        Array of shape (N, H, W, C) representing the image stack.

    Returns
    -------
    rejection_map : np.ndarray
        Array of shape (H, W, C) with boolean values indicating rejected pixels.
    """
    cprint("Computing rejection map...", end=" ", flush=True)
    rejection_map = bn.anynan(stack, axis=0)
    cprint("Done.")
    return rejection_map