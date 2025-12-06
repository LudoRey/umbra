import astropy.stats
import numpy as np

from umbra.common import disk, polar
from umbra.common.terminal import cprint
from umbra.integration import memory

def outlier_rejection(
    stack: np.ndarray | np.ma.MaskedArray,
    outlier_threshold: float,
) -> np.ma.MaskedArray:
    """
    Perform iterative sigma clipping on a stack of images.

    Parameters
    ----------
    stack : np.ndarray or np.ma.MaskedArray
        Array of shape (N, H, W, C) representing the image stack.
    outlier_threshold : float
        Threshold in units of standard deviation for clipping.

    Returns
    -------
    stack : np.ma.MaskedArray
        Masked array of shape (N, H, W, C) after outlier rejection. The data shares memory with the input array.
    """
    cprint(f"Rejecting outliers...", end=' ', flush=True)
    # astropy.stats.sigma_clip
    # cant use fast C algorithm as it converts to float64
    # cant use masked input as it converts to float64
    N = stack.shape[0]
    for _ in range(1):
        mean = np.ma.mean(stack, axis=0) # not memory efficient
        print("Computed mean")
        memory.print_memory_usage()
        std = np.ma.std(stack, axis=0) # not memory efficient
        print("Computed std")
        memory.print_memory_usage()
        new_mask = np.zeros_like(stack.mask, dtype=bool)
        # Broadcast mean and std to the shape of stack
        for i in range(N):
            new_mask[i] = np.abs(stack[i] - mean) / std > outlier_threshold
        print("Computed new mask")
        memory.print_memory_usage()
        # Combine with existing mask (in-place update)
        stack.mask |= new_mask
    cprint("Done.")
    return stack

def sigma_clip(
    masked_arr: np.ma.MaskedArray, 
    sigma: float = 3, 
    max_iters: int = 5, 
    axis: int | tuple[int] | None = None,
) -> np.ma.MaskedArray:
    """
    In-place, memory-efficient sigma clipping for a NumPy masked array.
    The mask is updated in-place, so input and output share memory.

    Parameters:
        masked_arr (np.ma.MaskedArray): Input masked array (mask may already exist).
        sigma (float): Sigma threshold for clipping.
        max_iters (int): Maximum number of iterations.
        axis (int or tuple of ints, optional): Axis or axes along which to perform sigma clipping.
    Returns:
        np.ma.MaskedArray: The same masked array, with updated mask.
    """
    axis = 0
    N = masked_arr.shape[axis]
    for _ in range(max_iters):
        # Compute mean and std along axis, ignoring masked values
        mean = np.ma.mean(masked_arr, axis=axis)
        std = np.ma.std(masked_arr, axis=axis)
        new_mask = np.zeros_like(masked_arr.mask, dtype=bool)
        # Broadcast mean and std to the shape of masked_arr
        for i in range(N):
            new_mask[i] = np.abs(masked_arr[i] - mean) > sigma * std
        # Combine with existing mask (in-place update)
        prev_mask = masked_arr.mask.copy()
        masked_arr.mask |= new_mask
        # Stop if no new values were masked
        if np.array_equal(masked_arr.mask, prev_mask):
            break
    return masked_arr

def moon_rejection(
    stack: np.ndarray | np.ma.MaskedArray,
    headers: list[dict],
) -> np.ma.MaskedArray:
    """
    Smooth reject the pixels that correspond to the moon.
    
    Parameters
    ----------
    stack : np.ndarray or np.ma.MaskedArray
        Array of shape (N, H, W, C) representing the image stack.
    headers : list of dict
        List of FITS headers corresponding to each image in the stack. Each header must contain the keywords "MOON-X", "MOON-Y", and "MOON-R".

    Returns
    -------
    stack : np.ma.MaskedArray
        Masked array of shape (N, H, W, C) after moon rejection.
    """
    cprint(f"Rejecting moon pixels...", end=' ', flush=True)
    stack = np.ma.masked_array(stack, mask=False)
    N, H, W, C = stack.shape
    # Track preferred index map to fill in all-masked pixels later
    max_dist_map = np.zeros((H, W))
    preferred_idx_map = np.zeros((H, W), dtype=np.int16)
    for i, header in enumerate(headers):
        # Compute moon mask and distance map
        x_c, y_c, radius = header["MOON-X"], header["MOON-Y"], header["MOON-R"]
        dist_map = polar.radius_map(x_c, y_c, (H, W))
        mask = dist_map <= radius
        # Mask pixels inside the moon mask
        stack.mask[i, mask] = True
        # Update preferred index map
        update_mask = dist_map > max_dist_map
        preferred_idx_map[update_mask] = i
        max_dist_map[update_mask] = dist_map[update_mask]
    stack = unmask_preferred(stack, preferred_idx_map)
    cprint("Done.")
    return stack

def compute_moon_weights(
    header: dict,
    extra_radius: int,
    smoothness: float,
    shape: tuple[int, int],
    return_dist: bool = False,
) -> np.ndarray:
    """
    Compute moon weights for an image. Inside the moon, weights are 0; outside the moon, weights are 1; in between, weights transition smoothly.

    Parameters
    ----------
    header : dict
        FITS header corresponding to the image. The header must contain the keywords "MOON-X", "MOON-Y", and "MOON-R".
    extra_radius : int
        Extra amount of pixels added to the radius of the moon mask.
    smoothness : float
        Smoothness of the mask in pixels.
    shape : tuple of int
        Shape (H, W).

    Returns
    -------
    moon_weights : np.ndarray
        Array of shape (H, W) with the moon weights for the image.
    """
    moon_x_c, moon_y_c, moon_radius = header["MOON-X"], header["MOON-Y"], header["MOON-R"]
    moon_radius += extra_radius
    dist_map = polar.radius_map(moon_x_c, moon_y_c, shape)
    moon_weights = 1 - np.clip((moon_radius - dist_map + smoothness) / smoothness, 0, 1)
    if return_dist:
        return moon_weights, dist_map
    return moon_weights

def unmask_preferred(
    stack: np.ma.MaskedArray,
    preferred_idx_map: np.ndarray,
) -> np.ma.MaskedArray:
    """
    If all images have a pixel masked, unmask it in the preferred image.

    Parameters
    ----------
    stack : np.ma.MaskedArray
        Masked array of shape (N, H, W, C) representing the image stack.
    preferred_idx_map : np.ndarray
        Array of shape (H, W) with the index of the preferred image for each pixel.

    Returns
    -------
    stack : np.ma.MaskedArray
        Stack with updated mask. Shares memory with the input array.
    """
    all_masked = np.all(stack.mask, axis=0)
    h_idx, w_idx, c_idx = np.nonzero(all_masked)
    n_idx = preferred_idx_map[h_idx, w_idx]
    stack.mask[n_idx, h_idx, w_idx, c_idx] = False
    return stack