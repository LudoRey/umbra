import gc
from collections.abc import Callable, Sequence
from pathlib import Path

import astropy.io.fits
import numpy as np

from umbra.common import coords, fits
from umbra.common.terminal import cprint
from umbra.common.typing import CheckStateCallback, ImageCallback
from umbra.integration import io, memory, rejection, reduce

# (stack, headers, region) -> weights of shape (N, H, W)
WeightFn = Callable[[np.ndarray, list[astropy.io.fits.Header], coords.Region], np.ndarray]


def integrate(
    filepaths: Sequence[Path | str],
    outlier_threshold: float,
    weight_fn: WeightFn | None = None,
    *,
    img_callback: ImageCallback = lambda _img: None,
    checkstate: CheckStateCallback = lambda: None,
) -> tuple[np.ndarray, astropy.io.fits.Header, np.ndarray | None]:
    """
    Stack a list of images into a single master image, in a memory-aware (chunked) manner.

    Outliers are always rejected (single-pass sigma clipping). The reduction is a plain mean
    ignoring NaNs, unless a ``weight_fn`` is provided, in which case a weighted average is used.

    Parameters
    ----------
    filepaths : sequence of paths
        The images to stack.
    outlier_threshold : float
        Threshold in units of standard deviation for sigma clipping.
    weight_fn : callable or None
        Optional ``(stack, headers, region) -> weights`` callback returning per-pixel
        weights of shape (N, H, W). When None, a uniform mean ignoring NaNs is computed.
    img_callback : callable
        Called with the current image after each chunk (for live preview).
    checkstate : callable
        Called to allow graceful cancellation.

    Returns
    -------
    img : np.ndarray
        The stacked image of shape (H, W) or (H, W, C).
    header : astropy.io.fits.Header
        The header common to all stacked images.
    total_weights : np.ndarray or None
        The average weights used per pixel, of shape (H, W) or (H, W, C). None when ``weight_fn`` is None.
    """
    num_images = len(filepaths)
    shape = fits.extract_shape(fits.read_fits_header(filepaths[0]))  # (H, W) or (H, W, C)

    # Compute available memory
    safe_memory_fraction = 0.8
    available_mem = int(memory.get_available_memory() * safe_memory_fraction)
    cprint(f"Using at most {safe_memory_fraction*100:.0f}% of available memory: {available_mem / 1000000:.2f} MB.")

    # Prepare output arrays and force allocation in order to compute available memory
    dtype = np.dtype(np.float32)
    img = np.zeros(shape, dtype=dtype)
    img[:] = np.nan
    total_weights = np.zeros(shape, dtype=dtype) if weight_fn is not None else None

    # Chunking based on available memory
    rows_ranges = memory.compute_rows_ranges_for_stack(num_images, shape, available_mem, dtype=dtype, has_weights=weight_fn is not None)
    num_chunks = len(rows_ranges)

    # Process each chunk
    headers: list[astropy.io.fits.Header] = []
    for chunk_idx, (row_start, row_end) in enumerate(rows_ranges, start=1):
        cprint(f"Processing chunk {chunk_idx}/{num_chunks} (rows {row_start} -> {row_end})", style="bold")
        region = coords.Region(width=shape[1], height=row_end-row_start, left=0, top=row_start)
        stack, headers = io.read_stack(filepaths, region, checkstate=checkstate)
        # Pixel rejection
        weights = weight_fn(stack, headers, region) if weight_fn is not None else None
        checkstate()
        rejection.outlier_rejection(stack, outlier_threshold)
        checkstate()
        # Update output arrays
        if weights is None:
            reduce.average_ignore_nan(stack, img[row_start:row_end])
        else:
            assert total_weights is not None
            reduce.weighted_average_ignore_nan(stack, weights, img[row_start:row_end], total_weights[row_start:row_end])
        checkstate()
        img_callback(img)
        # Free memory
        del stack, weights
        gc.collect()

    output_header = fits.intersect_headers(headers)
    return img, output_header, total_weights
