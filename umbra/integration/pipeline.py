import gc
from collections.abc import Callable, Sequence
from pathlib import Path

import astropy.io.fits
import numpy as np

from umbra.common import context, coords, fits, imageio
from umbra.common.terminal import cprint
from umbra.integration import io, memory, rejection, reduce

# (stack, headers, region) -> weights of shape (N, H, W)
WeightFn = Callable[[np.ndarray, list[astropy.io.fits.Header], coords.Region], np.ndarray]


def integrate(
    filepaths: Sequence[Path | str],
    outlier_threshold: float | None = None,
    weight_fn: WeightFn | None = None,
) -> tuple[np.ndarray, astropy.io.fits.Header, np.ndarray]:
    """
    Stack a list of images into a single master image, in a memory-aware (chunked) manner.

    Outliers are rejected via single-pass sigma clipping, unless ``outlier_threshold`` is None
    in which case rejection is skipped. The reduction is a plain mean ignoring NaNs, unless a
    ``weight_fn`` is provided, in which case a weighted average is used.

    When neither rejection nor weighting is requested (``outlier_threshold`` and ``weight_fn``
    both None), stacking is delegated to :func:`integrate_no_rejection`, a much simpler
    sequential accumulation path that avoids holding a stack in memory.

    Parameters
    ----------
    filepaths : sequence of paths
        The images to stack.
    outlier_threshold : float or None
        Threshold in units of standard deviation for sigma clipping. When None, no rejection
        is performed.
    weight_fn : callable or None
        Optional ``(stack, headers, region) -> weights`` callback returning per-pixel
        weights of shape (N, H, W). When None, a uniform mean ignoring NaNs is computed.

    Live preview (``context.emit_image``) and cancellation (``context.checkstate``)
    are read from the ambient pipeline context.

    Returns
    -------
    img : np.ndarray
        The stacked image of shape (H, W) or (H, W, C).
    header : astropy.io.fits.Header
        The header common to all stacked images.
    total_weights : np.ndarray
        The average weights used per pixel, of shape (H, W) or (H, W, C). Doubles as a rejection
        map. When ``weight_fn`` is None, it is the fraction of non-rejected frames per pixel.
    """
    if outlier_threshold is None and weight_fn is None:
        return integrate_no_rejection(filepaths)

    num_images = len(filepaths)
    shape = imageio.read_shape(filepaths[0])  # (H, W) or (H, W, C)

    # Compute available memory
    safe_memory_fraction = 0.8
    available_mem = int(memory.get_available_memory() * safe_memory_fraction)
    cprint(f"Using at most {safe_memory_fraction*100:.0f}% of available memory: {available_mem / 1000000:.2f} MB.")

    # Prepare output arrays and force allocation in order to compute available memory
    dtype = np.dtype(np.float32)
    img = np.zeros(shape, dtype=dtype)
    img[:] = np.nan
    total_weights = np.zeros(shape, dtype=dtype)

    # Chunking based on available memory
    rows_ranges = memory.compute_rows_ranges_for_stack(num_images, shape, available_mem, dtype=dtype, has_weights=weight_fn is not None)
    num_chunks = len(rows_ranges)

    # Process each chunk
    headers: list[astropy.io.fits.Header] = []
    for chunk_idx, (row_start, row_end) in enumerate(rows_ranges, start=1):
        cprint(f"Processing chunk {chunk_idx}/{num_chunks} (rows {row_start} -> {row_end})", style="bold")
        region = coords.Region(width=shape[1], height=row_end-row_start, left=0, top=row_start)
        stack, headers = io.read_stack(filepaths, region)
        # Pixel rejection
        weights = weight_fn(stack, headers, region) if weight_fn is not None else None
        context.checkstate()
        if outlier_threshold is not None:
            rejection.outlier_rejection(stack, outlier_threshold)
            context.checkstate()
        # Update output arrays
        if weights is None:
            reduce.average_ignore_nan(stack, img[row_start:row_end], total_weights[row_start:row_end])
        else:
            reduce.weighted_average_ignore_nan(stack, weights, img[row_start:row_end], total_weights[row_start:row_end])
        context.checkstate()
        context.emit_image(img)
        # Free memory
        del stack, weights
        gc.collect()

    output_header = fits.intersect(headers)
    return img, output_header, total_weights


def integrate_no_rejection(
    filepaths: Sequence[Path | str],
) -> tuple[np.ndarray, astropy.io.fits.Header, np.ndarray]:
    """
    Stack a list of images into a single master image via a plain mean.

    Without outlier rejection there is no need to hold a whole stack in memory: each frame is
    read sequentially and accumulated, then the running sum is divided by the frame count.

    Parameters
    ----------
    filepaths : sequence of paths
        The images to stack.

    Live preview (``context.emit_image``, the running mean after each frame) and
    cancellation (``context.checkstate``) are read from the ambient pipeline context.

    Returns
    -------
    img : np.ndarray
        The stacked image of shape (H, W) or (H, W, C).
    header : astropy.io.fits.Header
        The header common to all stacked images.
    total_weights : np.ndarray
        Uniform weights of ones (every frame contributes to every pixel), kept for interface
        parity with :func:`integrate`. Doubles as a rejection map.
    """
    num_images = len(filepaths)
    shape = imageio.read_shape(filepaths[0])  # (H, W) or (H, W, C)

    img = np.zeros(shape, dtype=np.float32)
    headers: list[astropy.io.fits.Header] = []
    for i, filepath in enumerate(filepaths, start=1):
        cprint(f"Accumulating image {i}/{num_images}", style="bold")
        data, header = imageio.read(filepath, verbose=False)
        img += data
        headers.append(header)
        context.checkstate()
        context.emit_image(img / i)

    img /= num_images
    context.emit_image(img)

    total_weights = np.ones(shape, dtype=np.float32)
    output_header = fits.intersect(headers)
    return img, output_header, total_weights
