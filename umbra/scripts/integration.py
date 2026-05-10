import gc
import os
from collections.abc import Sequence
from typing import cast

import astropy.io.fits
import numpy as np

from umbra.common import fits, coords
from umbra.common.terminal import cprint
from umbra import integration
from umbra.common.typing import CheckStateCallback, ImageCallback


# @trackers.track_info
def main(
    # IO
    registered_dir: str,
    stacks_dir: str,
    group_keywords: Sequence[str],
    # Outlier rejection
    outlier_threshold: float,
    # Moon rejection (optional, for sun-registered images)
    moon_rejection: bool = False,
    extra_radius_pixels: float = 0,
    smoothness: float = 0,
    # GUI interactions
    *,
    img_callback: ImageCallback = lambda _img: None,
    checkstate: CheckStateCallback = lambda: None,
) -> None:
    # Process each group
    grouped_filepaths = fits.get_grouped_filepaths(registered_dir, group_keywords)
    num_groups = len(grouped_filepaths)
    for group_idx, group_values in enumerate(grouped_filepaths.keys(), start=1):
        # Extract info about the group
        filepaths = grouped_filepaths[group_values]
        num_images = len(filepaths)
        header = fits.read_fits_header(filepaths[num_images // 2])
        shape = (cast(int, header["NAXIS2"]), cast(int, header["NAXIS1"]), cast(int, header["NAXIS3"])) # (H, W, C)
        group_identifier = ', '.join([f'{k}={v}' for k, v in zip(group_keywords, group_values)])
        cprint(f"Stacking {num_images} images from group {group_idx}/{num_groups} ({group_identifier})", style="bold")
            
        # Compute available memory
        safe_memory_fraction = 0.8
        available_mem = int(integration.memory.get_available_memory() * safe_memory_fraction)
        cprint(f"Using at most {safe_memory_fraction*100:.0f}% of available memory: {available_mem / 1000000:.2f} MB.")
        
        # Prepare output arrays and force allocation in order to compute available memory
        dtype = np.dtype(np.float32)
        img = np.zeros(shape, dtype=dtype)
        total_weights = np.zeros(shape, dtype=dtype)
        img[:] = np.nan
        total_weights[:] = 0
        
        # Chunking based on available memory
        rows_ranges = integration.memory.compute_rows_ranges_for_stack(num_images, shape, available_mem, dtype=dtype)
        num_chunks = len(rows_ranges)
        
        # Process each chunk
        headers: list[astropy.io.fits.Header] = []
        for chunk_idx, (row_start, row_end) in enumerate(rows_ranges, start=1):
            cprint(f"Processing chunk {chunk_idx}/{num_chunks} (rows {row_start} -> {row_end})", style="bold")
            region = coords.Region(width=shape[1], height=row_end-row_start, left=0, top=row_start)
            stack, headers = integration.io.read_stack(filepaths, region, checkstate=checkstate)
            # Pixel rejection
            if moon_rejection:
                weights = integration.rejection.moon_rejection(stack, headers, extra_radius_pixels, smoothness, region, checkstate=checkstate)
            else:
                weights = np.ones(stack.shape[0:3], dtype=stack.dtype)
            integration.rejection.outlier_rejection(stack, outlier_threshold, checkstate=checkstate)
            # Update output arrays
            integration.reduce.weighted_average_ignore_nan(stack, weights, img[row_start:row_end], total_weights[row_start:row_end])
            checkstate()
            img_callback(img)
            # Free memory
            del stack, weights
            gc.collect()
            
        cprint(f"Finished stacking group {group_idx}/{num_groups}.", color="green")
        output_header = fits.intersect_headers(headers)
        group_name = " - ".join([f"{group_keywords[i]}_{group_values[i]}" for i in range(len(group_keywords))])
        fits.save_as_fits(img, output_header, os.path.join(stacks_dir, f"{group_name}.fits"), checkstate=checkstate)
        fits.save_as_fits(total_weights, None, os.path.join(stacks_dir, f"{group_name}_rejection.fits"), checkstate=checkstate)

if __name__ == "__main__":
    import sys
    import yaml
    from umbra.common.terminal import ColorTerminalStream
    sys.stdout = ColorTerminalStream()

    with open("config_custom.yaml") as f:
        config = yaml.safe_load(f)

    main(**config["sun_integration"])
    main(**config["moon_integration"])
