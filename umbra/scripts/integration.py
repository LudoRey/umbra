import gc
import os
import numpy as np

from umbra.common import fits, coords, trackers
from umbra.common.terminal import cprint
from umbra import integration

@trackers.track_info
def main(
    # IO
    registered_dir,
    stacks_dir,
    group_keywords,
    # Outlier rejection
    outlier_threshold,
    # Moon rejection (optional, for sun-registered images)
    moon_rejection=False,
    extra_radius_pixels=None,
    smoothness=None,
    # GUI interactions
    *,
    img_callback=lambda img: None,
    checkstate=lambda: None
):
    # Process each group
    grouped_filepaths = fits.get_grouped_filepaths(registered_dir, group_keywords)
    num_groups = len(grouped_filepaths)
    for group_idx, group_values in enumerate(grouped_filepaths.keys(), start=1):
        # Extract info about the group
        filepaths = grouped_filepaths[group_values]
        num_images = len(filepaths)
        header = fits.read_fits_header(filepaths[num_images // 2])
        shape = (header["NAXIS2"], header["NAXIS1"], header["NAXIS3"]) # (H, W, C)
        group_identifier = ', '.join([f'{k}={v}' for k, v in zip(group_keywords, group_values)])
        cprint(f"Stacking {num_images} images from group {group_idx}/{num_groups} ({group_identifier})", style="bold")
            
        # Prepare output arrays and force allocation in order to compute available memory
        img = np.zeros(shape, dtype=np.float32)
        total_weights = np.zeros(shape, dtype=np.float32)
        img[:] = 0
        total_weights[:] = 0
        
        # Chunking based on available memory
        available_mem = integration.memory.get_available_memory()
        safe_memory_fraction = 0.8
        max_mem = int(available_mem * safe_memory_fraction)
        cprint(f"Using at most {safe_memory_fraction*100:.0f}% of available memory: {max_mem / 1000000:.2f} MB.")
        rows_ranges = integration.memory.compute_rows_ranges_for_stack(num_images, shape, max_mem)
        num_chunks = len(rows_ranges)
        
        # Process each chunk
        for chunk_idx, (row_start, row_end) in enumerate(rows_ranges, start=1):
            cprint(f"Processing chunk {chunk_idx}/{num_chunks} (rows {row_start} -> {row_end})", style="bold")
            region = coords.Region(width=shape[1], height=row_end-row_start, left=0, top=row_start)
            stack, headers = integration.io.read_stack(filepaths, (row_start, row_end), img_callback=img_callback, checkstate=checkstate)
            # Pixel rejection
            if moon_rejection:
                weights = integration.rejection.moon_rejection(stack, headers, extra_radius_pixels, smoothness, region, checkstate=checkstate)
            else:
                weights = np.ones(stack.shape[0:3], dtype=stack.dtype)
            integration.rejection.outlier_rejection(stack, outlier_threshold, checkstate=checkstate)
            # Update output arrays
            integration.reduce.weighted_average_ignore_nan(stack, weights, img[row_start:row_end], total_weights[row_start:row_end], img_callback=img_callback, checkstate=checkstate)
            # Free memory
            del stack, weights
            gc.collect()
            
        cprint(f"Finished stacking group {group_idx}/{num_groups}.", color="green")
        output_header = fits.extract_subheader(header, group_keywords+["MOON-X", "MOON-Y"])
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